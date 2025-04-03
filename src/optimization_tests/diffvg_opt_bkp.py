import os
import copy
import random

os.environ["TORCH_COMPILE_DISABLE"] = "1"

import pandas as pd
import cv2
import io
import json
import cairosvg
import torch
from src.score_original import VQAEvaluator, ImageProcessor, AestheticEvaluator
from src.score_gradient import (
    score_original,
    score_gradient,
    aesthetic_score_original,
    vqa_score_original,
    aesthetic_score_gradient,
    vqa_score_gradient,
    harmonic_mean,
    harmonic_mean_grad,
    score_gradient_ocr,
)
from src.preprocessing import apply_preprocessing_torch

from pathlib import Path
import io
from torch.nn import functional as F
import numpy as np
import requests
import torch
from PIL import Image
from tqdm.auto import tqdm
from src.utils import optimize_svg, svg_to_png, create_random_svg
from src.text_to_svg import text_to_svg, rgb_to_hex
import pydiffvg
import kagglehub


svg_constraints = kagglehub.package_import("metric/svg-constraints", bypass_confirmation=True)



def get_optimization_settings():
    # Create optimization settings
    settings = pydiffvg.SvgOptimizationSettings()
    
    # Configure optimization settings
    settings.global_override(["optimizer"], "Adam")
    settings.global_override(["color_lr"], 1e-1)
    settings.global_override(["alpha_lr"], 1e-1)
    settings.global_override(["paths", "shape_lr"], 1.0)  # Control point movement rate
    settings.global_override(["circles", "shape_lr"], 1.0)  # Control point movement rate
    settings.global_override(["transforms", "transform_lr"], 1.0)  # Transform modification rate
    
    # For filled shapes, optimize colors and transforms
    settings.global_override(["optimize_color"], True)
    settings.global_override(["optimize_alpha"], True)
    settings.global_override(["paths", "optimize_points"], True)
    settings.global_override(["circles", "optimize_center"], True)
    settings.global_override(["circles", "optimize_radius"], True)
    settings.global_override(["transforms", "optimize_transforms"], True)

    return settings


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False



def get_optimizer(
    params: torch.Tensor,
    lr: float = 0.01,
    weight_decay: float = 0.0,
    T_max: int = 100,
    eta_min: float = 0.0,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    optimizer = torch.optim.AdamW([params], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=T_max, eta_min=0.1 * lr
    )
    return optimizer, scheduler


def torch_to_pil(image: torch.Tensor) -> Image.Image:
    image = (image.detach() * 255.0).clamp(0, 255)
    image = image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    image = Image.fromarray(image)
    return image



def get_initial_svg(target_text: str, canvas_width: int = 384, canvas_height: int = 384, num_tiles: int = 4):
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_width}" height="{canvas_height}">\n'
    svg += f'  <rect width="{canvas_width}" height="{canvas_height}" fill="{rgb_to_hex(255, 255, 255)}" />\n'

    for i in range(num_tiles):
        for j in range(num_tiles):
            x = i * (canvas_width // num_tiles)
            y = j * (canvas_height // num_tiles)
            width = canvas_width // num_tiles
            height = canvas_height // num_tiles
            random_color = rgb_to_hex(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            
            svg += f'  <path d="M {x},{y} h {width} v {height} h {-width} z" fill="{random_color}" />\n'
            # svg += f'  <rect x="{x}" y="{y}" width="{canvas_width // num_tiles}" height="{canvas_height // num_tiles}" fill="{random_color}" />\n'
            # svg += f'  <ellipse cx="{x + canvas_width // num_tiles // 2}" cy="{y + canvas_height // num_tiles // 2}" rx="{canvas_width // num_tiles // 2}" ry="{canvas_height // num_tiles // 2}" fill="{random_color}" />\n'

    # Add text in a specific group with ID
    svg += f'<g id="text-group" class="no-optimize">\n'
    text_svg = text_to_svg(target_text, svg_width=canvas_width, svg_height=canvas_height)
    svg += "\n".join(text_svg.split("\n")[2:-1])
    svg += '</g>\n'
    svg += "</svg>"

    return svg


def optimize_diffvg(
    vqa_evaluator: VQAEvaluator,
    aesthetic_evaluator: AestheticEvaluator,
    target_text: str,
    canvas_width: int = 384,
    canvas_height: int = 384,
    num_iterations: int = 100,
    validation_steps: int = 10,
) -> Image.Image:
    vqa_evaluator.model.eval()
    vqa_evaluator.model.requires_grad_(False)
    aesthetic_evaluator.predictor.eval()
    aesthetic_evaluator.predictor.requires_grad_(False)
    aesthetic_evaluator.clip_model.eval()
    aesthetic_evaluator.clip_model.requires_grad_(False)

    pydiffvg.set_use_gpu(torch.cuda.is_available())
    
    svg_content = get_initial_svg(target_text, canvas_width, canvas_height, num_tiles=8)

    temp_svg_path = "/tmp/initial_svg.svg"
    with open(temp_svg_path, "w") as f:
        f.write(svg_content)

    settings = get_optimization_settings()
    # Disable optimization for the text group
    settings.undefault("text-group")
    settings.global_override(["paths", "optimize_points"], False)
    settings.global_override(["optimize_color"], False)
    settings.global_override(["transforms", "optimize_transforms"], False)
    
    # Add this custom override function that disables optimization for text paths
    def custom_settings_override(node_id, settings_dict):
        # If the ID starts with "text-" or whatever prefix you choose
        if node_id is not None and node_id.startswith("text-"):
            settings_dict["paths"]["optimize_points"] = False
            settings_dict["optimize_color"] = False
            settings_dict["transforms"]["optimize_transforms"] = False
        return settings_dict

    # Inject this function into the settings retrieval process
    original_retrieve = settings.retrieve
    def custom_retrieve(node_id):
        settings_dict, found = original_retrieve(node_id)
        return custom_settings_override(node_id, settings_dict), found

    settings.retrieve = custom_retrieve

    optim_svg = pydiffvg.OptimizableSvg(temp_svg_path, settings, verbose=False, device="cuda:0")

    validator = svg_constraints.SVGConstraints(max_svg_size=10000)
    best_svg = optimize_svg(optim_svg.write_xml())
        
    pbar = tqdm(total=num_iterations)

    for iter_idx in range(num_iterations):
        optim_svg.zero_grad()
        image = optim_svg.render(seed=iter_idx)

        img = image[:, :, :3].permute(2, 0, 1)
        img = F.interpolate(img.clamp(0, 1).unsqueeze(0), size=(384, 384), mode="bicubic", align_corners=False, antialias=True)[0].clamp(0, 1)
        
        ocr_loss = score_gradient_ocr(vqa_evaluator, img).mean()
        
        proc_img = apply_preprocessing_torch(img)
        aest_loss = (1.0 - aesthetic_score_gradient(aesthetic_evaluator, proc_img)).mean()

        loss = ocr_loss + aest_loss

        loss.backward()
        optim_svg.step()

        if iter_idx == 0 or (iter_idx + 1) % validation_steps == 0:
            pil_image = torch_to_pil(img).resize((384, 384))
            ocr_val_loss, text = vqa_evaluator.ocr(pil_image)

            proc_pil_img = ImageProcessor(copy.deepcopy(pil_image)).apply().image
            aest_val_loss = (1.0 - aesthetic_score_original(aesthetic_evaluator, proc_pil_img))
            val_loss = ocr_val_loss + aest_val_loss

            text = text.replace('\n', '')

        pbar.set_description(
            f"It {iter_idx}/{num_iterations} | "
            f"Loss: {loss.item():.3f} | "
            f"OCR Loss: {ocr_loss.item():.3f} | "
            f"Aest Loss: {aest_loss.item():.3f} | "
            f"Val Loss: {val_loss:.3f} | "
            f"OCR Val Loss: {ocr_val_loss:.3f} | "
            f"Aest Val Loss: {aest_val_loss:.3f} | "
            f"Text: `{text[:10]}` | "
            # f"LR: {scheduler.get_last_lr()[0]:.4f}"
        )
        pbar.update(1)


    image = torch_to_pil(img)

    return image


def evaluate_questions(
    vqa_evaluator: VQAEvaluator, aesthetic_evaluator: AestheticEvaluator, image: Image.Image
):
    df = pd.read_parquet("/home/mpf/code/kaggle/draw/question_generation_results.parquet")
    df = df[df["set"] == "test"].iloc[1:2]

    for index, row in tqdm(df.iterrows(), total=len(df)):
        description = row["description"]
        print(description)

        gt_questions_list = {
            "questions": [dct["question"] for dct in row["ground_truth"]],
            "choices_list": [dct["choices"].tolist() for dct in row["ground_truth"]],
            "answers": [dct["answer"] for dct in row["ground_truth"]],
        }
        gen_questions_list = {
            "questions": [dct["question"] for dct in row["generated"]],
            "choices_list": [dct["choices"].tolist() for dct in row["generated"]],
            "answers": [dct["answer"] for dct in row["generated"]],
        }

        score_gen = score_original(
            vqa_evaluator,
            aesthetic_evaluator,
            image,
            gen_questions_list["questions"],
            gen_questions_list["choices_list"],
            gen_questions_list["answers"],
        )

        score_gt = score_original(
            vqa_evaluator,
            aesthetic_evaluator,
            image,
            gt_questions_list["questions"],
            gt_questions_list["choices_list"],
            gt_questions_list["answers"],
        )

        print(f"Score Gen: {score_gen}")
        print(f"Score GT: {score_gt}")


def evaluate():
    seed_everything(42)

    vqa_evaluator = VQAEvaluator()
    aesthetic_evaluator = AestheticEvaluator()

    run_inference = True
    if run_inference:
        image = optimize_diffvg(
            vqa_evaluator=vqa_evaluator,
            aesthetic_evaluator=aesthetic_evaluator,
            target_text="purple pyramids spiraling around a bronze cone",
            num_iterations=100,
            validation_steps=1,
        )
        image.save("output.png")

    image = Image.open("output.png").resize((384, 384))
    # image = ImageProcessor(copy.deepcopy(image)).apply().image

    score, text = vqa_evaluator.ocr(image)
    print(f"Score: {score}")
    print(f"Text: `{text}`")

    evaluate_questions(vqa_evaluator, aesthetic_evaluator, image)


if __name__ == "__main__":
    evaluate()
