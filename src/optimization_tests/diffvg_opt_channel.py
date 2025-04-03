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

def svg_to_png_no_resize(svg_code: str) -> Image.Image:
    png_data = cairosvg.svg2png(bytestring=svg_code.encode('utf-8'))
    return Image.open(io.BytesIO(png_data)).convert('RGB')


def get_optimization_settings():
    # Create optimization settings
    settings = pydiffvg.SvgOptimizationSettings()

    # Configure optimization settings
    settings.global_override(["optimizer"], "Adam")
    settings.global_override(["color_lr"], 0.1)
    settings.global_override(["alpha_lr"], 0.1)
    settings.global_override(["paths", "shape_lr"], 0.5)
    settings.global_override(["circles", "shape_lr"], 0.5)
    settings.global_override(["transforms", "transform_lr"], 0.5)

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


def get_initial_svg(
    target_text: str, canvas_width: int = 384, canvas_height: int = 384, num_tiles: int = 4
):
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_width}" height="{canvas_height}">\n'
    svg += f'  <path id="background-0" d="M 0,0 h {canvas_width} v {canvas_height} h {-canvas_width} z" fill="{rgb_to_hex(0, 0, 0)}" />\n'
    svg += f'  <path id="background-1" d="M 0,0 h {canvas_width} v {int(0.6*canvas_height)} h {-canvas_width} z" fill="{rgb_to_hex(255, 255, 255)}" />\n'
    svg += f'  <path id="bg-0" d="M 0,0 h {canvas_width} v {int(0.6*canvas_height)} h {-canvas_width} z" fill="{rgb_to_hex(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))}" />\n'

    for i in range(num_tiles):
        for j in range(num_tiles):
            # steps = 4
            # if not(i < steps or j < steps or i >= num_tiles - steps or j >= num_tiles - steps):
            #     continue

            if j > num_tiles * 8/10:
                continue

            x = i * (canvas_width // num_tiles)
            y = j * (canvas_height // num_tiles)
            width = canvas_width // num_tiles
            height = canvas_height // num_tiles
            random_color = rgb_to_hex(
                random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
            )
            
            points_per_edge = random.randint(1, 5)

            # Create path with more control points
            if points_per_edge <= 1:
                # Original rectangle with 4 points
                svg += f'  <path d="M {x},{y} h {width} v {height} h {-width} z" fill="{random_color}" />\n'
            else:
                # Rectangle with subdivided edges for more control points
                path_data = f'M {x},{y} '
                
                # Top edge (left to right)
                for p in range(1, points_per_edge):
                    path_data += f'L {x + (width * p / points_per_edge)},{y} '
                path_data += f'L {x + width},{y} '
                
                # Right edge (top to bottom)
                for p in range(1, points_per_edge):
                    path_data += f'L {x + width},{y + (height * p / points_per_edge)} '
                path_data += f'L {x + width},{y + height} '
                
                # Bottom edge (right to left)
                for p in range(1, points_per_edge):
                    path_data += f'L {x + width - (width * p / points_per_edge)},{y + height} '
                path_data += f'L {x},{y + height} '
                
                # Left edge (bottom to top)
                for p in range(1, points_per_edge):
                    path_data += f'L {x},{y + height - (height * p / points_per_edge)} '
                path_data += 'z'
                
                svg += f'  <path d="{path_data}" fill="{random_color}" />\n'
            # svg += f'  <rect x="{x}" y="{y}" width="{canvas_width // num_tiles}" height="{canvas_height // num_tiles}" fill="{random_color}" />\n'
            # svg += f'  <ellipse cx="{x + canvas_width // num_tiles // 2}" cy="{y + canvas_height // num_tiles // 2}" rx="{canvas_width // num_tiles // 2}" ry="{canvas_height // num_tiles // 2}" fill="{random_color}" />\n'

    # text_svg = text_to_svg(target_text, svg_width=canvas_width, svg_height=canvas_height)
    # svg += "\n".join(text_svg.split("\n")[1:-1])
    svg += "</svg>"

    with open("initial_svg.svg", "w") as f:
        f.write(svg)
    
    print(f"Initial Length SVG: {len(svg.encode('utf-8'))}")

    return svg


def load_text_svg(
    svg_path: str, target_text: str, canvas_width: int = 384, canvas_height: int = 384, num_tiles: int = 4
):
    with open(svg_path) as f:
        svg = f.read().split("\n")
    
    text_svg = text_to_svg(target_text, svg_width=canvas_width, svg_height=canvas_height).split("\n")

    new_svg = [l for l in svg if not "text-path-" in l]
    new_svg += [l for l in text_svg if "text-path-" in l]
    new_svg = "\n".join(new_svg)
    new_svg = new_svg.replace("</svg>\n", "")
    new_svg += "</svg>"

    with open("initial_svg.svg", "w") as f:
        f.write(new_svg)

    return new_svg


def load_optmizer(svg: str, settings: pydiffvg.SvgOptimizationSettings) -> pydiffvg.OptimizableSvg:
    temp_svg_path = "/tmp/initial_svg.svg"
    with open(temp_svg_path, "w") as f:
        f.write(svg)
    
    opt = pydiffvg.OptimizableSvg(temp_svg_path, settings, optimize_background=True, verbose=False, device="cuda:0")
    return opt


def merge_svg(svg_bg: str, svg_text: str) -> str:
    svg_bg = svg_bg.strip().split("\n")
    svg_text = svg_text.strip().split("\n")
    svg_new = svg_bg[:-1] + svg_text[2:]
    svg_new = "\n".join(svg_new)

    return svg_new


def merge_images(img_bg: torch.Tensor, img_text: torch.Tensor) -> torch.Tensor:
    img_bg = img_bg[..., :1]
    img_text = img_text[..., 1:]
    img = torch.cat([img_bg, img_text], dim=-1)
    return img

def optimize_diffvg(
    vqa_evaluator: VQAEvaluator,
    aesthetic_evaluator: AestheticEvaluator,
    target_text: str,
    questions: list[str],
    choices_list: list[list[str]],
    answers: list[str],
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

    svg_bg = get_initial_svg(target_text, canvas_width, canvas_height, num_tiles=64)
    svg_text = text_to_svg(target_text, svg_width=canvas_width, svg_height=canvas_height)

    settings_bg = get_optimization_settings()
    settings_text = get_optimization_settings()
    
    settings_text.global_override(["optimize_color"], False)
    settings_text.global_override(["optimize_alpha"], False)
    settings_text.global_override(["paths", "optimize_points"], False)
    settings_text.global_override(["circles", "optimize_center"], False)
    settings_text.global_override(["circles", "optimize_radius"], False)
    settings_text.global_override(["transforms", "optimize_transforms"], False)

    opt_bg = load_optmizer(svg_bg, settings_bg)
    opt_text = load_optmizer(svg_text, settings_text)
    

    validator = svg_constraints.SVGConstraints(max_svg_size=10000)
    best_svg = merge_svg(opt_bg.write_xml(), opt_text.write_xml())
    best_val_loss = 1e8

    pbar = tqdm(total=num_iterations)

    for iter_idx in range(num_iterations):
        opt_bg.zero_grad()
        opt_text.zero_grad()
        
        img_bg = opt_bg.render(seed=iter_idx)
        img_text = opt_text.render(seed=iter_idx)
        image = merge_images(img_bg, img_text)
        
        img = image[:, :, :3].permute(2, 0, 1)
        # img = F.interpolate(
        #     img.clamp(0, 1).unsqueeze(0),
        #     size=(384, 384),
        #     mode="bicubic",
        #     align_corners=False,
        #     antialias=True,
        # )[0].clamp(0, 1)
        img_proc = apply_preprocessing_torch(img)

        # ocr_loss = score_gradient_ocr(vqa_evaluator, img, text="", response="<eos> " + target_text).mean()
        ocr_loss = score_gradient_ocr(vqa_evaluator, img, text="", response=None).mean()
        aest_loss = 1.0 - aesthetic_score_gradient(aesthetic_evaluator, img_proc).mean()

        loss = ocr_loss# + 2  * aest_loss

        if iter_idx == 0 or (iter_idx + 1) % validation_steps == 0:
            cur_svg = optimize_svg(merge_svg(opt_bg.write_xml(), opt_text.write_xml()))
            pil_image = svg_to_png_no_resize(cur_svg)
            pil_image_proc = ImageProcessor(copy.deepcopy(pil_image)).apply().image

            val_loss_ocr, text = vqa_evaluator.ocr(pil_image)
            val_loss_ocr = 1.0 - val_loss_ocr
            text = text.replace("\n", "")

            val_loss_aest = 1.0 - aesthetic_score_original(aesthetic_evaluator, pil_image )

            # val_loss = val_loss_ocr + val_loss_aest
            val_loss = val_loss_ocr

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_svg = merge_svg(opt_bg.write_xml(), opt_text.write_xml())

        pbar.set_description(
            f"It {iter_idx}/{num_iterations} | "
            f"Loss: {loss.item():.3f} | "
            f"OCR Loss: {ocr_loss.item():.3f} | "
            f"Aest Loss: {aest_loss.item():.3f} | "
            f"Val Loss: {val_loss:.3f} | "
            f"OCR Val Loss: {val_loss_ocr:.3f} | "
            f"Aest Val Loss: {val_loss_aest:.3f} | "
            f"Text: `{text[:10]}` | "
            # f"LR: {scheduler.get_last_lr()[0]:.4f}"
        )
        pbar.update(1)

        if val_loss < 0.001:
            break

        loss.backward()
        opt_bg.step()
        opt_text.step()


    best_svg = merge_svg(opt_bg.write_xml(), opt_text.write_xml())

    return best_svg


def evaluate():
    seed_everything(42)

    vqa_evaluator = VQAEvaluator()
    aesthetic_evaluator = AestheticEvaluator()

    run_inference = True
    if run_inference:
        df = pd.read_parquet("/home/mpf/code/kaggle/draw/question_generation_results.parquet")
        df = df[df["set"] == "test"].iloc[:1]

        for index, row in tqdm(df.iterrows(), total=len(df)):
            description = row["description"]
            row["ground_truth"] = row["ground_truth"][:4]
            row["generated"] = row["generated"][:4]

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

            svg = optimize_diffvg(
                vqa_evaluator=vqa_evaluator,
                aesthetic_evaluator=aesthetic_evaluator,
                target_text=description,
                questions=gt_questions_list["questions"],
                choices_list=gt_questions_list["choices_list"],
                answers=gt_questions_list["answers"],
                num_iterations=1000,
                validation_steps=10,
            )

            # with open("output.svg") as f:
            #     svg = f.read()

            with open("output.svg", "w") as f:
                f.write(svg)

            print(f"Length SVG: {len(svg.encode('utf-8'))}")

            opt_svg = optimize_svg(svg)

            with open("output_opt.svg", "w") as f:
                f.write(opt_svg)

            print(f"Length SVG Optimized: {len(opt_svg.encode('utf-8'))}")

            image = svg_to_png(opt_svg)
            image.save("output.png")
            
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

            print("\n\n")
            print(gt_questions_list["questions"])



if __name__ == "__main__":
    evaluate()
