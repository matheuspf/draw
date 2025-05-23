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
    score_gradient_ocr_1,
    score_gradient_ocr_2,
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


def format_prompt(prompt: str) -> str:
    return "<image>What is written in the image?\n" + prompt + "<eos>"


def score_gradient_ocr_3(
    evaluator: VQAEvaluator,
    image: torch.Tensor,
    target: list[str],
) -> torch.Tensor:
    image_shape = (
        evaluator.processor.image_processor.size["height"],
        evaluator.processor.image_processor.size["width"],
    )
    image = F.interpolate(
        image.unsqueeze(0), size=image_shape, mode="bicubic", align_corners=False, antialias=True
    )
    image = (image - 0.5) / 0.5

    answer_pos = (
        evaluator.processor(
            images=Image.new("RGB", image_shape),
            text=format_prompt(""),
            return_tensors="pt",
        ).input_ids.shape[-1]
        - 1
    )
    inputs = evaluator.processor(
        images=Image.new("RGB", image_shape),
        text=format_prompt(target),
        return_tensors="pt",
        # padding="longest",
    ).to("cuda:0")

    input_ids = copy.deepcopy(inputs["input_ids"][:, :-1])
    attention_mask = copy.deepcopy(inputs["attention_mask"][:, :-1])
    labels = copy.deepcopy(input_ids)
    labels[:, :answer_pos] = -100

    outputs = evaluator.model(
        pixel_values=image.repeat(len(input_ids), 1, 1, 1),
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
    loss = outputs.loss

    return loss


def svg_to_png_no_resize(svg_code: str) -> Image.Image:
    png_data = cairosvg.svg2png(bytestring=svg_code.encode("utf-8"))
    return Image.open(io.BytesIO(png_data)).convert("RGB")


def get_optimization_settings():
    # Create optimization settings
    settings = pydiffvg.SvgOptimizationSettings()

    lr = 1e-1

    # Configure optimization settings
    settings.global_override(["optimizer"], "Adam")
    settings.global_override(["color_lr"], lr)
    settings.global_override(["alpha_lr"], lr)
    settings.global_override(["paths", "shape_lr"], 10 * lr)
    settings.global_override(["circles", "shape_lr"], 10 * lr)
    settings.global_override(["transforms", "transform_lr"], 10 * lr)

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
    target_text: str,
    canvas_width: int = 384,
    canvas_height: int = 384,
    num_tiles: int = 4,
    points_per_edge: int = 1,
):
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_width}" height="{canvas_height}">\n'
    svg += f'  <path d="M 0,0 h {canvas_width} v {canvas_height} h {-canvas_width} z" fill="{rgb_to_hex(0, 0, 0)}" />\n'
    # svg += f'  <path id="background-0" d="M 0,0 h {canvas_width} v {canvas_height} h {-canvas_width} z" fill="{rgb_to_hex(255, 255, 255)}" />\n'
    # svg += f'  <path d="M 0,0 h {canvas_width} v {canvas_height//2} h {-canvas_width} z" fill="{rgb_to_hex(255, 255, 255)}" />\n'
    # svg += f'  <path d="M 0,0 h {canvas_width} v {int(0.5*canvas_height)} h {-canvas_width} z" fill="{rgb_to_hex(255, 255, 255)}" />\n'

    for i in range(num_tiles):
        for j in range(num_tiles):
            # steps = 4
            # if not(i < steps or j < steps or i >= num_tiles - steps or j >= num_tiles - steps):
            #     continue

            # if j > num_tiles // 2:
            #     continue

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
                path_data = f"M {x},{y} "

                # Top edge (left to right)
                for p in range(1, points_per_edge):
                    path_data += f"L {x + (width * p / points_per_edge)},{y} "
                path_data += f"L {x + width},{y} "

                # Right edge (top to bottom)
                for p in range(1, points_per_edge):
                    path_data += f"L {x + width},{y + (height * p / points_per_edge)} "
                path_data += f"L {x + width},{y + height} "

                # Bottom edge (right to left)
                for p in range(1, points_per_edge):
                    path_data += f"L {x + width - (width * p / points_per_edge)},{y + height} "
                path_data += f"L {x},{y + height} "

                # Left edge (bottom to top)
                for p in range(1, points_per_edge):
                    path_data += f"L {x},{y + height - (height * p / points_per_edge)} "
                path_data += "z"

                svg += f'  <path d="{path_data}" fill="{random_color}" />\n'
            # svg += f'  <rect x="{x}" y="{y}" width="{canvas_width // num_tiles}" height="{canvas_height // num_tiles}" fill="{random_color}" />\n'
            # svg += f'  <ellipse cx="{x + canvas_width // num_tiles // 2}" cy="{y + canvas_height // num_tiles // 2}" rx="{canvas_width // num_tiles // 2}" ry="{canvas_height // num_tiles // 2}" fill="{random_color}" />\n'

    # text_svg = text_to_svg(text="www.", svg_width=canvas_width, svg_height=canvas_height, x_position_frac=0.1, y_position_frac=0.1, font_size=50, color=(0, 0, 0))
    # svg += "\n".join(text_svg.split("\n")[1:-1])

    text_svg = text_to_svg(target_text, svg_width=canvas_width, svg_height=canvas_height, x_position_frac=0.2, y_position_frac=0.2, font_size=50, color=(255, 0, 0))
    svg += "\n".join(text_svg.split("\n")[1:-1])
    svg += "</svg>"
    # svg = optimize_svg(svg)

    with open("initial_svg.svg", "w") as f:
        f.write(svg)

    print(f"Initial Length SVG: {len(svg.encode('utf-8'))}")

    return svg


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
    points_per_edge: int = 1,
) -> Image.Image:
    vqa_evaluator.model.eval()
    vqa_evaluator.model.requires_grad_(False)
    aesthetic_evaluator.predictor.eval()
    aesthetic_evaluator.predictor.requires_grad_(False)
    aesthetic_evaluator.clip_model.eval()
    aesthetic_evaluator.clip_model.requires_grad_(False)

    pydiffvg.set_use_gpu(torch.cuda.is_available())

    svg_content = get_initial_svg(
        target_text, canvas_width, canvas_height, num_tiles=32, points_per_edge=points_per_edge
    )

    temp_svg_path = "/tmp/initial_svg.svg"
    with open(temp_svg_path, "w") as f:
        f.write(svg_content)

    settings = get_optimization_settings()

    text_path_ids = [f"text-path-{i}" for i in range(100)] + [f"background-{i}" for i in range(10)]
    for text_id in text_path_ids:
        text_settings = settings.undefault(text_id)
        text_settings["paths"]["optimize_points"] = False
        text_settings["optimize_color"] = False
        text_settings["optimize_alpha"] = False
        text_settings["optimize_transforms"] = False
        # text_settings["paths"]["shape_lr"] = 1e-1
        # text_settings["transforms"]["transform_lr"] = 1e-1
        # text_settings["color_lr"] = 1e-2
        # text_settings["alpha_lr"] = 1e-2

    optim_svg = pydiffvg.OptimizableSvg(
        temp_svg_path, settings, optimize_background=True, verbose=False, device="cuda:0"
    )

    validator = svg_constraints.SVGConstraints(max_svg_size=10000)
    best_svg = optimize_svg(optim_svg.write_xml())
    best_val_loss = 1e8

    pbar = tqdm(total=num_iterations)

    for iter_idx in range(num_iterations):
        optim_svg.zero_grad()
        image = optim_svg.render(seed=iter_idx)

        img = image[:, :, :3].permute(2, 0, 1)
        # img = F.interpolate(
        #     img.clamp(0, 1).unsqueeze(0),
        #     size=(384, 384),
        #     mode="bicubic",
        #     align_corners=False,
        #     antialias=True,
        # )[0].clamp(0, 1)
        # img = apply_preprocessing_torch(img)

        # loss = score_gradient_ocr(vqa_evaluator, img).mean()
        # loss = score_gradient_ocr_3(vqa_evaluator, img, target="www.")

        loss = score_gradient_ocr_1(vqa_evaluator, img, text="", response=None)
        loss.backward()

        loss2 = score_gradient_ocr_1(vqa_evaluator, apply_preprocessing_torch(img), text="", response=target_text)
        loss2.backward()

        optim_svg.step()

        if iter_idx == 0 or (iter_idx + 1) % validation_steps == 0:
            cur_svg = optimize_svg(optim_svg.write_xml())
            pil_image = svg_to_png_no_resize(cur_svg)
            # pil_image = ImageProcessor(copy.deepcopy(pil_image)).apply().image

            val_loss, text = vqa_evaluator.ocr(pil_image)
            val_loss = 1.0 - val_loss
            text = text.replace("\n", "/n")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_svg = cur_svg

        pbar.set_description(
            f"It {iter_idx}/{num_iterations} | "
            f"Loss: {loss.item():.3f} | "
            f"Val Loss: {val_loss:.3f} | "
            f"Text: `{text[:10]}` | "
        )
        pbar.update(1)

        if val_loss < 0.001:
            break

    best_svg = optim_svg.write_xml()

    return best_svg


def evaluate():
    seed_everything(42)

    vqa_evaluator = VQAEvaluator()
    aesthetic_evaluator = AestheticEvaluator()
    
    mean_score = 0.0   
    run_inference = True
    if run_inference:
        df = pd.read_parquet("/home/mpf/code/kaggle/draw/question_generation_results.parquet")
        df = df[df["set"] == "test"]#.iloc[1:2]

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
                canvas_width=384,
                canvas_height=384,
                questions=gt_questions_list["questions"],
                choices_list=gt_questions_list["choices_list"],
                answers=gt_questions_list["answers"],
                num_iterations=100,
                validation_steps=10,
                points_per_edge=3,
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

            image = svg_to_png_no_resize(opt_svg)
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
            
            mean_score += score_gt[0]

    mean_score /= len(df)
    print(f"Mean Score: {mean_score}")


if __name__ == "__main__":
    evaluate()
