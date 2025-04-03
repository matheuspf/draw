import os
import torch.nn as nn
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
    _check_inputs,
    _get_choice_tokens,
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


def svg_to_png_no_resize(svg_code: str) -> Image.Image:
    png_data = cairosvg.svg2png(bytestring=svg_code.encode('utf-8'))
    return Image.open(io.BytesIO(png_data)).convert('RGB')


svg_constraints = kagglehub.package_import("metric/svg-constraints", bypass_confirmation=True)



def format_prompt(prompt: str) -> str:
    return "<image>ocr\n" + prompt

def ocr_score(
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

    len_empty_prompt = evaluator.processor(
        images=Image.new("RGB", image_shape),
        text=format_prompt(""),
        return_tensors="pt",
    ).input_ids.shape[-1]
    inputs = evaluator.processor(
        images=Image.new("RGB", image_shape),
        text=format_prompt(target),
        return_tensors="pt",
        # padding="longest",
    ).to("cuda:0")
    inputs["pixel_values"] = image.repeat(1, 1, 1, 1)
    
    inputs["labels"] = copy.deepcopy(inputs.input_ids)
    inputs["labels"][:, :len_empty_prompt] = -100

    outputs = evaluator.model(**inputs)
    loss = outputs.loss

    return -loss



def get_optimization_settings():
    # Create optimization settings
    settings = pydiffvg.SvgOptimizationSettings()

    lr = 5e-2

    # Configure optimization settings
    settings.global_override(["optimizer"], "Adam")
    settings.global_override(["color_lr"], lr)
    settings.global_override(["alpha_lr"], lr)
    settings.global_override(["paths", "shape_lr"], 10*lr)
    settings.global_override(["circles", "shape_lr"], 10*lr)
    settings.global_override(["transforms", "transform_lr"], 10*lr)

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


def get_initial_svg(
    target_text: str, canvas_width: int = 384, canvas_height: int = 384, num_tiles: int = 4
):
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_width}" height="{canvas_height}">\n'
    svg += f'  <path id="bg-2" d="M 0,0 h {canvas_width} v {canvas_height} h {-canvas_width} z" fill="{rgb_to_hex(255, 255, 255)}" />\n'
    text_svg = text_to_svg("ABCDEF", svg_width=canvas_width, svg_height=canvas_height, x_position_frac=0.4, y_position_frac=0.4, color=(0, 0, 0), font_size=100)
    svg += "\n".join(text_svg.split("\n")[1:-1])
    svg += "</svg>"

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
) -> Image.Image:
    vqa_evaluator.model.eval()
    vqa_evaluator.model.requires_grad_(False)
    aesthetic_evaluator.predictor.eval()
    aesthetic_evaluator.predictor.requires_grad_(False)
    aesthetic_evaluator.clip_model.eval()
    aesthetic_evaluator.clip_model.requires_grad_(False)

    pydiffvg.set_use_gpu(torch.cuda.is_available())

    svg_content = get_initial_svg(target_text, canvas_width, canvas_height, num_tiles=32)

    temp_svg_path = "/tmp/initial_svg.svg"
    with open(temp_svg_path, "w") as f:
        f.write(svg_content)

    settings = get_optimization_settings()

    # text_path_ids = [f"text-path-{i}" for i in range(100)] + [f"background-{i}" for i in range(10)] + [f"bg-2"]
    # for text_id in text_path_ids:
    #     text_settings = settings.undefault(text_id)
    #     text_settings["paths"]["optimize_points"] = True
    #     text_settings["optimize_color"] = True
    #     text_settings["optimize_alpha"] = True
    #     text_settings["optimize_transforms"] = True
    #     text_settings["paths"]["shape_lr"] = 1.0
    #     text_settings["transforms"]["transform_lr"] = 1.0
    #     text_settings["color_lr"] = 0.1
    #     text_settings["alpha_lr"] = 0.1

    optim_svg = pydiffvg.OptimizableSvg(
        temp_svg_path, settings, optimize_background=True, verbose=False, device="cuda:0"
    )

    validator = svg_constraints.SVGConstraints(max_svg_size=10000)
    best_svg = optim_svg.write_xml()
    best_val_loss = -1e8

    pbar = tqdm(total=num_iterations)

    for iter_idx in range(num_iterations):
        optim_svg.zero_grad()
        image = optim_svg.render(seed=iter_idx)
        img = image[:, :, :3].permute(2, 0, 1).clamp(0, 1)

        # img = apply_preprocessing_torch(img)

        loss = ocr_score(vqa_evaluator, img, "X")

        if iter_idx == 0 or (iter_idx + 1) % validation_steps == 0:
            cur_svg = optimize_svg(optim_svg.write_xml())
            pil_image = svg_to_png_no_resize(cur_svg)
            # pil_image = ImageProcessor(copy.deepcopy(pil_image)).apply().image

            torch.cuda.empty_cache()
            val_loss, ocr_text = vqa_evaluator.ocr(pil_image)
            ocr_text = ocr_text.replace("\n", "")

            if val_loss > best_val_loss:
                best_val_loss = val_loss
                best_svg = cur_svg
            
        pbar.set_description(
            f"It {iter_idx}/{num_iterations} | "
            f"Loss: {loss.item():.3f} | "
            f"Val Loss: {val_loss:.3f} | "
            f"Text: `{ocr_text[:10]}` | "
        )
        pbar.update(1)

        loss = -loss
        loss.backward()
        optim_svg.step()

    best_svg = optim_svg.write_xml()

    return best_svg



def evaluate():
    seed_everything(42)

    vqa_evaluator = VQAEvaluator()
    aesthetic_evaluator = AestheticEvaluator()
    
    mean_score_gt = 0
    mean_score_gen = 0

    run_inference = True
    if run_inference:
        df = pd.read_parquet("/home/mpf/code/kaggle/draw/question_generation_results.parquet")
        df = df[df["set"] == "test"]#.iloc[:1]

        for index, row in tqdm(df.iterrows(), total=len(df)):
            num_ex = 4
            description = row["description"]
            print(description)

            gt_questions_list = {
                "questions": [dct["question"] for dct in row["ground_truth"]],
                "choices_list": [dct["choices"].tolist() for dct in row["ground_truth"]],
                "answers": [dct["answer"] for dct in row["ground_truth"]]
            }
            gen_questions_list = {
                "questions": [dct["question"] for dct in row["generated"]][:num_ex],
                "choices_list": [dct["choices"].tolist() for dct in row["generated"]][:num_ex],
                "answers": [dct["answer"] for dct in row["generated"]][:num_ex]
            }
            gen_questions_list_inpt = gen_questions_list

            run_optimization = True
            if run_optimization:
                svg = optimize_diffvg(
                    vqa_evaluator=vqa_evaluator,
                    aesthetic_evaluator=aesthetic_evaluator,
                    target_text=description,
                    questions=gen_questions_list_inpt["questions"],
                    choices_list=gen_questions_list_inpt["choices_list"],
                    answers=gen_questions_list_inpt["answers"],
                    canvas_width=384,
                    canvas_height=384,
                    num_iterations=1000,
                    validation_steps=10,
                )

            else:
                with open("output.svg") as f:
                    svg = f.read()

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

            mean_score_gt += score_gt[0]
            mean_score_gen += score_gen[0]

            break

    print(f"Mean Score GT: {mean_score_gt / len(df)}")
    print(f"Mean Score Gen: {mean_score_gen / len(df)}")

if __name__ == "__main__":
    evaluate()
