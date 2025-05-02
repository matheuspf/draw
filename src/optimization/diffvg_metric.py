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
from src.utils import optimize_svg, svg_to_png, create_random_svg, displace_svg_paths
from src.text_to_svg import text_to_svg, rgb_to_hex
import pydiffvg
import kagglehub


def svg_to_png_no_resize(svg_code: str) -> Image.Image:
    png_data = cairosvg.svg2png(bytestring=svg_code.encode('utf-8'))
    return Image.open(io.BytesIO(png_data)).convert('RGB')


svg_constraints = kagglehub.package_import("metric/svg-constraints", bypass_confirmation=True)


def vqa_caption_score_gradient(
    evaluator: VQAEvaluator,
    image: torch.Tensor,
    target_text: str,
    prefix: str = "<image>caption en\n",
) -> torch.Tensor:
    image_shape = (
        evaluator.processor.image_processor.size["height"],
        evaluator.processor.image_processor.size["width"],
    )
    image = F.interpolate(
        image.unsqueeze(0), size=image_shape, mode="bicubic", align_corners=False, antialias=True
    )
    image = (image - 0.5) / 0.5
    
    inputs = evaluator.processor(
        images=Image.new("RGB", image_shape),
        text=prefix,
        return_tensors="pt",
        suffix=target_text,
    ).to("cuda:0")
    inputs["pixel_values"] = image
    
    outputs = evaluator.model(**inputs)
    loss = outputs.loss

    return loss



def get_optimization_settings():
    # Create optimization settings
    settings = pydiffvg.SvgOptimizationSettings()

    lr = 1e-2

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


def get_initial_svg(svg_path: str):
    with open(svg_path, "r") as f:
        svg_content = f.read()
    return svg_content


def convert_polygons_to_paths(svg_string):
    """
    Convert SVG polygon and polyline elements to path elements.
    
    Args:
        svg_string (str): The SVG content as a string
        
    Returns:
        str: The converted SVG with polygons and polylines replaced by paths
    """
    import re
    
    # Convert polygon points to path d with closing 'z'
    svg_string = re.sub(
        r'<polygon([\w\W]+?)points=(["\'])([\.\d, -]+?)(["\'])', 
        r'<path\1d=\2M\3z\4', 
        svg_string
    )
    
    # Convert polyline points to path d without closing
    svg_string = re.sub(
        r'<polyline([\w\W]+?)points=(["\'])([\.\d, -]+?)(["\'])', 
        r'<path\1d=\2M\3\4', 
        svg_string
    )
    
    return svg_string


def optimize_diffvg(
    vqa_evaluator: VQAEvaluator,
    aesthetic_evaluator: AestheticEvaluator,
    initial_svg: str,
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

    initial_svg = convert_polygons_to_paths(initial_svg)
    temp_svg_path = "/tmp/initial_svg.svg"
    with open(temp_svg_path, "w") as f:
        f.write(initial_svg)

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
        temp_svg_path, settings, optimize_background=False, verbose=False, device="cuda:0"
    )

    validator = svg_constraints.SVGConstraints(max_svg_size=10000)
    best_svg = optim_svg.write_xml()
    best_val_loss = -1e8

    pbar = tqdm(total=num_iterations)

    for iter_idx in range(num_iterations):
        optim_svg.zero_grad()
        image = optim_svg.render(seed=iter_idx)
        img = image[:, :, :3].permute(2, 0, 1).clamp(0, 1)

        img = apply_preprocessing_torch(img)

        # vqa_loss = vqa_score_gradient(vqa_evaluator, img, questions, choices_list, answers)
        vqa_loss = vqa_caption_score_gradient(vqa_evaluator, img, target_text)
        aest_loss = aesthetic_score_gradient(aesthetic_evaluator, img).mean()

        # loss = vqa_loss
        loss = 1.0 * vqa_loss + aest_loss
        
        # loss = 1.0 - harmonic_mean_grad(vqa_loss, aest_loss, beta=0.5)
        # loss = (vqa_loss + aest_loss).mean()

        loss.backward()
        optim_svg.step()

        if iter_idx == 0 or (iter_idx + 1) % validation_steps == 0:
            cur_svg = optimize_svg(optim_svg.write_xml())
            pil_image = svg_to_png_no_resize(cur_svg)
            # pil_image = Image.fromarray((image[:, :, :3] * 255).detach().cpu().numpy().astype(np.uint8)).convert("RGB")

            torch.cuda.empty_cache()
            val_loss, vqa_val_loss, aest_val_loss, ocr_loss, ocr_text = score_original(
                vqa_evaluator,
                aesthetic_evaluator,
                pil_image,
                questions,
                choices_list,
                answers,
            )
            ocr_text = ocr_text.replace("\n", "")

            if val_loss > best_val_loss:
                best_val_loss = val_loss
                best_svg = cur_svg
            
        pbar.set_description(
            f"It {iter_idx}/{num_iterations} | "
            # f"VQA Loss: {vqa_loss.item():.3f} | "
            # f"Aest Loss: {aest_loss.item():.3f} | "
            f"Loss: {loss.item():.3f} | "
            f"VQA Val Loss: {vqa_val_loss:.3f} | "
            f"Aest Val Loss: {aest_val_loss:.3f} | "
            f"Val Loss: {val_loss:.3f} | "
            f"OCR Loss: {ocr_loss:.3f} | "
            # f"Text: `{ocr_text[:10]}` | "
        )
        pbar.update(1)

    # best_svg = optim_svg.write_xml()

    return best_svg


def describe(evaluator, image):
    inputs = (
        evaluator.processor(
            text='<image>describe en\n',
            images=image,
            return_tensors='pt',
        )
        .to(torch.float16)
        .to(evaluator.model.device)
    )
    input_len = inputs['input_ids'].shape[-1]

    with torch.inference_mode():
        outputs = evaluator.model.generate(**inputs, max_new_tokens=32, do_sample=False)
        outputs = outputs[0][input_len:]
        decoded = evaluator.processor.decode(outputs, skip_special_tokens=True)

    return decoded



def evaluate():
    seed_everything(42)

    vqa_evaluator = VQAEvaluator()
    aesthetic_evaluator = AestheticEvaluator()
    
    mean_score_gt = 0
    mean_score_gt_aest = 0
    mean_score_gt_vqa = 0
    mean_score_gen = 0

    # df = pd.read_parquet("/home/mpf/code/kaggle/draw/question_generation_results.parquet")
    # df = df[df["set"] == "test"]#.iloc[:1]

    # df_svg = pd.read_parquet("/home/mpf/code/kaggle/draw/src/subs/train_df.parquet")
    # df_svg = df_svg[df_svg["id"].isin(df["id"])]
    # df_svg = df_svg[["id", "svg"]]
    
    # df = df.merge(df_svg, on="id", how="right")
    
    
    # df = pd.read_parquet("/home/mpf/code/kaggle/draw/src/subs/train_df_multi_100.parquet")
    df = pd.read_parquet("/home/mpf/code/kaggle/draw/src/subs/train_df_poly_100_bottom.parquet")
    df = df[df["split"] == "validation"].reset_index(drop=True)
    # df = df[["id", "svg"]].reset_index(drop=True)
    # df_org = pd.read_parquet("/home/mpf/code/kaggle/draw/src/data/generated/qa_dataset_train.parquet")
    # df = df.merge(df_org, on="id", how="left")
    # df = df[df["split"] == "validation"].reset_index(drop=True)


    for index, row in tqdm(df.iterrows(), total=len(df)):
        num_ex = 4
        description = row["description"]
        print(description)

        # gt_questions_list = {
        #     "questions": [dct["question"] for dct in row["ground_truth"]][:num_ex],
        #     "choices_list": [dct["choices"].tolist() for dct in row["ground_truth"]][:num_ex],
        #     "answers": [dct["answer"] for dct in row["ground_truth"]][:num_ex]
        # }
        # gen_questions_list = {
        #     "questions": [dct["question"] for dct in row["generated"]][:num_ex],
        #     "choices_list": [dct["choices"].tolist() for dct in row["generated"]][:num_ex],
        #     "answers": [dct["answer"] for dct in row["generated"]][:num_ex]
        # }
        # gen_questions_list_inpt = gt_questions_list

        run_optimization = False
        if run_optimization:
            svg = optimize_diffvg(
                vqa_evaluator=vqa_evaluator,
                aesthetic_evaluator=aesthetic_evaluator,
                target_text=description,
                # initial_svg=row["svg"],
                initial_svg=open("/home/mpf/code/kaggle/draw/output_aest.svg").read(),
                questions=json.loads(row["question"]),
                choices_list=json.loads(row["choices"]),
                answers=json.loads(row["answer"]),
                canvas_width=384,
                canvas_height=384,
                num_iterations=100,
                validation_steps=20,
            )

        else:
            with open("output_aest.svg") as f:
                svg = f.read()
            
            # svg = displace_svg_paths(svg, 32, 32, scale=1.0)

            svg = svg.strip().split("\n")[2:-1]
            svg = [
                '<defs>',
                '<clipPath id="cut">',
                '<rect x="32" y="32" width="64" height="64" />',
                # '<rect x="32" y="32" width="32" height="32" />',
                # '<rect x="32" y="320" width="32" height="32" />',
                # '<rect x="320" y="32" width="32" height="32" />',
                # '<rect x="320" y="320" width="32" height="32" />',
                '</clipPath>',
                '</defs>',
                '<g clip-path="url(#cut)">',
                *svg,
                '</g>'
            ]
            svg = "\n".join(svg)
            
            bg_svg = convert_polygons_to_paths(row["svg"])
            
            svg = bg_svg + svg
            svg = svg.replace("</svg>", "") + "</svg>"
            
            image = svg_to_png_no_resize(svg)

            # with open("output_aest_128.svg") as f:
            #     svg = f.read()
            
            # img = svg_to_png_no_resize(svg)#.resize((32, 32))
            # bg = svg_to_png_no_resize(row["svg"])#.resize((384, 384))
            

            # bg = torch.from_numpy(np.array(bg)).permute(2, 0, 1).float() / 255.0
            # img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0  
            
            # pos = 64, 64
            # bg[:, pos[0]:pos[0]+img.shape[1], pos[1]:pos[1]+img.shape[2]] = img
            # image = Image.fromarray((bg * 255).detach().permute(1, 2, 0).cpu().numpy().astype(np.uint8)).convert("RGB")

            # image.save("output.png")
            # exit()

            


        with open("output.svg", "w") as f:
            f.write(svg)

        print(f"Length SVG: {len(svg.encode('utf-8'))}")

        opt_svg = optimize_svg(svg)

        with open("output_opt.svg", "w") as f:
            f.write(opt_svg)

        print(f"Length SVG Optimized: {len(opt_svg.encode('utf-8'))}")

        # image = svg_to_png_no_resize(opt_svg)
        # image.save("output.png")

        # score_gen = score_original(
        #     vqa_evaluator,
        #     aesthetic_evaluator,
        #     image,
        #     gen_questions_list["questions"],
        #     gen_questions_list["choices_list"],
        #     gen_questions_list["answers"],
        # )

        score_gt = score_original(
            vqa_evaluator,
            aesthetic_evaluator,
            image,
            json.loads(row["question"]),
            json.loads(row["choices"]),
            json.loads(row["answer"]),
        )
        print(f"Score GT: {score_gt}")

        mean_score_gt += score_gt[0]
        mean_score_gt_vqa += score_gt[1]
        mean_score_gt_aest += score_gt[2]

    print(f"Mean Score GT: {mean_score_gt / len(df)}")
    print(f"Mean Score GT VQA: {mean_score_gt_vqa / len(df)}")
    print(f"Mean Score GT Aest: {mean_score_gt_aest / len(df)}")

if __name__ == "__main__":
    evaluate()

    
    # svg = convert_polygons_to_paths(df.iloc[0]["svg"])
    
    # # with open("output_aest_128.svg") as f:
    # #     svg = f.read()
    
    # svg = optimize_svg(svg)

    # with open("output_opt.svg", "w") as f:
    #     f.write(svg)

    # print(f"Length SVG Optimized: {len(svg.encode('utf-8'))}")
