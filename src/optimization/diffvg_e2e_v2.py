import os
import kornia
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
from datasets import load_dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


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


def svg_to_png_no_resize(svg_code: str) -> Image.Image:
    png_data = cairosvg.svg2png(bytestring=svg_code.encode('utf-8'))
    img_pil = Image.open(io.BytesIO(png_data)).convert('RGB')
    return img_pil


def get_optimization_settings():
    # Create optimization settings
    settings = pydiffvg.SvgOptimizationSettings()

    lr = 5e-3

    # Configure optimization settings
    settings.global_override(["optimizer"], "Adam")
    settings.global_override(["color_lr"], lr)
    settings.global_override(["alpha_lr"], lr)
    settings.global_override(["paths", "shape_lr"], 10*lr)
    settings.global_override(["circles", "shape_lr"], 10*lr)
    settings.global_override(["transforms", "transform_lr"], 10*lr)
    
    # Configure gradient optimization settings
    settings.global_override(["gradients", "optimize_stops"], True)
    settings.global_override(["gradients", "stop_lr"], lr)
    settings.global_override(["gradients", "optimize_color"], True)
    settings.global_override(["gradients", "color_lr"], lr)
    settings.global_override(["gradients", "optimize_alpha"], True)
    settings.global_override(["gradients", "alpha_lr"], lr)
    settings.global_override(["gradients", "optimize_location"], True)
    settings.global_override(["gradients", "location_lr"], 10*lr)

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


def get_initial_svg(
    mask: torch.Tensor,
    canvas_width: int = 384,
    canvas_height: int = 384,
    num_tiles: int = 4,
    points_per_edge: int = 1,
    tile_split: int = 4
):
    # Start with the SVG header
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_width}" height="{canvas_height}" xmlns:xlink="http://www.w3.org/1999/xlink">\n'
    
    # Add a white background
    svg += f'  <path d="M 0,0 h {canvas_width} v {canvas_height} h {-canvas_width} z" fill="{rgb_to_hex(255, 255, 255)}" />\n'

    tile_size_width = canvas_width // (num_tiles * tile_split)
    tile_size_height = canvas_height // (num_tiles * tile_split)

    for i in range(num_tiles * tile_split):
        for j in range(num_tiles * tile_split):
            if mask[0, j * tile_size_height, i * tile_size_width] == 0:
                continue
            
            # if j > 0.5 * num_tiles:
            #     continue

            x = i * tile_size_width
            y = j * tile_size_height
            width = tile_size_width
            height = tile_size_height
            
            fill = rgb_to_hex(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            # points_per_edge = random.randint(1, 3)
            points_per_edge = 2

            # Create path with more control points
            if points_per_edge <= 1:
                # Original rectangle with 4 points
                svg += f'  <path d="M {x},{y} h {width} v {height} h {-width} z" fill="{fill}" />\n'
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

                svg += f'  <path d="{path_data}" fill="{fill}" />\n'

    # # Add text SVG
    # text_svg = text_to_svg("A", svg_width=canvas_width, svg_height=canvas_height, color=(255, 255, 255), x_position_frac=0.1, y_position_frac=0.2, font_size=50)
    # svg += "\n".join(text_svg.split("\n")[1:-1])

    svg += "</svg>"

    with open("initial_svg.svg", "w") as f:
        f.write(svg)

    print(f"Initial Length SVG: {len(svg.encode('utf-8'))}")
    opt_svg = optimize_svg(svg)
    print(f"Optimized Length SVG: {len(opt_svg.encode('utf-8'))}")

    return svg




def optimize_diffvg(
    bg_image: Image.Image,
    bg_svg: str,
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
    num_tiles: int = 12,
    tile_split: int = 4
) -> Image.Image:
    pydiffvg.set_use_gpu(torch.cuda.is_available())

    # bg_image = bg_image.resize((canvas_width, canvas_height))
    # bg_image = np.array(bg_image)
    # bg_image = torch.from_numpy(bg_image).cuda().permute(2, 0, 1).float() / 255.0

    # mask = torch.zeros((3, canvas_height, canvas_width), dtype=torch.float32, device="cuda:0")
    # mask[:, 32:96, 32:96] = 1

    # initial_svg = get_initial_svg(mask, canvas_width, canvas_height, num_tiles=num_tiles, tile_split=tile_split)

    with open("output_aest.svg") as f:
        initial_svg = f.read()

    initial_svg = initial_svg.replace('style="', 'style="fill-opacity:0.5;')
    bg_svg = bg_svg.replace("<polygon", '<polygon id="text-path-0"')
    bg_svg = "\n".join([x for x in bg_svg.split("\n") if "<rect" not in x])
    
    initial_svg = merge_svgs(bg_svg, initial_svg)

    temp_svg_path = "/tmp/initial_svg.svg"
    with open(temp_svg_path, "w") as f:
        f.write(initial_svg)

    settings = get_optimization_settings()

    text_path_ids = [f"text-path-{i}" for i in range(200)] + [f"background-{i}" for i in range(10)]
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

    best_svg = optim_svg.write_xml()
    best_val_loss = -1e8
    
    grad_accumulation_steps = 1

    pbar = tqdm(total=num_iterations)

    for iter_idx in range(num_iterations):
        optim_svg.zero_grad()
        image = optim_svg.render(seed=iter_idx)
        img = image[:, :, :3].permute(2, 0, 1).clamp(0, 1)

        # img = img * mask + bg_image * (1 - mask)
        
        # img_bkp = img.detach().clone()
        
        crop_frac = 0.05
        random_size = int(random.uniform(1.0 - crop_frac, 1.0) * image.shape[1])
        img = kornia.augmentation.RandomCrop((random_size, random_size))(img.unsqueeze(0)).squeeze(0)

        img = apply_preprocessing_torch(img)

        loss = aesthetic_score_gradient(aesthetic_evaluator, img).mean()

        if iter_idx == 0 or (iter_idx + 1) % validation_steps == 0:
            torch.cuda.empty_cache()

            # cur_svg = optim_svg.write_xml()
            # pil_image = Image.fromarray((img_bkp * 255).detach().permute(1, 2, 0).cpu().numpy().astype(np.uint8)).convert("RGB")

            # cur_svg = merge_svgs(bg_svg, optim_svg.write_xml())
            cur_svg = optim_svg.write_xml()
            pil_image = svg_to_png_no_resize(cur_svg)
            
            # with open("output.svg", "w") as f:
            #     f.write(cur_svg)
            
            # pil_image.save("output.png")

            pil_image = ImageProcessor(pil_image).apply().image
            val_loss = aesthetic_score_original(aesthetic_evaluator, pil_image)

            if val_loss > best_val_loss:
                best_val_loss = val_loss
                best_svg = cur_svg
            
        pbar.set_description(
            f"It {iter_idx}/{num_iterations} | "
            f"Loss: {loss.item():.3f} | "
            f"Val Loss: {val_loss:.3f} | "
        )
        pbar.update(1)

        loss = -loss / grad_accumulation_steps
        loss.backward()
        
        if (iter_idx + 1) % grad_accumulation_steps == 0:
            optim_svg.step()

    print(f"Best loss: {best_val_loss}")

    return best_svg



def merge_svgs(bg_svg: str, svg: str):
    svg = svg.strip().split("\n")[2:-1]
    # svg = [
    #     '<defs>',
    #     '<clipPath id="cut">',
    #     '<rect x="32" y="32" width="64" height="64" />',
    #     '</clipPath>',
    #     '</defs>',
    #     '<g clip-path="url(#cut)">',
    #     *svg,
    #     '</g>'
    # ]
    svg = "\n".join(svg)
    
    bg_svg = convert_polygons_to_paths(bg_svg)
    
    svg = bg_svg + svg
    svg = svg.replace("</svg>", "") + "</svg>"

    return svg


def evaluate():
    seed_everything(42)

    vqa_evaluator = VQAEvaluator()
    vqa_evaluator.model.eval()
    vqa_evaluator.model.requires_grad_(False)

    aesthetic_evaluator = AestheticEvaluator()
    aesthetic_evaluator.predictor.eval()
    aesthetic_evaluator.predictor.requires_grad_(False)
    aesthetic_evaluator.clip_model.eval()
    aesthetic_evaluator.clip_model.requires_grad_(False)

    # df = pd.read_parquet("/home/mpf/code/kaggle/draw/src/subs/train_df_poly_100_bottom.parquet")
    # df = df[df["split"] == "validation"].reset_index(drop=True)

    df = pd.read_parquet("/home/mpf/code/kaggle/draw/src/bkp_subs/train_df_org_poly_100_bottom.parquet")

    mean_score_gt = 0
    mean_score_gt_vqa = 0
    mean_score_gt_aest = 0

    for index, row in tqdm(df.iterrows(), total=len(df)):
        print(row["description"])
        
        run_optimization = True
        if run_optimization:
            png_data = cairosvg.svg2png(bytestring=row["svg"].encode('utf-8'))
            bg_image = Image.open(io.BytesIO(png_data)).convert('RGB')
            
            svg = optimize_diffvg(
                bg_image=bg_image,
                bg_svg=row["svg"],
                vqa_evaluator=vqa_evaluator,
                aesthetic_evaluator=aesthetic_evaluator,
                target_text=row["description"],
                questions=json.loads(row["question"]),
                choices_list=json.loads(row["choices"]),
                answers=json.loads(row["answer"]),
                canvas_width=384,
                canvas_height=384,
                num_iterations=100,
                validation_steps=20,
                num_tiles=32,
                tile_split=1
            )

        else:
            with open("output.svg") as f:
                svg = f.read()

        # svg = merge_svgs(row["svg"], svg)

        with open("output.svg", "w") as f:
            f.write(svg)
            
        opt_svg = optimize_svg(svg)

        with open("output_opt.svg", "w") as f:
            f.write(opt_svg)

        print(f"Length SVG: {len(opt_svg.encode('utf-8'))}")

        image = svg_to_png_no_resize(opt_svg)
        image.save("output.png")
        
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
