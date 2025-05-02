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
from bs4 import BeautifulSoup


def convert_polygons_to_paths(svg_string):
    """
    Convert SVG polygon, polyline, and rect elements to path elements.
    
    Args:
        svg_string (str): The SVG content as a string
        
    Returns:
        str: The converted SVG with polygons, polylines, and rects replaced by paths
    """
    soup = BeautifulSoup(svg_string, 'xml')
    
    # Convert all rect elements to path
    for rect in soup.find_all('rect'):
        x = float(rect.get('x', 0))
        y = float(rect.get('y', 0))
        width = float(rect.get('width', 0))
        height = float(rect.get('height', 0))
        
        # Create a new path element
        path = soup.new_tag('path')
        
        # Copy all attributes from rect to path
        for attr, value in rect.attrs.items():
            if attr not in ['x', 'y', 'width', 'height']:
                path[attr] = value
                
        # Set the path data
        path['d'] = f"M {x},{y} h {width} v {height} h {-width} z"
        
        # Replace rect with path
        rect.replace_with(path)
    
    # Convert polygon and polyline (keeping your original code)
    result = str(soup)
    import re
    
    # Convert polygon points to path d with closing 'z'
    result = re.sub(
        r'<polygon([\w\W]+?)points=(["\'])([\.\d, -]+?)(["\'])', 
        r'<path\1d=\2M\3z\4', 
        result
    )
    
    # Convert polyline points to path d without closing
    result = re.sub(
        r'<polyline([\w\W]+?)points=(["\'])([\.\d, -]+?)(["\'])', 
        r'<path\1d=\2M\3\4', 
        result
    )
    
    return result



def proc_svg(svg):
    svg = convert_polygons_to_paths(svg)

    svg_lines = svg.replace(">", ">\n").strip().split("\n")
    svg_lines = [line for line in svg_lines if line.strip()]
    initial_lines = svg_lines[:2]
    svg_lines = svg_lines[2:-2]
    svg_lines = [line for line in svg_lines if not "<g" in line and not "</g" in line]
    svg_lines = [(line.replace("/>", ' fill-opacity=".5"/>') if "<path" in line and idx > 0 else line) for idx, line in enumerate(svg_lines)]
    svg_lines = initial_lines + svg_lines + ["</svg>"]

    svg = "\n".join(svg_lines)

    # svg += text_to_svg("A", x_position_frac=0.9, y_position_frac=0.9, font_size=45, color=(255, 255, 255)).split("\n")[1]
    svg += text_to_svg("O", x_position_frac=0.6, y_position_frac=0.85, font_size=60, color=(255, 255, 255)).split("\n")[1]
    svg += text_to_svg("C", x_position_frac=0.75, y_position_frac=0.85, font_size=60, color=(0, 0, 0)).split("\n")[1]

    svg = svg.replace("</svg>", "") + "\n</svg>"

    return svg


def load_svg_dataset(split="train", canvas_height=224, canvas_width=224):

    df = pd.read_parquet("/home/mpf/code/kaggle/draw/src/bkp_subs/train_df_poly_100_bottom.parquet")
    
    # df_org = pd.read_parquet("/home/mpf/code/kaggle/draw/src/data/generated/qa_dataset_train.parquet")
    # df = df.merge(df_org, on="id", how="left")
    
    df = df[df["split"] == split].reset_index(drop=True)
    svgs = df["svg"].tolist()

    images_list = []
    svgs_list = []

    for svg in svgs:
        svg = proc_svg(svg)

        try:
            png_data = cairosvg.svg2png(bytestring=svg.encode('utf-8'))
            img = Image.open(io.BytesIO(png_data)).convert('RGB')
            img = img.resize((canvas_width, canvas_height))
        except Exception as e:
            continue

        img = np.array(img)
        img_torch = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        images_list.append(img_torch)
        svgs_list.append(svg)
        
    return images_list, svgs_list



def svg_to_png_no_resize_background(svg_code: str, bg_torch: torch.Tensor) -> Image.Image:
    png_data = cairosvg.svg2png(bytestring=svg_code.encode('utf-8'))
    img_pil = Image.open(io.BytesIO(png_data)).convert('RGB')

    img = torch.from_numpy(np.array(img_pil)).permute(2, 0, 1).float() / 255.0
    # img = torch.cat([img, bg_torch], dim=1)
    pos = 32, 32
    bg_torch[:, pos[0]:pos[0]+img.shape[1], pos[1]:pos[1]+img.shape[2]] = img

    img_pil = Image.fromarray((bg_torch * 255).detach().permute(1, 2, 0).cpu().numpy().astype(np.uint8)).convert("RGB")
    
    return img_pil


def svg_to_png_no_resize(svg_code: str) -> Image.Image:
    png_data = cairosvg.svg2png(bytestring=svg_code.encode('utf-8'))
    img_pil = Image.open(io.BytesIO(png_data)).convert('RGB')
    return img_pil



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



def merge_svgs(bg_svg: str, aest_svg: str):
    aest_svg = aest_svg.strip().split("\n")[2:-1]
    # aest_svg = [
    #     '<g id="aesthetics" fill-opacity="0.5">',
    #     *aest_svg,
    #     "</g>"
    # ]
    aest_svg = "\n".join(aest_svg)
    svg = bg_svg + '\n' + aest_svg
    svg = svg.replace("</svg>", "") + "</svg>"

    return svg



def apply_random_crop_resize_seed(image: Image.Image, crop_percent=0.05, seed=42):
    rs = np.random.RandomState(seed)
    
    width, height = image.size
    crop_pixels_w = int(width * crop_percent)
    crop_pixels_h = int(height * crop_percent)

    left = rs.randint(0, crop_pixels_w + 1)
    top = rs.randint(0, crop_pixels_h + 1)
    right = width - rs.randint(0, crop_pixels_w + 1)
    bottom = height - rs.randint(0, crop_pixels_h + 1)

    image = image.crop((left, top, right, bottom))
    image = image.resize((width, height), Image.BILINEAR)

    return image




def optimize_diffvg(
    vqa_evaluator: VQAEvaluator,
    aesthetic_evaluator: AestheticEvaluator,
    target_text: str,
    svg: str,
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
    print(f"Target text: {target_text}\n")

    pydiffvg.set_use_gpu(torch.cuda.is_available())

    initial_svg = proc_svg(svg)

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

    best_svg = optim_svg.write_xml()
    best_val_loss = -1e8
    initial_loss = None
    
    grad_accumulation_steps = 1

    

    pbar = tqdm(total=num_iterations)

    for iter_idx in range(num_iterations):
        optim_svg.zero_grad()
        image = optim_svg.render(seed=iter_idx)
        img = image[:, :, :3].permute(2, 0, 1).clamp(0, 1)

        crop_frac = 0.05
        random_size = int(random.uniform(1.0 - crop_frac, 1.0) * image.shape[1])
        img = kornia.augmentation.RandomCrop((random_size, random_size))(img.unsqueeze(0)).squeeze(0)

        img = apply_preprocessing_torch(img)

        loss = aesthetic_score_gradient(aesthetic_evaluator, img).mean()

        if iter_idx == 0 or (iter_idx + 1) % validation_steps == 0:
            torch.cuda.empty_cache()
            aest_svg = optim_svg.write_xml()
            loss_total, vqa_loss_total, aest_loss_total, ocr_loss_total = 0.0, 0.0, 0.0, 0.0
            
            with open("output.svg", "w") as f:
                f.write(aest_svg)

            num_evals = 4
            for val_idx in range(num_evals):
                cur_svg = optimize_svg(aest_svg)
                pil_image = svg_to_png_no_resize(cur_svg)

                pil_image = apply_random_crop_resize_seed(pil_image, crop_percent=0.03, seed=iter_idx)

                val_loss, vqa_val_loss, aest_val_loss, ocr_loss, ocr_text = score_original(
                    vqa_evaluator,
                    aesthetic_evaluator,
                    pil_image,
                    questions,
                    choices_list,
                    answers,
                )
                
                loss_total += val_loss
                vqa_loss_total += vqa_val_loss
                aest_loss_total += aest_val_loss
                ocr_loss_total += ocr_loss

            loss_total /= num_evals
            vqa_loss_total /= num_evals
            aest_loss_total /= num_evals
            ocr_loss_total /= num_evals

            if initial_loss is None:
                initial_loss = loss_total

            else:
                if loss_total > best_val_loss:
                    best_val_loss = loss_total
                    best_svg = aest_svg

            
        pbar.set_description(
            f"It {iter_idx}/{num_iterations} | "
            f"Loss: {loss.item():.3f} | "
            f"Val Loss: {loss_total:.3f} | "
            f"Val VQA Loss: {vqa_loss_total:.3f} | "
            f"Val Aest Loss: {aest_loss_total:.3f} | "
            f"Val OCR Loss: {ocr_loss_total:.3f} | "
        )
        pbar.update(1)

        loss = -loss / grad_accumulation_steps
        loss.backward()
        
        if (iter_idx + 1) % grad_accumulation_steps == 0:
            optim_svg.step()

    # best_svg = optim_svg.write_xml()

    print("\n", "="*100, "\n")
    print(f"Best loss: {best_val_loss}")
    print(f"Initial loss: {initial_loss}")
    print(f"Best loss / Initial loss: {best_val_loss / initial_loss}")
    print("="*100, "\n\n")

    return best_svg



def evaluate():
    seed_everything(42)

    # vqa_evaluator = None
    vqa_evaluator = VQAEvaluator()
    vqa_evaluator.model.eval()
    vqa_evaluator.model.requires_grad_(False)

    aesthetic_evaluator = AestheticEvaluator()
    aesthetic_evaluator.predictor.eval()
    aesthetic_evaluator.predictor.requires_grad_(False)
    aesthetic_evaluator.clip_model.eval()
    aesthetic_evaluator.clip_model.requires_grad_(False)

    mean_score_gt = 0
    mean_score_gen = 0
    
    df = pd.read_parquet("/home/mpf/code/kaggle/draw/src/bkp_subs/train_df_poly_100_bottom.parquet")

    # df = pd.read_parquet("/home/mpf/code/kaggle/draw/question_generation_results.parquet")
    # df = df[df["set"] == "test"].iloc[:1]

    # df_svg = pd.read_parquet("/home/mpf/code/kaggle/draw/src/subs/train_df.parquet")
    # df_svg = df_svg[df_svg["id"].isin(df["id"])]
    # df_svg = df_svg[["id", "svg"]]
    # df = df.merge(df_svg, on="id", how="right")

    for index, row in tqdm(df.iterrows(), total=len(df)):
        run_optimization = True
        if run_optimization:
            svg = optimize_diffvg(
                vqa_evaluator=vqa_evaluator,
                aesthetic_evaluator=aesthetic_evaluator,
                target_text=row["description"],
                questions=json.loads(row["question"]),
                choices_list=json.loads(row["choices"]),
                answers=json.loads(row["answer"]),
                svg=row["svg"],
                canvas_width=384,
                canvas_height=384,
                num_iterations=1000,
                validation_steps=200,
                num_tiles=384//16,
                tile_split=1,
            )

        else:
            with open("output_aest.svg") as f:
                svg = f.read()

        with open("output_aest.svg", "w") as f:
            f.write(svg)

        opt_svg = optimize_svg(svg)

        with open("output_opt.svg", "w") as f:
            f.write(opt_svg)

        print(f"Length SVG: {len(opt_svg.encode('utf-8'))}")

        image = svg_to_png_no_resize(opt_svg)
        image.save("output.png")


if __name__ == "__main__":
    evaluate()
