import os
import kornia
import torch.nn as nn
import copy
import random

# os.environ["TORCH_COMPILE_DISABLE"] = "1"

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


def load_svg_dataset(split="train", canvas_height=224, canvas_width=224):

    # df = pd.read_parquet("/home/mpf/code/kaggle/draw/src/bkp_subs/train_df_poly_100_bottom.parquet")

    # df = pd.read_parquet(kagglehub.dataset_download('tomirol/trainpolyqa', path='train_df_poly_100_bottom.parquet'))
    # df = pd.read_parquet("/home/mpf/code/kaggle/draw/src/subs/train_df_sdxl_vtracer.parquet")
    
    df = pd.read_parquet("/home/mpf/code/kaggle/draw/sub_reno_imagereward_prompt.parquet")
    
    # df_org = pd.read_parquet("/home/mpf/code/kaggle/draw/src/data/generated/qa_dataset_train.parquet")
    # df = df.merge(df_org, on="id", how="left")
    
    df = df[df["split"] == split].reset_index(drop=True)
    svgs = df["svg"].tolist()

    images_list = []
    svgs_list = []

    for svg in svgs:
        svg_lines = svg.replace(">", ">\n").strip().split("\n")
        svg_lines = svg_lines[:-2]
        svg = "\n".join(svg_lines)

        x_position_frac = 0.85
        y_position_frac = 0.9
        x_pos = int(canvas_width * (x_position_frac))
        y_pos = int(canvas_height * (y_position_frac))
        sz = 24
        svg += f'<path id="text-path-5" d="M {int(x_pos-sz/8)},{int(y_pos-sz*4/5)} h {sz} v {sz} h -{sz} z" fill="{rgb_to_hex(0, 0, 0)}" />\n'
        svg += text_to_svg("O", x_position_frac=x_position_frac, y_position_frac=y_position_frac, font_size=24, color=(255, 255, 255), font_path="/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf").split("\n")[1]
        svg = svg.replace("</svg>", "") + "</svg>"
        svg = convert_polygons_to_paths(svg)

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


def svg_to_png_no_resize(svg_code: str) -> Image.Image:
    png_data = cairosvg.svg2png(bytestring=svg_code.encode('utf-8'))
    img_pil = Image.open(io.BytesIO(png_data)).convert('RGB')
    return img_pil



def get_adaptive_optimization_settings(learning_rate_scale=1.0, max_width=4.0):
    # Create optimization settings with adaptive learning rates
    settings = pydiffvg.SvgOptimizationSettings()

    base_lr = 1e-3 * learning_rate_scale

    # Configure optimization settings with different rates for different parameters
    settings.global_override(["optimizer"], "Adam")
    settings.global_override(["color_lr"], base_lr)
    settings.global_override(["alpha_lr"], base_lr)
    settings.global_override(["paths", "shape_lr"], 10*base_lr)  # Higher LR for shape points
    settings.global_override(["circles", "shape_lr"], 10*base_lr)
    settings.global_override(["transforms", "transform_lr"], 10*base_lr)
    
    # Configure stroke width optimization
    settings.global_override(["paths", "stroke_width_lr"], base_lr)
    settings.global_override(["paths", "min_stroke_width"], 1.0)
    settings.global_override(["paths", "max_stroke_width"], max_width)
    
    # Configure gradient optimization settings
    settings.global_override(["gradients", "optimize_stops"], True)
    settings.global_override(["gradients", "stop_lr"], base_lr)
    settings.global_override(["gradients", "optimize_color"], True)
    settings.global_override(["gradients", "color_lr"], base_lr)
    settings.global_override(["gradients", "optimize_alpha"], True)
    settings.global_override(["gradients", "alpha_lr"], base_lr)
    settings.global_override(["gradients", "optimize_location"], True)
    settings.global_override(["gradients", "location_lr"], 10*base_lr)

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

    fill = rgb_to_hex(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    
    # Add a white background
    # svg += f'  <path d="M 0,0 h {canvas_width} v 64 h {-canvas_width} z" fill="{rgb_to_hex(255, 255, 255)}" />\n'
    s, e = 32, 48
    # svg += f'  <path id="background-0" d="M {s},{s} h {e} v {e} h -{e} z" fill="{fill}" />\n'
    svg += f'  <path id="background-0" d="M {s-8},{s-8} h {e+16} v {e+16} h -{e+16} z" fill="{fill}" />\n'
    # svg += f'  <path id="background-1" d="M {canvas_width-s-e},{s} h {e} v {e} h -{e} z" fill="{rgb_to_hex(255, 255, 255)}" />\n'
    # svg += f'  <path id="background-2" d="M {s},{canvas_height-s-e} h {e} v {e} h -{e} z" fill="{fill}" />\n'
    # svg += f'  <path id="background-3" d="M {canvas_width-s-e},{canvas_height-s-e} h {e} v {e} h -{e} z" fill="{rgb_to_hex(255, 255, 255)}" />\n'

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

            points_per_edge = random.randint(2, 6)
            # points_per_edge = 3

            # Create path with more control points
            if points_per_edge <= 1:
                # Original rectangle with 4 points
                svg += f'  <path d="M {x},{y} h {width} v {height} h {-width} z" fill="{fill}" fill-opacity="1.0" />\n'
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

                svg += f'  <path d="{path_data}" fill="{fill}" fill-opacity="1.0" />\n'
                # svg += f'  <path d="{path_data}" fill="{fill}" />\n'

    # # Add text SVG
    # text_svg = text_to_svg("A", svg_width=canvas_width, svg_height=canvas_height, color=(255, 255, 255), x_position_frac=0.1, y_position_frac=0.2, font_size=50)
    # svg += "\n".join(text_svg.split("\n")[1:-1])

    # svg += text_to_svg("O", x_position_frac=0.6, y_position_frac=0.85, font_size=60, color=(255, 255, 255), font_path="/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf").split("\n")[1]
    # svg += text_to_svg("C", x_position_frac=0.75, y_position_frac=0.85, font_size=60, color=(0, 0, 0), font_path="/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf").split("\n")[1]
    svg += "</svg>"

    with open("initial_svg.svg", "w") as f:
        f.write(svg)

    print(f"Initial Length SVG: {len(svg.encode('utf-8'))}")
    opt_svg = optimize_svg(svg)
    print(f"Optimized Length SVG: {len(opt_svg.encode('utf-8'))}")

    return svg


def get_initial_random_svg(
    mask: torch.Tensor,
    canvas_width: int = 384,
    canvas_height: int = 384,
    num_tiles: int = 4,
    points_per_edge: int = 1,
    tile_split: int = 4,
    use_strokes: bool = False,
    max_stroke_width: float = 5.0
):
    # Start with the SVG header
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_width}" height="{canvas_height}" xmlns:xlink="http://www.w3.org/1999/xlink">\n'

    fill = rgb_to_hex(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    
    s, e = 32, 48
    # svg += f'  <path id="background-0" d="M {s},{s} h {e} v {e} h -{e} z" fill="{fill}" />\n'
    svg += f'  <path id="background-0" d="M {s-8},{s-8} h {e+16} v {e+16} h -{e+16} z" fill="{fill}" />\n'

    num_paths = (num_tiles * tile_split) ** 2 // 4
    for i in range(num_paths):
        if random.random() < 0.2:  # Skip some paths randomly to create diversity
            continue
            
        # Generate random position within the mask area
        valid_pos = False
        for _ in range(100):  # Try a few times to find valid position
            x = random.randint(0, canvas_width - 1)
            y = random.randint(0, canvas_height - 1)
            if x >= s and x < s+e and y >= s and y < s+e:
                if mask[0, y, x] > 0:
                    valid_pos = True
                    break
        
        if not valid_pos:
            continue
            
        # Random path size
        path_width = random.randint(5, 20)
        path_height = random.randint(5, 20)
        
        # Ensure path stays within the masked area
        x = max(min(x, s+e-path_width), s)
        y = max(min(y, s+e-path_height), s)
        
        # Choose random colors for fill or stroke
        fill_color = rgb_to_hex(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        stroke_color = rgb_to_hex(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        stroke_width = random.uniform(1.0, max_stroke_width)
        
        # Create path based on blob approach from painterly_rendering.py
        if use_strokes and random.random() < 0.7:
            # Create stroke-based path (not closed)
            num_segments = random.randint(1, 3)
            path_data = f"M {x},{y} "
            
            p0 = (x, y)
            for j in range(num_segments):
                radius = random.uniform(3.0, 15.0)
                p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
                p2 = (p1[0] + radius * (random.random() - 0.5), p1[1] + radius * (random.random() - 0.5))
                p3 = (p2[0] + radius * (random.random() - 0.5), p2[1] + radius * (random.random() - 0.5))
                
                # Add bezier curve
                path_data += f"C {p1[0]},{p1[1]} {p2[0]},{p2[1]} {p3[0]},{p3[1]} "
                p0 = p3
            
            # Create a stroke-based path (no fill)
            svg += f'  <path d="{path_data}" fill="none" stroke="{stroke_color}" stroke-width="{stroke_width}" stroke-linecap="round" />\n'
        else:
            # Create filled path (closed)
            num_segments = random.randint(3, 5)
            path_data = f"M {x},{y} "
            
            p0 = (x, y)
            for j in range(num_segments):
                radius = random.uniform(2.0, 10.0)
                p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
                p2 = (p1[0] + radius * (random.random() - 0.5), p1[1] + radius * (random.random() - 0.5))
                p3 = (p2[0] + radius * (random.random() - 0.5), p2[1] + radius * (random.random() - 0.5))
                
                # Add bezier curve
                path_data += f"C {p1[0]},{p1[1]} {p2[0]},{p2[1]} {p3[0]},{p3[1]} "
                p0 = p3
            
            path_data += "Z"  # Close path
            
            # Random opacity for variety
            opacity = random.uniform(0.7, 1.0)
            svg += f'  <path d="{path_data}" fill="{fill_color}" fill-opacity="{opacity}" />\n'

    svg += "</svg>"

    with open("initial_random_svg.svg", "w") as f:
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


def clamp_svg_to_mask(svg_root, s, e):
    def clamp_node(node):
        if hasattr(node, 'paths'):
            for path in getattr(node, 'paths', []):
                path.points.data[:, 0].clamp_(s, e - 1e-3)
                path.points.data[:, 1].clamp_(s, e - 1e-3)
        for child in getattr(node, 'children', []):
            clamp_node(child)
    clamp_node(svg_root)


def optimize_svg_with_schedule(
    optim_svg, 
    target_img, 
    num_iterations=100, 
    validation_steps=10,
    lr_schedule=None,
    use_lpips_loss=False
):
    """
    Optimize SVG with a learning rate schedule and LPIPS perception loss option.
    Similar to painterly_rendering.py optimization process.
    """
    if lr_schedule is None:
        # Default schedule: start with high lr, gradually decrease
        lr_schedule = [(0, 1.0), (num_iterations//3, 0.5), (2*num_iterations//3, 0.1)]
    
    if use_lpips_loss:
        try:
            import ttools.modules
            perception_loss = ttools.modules.LPIPS().to(pydiffvg.get_device())
        except ImportError:
            print("Warning: ttools not found. Using L2 loss instead of LPIPS.")
            use_lpips_loss = False
    
    best_loss = float('inf')
    best_svg = optim_svg.write_xml()
    
    # Initialize progress tracking
    pbar = tqdm(total=num_iterations)
    
    # Optimize with changing learning rates
    for iter_idx in range(num_iterations):
        # Update learning rate according to schedule
        current_lr_scale = 1.0
        for schedule_iter, lr_scale in lr_schedule:
            if iter_idx >= schedule_iter:
                current_lr_scale = lr_scale
        
        # Apply learning rate scale to optimizer
        for param_group in optim_svg.optimizer.param_groups:
            param_group['lr'] = param_group['initial_lr'] * current_lr_scale
            
        # Zero gradients
        optim_svg.zero_grad()
        
        # Render current state
        img = optim_svg.render(seed=iter_idx)
        rendered_img = img[:, :, :3].permute(2, 0, 1).clamp(0, 1).unsqueeze(0)
        
        # Prepare target
        target_tensor = target_img.unsqueeze(0).to(rendered_img.device)
        
        # Compute loss
        if use_lpips_loss:
            loss = perception_loss(rendered_img, target_tensor) + (rendered_img.mean() - target_tensor.mean()).pow(2)
        else:
            loss = (rendered_img - target_tensor).pow(2).mean()
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        optim_svg.step()
        
        # Validate and save best result
        if iter_idx % validation_steps == 0 or iter_idx == num_iterations - 1:
            with torch.no_grad():
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_svg = optim_svg.write_xml()
            
        # Update progress
        pbar.set_description(f"Iter {iter_idx}/{num_iterations} | Loss: {loss.item():.5f} | LR: {current_lr_scale:.3f}")
        pbar.update(1)
    
    pbar.close()
    return best_svg, best_loss


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
    num_tiles: int = 12,
    tile_split: int = 4
) -> Image.Image:
    pydiffvg.set_use_gpu(torch.cuda.is_available())

    background_images, background_svgs = load_svg_dataset(split="train", canvas_width=canvas_width, canvas_height=canvas_height)
    background_val_images, background_val_svgs = load_svg_dataset(split="validation", canvas_width=canvas_width, canvas_height=canvas_height)
    np.random.shuffle(background_images)
    np.random.shuffle(background_val_images)
    background_val_images = background_val_images[:50]
    # background_images = background_val_images
    # background_images = background_images#[:20]
    
    
    
    tile_width = canvas_width // num_tiles
    tile_height = canvas_height // num_tiles

    s, e = 32, 32+48
    mask = torch.zeros((3, canvas_height, canvas_width), dtype=torch.float32, device="cuda:0")
    mask[:, s:e, s:e] = 1
    # mask[:, s:e, -e:-s] = 1
    # mask[:, -e:-s, s:e] = 1
    # mask[:, -e:-s, -e:-s] =1

    initial_svg = get_initial_svg(mask, canvas_width, canvas_height, num_tiles=num_tiles, tile_split=tile_split)
    # initial_svg = get_initial_random_svg(mask, canvas_width, canvas_height, num_tiles=num_tiles, tile_split=tile_split)

    # with open("output_vtracer_96_0.708.svg", "r") as f:
        # initial_svg = f.read()

    print(f"Num train: {len(background_images)}")
    print(f"Num eval: {len(background_val_images)}")
    print(f"Target text: {target_text}")

    # initial_svg = convert_polygons_to_paths(pd.read_parquet("/home/mpf/code/kaggle/draw/src/subs/train_df_poly_100.parquet")["svg"].iloc[2])
    

    temp_svg_path = "/tmp/initial_svg.svg"
    with open(temp_svg_path, "w") as f:
        f.write(initial_svg)

    settings = get_adaptive_optimization_settings()

    text_path_ids = [f"text-path-{i}" for i in range(100)] + [f"background-{i}" for i in range(10)]
    for text_id in text_path_ids:
        text_settings = settings.undefault(text_id)
        text_settings["paths"]["optimize_points"] = False
        text_settings["paths"]["optimize_transforms"] = False
        text_settings["paths"]["optimize_color"] = True
        text_settings["paths"]["optimize_alpha"] = True
        text_settings["optimize_color"] = True
        text_settings["optimize_alpha"] = True
        text_settings["optimize_transforms"] = True
        # text_settings["paths"]["shape_lr"] = 1e-1
        # text_settings["transforms"]["transform_lr"] = 1e-1
        # text_settings["color_lr"] = 1e-2
        # text_settings["alpha_lr"] = 1e-2

    optim_svg = pydiffvg.OptimizableSvg(
        temp_svg_path, settings, optimize_background=False, verbose=False, device="cuda:0"
    )

    best_svg, best_loss = optimize_svg_with_schedule(
        optim_svg, 
        background_images[0], 
        num_iterations=num_iterations, 
        validation_steps=validation_steps,
        use_lpips_loss=True
    )

    print(f"Best loss: {best_loss}")

    return best_svg, best_loss



def evaluate():
    seed_everything(42)

    vqa_evaluator = None
    # vqa_evaluator = VQAEvaluator()
    # vqa_evaluator.model.eval()
    # vqa_evaluator.model.requires_grad_(False)

    aesthetic_evaluator = AestheticEvaluator()
    aesthetic_evaluator.predictor.eval()
    aesthetic_evaluator.predictor.requires_grad_(False)
    aesthetic_evaluator.clip_model.eval()
    aesthetic_evaluator.clip_model.requires_grad_(False)

    mean_score_gt = 0
    mean_score_gen = 0

    svg, score = optimize_diffvg(
        vqa_evaluator=None,
        aesthetic_evaluator=aesthetic_evaluator,
        target_text="",
        questions=[],
        choices_list=[],
        answers=[],
        canvas_width=384,
        canvas_height=384,
        num_iterations=50000,
        validation_steps=500,
        num_tiles=384//8,
        tile_split=1
    )

    with open(f"output_vtracer_96_{score:.3f}.svg", "w") as f:
        f.write(svg)

    opt_svg = optimize_svg(svg)

    with open("output_opt.svg", "w") as f:
        f.write(opt_svg)

    print(f"Length SVG: {len(opt_svg.encode('utf-8'))}")

    image = svg_to_png_no_resize(opt_svg)
    image.save("output.png")


if __name__ == "__main__":
    evaluate()
