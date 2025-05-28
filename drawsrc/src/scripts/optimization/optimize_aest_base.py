import os
import kornia
import torch.nn as nn
import copy
import random

import pandas as pd
import cv2
import io
import json
import cairosvg
import torch
from src.score_original import VQAEvaluator, ImageProcessor, AestheticEvaluator
from src.score_gradient import aesthetic_score_original, aesthetic_score_gradient
from src.preprocessing import apply_preprocessing_torch

from pathlib import Path
import io
from torch.nn import functional as F
import numpy as np
import requests
import torch
from PIL import Image
from tqdm.auto import tqdm
from src.utils import optimize_svg, svg_to_png, displace_svg_paths
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


def load_svg_dataset(split="train", canvas_height=384, canvas_width=384):
    # df = pd.read_parquet("/home/mpf/code/kaggle/draw/src/subs/train_df_sdxl_vtracer.parquet")
    df = pd.read_parquet("/home/mpf/code/kaggle/draw/sub_reno_pali_3b_224.parquet")
    # df = pd.read_parquet("/home/mpf/code/kaggle/draw/sub_reno_imagereward_aest.parquet")
    
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
        # svg += text_to_svg("O", x_position_frac=0.8, y_position_frac=0.9, font_size=60, color=(0, 0, 0), font_path="/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf").split("\n")[1]
        svg = svg.replace("</svg>", "") + "</svg>"
        svg = convert_polygons_to_paths(svg)
        # svg = optimize_svg(svg)

        png_data = cairosvg.svg2png(bytestring=svg.encode('utf-8'))
        img = Image.open(io.BytesIO(png_data)).convert('RGB')
        img = img.resize((canvas_width, canvas_height))

        img = np.array(img)
        img_torch = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        images_list.append(img_torch)
        svgs_list.append(svg)
        
    return images_list, svgs_list


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
    
    # # Configure gradient optimization settings
    # settings.global_override(["gradients", "optimize_stops"], True)
    # settings.global_override(["gradients", "stop_lr"], lr)
    # settings.global_override(["gradients", "optimize_color"], True)
    # settings.global_override(["gradients", "color_lr"], lr)
    # settings.global_override(["gradients", "optimize_alpha"], True)
    # settings.global_override(["gradients", "alpha_lr"], lr)
    # settings.global_override(["gradients", "optimize_location"], True)
    # settings.global_override(["gradients", "location_lr"], 10*lr)

    # For filled shapes, optimize colors and transforms
    settings.global_override(["optimize_color"], True)
    settings.global_override(["optimize_alpha"], True)
    settings.global_override(["paths", "optimize_points"], True)
    # settings.global_override(["circles", "optimize_center"], True)
    # settings.global_override(["circles", "optimize_radius"], True)
    # settings.global_override(["transforms", "optimize_transforms"], True)

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
    width: int,
    height: int,
    tile_size: int,
):
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" xmlns:xlink="http://www.w3.org/1999/xlink">\n'

    fill = rgb_to_hex(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    
    svg += f'  <path id="background-0" d="M 0,0 h {width} v {height} h -{width} z" fill="{fill}" />\n'
    # svg += f'  <path d="M 0,0 h {width} v {height} h -{width} z" fill="{fill}" />\n'

    assert width % tile_size == 0
    assert height % tile_size == 0

    num_tiles_width = width // tile_size
    num_tiles_height = height // tile_size

    for i in range(num_tiles_width):
        for j in range(num_tiles_height):
            x = i * tile_size
            y = j * tile_size
            width = tile_size
            height = tile_size

            fill = rgb_to_hex(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            # points_per_edge = random.randint(1, 5)
            points_per_edge = 2


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

                svg += f'  <path d="{path_data}" fill="{fill}" />\n'

    svg += "</svg>"

    with open("initial_svg.svg", "w") as f:
        f.write(svg)

    print(f"Initial Length SVG: {len(svg.encode('utf-8'))}")
    opt_svg = optimize_svg(svg)
    print(f"Optimized Length SVG: {len(opt_svg.encode('utf-8'))}")

    return svg



# def merge_svgs(bg_svg: str, aest_svg: str):
#     aest_svg = aest_svg.strip().split("\n")[2:-1]
#     aest_svg = "\n".join(aest_svg)
#     svg = bg_svg + '\n' + aest_svg
#     svg = svg.replace("</svg>", "") + "</svg>"

#     return svg



def merge_svgs(bg_svg: str, aest_svg: str, pos_x: int = 0, pos_y: int = 0):
    aest_svg = displace_svg_paths(aest_svg, pos_x, pos_y)
    aest_svg = aest_svg.strip().split("\n")[2:-1]

    # aest_svg = [
    #     f'<g transform="translate({pos_x},{pos_y})">',
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


def clamp_svg_canvas(svg_root, width, height):
    def clamp_node(node):
        if hasattr(node, 'paths'):
            for path in getattr(node, 'paths', []):
                path.points.data[:, 0].clamp_(0, width - 1e-3)
                path.points.data[:, 1].clamp_(0, height - 1e-3)
        for child in getattr(node, 'children', []):
            clamp_node(child)
    clamp_node(svg_root)

    return svg_root

def optimize_diffvg(
    aesthetic_evaluator: AestheticEvaluator,
    width: int = 96,
    height: int = 96,
    num_iterations: int = 100,
    validation_steps: int = 10,
    tile_size: int = 16,
    pos_x: int = 0,
    pos_y: int = 0,
) -> Image.Image:
    pydiffvg.set_use_gpu(torch.cuda.is_available())

    background_images, background_svgs = load_svg_dataset(split="train")
    background_val_images, background_val_svgs = load_svg_dataset(split="validation")

    np.random.shuffle(background_images)
    np.random.shuffle(background_val_images)
    background_val_images = background_val_images[:50]
    

    initial_svg = get_initial_svg(width, height, tile_size)

    print(f"Num train: {len(background_images)}")
    print(f"Num eval: {len(background_val_images)}")

    temp_svg_path = "/tmp/initial_svg.svg"
    with open(temp_svg_path, "w") as f:
        f.write(initial_svg)

    settings = get_optimization_settings()

    text_path_ids = [f"text-path-{i}" for i in range(100)] + [f"background-{i}" for i in range(10)]
    for text_id in text_path_ids:
        text_settings = settings.undefault(text_id)
        text_settings["paths"]["optimize_points"] = False
        text_settings["paths"]["optimize_color"] = True
        text_settings["paths"]["optimize_alpha"] = True

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

        bg = background_images[iter_idx % len(background_images)].to("cuda:0")
        bg_svg = background_svgs[iter_idx % len(background_svgs)]

        bg[:, pos_y:pos_y+img.shape[1], pos_x:pos_x+img.shape[2]] = img
        img = bg

        if iter_idx > 1000:
            # if np.random.rand() < 0.25:
            #     xx = np.random.rand()
            #     if xx < 1/4:
            #         img = img[:, :int(384*0.97), :int(384*0.97)]
            #     elif xx < 2/4:
            #         img = img[:, -int(384*0.97):, :int(384*0.97)]
            #     elif xx < 3/4:
            #         img = img[:, :int(384*0.97), -int(384*0.97):]
            #     else:
            #         img = img[:, -int(384*0.97):, -int(384*0.97):]
            #     img = img.unsqueeze(0)

            # else:
            crop_frac = 0.05
            random_size = int(random.uniform(1.0 - crop_frac, 1.0) * img.shape[1])
            img = kornia.augmentation.RandomCrop((384, 384))(img.unsqueeze(0))

            img = F.interpolate(img, size=(384, 384), mode="bicubic", align_corners=False, antialias=True).squeeze(0)

        img = apply_preprocessing_torch(img)

        loss = aesthetic_score_gradient(aesthetic_evaluator, img).mean()

        if iter_idx == 0 or (iter_idx + 1) % validation_steps == 0:
            aest_svg = optim_svg.write_xml()
            val_loss = 0.0
            
            for val_idx, (bg_val, bg_val_svg) in enumerate(zip(background_val_images, background_val_svgs)):
                torch.cuda.empty_cache()

                cur_svg = optimize_svg(merge_svgs(bg_val_svg, aest_svg, pos_x=pos_x, pos_y=pos_y))
                pil_image = svg_to_png_no_resize(cur_svg)

                # pil_image = Image.fromarray((img * 255).detach().permute(1, 2, 0).cpu().numpy().astype(np.uint8)).convert("RGB")

                pil_image = apply_random_crop_resize_seed(pil_image, crop_percent=0.03, seed=val_idx)
                pil_image = ImageProcessor(pil_image, crop=False).apply().image
                vl = aesthetic_evaluator.score(image=pil_image)
                val_loss += vl

            val_loss /= len(background_val_images)

            if val_loss > best_val_loss:
                best_val_loss = val_loss
                best_svg = aest_svg
            
            with open("output.svg", "w") as f:
                f.write(cur_svg)
            
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
            optim_svg = clamp_svg_canvas(optim_svg, width, height)

    # best_svg = optim_svg.write_xml()

    print(f"Best loss: {best_val_loss}")

    return best_svg, best_val_loss



def evaluate():
    seed_everything(42)

    aesthetic_evaluator = AestheticEvaluator()
    aesthetic_evaluator.predictor.eval()
    aesthetic_evaluator.predictor.requires_grad_(False)
    aesthetic_evaluator.clip_model.eval()
    aesthetic_evaluator.clip_model.requires_grad_(False)

    svg, score = optimize_diffvg(
        aesthetic_evaluator=aesthetic_evaluator,
        width=96,
        height=96,
        tile_size=16,
        pos_x=32,
        pos_y=32,
        num_iterations=50000,
        validation_steps=1000,
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
