import os
import kornia
import torch.nn as nn
import copy
import random
import math

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

    df = pd.read_parquet(kagglehub.dataset_download('tomirol/trainpolyqa', path='train_df_poly_100_bottom.parquet'))
    
    # df_org = pd.read_parquet("/home/mpf/code/kaggle/draw/src/data/generated/qa_dataset_train.parquet")
    # df = df.merge(df_org, on="id", how="left")
    
    df = df[df["split"] == split].reset_index(drop=True)
    svgs = df["svg"].tolist()

    images_list = []
    svgs_list = []

    for svg in svgs:
        # tag = "</g>"
        # bg_idx = svg.rfind(tag) + len(tag)
        # svg = svg[:bg_idx] + "</svg>"

        # with open("output.svg", "w") as f:
        #     f.write(svg)
        # exit()
        
        svg_lines = svg.replace(">", ">\n").strip().split("\n")
        svg_lines = svg_lines[:-2]
        svg = "\n".join(svg_lines)
        # svg += text_to_svg("A", x_position_frac=0.9, y_position_frac=0.9, font_size=45, color=(255, 255, 255), font_path="/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf").split("\n")[1]
        svg += text_to_svg("O", x_position_frac=0.6, y_position_frac=0.85, font_size=60, color=(255, 255, 255), font_path="/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf").split("\n")[1]
        svg += text_to_svg("C", x_position_frac=0.75, y_position_frac=0.85, font_size=60, color=(0, 0, 0), font_path="/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf").split("\n")[1]
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
    s, e = 32, 96
    fill = rgb_to_hex(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    # svg += f'  <path id="background-0" d="M 0,0 h {canvas_width} v {e} h {-canvas_width} z" fill="{rgb_to_hex(255, 255, 255)}" />\n'
    svg += f'  <path id="background-0" d="M {s},{s} h {e} v {e} h -{e} z" fill="{fill}" />\n'
    # svg += f'  <path id="background-0" d="M {s},{s} h {e} v {e} h -{e} z" fill="{fill}" />\n'
    # svg += f'  <path id="background-1" d="M {canvas_width-s-e},{s} h {e} v {e} h -{e} z" fill="{rgb_to_hex(255, 255, 255)}" />\n'
    # svg += f'  <path id="background-2" d="M {s},{canvas_height-s-e} h {e} v {e} h -{e} z" fill="{rgb_to_hex(255, 255, 255)}" />\n'
    # svg += f'  <path id="background-3" d="M {canvas_width-s-e},{canvas_height-s-e} h {e} v {e} h -{e} z" fill="{fill}" />\n'


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

            points_per_edge = random.randint(1, 5)
            # points_per_edge = 2

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

    return svg


def get_initial_svg_random(
    mask: torch.Tensor,
    canvas_width: int = 384,
    canvas_height: int = 384,
    num_paths: int = 40,
    min_vertices: int = 3,
    max_vertices: int = 8,
    min_radius: float = 10.0,
    max_radius: float = 60.0,
) -> str:

    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_width}" height="{canvas_height}" xmlns:xlink="http://www.w3.org/1999/xlink">\n'

    # svg += f'  <path d="M 0,0 H {canvas_width} V {canvas_height} H 0 Z" fill="{rgb_to_hex(255,255,255)}" />\n'

    mask_np = mask.cpu().numpy()[0]
    H, W = mask_np.shape

    for i in range(num_paths):
        # Try to find a centroid within the mask
        for _ in range(100):
            cx = random.randint(0, W - 1)
            cy = random.randint(0, H - 1)
            if mask_np[cy, cx] > 0.5:
                break
        else:
            continue  # Could not find a valid centroid

        num_vertices = random.randint(min_vertices, max_vertices)
        angle_offset = random.uniform(0, 2 * math.pi)
        radius = random.uniform(min_radius, max_radius)

        # Generate polygon vertices
        points = []
        for v in range(num_vertices):
            angle = angle_offset + 2 * math.pi * v / num_vertices
            r = radius * random.uniform(0.7, 1.3)
            x = cx + r * math.cos(angle)
            y = cy + r * math.sin(angle)
            # Clamp to canvas
            x = min(max(x, 0), canvas_width - 1)
            y = min(max(y, 0), canvas_height - 1)
            points.append((x, y))

        # Build SVG path string
        path_d = "M " + " ".join(f"{x:.2f},{y:.2f}" for x, y in points) + " Z"
        fill = rgb_to_hex(random.randint(0,255), random.randint(0,255), random.randint(0,255))
        svg += f'  <path d="{path_d}" fill="{fill}" />\n'

    svg += "</svg>"
    return svg


def merge_svgs(bg_svg: str, aest_svg: str):
    aest_svg = aest_svg.strip().split("\n")[2:-1]
    # aest_svg = [
    #     '<defs>',
    #     '<clipPath id="cut">',
    #     '<rect x="16" y="16" width="112" height="112" />',
    #     '</clipPath>',
    #     '</defs>',
    #     '<g clip-path="url(#cut)">',
    #     *aest_svg,
    #     '</g>'
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

    s, e = 32, 32+96
    org_mask = torch.zeros((1, canvas_height, canvas_width), dtype=torch.float32, device="cuda:0")
    # org_mask[:, s:e, :] = 1
    org_mask[:, s:e, s:e] = 1
    # org_mask[:, s:e, -e:-s] = 1
    # org_mask[:, -e:-s, s:e] = 1
    # org_mask[:, -e:-s, -e:-s] =1


    initial_svg = get_initial_svg(org_mask, canvas_width, canvas_height, num_tiles=num_tiles, tile_split=tile_split)

    # initial_svg = get_initial_svg_random(org_mask, canvas_width, canvas_height, num_paths=50, min_vertices=3, max_vertices=10, min_radius=10.0, max_radius=50.0)

    # with open("/home/mpf/code/kaggle/draw/output_aest_650.svg", "r") as f:
    #     initial_svg = f.read()


    with open("initial_svg.svg", "w") as f:
        f.write(initial_svg)


    print(f"Initial Length SVG: {len(initial_svg.encode('utf-8'))}")
    opt_svg = optimize_svg(initial_svg)
    print(f"Optimized Length SVG: {len(opt_svg.encode('utf-8'))}")



    print(f"Num train: {len(background_images)}")
    print(f"Num eval: {len(background_val_images)}")
    print(f"Target text: {target_text}")

    # initial_svg = convert_polygons_to_paths(pd.read_parquet("/home/mpf/code/kaggle/draw/src/subs/train_df_poly_100.parquet")["svg"].iloc[2])
    

    temp_svg_path = "/tmp/initial_svg.svg"
    with open(temp_svg_path, "w") as f:
        f.write(initial_svg)

    settings = get_optimization_settings()

    text_path_ids = [f"text-path-{i}" for i in range(100)] + [f"background-{i}" for i in range(10)]
    for text_id in text_path_ids:
        text_settings = settings.undefault(text_id)
        text_settings["paths"]["optimize_points"] = False
        # text_settings["paths"]["optimize_points"] = True
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

        
        with torch.no_grad():
            mask = (img < 1e-6).all(dim=0).unsqueeze(0).float()
        img = img + bg * mask


        # img = img * org_mask + bg * (1.0 - org_mask)


        crop_frac = 0.05
        random_size = int(random.uniform(1.0 - crop_frac, 1.0) * image.shape[1])
        img = kornia.augmentation.RandomCrop((random_size, random_size))(img.unsqueeze(0)).squeeze(0)

        # pos = 32, 32
        # pos = (pos[0] + random.randint(-10, 10+1), pos[1] + random.randint(-10, 10+1))
        # bg[:, pos[0]:pos[0]+img.shape[1], pos[1]:pos[1]+img.shape[2]] = img

        img = apply_preprocessing_torch(img)

        loss = aesthetic_score_gradient(aesthetic_evaluator, img).mean()
        
        # norm_loss = 1.0 * ((1.0 - mask) * (1.0 - org_mask)).mean()
        # loss = - loss + norm_loss
        # loss = - loss * (1.0 - norm_loss)

        if iter_idx == 0 or (iter_idx + 1) % validation_steps == 0:
            aest_svg = optim_svg.write_xml()
            val_loss = 0.0
            
            for val_idx, (bg_val, bg_val_svg) in enumerate(zip(background_val_images, background_val_svgs)):
                torch.cuda.empty_cache()

                cur_svg = optimize_svg(merge_svgs(bg_val_svg, aest_svg))

                pil_image = svg_to_png_no_resize(cur_svg)

                # pil_image = Image.fromarray((img_bkp * 255).detach().permute(1, 2, 0).cpu().numpy().astype(np.uint8)).convert("RGB")

                # pil_image = svg_to_png_no_resize_background(cur_svg, bg_val)
                # pil_image = svg_to_png_no_resize(cur_svg)
                # pil_image = Image.fromarray((img * 255).detach().permute(1, 2, 0).cpu().numpy().astype(np.uint8)).convert("RGB")

                pil_image = apply_random_crop_resize_seed(pil_image, crop_percent=0.03, seed=iter_idx)
                pil_image = ImageProcessor(pil_image).apply().image
                vl = aesthetic_score_original(aesthetic_evaluator, pil_image)
                val_loss += vl

            val_loss /= len(background_val_images)

            if val_loss > best_val_loss:
                best_val_loss = val_loss
                best_svg = aest_svg
            
            with open("output_96.svg", "w") as f:
                f.write(cur_svg)
                # f.write(aest_svg)

            # diff = 1e1 * ((1.0 - mask) * (1.0 - org_mask))
            # diff = (diff.permute(1, 2, 0) * 255).cpu().repeat(1, 1, 3).numpy().astype(np.uint8)
            # Image.fromarray(diff).convert("RGB").save("diff.png")


            
        pbar.set_description(
            f"It {iter_idx}/{num_iterations} | "
            f"Loss: {loss.item():.3f} | "
            # f"Norm Loss: {norm_loss.item():.3f} | "
            f"Val Loss: {val_loss:.3f} | "
        )
        pbar.update(1)

        loss = loss / grad_accumulation_steps
        loss.backward()
        
        if (iter_idx + 1) % grad_accumulation_steps == 0:
            optim_svg.step()

    # best_svg = optim_svg.write_xml()

    print(f"Best loss: {best_val_loss}")

    return best_svg, best_val_loss



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

    svg, best_val_loss = optimize_diffvg(
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
        num_tiles=384//16,
        tile_split=1
    )

    with open(f"output_aest_96_{best_val_loss:.3f}.svg", "w") as f:
        f.write(svg)

    opt_svg = optimize_svg(svg)

    with open("output_opt.svg", "w") as f:
        f.write(opt_svg)

    print(f"Length SVG: {len(opt_svg.encode('utf-8'))}")

    image = svg_to_png_no_resize(opt_svg)
    image.save("output.png")





if __name__ == "__main__":
    evaluate()
