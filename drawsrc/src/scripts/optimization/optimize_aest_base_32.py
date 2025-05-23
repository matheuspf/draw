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


def get_initial_diffvg_scene(
    mask: torch.Tensor,
    canvas_width: int = 384,
    canvas_height: int = 384,
    num_paths: int = 512,
    max_width: float = 2.0,
    use_blob: bool = False,
    device: str = "cuda:0",
    max_attempts_per_shape: int = 20,
):
    """
    Create initial diffvg shapes and shape_groups, similar to painterly_rendering.py,
    but ensure exactly num_paths shapes, each placed randomly inside the mask.
    """
    shapes = []
    shape_groups = []
    mask_np = mask[0].cpu().numpy()  # Assume mask shape [3, H, W], use first channel
    H, W = mask_np.shape

    def is_point_in_mask(x, y):
        xi = int(np.clip(x, 0, W - 1))
        yi = int(np.clip(y, 0, H - 1))
        return mask_np[yi, xi] > 0.5

    for shape_idx in range(num_paths):
        for attempt in range(max_attempts_per_shape):
            # Randomly pick a starting point inside the mask
            yx = np.argwhere(mask_np > 0.5)
            if len(yx) == 0:
                raise ValueError("Mask is empty!")
            y0, x0 = yx[np.random.randint(len(yx))]
            x0 = float(x0)
            y0 = float(y0)

            num_segments = random.randint(1, 3)
            num_control_points = torch.zeros(num_segments, dtype=torch.int32) + 2
            points = []
            p0 = (
                x0 + random.uniform(-5, 5),
                y0 + random.uniform(-5, 5),
            )
            points.append(p0)
            valid = True
            p_last = p0
            for k in range(num_segments):
                radius = random.uniform(10, 30)
                p1 = (
                    p_last[0] + radius * (random.random() - 0.5),
                    p_last[1] + radius * (random.random() - 0.5),
                )
                p2 = (
                    p1[0] + radius * (random.random() - 0.5),
                    p1[1] + radius * (random.random() - 0.5),
                )
                p3 = (
                    p2[0] + radius * (random.random() - 0.5),
                    p2[1] + radius * (random.random() - 0.5),
                )
                for px, py in [p1, p2, p3]:
                    if not is_point_in_mask(px, py):
                        valid = False
                        break
                if not valid:
                    break
                points.extend([p1, p2, p3])
                p_last = p3
            if not valid:
                continue  # Try again
            points = torch.tensor(points, device=device)
            path = pydiffvg.Path(
                num_control_points=num_control_points,
                points=points,
                stroke_width=torch.tensor(1.0, device=device),
                is_closed=False,
            )
            shapes.append(path)
            color = torch.tensor(
                [random.random(), random.random(), random.random(), random.random()],
                device=device,
            )
            group = pydiffvg.ShapeGroup(
                shape_ids=torch.tensor([len(shapes) - 1], device=device),
                fill_color=None,
                stroke_color=color,
            )
            shape_groups.append(group)
            break  # Success, go to next shape
        else:
            print(f"Warning: Could not place shape {shape_idx} inside mask after {max_attempts_per_shape} attempts.")
    return shapes, shape_groups


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
    s, e = 8, 64
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
    tile_split: int = 4,
    num_paths: int = 100,
) -> Image.Image:
    pydiffvg.set_use_gpu(torch.cuda.is_available())
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    background_images, background_svgs = load_svg_dataset(split="train", canvas_width=canvas_width, canvas_height=canvas_height)
    background_val_images, background_val_svgs = load_svg_dataset(split="validation", canvas_width=canvas_width, canvas_height=canvas_height)
    np.random.shuffle(background_images)
    np.random.shuffle(background_val_images)
    background_val_images = background_val_images[:50]

    s, e = 8, 8+64
    mask = torch.zeros((3, canvas_height, canvas_width), dtype=torch.float32, device=device)
    mask[:, s:e, s:e] = 1

    # --- NEW: create initial diffvg scene ---
    shapes, shape_groups = get_initial_diffvg_scene(
        mask, canvas_width, canvas_height, num_paths=num_paths, max_width=2.0, use_blob=False, device=device
    )

    # --- Set up optimizers as in painterly_rendering.py ---
    points_vars = []
    stroke_width_vars = []
    color_vars = []
    for path in shapes:
        path.points.requires_grad = True
        points_vars.append(path.points)
        path.stroke_width.requires_grad = True
        stroke_width_vars.append(path.stroke_width)
    for group in shape_groups:
        group.stroke_color.requires_grad = True
        color_vars.append(group.stroke_color)

    points_optim = torch.optim.Adam(points_vars, lr=1.0)
    width_optim = torch.optim.Adam(stroke_width_vars, lr=0.1)
    color_optim = torch.optim.Adam(color_vars, lr=0.01)

    best_svg = None
    best_val_loss = -1e8
    grad_accumulation_steps = 1
    pbar = tqdm(total=num_iterations)
    
    
    pydiffvg.save_svg("output.svg", canvas_width, canvas_height, shapes, shape_groups)
    with open("output.svg", "r") as f:
        initial_svg = f.read()
    print(f"Initial SVG length: {len(initial_svg.encode('utf-8'))}")


    render = pydiffvg.RenderFunction.apply

    for iter_idx in range(num_iterations):
        points_optim.zero_grad()
        width_optim.zero_grad()
        color_optim.zero_grad()

        # --- Render current image ---
        scene_args = pydiffvg.RenderFunction.serialize_scene(
            canvas_width, canvas_height, shapes, shape_groups
        )
        image = render(
            canvas_width, canvas_height, 2, 2, iter_idx, None, *scene_args
        )
        img = image[:, :, :3].permute(2, 0, 1).clamp(0, 1)

        bg = background_images[iter_idx % len(background_images)].to(device)
        bg_svg = background_svgs[iter_idx % len(background_svgs)]

        crop_frac = 0.05
        random_size = int(random.uniform(1.0 - crop_frac, 1.0) * image.shape[1])

        xx = np.random.rand()
        if xx < 1/4:
            img = img[:, :int(384*0.97), :int(384*0.97)]
        elif xx < 2/4:
            img = img[:, -int(384*0.97):, :int(384*0.97)]
        elif xx < 3/4:
            img = img[:, :int(384*0.97), -int(384*0.97):]
        else:
            img = img[:, -int(384*0.97):, -int(384*0.97):]
        img = img.unsqueeze(0)
        
        
        # img = kornia.augmentation.RandomCrop((random_size, random_size))(img.unsqueeze(0))
        img = F.interpolate(img, size=(384, 384), mode="bicubic", align_corners=False, antialias=True).squeeze(0)

        img = apply_preprocessing_torch(img)

        loss = aesthetic_score_gradient(aesthetic_evaluator, img).mean()

        # --- Validation and best SVG saving ---
        if iter_idx == 0 or (iter_idx + 1) % validation_steps == 0:
            svg_path = "aest.svg"
            pydiffvg.save_svg(svg_path, canvas_width, canvas_height, shapes, shape_groups)
            with open(svg_path, "r") as f:
                aest_svg = f.read()

            val_loss = 0.0

            for val_idx, (bg_val, bg_val_svg) in enumerate(zip(background_val_images, background_val_svgs)):
                torch.cuda.empty_cache()
                cur_svg = optimize_svg(merge_svgs(bg_val_svg, aest_svg))
                pil_image = svg_to_png_no_resize(cur_svg)
                pil_image = apply_random_crop_resize_seed(pil_image, crop_percent=0.03, seed=iter_idx)
                pil_image = ImageProcessor(pil_image, crop=False).apply().image
                vl = aesthetic_score_original(aesthetic_evaluator, pil_image)
                val_loss += vl

            val_loss /= len(background_val_images)

            if val_loss > best_val_loss:
                best_val_loss = val_loss
                best_svg = cur_svg

            with open("output.svg", "w") as f:
                f.write(cur_svg)

        pbar.set_description(
            f"It {iter_idx}/{num_iterations} | "
            f"Loss: {loss.item():.3f} | "
            f"Val Loss: {val_loss:.3f} | "
        )
        pbar.update(1)

        # --- Backprop and step ---
        loss = -loss / grad_accumulation_steps
        loss.backward()
        if (iter_idx + 1) % grad_accumulation_steps == 0:
            points_optim.step()
            width_optim.step()
            color_optim.step()
            # Clamp as needed
            
            for path in shapes:
                path.stroke_width.data.clamp_(1.0, 2.0)
                path.points.data[:, 0].clamp_(0, canvas_width - 1e-3)
                path.points.data[:, 1].clamp_(0, canvas_height - 1e-3)
            for group in shape_groups:
                group.stroke_color.data.clamp_(0.0, 1.0)

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
        num_tiles=384//64,
        tile_split=1,
        num_paths=100
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
