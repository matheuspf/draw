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
from datasets import load_dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


def load_svg_dataset():
    df = pd.read_parquet("/home/mpf/code/kaggle/draw/src/subs/train_df_bkp.parquet")
    svgs = df["svg"].tolist()

    df_sel = pd.read_parquet("/home/mpf/code/kaggle/draw/question_generation_results.parquet")
    df_sel = df_sel[df_sel["set"] == "test"].iloc[:1]

    df = df[~df["id"].isin(df_sel["id"])]


    images_list = []

    for svg in svgs:
        try:
            png_data = cairosvg.svg2png(bytestring=svg.encode('utf-8'))
            img = Image.open(io.BytesIO(png_data)).convert('RGB')
        except Exception as e:
            continue

        img = np.array(img)
        img_torch = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        images_list.append(img_torch)

    return images_list


# def load_svg_dataset():
#     dataset = load_dataset("starvector/text2svg-stack")
    
#     # Get the total size of the dataset
#     total_size = len(dataset["train"])
    
#     # Define sample size
#     sample_size = 5000
    
#     # Generate random indices
#     random_indices = random.sample(range(total_size), min(sample_size, total_size))
#     random_range = random_indices
    
#     dataset = dataset["train"].select(random_range)

#     images_list = []
    
#     # Define the transformation
#     transform = A.Compose([
#         # A.LongestMaxSize(max_size=384),  # Resize preserving aspect ratio
#         # A.PadIfNeeded(
#         #     min_height=384//2,
#         #     min_width=384,
#         #     border_mode=0,  # constant padding
#         #     value=[0, 0, 0]  # black background
#         # ),
#         A.Resize(height=384//2, width=384)
#     ])

#     for svg in tqdm(dataset["Svg"]):
#         try:
#             png_data = cairosvg.svg2png(bytestring=svg.encode('utf-8'))
#             img = Image.open(io.BytesIO(png_data)).convert('RGB')
#         except Exception as e:
#             continue
        
#         # Convert to numpy array for albumentations
#         img_np = np.array(img)

#         if np.mean(img_np == 0) > 0.5:
#             continue

#         # Apply transformation
#         transformed = transform(image=img_np)
#         img_transformed = transformed["image"]

#         img_torch = torch.from_numpy(img_transformed).permute(2, 0, 1).float() / 255.0

#         # img_pil = Image.fromarray((img_torch * 255).detach().permute(1, 2, 0).cpu().numpy().astype(np.uint8)).convert("RGB")
#         # img_pil.save("img_0.png")
#         # import pdb; pdb.set_trace()
        
#         images_list.append(img_torch)

#     return images_list



def svg_to_png_no_resize_background(svg_code: str, bg_torch: torch.Tensor) -> Image.Image:
    png_data = cairosvg.svg2png(bytestring=svg_code.encode('utf-8'))
    img_pil = Image.open(io.BytesIO(png_data)).convert('RGB')

    img = torch.from_numpy(np.array(img_pil)).permute(2, 0, 1).float() / 255.0
    # img = torch.cat([img, bg_torch], dim=1)
    pos = 64, 64
    bg_torch[:, pos[0]:pos[0]+img.shape[1], pos[1]:pos[1]+img.shape[2]] = img

    img_pil = Image.fromarray((bg_torch * 255).detach().permute(1, 2, 0).cpu().numpy().astype(np.uint8)).convert("RGB")
    
    return img_pil


def svg_to_png_no_resize(svg_code: str) -> Image.Image:
    png_data = cairosvg.svg2png(bytestring=svg_code.encode('utf-8'))
    img_pil = Image.open(io.BytesIO(png_data)).convert('RGB')

    img = torch.from_numpy(np.array(img_pil)).permute(2, 0, 1).float() / 255.0
    bg = torch.rand_like(img)
    img = torch.cat([img, bg], dim=1)

    img_pil = Image.fromarray((img * 255).detach().permute(1, 2, 0).cpu().numpy().astype(np.uint8)).convert("RGB")
    
    return img_pil


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
    target_text: str,
    canvas_width: int = 384,
    canvas_height: int = 384,
    num_tiles: int = 4,
    points_per_edge: int = 1,
):
    # Start with the SVG header
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_width}" height="{canvas_height}" xmlns:xlink="http://www.w3.org/1999/xlink">\n'
    
    # # Add definitions section for gradients
    # svg += '  <defs>\n'
    
    # # Create several linear gradients with random colors
    # num_gradients = 5
    # for i in range(num_gradients):
    #     start_color = rgb_to_hex(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    #     end_color = rgb_to_hex(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        
    #     # Create a horizontal gradient - using absolute coordinates
    #     svg += f'    <linearGradient id="grad-h-{i}" x1="0" y1="{canvas_height//2}" x2="{canvas_width}" y2="{canvas_height//2}">\n'
    #     svg += f'      <stop offset="0" stop-color="{start_color}" stop-opacity="1" />\n'
    #     svg += f'      <stop offset="1" stop-color="{end_color}" stop-opacity="1" />\n'
    #     svg += '    </linearGradient>\n'
        
    #     # Create a vertical gradient - using absolute coordinates
    #     svg += f'    <linearGradient id="grad-v-{i}" x1="{canvas_width//2}" y1="0" x2="{canvas_width//2}" y2="{canvas_height}">\n'
    #     svg += f'      <stop offset="0" stop-color="{start_color}" stop-opacity="1" />\n'
    #     svg += f'      <stop offset="1" stop-color="{end_color}" stop-opacity="1" />\n'
    #     svg += '    </linearGradient>\n'
        
    #     # Create a diagonal gradient - using absolute coordinates
    #     svg += f'    <linearGradient id="grad-d-{i}" x1="0" y1="0" x2="{canvas_width}" y2="{canvas_height}">\n'
    #     svg += f'      <stop offset="0" stop-color="{start_color}" stop-opacity="1" />\n'
    #     svg += f'      <stop offset="0.5" stop-color="{rgb_to_hex(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))}" stop-opacity="1" />\n'
    #     svg += f'      <stop offset="1" stop-color="{end_color}" stop-opacity="1" />\n'
    #     svg += '    </linearGradient>\n'
        
    #     # # Create a radial gradient - using absolute coordinates
    #     # cx = canvas_width * random.uniform(0.3, 0.7)
    #     # cy = canvas_height * random.uniform(0.3, 0.7)
    #     # r = min(canvas_width, canvas_height) * random.uniform(0.5, 0.8)
    #     # fx = cx - (canvas_width * 0.1)
    #     # fy = cy - (canvas_height * 0.1)
    #     # svg += f'    <radialGradient id="grad-r-{i}" cx="{cx}" cy="{cy}" r="{r}" fx="{fx}" fy="{fy}" gradientUnits="userSpaceOnUse">\n'
    #     # svg += f'      <stop offset="0" stop-color="{start_color}" stop-opacity="1" />\n'
    #     # svg += f'      <stop offset="0.7" stop-color="{rgb_to_hex(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))}" stop-opacity="1" />\n'
    #     # svg += f'      <stop offset="1" stop-color="{end_color}" stop-opacity="1" />\n'
    #     # svg += '    </radialGradient>\n'
    
    # svg += '  </defs>\n'
    
    # Add a white background
    svg += f'  <path d="M 0,0 h {canvas_width} v {canvas_height} h {-canvas_width} z" fill="{rgb_to_hex(255, 255, 255)}" />\n'

    for i in range(num_tiles):
        for j in range(num_tiles):
            # if j > 0.5 * num_tiles:
            #     continue

            x = i * (canvas_width // num_tiles)
            y = j * (canvas_height // num_tiles)
            width = canvas_width // num_tiles
            height = canvas_height // num_tiles
            
            # Randomly choose between a solid color or a gradient
            # use_gradient = random.random() > 0.5
            use_gradient = False
            
            if use_gradient:
                # Choose a random gradient type and index
                # grad_type = random.choice(['h', 'v', 'd', 'r'])
                grad_type = random.choice(['h', 'v', 'd'])
                grad_idx = random.randint(0, num_gradients - 1)
                fill = f'url(#grad-{grad_type}-{grad_idx})'
            else:
                # Use a solid color
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
    num_tiles: int = 4,
) -> Image.Image:
    pydiffvg.set_use_gpu(torch.cuda.is_available())

    background_images = load_svg_dataset()
    np.random.shuffle(background_images)
    background_val_images = background_images[:1]
    background_images = background_images[1:]

    # initial_svg = get_initial_svg(target_text, canvas_width, canvas_height // 2, num_tiles=num_tiles)
    initial_svg = get_initial_svg(target_text, 128, 128, 8)
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

        bg = background_images[iter_idx % len(background_images)].to("cuda:0")

        pos = 64, 64
        pos = (pos[0] + random.randint(-10, 10+1), pos[1] + random.randint(-10, 10+1))
        bg[:, pos[0]:pos[0]+img.shape[1], pos[1]:pos[1]+img.shape[2]] = img

        # img = torch.cat([img, bg], dim=1)

        img = apply_preprocessing_torch(bg)

        loss = aesthetic_score_gradient(aesthetic_evaluator, img).mean()

        if iter_idx == 0 or (iter_idx + 1) % validation_steps == 0:
            val_loss = 0.0
            
            for bg_val in background_val_images:
                cur_svg = optimize_svg(optim_svg.write_xml())
                pil_image = svg_to_png_no_resize_background(cur_svg, bg_val)
                # pil_image = svg_to_png_no_resize(cur_svg)
                # pil_image = Image.fromarray((img * 255).detach().permute(1, 2, 0).cpu().numpy().astype(np.uint8)).convert("RGB")

                torch.cuda.empty_cache()
                pil_image = ImageProcessor(pil_image).apply().image
                vl = aesthetic_score_original(aesthetic_evaluator, pil_image)
                val_loss += vl
            val_loss /= len(background_val_images)

            if val_loss > best_val_loss:
                best_val_loss = val_loss
                best_svg = cur_svg
            
        pbar.set_description(
            f"It {iter_idx}/{num_iterations} | "
            f"Loss: {loss.item():.3f} | "
            f"Val Loss: {val_loss:.3f} | "
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
    vqa_evaluator.model.eval()
    vqa_evaluator.model.requires_grad_(False)

    aesthetic_evaluator = AestheticEvaluator()
    aesthetic_evaluator.predictor.eval()
    aesthetic_evaluator.predictor.requires_grad_(False)
    aesthetic_evaluator.clip_model.eval()
    aesthetic_evaluator.clip_model.requires_grad_(False)
    
    mean_score_gt = 0
    mean_score_gen = 0

    df = pd.read_parquet("/home/mpf/code/kaggle/draw/question_generation_results.parquet")
    df = df[df["set"] == "test"].iloc[:1]

    # df_svg = pd.read_parquet("/home/mpf/code/kaggle/draw/src/subs/train_df.parquet")
    # df_svg = df_svg[df_svg["id"].isin(df["id"])]
    # df_svg = df_svg[["id", "svg"]]
    # df = df.merge(df_svg, on="id", how="right")

    for index, row in tqdm(df.iterrows(), total=len(df)):
        num_ex = 4
        description = row["description"]
        print(description)

        gt_questions_list = {
            "questions": [dct["question"] for dct in row["ground_truth"]][:num_ex],
            "choices_list": [dct["choices"].tolist() for dct in row["ground_truth"]][:num_ex],
            "answers": [dct["answer"] for dct in row["ground_truth"]][:num_ex]
        }
        gen_questions_list = {
            "questions": [dct["question"] for dct in row["generated"]][:num_ex],
            "choices_list": [dct["choices"].tolist() for dct in row["generated"]][:num_ex],
            "answers": [dct["answer"] for dct in row["generated"]][:num_ex]
        }
        
        gen_questions_list_inpt = gt_questions_list

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
                num_iterations=5000,
                validation_steps=200,
                num_tiles=16,
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

    print(f"Mean Score GT: {mean_score_gt / len(df)}")
    print(f"Mean Score Gen: {mean_score_gen / len(df)}")

if __name__ == "__main__":
    evaluate()
