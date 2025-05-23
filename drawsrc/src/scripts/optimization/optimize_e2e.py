import os
import ast
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
from src.score_original import VQAEvaluator, ImageProcessor, AestheticEvaluator, harmonic_mean
from src.score_gradient import aesthetic_score_original, aesthetic_score_gradient, score_original
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



def harmonic_mean_grad(a: torch.Tensor, b: torch.Tensor, beta: float = 1.0, eps: float = 1e-8) -> torch.Tensor:
    numerator = (1 + beta**2) * (a * b)
    denominator = beta**2 * a + b + eps
    return numerator / denominator


# def vqa_yes_probability(vqa_evaluator, image, description, img_size=(224, 224)):
#     # prompt = f"<image>answer en Question: Does this image contain {description}? Answer Yes or No.\nAnswer:"
#     prompt = description
#     inputs = vqa_evaluator.processor(
#         images=Image.new("RGB", img_size),
#         text=[prompt],
#         return_tensors="pt",
#     ).to("cuda:0")

#     img = F.interpolate(image.unsqueeze(0), size=img_size, mode="bicubic", align_corners=False, antialias=True)
#     img = (img - 0.5) / 0.5
#     inputs["pixel_values"] = img

#     outputs = vqa_evaluator.model(**inputs)
#     logits = outputs.logits[:, -1, :]  # Get logits for the last token
    
#     # Get token IDs for "Yes" and " Yes" (with space)
#     yes_token = vqa_evaluator.processor.tokenizer.encode("Yes", add_special_tokens=False)[0]
#     yes_token_with_space = vqa_evaluator.processor.tokenizer.encode(" Yes", add_special_tokens=False)[0]
    
#     # Create masked logits with only the relevant tokens
#     masked_logits = torch.full_like(logits, float('-inf'))
#     masked_logits[0, yes_token] = logits[0, yes_token]
#     masked_logits[0, yes_token_with_space] = logits[0, yes_token_with_space]
    
#     # Get token IDs for "No" and " No" (with space)
#     no_token = vqa_evaluator.processor.tokenizer.encode("No", add_special_tokens=False)[0]
#     no_token_with_space = vqa_evaluator.processor.tokenizer.encode(" No", add_special_tokens=False)[0]
    
#     # Add "No" tokens to masked logits
#     masked_logits[0, no_token] = logits[0, no_token]
#     masked_logits[0, no_token_with_space] = logits[0, no_token_with_space]
    
#     # Apply softmax to get probabilities
#     probabilities = torch.softmax(masked_logits, dim=-1)
    
#     # Calculate total probability of "Yes"
#     yes_prob = probabilities[0, yes_token] + probabilities[0, yes_token_with_space]
    
#     return yes_prob




def vqa_yes_probability(vqa_evaluator, image, prompts):
    # img_size = (224, 224) if "224" in vqa_evaluator.model_id else (448, 448)
    img_size = (224, 224)
    inputs = vqa_evaluator.processor(
        images=[Image.new("RGB", img_size)] * len(prompts),
        text=prompts,
        return_tensors="pt",
        padding=True
    ).to("cuda:0")

    img = F.interpolate(image.unsqueeze(0), size=img_size, mode="bicubic", align_corners=False, antialias=True)
    img = (img - 0.5) / 0.5
    inputs["pixel_values"] = img.repeat(len(prompts), 1, 1, 1)

    outputs = vqa_evaluator.model(**inputs)
    logits = outputs.logits[:, -1, :]  # Get logits for the last token
    
    # Get token IDs for "Yes" and " Yes" (with space)
    yes_token = vqa_evaluator.processor.tokenizer.encode("Yes", add_special_tokens=False)[0]
    yes_token_with_space = vqa_evaluator.processor.tokenizer.encode(" Yes", add_special_tokens=False)[0]
    
    # Get token IDs for "No" and " No" (with space)
    no_token = vqa_evaluator.processor.tokenizer.encode("No", add_special_tokens=False)[0]
    no_token_with_space = vqa_evaluator.processor.tokenizer.encode(" No", add_special_tokens=False)[0]
    
    # Create masked logits with only the relevant tokens
    masked_logits = torch.full_like(logits, float('-inf'))
    masked_logits[:, yes_token] = logits[:, yes_token]
    masked_logits[:, yes_token_with_space] = logits[:, yes_token_with_space]
    masked_logits[:, no_token] = logits[:, no_token]
    masked_logits[:, no_token_with_space] = logits[:, no_token_with_space]
    
    # Apply softmax to get probabilities
    probabilities = torch.softmax(masked_logits, dim=-1)
    
    # Calculate total probability of "Yes"
    yes_prob = probabilities[:, yes_token] + probabilities[:, yes_token_with_space]
    
    # Average across batch dimension
    yes_prob = yes_prob.mean(dim=0)

    return yes_prob



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
    df = pd.read_parquet("/home/mpf/code/kaggle/draw/sub_reno_imagereward_prompt.parquet")
    
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

    lr = 3e-3

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
    
    # svg += f'  <path id="background-0" d="M 0,0 h {width} v {height} h -{width} z" fill="{fill}" fill-opacity="1.0" />\n'
    svg += f'  <path d="M 0,0 h {width} v {height} h -{width} z" fill="{fill}" fill-opacity="1.0" />\n'

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

            points_per_edge = random.randint(1, 5)
            # points_per_edge = 2


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


def optimize_diffvg(
    vqa_evaluator: VQAEvaluator,
    vqa_evaluator_2: VQAEvaluator,
    aesthetic_evaluator: AestheticEvaluator,
    description: str,
    initial_svg: str,
    questions: list[str],
    choices_list: list[list[str]],
    answers: list[str],
    num_iterations: int = 100,
    validation_steps: int = 10,
    num_eval: int = 10,
) -> Image.Image:
    pydiffvg.set_use_gpu(torch.cuda.is_available())
    
    print(f"\n ---- Description: `{description}`    -    len(questions): {len(questions)} ---- \n")
    print(questions)

    temp_svg_path = "/tmp/initial_svg.svg"
    with open(temp_svg_path, "w") as f:
        f.write(initial_svg)

    settings = get_optimization_settings()

    text_path_ids = [f"text-path-{i}" for i in range(100)] + [f"background-{i}" for i in range(10)]
    for text_id in text_path_ids:
        text_settings = settings.undefault(text_id)
        text_settings["paths"]["optimize_points"] = False
        text_settings["paths"]["optimize_color"] = False
        text_settings["paths"]["optimize_alpha"] = False

    optim_svg = pydiffvg.OptimizableSvg(
        temp_svg_path, settings, optimize_background=False, verbose=False, device="cuda:0"
    )

    best_svg = optim_svg.write_xml()
    best_val_loss = -1e8
    initial_val_loss = -1e8
    
    grad_accumulation_steps = 1

    pbar = tqdm(total=num_iterations)

    torch.cuda.empty_cache()

    for iter_idx in range(num_iterations):
        if iter_idx == 0 or (iter_idx + 1) % validation_steps == 0:
            cur_svg = optimize_svg(optim_svg.write_xml())
            pil_image_org = svg_to_png_no_resize(cur_svg)
            val_loss, val_vqa_loss, val_aest_loss = 0.0, 0.0, 0.0
            
            for val_idx in range(num_eval):
                pil_image = pil_image_org.copy()
                # pil_image = Image.fromarray((img * 255).detach().permute(1, 2, 0).cpu().numpy().astype(np.uint8)).convert("RGB")

                pil_image = apply_random_crop_resize_seed(pil_image, crop_percent=0.03, seed=val_idx)
                pil_image = ImageProcessor(pil_image, crop=False).apply().image

                vqa_score = vqa_evaluator_2.score(questions=questions[1::2], choices=choices_list[1::2], answers=answers[1::2], image=pil_image, n=8)[0]
                # aest_score = aesthetic_evaluator.score(image=pil_image)
                # score = harmonic_mean(vqa_score, aest_score, beta=0.5)
                aest_score = vqa_score
                score = vqa_score
                
                val_loss += score
                val_vqa_loss += vqa_score
                val_aest_loss += aest_score

            val_loss /= num_eval
            val_vqa_loss /= num_eval
            val_aest_loss /= num_eval

            if val_loss > best_val_loss:
                best_val_loss = val_loss
                best_svg = cur_svg
                
                with open("output_best.svg", "w") as f:
                    f.write(best_svg)
            
            with open("output.svg", "w") as f:
                f.write(cur_svg)
            
            if iter_idx == 0:
                initial_val_loss = val_loss
            
            # if val_loss > 0.95:
            #     print(f"Val loss: {val_loss} -- breaking")
            #     break


        optim_svg.zero_grad()
        image = optim_svg.render(seed=iter_idx)
        img = image[:, :, :3].permute(2, 0, 1).clamp(0, 1)

        crop_frac = 0.03
        random_size = int(random.uniform(1.0 - crop_frac, 1.0) * img.shape[1])
        img = kornia.augmentation.RandomCrop((random_size, random_size))(img.unsqueeze(0))
        img = F.interpolate(img, size=(384, 384), mode="bicubic", align_corners=False, antialias=True).squeeze(0)
        img = apply_preprocessing_torch(img)

        # prompt = description
        prompt = [f"<image>answer en Question: Does this image contain {description}? Answer Yes or No.\nAnswer:"] + \
            [f"<image>answer en Question: {q}. Answer: {a}. Answer Yes or No. Answer:" for q, a in zip(questions[::2], answers[::2])]
        vqa_loss = vqa_yes_probability(vqa_evaluator, img, prompt)
        # aest_loss = aesthetic_score_gradient(aesthetic_evaluator, img).mean()
        # loss = (vqa_loss + aest_loss) / 2
        # loss = harmonic_mean_grad(vqa_loss, aest_loss, beta=0.5)
        loss = vqa_loss
        aest_loss = vqa_loss

        loss = -loss / grad_accumulation_steps
        loss.backward()

        if (iter_idx + 1) % grad_accumulation_steps == 0:
            optim_svg.step()

        pbar.set_description(
            f"It {iter_idx}/{num_iterations} | "
            f"Loss: {-loss.item():.3f} | "
            f"VQA Loss: {vqa_loss.item():.3f} | "
            f"Aest Loss: {aest_loss.item():.3f} | "
            f"Val Loss: {val_loss:.3f} | "
            f"Val VQA Loss: {val_vqa_loss:.3f} | "
            f"Val Aest Loss: {val_aest_loss:.3f} | "
        )
        pbar.update(1)

    best_val_loss = val_loss

    print(f"Best loss: {best_val_loss}")
    print(f"Initial loss: {initial_val_loss}")
    print(f"Improvement: {(best_val_loss - initial_val_loss):.4f}")

    return best_svg, initial_val_loss, best_val_loss



def evaluate():
    seed_everything(42)

    vqa_evaluator = VQAEvaluator('google/paligemma-2/transformers/paligemma2-3b-mix-224')
    vqa_evaluator.model.eval()
    vqa_evaluator.model.requires_grad_(False)

    vqa_evaluator_2 = VQAEvaluator('google/paligemma-2/transformers/paligemma2-10b-mix-448')
    vqa_evaluator_2.model.eval()
    vqa_evaluator_2.model.requires_grad_(False)


    aesthetic_evaluator = AestheticEvaluator()
    aesthetic_evaluator.predictor.eval()
    aesthetic_evaluator.predictor.requires_grad_(False)
    aesthetic_evaluator.clip_model.eval()
    aesthetic_evaluator.clip_model.requires_grad_(False)

    df = pd.read_parquet("/home/mpf/code/kaggle/draw/sub_reno_pali_3b_224.parquet")
    # df = pd.read_parquet("/home/mpf/code/kaggle/draw/sub_reno_imagereward_aest.parquet")
    for colname in ['question', 'choices', 'answer']:
        df[colname] = df[colname].apply(ast.literal_eval)
    
    df = df.iloc[:20].copy()
    final_scores = []
    initial_scores = []
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        description = row["description"]
        questions = row["question"]
        choices_list = row["choices"]
        answers = row["answer"]

        sz = min(len(questions), len(choices_list), len(answers), 8)
        questions = questions[:sz]
        choices_list = choices_list[:sz]
        answers = answers[:sz]

        svg, initial_score, final_score = optimize_diffvg(
            vqa_evaluator=vqa_evaluator,
            vqa_evaluator_2=vqa_evaluator_2,
            aesthetic_evaluator=aesthetic_evaluator,
            description=description,
            initial_svg=row["svg"],
            questions=questions,
            choices_list=choices_list,
            answers=answers,
            num_iterations=50,
            validation_steps=50,
        )

        final_scores.append(final_score)
        initial_scores.append(initial_score)

        print(f"Length SVG: {len(optimize_svg(svg).encode('utf-8'))}")
    
    diff_scores = [final_scores[i] - initial_scores[i] for i in range(len(final_scores))]

    print(f"Mean initial: {np.mean(initial_scores)}")
    print(f"Mean final: {np.mean(final_scores)}")
    print(f"Mean diff: {np.mean(diff_scores)}")


if __name__ == "__main__":
    evaluate()
