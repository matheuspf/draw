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
from src.score_gradient import score_original
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


svg_constraints = kagglehub.package_import("metric/svg-constraints", bypass_confirmation=True)

def svg_to_png_no_resize(svg_code: str) -> Image.Image:
    png_data = cairosvg.svg2png(bytestring=svg_code.encode("utf-8"))
    return Image.open(io.BytesIO(png_data)).convert("RGB")


def get_optimization_settings():
    # Create optimization settings
    settings = pydiffvg.SvgOptimizationSettings()

    lr = 1e-1

    # Configure optimization settings
    settings.global_override(["optimizer"], "Adam")
    settings.global_override(["color_lr"], lr)
    settings.global_override(["alpha_lr"], lr)
    settings.global_override(["paths", "shape_lr"], 10 * lr)
    settings.global_override(["circles", "shape_lr"], 10 * lr)
    settings.global_override(["transforms", "transform_lr"], 10 * lr)

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


def torch_to_pil(image: torch.Tensor) -> Image.Image:
    image = (image.detach() * 255.0).clamp(0, 255)
    image = image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    image = Image.fromarray(image)
    return image


def optimize_diffvg(
    input_svg_path: str,
    target_image_path: str,
    num_iterations: int = 100,
) -> Image.Image:
    pydiffvg.set_use_gpu(torch.cuda.is_available())

    target_image = Image.open(target_image_path)
    target_image = torch.from_numpy(np.array(target_image)).permute(2, 0, 1).unsqueeze(0)
    target_image = (target_image.float() / 255.0).clamp(0, 1).cuda()

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
        input_svg_path, settings, optimize_background=True, verbose=False, device="cuda:0"
    )

    best_svg = optimize_svg(optim_svg.write_xml())

    loss_fn = nn.MSELoss()

    pbar = tqdm(total=num_iterations)

    for iter_idx in range(num_iterations):
        optim_svg.zero_grad()
        image = optim_svg.render(seed=iter_idx)

        img = image[:, :, :3].permute(2, 0, 1)
        
        loss = loss_fn(img, target_image)
        loss.backward()

        pbar.set_description(
            f"It {iter_idx}/{num_iterations} | "
            f"Loss: {loss.item():.3f} | "
        )
        pbar.update(1)

    best_svg = optim_svg.write_xml()

    return best_svg


def evaluate():
    svg = optimize_diffvg(
        input_svg_path="/home/mpf/Downloads/r1.svg",
        target_image_path="/home/mpf/Downloads/r1.jpg",
        num_iterations=100,
    )

    with open("output.svg", "w") as f:
        f.write(svg)


if __name__ == "__main__":
    evaluate()
