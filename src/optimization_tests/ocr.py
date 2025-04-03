import os
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
from src.utils import optimize_svg


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def svg_to_png(svg_code: str, size: tuple = (384, 384)) -> Image.Image:
    if "viewBox" not in svg_code:
        svg_code = svg_code.replace("<svg", f'<svg viewBox="0 0 {size[0]} {size[1]}"')

    png_data = cairosvg.svg2png(bytestring=svg_code.encode("utf-8"))
    return Image.open(io.BytesIO(png_data)).convert("RGB").resize(size)


def png_to_svg(
    img,
    image_size=(384, 384),
    pixel_size=1,
):
    if isinstance(img, (str, Path)):
        img = Image.open(img).convert("RGB")

    width, height = img.size
    svg_content = f'<svg viewBox="0 0 {image_size[0]} {image_size[1]}">\n'

    for y in range(height):
        for x in range(width):
            r, g, b = img.getpixel((x, y))
            svg_content += f'    <rect x="{x * pixel_size}" y="{y * pixel_size}" width="{pixel_size}" height="{pixel_size}" fill="rgb({r},{g},{b})" />\n'

    svg_content += "</svg>"
    svg_content = svg_content.replace("\n", "").replace("    ", "").replace(" />", "/>")
    svg_content = optimize_svg(svg_content)

    return svg_content



# def get_initial_image(
#     dim: tuple[int],
#     mean_value: float = 0.5,
#     std_value: float = 0.1,
#     low: float = 0.0,
#     high: float = 1.0,
# ) -> torch.Tensor:
#     embedding = torch.randn(dim, dtype=torch.float32) * std_value
#     embedding = embedding + mean_value
#     embedding = torch.clamp(embedding, low, high)
#     # embedding = 0.5 * torch.ones(dim, dtype=torch.float32)
#     embedding = embedding.to("cuda:0")
#     embedding.requires_grad_(True)
#     return embedding



def get_initial_image(
    dim: tuple[int],
    **kwargs,
) -> torch.Tensor:
    embd = cv2.imread("/home/mpf/code/kaggle/draw/t3.png")
    # embd = cv2.imread("/home/mpf/code/kaggle/draw/eos.png")
    embd = cv2.cvtColor(embd, cv2.COLOR_BGR2RGB)
    embd = cv2.resize(embd, dim[-2:], interpolation=cv2.INTER_AREA)
    embd = embd.astype(np.float32) / 255.0
    embd = torch.tensor(embd, dtype=torch.float32, device="cuda:0")
    embd = embd.permute((2, 0, 1))
    embd.requires_grad_(True)
    return embd


def get_optimizer(
    params: torch.Tensor,
    lr: float = 0.01,
    weight_decay: float = 0.0,
    T_max: int = 100,
    eta_min: float = 0.0,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    optimizer = torch.optim.AdamW([params], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=T_max, eta_min=0.1 * lr
    )
    return optimizer, scheduler


def torch_to_pil(image: torch.Tensor) -> Image.Image:
    image = (image.detach() * 255.0).clamp(0, 255)
    image = image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    image = Image.fromarray(image)
    return image


def optimize_pixel_wise(
    vqa_evaluator: VQAEvaluator,
    aesthetic_evaluator: AestheticEvaluator,
    num_iterations: int = 100,
    learning_rate: float = 1e-1,
    weight_decay: float = 0.0,
    validation_steps: int = 10,
    image_shape: tuple[int] = (3, 384, 384),
) -> Image.Image:
    vqa_evaluator.model.eval()
    vqa_evaluator.model.requires_grad_(False)
    aesthetic_evaluator.predictor.eval()
    aesthetic_evaluator.predictor.requires_grad_(False)
    aesthetic_evaluator.clip_model.eval()
    aesthetic_evaluator.clip_model.requires_grad_(False)

    image = get_initial_image(image_shape)

    starting_image = image.detach().clone()
    starting_image = F.interpolate(starting_image.clamp(0, 1).unsqueeze(0), size=(384, 384), mode="bicubic", align_corners=False, antialias=True)[0].clamp(0, 1)

    sz = 20
    mask = torch.zeros_like(starting_image)
    mask[:, :sz, :] = 1.0
    mask[:, -sz:, :] = 1.0
    mask[:, :, :sz] = 1.0
    mask[:, :, -sz:] = 1.0
    # mask[:, mask.shape[1]//2 - sz//2:mask.shape[1]//2 + sz//2, :] = 1.0
    # mask[:, :, mask.shape[2]//2 - sz//2:mask.shape[2]//2 + sz//2] = 1.0
    # mask[...] = 1.0
    # mask[:2, ...] = 1.0

    optimizer, scheduler = get_optimizer(
        image, lr=learning_rate, weight_decay=weight_decay, T_max=num_iterations, eta_min=0.0
    )
    pbar = tqdm(total=num_iterations)

    for iter_idx in range(num_iterations):
        optimizer.zero_grad()

        img = image
        img = F.interpolate(img.clamp(0, 1).unsqueeze(0), size=(384, 384), mode="bicubic", align_corners=False, antialias=True)[0].clamp(0, 1)

        img = img * mask + (1.0 - mask) * starting_image

        # img = apply_preprocessing_torch(img)
        loss = score_gradient_ocr(vqa_evaluator, img).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if iter_idx == 0 or (iter_idx + 1) % validation_steps == 0:
            pil_image = torch_to_pil(img).resize((384, 384))
            # pil_image = ImageProcessor(copy.deepcopy(pil_image)).apply().image
            val_loss, text = vqa_evaluator.ocr(pil_image)
            text = text.replace('\n', '')

        pbar.set_description(
            f"Iteration {iter_idx}/{num_iterations} | "
            f"Loss: {loss.item():.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Text: `{text}` | "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )
        pbar.update(1)

        if 1.0 - val_loss < 1e-3:
            break

    image = torch_to_pil(img)

    return image


def evaluate_questions(
    vqa_evaluator: VQAEvaluator, aesthetic_evaluator: AestheticEvaluator, image: Image.Image
):
    df = pd.read_parquet("/home/mpf/code/kaggle/draw/question_generation_results.parquet")
    df = df[df["set"] == "test"].iloc[1:2]

    for index, row in tqdm(df.iterrows(), total=len(df)):
        description = row["description"]
        print(description)

        gt_questions_list = {
            "questions": [dct["question"] for dct in row["ground_truth"]],
            "choices_list": [dct["choices"].tolist() for dct in row["ground_truth"]],
            "answers": [dct["answer"] for dct in row["ground_truth"]],
        }
        gen_questions_list = {
            "questions": [dct["question"] for dct in row["generated"]],
            "choices_list": [dct["choices"].tolist() for dct in row["generated"]],
            "answers": [dct["answer"] for dct in row["generated"]],
        }

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


def evaluate():
    seed_everything(42)

    vqa_evaluator = VQAEvaluator()
    aesthetic_evaluator = AestheticEvaluator()

    run_inference = True
    if run_inference:
        image = optimize_pixel_wise(
            vqa_evaluator=vqa_evaluator,
            aesthetic_evaluator=aesthetic_evaluator,
            num_iterations=100,
            validation_steps=1,
            learning_rate=0.1,
            image_shape=(3, 384, 384),
        )
        image.save("output.png")

    image = Image.open("output.png").resize((384, 384))
    # image = ImageProcessor(copy.deepcopy(image)).apply().image
    image.save("output_processed.png")

    score, text = vqa_evaluator.ocr(image)
    print(f"Score: {score}")
    print(f"Text: `{text}`")

    evaluate_questions(vqa_evaluator, aesthetic_evaluator, image)


if __name__ == "__main__":
    evaluate()
