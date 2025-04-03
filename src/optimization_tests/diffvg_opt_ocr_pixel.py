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
from src.score_gradient import score_original, score_gradient_ocr_1, score_gradient_ocr_2
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


def score_gradient_ocr_3(
    evaluator: VQAEvaluator,
    image: torch.Tensor,
    target: list[str],
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
        text="<image>ocr\n",
        suffix=target,
        return_tensors="pt",
    ).to("cuda:0")
    inputs["pixel_values"] = image

    outputs = evaluator.model(**inputs)
    loss = outputs.loss

    return loss


def svg_to_png_no_resize(svg_code: str) -> Image.Image:
    png_data = cairosvg.svg2png(bytestring=svg_code.encode("utf-8"))
    return Image.open(io.BytesIO(png_data)).convert("RGB")


svg_constraints = kagglehub.package_import("metric/svg-constraints", bypass_confirmation=True)


def format_prompt(prompt: str) -> str:
    return "<image>What is written in the image?\n" + prompt + "<eos>"


def svg_to_png_no_resize(svg_code: str) -> Image.Image:
    png_data = cairosvg.svg2png(bytestring=svg_code.encode("utf-8"))
    return Image.open(io.BytesIO(png_data)).convert("RGB")


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def get_initial_image(
    dim: tuple[int],
    mean_value: float = 0.5,
    std_value: float = 0.1,
    low: float = 0.0,
    high: float = 1.0,
) -> torch.Tensor:
    embedding = torch.randn(dim, dtype=torch.float32) * std_value
    embedding = embedding + mean_value
    embedding = torch.clamp(embedding, low, high)
    # embedding = 0.5 * torch.ones(dim, dtype=torch.float32)
    embedding = embedding.to("cuda:0")
    embedding.requires_grad_(True)
    return embedding


# def get_initial_image(
#     dim: tuple[int],
#     **kwargs,
# ) -> torch.Tensor:
#     # embd = cv2.imread("/home/mpf/Downloads/f2.jpg")
#     embd = cv2.imread("/home/mpf/code/kaggle/draw/t1.png")
#     embd = cv2.cvtColor(embd, cv2.COLOR_BGR2RGB)
#     embd = cv2.resize(embd, dim[-2:], interpolation=cv2.INTER_AREA)
#     embd = embd.astype(np.float32) / 255.0
#     embd = torch.tensor(embd, dtype=torch.float32, device="cuda:0")
#     embd = embd.permute((2, 0, 1))
#     embd.requires_grad_(True)
#     return embd


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


def optimize(
    vqa_evaluator: VQAEvaluator,
    aesthetic_evaluator: AestheticEvaluator,
    target_text: str,
    questions: list[str],
    choices_list: list[list[str]],
    answers: list[str],
    dim: tuple[int] = (3, 384, 384),
    num_iterations: int = 100,
    validation_steps: int = 10,
    learning_rate: float = 1e-1,
) -> Image.Image:
    vqa_evaluator.model.eval()
    vqa_evaluator.model.requires_grad_(False)
    aesthetic_evaluator.predictor.eval()
    aesthetic_evaluator.predictor.requires_grad_(False)
    aesthetic_evaluator.clip_model.eval()
    aesthetic_evaluator.clip_model.requires_grad_(False)

    image = get_initial_image(dim=dim)

    text_image = svg_to_png_no_resize(
        text_to_svg(
            text=target_text,
            svg_height=dim[1],
            svg_width=dim[2],
            x_position_frac=0.1,
            y_position_frac=0.3,
            font_size=50,
            color=(255, 0, 0)
        )
    )
    text_image.save("text_image.png")
    # text_image = text_image.transpose(Image.FLIP_TOP_BOTTOM)
    text_image = torch.from_numpy(np.array(text_image)).to("cuda:0").float() / 255.0
    text_image = text_image.permute(2, 0, 1)

    optimizer, scheduler = get_optimizer(image, lr=learning_rate)

   
    # mask = torch.ones_like(image)
    # skip = 4
    # mask[:, ::skip, ::skip] = 0.0

    sz = 14*2
    mask = torch.zeros_like(image)
    mask[:, :sz, :] = 1.0
    mask[:, -sz:, :] = 1.0
    mask[:, :, :sz] = 1.0
    mask[:, :, -sz:] = 1.0
    # mask[:, mask.shape[1]//2 - sz//2:mask.shape[1]//2 + sz//2, :] = 1.0
    # mask[:, :, mask.shape[2]//2 - sz//2:mask.shape[2]//2 + sz//2] = 1.0
    # mask[...] = 1.0
    # mask[:2, ...] = 1.0
    
    def get_image(image, text_image):
        # return torch.cat([image[:, :image.shape[1]//2, :], text_image, image[:, image.shape[1]//2:, :]], dim=1)
        return image * mask + (1.0 - mask) * text_image

    best_val_loss = -1e8
    best_img = None

    initial_image = torch_to_pil(get_image(image, text_image))
    initial_image.save("initial_image.png")
    
    pbar = tqdm(total=num_iterations)

    for iter_idx in range(num_iterations):
        optimizer.zero_grad()

        # img = image
        # img = torch.cat([image, text_image], dim=1)
        img = get_image(image, text_image)

        image_shape = (
            vqa_evaluator.processor.image_processor.size["height"],
            vqa_evaluator.processor.image_processor.size["width"],
        )
        # img = F.interpolate(
        #     img.unsqueeze(0), size=image_shape, mode="bicubic", align_corners=False, antialias=True
        # )[0]
        img = (img - 0.5) / 0.5
        # img = apply_preprocessing_torch(img)

        # loss = score_gradient_ocr_1(vqa_evaluator, img, text="", response=None)
        # loss = score_gradient_ocr_2(vqa_evaluator, img, response="www.")
        loss = score_gradient_ocr_3(vqa_evaluator, img, "<eos>")
        # loss = score_gradient_ocr_3(vqa_evaluator, apply_preprocessing_torch(img), target_text)
        

        torch.cuda.empty_cache()

        if iter_idx == 0 or (iter_idx + 1) % validation_steps == 0:
            torch.cuda.empty_cache()
            pil_image = torch_to_pil(img)
            val_loss, vqa_val_loss, aest_val_loss, ocr_loss, ocr_text = score_original(
                vqa_evaluator, aesthetic_evaluator, pil_image, questions, choices_list, answers
            )
            ocr_text = ocr_text.replace("\n", "")

            if val_loss > best_val_loss:
                best_val_loss = val_loss
                best_img = pil_image

        pbar.set_description(
            f"It {iter_idx}/{num_iterations} | "
            f"Loss: {loss.item():.3f} | "
            f"Val Loss: {val_loss:.3f} | "
            f"Val VQA Loss: {vqa_val_loss:.3f} | "
            f"Val Aest Loss: {aest_val_loss:.3f} | "
            f"Val OCR Loss: {ocr_loss:.3f} | "
            f"Text: `{ocr_text[:10]}` | "
        )
        pbar.update(1)

        loss.backward()
        optimizer.step()
        scheduler.step()

    return best_img


def evaluate():
    seed_everything(42)

    vqa_evaluator = VQAEvaluator()
    aesthetic_evaluator = AestheticEvaluator()

    df = pd.read_parquet("/home/mpf/code/kaggle/draw/question_generation_results.parquet")
    df = df[df["set"] == "test"]#.iloc[2:3]

    for index, row in tqdm(df.iterrows(), total=len(df)):
        description = row["description"]
        # row["ground_truth"] = row["ground_truth"][:4]
        # row["generated"] = row["generated"][:4]

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

        run_inference = True
        if run_inference:
            image = optimize(
                vqa_evaluator=vqa_evaluator,
                aesthetic_evaluator=aesthetic_evaluator,
                target_text=description,
                questions=gen_questions_list["questions"],
                choices_list=gen_questions_list["choices_list"],
                answers=gen_questions_list["answers"],
                num_iterations=200,
                validation_steps=20,
                learning_rate=1e-1,
                dim=(3, 448, 448),
            )
            image.save("output.png")

        image = Image.open("output.png")  # .resize((224, 224), Image.Resampling.NEAREST)
        # image = Image.open("t1.png").resize((384, 384))

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


if __name__ == "__main__":
    evaluate()
