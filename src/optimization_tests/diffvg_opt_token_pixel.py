import os
import torch.nn as nn
import copy
import random
from src.display.display_word_emb import display_word_emb

os.environ["TORCH_COMPILE_DISABLE"] = "1"

import pandas as pd
import cv2
import io
import json
import cairosvg
import torch
from src.score_original import VQAEvaluator, ImageProcessor, AestheticEvaluator
from src.score_gradient import score_original, vqa_score_gradient
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


def svg_to_png_no_resize(svg_code: str) -> Image.Image:
    png_data = cairosvg.svg2png(bytestring=svg_code.encode("utf-8"))
    return Image.open(io.BytesIO(png_data)).convert("RGB")


svg_constraints = kagglehub.package_import("metric/svg-constraints", bypass_confirmation=True)


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
    target_text = target_text[0].upper() + target_text[1:]

    vqa_evaluator.model.eval()
    vqa_evaluator.model.requires_grad_(False)
    aesthetic_evaluator.predictor.eval()
    aesthetic_evaluator.predictor.requires_grad_(False)
    aesthetic_evaluator.clip_model.eval()
    aesthetic_evaluator.clip_model.requires_grad_(False)

    image_shape = (
        vqa_evaluator.processor.image_processor.size["height"],
        vqa_evaluator.processor.image_processor.size["width"],
    )

    image = get_initial_image(dim=dim)
    optimizer, scheduler = get_optimizer(image, lr=learning_rate)

    best_val_loss = -1e8
    best_img = None

    pbar = tqdm(total=num_iterations)

    print(f"Description: {target_text}\n\n")

    for iter_idx in range(num_iterations):
        optimizer.zero_grad()
        
        prefix_text_list = [
            # "<image>ocr\n",
            # "<image>cap en\n",
            # "<image>caption en\n",
            # "<image>describe en\n",
            "<image>answer en Question: What is represented in the image?\n Choices:\n"
        ]

        for prefix_text in prefix_text_list:
            img = image.clamp(0, 1)
            # img = F.interpolate(
            #     img.unsqueeze(0), size=image_shape, mode="bicubic", align_corners=False, antialias=True
            # )[0]
            img = (img - 0.5) / 0.5
            # img = apply_preprocessing_torch(img)

            inputs = vqa_evaluator.processor(
                images=Image.new("RGB", image_shape),
                text=[prefix_text],
                return_tensors="pt",
                suffix=[target_text + "<eos>"],
            ).to("cuda:0")
            inputs["pixel_values"] = img.unsqueeze(0)
            
            outputs = vqa_evaluator.model(**inputs)
            loss = outputs.loss / len(prefix_text_list)
            loss.backward()

        if iter_idx == 0 or (iter_idx + 1) % validation_steps == 0:
            with torch.no_grad():
                vqa_loss = vqa_score_gradient(vqa_evaluator, image, questions, choices_list, answers).item()

            torch.cuda.empty_cache()
            pil_image = torch_to_pil(image)#.resize((224, 224), Image.Resampling.NEAREST)
            val_loss, vqa_val_loss, aest_val_loss, ocr_loss, ocr_text = score_original(
                vqa_evaluator, aesthetic_evaluator, pil_image, questions, choices_list, answers, apply_preprocessing=False
            )
            ocr_text = ocr_text.replace("\n", "")

            # pil_image = ImageProcessor(pil_image).apply().image

            val_prefix_text = prefix_text_list[-1]
            inputs = vqa_evaluator.processor(
                images=[pil_image],
                text=[val_prefix_text],
                return_tensors="pt",
            ).to("cuda:0")

            output = vqa_evaluator.model.generate(**inputs, max_new_tokens=128)
            input_len = len(inputs["input_ids"][0])
            pred_text = vqa_evaluator.processor.tokenizer.decode(output[0][input_len:]).replace('\n', '')

            if vqa_val_loss > best_val_loss:
                best_val_loss = vqa_val_loss
                best_img = pil_image


        pbar.set_description(
            f"It {iter_idx}/{num_iterations} | "
            f"Loss: {loss.item():.3f} | "
            # f"Val Loss: {val_loss:.3f} | "
            f"VQA Loss: {vqa_loss:.3f} | "
            f"Val VQA Loss: {vqa_val_loss:.3f} | "
            # f"Val Aest Loss: {aest_val_loss:.3f} | "
            # f"Val OCR Loss: {ocr_loss:.3f} | "
            # f"OCR Text: `{ocr_text[:10]}` | "
            f"Pred Text: `{pred_text[:70]}` | "
        )
        pbar.update(1)

        # if vqa_val_loss > 0.9:
        #     break

        # loss.backward()
        optimizer.step()
        scheduler.step()
    
    # display_word_emb(vqa_evaluator, torch_to_pil(image), target_text)

    # print("\n", flush=True)
    # print(f"\nBest VQA Loss: {best_val_loss:.3f}\n")

    return best_img


def evaluate():
    seed_everything(42)

    vqa_evaluator = VQAEvaluator()
    aesthetic_evaluator = AestheticEvaluator()

    df = pd.read_parquet("/home/mpf/code/kaggle/draw/question_generation_results.parquet")
    df = df[df["set"] == "test"]#.iloc[3:]

    average_gen_loss = 0.0
    average_gt_loss = 0.0

    for index, row in tqdm(df.iterrows(), total=len(df)):
        torch.cuda.empty_cache()

        description = row["description"]
        num_ex = 4

        gt_questions_list = {
            "questions": [dct["question"] for dct in row["ground_truth"]],
            "choices_list": [dct["choices"].tolist() for dct in row["ground_truth"]],
            "answers": [dct["answer"] for dct in row["ground_truth"]],
        }
        gen_questions_list = {
            "questions": [dct["question"] for dct in row["generated"]][:num_ex],
            "choices_list": [dct["choices"].tolist() for dct in row["generated"]][:num_ex],
            "answers": [dct["answer"] for dct in row["generated"]][:num_ex],
        }

        run_inference = True
        if run_inference:
            image = optimize(
                vqa_evaluator=vqa_evaluator,
                aesthetic_evaluator=aesthetic_evaluator,
                target_text=description,
                questions=gt_questions_list["questions"],
                choices_list=gt_questions_list["choices_list"],
                answers=gt_questions_list["answers"],
                num_iterations=500,
                validation_steps=50,
                learning_rate=1e-2,
                dim=(3, 448//2, 448//2),
            )
            image.save("output.png")

        image = Image.open("output.png")#.resize((224, 224), Image.Resampling.NEAREST)
        # image = Image.open("t1.png").resize((384, 384))

        score_gen = score_original(
            vqa_evaluator,
            aesthetic_evaluator,
            image,
            gen_questions_list["questions"],
            gen_questions_list["choices_list"],
            gen_questions_list["answers"],
            apply_preprocessing=False,
        )

        score_gt = score_original(
            vqa_evaluator,
            aesthetic_evaluator,
            image,
            gt_questions_list["questions"],
            gt_questions_list["choices_list"],
            gt_questions_list["answers"],
            apply_preprocessing=False,
        )

        print(f"Score Gen: {score_gen}")
        print(f"Score GT: {score_gt}")
        
        average_gen_loss += score_gen[1]
        average_gt_loss += score_gt[1]

    print("\n\n", "="*50, "\n")
    print(f"Average Gen Loss: {average_gen_loss / len(df)}")
    print(f"Average GT Loss: {average_gt_loss / len(df)}")


if __name__ == "__main__":
    evaluate()
