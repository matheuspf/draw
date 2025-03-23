import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import concurrent
import pandas as pd
import cv2
import io
import logging
import re
import json
import cairosvg
import kagglehub
import torch
from lxml import etree
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.models.siglip.modeling_siglip import SiglipModel
from transformers.models.siglip.processing_siglip import SiglipProcessor
from src.score_original import VQAEvaluator, ImageProcessor, AestheticEvaluator
from src.score_gradient import score_original, score_gradient


from pathlib import Path
import io
from torch.nn import functional as F
import numpy as np
import requests
import torch
from PIL import Image
from tqdm.auto import tqdm


def get_initial_image(
    dim: tuple[int],
    mean_value: float = 0.0,
    std_value: float = 0.1,
    low: float = -1.0,
    high: float = 1.0,
) -> torch.Tensor:
    # embedding = torch.randn(dim, dtype=torch.float32) * std_value
    # embedding = embedding + mean_value
    # embedding = torch.clamp(embedding, low, high)
    embedding = 0.5 * torch.ones(dim, dtype=torch.float32)
    embedding = embedding.to("cuda:0")
    embedding.requires_grad_(True)
    return embedding


# def get_initial_image(
#     dim: tuple[int],
#     **kwargs,
# ) -> torch.Tensor:
#     embd = cv2.imread("/home/mpf/code/kaggle/draw/ocr.png")
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


def optimize_pixel_wise(
    vqa_evaluator: VQAEvaluator,
    aesthetic_evaluator: AestheticEvaluator,
    description: str,
    questions: list[str],
    choices_list: list[list[str]],
    answers: list[str],
    image_shape: tuple[int] = (3, 224, 224),
    num_iterations: int = 100,
    learning_rate: float = 1e-1,
    weight_decay: float = 0.0,
    validation_steps: int = 10,
) -> Image.Image:
    vqa_evaluator.model.eval()
    vqa_evaluator.model.requires_grad_(False)
    aesthetic_evaluator.predictor.eval()
    aesthetic_evaluator.predictor.requires_grad_(False)
    aesthetic_evaluator.clip_model.eval()
    aesthetic_evaluator.clip_model.requires_grad_(False)

    image = get_initial_image(image_shape)

    starting_image = image.detach().clone()

    optimizer, scheduler = get_optimizer(
        image, lr=learning_rate, weight_decay=weight_decay, T_max=num_iterations, eta_min=0.0
    )
    pbar = tqdm(total=num_iterations)

    for iter_idx in range(num_iterations):
        optimizer.zero_grad()
        loss, _, _ = score_gradient(
            vqa_evaluator, aesthetic_evaluator, image, questions, choices_list, answers
        )
        loss = -loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        if iter_idx == 0 or (iter_idx + 1) % validation_steps == 0:
            val_loss, _, _, _ = score_original(
                vqa_evaluator, aesthetic_evaluator, torch_to_pil(image), questions, choices_list, answers
            )
            loss_diff = np.abs(-loss.item() - val_loss)

        pbar.set_description(
            f"Iteration {iter_idx}/{num_iterations} | "
            f"Loss: {-loss.item():.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Loss Diff: {loss_diff:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )
        pbar.update(1)

    final_diff = (starting_image - image).abs()
    final_diff_img = torch_to_pil(final_diff)

    image = torch_to_pil(image)

    return image


def evaluate_dataset():
    vqa_evaluator = VQAEvaluator()
    aesthetic_evaluator = AestheticEvaluator()

    df = pd.read_parquet("/home/mpf/code/kaggle/draw/question_generation_results.parquet")
    df = df[df["set"] == "test"].head(1)

    for index, row in tqdm(df.iterrows(), total=len(df)):
        description = row["description"]

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
        
        # image = optimize_pixel_wise(
        #     vqa_evaluator=vqa_evaluator,
        #     aesthetic_evaluator=aesthetic_evaluator,
        #     description=description,
        #     questions=gen_questions_list["questions"],
        #     choices_list=gen_questions_list["choices_list"],
        #     answers=gen_questions_list["answers"],
        #     num_iterations=100,
        #     validation_steps=5,
        #     learning_rate=1e-1,
        #     image_shape=(3, 224, 224),
        # )
        # image.save("output.png")



        image = Image.open("output.png")#.resize((384, 384))

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


def test_eval():
    vqa_evaluator = VQAEvaluator()
    aesthetic_evaluator = AestheticEvaluator()

    description = "a yellow forest at dusk"
    questions = [
        "What is the main setting of the image?",
        # "Is there anything yellow in the image?",
        # "What time of day is suggested in the image?",
        # "What color is prominently featured in the image?",
        # "What do you see in the image?",
        # "Is yellow NOT in the image?",
    ]
    choices_list = [
        ["beach", "desert", "forest", "mountain"],
        # ["no", "yes"],
        # ["dawn", "dusk", "midday", "midnight"],
        # ["green", "orange", "yellow", "white"],
        # ["red", "black", "white", "yellow"],
        # ["no", "yes"],
    ]
    answers = [
        "forest",
        # "yes",
        # "dusk",
        # "yellow",
        # "yellow",
        # "no",
    ]

    # image = Image.open("output.png")#.resize((384, 384))
    # scores = score_original(
    #     vqa_evaluator, aesthetic_evaluator, image, questions, choices_list, answers
    # )
    # print(f"Score: {scores}")
    # exit()

    image = optimize_pixel_wise(
        vqa_evaluator=vqa_evaluator,
        aesthetic_evaluator=aesthetic_evaluator,
        description=description,
        questions=questions,
        choices_list=choices_list,
        answers=answers,
        num_iterations=50,
        validation_steps=5,
        learning_rate=1e-1,
        image_shape=(3, 224, 224),
    )

    image.save("output.png")


if __name__ == "__main__":
    evaluate_dataset()
    # test_eval()
