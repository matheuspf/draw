import concurrent
import cv2
import io
import logging
import re

import cairosvg
import kagglehub
import torch
from lxml import etree
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.models.siglip.modeling_siglip import SiglipModel
from transformers.models.siglip.processing_siglip import SiglipProcessor
from src.score_original import VQAEvaluator, ImageProcessor
from src.score_gradient import vqa_score_original, vqa_score_gradient
from src.preprocessing import apply_preprocessing_torch


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
#     embd = cv2.imread("/home/mpf/Downloads/f1.jpeg")
#     embd = cv2.cvtColor(embd, cv2.COLOR_BGR2RGB)
#     embd = cv2.resize(embd, dim[-2:], interpolation=cv2.INTER_AREA)
#     embd = embd.astype(np.float32) / 255.0
#     embd = torch.tensor(embd, dtype=torch.float32, device="cuda:0")
#     embd = embd.permute((2, 0, 1))
#     embd.requires_grad_(True)
#     return embd


def get_optimizer(
    params: torch.Tensor, lr: float = 0.01, weight_decay: float = 0.0
) -> torch.optim.Optimizer:
    return torch.optim.AdamW([params], lr=lr)  # , weight_decay=weight_decay)
    # return torch.optim.LBFGS([params], lr=lr, max_iter=10, history_size=100, line_search_fn="strong_wolfe")


def torch_to_pil(image: torch.Tensor) -> Image.Image:
    image = (image.detach() * 255.0).clamp(0, 255)
    image = image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    image = Image.fromarray(image)
    return image


def optimize_pixel_wise(
    evaluator: VQAEvaluator,
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
    evaluator.model.eval()
    evaluator.model.requires_grad_(False)

    image = get_initial_image(image_shape)

    starting_image = image.detach().clone()

    optimizer = get_optimizer(image, lr=learning_rate, weight_decay=weight_decay)
    pbar = tqdm(total=num_iterations)

    for iter_idx in range(num_iterations):
        optimizer.zero_grad()
        image_preproc = apply_preprocessing_torch(image)
        loss = -vqa_score_gradient(evaluator, image_preproc, questions, choices_list, answers)
        loss.backward()
        optimizer.step()

        if iter_idx == 0 or (iter_idx + 1) % validation_steps == 0:
            pil_image = ImageProcessor(torch_to_pil(image)).apply().image
            val_loss = vqa_score_original(evaluator, pil_image, questions, choices_list, answers)
            loss_diff = np.abs(-loss.item() - val_loss)

        pbar.set_description(
            f"Iteration {iter_idx}/{num_iterations} | Loss: {-loss.item():.4f} | Val Loss: {val_loss:.4f} | Loss Diff: {loss_diff:.4f}"
        )
        pbar.update(1)

    final_diff = (starting_image - image).abs()
    final_diff_img = torch_to_pil(final_diff)

    image = torch_to_pil(image)

    return image


if __name__ == "__main__":
    evaluator = VQAEvaluator()
    

    

    description = "a yellow forest at dusk"
    questions = [
        # "What is the main setting of the image?",
        # "Is there anything yellow in the image?",
        # "What time of day is suggested in the image?",
        # "What color is prominently featured in the image?",
        "What color do you see in the image?",
    ]
    choices_list = [
        # ["beach", "desert", "forest", "mountain"],
        # ["no", "yes"],
        # ["dawn", "dusk", "midday", "midnight"],
        # ["green", "orange", "yellow", "white"],
        ["red", "black", "white", "yellow"],
    ]
    answers = [
        # "forest",
        # "yes",
        # "dusk",
        # "yellow",
        "yellow",
    ]

    image = Image.open("output.png").resize((384, 384))
    image = ImageProcessor(image).apply().image
    score = vqa_score_original(evaluator, image, questions, choices_list, answers)
    print(f"Score: {score}")
    exit()


    image = optimize_pixel_wise(
        evaluator=evaluator,
        description=description,
        questions=questions,
        choices_list=choices_list,
        answers=answers,
        num_iterations=50,
        validation_steps=5,
        learning_rate=5e-2,
        image_shape=(3, 16, 16),
    )

    image.save("output.png")
