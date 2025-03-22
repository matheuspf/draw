import string
import copy
import cv2

import numpy as np
import torch
from src.score_original import VQAEvaluator, ImageProcessor, AestheticEvaluator
from src.preprocessing import apply_preprocessing_torch
from PIL import Image
from torch.nn import functional as F
from src.score_gradient import (
    vqa_score_original,
    vqa_score_gradient,
    aesthetic_score_gradient,
    aesthetic_score_original,
    score_original,
    score_gradient,
)


def test_image_inputs():
    with_preprocessing = True

    evaluator = VQAEvaluator()
    # image_pil = Image.open("/home/mpf/code/kaggle/draw/org.png")
    image_pil = Image.open("/home/mpf/code/label_studio/examples_poe2/poe_0.jpg")
    torch_image = torch.tensor(np.array(image_pil), dtype=torch.float32, device="cuda:0")
    torch_image = (torch_image / 255.0).permute(2, 0, 1)

    if with_preprocessing:
        image_processor = ImageProcessor(copy.deepcopy(image_pil))
        image_processor.apply()
        image_pil = image_processor.image

        torch_image = apply_preprocessing_torch(torch_image)

    inputs = evaluator.processor(
        images=image_pil,
        text="",
        return_tensors="pt",
        padding="longest",
    ).to("cuda:0")
    org_image = inputs["pixel_values"]

    torch_image = F.interpolate(
        torch_image.unsqueeze(0),
        size=(org_image.shape[-2], org_image.shape[-1]),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    torch_image = (torch_image - 0.5) / 0.5

    diff = (org_image - torch_image).abs()

    print(org_image.max(), org_image.min(), org_image.mean())
    print(torch_image.max(), torch_image.min(), torch_image.mean())
    print(diff.max(), diff.min(), diff.mean())

    diff_img = (
        (diff[0] * 255)
        .clamp(0, 255)
        .permute(1, 2, 0)
        .round()
        .detach()
        .cpu()
        .numpy()
        .astype(np.uint8)
    )
    cv2.imwrite("diff_grad.png", diff_img)

    # assert diff.max() < 0.1 and diff.mean() < 0.01


def test_vqa_score_gradient():
    evaluator = VQAEvaluator()

    description = "a purple forest at dusk"
    question = "What is the main setting of the image?"
    choices = ["beach", "desert", "forest", "mountain"]
    answer = "forest"

    mean_distance = 0.0

    image_list = [
        "/home/mpf/code/kaggle/draw/org.png",
        "/home/mpf/code/kaggle/draw/processed.png",
        "/home/mpf/code/vision/base.png",
        "/home/mpf/code/label_studio/examples_poe2/poe_0.jpg",
        "/home/mpf/code/label_studio/examples_poe2/poe_1.jpg",
        "/home/mpf/code/label_studio/examples_poe2/poe_2.jpg",
        "/home/mpf/code/label_studio/examples_poe2/poe_3.jpg",
        "/home/mpf/code/label_studio/examples_poe2/poe_4.jpg",
        "/home/mpf/code/label_studio/examples_poe2/poe_5.jpg",
        "/home/mpf/Downloads/f1.jpeg",
        "/home/mpf/Downloads/f2.jpg",
        "/home/mpf/Downloads/f3.jpg",
    ]

    for image_path in image_list:
        org_image = Image.open(image_path)

        image_processor = ImageProcessor(copy.deepcopy(org_image))
        image_processor.apply()
        image = image_processor.image

        original_score = vqa_score_original(evaluator, image, question, choices, answer)

        torch_image = torch.tensor(np.array(org_image), dtype=torch.float32, device="cuda:0")
        torch_image = (torch_image / 255.0).permute(2, 0, 1)

        torch_image = apply_preprocessing_torch(torch_image).unsqueeze(0)
        gradient_score = vqa_score_gradient(
            evaluator, torch_image, question, choices, answer
        ).item()

        diff = np.abs(original_score - gradient_score)

        print(f"image_path: {image_path}")
        print(f"original_score: {original_score}")
        print(f"gradient_score: {gradient_score}")
        print(f"diff: {diff}")
        print("-" * 30)

        mean_distance += diff / len(image_list)

    print("\n\n")
    print(f"mean_distance: {mean_distance}")


def test_aesthetic_score_gradient():
    evaluator = AestheticEvaluator()

    image_list = [
        "/home/mpf/code/kaggle/draw/org.png",
        "/home/mpf/code/kaggle/draw/processed.png",
        "/home/mpf/code/vision/base.png",
    ]

    mean_distance = 0.0

    for image_path in image_list:
        org_image = Image.open(image_path)
        image_processor = ImageProcessor(copy.deepcopy(org_image))
        image_processor.apply()
        image = image_processor.image

        original_score = aesthetic_score_original(evaluator, image)

        torch_image = torch.tensor(np.array(org_image), dtype=torch.float32, device="cuda:0")
        torch_image = (torch_image / 255.0).permute(2, 0, 1)

        torch_image = apply_preprocessing_torch(torch_image)
        gradient_score = aesthetic_score_gradient(evaluator, torch_image).item()

        diff = np.abs(original_score - gradient_score)

        print(f"image_path: {image_path}")
        print(f"original_score: {original_score}")
        print(f"gradient_score: {gradient_score}")
        print(f"diff: {diff}")
        print("-" * 30)

        mean_distance += diff / len(image_list)


def test_score_gradient():
    vqa_evaluator = VQAEvaluator()
    aesthetic_evaluator = AestheticEvaluator()

    description = "a purple forest at dusk"
    question = "What is the main setting of the image?"
    choices = ["beach", "desert", "forest", "mountain"]
    answer = "forest"

    mean_distance = 0.0

    image_list = [
        "/home/mpf/code/kaggle/draw/org.png",
        "/home/mpf/code/kaggle/draw/processed.png",
        "/home/mpf/code/vision/base.png",
        "/home/mpf/code/label_studio/examples_poe2/poe_0.jpg",
        "/home/mpf/code/label_studio/examples_poe2/poe_1.jpg",
        "/home/mpf/code/label_studio/examples_poe2/poe_2.jpg",
        "/home/mpf/code/label_studio/examples_poe2/poe_3.jpg",
        "/home/mpf/code/label_studio/examples_poe2/poe_4.jpg",
        "/home/mpf/code/label_studio/examples_poe2/poe_5.jpg",
        "/home/mpf/Downloads/f1.jpeg",
        "/home/mpf/Downloads/f2.jpg",
        "/home/mpf/Downloads/f3.jpg",
    ]

    for image_path in image_list:
        org_image = Image.open(image_path)

        image_processor = ImageProcessor(copy.deepcopy(org_image))
        image_processor.apply()
        image = image_processor.image

        original_score, _, _ = score_original(
            vqa_evaluator, aesthetic_evaluator, image, [question], [choices], [answer]
        )

        torch_image = torch.tensor(np.array(org_image), dtype=torch.float32, device="cuda:0")
        torch_image = (torch_image / 255.0).permute(2, 0, 1)

        torch_image = apply_preprocessing_torch(torch_image)
        gradient_score, _, _ = score_gradient(
            vqa_evaluator, aesthetic_evaluator, torch_image, [question], [choices], [answer]
        )

        diff = np.abs(original_score - gradient_score.item())

        print(f"image_path: {image_path}")
        print(f"original_score: {original_score}")
        print(f"gradient_score: {gradient_score.item()}")
        print(f"diff: {diff}")
        print("-" * 30)

        mean_distance += diff / len(image_list)

    print("\n\n")
    print(f"mean_distance: {mean_distance}")


# test_vqa_score_gradient()
# test_aesthetic_score_gradient()
test_score_gradient()
