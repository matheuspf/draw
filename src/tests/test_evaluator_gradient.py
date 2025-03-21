import string
import copy
import cv2

import numpy as np
import torch
from src.score_original import VQAEvaluator, ImageProcessor
from src.preprocessing import apply_preprocessing_torch
from PIL import Image
from torch.nn import functional as F

def vqa_score_original(
    evaluator: VQAEvaluator,
    image: Image.Image,
    question: str,
    choices: list[str],
    answer: str,
) -> float:
    return evaluator.score(
        questions=[question],
        choices=[choices],
        answers=[answer],
        image=image,
    )


def _check_inputs(evaluator: VQAEvaluator, choices: list[str], answer: str):
    if not answer or not choices:
        raise ValueError(f"Invalid answer format: |{answer}|, possible choices: |{choices}|")

    if not answer in choices:
        raise ValueError(f"Invalid answer format: |{answer}|, possible choices: |{choices}|")
    
    try:
        first_token, first_token_with_space = _get_choice_tokens(evaluator)
    except ValueError as e:
        raise ValueError(f"Invalid token mapping at get_choice_tokens: {e}")
    
    if first_token.min() < 0 or first_token.max() >= evaluator.processor.tokenizer.vocab_size:
        raise ValueError(f"Invalid first token: {first_token}")
    
    if first_token_with_space.min() < 0 or first_token_with_space.max() >= evaluator.processor.tokenizer.vocab_size:
        raise ValueError(f"Invalid first token with space: {first_token_with_space}")

    decode_string = evaluator.processor.tokenizer.decode(first_token)
    decode_string_with_space = evaluator.processor.tokenizer.decode(first_token_with_space)

    assert decode_string == string.ascii_uppercase
    assert decode_string_with_space == ' ' + ' '.join(string.ascii_uppercase)


def _get_choice_tokens(evaluator: VQAEvaluator) -> torch.Tensor:
    letters = string.ascii_uppercase
    first_token = torch.tensor([
        evaluator.processor.tokenizer.encode(letter_token, add_special_tokens=False)[0]
        for letter_token in letters
    ], dtype=torch.long)    
    first_token_with_space = torch.tensor([
        evaluator.processor.tokenizer.encode(" " + letter_token, add_special_tokens=False)[0]
        for letter_token in letters
    ], dtype=torch.long)

    return first_token, first_token_with_space


def vqa_score_gradient(
    evaluator: VQAEvaluator,
    image: torch.Tensor,
    question: str,
    choices: list[str],
    answer: str,
    # img_shape=(224, 224),
    img_shape=(448, 448),
) -> torch.Tensor:
    _check_inputs(evaluator, choices, answer)

    image = F.interpolate(
        image, size=img_shape, mode="bicubic", align_corners=False, antialias=True
    )
    image = (image - 0.5) / 0.5

    prompt = evaluator.format_prompt(question, choices)
    inputs = evaluator.processor(
        # images=Image.new("RGB", img_shape),
        images=Image.open("/home/mpf/code/kaggle/draw/org.png"),
        text=prompt,
        return_tensors='pt',
        padding='longest',
    ).to('cuda:0')
    inputs["pixel_values"] = image

    outputs = evaluator.model(**inputs)
    logits = outputs.logits[:, -1, :]

    first_token, first_token_with_space = _get_choice_tokens(evaluator)

    masked_logits = torch.full_like(logits, float('-inf'))
    masked_logits[:, first_token] = logits[:, first_token]
    masked_logits[:, first_token_with_space] = logits[:, first_token_with_space]
    probabilities = torch.softmax(masked_logits, dim=-1)

    choice_probabilities = probabilities[:, first_token] + probabilities[:, first_token_with_space]
    total_prob = choice_probabilities.sum(dim=-1)

    choice_probabilities = choice_probabilities / total_prob

    answer_index = choices.index(answer)
    answer_probability = choice_probabilities[0, answer_index]

    return answer_probability



def test_image_inputs():
    with_preprocessing = True
    




    evaluator = VQAEvaluator()
    # image_pil = Image.open("/home/mpf/code/kaggle/draw/org.png")
    image_pil = Image.open("/home/mpf/code/label_studio/examples_poe2/poe_0.jpg")
    torch_image = torch.tensor(np.array(image_pil), dtype=torch.float32, device='cuda:0')
    torch_image = (torch_image / 255.0).permute(2, 0, 1)

    if with_preprocessing:
        image_processor = ImageProcessor(copy.deepcopy(image_pil))
        image_processor.apply()
        image_pil = image_processor.image
        
        torch_image = apply_preprocessing_torch(torch_image)
    
    inputs = evaluator.processor(
        images=image_pil,
        text="",
        return_tensors='pt',
        padding='longest',
    ).to('cuda:0')
    org_image = inputs["pixel_values"]

    torch_image = F.interpolate(
        torch_image.unsqueeze(0), size=(org_image.shape[-2], org_image.shape[-1]), mode="bicubic", align_corners=False, antialias=True
    )
    torch_image = (torch_image - 0.5) / 0.5

    diff = (org_image - torch_image).abs()

    print(org_image.max(), org_image.min(), org_image.mean())
    print(torch_image.max(), torch_image.min(), torch_image.mean())
    print(diff.max(), diff.min(), diff.mean())

    diff_img = (diff[0] * 255).clamp(0, 255).permute(1, 2, 0).round().detach().cpu().numpy().astype(np.uint8)
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

        torch_image = torch.tensor(np.array(org_image), dtype=torch.float32, device='cuda:0')
        torch_image = (torch_image / 255.0).permute(2, 0, 1)

        torch_image = apply_preprocessing_torch(torch_image).unsqueeze(0)
        gradient_score = vqa_score_gradient(evaluator, torch_image, question, choices, answer).item()

        diff = np.abs(original_score - gradient_score)

        print(f"image_path: {image_path}")
        print(f"original_score: {original_score}")
        print(f"gradient_score: {gradient_score}")
        print(f"diff: {diff}")
        print("-" * 30)
        
        mean_distance += diff / len(image_list)

    print("\n\n")
    print(f"mean_distance: {mean_distance}")


test_vqa_score_gradient()


# test_image_inputs()

