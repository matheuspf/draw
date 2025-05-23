import string
import copy
import torch
from src.score_original import VQAEvaluator, AestheticEvaluator, harmonic_mean, ImageProcessor
from src.preprocessing import apply_preprocessing_torch
from PIL import Image
from torch.nn import functional as F


def harmonic_mean_grad(a: float, b: float, beta: float = 1.0) -> float:
    return (1 + beta**2) * (a * b) / (beta**2 * a + b)


def score_original(
    vqa_evaluator: VQAEvaluator,
    aesthetic_evaluator: AestheticEvaluator,
    image: Image.Image,
    questions: list[str],
    choices_list: list[list[str]],
    answers: list[str],
    apply_preprocessing: bool = True,
) -> float:
    if apply_preprocessing:
        proc_image = ImageProcessor(copy.deepcopy(image)).apply().image
    else:
        proc_image = image
    
    vqa_score = vqa_score_original(vqa_evaluator, proc_image, questions, choices_list, answers)
    aesthetic_score = aesthetic_score_original(aesthetic_evaluator, proc_image)
    ocr_score = vqa_evaluator.ocr(image)
    score = harmonic_mean(vqa_score, aesthetic_score, beta=0.5) * ocr_score[0]

    return score, vqa_score, aesthetic_score, ocr_score[0], ocr_score[1]


def vqa_score_original(
    evaluator: VQAEvaluator,
    image: Image.Image,
    questions: list[str],
    choices_list: list[list[str]],
    answers: list[str],
) -> float:
    return evaluator.score(
        questions=questions,
        choices=choices_list,
        answers=answers,
        image=image,
    )

def aesthetic_score_original(
    evaluator: AestheticEvaluator,
    image: Image.Image,
) -> float:
    return evaluator.score(image=image)



def _check_inputs(evaluator: VQAEvaluator, choices_list: list[list[str]], answers: list[str]):
    for choices, answer in zip(choices_list, answers):
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

        if (
            first_token_with_space.min() < 0
            or first_token_with_space.max() >= evaluator.processor.tokenizer.vocab_size
        ):
            raise ValueError(f"Invalid first token with space: {first_token_with_space}")

        decode_string = evaluator.processor.tokenizer.decode(first_token)
        decode_string_with_space = evaluator.processor.tokenizer.decode(first_token_with_space)

        assert decode_string == string.ascii_uppercase
        assert decode_string_with_space == " " + " ".join(string.ascii_uppercase)


def _get_choice_tokens(evaluator: VQAEvaluator) -> torch.Tensor:
    letters = string.ascii_uppercase
    first_token = torch.tensor(
        [
            evaluator.processor.tokenizer.encode(letter_token, add_special_tokens=False)[0]
            for letter_token in letters
        ],
        dtype=torch.long,
    )
    first_token_with_space = torch.tensor(
        [
            evaluator.processor.tokenizer.encode(" " + letter_token, add_special_tokens=False)[0]
            for letter_token in letters
        ],
        dtype=torch.long,
    )

    return first_token, first_token_with_space



def aesthetic_score_gradient(
    evaluator: AestheticEvaluator,
    image: torch.Tensor,
) -> torch.Tensor:
    image = F.interpolate(
        image.unsqueeze(0), size=(224, 224), mode="bicubic", align_corners=False, antialias=True
    )
    image = (
        image
        - torch.tensor([0.48145466, 0.4578275, 0.40821073], device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
    ) / torch.tensor([0.26862954, 0.26130258, 0.27577711], device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
    image_features = evaluator.clip_model.encode_image(image)
    image_features_norm = (image_features / image_features.norm(dim=-1, keepdim=True)).float()
    score = evaluator.predictor(image_features_norm) / 10.0  # scale to [0, 1]
    return score


def vqa_score_gradient(
    evaluator: VQAEvaluator,
    image: torch.Tensor,
    questions: list[str],
    choices_list: list[list[str]],
    answers: list[str],
) -> torch.Tensor:
    _check_inputs(evaluator, choices_list, answers)

    image_shape = (
        evaluator.processor.image_processor.size["height"],
        evaluator.processor.image_processor.size["width"],
    )
    image = F.interpolate(
        image.unsqueeze(0), size=image_shape, mode="bicubic", align_corners=False, antialias=True
    )
    image = (image - 0.5) / 0.5

    prompts = [evaluator.format_prompt(question, choice) for question, choice in zip(questions, choices_list)]
    inputs = evaluator.processor(
        images=Image.new("RGB", image_shape),
        text=prompts,
        return_tensors="pt",
        padding="longest",
    ).to("cuda:0")
    inputs["pixel_values"] = image.repeat(len(prompts), 1, 1, 1)

    outputs = evaluator.model(**inputs)
    logits = outputs.logits[:, -1, :]

    first_token, first_token_with_space = _get_choice_tokens(evaluator)

    masked_logits = torch.full_like(logits, float("-inf"))
    masked_logits[:, first_token] = logits[:, first_token]
    masked_logits[:, first_token_with_space] = logits[:, first_token_with_space]
    probabilities = torch.softmax(masked_logits, dim=-1)

    choice_probabilities = probabilities[:, first_token] + probabilities[:, first_token_with_space]
    total_prob = torch.sum(choice_probabilities, dim=-1, keepdim=True)

    choice_probabilities = choice_probabilities / total_prob

    answer_index = torch.tensor([choices.index(answer) for choices, answer in zip(choices_list, answers)])
    arange_index = torch.arange(len(answers))
    answer_probability = choice_probabilities[arange_index, answer_index]
    answer_probability = answer_probability.mean()

    # entropy = -torch.sum(choice_probabilities * torch.log(choice_probabilities + 1e-10), dim=1)
    # entropy_term = 0.1 * entropy.mean()  # Adjust coefficient as needed
    # answer_probability = answer_probability.mean() + entropy_term

    return answer_probability


def score_gradient_ocr(
    evaluator: VQAEvaluator,
    image: torch.Tensor,
    text: str = "",
    response: str | None = None,
    prefix: str = "<image>ocr\n"
) -> torch.Tensor:
    # return score_gradient_ocr_1(evaluator, image, text="", response="<eos>  purple pyramids spiraling around a bronze cone")
    # return score_gradient_ocr_1(evaluator, image, text="", response="<eos> crimson rectangles forming a chaotic grid")


    # return score_gradient_ocr_1(evaluator, image, text=text, response=response, prefix=prefix)
    # return score_gradient_ocr_2(evaluator, image, text=text, response=response)

    # return score_gradient_ocr_1(evaluator, image, text="", response=None, prefix=prefix)

    # return score_gradient_ocr_2(evaluator, image, text="", response="www\n", prefix=prefix)

    return score_gradient_ocr_1(evaluator, image, text="", response=None, prefix=prefix)

    return score_gradient_ocr_1(evaluator, image, text="", response="www.", prefix=prefix) \
            + score_gradient_ocr_1(evaluator, image, text="www.", response=None, prefix=prefix)


def score_gradient_ocr_1(
    evaluator: VQAEvaluator,
    image: torch.Tensor,
    text: str = "",
    response: str | None = None,
    prefix: str = "<image>ocr\n"
) -> torch.Tensor:
    response = response or evaluator.processor.tokenizer.eos_token
    
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
        text=prefix + text,
        return_tensors="pt",
        padding="longest",
    ).to("cuda:0")

    inputs["pixel_values"] = image
    outputs = evaluator.model(**inputs)

    logits = outputs.logits[:, -1, :]
    target_id = evaluator.processor.tokenizer.encode(response, add_special_tokens=False)[0]
    target_tensor = torch.tensor([target_id], device=logits.device)

    loss = F.cross_entropy(logits, target_tensor)
    
    return loss


def score_gradient_ocr_2(
    evaluator: VQAEvaluator,
    image: torch.Tensor,
    response: str | None = None,
    prefix: str = "<image>ocr\n"
) -> torch.Tensor:
    response = response or evaluator.processor.tokenizer.eos_token
    
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
        suffix=response,
        return_tensors="pt",
        padding="longest",
    ).to("cuda:0")
    inputs["pixel_values"] = image

    outputs = evaluator.model(**inputs)
    loss = outputs.loss

    return loss


def score_gradient(
    vqa_evaluator: VQAEvaluator,
    aesthetic_evaluator: AestheticEvaluator,
    image: torch.Tensor,
    questions: list[str],
    choices_list: list[list[str]],
    answers: list[str],
    apply_preprocessing: bool = True,
) -> torch.Tensor:
    if apply_preprocessing:
        proc_image = apply_preprocessing_torch(image)
    else:
        proc_image = image

    vqa_score_grad = vqa_score_gradient(vqa_evaluator, proc_image, questions, choices_list, answers)
    aesthetic_score_grad = aesthetic_score_gradient(aesthetic_evaluator, proc_image)
    score_grad = harmonic_mean_grad(vqa_score_grad, aesthetic_score_grad, beta=0.5)

    # score_grad_ocr = score_gradient_ocr(vqa_evaluator, image)
    # score_grad = score_grad * score_grad_ocr
    # print("\n", score_grad_ocr.item())
    score_grad_ocr = 0.0

    return score_grad, vqa_score_grad, aesthetic_score_grad, score_grad_ocr


