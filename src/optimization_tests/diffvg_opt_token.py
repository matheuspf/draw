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
    score_gradient_ocr_1,
    score_gradient_ocr_2,
    _check_inputs,
    _get_choice_tokens,
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
from src.utils import optimize_svg, svg_to_png, create_random_svg
from src.text_to_svg import text_to_svg, rgb_to_hex
import pydiffvg
import kagglehub


def svg_to_png_no_resize(svg_code: str) -> Image.Image:
    png_data = cairosvg.svg2png(bytestring=svg_code.encode('utf-8'))
    return Image.open(io.BytesIO(png_data)).convert('RGB')


svg_constraints = kagglehub.package_import("metric/svg-constraints", bypass_confirmation=True)



def format_prompt(prompt: str) -> str:
    return "<image>ocr\n" + prompt

def ocr_score(
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

    len_empty_prompt = evaluator.processor(
        images=Image.new("RGB", image_shape),
        text=format_prompt(""),
        return_tensors="pt",
    ).input_ids.shape[-1]
    inputs = evaluator.processor(
        images=Image.new("RGB", image_shape),
        text=format_prompt(target),
        return_tensors="pt",
        # padding="longest",
    ).to("cuda:0")
    inputs["pixel_values"] = image.repeat(1, 1, 1, 1)
    
    inputs["labels"] = copy.deepcopy(inputs.input_ids)
    inputs["labels"][:, :len_empty_prompt] = -100

    outputs = evaluator.model(**inputs)
    loss = outputs.loss

    return -loss



def get_optimization_settings():
    # Create optimization settings
    settings = pydiffvg.SvgOptimizationSettings()

    lr = 1e-2

    # Configure optimization settings
    settings.global_override(["optimizer"], "Adam")
    settings.global_override(["color_lr"], lr)
    settings.global_override(["alpha_lr"], lr)
    settings.global_override(["paths", "shape_lr"], 10*lr)
    settings.global_override(["circles", "shape_lr"], 10*lr)
    settings.global_override(["transforms", "transform_lr"], 10*lr)

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


def map_text_to_embeddings(evaluator, text):
    tokenization_result = evaluator.processor.tokenizer([text], return_tensors="pt", return_offsets_mapping=True).to("cuda:0")
    input_ids = tokenization_result.input_ids
    offsets = tokenization_result.offset_mapping[0]
    
    inputs_embeds = evaluator.model.get_input_embeddings()(input_ids)
    
    word_to_embedding = []
    for i, (start, end) in enumerate(offsets):
        start = start.item()
        end = end.item()
        if start == end:
            continue
        word = text[start:end]
        word_to_embedding.append((word, (start, end), inputs_embeds[0, i]))

    return word_to_embedding




def get_initial_svg(
    target_text: str,
    canvas_width: int = 384,
    canvas_height: int = 384,
    num_tiles: int = 4,
    points_per_edge: int = 1,
):
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_width}" height="{canvas_height}">\n'
    svg += f'  <path id="background-0" d="M 0,0 h {canvas_width} v {canvas_height} h {-canvas_width} z" fill="{rgb_to_hex(0, 0, 0)}" />\n'
    # svg += f'  <path id="background-0" d="M 0,0 h {canvas_width} v {canvas_height} h {-canvas_width} z" fill="{rgb_to_hex(255, 255, 255)}" />\n'
    # svg += f'  <path d="M 0,0 h {canvas_width} v {canvas_height//2} h {-canvas_width} z" fill="{rgb_to_hex(255, 255, 255)}" />\n'
    # svg += f'  <path d="M 0,0 h {canvas_width} v {int(0.5*canvas_height)} h {-canvas_width} z" fill="{rgb_to_hex(255, 255, 255)}" />\n'

    for i in range(num_tiles):
        for j in range(num_tiles):
            # steps = 4
            # if not(i < steps or j < steps or i >= num_tiles - steps or j >= num_tiles - steps):
            #     continue

            # if j > num_tiles // 4 or i > num_tiles // 4:
            #     continue

            if j > 0.5 * num_tiles:
                continue

            x = i * (canvas_width // num_tiles)
            y = j * (canvas_height // num_tiles)
            width = canvas_width // num_tiles
            height = canvas_height // num_tiles
            random_color = rgb_to_hex(
                random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
            )

            points_per_edge = random.randint(1, 5)

            # Create path with more control points
            if points_per_edge <= 1:
                # Original rectangle with 4 points
                svg += f'  <path d="M {x},{y} h {width} v {height} h {-width} z" fill="{random_color}" />\n'
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

                svg += f'  <path d="{path_data}" fill="{random_color}" />\n'
            # svg += f'  <rect x="{x}" y="{y}" width="{canvas_width // num_tiles}" height="{canvas_height // num_tiles}" fill="{random_color}" />\n'
            # svg += f'  <ellipse cx="{x + canvas_width // num_tiles // 2}" cy="{y + canvas_height // num_tiles // 2}" rx="{canvas_width // num_tiles // 2}" ry="{canvas_height // num_tiles // 2}" fill="{random_color}" />\n'

    # text_svg = text_to_svg(text="A", svg_width=canvas_width, svg_height=canvas_height, x_position_frac=0.2, y_position_frac=0.2, font_size=50, color=(0, 0, 0))
    # svg += "\n".join(text_svg.split("\n")[1:-1])

    text_svg = text_to_svg(target_text, svg_width=canvas_width, svg_height=canvas_height, color=(255, 255, 255), x_position_frac=0.1, y_position_frac=0.7, font_size=40)
    svg += "\n".join(text_svg.split("\n")[1:-1])

    svg += "</svg>"
    # svg = optimize_svg(svg)

    with open("initial_svg.svg", "w") as f:
        f.write(svg)

    print(f"Initial Length SVG: {len(svg.encode('utf-8'))}")

    return svg


def optimize_diffvg(
    vqa_evaluator: VQAEvaluator,
    aesthetic_evaluator: AestheticEvaluator,
    target_text: str,
    questions: list[str],
    choices_list: list[list[str]],
    answers: list[str],
    canvas_width: int = 384,
    canvas_height: int = 384,
    num_iterations: int = 100,
    validation_steps: int = 10,
) -> Image.Image:
    vqa_evaluator.model.eval()
    vqa_evaluator.model.requires_grad_(False)
    aesthetic_evaluator.predictor.eval()
    aesthetic_evaluator.predictor.requires_grad_(False)
    aesthetic_evaluator.clip_model.eval()
    aesthetic_evaluator.clip_model.requires_grad_(False)

    pydiffvg.set_use_gpu(torch.cuda.is_available())

    svg_content = get_initial_svg(target_text, canvas_width, canvas_height, num_tiles=448//8)

    temp_svg_path = "/tmp/initial_svg.svg"
    with open(temp_svg_path, "w") as f:
        f.write(svg_content)

    settings = get_optimization_settings()

    text_path_ids = [f"text-path-{i}" for i in range(100)] + [f"background-{i}" for i in range(10)] + [f"bg-2"]
    for text_id in text_path_ids:
        text_settings = settings.undefault(text_id)
        text_settings["paths"]["optimize_points"] = False
        text_settings["optimize_color"] = False
        text_settings["optimize_alpha"] = False
        text_settings["optimize_transforms"] = False
        # text_settings["paths"]["shape_lr"] = 1.0
        # text_settings["transforms"]["transform_lr"] = 1.0
        # text_settings["color_lr"] = 0.1
        # text_settings["alpha_lr"] = 0.1

    optim_svg = pydiffvg.OptimizableSvg(
        temp_svg_path, settings, optimize_background=True, verbose=False, device="cuda:0"
    )
    
    # target_text = "<eos><eos><eos>"
    
    text_pos_info = map_text_to_embeddings(vqa_evaluator, target_text)
    text_embds = torch.stack([x[2] for x in text_pos_info], dim=0)
    

    validator = svg_constraints.SVGConstraints(max_svg_size=10000)
    best_svg = optim_svg.write_xml()
    best_val_loss = -1e8

    pbar = tqdm(total=num_iterations)

    for iter_idx in range(num_iterations):
        optim_svg.zero_grad()
        image = optim_svg.render(seed=iter_idx)

        img = image[:, :, :3].permute(2, 0, 1).clamp(0, 1).unsqueeze(0)

        
        image_shape = (
            vqa_evaluator.processor.image_processor.size["height"],
            vqa_evaluator.processor.image_processor.size["width"],
        )
        img = F.interpolate(
            img, size=image_shape, mode="bicubic", align_corners=False, antialias=True
        )
        img = (img - 0.5) / 0.5
        # img = apply_preprocessing_torch(img)
                
        
        inputs = vqa_evaluator.processor(
            images=Image.new("RGB", image_shape),
            text=format_prompt(target_text),
            return_tensors="pt",
            # padding="longest",
        ).to("cuda:0")
        inputs["pixel_values"] = img
        
            
        outputs = vqa_evaluator.model(**inputs)
            
        
        
        # image_features = vqa_evaluator.model.get_image_features(img)

        # sz = 4
        # image_features = image_features.permute((0, 2, 1))
        # image_features = image_features.reshape((image_features.shape[0], image_features.shape[1], image_features.shape[2] // sz, sz))
        # image_features = image_features.sum(dim=-1)
        # image_features = image_features.permute((0, 2, 1))
        # image_features = image_features[:, -len(text_embds):]
        

        # sz = 10
        # arange_idx = torch.arange(image_features.shape[1] - len(text_embds) * sz, image_features.shape[1], sz)
        # image_features = image_features[:, arange_idx]

        # image_features = image_features[0, :len(text_embds)]
        # image_logits = vqa_evaluator.model.language_model.lm_head(image_features)

        
        labels = vqa_evaluator.processor.tokenizer(target_text, return_tensors="pt").to("cuda:0").input_ids[0]
        image_logits = outputs.logits[0, :len(labels)]

        print(vqa_evaluator.processor.tokenizer.decode(image_logits.argmax(dim=-1)))
        
        loss = F.cross_entropy(image_logits, labels)
        
        # loss = -(image_features - text_embds).pow(2).sum(dim=-1).mean()
        # loss = F.cosine_similarity(image_features / image_features.norm(dim=-1, keepdim=True), text_embds / text_embds.norm(dim=-1, keepdim=True), dim=-1).mean()



        if iter_idx == 0 or (iter_idx + 1) % validation_steps == 0:
            cur_svg = optimize_svg(optim_svg.write_xml())
            pil_image = svg_to_png_no_resize(cur_svg)
            # pil_image = ImageProcessor(copy.deepcopy(pil_image)).apply().image

            torch.cuda.empty_cache()
            val_loss, vqa_val_loss, aest_val_loss, ocr_loss, ocr_text = score_original(
                vqa_evaluator,
                aesthetic_evaluator,
                pil_image,
                questions,
                choices_list,
                answers,
            )
            
            ocr_text = ocr_text.replace("\n", "")

            if val_loss > best_val_loss:
                best_val_loss = val_loss
                best_svg = cur_svg
            
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

        loss = loss
        loss.backward()
        optim_svg.step()

    best_svg = optim_svg.write_xml()

    return best_svg



def evaluate():
    seed_everything(42)

    vqa_evaluator = VQAEvaluator()
    aesthetic_evaluator = AestheticEvaluator()
    
    mean_score_gt = 0
    mean_score_gen = 0

    run_inference = True
    if run_inference:
        df = pd.read_parquet("/home/mpf/code/kaggle/draw/question_generation_results.parquet")
        df = df[df["set"] == "test"].iloc[1:2]

        for index, row in tqdm(df.iterrows(), total=len(df)):
            num_ex = 4
            description = row["description"]
            print(description)

            gt_questions_list = {
                "questions": [dct["question"] for dct in row["ground_truth"]],
                "choices_list": [dct["choices"].tolist() for dct in row["ground_truth"]],
                "answers": [dct["answer"] for dct in row["ground_truth"]]
            }
            gen_questions_list = {
                "questions": [dct["question"] for dct in row["generated"]][:num_ex],
                "choices_list": [dct["choices"].tolist() for dct in row["generated"]][:num_ex],
                "answers": [dct["answer"] for dct in row["generated"]][:num_ex]
            }
            gen_questions_list_inpt = gen_questions_list

            run_optimization = True
            if run_optimization:
                svg = optimize_diffvg(
                    vqa_evaluator=vqa_evaluator,
                    aesthetic_evaluator=aesthetic_evaluator,
                    target_text=description,
                    questions=gen_questions_list_inpt["questions"],
                    choices_list=gen_questions_list_inpt["choices_list"],
                    answers=gen_questions_list_inpt["answers"],
                    canvas_width=224,
                    canvas_height=224,
                    num_iterations=1000,
                    validation_steps=50,
                )

            else:
                with open("output.svg") as f:
                    svg = f.read()

            with open("output.svg", "w") as f:
                f.write(svg)

            print(f"Length SVG: {len(svg.encode('utf-8'))}")

            opt_svg = optimize_svg(svg)

            with open("output_opt.svg", "w") as f:
                f.write(opt_svg)

            print(f"Length SVG Optimized: {len(opt_svg.encode('utf-8'))}")

            image = svg_to_png_no_resize(opt_svg)
            image.save("output.png")

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

            mean_score_gt += score_gt[0]
            mean_score_gen += score_gen[0]

    print(f"Mean Score GT: {mean_score_gt / len(df)}")
    print(f"Mean Score Gen: {mean_score_gen / len(df)}")



def test_token(evaluator):
    
    image_pil = Image.open("t1.png")
    image = torch.from_numpy(np.array(image_pil)).permute(2, 0, 1).float().to("cuda:0") / 255.0
    

    image_shape = (
        evaluator.processor.image_processor.size["height"],
        evaluator.processor.image_processor.size["width"],
    )
    image = F.interpolate(
        image.unsqueeze(0), size=image_shape, mode="bicubic", align_corners=False, antialias=True
    )
    image = (image - 0.5) / 0.5

    
    text = "A A"

    inputs = evaluator.processor(
        images=image_pil,
        # text="<image>ocr\n",
        text="<image>" + text,
        return_tensors="pt",
    ).to("cuda:0")

    
    inputs_text = evaluator.processor.tokenizer(text, return_tensors="pt").to("cuda:0")
    
    image_feature_size = 1024 if image_shape[0] == 448 else 256
    
    image_features = evaluator.model.get_image_features(image)

    print(image_features.shape)

    import pdb; pdb.set_trace()

    print(inputs.input_ids[:, -20:])
    print("\n\n")
    print(inputs_text.input_ids[:, -20:])
    assert inputs_text.input_ids[0, 0] == inputs.input_ids[0, image_feature_size+1]
    assert inputs_text.input_ids[0, 1] == inputs.input_ids[0, image_feature_size+2]
    
    import pdb; pdb.set_trace()
    inputs_embeds = evaluator.model.get_input_embeddings()(inputs.input_ids)

    text_to_embeds_mapping = map_text_to_embeddings(evaluator, text, image_feature_size)
    print("Text to embeddings mapping:")
    for word, embedding in text_to_embeds_mapping.items():
        print(f"Word: {word}, Embedding shape: {embedding.shape}")
    

    # inputs["pixel_values"] = image.repeat(1, 1, 1, 1)
    
    inputs["labels"] = copy.deepcopy(inputs.input_ids)

    outputs = evaluator.model(**inputs)
    loss = outputs.loss

    return -loss


if __name__ == "__main__":
    evaluate()
    
    # evaluator = VQAEvaluator()
    # test_token(evaluator)

    # # text = "A A B C B"
    # # dct = map_text_to_embeddings(evaluator, text)
    # # print([x[:2] for x in dct])
    # # import pdb; pdb.set_trace()
