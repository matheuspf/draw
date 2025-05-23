import pandas as pd
import ast
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from src.scripts.data.generate_svgs import clamp_svg
from src.score_original import VQAEvaluator, AestheticEvaluator, ImageProcessor, harmonic_mean, svg_to_png, device_0
from src.text_to_svg import text_to_svg


def process_row(
    row: dict,
    vqa_evaluator: VQAEvaluator,
    aesthetic_evaluator: AestheticEvaluator,
    random_seed: int = 42,
    max_questions: int = 4
) -> tuple[float, float, float, float, list[dict]]:

    questions = row["question"][:max_questions]
    choices = row["choices"][:max_questions]
    answers = row["answer"][:max_questions]
    
    svg_path = row["svg_path"]

    with open(svg_path, "r", encoding="utf-8") as f:
        svg = f.read()

    svg = clamp_svg(svg)
    svg += text_to_svg("O", x_position_frac=0.75, y_position_frac=0.85, font_size=60, color=(0, 0, 0), font_path="/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf").split("\n")[1]
    svg = svg.replace("</svg>", "") + "</svg>"

    image = svg_to_png(svg)

    processor = ImageProcessor(image=image, seed=random_seed, crop=False).apply()
    proc_img = processor.image.copy()

    aest_score = aesthetic_evaluator.score(proc_img)
    vqa_score, batched_choice_probabilities, logits_dict = vqa_evaluator.score(
        questions, choices, answers, proc_img, n=len(choices)
    )
    
    # processor.reset().apply_random_crop_resize().apply_jpeg_compression(quality=90)
    # ocr_score = vqa_evaluator.ocr(processor.image)
    ocr_score = 1.0

    instance_score = harmonic_mean(vqa_score, aest_score, beta=0.5) * ocr_score

    return instance_score, vqa_score, aest_score, ocr_score, batched_choice_probabilities, logits_dict


def generate_scores(
    dataset_path: Path = Path("/home/mpf/code/kaggle/draw/data/vtracer/vtracer_dataset.parquet"),
    out_folder: Path = Path("/home/mpf/code/kaggle/draw/data/vtracer"),
    random_seed: int = 42
) -> None:
    svg_df = pd.read_parquet(dataset_path)
    svg_df = svg_df.drop_duplicates(subset=["description", "sd_seed", "prompt"])
    svg_df = svg_df[svg_df["sd_seed"].isin([0])]
    # svg_df = svg_df[svg_df["prompt_name"].isin(list(set(svg_df["prompt_name"]))[:2])]
    svg_df.drop("category", axis=1, inplace=True)

    questions_df = pd.read_parquet("/home/mpf/code/kaggle/draw/data/questions_groq_llama4_maverick.parquet")

    df = pd.merge(svg_df, questions_df, on="description", how="left")

    print(f"Merged {len(svg_df)} SVGs with {len(questions_df)} questions into {len(df)} rows")
    print(f"Number of unique questions: {len(df['question'].unique())}")

    for colname in ['question', 'choices', 'answer']:
        df[colname] = df[colname].apply(ast.literal_eval)
    
    vqa_evaluator = VQAEvaluator()
    aesthetic_evaluator = AestheticEvaluator(device_0)

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        res = process_row(row, vqa_evaluator, aesthetic_evaluator, random_seed=random_seed, max_questions=4)
        results.append(res)

    scores, vqa_scores, aest_scores, ocr_scores, batched_probs, logits_dict = zip(*results)
    df["score"] = scores
    df["vqa_score"] = vqa_scores
    df["aest_score"] = aest_scores
    df["ocr_score"] = ocr_scores
    df["batched_choice_probabilities"] = [repr(prob) for prob in batched_probs]
    df["logits_dict"] = [repr(logits) for logits in logits_dict]
    
    for colname in ['question', 'choices', 'answer']:
        df[colname] = df[colname].apply(repr)

    out_folder.mkdir(parents=True, exist_ok=True)
    out_path = out_folder / "vtracer_dataset_scores.parquet"
    df.to_parquet(out_path)

    print(f"Saved scored dataset to {out_path}")


if __name__ == "__main__":
    generate_scores()
