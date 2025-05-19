import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from src.scripts.data.generate_svgs import clamp_svg
from src.score_original import VQAEvaluator, AestheticEvaluator, ImageProcessor, harmonic_mean, svg_to_png, device_0


def process_row(
    row: dict,
    vqa_evaluator: VQAEvaluator,
    aesthetic_evaluator: AestheticEvaluator,
    rng: np.random.RandomState
) -> tuple[float, float, float, float, list[dict]]:
    """
    Process a single row: read SVG, clamp, convert to PNG, apply defenses, and compute scores.
    Returns: (instance_score, vqa_score, aest_score, ocr_score, batched_choice_probabilities)
    """
    svg_path = row["svg_path"]
    with open(svg_path, "r", encoding="utf-8") as f:
        svg = f.read()
    # clamp and convert to PNG
    clamped = clamp_svg(svg)
    image = svg_to_png(clamped)
    # apply defenses
    seed = rng.randint(0, np.iinfo(np.int32).max)
    processor = ImageProcessor(image=image, seed=seed).apply()
    proc_img = processor.image.copy()
    # aesthetic score
    aest_score = aesthetic_evaluator.score(proc_img)
    # VQA score
    vqa_score, batched_choice_probabilities = vqa_evaluator.score(
        row["question"], row["choices"], row["answer"], proc_img, n=2
    )
    # OCR score
    processor.reset().apply_random_crop_resize().apply_jpeg_compression(quality=90)
    ocr_score = vqa_evaluator.ocr(processor.image)
    # combined instance score
    instance_score = harmonic_mean(vqa_score, aest_score, beta=0.5) * ocr_score
    return instance_score, vqa_score, aest_score, ocr_score, batched_choice_probabilities


def generate_scores(
    dataset_path: Path = Path("/home/mpf/code/kaggle/draw/data/vtracer/vtracer_dataset.parquet"),
    out_folder: Path = Path("/home/mpf/code/kaggle/draw/data/vtracer"),
    max_workers: int = 1,
    random_seed: int = 0
) -> None:
    """
    Read the generated SVG dataset, score each entry, and write a new parquet with additional columns.
    """
    # load dataset
    df = pd.read_parquet(dataset_path)
    # prepare evaluators and RNG
    vqa_evaluator = VQAEvaluator()
    aesthetic_evaluator = AestheticEvaluator(device_0)
    rng = np.random.RandomState(random_seed)
    # convert to records for mapping
    rows = df.to_dict(orient="records")
    process_func = partial(
        process_row,
        vqa_evaluator=vqa_evaluator,
        aesthetic_evaluator=aesthetic_evaluator,
        rng=rng,
    )
    # process in parallel (or serial if max_workers=1)
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for res in tqdm(executor.map(process_func, rows), total=len(rows)):
            results.append(res)
    # unpack
    scores, vqa_scores, aest_scores, ocr_scores, batched_probs = zip(*results)
    df["score"] = scores
    df["vqa_score"] = vqa_scores
    df["aest_score"] = aest_scores
    df["ocr_score"] = ocr_scores
    df["batched_choice_probabilities"] = batched_probs
    # save
    out_folder.mkdir(parents=True, exist_ok=True)
    out_path = out_folder / "vtracer_dataset_with_scores.parquet"
    df.to_parquet(out_path)
    print(f"Saved scored dataset to {out_path}")


if __name__ == "__main__":
    generate_scores()
