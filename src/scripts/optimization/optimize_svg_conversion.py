import argparse
import io
import json
import importlib.util
from functools import partial
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from PIL import Image
from tqdm import tqdm
import cairosvg
import wandb
import os
from concurrent.futures import ProcessPoolExecutor

from src.scripts.data.generate_svgs import VTracerConfig, svg_conversion, clamp_svg

USE_SSIM = importlib.util.find_spec("skimage.metrics") is not None
if USE_SSIM:
    from skimage.metrics import structural_similarity as ssim

# Number of example images (orig, recon, diff) to log to wandb for each trial
NUM_DISPLAY_IMAGES = 20


def _to_array(img: Image.Image) -> np.ndarray:
    return np.asarray(img, dtype=np.float32)


def _image_similarity(img_a: Image.Image, img_b: Image.Image) -> float:
    arr_a = _to_array(img_a)
    arr_b = _to_array(img_b.resize(img_a.size)) if arr_a.shape != _to_array(img_b).shape else _to_array(img_b)
    if USE_SSIM:
        if arr_a.ndim == 3 and arr_a.shape[2] == 3:
            arr_a = np.dot(arr_a[..., :3], [0.299, 0.587, 0.114])
            arr_b = np.dot(arr_b[..., :3], [0.299, 0.587, 0.114])
        similarity, _ = ssim(arr_a, arr_b, full=True, data_range=255)
        return float(similarity)
    mse = np.mean((arr_a - arr_b) ** 2)
    return 1.0 - min(mse / (255.0 ** 2), 1.0)


def svg_to_png(svg_str: str, size: tuple[int, int]) -> Image.Image:
    png_bytes = cairosvg.svg2png(bytestring=svg_str.encode(), output_width=size[0], output_height=size[1])
    return Image.open(io.BytesIO(png_bytes)).convert("RGB")


def build_config_from_trial(trial: optuna.trial.Trial) -> VTracerConfig:
    return VTracerConfig(
        # image_size=trial.suggest_categorical("image_size", [(96, 96), (128, 128), (192, 192), (256, 256)]),
        image_size=(192, 192),
        mode="polygon",
        hierarchical="stacked",
        colormode="color",
        filter_speckle=trial.suggest_int("filter_speckle", 2, 20),
        color_precision=trial.suggest_int("color_precision", 4, 8),
        layer_difference=trial.suggest_int("layer_difference", 4, 100),
        corner_threshold=trial.suggest_int("corner_threshold", 10, 120),
        length_threshold=trial.suggest_float("length_threshold", 3.0, 10.0),
        max_iterations=10,
        splice_threshold=trial.suggest_int("splice_threshold", 10, 90),
        path_precision=trial.suggest_int("path_precision", 6, 12),
        simplify_poligons=trial.suggest_categorical("simplify_poligons", [True, False]),
        preserve_topology=True,
        tolerance=trial.suggest_float("tolerance", 0.5, 10.0),
    )


def _compute_similarity(path: str, cfg: VTracerConfig) -> float:
    img = Image.open(path).convert("RGB")
    svg = svg_conversion(img, cfg)
    svg = clamp_svg(svg, cfg)
    recon = svg_to_png(svg, size=cfg.image_size)
    return _image_similarity(img, recon.resize(img.size).convert("RGB"))


def _diff_image(img_a: Image.Image, img_b: Image.Image) -> Image.Image:
    arr_a = _to_array(img_a)
    arr_b = _to_array(img_b.resize(img_a.size)) if arr_a.shape != _to_array(img_b).shape else _to_array(img_b)
    diff = arr_a - arr_b  # range roughly [-255, 255]
    diff_mean = diff.mean(axis=-1)  # shape (H, W)
    # Min-max normalize to [0, 255]
    min_val, max_val = diff_mean.min(), diff_mean.max()
    if max_val - min_val < 1e-8:
        norm = np.zeros_like(diff_mean, dtype=np.uint8)
    else:
        norm = ((diff_mean - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    return Image.fromarray(norm, mode="L")


def _generate_log_images(path: str, idx: int, config: VTracerConfig, trial_number: int) -> list[wandb.Image]:
    img = Image.open(path).convert("RGB")
    svg = clamp_svg(svg_conversion(img, config), config)
    recon = svg_to_png(svg, size=config.image_size)

    orig = img.resize(config.image_size)
    diff = _diff_image(orig, recon)

    return [
        wandb.Image(orig, caption=f"orig_{trial_number}_{idx}"),
        wandb.Image(recon, caption=f"recon_{trial_number}_{idx}"),
        wandb.Image(diff, caption=f"diff_{trial_number}_{idx}")
    ]


def objective(paths: list[str], run, trial: optuna.trial.Trial) -> float:
    config = build_config_from_trial(trial)

    sims: list[float] = []
    compute_similarity_func = partial(_compute_similarity, cfg=config)
    with ProcessPoolExecutor(max_workers=16) as executor:
        for i, sim in enumerate(executor.map(compute_similarity_func, paths), 1):
            sims.append(sim)
            if i % 5 == 0:
                trial.report(float(np.mean(sims)), i)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

    first_n = min(NUM_DISPLAY_IMAGES, len(paths))
    log_images: list[wandb.Image] = []

    if first_n > 0:
        generate_images_func = partial(_generate_log_images, config=config, trial_number=trial.number)
        with ProcessPoolExecutor(max_workers=16) as executor:
            results = executor.map(generate_images_func, paths[:first_n], range(first_n))
            for result_list in results:
                log_images.extend(result_list)

    score = float(np.mean(sims))
    metrics = {"score": score, **{f"param/{k}": v for k, v in trial.params.items()}}
    run.log(metrics, step=trial.number)
    if log_images:
        run.log({f"images/trial_{trial.number}": log_images}, step=trial.number)
    return score


def load_image_paths(dataset: Path, sample_size: int | None = 30) -> list[str]:
    df = pd.read_parquet(dataset)
    if sample_size is not None and sample_size < len(df):
        df = df.sample(sample_size, random_state=42)
    return df["image_path"].tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("/home/mpf/code/kaggle/draw/data/sdxl/sdxl_dataset.parquet"))
    parser.add_argument("--output", type=Path, default=Path("/home/mpf/code/kaggle/draw/data/vtracer/best_vtracer_config.json"))
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--sample-size", type=int, default=500)
    args = parser.parse_args()

    run = wandb.init(project="svg_optimization", config={"dataset": str(args.dataset), "sample_size": args.sample_size, "trials": args.trials})
    paths = load_image_paths(args.dataset, args.sample_size)
    obj = partial(objective, paths, run)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(obj, n_trials=args.trials, show_progress_bar=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"score": study.best_value, "params": study.best_trial.params}, f, indent=2)

    run.summary["best_score"] = study.best_value
    run.summary["best_params"] = study.best_trial.params
    run.finish()

    print("Best similarity:", study.best_value)
    print("Best parameters saved to", args.output)


if __name__ == "__main__":
    main() 


# # 5250
# wandb: Run summary:
# wandb:              best_score 0.67557
# wandb:   param/color_precision 5
# wandb:  param/corner_threshold 16
# wandb:    param/filter_speckle 4
# wandb:  param/layer_difference 78
# wandb:  param/length_threshold 7.46592
# wandb:    param/path_precision 7
# wandb: param/simplify_poligons False
# wandb:  param/splice_threshold 87
# wandb:         param/tolerance 7.16844
# wandb:                   score 0.67519

# # 9500
# wandb: Run summary:
# wandb:              best_score 0.70443
# wandb:   param/color_precision 5
# wandb:  param/corner_threshold 113
# wandb:    param/filter_speckle 7
# wandb:  param/layer_difference 62
# wandb:  param/length_threshold 9.53395
# wandb:    param/path_precision 8
# wandb: param/simplify_poligons False
# wandb:  param/splice_threshold 50
# wandb:         param/tolerance 5.92073
# wandb:                   score 0.6992