import torch
import copy
from PIL import Image
from tqdm import tqdm
import numpy as np
from pathlib import Path
import multiprocessing as mp
from functools import partial


def process_image(image, output_path, split, out_name=None):
    from src.score_original import ImageProcessor
    from src.preprocessing import apply_preprocessing_torch

    """Process a single image for dataset preparation"""
    if isinstance(image, (str, Path)):
        out_name = image.stem
        image = Image.open(image).convert("RGB")

    output_image = ImageProcessor(copy.deepcopy(image)).apply().image

    # torch_image = torch.from_numpy(np.array(image)).permute(2, 0, 1)
    # torch_image = torch_image.float().cuda() / 255.0
    # torch_image = apply_preprocessing_torch(torch_image)

    # torch_image = (torch_image * 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    # input_image = Image.fromarray(torch_image)
    input_image = image

    input_image.save(output_path / split / "input" / f"{out_name}.png")
    output_image.save(output_path / split / "output" / f"{out_name}.png")


def generate_random_images(output_path, split, num_samples=100):
    std_value: float = 0.1
    mean_value: float = 0.5
    low: float = 0.0
    high: float = 1.0

    for idx in tqdm(range(num_samples), desc=f"Generating random images for {split}"):
        random_image = np.random.randn(224, 224, 3)
        random_image = random_image * std_value
        random_image = random_image + mean_value
        random_image = np.clip(random_image, 0.0, 1.0)
        random_image = (random_image * 255).astype(np.uint8)
        random_image = Image.fromarray(random_image).convert("RGB")
        process_image(random_image, output_path, split, f"random_{idx}")


def prepare_datasets(
    data_path: str | Path = "/home/mpf/code/yolov9/coco/images/val2017",
    max_samples: int = 1000,
    val_split: float = 0.05,
    output_path: str | Path = "./segmentation_data",
    num_workers: int = 24
):
    dataset_path = Path(data_path).resolve()
    output_path = Path(output_path).resolve()

    if output_path.exists():
        return

    # Create directories
    (output_path / "train" / "input").mkdir(parents=True, exist_ok=True)
    (output_path / "train" / "output").mkdir(parents=True, exist_ok=True)
    (output_path / "val" / "input").mkdir(parents=True, exist_ok=True)
    (output_path / "val" / "output").mkdir(parents=True, exist_ok=True)

    images_list = sorted(list(dataset_path.glob("*.jpg")))[:max_samples]
    np.random.shuffle(images_list)

    val_size = int(len(images_list) * val_split)
    
    # Prepare arguments for multiprocessing
    val_images = images_list[:val_size]
    train_images = images_list[val_size:]
    
    # Create process pool
    with mp.Pool(num_workers) as pool:
        # Process validation images
        val_process_fn = partial(process_image, output_path=output_path, split="val")
        list(tqdm(pool.imap(val_process_fn, val_images), total=len(val_images), desc="Processing validation images"))
        
        # Process training images
        train_process_fn = partial(process_image, output_path=output_path, split="train")
        list(tqdm(pool.imap(train_process_fn, train_images), total=len(train_images), desc="Processing training images"))

    generate_random_images(output_path, "train", num_samples=100)
    generate_random_images(output_path, "val", num_samples=100)

