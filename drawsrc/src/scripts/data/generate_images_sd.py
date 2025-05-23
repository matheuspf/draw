import kagglehub
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import json

from diffusers import StableDiffusionPipeline, AutoPipelineForText2Image
from pydantic import BaseModel

from PIL import Image
import torch


DEVICE_0 = "cuda:0"


class SDConfig(BaseModel):
    width: int = 384
    height: int = 384

    prompt_prefixes: list[str] = [
        "Vector style image of",
        "Flat icon illustration of",
        "Vector art depiction of",
        "SVG style graphic of",
        # "Geometric vector design of",
        #"Clean vector outline of"
    ]
    prompt_suffixes: list[str] = [
        "flat design, pastel colors, and simple shapes",
        "minimalist design, solid colors, clean edges",
        "graphic illustration, bold lines, vibrant flat colors",
        "simple vector shapes, limited color palette, 2D",
        # "high contrast colors, simplified details",
        #"bold shapes, primary colors, clear hierarchy"
    ]
    negative_prompts: list[str] = [
        "other colors, detailed",
        "other colors, photorealistic, texture, shadows, gradients",
        "blurry, noisy, complex patterns, 3D rendering",
        "realistic, detailed background, rasterized, photographic",
        # "detailed shading, reflections, dimensional effects, realism"
        #"other colors, textured, ornate details, busy composition, hand-drawn look"
    ]
    
    num_inference_steps: int = 4
    guidance_scale: float = 0.0



def load_sd_model(device=DEVICE_0):
    sdxl_path = kagglehub.dataset_download('tomirol/sdxlturbo')
    pipe = AutoPipelineForText2Image.from_pretrained(sdxl_path, torch_dtype=torch.float16, variant="fp16")
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def generate_sd_image(pipe: StableDiffusionPipeline, prompt: str, config: SDConfig = SDConfig(), num_images_per_prompt: int = 1):
    images = []
    prompts = []
    for idx, (prompt_prefix, prompt_suffix, negative_prompt) in enumerate(zip(config.prompt_prefixes, config.prompt_suffixes, config.negative_prompts)):
        generator = torch.Generator(device=DEVICE_0).manual_seed(idx)
        prompt = f'{prompt_prefix} {prompt} {prompt_suffix}'
        prompts += [prompt] * num_images_per_prompt
        images += pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
            generator=generator,
            num_images_per_prompt=num_images_per_prompt,
        ).images

    return images, prompts



def generate_sd_images(
    descriptions_path = Path("/home/mpf/code/kaggle/draw/data/descriptions_gemini_2.json"),
    out_folder = Path("/home/mpf/code/kaggle/draw/data/sdxl"),
    num_images_per_prompt = 10
):
    imgs_folder = out_folder / "images"
    imgs_folder.mkdir(parents=True, exist_ok=True)
    
    descriptions_path = Path("/home/mpf/code/kaggle/draw/data/descriptions_gemini_2.json")
    out_folder = Path("/home/mpf/code/kaggle/draw/data/sdxl")
    imgs_folder = out_folder / "images"
    imgs_folder.mkdir(parents=True, exist_ok=True)
    
    with open(descriptions_path, 'r') as f:
        descriptions = json.load(f)

    pipe = load_sd_model()
    dataset = {
        "category": [],
        "description": [],
        "prompt": [],
        "image_name": [],
        "prompt_name": [],
        "image_path": [],
        "sd_seed": []
    }

    config = SDConfig()

    for category, descriptions in tqdm(descriptions.items(), total=len(descriptions)):
        for description in tqdm(descriptions, total=len(descriptions)):
            for prompt_idx, (prompt_prefix, prompt_suffix, negative_prompt) in enumerate(zip(config.prompt_prefixes, config.prompt_suffixes, config.negative_prompts)):
                generator = torch.Generator(device=DEVICE_0).manual_seed(prompt_idx)
                prompt = f'{prompt_prefix} {description} {prompt_suffix}'
                images = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=config.num_inference_steps,
                    guidance_scale=config.guidance_scale,
                    generator=generator,
                    num_images_per_prompt=num_images_per_prompt,
                ).images

                img_name = description.replace(" ", "_").replace(",", "_")[:50]
                prompt_name = (prompt_prefix + "_" + prompt_suffix).replace(" ", "_").replace(",", "_")[:50]

                for seed_idx, image in enumerate(images):
                    img_out_p = imgs_folder / category / img_name / prompt_name
                    img_out_p.mkdir(parents=True, exist_ok=True)
                    
                    img_out_path = str(img_out_p / f"{seed_idx}.png")
                    image.save(img_out_path)

                    dataset["category"].append(category)
                    dataset["description"].append(description)
                    dataset["prompt"].append(prompt)
                    dataset["image_name"].append(img_name)
                    dataset["prompt_name"].append(prompt_name)
                    dataset["sd_seed"].append(seed_idx)
                    dataset["image_path"].append(img_out_path)
            
        pd.DataFrame(dataset).to_parquet(out_folder / "sdxl_dataset.parquet")

if __name__ == "__main__":
    generate_sd_images()
