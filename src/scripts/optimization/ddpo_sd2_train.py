# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
python examples/scripts/ddpo.py \
    --num_epochs=200 \
    --train_gradient_accumulation_steps=1 \
    --sample_num_steps=50 \
    --sample_batch_size=6 \
    --train_batch_size=3 \
    --sample_num_batches_per_epoch=4 \
    --per_prompt_stat_tracking=True \
    --per_prompt_stat_tracking_buffer_size=32 \
    --tracker_project_name="stable_diffusion_training" \
    --log_with="wandb"
"""

import os
from dataclasses import dataclass, field

import numpy as np
import ast
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from transformers import CLIPModel, CLIPProcessor, HfArgumentParser, is_torch_npu_available, is_torch_xpu_available

from trl import DDPOConfig, DDPOTrainer, DefaultDDPOStableDiffusionPipeline
from src.score_original import VQAEvaluator, svg_to_png
import pandas as pd
import numpy as np
from PIL import Image

import vtracer
import tempfile
import os

from picosvg.svg import SVG
import scour.scour
from skimage.metrics import structural_similarity as ssim



@dataclass
class ScriptArguments:
    pretrained_model: str = field(
        # default="runwayml/stable-diffusion-v1-5", metadata={"help": "Pretrained model to use."}
        default="stabilityai/stable-diffusion-2-1", metadata={"help": "Pretrained model to use."}
    )
    pretrained_revision: str = field(default="main", metadata={"help": "Pretrained model revision to use."})
    hf_hub_model_id: str = field(
        default="ddpo-finetuned-stable-diffusion", metadata={"help": "HuggingFace repo to save model weights to."}
    )
    hf_hub_aesthetic_model_id: str = field(
        default="trl-lib/ddpo-aesthetic-predictor",
        metadata={"help": "Hugging Face model ID for aesthetic scorer model weights."},
    )
    hf_hub_aesthetic_model_filename: str = field(
        default="aesthetic-model.pth",
        metadata={"help": "Hugging Face model filename for aesthetic scorer model weights."},
    )
    use_lora: bool = field(default=True, metadata={"help": "Whether to use LoRA."})


def optimize_svg_picosvg(svg):
    svg_instance = SVG.fromstring(svg)
    svg_instance.topicosvg(inplace=True)
    return svg_instance.tostring(pretty_print=False)


def optimize_svg(svg):
    options = scour.scour.parse_args([
        '--enable-viewboxing',
        '--enable-id-stripping',
        '--enable-comment-stripping',
        '--shorten-ids',
        '--indent=none',
        '--strip-xml-prolog',
        '--remove-metadata',
        '--remove-descriptive-elements',
        '--disable-embed-rasters',
        '--enable-viewboxing',
        '--create-groups',
        '--renderer-workaround',
        '--set-precision=2',
    ])

    svg = scour.scour.scourString(svg, options)
    
    svg = svg.replace('id=""', '')
    svg = svg.replace('version="1.0"', '')
    svg = svg.replace('version="1.1"', '')
    svg = svg.replace('version="2.0"', '')
    svg = svg.replace('  ', ' ')
    svg = svg.replace('>\n', '>')
    
    return svg


def svg_conversion(img, image_size=(384,384)):
    tmp_dir = tempfile.TemporaryDirectory()
    # Open the image, resize it, and save it to the temporary directory
    resized_img = img.resize(image_size)
    tmp_file_path = os.path.join(tmp_dir.name, "tmp.png")
    resized_img = resized_img.convert("RGB")
    resized_img.save(tmp_file_path)
    
    svg_path = os.path.join(tmp_dir.name, "gen_svg.svg")
    vtracer.convert_image_to_svg_py(
                tmp_file_path,
                svg_path,
                colormode="color",  # ["color"] or "binary"
                # hierarchical="cutout",  # ["stacked"] or "cutout"
                hierarchical="stacked",  # ["stacked"] or "cutout"
                mode="polygon",  # ["spline"] "polygon", or "none"
                filter_speckle=4,  # default: 4
                color_precision=6,  # default: 6
                layer_difference=16,  # default: 16
                corner_threshold=60,  # default: 60
                length_threshold=4.0,  # in [3.5, 10] default: 4.0
                max_iterations=10,  # default: 10
                splice_threshold=45,  # default: 45
                path_precision=8,  # default: 8
            )

    with open(svg_path, 'r', encoding='utf-8') as f:
        svg = f.read()
    
    svg = optimize_svg_picosvg(svg)
    
    return svg


def get_data_df():
    train_categories = [
        "landscape",
        "fashion",
        "food",
        "activity",
        "architecture",
        "tools",
        "animals",
        "plants",
        "vehicles",
        "toys",
        # "signs",
        # "symbols",
        # "nature",
        # "geometry",
        # "color",
        # "industry",
        # "electronics",
        # "planes",
        # "rockets",
        # "emojis",
    ]
    
    df = pd.read_parquet("/home/mpf/code/kaggle/draw/data/questions_groq_llama4_maverick.parquet")
    df = df[df["category"].isin(train_categories)]

    for key in ["question", "choices", "answer"]:
        df[key] = df[key].apply(ast.literal_eval)

    return df



def get_score_fn():
    # evaluator = VQAEvaluator()
    df = get_data_df()
    
    def _fn(images, prompts, metadata):
        scores = []
        for image, prompt in zip(images, prompts):
            org_img = Image.fromarray((image * 255).round().clamp(0, 255).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8))
            svg = svg_conversion(org_img, (192, 192))
            svg = optimize_svg(svg)
            img = svg_to_png(svg)
            
            org_img = org_img.resize(img.size)
            org_img_np = np.asarray(org_img)
            img_np = np.asarray(img)
            
            
            # row = df[df["description"] == prompt].iloc[0]
            # score = evaluator.score(row["question"], row["choices"], row["answer"], img)
            
            score = ssim(org_img_np, img_np, channel_axis=-1, data_range=255)

            # diff = np.linalg.norm(org_img_np - img_np, axis=-1)
            # score = -np.mean(diff)
            
            scores.append(score)

        return torch.tensor(scores, device=images.device, dtype=torch.float32), {}

    return _fn


def get_prompt_fn():
    df = get_data_df()

    def _fn():
        description = np.random.choice(df["description"])
        prompt = "Vector style image of " + description + " flat design, pastel colors, and simple shapes."
        return prompt, {}


    return _fn


def image_outputs_logger(image_data, global_step, accelerate_logger):
    # For the sake of this example, we will only log the last batch of images
    # and associated data
    result = {}
    images, prompts, _, rewards, _ = image_data[-1]

    for i, image in enumerate(images):
        prompt = prompts[i]
        reward = rewards[i].item()
        result[f"{prompt:.25} | {reward:.2f}"] = image.unsqueeze(0).float()

    accelerate_logger.log_images(
        result,
        step=global_step,
    )


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, DDPOConfig))
    script_args, training_args = parser.parse_args_into_dataclasses()
    training_args.project_kwargs = {
        "logging_dir": "./logs",
        "automatic_checkpoint_naming": True,
        "total_limit": 5,
        "project_dir": "./save",
    }

    pipeline = DefaultDDPOStableDiffusionPipeline(
        script_args.pretrained_model,
        pretrained_model_revision=script_args.pretrained_revision,
        use_lora=script_args.use_lora,
    )

    trainer = DDPOTrainer(
        training_args,
        get_score_fn(),
        get_prompt_fn(),
        pipeline,
        image_samples_hook=image_outputs_logger,
    )

    trainer.train()

    trainer.save_model(training_args.output_dir)


# python src/scripts/ddpo.py --num_epochs=200 --train_gradient_accumulation_steps=1 --sample_num_steps=50 --sample_batch_size=6 --train_batch_size=3 --sample_num_batches_per_epoch=4 --per_prompt_stat_tracking=True --per_prompt_stat_tracking_buffer_size=32 --tracker_project_name="stable_diffusion_training" --log_with="wandb"