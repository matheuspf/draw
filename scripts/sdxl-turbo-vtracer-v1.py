#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install svgwrite svgpathtools
# !pip install cairosvg bitsandbytes
# !pip install git+https://github.com/openai/CLIP.git
# !pip install -q opencv-python scikit-image pillow
# !pip install scour cssutils
# !pip install vtracer picosvg
# !pip install transformers==4.51.0
# !pip install vllm==0.7.1


# In[2]:


#| default_exp core


# In[3]:


#| export

import kagglehub
import shutil
import os
import sys
from pathlib import Path
import subprocess

# diffvg_path = kagglehub.dataset_download('tomirol/diffvg')
# out_path = Path("/tmp/diffvg").resolve()

# if not out_path.exists():
#     shutil.copytree(Path(diffvg_path) / "diffvg", str(out_path))
#     output = subprocess.check_output(f"pip uninstall tensorflow -y && cd {str(out_path)} && python setup.py install", shell=True, text=True)

# sys.path.append(str(out_path / "dist/diffvg-0.0.1-py3.10-linux-x86_64.egg"))


# In[4]:


#| export

draw_src_path = kagglehub.dataset_download('tomirol/drawsrc')

out_path = Path("/tmp/drawsrc")

if not out_path.exists():
    shutil.copytree(str(draw_src_path), out_path)

sys.path.append(str(out_path))


# In[5]:


#| export

primitive_path = kagglehub.dataset_download('tomirol/primitive')

out_path = Path("/tmp/primitive")

if not out_path.exists():
    shutil.copy(Path(primitive_path) / "primitive", out_path)
    subprocess.check_output(f"chmod +x {out_path}", shell=True, text=True)


# In[6]:


#| export

import sys
import site
from importlib import invalidate_caches

# Refresh sys.path
site.main()
invalidate_caches()


# In[7]:


#| export

# import os
# os.environ["TORCH_COMPILE_DISABLE"] = "1"

from pydantic import BaseModel
from PIL import Image
import re
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
import kagglehub
import subprocess
import scour.scour
import time


import os
import kornia
import torch.nn as nn
import copy
import random
import pandas as pd
import cv2
import json
import cairosvg
from pathlib import Path
import io
from torch.nn import functional as F
import numpy as np
import torch
from tqdm.auto import tqdm
import pydiffvg


# from src.score_original import VQAEvaluator, ImageProcessor, AestheticEvaluator
from src.score_gradient import (
    score_original,
    score_gradient,
    aesthetic_score_original,
    vqa_score_original,
    aesthetic_score_gradient,
    vqa_score_gradient,
)
from src.preprocessing import apply_preprocessing_torch

from src.text_to_svg import text_to_svg


device_0 = f"cuda:{max(0, torch.cuda.device_count() - 2)}"
device_1 = f"cuda:{max(0, torch.cuda.device_count() - 1)}"


# In[8]:


#| export

import pandas as pd
import numpy as np
import kagglehub
import json
import random
import torch
import vllm


# def get_model_question_gen(model_name: str = "qwen-lm/qwen2.5/transformers/7b-instruct-awq/1", device="cuda:1"):
def get_model_question_gen(model_name: str = "qwen-lm/qwen2.5/transformers/7b-instruct-gptq-int4/1", device="cuda:1"):
    model_path = kagglehub.model_download(model_name)
    model = vllm.LLM(
        model_path,
        quantization="awq" if "awq" in model_name else "gptq",
        tensor_parallel_size=1, 
        gpu_memory_utilization=0.5, 
        trust_remote_code=True,
        dtype=torch.float16, 
        enforce_eager=True,
        max_model_len=1280,
        device=device
    )
    tokenizer = model.get_tokenizer()
    return model, tokenizer


def generate_text_question_gen(
    model,
    tokenizer,
    prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    max_new_tokens: int = 2560,
    temperature: float = 0.7,
    top_p: float = 0.9,
    seed: int = 42
):
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
    outputs = model.chat(
        messages,
        vllm.SamplingParams(
            n=1,
            top_p=top_p,
            temperature=temperature,
            seed=seed,
            skip_special_tokens=False,
            max_tokens=max_new_tokens,
        ),
        use_tqdm=False
    )
    response = outputs[0].outputs[0].text

    return response


def load_data_drawing_with_llms():
    # Load descriptions and questions
    # drawing_with_llms_path = kagglehub.competition_download('drawing-with-llms')
    drawing_with_llms_path = kagglehub.dataset_download('tomirol/drawcomp')

    descriptions_df = pd.read_csv(f'{drawing_with_llms_path}/train.csv')
    questions_df = pd.read_parquet(f'{drawing_with_llms_path}/questions.parquet')
    
    # Create a dictionary with all questions for each ID
    questions_by_id = {}
    for _, row in questions_df.iterrows():
        if row['id'] not in questions_by_id:
            questions_by_id[row['id']] = []
            
        # Convert choices string to actual list
        # choices = row['choices'].strip("[]").replace("'", "").split()
        choices = row["choices"].tolist()
        
        questions_by_id[row['id']].append({
            'question': row['question'],
            'choices': choices,
            'answer': row['answer']
        })
    
    # Combine descriptions with their questions
    combined_data = []
    for _, row in descriptions_df.iterrows():
        item_id = row['id']
        if item_id in questions_by_id:
            combined_data.append({
                'id': item_id,
                'description': row['description'],
                'questions': questions_by_id[item_id]
            })
    
    return combined_data

def create_few_shot_prompt(examples, test_description):
    prompt = \
"""Generate questions, multiple-choice options, and answers based on a description.
Format your response as JSON. At least two questions should be multiple-choice questions, instead of yes/no.

Note how all questions are related to the image - the user will only see the image.

Generate a total of 4 distinct questions for a given description.

Examples:

"""
    for example in examples:
        prompt += f"Description: {example['description']}\n"
        prompt += f"Questions: {json.dumps(example['questions'], indent=2)}\n\n"
    
    prompt += f"Now generate questions for this description:\n"
    prompt += f"Description: {test_description}\n"
    prompt += "Questions:"
    
    return prompt

def parse_generated_questions(generated_text):
    """Parse the generated text to extract questions, choices, and answers"""
    try:
        # Try to parse the response as JSON
        start_idx = generated_text.find('[')
        end_idx = generated_text.rfind(']') + 1
        
        if start_idx != -1 and end_idx != -1:
            json_str = generated_text[start_idx:end_idx]
            return json.loads(json_str)
        return []
    except Exception as e:
        print(f"Error parsing generated questions: {e}")
        return []
 
 
class QuestionGenerator:
    def __init__(self, device="cuda:1"):
        self.model, self.tokenizer = get_model_question_gen(device=device)
        self.system_prompt = "You are an AI assistant that generates multiple-choice questions for descriptions. Always respond with valid JSON."
        self.data = load_data_drawing_with_llms()

        few_shot_count = 3
        # np.random.seed(42)
        # np.random.shuffle(self.data)
        self.few_shot_examples = self.data[:few_shot_count]

    def __call__(self, prompt: str) -> str:
        full_prompt = create_few_shot_prompt(self.few_shot_examples, prompt)
        response = generate_text_question_gen(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=full_prompt,
            system_prompt=self.system_prompt
        )
        # generated_questions = parse_generated_questions(response.strip())
        
        return response


# In[9]:


#| export


class BaseConfig(BaseModel):
    width: int = 384
    height: int = 384

class SDConfig(BaseConfig):
    stable_diffusion_path: str = kagglehub.model_download("stabilityai/stable-diffusion-v2/pytorch/1/1")
    
    # prompt_prefixes: list[str] = ["Simple, classic image of"]
    # prompt_suffixes: list[str] = ["with flat color blocks, beautiful, minimal details, solid colors only"]
    # negative_prompts: list[str] = ["lines, framing, hatching, background, textures, patterns, details, outlines"]

    # prompt_prefixes: list[str] = ["Simple, classic image of"]
    # prompt_suffixes: list[str] = ["beautiful, minimal details, avoid other colors"]
    # negative_prompts: list[str] = ["other colors, detailed"]

    # prompt_prefixes: list[str] = ["Image of"]
    # prompt_suffixes: list[str] = ["(((lineart))),((low detail)),(simple),high contrast,sharp,2 bit, (((clean ink anime illustration))),Studio Ghibli,Makoto Shinkai,Hayao Miyazaki,Audrey Kawasaki"]
    # negative_prompts: list[str] = ["(((text))),((color)),(shading),background,noise,dithering,gradient,detailed,out of frame,ugly,error,Illustration, watermark"]

    
    # prompt_prefixes: list[str] = ["Vector style image of"]
    # prompt_suffixes: list[str] = ["flat design, pastel colors, and simple shapes"]
    # negative_prompts: list[str] = ["other colors, detailed"]


    prompt_prefixes: list[str] = [
        "Vector style image of",
        "Flat icon illustration of",
        "Vector art depiction of",
        "SVG style graphic of",
        "Geometric vector design of",
        "Clean vector outline of"
    ]
    prompt_suffixes: list[str] = [
        "flat design, pastel colors, and simple shapes",
        "minimalist design, solid colors, clean edges",
        "graphic illustration, bold lines, vibrant flat colors",
        "simple vector shapes, limited color palette, 2D",
        "high contrast colors, simplified details",
        "bold shapes, primary colors, clear hierarchy"
    ]
    negative_prompts: list[str] = [
        "other colors, detailed",
        "other colors, photorealistic, texture, shadows, gradients",
        "blurry, noisy, complex patterns, 3D rendering",
        "realistic, detailed background, rasterized, photographic",
        "detailed shading, reflections, dimensional effects, realism"
        "other colors, textured, ornate details, busy composition, hand-drawn look"
    ]



    # num_inference_steps: int = 50
    # guidance_scale: int = 20

    num_inference_steps: int = 4
    guidance_scale: int = 0


class TextConfig(BaseConfig):
    x_position_frac: float = 0.9
    y_position_frac: float = 0.9
    font_size: int = 45
    color: tuple[int, int, int] = (255, 255, 255)
    font_path: str = ""


class PrimitiveConfig(BaseConfig):
    mode: int = 8
    num_shapes: int = 100


class DiffvgConfig(BaseConfig):
    num_iterations: int = 40
    validation_steps: int = 20
    # base_svg_path: str = str(Path(kagglehub.dataset_download('tomirol/aestsvg')) / "output_aest_650.svg")
    # base_svg_path: str = str(Path(kagglehub.dataset_download('tomirol/aestsvg')) / "output_vtracer_0.670.svg")
    base_svg_path: str = "/home/mpf/code/kaggle/draw/output_vtracer_96_bottom_0.659.svg"



# In[10]:


#| export

def generate_primitive_svg(
    image: Image.Image,
    size: int = 384,
    mode: int = 8,
    num_shapes: int = 100,
    temp_path: str = "/tmp/image.png",
    max_size: int = 5500
) -> str:
    temp_path_svg = temp_path.replace(".png", ".svg")
    image = image.resize((size, size))

    image.save(temp_path)

    args = [
        "/tmp/primitive",
        "-i",
        temp_path,
        "-o",
        temp_path_svg,
        "-n",
        str(num_shapes),
        "-m",
        str(mode),
        "-r",
        f"{size}",
        "-s",
        f"{size}",
    ]
    subprocess.run(args)

    with open(temp_path_svg, "r") as f:
        svg = f.read()
    
    svg = polygon_to_path(svg)
    
    cur_svg = svg
    svg_lines = svg.strip().split("\n")
    keep_idx = [1] * len(svg_lines)
    cur_idx = len(svg_lines) - 1

    while len(optimize_svg(cur_svg).encode('utf-8')) > max_size:
        if "<path" in svg_lines[cur_idx]:
            keep_idx[cur_idx] = 0

        cur_idx -= 1
        cur_svg = "\n".join([line for i, line in enumerate(svg_lines) if keep_idx[i]])
        
    print(f"Avg keep: {np.mean(keep_idx)}")
    print(len(optimize_svg(cur_svg).encode('utf-8')))
    
    svg = cur_svg

    svg += text_to_svg("O", x_position_frac=0.75, y_position_frac=0.85, font_size=60, color=(255, 255, 255), font_path="/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf").split("\n")[1]
    # svg += text_to_svg("O", x_position_frac=0.6, y_position_frac=0.85, font_size=60, color=(255, 255, 255), font_path="/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf").split("\n")[1]
    # svg += text_to_svg("C", x_position_frac=0.75, y_position_frac=0.85, font_size=60, color=(0, 0, 0), font_path="/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf").split("\n")[1]

    svg = svg.replace("</svg>", "") + "</svg>"

    return svg


# In[11]:


#| export

def generate_sd_image(pipe: StableDiffusionPipeline, prompt: str, config: SDConfig, num_images_per_prompt: int = 1) -> list[Image.Image]:
    images = []
    for idx, (prompt_prefix, prompt_suffix, negative_prompt) in enumerate(zip(config.prompt_prefixes, config.prompt_suffixes, config.negative_prompts)):
        generator = torch.Generator(device=device_0).manual_seed(idx)
        images += pipe(
            prompt=f'{prompt_prefix} {prompt} {prompt_suffix}',
            negative_prompt=negative_prompt,
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
            generator=generator,
            num_images_per_prompt=num_images_per_prompt,
        ).images

    return images


# In[12]:


#| export

from picosvg.svg import SVG

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


# In[13]:


#| export

import vtracer
import tempfile
import os

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


# In[14]:


#| export

from svgpathtools import parse_path, Path, Line, CubicBezier, QuadraticBezier, Arc
from shapely.geometry import LineString, Polygon
from shapely.ops import polygonize
from lxml import etree
import re

def path_to_coords(path):
    coords = []
    for seg in path:
        if isinstance(seg, (Line, CubicBezier, QuadraticBezier, Arc)):
            coords.append((seg.start.real, seg.start.imag))
    # Add the last point
    if path:
        coords.append((path[-1].end.real, path[-1].end.imag))
    return coords

def coords_to_path(coords, close=False):
    if not coords:
        return ""
    d = f"M{coords[0][0]} {coords[0][1]}"
    for x, y in coords[1:]:
        d += f" L{x} {y}"
    if close:
        d += " Z"
    return d

def simplify_svg(svg_string, tolerance=1.0, preserve_topology=True):
    # Parse SVG
    root = etree.fromstring(svg_string.encode())
    ns = {'svg': root.nsmap.get(None, '')}
    for elem in root.xpath('//svg:path', namespaces=ns):
        d = elem.attrib['d']
        path = parse_path(d)
        coords = path_to_coords(path)
        if not coords:
            continue
        # Detect if path is closed
        close = (coords[0] == coords[-1])
        # Use Polygon if closed, else LineString
        if close and len(coords) > 3:
            geom = Polygon(coords)
        else:
            geom = LineString(coords)
        simplified = geom.simplify(tolerance, preserve_topology=preserve_topology)
        # Convert back to path
        if simplified.is_empty:
            continue
        if hasattr(simplified, 'exterior'):
            new_coords = list(simplified.exterior.coords)
            new_d = coords_to_path(new_coords, close=True)
        else:
            new_coords = list(simplified.coords)
            new_d = coords_to_path(new_coords, close=False)
        elem.attrib['d'] = new_d
    return etree.tostring(root, pretty_print=True).decode()


# In[15]:


#| export

import svgpathtools
from svgpathtools import svg2paths2, wsvg, Path
import tempfile
import os

def path_area(path: Path) -> float:
    """Calculate the signed area of a closed path (polygon)."""
    area = 0.0
    for seg in path:
        try:
            area += 0.5 * abs(seg.start.real * seg.end.imag - seg.end.real * seg.start.imag)
        except Exception:
            pass
    return area

def remove_smallest_paths_svg(svg: str, min_bytes: int = 10000) -> str:
    # Write SVG to a temp file for svgpathtools
    with tempfile.NamedTemporaryFile(delete=False, suffix=".svg") as tmp:
        tmp.write(svg.encode('utf-8'))
        temp_path = tmp.name

    paths, attributes, svg_attributes = svg2paths2(temp_path)
    if not paths or len(paths) == 1:
        os.remove(temp_path)
        return svg

    # Calculate areas and sort paths by area descending
    indexed = list(enumerate(paths))
    areas = [(i, abs(path_area(p))) for i, p in indexed]
    areas.sort(key=lambda x: x[1], reverse=True)
    # sorted_indices = [i for i, _ in areas]
    sorted_indices = list(range(len(areas)))

    # Binary search for the minimum number of largest paths to keep
    left, right = 1, len(paths)
    best_svg = None

    while left <= right:
        mid = (left + right) // 2
        keep_indices = sorted_indices[:mid]
        keep_paths = [paths[i] for i in keep_indices]
        keep_attributes = [attributes[i] for i in keep_indices]

        # Write new SVG to temp file
        wsvg(keep_paths, attributes=keep_attributes, svg_attributes=svg_attributes, filename=temp_path)
        with open(temp_path, "r") as f:
            candidate_svg = f.read()
        candidate_svg_opt = optimize_svg(candidate_svg)
        # candidate_svg = candidate_svg.replace("<defs/>\n", "")

        if len(candidate_svg_opt.encode('utf-8')) > min_bytes:
            best_svg = candidate_svg
            right = mid - 1
        else:
            left = mid + 1

    os.remove(temp_path)
    return best_svg if best_svg is not None else svg


# In[16]:


#| export

import ast
import io
import math
import statistics
import string

import cairosvg
import clip
import cv2
import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from more_itertools import chunked
from PIL import Image, ImageFilter
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    PaliGemmaForConditionalGeneration,
)

svg_constraints = kagglehub.package_import('metric/svg-constraints', bypass_confirmation=True)


class ParticipantVisibleError(Exception):
    pass


def score(
    solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str, random_seed: int = 0
) -> float:
    """Calculates a fidelity score by comparing generated SVG images to target text descriptions.

    Parameters
    ----------
    solution : pd.DataFrame
        A DataFrame containing target questions, choices, and answers about an SVG image.
    submission : pd.DataFrame
        A DataFrame containing generated SVG strings. Must have a column named 'svg'.
    row_id_column_name : str
        The name of the column containing row identifiers. This column is removed before scoring.
    random_seed : int
        A seed to set the random state.

    Returns
    -------
    float
        The mean fidelity score (a value between 0 and 1) representing the average similarity between the generated SVGs and their descriptions.
        A higher score indicates better fidelity.

    Raises
    ------
    ParticipantVisibleError
        If the 'svg' column in the submission DataFrame is not of string type or if validation of the SVG fails.

    Examples
    --------
    >>> import pandas as pd
    >>> solution = pd.DataFrame({
    ...     'id': ["abcde"],
    ...     'question': ['["Is there a red circle?", "What shape is present?"]'],
    ...     'choices': ['[["yes", "no"], ["square", "circle", "triangle", "hexagon"]]'],
    ...     'answer': ['["yes", "circle"]'],
    ... })
    >>> submission = pd.DataFrame({
    ...     'id': ["abcde"],
    ...     'svg': ['<svg viewBox="0 0 100 100"><circle cx="50" cy="50" r="40" fill="red"/></svg>'],
    ... })
    >>> score(solution, submission, 'row_id', random_seed=42)
    0...
    """
    # Convert solution fields to list dtypes and expand
    for colname in ['question', 'choices', 'answer']:
        solution[colname] = solution[colname].apply(ast.literal_eval)
    solution = solution.explode(['question', 'choices', 'answer'])

    # Validate
    if not pd.api.types.is_string_dtype(submission.loc[:, 'svg']):
        raise ParticipantVisibleError('svg must be a string.')

    # Check that SVG code meets defined constraints
    constraints = svg_constraints.SVGConstraints()
    try:
        for svg in submission.loc[:, 'svg']:
            constraints.validate_svg(svg)
    except:
        with open("svg_error.svg", "w") as f:
            f.write(svg)
        print(len(svg.encode('utf-8')))
        raise ParticipantVisibleError('SVG code violates constraints.')

    # Score
    vqa_evaluator = VQAEvaluator()
    aesthetic_evaluator = AestheticEvaluator(device_0)

    results = []
    all_scores = []
    rng = np.random.RandomState(random_seed)
    try:
        df = solution.merge(submission, on='id')
        pbar = tqdm(list(set(df["id"])))
        for i, (_, group) in enumerate(df.loc[
            :, ['id', 'question', 'choices', 'answer', 'svg']
        ].groupby('id')):
            questions, choices, answers, svg = [
                group[col_name].to_list()
                for col_name in group.drop('id', axis=1).columns
            ]
            svg = svg[0]  # unpack singleton from list
            group_seed = rng.randint(0, np.iinfo(np.int32).max)
            image_processor = ImageProcessor(image=svg_to_png(svg), seed=group_seed).apply()
            image = image_processor.image.copy()
            aesthetic_score = aesthetic_evaluator.score(image)
            vqa_score, batched_choice_probabilities = vqa_evaluator.score(questions, choices, answers, image, n=2)
            image_processor.reset().apply_random_crop_resize().apply_jpeg_compression(quality=90)
            ocr_score = vqa_evaluator.ocr(image_processor.image)
            instance_score = (
                harmonic_mean(vqa_score, aesthetic_score, beta=0.5) * ocr_score
            )
            results.append(instance_score)
            all_scores.append((instance_score, vqa_score, aesthetic_score, ocr_score, batched_choice_probabilities))
            pbar.update(1)

    except:
        raise ParticipantVisibleError('SVG failed to score.')

    fidelity = statistics.mean(results)
    return float(fidelity), all_scores



class VQAEvaluator:
    """Evaluates images based on their similarity to a given text description using multiple choice questions."""

    def __init__(self, model_id: str = 'google/paligemma-2/transformers/paligemma2-10b-mix-448', device=device_1):
        self.device = device
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        self.letters = string.ascii_uppercase
        self.model_path = kagglehub.model_download(model_id)
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.model_path,
            low_cpu_mem_usage=True,
            quantization_config=self.quantization_config,
            device_map=self.device
        ).to(self.device)

    def score(self, questions, choices, answers, image, n=4):
        scores = []
        batched_choice_probabilities = []
        batches = (chunked(qs, n) for qs in [questions, choices, answers])
        for question_batch, choice_batch, answer_batch in zip(*batches, strict=True):
            res = self.score_batch(
                image,
                question_batch,
                choice_batch,
                answer_batch,
            )
            scores.extend(res[0])
            batched_choice_probabilities.extend(res[1])
        return statistics.mean(scores), batched_choice_probabilities

    def get_description(self, image: Image.Image, prefix: str = "<image>cap en\n") -> str:
        inputs = self.processor(
            images=image,
            text=prefix,
            return_tensors="pt",
            # suffix=description,
        ).to(self.device)
        input_len = inputs['input_ids'].shape[-1]

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=128, do_sample=False)

        outputs = outputs[0][input_len:]
        decoded = self.processor.decode(outputs, skip_special_tokens=True)

        return decoded

    def score_batch(
        self,
        image: Image.Image,
        questions: list[str],
        choices_list: list[list[str]],
        answers: list[str],
    ) -> list[float]:
        """Evaluates the image based on multiple choice questions and answers.

        Parameters
        ----------
        image : PIL.Image.Image
            The image to evaluate.
        questions : list[str]
            List of questions about the image.
        choices_list : list[list[str]]
            List of lists of possible answer choices, corresponding to each question.
        answers : list[str]
            List of correct answers from the choices, corresponding to each question.

        Returns
        -------
        list[float]
            List of scores (values between 0 and 1) representing the probability of the correct answer for each question.
        """
        prompts = [
            self.format_prompt(question, choices)
            for question, choices in zip(questions, choices_list, strict=True)
        ]
        batched_choice_probabilities = self.get_choice_probability(
            image, prompts, choices_list
        )

        scores = []
        for i, _ in enumerate(questions):
            choice_probabilities = batched_choice_probabilities[i]
            answer = answers[i]
            answer_probability = 0.0
            for choice, prob in choice_probabilities.items():
                if choice == answer:
                    answer_probability = prob
                    break
            scores.append(answer_probability)

        # pred_description = self.get_description(image)
        batched_choice_probabilities = [{"question": questions[i], "choices": batched_choice_probabilities[i], "answer": answers[i], "pred_description": ""} for i in range(len(questions))]
        return scores, batched_choice_probabilities

    def format_prompt(self, question: str, choices: list[str]) -> str:
        prompt = f'<image>answer en Question: {question}\nChoices:\n'
        for i, choice in enumerate(choices):
            prompt += f'{self.letters[i]}. {choice}\n'
        return prompt

    def mask_choices(self, logits, choices_list):
        """Masks logits for the first token of each choice letter for each question in the batch."""
        batch_size = logits.shape[0]
        masked_logits = torch.full_like(logits, float('-inf'))

        for batch_idx in range(batch_size):
            choices = choices_list[batch_idx]
            for i in range(len(choices)):
                letter_token = self.letters[i]

                first_token = self.processor.tokenizer.encode(
                    letter_token, add_special_tokens=False
                )[0]
                first_token_with_space = self.processor.tokenizer.encode(
                    ' ' + letter_token, add_special_tokens=False
                )[0]

                if isinstance(first_token, int):
                    masked_logits[batch_idx, first_token] = logits[
                        batch_idx, first_token
                    ]
                if isinstance(first_token_with_space, int):
                    masked_logits[batch_idx, first_token_with_space] = logits[
                        batch_idx, first_token_with_space
                    ]

        return masked_logits

    def get_choice_probability(self, image, prompts, choices_list) -> list[dict]:
        inputs = self.processor(
            images=[image] * len(prompts),
            text=prompts,
            return_tensors='pt',
            padding='longest',
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :]  # Logits for the last (predicted) token
            masked_logits = self.mask_choices(logits, choices_list)
            probabilities = torch.softmax(masked_logits, dim=-1)

        batched_choice_probabilities = []
        for batch_idx in range(len(prompts)):
            choice_probabilities = {}
            choices = choices_list[batch_idx]
            for i, choice in enumerate(choices):
                letter_token = self.letters[i]
                first_token = self.processor.tokenizer.encode(
                    letter_token, add_special_tokens=False
                )[0]
                first_token_with_space = self.processor.tokenizer.encode(
                    ' ' + letter_token, add_special_tokens=False
                )[0]

                prob = 0.0
                if isinstance(first_token, int):
                    prob += probabilities[batch_idx, first_token].item()
                if isinstance(first_token_with_space, int):
                    prob += probabilities[batch_idx, first_token_with_space].item()
                choice_probabilities[choice] = prob

            # Renormalize probabilities for each question
            total_prob = sum(choice_probabilities.values())
            if total_prob > 0:
                renormalized_probabilities = {
                    choice: prob / total_prob
                    for choice, prob in choice_probabilities.items()
                }
            else:
                renormalized_probabilities = (
                    choice_probabilities  # Avoid division by zero if total_prob is 0
                )
            batched_choice_probabilities.append(renormalized_probabilities)

        return batched_choice_probabilities

    def ocr(self, image, free_chars=4):
        inputs = (
            self.processor(
                text='<image>ocr\n',
                images=image,
                return_tensors='pt',
            )
            .to(torch.float16)
            .to(self.device)
        )
        input_len = inputs['input_ids'].shape[-1]

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=32, do_sample=False)
            outputs = outputs[0][input_len:]
            decoded = self.processor.decode(outputs, skip_special_tokens=True)

        num_char = len(decoded)

        # Exponentially decreasing towards 0.0 if more than free_chars detected
        return min(1.0, math.exp(-num_char + free_chars))


class AestheticPredictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)


class AestheticEvaluator:
    def __init__(self, device=device_1):
        self.device = device
        # self.model_path = '/kaggle/input/sac-logos-ava1-l14-linearmse/sac+logos+ava1-l14-linearMSE.pth'
        # self.clip_model_path = '/kaggle/input/openai-clip-vit-large-patch14/ViT-L-14.pt'
        self.model_path = str(kagglehub.model_download("jiazhuang/sac-logos-ava1-l14-linearmse/Transformers/default/1", path="sac+logos+ava1-l14-linearMSE.pth"))
        self.clip_model_path = str(kagglehub.model_download("jiazhuang/clip-vit-large-patch14/Transformers/default/1", path="ViT-L-14.pt"))
        self.predictor, self.clip_model, self.preprocessor = self.load()

    def load(self):
        """Loads the aesthetic predictor model and CLIP model."""
        state_dict = torch.load(self.model_path, weights_only=True, map_location=self.device)

        # CLIP embedding dim is 768 for CLIP ViT L 14
        predictor = AestheticPredictor(768)
        predictor.load_state_dict(state_dict)
        predictor.to(self.device)
        predictor.eval()
        clip_model, preprocessor = clip.load(self.clip_model_path, device=self.device)

        return predictor, clip_model, preprocessor

    def score(self, image: Image.Image) -> float:
        """Predicts the CLIP aesthetic score of an image."""
        image = self.preprocessor(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)
            # l2 normalize
            image_features /= image_features.norm(dim=-1, keepdim=True)
            image_features = image_features.cpu().detach().numpy()

        score = self.predictor(torch.from_numpy(image_features).to(self.device).float())

        return score.item() / 10.0  # scale to [0, 1]


def harmonic_mean(a: float, b: float, beta: float = 1.0) -> float:
    """
    Calculate the harmonic mean of two values, weighted using a beta parameter.

    Args:
        a: First value (e.g., precision)
        b: Second value (e.g., recall)
        beta: Weighting parameter

    Returns:
        Weighted harmonic mean
    """
    # Handle zero values to prevent division by zero
    if a <= 0 or b <= 0:
        return 0.0
    return (1 + beta**2) * (a * b) / (beta**2 * a + b)


def svg_to_png(svg_code: str, size: tuple = (384, 384)) -> Image.Image:
    """
    Converts an SVG string to a PNG image using CairoSVG.

    If the SVG does not define a `viewBox`, it will add one using the provided size.

    Parameters
    ----------
    svg_code : str
        The SVG string to convert.
    size : tuple[int, int], default=(384, 384)
        The desired size of the output PNG image (width, height).

    Returns
    -------
    PIL.Image.Image
        The generated PNG image.
    """
    # Ensure SVG has proper size attributes
    if 'viewBox' not in svg_code:
        svg_code = svg_code.replace('<svg', f'<svg viewBox="0 0 {size[0]} {size[1]}"')

    # Convert SVG to PNG
    png_data = cairosvg.svg2png(bytestring=svg_code.encode('utf-8'))
    return Image.open(io.BytesIO(png_data)).convert('RGB').resize(size)


class ImageProcessor:
    def __init__(self, image: Image.Image, seed=None, crop=True):
        """Initialize with either a path to an image or a PIL Image object."""
        self.image = image
        self.original_image = self.image.copy()
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random

    def reset(self):
        self.image = self.original_image.copy()
        return self
    
    def visualize_comparison(
        self,
        original_name='Original',
        processed_name='Processed',
        figsize=(10, 5),
        show=True,
    ):
        """Display original and processed images side by side."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        ax1.imshow(np.asarray(self.original_image))
        ax1.set_title(original_name)
        ax1.axis('off')

        ax2.imshow(np.asarray(self.image))
        ax2.set_title(processed_name)
        ax2.axis('off')

        title = f'{original_name} vs {processed_name}'
        fig.suptitle(title)
        fig.tight_layout()
        if show:
            plt.show()
        return fig

    def apply_median_filter(self, size=3):
        """Apply median filter to remove outlier pixel values.

        Args:
            size: Size of the median filter window.
        """
        self.image = self.image.filter(ImageFilter.MedianFilter(size=size))
        return self

    def apply_bilateral_filter(self, d=9, sigma_color=75, sigma_space=75):
        """Apply bilateral filter to smooth while preserving edges.

        Args:
            d: Diameter of each pixel neighborhood
            sigma_color: Filter sigma in the color space
            sigma_space: Filter sigma in the coordinate space
        """
        # Convert PIL Image to numpy array for OpenCV
        img_array = np.asarray(self.image)

        # Apply bilateral filter
        filtered = cv2.bilateralFilter(img_array, d, sigma_color, sigma_space)

        # Convert back to PIL Image
        self.image = Image.fromarray(filtered)
        return self

    def apply_fft_low_pass(self, cutoff_frequency=0.5):
        """Apply low-pass filter in the frequency domain using FFT.

        Args:
            cutoff_frequency: Normalized cutoff frequency (0-1).
                Lower values remove more high frequencies.
        """
        # Convert to numpy array, ensuring float32 for FFT
        img_array = np.array(self.image, dtype=np.float32)

        # Process each color channel separately
        result = np.zeros_like(img_array)
        for i in range(3):  # For RGB channels
            # Apply FFT
            f = np.fft.fft2(img_array[:, :, i])
            fshift = np.fft.fftshift(f)

            # Create a low-pass filter mask
            rows, cols = img_array[:, :, i].shape
            crow, ccol = rows // 2, cols // 2
            mask = np.zeros((rows, cols), np.float32)
            r = int(min(crow, ccol) * cutoff_frequency)
            center = [crow, ccol]
            x, y = np.ogrid[:rows, :cols]
            mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
            mask[mask_area] = 1

            # Apply mask and inverse FFT
            fshift_filtered = fshift * mask
            f_ishift = np.fft.ifftshift(fshift_filtered)
            img_back = np.fft.ifft2(f_ishift)
            img_back = np.real(img_back)

            result[:, :, i] = img_back

        # Clip to 0-255 range and convert to uint8 after processing all channels
        result = np.clip(result, 0, 255).astype(np.uint8)

        # Convert back to PIL Image
        self.image = Image.fromarray(result)
        return self

    def apply_jpeg_compression(self, quality=85):
        """Apply JPEG compression.

        Args:
            quality: JPEG quality (0-95). Lower values increase compression.
        """
        buffer = io.BytesIO()
        self.image.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        self.image = Image.open(buffer)
        return self

    def apply_random_crop_resize(self, crop_percent=0.05):
        """Randomly crop and resize back to original dimensions.

        Args:
            crop_percent: Percentage of image to crop (0-0.4).
        """
        width, height = self.image.size
        crop_pixels_w = int(width * crop_percent)
        crop_pixels_h = int(height * crop_percent)

        left = self.rng.randint(0, crop_pixels_w + 1)
        top = self.rng.randint(0, crop_pixels_h + 1)
        right = width - self.rng.randint(0, crop_pixels_w + 1)
        bottom = height - self.rng.randint(0, crop_pixels_h + 1)

        self.image = self.image.crop((left, top, right, bottom))
        self.image = self.image.resize((width, height), Image.BILINEAR)
        return self

    def apply(self):
        """Apply an ensemble of defenses."""
        return (
            self.apply_random_crop_resize(crop_percent=0.03)
            .apply_jpeg_compression(quality=95)
            .apply_median_filter(size=9)
            .apply_fft_low_pass(cutoff_frequency=0.5)
            .apply_bilateral_filter(d=5, sigma_color=75, sigma_space=75)
            .apply_jpeg_compression(quality=92)
        )


# In[17]:


#| export

import svgpathtools
from svgpathtools import document, wsvg

def displace_svg_paths(svg, x_offset = 0, y_offset = 0, scale=0.5) -> str:
    temp_path = "/tmp/temp.svg"
    with open(temp_path, "w") as f:
        f.write(svg)
    
    paths, attributes, svg_attributes = svgpathtools.svg2paths2(temp_path)
    displacement = complex(x_offset, y_offset)
    
    for i, path in enumerate(paths):
        paths[i] = path.scaled(scale).translated(displacement)

    wsvg(paths, attributes=attributes, svg_attributes=svg_attributes, filename=temp_path)

    with open(temp_path) as f:
        svg = f.read()
    
    svg = svg.replace("<defs/>\n", "")

    return svg


# In[18]:


#| export

from diffusers import AutoPipelineForText2Image
import torch

def load_sd_model(device=device_0):
    sdxl_path = kagglehub.dataset_download('tomirol/sdxlturbo')
    pipe = AutoPipelineForText2Image.from_pretrained(sdxl_path, torch_dtype=torch.float16, variant="fp16")
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


# In[19]:


#| export

def polygon_to_path(svg: str) -> str:
    svg = re.sub(
        r'<polygon([\w\W]+?)points=(["\'])([\.\d, -]+?)(["\'])', 
        r'<path\1d=\2M\3z\4', 
        svg
    )
    svg = re.sub(
        r'<polyline([\w\W]+?)points=(["\'])([\.\d, -]+?)(["\'])', 
        r'<path\1d=\2M\3\4', 
        svg
    )
    return svg

def merge_svgs(bg_svg: str, aest_svg: str):
    aest_svg = aest_svg.strip().split("\n")[2:-1]
    # aest_svg = aest_svg.strip().split("\n")[2:-2]
    
    # aest_svg = [
    #     '<g clip-path="polygon(32px 32px, 80px 32px, 80px 80px, 32px 80px)">',
    #     # '<svg x="32" y="32" width="48" height="48" viewBox="32 32 48 48" overflow="hidden">',
    #     *aest_svg,
    #     '</g>'
    # ]
    aest_svg = "\n".join(aest_svg)
    
    # bg_svg = polygon_to_path(bg_svg)
    svg = bg_svg + '\n' + aest_svg
    svg = svg.replace("</svg>", "") + "</svg>"

    return svg


def svg_to_png_no_resize(svg_code: str) -> Image.Image:
    png_data = cairosvg.svg2png(bytestring=svg_code.encode('utf-8'))
    img_pil = Image.open(io.BytesIO(png_data)).convert('RGB')
    return img_pil


def apply_random_crop_resize_seed(image: Image.Image, crop_percent=0.05, seed=42):
    rs = np.random.RandomState(seed)
    
    width, height = image.size
    crop_pixels_w = int(width * crop_percent)
    crop_pixels_h = int(height * crop_percent)

    left = rs.randint(0, crop_pixels_w + 1)
    top = rs.randint(0, crop_pixels_h + 1)
    right = width - rs.randint(0, crop_pixels_w + 1)
    bottom = height - rs.randint(0, crop_pixels_h + 1)

    image = image.crop((left, top, right, bottom))
    image = image.resize((width, height), Image.BILINEAR)

    return image


def get_optimization_settings():
    settings = pydiffvg.SvgOptimizationSettings()

    lr = 5e-3

    settings.global_override(["optimizer"], "Adam")
    settings.global_override(["color_lr"], lr)
    settings.global_override(["alpha_lr"], lr)
    settings.global_override(["paths", "shape_lr"], 10*lr)
    settings.global_override(["circles", "shape_lr"], 10*lr)
    settings.global_override(["transforms", "transform_lr"], 10*lr)
    
    settings.global_override(["gradients", "optimize_stops"], True)
    settings.global_override(["gradients", "stop_lr"], lr)
    settings.global_override(["gradients", "optimize_color"], True)
    settings.global_override(["gradients", "color_lr"], lr)
    settings.global_override(["gradients", "optimize_alpha"], True)
    settings.global_override(["gradients", "alpha_lr"], lr)
    settings.global_override(["gradients", "optimize_location"], True)
    settings.global_override(["gradients", "location_lr"], 10*lr)

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


def generate_diffvg_svg(
    bg_svg: str,
    base_svg_path: str,
    evaluator: AestheticEvaluator,
    width: int = 384,
    height: int = 384,
    num_iterations: int = 100,
    validation_steps: int = 10,
    device: str = device_1,
    num_eval: int = 4
) -> Image.Image:
    pydiffvg.set_use_gpu(True)

    evaluator.predictor.to("cuda:1").eval()
    evaluator.clip_model.to("cuda:1").eval()
    evaluator.predictor.requires_grad_(False)
    evaluator.clip_model.requires_grad_(False)

    bg_image = svg_to_png_no_resize(bg_svg)
    bg_image = bg_image.resize((width, height))
    bg_image = np.array(bg_image)
    bg_image = torch.from_numpy(bg_image).to("cuda:1").permute(2, 0, 1).float() / 255.0

    # mask = torch.zeros((3, height, width), dtype=torch.float32, device=device)
    # mask[:, 32:80, 32:80] = 1

    settings = get_optimization_settings()

    text_path_ids = [f"text-path-{i}" for i in range(100)] + [f"background-{i}" for i in range(10)]
    for text_id in text_path_ids:
        text_settings = settings.undefault(text_id)
        text_settings["paths"]["optimize_points"] = False
        text_settings["optimize_color"] = True
        text_settings["optimize_alpha"] = True
        text_settings["optimize_transforms"] = False

    optim_svg = pydiffvg.OptimizableSvg(
        base_svg_path, settings, optimize_background=False, verbose=False, device="cuda:1"
    )

    best_svg = optimize_svg(merge_svgs(bg_svg, optim_svg.write_xml()))
    best_val_loss = -1e8
    
    grad_accumulation_steps = 1

    pbar = tqdm(total=num_iterations)

    for iter_idx in range(num_iterations):
        optim_svg.zero_grad()
        image = optim_svg.render(seed=iter_idx)
        img = image[:, :, :3].permute(2, 0, 1).clamp(0, 1).to("cuda:1")

        # img = img * mask + bg_image * (1 - mask)

        mask = (img == 0).all(dim=0).unsqueeze(0).float()
        bg_image = bg_image.to(mask.device)
        img = (1.0 - mask) * img + mask * bg_image.to(mask.device)
        
        crop_frac = 0.05
        random_size = int(random.uniform(1.0 - crop_frac, 1.0) * image.shape[1])
        img = kornia.augmentation.RandomCrop((random_size, random_size))(img.unsqueeze(0)).squeeze(0)

        img = apply_preprocessing_torch(img)

        loss = aesthetic_score_gradient(evaluator, img).mean()

        if iter_idx == 0 or (iter_idx + 1) % validation_steps == 0:
            torch.cuda.empty_cache()
            
            aest_svg = optim_svg.write_xml()
            cur_svg = merge_svgs(bg_svg, aest_svg)
            cur_svg_opt = optimize_svg(cur_svg)
            pil_image = svg_to_png_no_resize(cur_svg_opt)
            val_loss = 0.0
            
            for eval_idx in range(num_eval):
                pil_image_eval = pil_image.copy()
                pil_image_eval = apply_random_crop_resize_seed(pil_image_eval, crop_percent=0.03, seed=eval_idx)
                # pil_image_eval = ImageProcessor(pil_image_eval).apply().image
                pil_image_eval = (ImageProcessor(pil_image_eval)
                    # .apply_random_crop_resize(crop_percent=0.03)
                    .apply_jpeg_compression(quality=95)
                    .apply_median_filter(size=9)
                    .apply_fft_low_pass(cutoff_frequency=0.5)
                    .apply_bilateral_filter(d=5, sigma_color=75, sigma_space=75)
                    .apply_jpeg_compression(quality=92)
                ).image
                
                val_loss += aesthetic_score_original(evaluator, pil_image_eval)

            val_loss /= num_eval

            # pil_image = Image.fromarray((img_bkp.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)).convert("RGB")

            if val_loss > best_val_loss:
                best_val_loss = val_loss
                best_svg = cur_svg_opt
            
        pbar.set_description(
            f"It {iter_idx}/{num_iterations} | "
            f"Loss: {loss.item():.3f} | "
            f"Val Loss: {val_loss:.3f} | "
        )
        pbar.update(1)

        loss = -loss / grad_accumulation_steps
        loss.backward()
        
        if (iter_idx + 1) % grad_accumulation_steps == 0:
            optim_svg.step()

    print(f"Best loss: {best_val_loss}")

    return best_svg


# In[20]:


#| export

def harmonic_mean(a: float, b: float, beta: float = 1.0) -> float:
    if a <= 0 or b <= 0:
        return 0.0
    return (1 + beta**2) * (a * b) / (beta**2 * a + b)


def to_svg_vtracer(img, sz=192):
    bg_svg = svg_conversion(img, image_size=(sz, sz))
    bg_svg = polygon_to_path(bg_svg)
    bg_svg = displace_svg_paths(bg_svg, x_offset=0, y_offset=0, scale=384 / sz)
    bg_svg = simplify_svg(bg_svg, tolerance=5.0, preserve_topology=True)
    bg_svg = remove_smallest_paths_svg(bg_svg, min_bytes=5500)
    bg_svg += text_to_svg("O", x_position_frac=0.75, y_position_frac=0.85, font_size=60, color=(0, 0, 0), font_path="/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf").split("\n")[1]
    bg_svg = bg_svg.replace("</svg>", "") + "</svg>"
    bg_svg = optimize_svg(bg_svg)
    bg_svg = bg_svg.replace(
        f'<svg baseProfile="full" viewBox="0 0 {sz} {sz}" xmlns="http://www.w3.org/2000/svg">',
        '<svg viewBox="0 0 384 384" xmlns="http://www.w3.org/2000/svg">'
    )
    return bg_svg




# class Model:
#     def __init__(self):
#         self.df = pd.read_parquet("./train_df_6.parquet")
    
#     def predict(self, prompt: str) -> str:
#         row = self.df[self.df["description"] == prompt].iloc[0]

#         bg_svg = row["svg"]
#         bg_svg = bg_svg[:bg_svg.rfind("</g>")]# + "</g></svg>"
#         import pdb; pdb.set_trace()

#         with open("./output.svg", "w") as f:
#             f.write(bg_svg)
        
#         # bg_svg += text_to_svg("O", x_position_frac=0.75, y_position_frac=0.85, font_size=60, color=(255, 255, 255), font_path="/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf").split("\n")[1]
        
#         with open("./output_96_0.696.svg") as f:
#             aest_svg = f.read()
        
#         svg = optimize_svg((merge_svgs(bg_svg, aest_svg)))


        
#         return svg


class Model:
    def __init__(self):
        self.device_0 = device_0
        self.device_1 = device_1
        
        self.sd_config = SDConfig()
        self.primitive_config = PrimitiveConfig()
        self.diffvg_config = DiffvgConfig()


        self.question_gen = QuestionGenerator(device_0)
        
        self.sd_pipe = load_sd_model(self.device_0)

        # self.sd_scheduler = DDIMScheduler.from_pretrained(self.sd_config.stable_diffusion_path, subfolder="scheduler")
        # self.sd_pipe = StableDiffusionPipeline.from_pretrained(
        #     self.sd_config.stable_diffusion_path,
        #     scheduler=self.sd_scheduler,
        #     torch_dtype=torch.float16,
        #     safety_checker=None
        # ).to(self.device_1)
        

        self.aesthetic_evaluator = AestheticEvaluator(device=self.device_1)
        self.aesthetic_evaluator.predictor.to(self.device_1).eval()
        self.aesthetic_evaluator.clip_model.to(self.device_1).eval()
        self.aesthetic_evaluator.predictor.requires_grad_(False)
        self.aesthetic_evaluator.clip_model.requires_grad_(False)

        self.vqa_evaluator = VQAEvaluator('google/paligemma-2/transformers/paligemma2-10b-mix-448', device=self.device_1)
        self.vqa_evaluator.model.to(self.device_1).eval()
        self.vqa_evaluator.model.requires_grad_(False)

        with open(self.diffvg_config.base_svg_path, "r") as f:
            self.base_aest_svg = f.read()


    def get_questions_answers(self, prompt, max_sz=4):
        questions_str = self.question_gen(prompt)
        questions_str = questions_str.replace("```json", "").replace("```", "").strip()

        if questions_str[0] != "[":
            questions_str = "[" + questions_str

        if questions_str[-1] != "]":
            questions_str = questions_str + "]"

        questions_dict = ast.literal_eval(questions_str)
        questions = [dct["question"] for dct in questions_dict]
        choices = [dct["choices"] for dct in questions_dict]
        answers = [dct["answer"] for dct in questions_dict]
        
        sz = min(len(questions), len(choices), len(answers), max_sz)
        questions = questions[:sz]
        choices = choices[:sz]
        answers = answers[:sz]

        return questions, choices, answers


    def score(self, raw_pil_image, prompt, questions, choices, answers):
        # Process image for VQA and Aesthetic scores
        ip_main = ImageProcessor(image=raw_pil_image.copy(), seed=42).apply()
        processed_image_for_vqa_aes = ip_main.image.copy()

        aesthetic_score_val = self.aesthetic_evaluator.score(processed_image_for_vqa_aes)
        vqa_score_val, _ = self.vqa_evaluator.score(questions, choices, answers, processed_image_for_vqa_aes, n=4)

        # Process image for OCR score
        ip_ocr_branch = ImageProcessor(image=raw_pil_image.copy(), seed=42)
        ip_ocr_branch.reset().apply_random_crop_resize().apply_jpeg_compression(quality=90)
        ocr_image = ip_ocr_branch.image.copy()
        ocr_score_val = self.vqa_evaluator.ocr(ocr_image)
        
        final_score_val = harmonic_mean(vqa_score_val, aesthetic_score_val, beta=0.5) * ocr_score_val
        
        return final_score_val, vqa_score_val, aesthetic_score_val, ocr_score_val


    def predict(self, prompt: str, prompt_idx: int = 0) -> list:
        torch.cuda.empty_cache()

        imgs_pil = generate_sd_image(
            pipe=self.sd_pipe,
            prompt=prompt,
            config=self.sd_config,
            num_images_per_prompt=1
        )
        bg_svgs_list = [to_svg_vtracer(img, sz=192) for img in imgs_pil]
        svgs_list = [optimize_svg(merge_svgs(bg_svg, self.base_aest_svg)) for bg_svg in bg_svgs_list]
        imgs_for_scoring = [svg_to_png(svg) for svg in svgs_list]

        results_list = []

        # Sanitize prompt for directory name
        sanitized_prompt = re.sub(r'[^\w\s-]', '', prompt).strip()
        sanitized_prompt = re.sub(r'[-\s]+', '_', sanitized_prompt)
        sanitized_prompt = sanitized_prompt[:50] # Truncate to 50 chars
        if not sanitized_prompt: # Handle empty prompt after sanitization
            sanitized_prompt = f"prompt_{prompt_idx}"

        base_output_dir = "debug_svgs_per_prompt"
        prompt_specific_dir = os.path.join(base_output_dir, sanitized_prompt)

        try:
            questions, choices, answers = self.get_questions_answers(prompt)
            
            for i, (pil_img_for_score, svg_str) in enumerate(zip(imgs_for_scoring, svgs_list)):
                final_score, vqa_score, aesthetic_score, ocr_score = self.score(
                    pil_img_for_score, prompt, questions, choices, answers
                )

                os.makedirs(prompt_specific_dir, exist_ok=True)
                svg_filename = f"candidate_{i}.svg"
                full_svg_path = os.path.join(prompt_specific_dir, svg_filename)
                
                with open(full_svg_path, "w") as f:
                    f.write(svg_str)
                
                print(f"Saved {full_svg_path}. Metrics: Final={final_score:.4f}, VQA={vqa_score:.4f}, Aesthetic={aesthetic_score:.4f}, OCR={ocr_score:.4f}")

                results_list.append({
                    "svg_string": svg_str,
                    "svg_filename": full_svg_path,
                    "final_score": final_score,
                    "vqa_score": vqa_score,
                    "aesthetic_score": aesthetic_score,
                    "ocr_score": ocr_score,
                    "source_prompt": prompt,
                    "prompt_index": prompt_idx,
                    "candidate_index": i,
                    "prompt_directory": prompt_specific_dir
                })

        except Exception as e:
            print(f"Error during processing prompt '{prompt}' (index {prompt_idx}): {e}")
        
        return results_list

        # final_svg_opt = generate_diffvg_svg(
        #     bg_svg=bg_svg,
        #     base_svg_path=self.diffvg_config.base_svg_path,
        #     evaluator=self.aesthetic_evaluator,
        #     width=self.diffvg_config.width,
        #     height=self.diffvg_config.height,
        #     num_iterations=self.diffvg_config.num_iterations,
        #     validation_steps=self.diffvg_config.validation_steps,
        #     device=self.device_1
        # )

        # return final_svg_opt


# In[21]:


# model = Model()

# svg = model.predict("a blue desert at morning")
# img = svg_to_png(svg)

# with open("output.svg", "w") as f:
#     f.write(svg)

# img.save("output.png")


# In[22]:


# # ORG Train
# drawing_with_llms_path = kagglehub.competition_download('drawing-with-llms')
# train_df = pd.read_csv(f'{drawing_with_llms_path}/train.csv')

# train_question_df = pd.read_parquet(f'{drawing_with_llms_path}/questions.parquet')

# train_df = pd.merge(train_df, train_question_df, how='left', on='id')
# train_df = train_df.groupby('id').apply(lambda df: df.to_dict(orient='list'), include_groups=False)
# train_df = train_df.reset_index(name='qa')

# train_df["description"] = train_df.qa.apply(lambda qa: qa['description'][0])
# train_df["question"] = train_df.qa.apply(lambda qa: str(json.dumps(qa['question'], ensure_ascii=False)))
# train_df["answer"] = train_df.qa.apply(lambda qa: str(json.dumps(qa['answer'], ensure_ascii=False)))
# train_df["choices"] = train_df.qa.apply(lambda qa: str(json.dumps([x.tolist() for x in qa['choices']], ensure_ascii=False)))

# train_df = train_df.drop("qa", axis=1)
# train_df.head()


# In[ ]:





# In[23]:


# Gen train
train_df = pd.read_parquet(kagglehub.dataset_download("tomirol/qadataset") + "/qa_dataset_train.parquet")
train_df = train_df.iloc[-20:-10].copy()


# In[24]:


model = Model()

descriptions = train_df["description"].tolist()
ids = train_df["id"].tolist() # Capture IDs if needed for associating results
all_prompts_results = []

for idx, (desc_id, d_prompt) in enumerate(tqdm(zip(ids, descriptions), total=len(descriptions))):
    list_of_svg_metric_dicts = model.predict(d_prompt, prompt_idx=idx) 
    for res_dict in list_of_svg_metric_dicts:
        res_dict["original_id"] = desc_id # Add original ID for traceability
    all_prompts_results.extend(list_of_svg_metric_dicts)

# train_df["svg"] = svg_list # This is replaced by the new detailed output
# train_df.head()

if all_prompts_results:
    results_df = pd.DataFrame(all_prompts_results)
    results_df.to_parquet("./debug_all_svgs_with_metrics.parquet")
    print("\nSaved all SVG data and metrics to ./debug_all_svgs_with_metrics.parquet")
else:
    print("\nNo results generated to save.")


# In[25]:


# train_df.to_parquet("./train_df.parquet") # Original saving, now replaced
# train_df = pd.read_parquet("/tmp/train_df.parquet")


# In[26]:


del model
torch.cuda.empty_cache()


# In[27]:


# The following final evaluation section is commented out as it relies on 
# the old structure of train_df and submission_df.
# As per the request, "the final evaluation does not need to be done of course."

# solution = train_df[["id", "question", "choices", "answer"]].copy()
# submission = train_df[["id", "svg"]].copy()

# train_score, all_scores = score(copy.deepcopy(solution), copy.deepcopy(submission), "row_id", random_seed=42)
# print("\n\n")
# print(f"Scored {train_score: .5f} on {len(submission)} images")
# print(train_score)
# print(np.mean([x[1] for x in all_scores]))
# print(np.mean([x[2] for x in all_scores]))
# print(np.mean([x[3] for x in all_scores]))


# with open("results.json", "w") as f:
#     json.dump({
#         "final_score": float(train_score),
#         "vqa_score": float(np.mean([x[1] for x in all_scores])),
#         "aesthetic_score": float(np.mean([x[2] for x in all_scores])),
#         "ocr_score": float(np.mean([x[3] for x in all_scores])),
#     }, f)



# 0.741260364995869
# 0.7841644500482498
# 0.6368971585004757
# 1.0