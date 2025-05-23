import os
import string
import torch.nn as nn
import copy
import random

os.environ["TORCH_COMPILE_DISABLE"] = "1"

import pandas as pd
import kagglehub
import cv2
import io
import json
import cairosvg
import torch
from src.score_original import VQAEvaluator, ImageProcessor, AestheticEvaluator
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



def svg_to_png_no_resize(svg_code: str) -> Image.Image:
    png_data = cairosvg.svg2png(bytestring=svg_code.encode("utf-8"))
    return Image.open(io.BytesIO(png_data)).convert("RGB")


def visualize_tokens(decoded, image_feature_size, block_size=100):
    """
    Creates a visualization of decoded tokens in a grid layout.
    
    Args:
        decoded: 2D list of decoded tokens
        image_feature_size: Size of the token grid (assumed square)
        block_size: Size of each token block in pixels
    
    Returns:
        PIL Image with the visualization
    """
    # Create a blank image (white background)
    image_width = image_feature_size * block_size
    image_height = image_feature_size * block_size
    img = np.ones((image_height, image_width, 3), dtype=np.uint8) * 255
    
    # Draw grid lines
    for i in range(image_feature_size + 1):
        # Horizontal lines
        cv2.line(img, (0, i * block_size), (image_width, i * block_size), (200, 200, 200), 1)
        # Vertical lines
        cv2.line(img, (i * block_size, 0), (i * block_size, image_height), (200, 200, 200), 1)
    
    # Add text for each token
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2
    font_color = (0, 0, 0)  # Black text
    
    for i in range(image_feature_size):
        for j in range(image_feature_size):
            token_text = str(decoded[i][j])
            
            # Skip empty tokens or just spaces
            if not token_text.strip():
                continue
                
            # Calculate position for text
            x = j * block_size + 10  # Add padding
            y = i * block_size + block_size // 2  # Center vertically
            
            # Get text size to check if it fits in the block
            (text_width, text_height), _ = cv2.getTextSize(token_text, font, font_scale, font_thickness)
            
            # Check if text width exceeds block width (with some margin)
            if text_width > block_size - 30:
                # Split text in half (roughly)
                split_point = len(token_text) // 2
                # Try to find a space near the split point
                space_pos = token_text.find(' ', split_point - 5)
                if space_pos > 0 and space_pos < split_point + 5:
                    split_point = space_pos
                
                line1 = token_text[:split_point].strip() + "-"
                line2 = token_text[split_point:].strip()
                
                # Draw first line
                y1 = i * block_size + block_size // 3
                cv2.putText(img, line1, (x, y1), font, font_scale, font_color, font_thickness)
                
                # Draw second line
                y2 = i * block_size + 2 * block_size // 3
                cv2.putText(img, line2, (x, y2), font, font_scale, font_color, font_thickness)
            else:
                # Adjust y position to better center text
                y += text_height // 2
                
                # Draw text as a single line
                cv2.putText(img, token_text, (x, y), font, font_scale, font_color, font_thickness)
    
    # Convert to PIL Image
    return Image.fromarray(img)

def overlay_images(original_image, visualization, alpha=0.6):
    """
    Overlays the visualization on top of the original image with alpha blending.
    
    Args:
        original_image: PIL Image of the original input
        visualization: PIL Image of the token visualization
        alpha: Opacity of the visualization (0-1)
        
    Returns:
        PIL Image with the blended result
    """
    # Resize original image to match visualization dimensions
    orig_resized = original_image.resize(visualization.size, Image.LANCZOS)
    
    # Convert both images to numpy arrays
    orig_array = np.array(orig_resized)
    vis_array = np.array(visualization)
    
    # Blend the images
    blended = cv2.addWeighted(orig_array, 1-alpha, vis_array, alpha, 0)
    
    return Image.fromarray(blended)



def format_prompt(question: str, choices: list[str]) -> str:
    letters = string.ascii_uppercase
    prompt = f'<image>answer en Question: {question}\nChoices:\n'
    for i, choice in enumerate(choices):
        prompt += f'{letters[i]}. {choice}\n'
        # prompt += f'{choice}\n'
    return prompt


def test_word_embd(evaluator, description: str, question: str, choices: list[str], answer: str):
    image_shape = (
        evaluator.processor.image_processor.size["height"],
        evaluator.processor.image_processor.size["width"],
    )
    image_feature_size = 32
    
    answer = string.ascii_uppercase[choices.index(answer)]
    
    with torch.no_grad():
        inputs = evaluator.processor(
            images=Image.new("RGB", (image_shape[1], image_shape[0])),
            # text="<image>answer en Question: Is the object in the image a sphere?\n",
            text=format_prompt(question, choices),
            return_tensors="pt",
        ).to("cuda:0")
        
        target_tokens = evaluator.processor.tokenizer.encode(description + " ", add_special_tokens=False)
        target_embd = [evaluator.model.language_model.lm_head.weight[target_token].to("cuda:0") for target_token in target_tokens]
        # image_features = target_embd.unsqueeze(0).unsqueeze(0).repeat(1, image_feature_size, image_feature_size).reshape(1, -1, evaluator.model.config.text_config.hidden_size) 
        image_features = torch.zeros((1, image_feature_size * image_feature_size, evaluator.model.config.text_config.hidden_size), dtype=target_embd[0].dtype).to("cuda:0")

        # for i in range(image_features.shape[1]):
        
        # for s in range(20, image_features.shape[1] - 20, 50):
        #     for i in range(len(target_embd)):
        #         image_features[:, i+s, :] = target_embd[i % len(target_embd)]
        
        for i in range(len(target_embd)):
            image_features[:, i + 1, :] = target_embd[i % len(target_embd)]

        # target_token = evaluator.processor.tokenizer.encode("ball", add_special_tokens=False)[0]
        # target_embd = evaluator.model.language_model.lm_head.weight[target_token].to("cuda:0")
        # image_features[:, ::2, :] = target_embd

        inputs_embeds = evaluator.model.get_input_embeddings()(inputs["input_ids"])
        special_image_mask = (inputs["input_ids"] == evaluator.model.config.image_token_index).unsqueeze(-1)
        special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)
        inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)
        
        outputs = evaluator.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=inputs["attention_mask"],
            max_new_tokens=128,
            do_sample=False,
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
        )
        input_len = inputs["input_ids"].shape[1]
        # output_tokens = outputs[0][input_len:]
        output_tokens = outputs
        decoded = evaluator.processor.batch_decode(output_tokens, skip_special_tokens=True)[0]

        acc = decoded.strip().lower() == answer.strip().lower()

        print(question)
        print(decoded, " | ", answer, " | ", acc)
        print("="*50)
        
        return acc



if __name__ == "__main__": 
    # questions_df = pd.read_parquet("/kaggle/input/questions.parquet")
    # train_df = pd.read_csv("/kaggle/input/train.csv")
    # train_df = train_df[["id", "description"]]
    # train_df = train_df.merge(questions_df, on="id")
    # train_df["choices"] = train_df.choices.apply(lambda x: x.tolist())

    svg_val_path = kagglehub.dataset_download('raresbarbantan/draw-svg-validation')
    val_df = pd.read_csv(f'{svg_val_path}/validation.csv')
    val_df = val_df.groupby('id').apply(lambda df: df.to_dict(orient='list'), include_groups=False)
    val_df = val_df.reset_index(name='qa')
    val_df['description'] = val_df.qa.apply(lambda qa: qa['description'][0])
    val_df['question'] = val_df.qa.apply(lambda qa: qa['question'])
    val_df['answer'] = val_df.qa.apply(lambda qa: qa['answer'])
    val_df['choices'] = val_df.qa.apply(lambda qa: [eval(x) for x in qa['choices']])
    train_df = val_df.drop("qa", axis=1)
    train_df = train_df.explode(['question', 'choices', 'answer'])
    # train_df = train_df.head(10)
   
    evaluator = VQAEvaluator()
    
    acc_list_map = {}
    
    for _, row in tqdm(train_df.iterrows(), total=len(train_df)):
        description = row["description"]
        question = row["question"]
        choices = row["choices"]
        answer = row["answer"]

        acc = test_word_embd(evaluator, description, question, choices, answer)
        acc_list_map.setdefault(description, []).append(acc)
    
    all_acc = 0.0

    for description, acc_list in acc_list_map.items():
        acc_list_map[description] = sum(acc_list) / len(acc_list)
        all_acc += acc_list_map[description]
    
    print(f"Accuracy: {all_acc / len(acc_list_map)}")


