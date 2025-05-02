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

def display_word_emb(evaluator, image_pil="t1.png", prompt_list=["cap en\n", "ocr\n"]):
    image_shape = (
        evaluator.processor.image_processor.size["height"],
        evaluator.processor.image_processor.size["width"],
    )
    image_feature_size = (32 if image_shape[0] == 448 else 16)

    if isinstance(image_pil, (str, Path)):
        image_pil = Image.open(image_pil)

    image_pil = image_pil.resize((image_shape[1], image_shape[0]))

    
    def decode_tokens(image_tokens):
        decoded = evaluator.processor.batch_decode(image_tokens.reshape((-1, 1)), skip_special_tokens=True)
        image_tokens = image_tokens.reshape(image_feature_size, image_feature_size)
        decoded = [[decoded[i*image_feature_size+j] for j in range(image_feature_size)] for i in range(image_feature_size)]
        visualization = visualize_tokens(decoded, image_feature_size, block_size=100)
        overlaid = overlay_images(image_pil, visualization, alpha=0.6)
        return overlaid
    
    with torch.no_grad():
        image = torch.from_numpy(np.array(image_pil)).permute(2, 0, 1).float().to("cuda:0") / 255.0
        # image = F.interpolate(
        #     image.unsqueeze(0), size=image_shape, mode="bicubic", align_corners=False, antialias=True
        # )
        image = (image - 0.5) / 0.5

        image_features = evaluator.model.get_image_features(image.unsqueeze(0))[0]
        image_logits = evaluator.model.language_model.lm_head(image_features)
        image_tokens = image_logits.argmax(dim=-1)

    overlay_image = decode_tokens(image_tokens)
    overlay_image.save("token_visualization_vision.png")

    for prompt in tqdm(prompt_list):
        with torch.no_grad():
            inputs = evaluator.processor(
                images=image_pil,
                text=f"<image>{prompt}",
                return_tensors="pt",
            ).to("cuda:0")

            outputs = evaluator.model(**inputs)
            image_tokens = outputs.logits[0, :image_feature_size**2, :].argmax(dim=-1)

        overlay_image = decode_tokens(image_tokens)
        overlay_image.save(f"token_visualization_llm_{prompt.strip()}.png")

if __name__ == "__main__":
    evaluator = VQAEvaluator()
    display_word_emb(evaluator, image_pil="/home/mpf/Downloads/r1.jpg")
