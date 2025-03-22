import concurrent
import io
import logging
import re

import cairosvg
import kagglehub
import torch
from lxml import etree
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.models.siglip.modeling_siglip import SiglipModel
from transformers.models.siglip.processing_siglip import SiglipProcessor


from pathlib import Path
import io
from torch.nn import functional as F
import numpy as np
import requests
import torch
from PIL import Image
from tqdm.auto import tqdm


avg_final_loss = []

def generate_svg_from_image(img, pixel_size=1, image_size=(384, 384)):
    # img = Image.open(input_image_path).convert("RGB") 

    # Get image dimensions
    width, height = img.size

    # Start the SVG file
    svg_content = f'<svg viewBox="0 0 {image_size[0]} {image_size[1]}">\n'

    # Loop through each pixel in the image
    for y in range(height):
        for x in range(width):
            # Get the RGB color of the pixel
            r, g, b = img.getpixel((x, y))
            # Create a rectangle for the pixel
            svg_content += f'    <rect x="{x * pixel_size}" y="{y * pixel_size}" width="{pixel_size}" height="{pixel_size}" fill="rgb({r},{g},{b})" />\n'

    # Close the SVG file
    svg_content += '</svg>'
    svg_content = svg_content.replace("\n", "").replace("    ", "").replace(" />", "/>")

    return svg_content


def svg_to_png(svg_code: str, size: tuple = (384, 384)) -> Image.Image:
    if 'viewBox' not in svg_code:
        svg_code = svg_code.replace(
            '<svg', f'<svg viewBox="0 0 {size[0]} {size[1]}"'
        )

    png_data = cairosvg.svg2png(bytestring=svg_code.encode('utf-8'))
    return Image.open(io.BytesIO(png_data)).convert('RGB').resize(size)

def get_initial_embedding(
    dim: tuple[int],
    mean_value: float = 0.0,
    std_value: float = 0.1,
    low: float = -1.0,
    high: float = 1.0,
) -> torch.Tensor:
    embedding = torch.randn(dim, dtype=torch.float32) * std_value
    embedding = embedding + mean_value
    embedding = torch.clamp(embedding, low, high)
    return embedding


# def get_initial_embedding(
#     dim: tuple[int],
#     mean_value: float = 0.0,
#     std_value: float = 0.5,
#     low: float = -1.0,
#     high: float = 1.0,
# ) -> torch.Tensor:
#     embd = Image.open("output.png")
#     # embd = embd.resize((12, 12), Image.Resampling.NEAREST)
#     embd = np.asarray(embd)
#     embd = embd[None, ...]
#     embd = embd.astype(np.float32)
#     embd = (embd - 128) / 128
#     embd = torch.tensor(embd)
#     embd = embd.permute((0, 3, 1, 2))
#     return embd



def get_optimizer(
    params: torch.Tensor, lr: float = 0.01, weight_decay: float = 0.0
) -> torch.optim.Optimizer:
    return torch.optim.AdamW([params], lr=lr)#, weight_decay=weight_decay)
    # return torch.optim.LBFGS([params], lr=lr, max_iter=10, history_size=100, line_search_fn="strong_wolfe")

def get_loss_embds(image_features, text_features):
    # Normalize features
    image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)

    # Calculate similarity scores
    similarities = (image_features_norm @ text_features_norm.T).squeeze()

    loss = -similarities.mean()

    return loss


def get_loss(outputs):
    image_features = outputs.image_embeds
    text_features = outputs.text_embeds
    return get_loss_embds(image_features, text_features)


def get_vision_embds(model, pixel_values):
    vision_outputs = model.vision_model(
        pixel_values=pixel_values,
        output_attentions=model.config.output_attentions,
        output_hidden_states=model.config.output_hidden_states,
        return_dict=model.config.use_return_dict,
        interpolate_pos_encoding=False
    )
    image_embeds = vision_outputs[1]
    image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)

    return image_embeds

def get_text_embds(model, inputs):
    text_outputs = model.text_model(
        **inputs,
        output_attentions=model.config.output_attentions,
        output_hidden_states=model.config.output_hidden_states,
        return_dict=model.config.use_return_dict,
    )
    text_embeds = text_outputs[1]
    text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

    return text_embeds


def optimize_embedding(
    model: torch.nn.Module,
    processor: AutoProcessor,
    target_text: str,
    embedding_shape: tuple[int],
    num_iterations: int = 500,
    learning_rate: float = 0.05,
    regularization: float = 0.001,
    device: str = "cuda",
    sample_pixels: int = 4
) -> torch.Tensor:
    model.to(device)
    model.eval()  # Set to evaluation mode

    # Process the target text
    text_inputs = processor.tokenizer(
        [target_text], padding="max_length", return_tensors="pt"
    ).to(device)

    # Initialize the embedding - create a proper leaf tensor
    # Use nn.Parameter to ensure it's a leaf tensor that can be optimized
    embedding_shape = (embedding_shape[0], embedding_shape[1], embedding_shape[2] // sample_pixels, embedding_shape[3] // sample_pixels)
    embedding = torch.nn.Parameter(
        get_initial_embedding(embedding_shape).detach().to(device)
    )

    # Create optimizer with the parameter
    optimizer = get_optimizer(embedding, lr=learning_rate)#, weight_decay=regularization)

    pbar = tqdm(total=num_iterations)

    with torch.no_grad():
        text_embds = get_text_embds(model, text_inputs)
    

    # Optimization loop
    for i in range(num_iterations):
        
        # # Forward pass
        with torch.set_grad_enabled(True):
        # def closure():
            optimizer.zero_grad()
            embedding_int = F.interpolate(embedding, scale_factor=sample_pixels, mode="nearest")

            # Create inputs with our optimizable embedding
            inputs = {**text_inputs, "pixel_values": embedding_int}

            # outputs = model(**inputs)
            vision_embds = get_vision_embds(model, embedding_int)


            loss = get_loss_embds(text_embds, vision_embds)

            # # Backward pass
            loss.backward()
            optimizer.step()

            # return loss

        # loss = optimizer.step(closure)

        pbar.set_description(
            f"Iteration {i}/{num_iterations} | Loss: {loss:.4f}"
        )
        pbar.update(1)

    return embedding.detach()


def embedding_to_image(
    embedding: torch.Tensor, processor: AutoProcessor, model: torch.nn.Module
) -> Image.Image:
    # Make sure embedding is detached from computation graph and on CPU
    embedding = embedding.detach().cpu()
    
    # Get image processor parameters
    image_processor = processor.image_processor
    mean = image_processor.image_mean
    std = image_processor.image_std
    
    # Create copy of tensor
    image_tensor = embedding.clone()
    
    # Denormalize
    for i in range(3):
        image_tensor[0, i] = image_tensor[0, i] * std[i] + mean[i]
    
    # Ensure values are in [0,1] range
    image_tensor = torch.clamp(image_tensor, 0.0, 1.0)
    
    # Scale to [0,255] for PIL
    image_tensor = (image_tensor * 255).to(torch.uint8)
    
    # Convert to PIL Image (rearrange dimensions)
    pil_image = Image.fromarray(image_tensor[0].permute(1, 2, 0).numpy())
    pil_image = pil_image.convert("RGB") 
    
    return pil_image

def run_model(model, processor, text, image, device="cuda"):
    inputs = processor(text=[text], images=image, padding="max_length", return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
    
    return outputs


def run_optimization_with_display(
    model: torch.nn.Module,
    processor: AutoProcessor,
    target_text: str,
    embedding_shape: tuple[int] = (1, 3, 384, 384),
    num_iterations: int = 500,
    learning_rate: float = 2e-2,
    regularization: float = 0.001,
    display_every: int = 5,
    sample_pixels: int = 32,
    save_path: str = None,
) -> str:
    # For storing optimization history
    loss_history = []
    
    # Progress bar
    pbar = tqdm(total=num_iterations)
    
    # Run the optimization
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Target text]: '{target_text}'")
    
    optimized_embedding = optimize_embedding(
        model=model,
        processor=processor,
        target_text=target_text,
        embedding_shape=embedding_shape,
        num_iterations=num_iterations,
        learning_rate=learning_rate,
        regularization=regularization,
        device=device,
        sample_pixels=sample_pixels,
    )
    
    pbar.close()
    
    # Create the final image
    final_image = embedding_to_image(optimized_embedding, processor, model)

    svg = generate_svg_from_image(final_image, pixel_size=sample_pixels)

    final_image = svg_to_png(svg)

    outputs = run_model(model, processor, target_text, final_image, device)

    loss = get_loss(outputs)

    print(f"Final Loss: {loss.item()}")
    avg_final_loss.append(loss.item())


    return svg



svg_constraints = kagglehub.package_import('metric/svg-constraints', bypass_confirmation=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model:
    def __init__(self):
         # Quantization Configuration
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        self.model_path = "/home/mpf/code/siglip"
        # self.model_path = kagglehub.model_download('aishikai/google-siglip-so400m-patch14-384/transformers/default')
        self.processor = SiglipProcessor.from_pretrained(self.model_path)
        self.model = SiglipModel.from_pretrained(
            self.model_path,
            device_map="cuda",
            # quantization_config=quantization_config,
        )
        self.default_svg = """<svg width="256" height="256" viewBox="0 0 256 256"><circle cx="50" cy="50" r="40" fill="red" /></svg>"""
        self.constraints = svg_constraints.SVGConstraints()
        self.timeout_seconds = 90

    # You could try increasing `max_new_tokens`
    def predict(self, description: str, max_new_tokens=512) -> str:
        description = "SVG illustration of " + description
        def generate_svg():
            # return self.default_svg
            svg = run_optimization_with_display(
                model=self.model,
                processor=self.processor,
                target_text=description,
                sample_pixels=32,
                num_iterations=100,
                learning_rate=5e-2
            )
            return svg

        return generate_svg()


if __name__ == "__main__":
    # from scorer import run_model

    # prediction_df, score_value = run_model_eval(
    #     Model(),
    #     data_path="/kaggle/input/kaggle_evaluation/test.csv",
    #     output_path="output.csv",
    #     run_score=True,
    #     row_id_column_name="id",
    #     skip_constraints=True
    # )
    
    # print("Score final loss: ", score_value)

    # print("Final mean loss: ", np.mean(avg_final_loss))

    # import pdb
    # pdb.set_trace()
    
    
    model = Model()
    text = "a beautiful sunset over the ocean"

    svg = model.predict(text)
    # print(svg)

    image = svg_to_png(svg)
    image.save("output.png")

    # Replace the display line with file writing
    with open("output.svg", "w") as svg_file:
        svg_file.write(svg)

    outputs = run_model(model.model, model.processor, "SVG illustration of " + text, image, DEVICE)

    loss = get_loss(outputs)

    print(f"Final loss: {loss.item()}")
