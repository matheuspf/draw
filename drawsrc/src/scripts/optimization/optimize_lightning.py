import os
import copy
import io
import random
import json
import torch
import numpy as np
import pandas as pd
import cairosvg
import cv2
import kornia
import pydiffvg
import lightning as L
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm
from torch.nn import functional as F
from typing import List, Tuple, Dict, Optional, Union
from torch.utils.data import Dataset, DataLoader

from src.score_original import VQAEvaluator, ImageProcessor, AestheticEvaluator
from src.score_gradient import (
    aesthetic_score_original,
    aesthetic_score_gradient,
    vqa_score_original,
    vqa_score_gradient
)
from src.preprocessing import apply_preprocessing_torch
from src.utils import optimize_svg, svg_to_png, create_random_svg
from src.text_to_svg import text_to_svg, rgb_to_hex


class SVGDataset(Dataset):
    def __init__(
        self, 
        split="train", 
        canvas_height=384, 
        canvas_width=384,
        data_path="/home/mpf/code/kaggle/draw/sub_reno_imagereward_prompt.parquet"
    ):
        self.canvas_height = canvas_height
        self.canvas_width = canvas_width
        
        df = pd.read_parquet(data_path)
        df = df[df["split"] == split].reset_index(drop=True)
        self.svgs = df["svg"].tolist()

    def __len__(self):
        return len(self.svgs)
    
    def __getitem__(self, idx):
        svg = self.convert_svg(self.svgs[idx])
        try:
            img_tensor = self.svg_to_tensor(svg)
            return {"svg": svg, "image": img_tensor}
        except Exception as e:
            # Fallback to a simpler SVG if conversion fails
            return {"svg": self.create_fallback_svg(), "image": torch.zeros(3, self.canvas_height, self.canvas_width)}
    
    def convert_svg(self, svg):
        """Process the SVG to add text and convert polygons to paths"""
        svg_lines = svg.replace(">", ">\n").strip().split("\n")
        svg_lines = svg_lines[:-2]
        svg = "\n".join(svg_lines)

        x_position_frac = 0.85
        y_position_frac = 0.9
        x_pos = int(self.canvas_width * (x_position_frac))
        y_pos = int(self.canvas_height * (y_position_frac))
        sz = 24
        svg += f'<path id="text-path-5" d="M {int(x_pos-sz/8)},{int(y_pos-sz*4/5)} h {sz} v {sz} h -{sz} z" fill="{rgb_to_hex(0, 0, 0)}" />\n'
        svg += text_to_svg("O", x_position_frac=x_position_frac, y_position_frac=y_position_frac, font_size=24, color=(255, 255, 255), font_path="/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf").split("\n")[1]
        svg = svg.replace("</svg>", "") + "</svg>"
        svg = self.convert_polygons_to_paths(svg)
        return svg
    
    def svg_to_tensor(self, svg):
        """Convert SVG to tensor"""
        png_data = cairosvg.svg2png(bytestring=svg.encode('utf-8'))
        img = Image.open(io.BytesIO(png_data)).convert('RGB')
        img = img.resize((self.canvas_width, self.canvas_height))
        img = np.array(img)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return img_tensor
    
    def convert_polygons_to_paths(self, svg_string):
        """Convert SVG polygon and polyline elements to path elements"""
        import re
        
        # Convert polygon points to path d with closing 'z'
        svg_string = re.sub(
            r'<polygon([\w\W]+?)points=(["\'])([\.\d, -]+?)(["\'])', 
            r'<path\1d=\2M\3z\4', 
            svg_string
        )
        
        # Convert polyline points to path d without closing
        svg_string = re.sub(
            r'<polyline([\w\W]+?)points=(["\'])([\.\d, -]+?)(["\'])', 
            r'<path\1d=\2M\3\4', 
            svg_string
        )
        
        return svg_string
    
    def create_fallback_svg(self):
        """Create a simple fallback SVG if conversion fails"""
        svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.canvas_width}" height="{self.canvas_height}" xmlns:xlink="http://www.w3.org/1999/xlink">\n'
        svg += f'  <rect width="{self.canvas_width}" height="{self.canvas_height}" fill="white" />\n'
        svg += '</svg>'
        return svg


class SVGDataModule(L.LightningDataModule):
    def __init__(
        self, 
        data_path="/home/mpf/code/kaggle/draw/sub_reno_imagereward_prompt.parquet",
        canvas_width=384, 
        canvas_height=384,
        batch_size=1,
        num_workers=4
    ):
        super().__init__()
        self.data_path = data_path
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def setup(self, stage=None):
        self.train_dataset = SVGDataset(
            split="train", 
            canvas_height=self.canvas_height,
            canvas_width=self.canvas_width,
            data_path=self.data_path
        )
        
        self.val_dataset = SVGDataset(
            split="validation", 
            canvas_height=self.canvas_height,
            canvas_width=self.canvas_width,
            data_path=self.data_path
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )


class SVGOptimizer(L.LightningModule):
    def __init__(
        self,
        canvas_width=384,
        canvas_height=384,
        learning_rate=1e-2,
        num_paths=64,
        max_width=4.0,
        use_blob=True,
        mask_region=None,
        val_samples=10
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        
        # Set up aesthetic evaluator
        self.aesthetic_evaluator = AestheticEvaluator()
        self.aesthetic_evaluator.predictor.eval()
        self.aesthetic_evaluator.predictor.requires_grad_(False)
        self.aesthetic_evaluator.clip_model.eval()
        self.aesthetic_evaluator.clip_model.requires_grad_(False)
        
        # Set up canvas size
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        
        # Set up region to optimize (default: top-left corner)
        if mask_region is None:
            self.mask_region = {"s": 8, "e": 72}  # s, e define the rectangular region
        else:
            self.mask_region = mask_region
            
        # Initialize shapes and parameters
        self.init_shapes()
        
        # Best model tracking
        self.best_val_score = -float('inf')
        self.best_shapes = None
        self.best_shape_groups = None
        
        # Validation samples
        self.val_samples = val_samples
        
        # Set up pydiffvg
        pydiffvg.set_use_gpu(torch.cuda.is_available())
        self.render_fn = pydiffvg.RenderFunction.apply
        
        # Initialize validation scores list
        self.validation_scores = []
        
    def init_shapes(self):
        """Initialize shapes to optimize"""
        s, e = self.mask_region["s"], self.mask_region["e"]
        region_width = e - s
        region_height = e - s
        
        self.shapes = []
        self.shape_groups = []
        
        if self.hparams.use_blob:
            # Create blob-like shapes like in painterly_rendering.py
            for i in range(self.hparams.num_paths):
                num_segments = random.randint(3, 5)
                num_control_points = torch.zeros(num_segments, dtype=torch.int32) + 2
                points = []
                p0 = (random.random(), random.random())
                points.append(p0)
                for j in range(num_segments):
                    radius = 0.05
                    p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
                    p2 = (p1[0] + radius * (random.random() - 0.5), p1[1] + radius * (random.random() - 0.5))
                    p3 = (p2[0] + radius * (random.random() - 0.5), p2[1] + radius * (random.random() - 0.5))
                    points.append(p1)
                    points.append(p2)
                    if j < num_segments - 1:
                        points.append(p3)
                        p0 = p3
                points = torch.tensor(points)
                points[:, 0] = points[:, 0] * region_width + s
                points[:, 1] = points[:, 1] * region_height + s
                
                path = pydiffvg.Path(
                    num_control_points=num_control_points,
                    points=points,
                    stroke_width=torch.tensor(1.0),
                    is_closed=True
                )
                self.shapes.append(path)
                
                path_group = pydiffvg.ShapeGroup(
                    shape_ids=torch.tensor([len(self.shapes) - 1]),
                    fill_color=torch.tensor([random.random(), random.random(), random.random(), random.random()])
                )
                self.shape_groups.append(path_group)
        else:
            # Create stroke-based paths
            for i in range(self.hparams.num_paths):
                num_segments = random.randint(1, 3)
                num_control_points = torch.zeros(num_segments, dtype=torch.int32) + 2
                points = []
                p0 = (random.random(), random.random())
                points.append(p0)
                for j in range(num_segments):
                    radius = 0.05
                    p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
                    p2 = (p1[0] + radius * (random.random() - 0.5), p1[1] + radius * (random.random() - 0.5))
                    p3 = (p2[0] + radius * (random.random() - 0.5), p2[1] + radius * (random.random() - 0.5))
                    points.append(p1)
                    points.append(p2)
                    points.append(p3)
                    p0 = p3
                points = torch.tensor(points)
                points[:, 0] = points[:, 0] * region_width + s
                points[:, 1] = points[:, 1] * region_height + s
                
                path = pydiffvg.Path(
                    num_control_points=num_control_points,
                    points=points,
                    stroke_width=torch.tensor(1.0),
                    is_closed=False
                )
                self.shapes.append(path)
                
                path_group = pydiffvg.ShapeGroup(
                    shape_ids=torch.tensor([len(self.shapes) - 1]),
                    fill_color=None,
                    stroke_color=torch.tensor([random.random(), random.random(), random.random(), random.random()])
                )
                self.shape_groups.append(path_group)
    
    def setup_optimizers(self):
        """Set up optimizers for different shape parameters"""
        # Variables to optimize
        self.points_vars = []
        self.stroke_width_vars = []
        self.color_vars = []
        
        # Set up parameters for optimization
        for path in self.shapes:
            path.points.requires_grad = True
            self.points_vars.append(path.points)
        
        if not self.hparams.use_blob:
            for path in self.shapes:
                path.stroke_width.requires_grad = True
                self.stroke_width_vars.append(path.stroke_width)
        
        if self.hparams.use_blob:
            for group in self.shape_groups:
                group.fill_color.requires_grad = True
                self.color_vars.append(group.fill_color)
        else:
            for group in self.shape_groups:
                group.stroke_color.requires_grad = True
                self.color_vars.append(group.stroke_color)
        
        # Define optimizers
        self.points_optim = torch.optim.Adam(self.points_vars, lr=self.hparams.learning_rate)
        if len(self.stroke_width_vars) > 0:
            self.width_optim = torch.optim.Adam(self.stroke_width_vars, lr=self.hparams.learning_rate * 0.1)
        self.color_optim = torch.optim.Adam(self.color_vars, lr=self.hparams.learning_rate * 0.01)
    
    def configure_optimizers(self):
        # Set up optimizers
        self.setup_optimizers()
        
        # Return optimizers (for Lightning to manage)
        optimizers = [self.points_optim, self.color_optim]
        if len(self.stroke_width_vars) > 0:
            optimizers.append(self.width_optim)
            
        return optimizers
    
    def render(self, seed=0):
        """Render the current state of shapes into an image"""
        scene_args = pydiffvg.RenderFunction.serialize_scene(
            self.canvas_width, 
            self.canvas_height, 
            self.shapes, 
            self.shape_groups
        )
        
        img = self.render_fn(
            self.canvas_width,   # width
            self.canvas_height,  # height
            2,                   # num_samples_x
            2,                   # num_samples_y
            seed,                # seed
            None,                # background_image
            *scene_args
        )
        
        # Compose img with white background
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device=img.device) * (1 - img[:, :, 3:4])
        
        return img
    
    def apply_random_crop_resize(self, img, crop_percent=0.05, seed=None):
        """Apply random crop and resize augmentation"""
        if seed is not None:
            torch.manual_seed(seed)
            
        crop_frac = crop_percent
        random_size = int(torch.rand(1).item() * crop_frac * img.shape[-1] + (1.0 - crop_frac) * img.shape[-1])
        
        # Use kornia for differentiable augmentations
        cropped = kornia.augmentation.RandomCrop((random_size, random_size))(img.unsqueeze(0))
        
        # Resize back to original size
        resized = F.interpolate(
            cropped, 
            size=(self.canvas_height, self.canvas_width), 
            mode="bicubic", 
            align_corners=False, 
            antialias=True
        ).squeeze(0)
        
        return resized
    
    def clamp_shapes_to_mask(self):
        """Ensure shapes stay within the mask region"""
        s, e = self.mask_region["s"], self.mask_region["e"]
        
        for path in self.shapes:
            path.points.data[:, 0].clamp_(s, e - 1e-3)
            path.points.data[:, 1].clamp_(s, e - 1e-3)
            
        if not self.hparams.use_blob:
            for path in self.shapes:
                path.stroke_width.data.clamp_(1.0, self.hparams.max_width)
        
        if self.hparams.use_blob:
            for group in self.shape_groups:
                group.fill_color.data.clamp_(0.0, 1.0)
        else:
            for group in self.shape_groups:
                group.stroke_color.data.clamp_(0.0, 1.0)
    
    def on_train_start(self):
        # Save initial state
        self.save_svg("initial.svg")
    
    def training_step(self, batch, batch_idx):
        # Manual optimization
        optimizers = self.optimizers()
        for opt in optimizers:
            opt.zero_grad()

        # Render the current state
        img = self.render(seed=self.current_epoch * 1000 + batch_idx)
        img = img[:, :, :3].permute(2, 0, 1).clamp(0, 1)

        # Apply augmentation
        img = self.apply_random_crop_resize(img, crop_percent=0.05, seed=batch_idx)

        # Apply preprocessing (normalize)
        img = apply_preprocessing_torch(img)

        # Calculate aesthetic score and gradients
        loss = -aesthetic_score_gradient(self.aesthetic_evaluator, img).mean()

        # Backward and optimizer step
        self.manual_backward(loss)
        for opt in optimizers:
            opt.step()

        # Clamp shapes to mask region after optimization step
        self.clamp_shapes_to_mask()

        # Log metrics
        self.log("train_loss", loss, prog_bar=True)

        # Save SVG periodically
        if batch_idx % 10 == 0:
            self.save_svg(f"iter_{self.current_epoch}_{batch_idx}.svg")
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        if batch_idx >= self.val_samples:
            return None
            
        # Render current state
        img = self.render(seed=42 + batch_idx)
        img = img[:, :, :3].permute(2, 0, 1).clamp(0, 1)
        
        # Convert to PIL image for non-differentiable evaluation
        pil_image = Image.fromarray(
            (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        ).convert("RGB")
        
        # Apply random crop/resize
        pil_image = self.apply_random_crop_resize_pil(pil_image, crop_percent=0.03, seed=batch_idx)
        
        # Process image for evaluation
        processed_image = ImageProcessor(pil_image, crop=False).apply().image
        
        # Calculate aesthetic score
        score = aesthetic_score_original(self.aesthetic_evaluator, processed_image)
        
        # Log score
        self.log("val_aesthetic_score", score, prog_bar=True)
        
        # Store score for epoch-end processing
        if not hasattr(self, 'validation_scores'):
            self.validation_scores = []
        self.validation_scores.append(score)
    
    def apply_random_crop_resize_pil(self, image, crop_percent=0.05, seed=None):
        """Apply random crop and resize to PIL image"""
        if seed is not None:
            random.seed(seed)
            
        width, height = image.size
        crop_pixels_w = int(width * crop_percent)
        crop_pixels_h = int(height * crop_percent)

        left = random.randint(0, crop_pixels_w + 1)
        top = random.randint(0, crop_pixels_h + 1)
        right = width - random.randint(0, crop_pixels_w + 1)
        bottom = height - random.randint(0, crop_pixels_h + 1)

        image = image.crop((left, top, right, bottom))
        image = image.resize((width, height), Image.BILINEAR)

        return image
    
    def on_validation_epoch_end(self):
        """
        Called at the end of validation to compute metrics based on collected outputs.
        Replaces the deprecated validation_epoch_end method.
        """
        # Get the validation scores we've collected during validation steps
        if hasattr(self, 'validation_scores') and self.validation_scores:
            mean_score = sum(self.validation_scores) / len(self.validation_scores)
            
            # Save best model
            if mean_score > self.best_val_score:
                self.best_val_score = mean_score
                self.best_shapes = copy.deepcopy(self.shapes)
                self.best_shape_groups = copy.deepcopy(self.shape_groups)
                self.save_svg(f"best_{mean_score:.3f}.svg")
                
            # Log the mean validation score
            self.log("val_mean_score", mean_score)
            
            # Clear the scores for the next epoch
            self.validation_scores = []
    
    def save_svg(self, filename):
        """Save current state as SVG"""
        pydiffvg.save_svg(
            filename,
            self.canvas_width,
            self.canvas_height,
            self.shapes,
            self.shape_groups
        )
    
    def export_best_svg(self, filename="best_final.svg"):
        """Export the best model found during training"""
        if self.best_shapes and self.best_shape_groups:
            pydiffvg.save_svg(
                filename,
                self.canvas_width,
                self.canvas_height,
                self.best_shapes,
                self.best_shape_groups
            )
            
            # Also optimize the SVG for size
            with open(filename, "r") as f:
                svg_content = f.read()
                
            optimized_svg = optimize_svg(svg_content)
            
            with open(f"optimized_{filename}", "w") as f:
                f.write(optimized_svg)
                
            return optimized_svg
        else:
            return None


# Main function to run training
def main():
    # Set seed for reproducibility
    seed_everything(42)
    
    # Set up data module
    data_module = SVGDataModule(
        canvas_width=384,
        canvas_height=384,
        batch_size=1,
        num_workers=4
    )
    
    # Set up model
    model = SVGOptimizer(
        canvas_width=384,
        canvas_height=384,
        learning_rate=1e-2,
        num_paths=64,
        max_width=4.0,
        use_blob=True,
        mask_region={"s": 8, "e": 72},
        val_samples=10
    )
    
    # Set up trainer
    trainer = L.Trainer(
        max_epochs=50,
        accelerator="auto",
        devices=1,
        # val_check_interval=10,
        check_val_every_n_epoch=10,
        log_every_n_steps=10,
        enable_checkpointing=True,
        default_root_dir="./lightning_logs"
    )
    
    # Train the model
    trainer.fit(model, datamodule=data_module)
    
    # Export the best model
    final_svg = model.export_best_svg("best_aesthetic.svg")
    print(f"Final SVG length: {len(final_svg.encode('utf-8'))}")
    
    # Convert to PNG for visualization
    svg_to_png(final_svg, "best_aesthetic.png")


def seed_everything(seed):
    """Set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    main()
