import torch
import numpy as np
from kornia.filters import bilateral_blur

def apply_bilateral_filter_torch(image: torch.Tensor, d=5, sigma_color=75/255.0, sigma_space=(1.4, 1.4)) -> torch.Tensor:
    filtered_torch = bilateral_blur(
        image.unsqueeze(0),
        kernel_size=(d, d),
        sigma_color=sigma_color,
        sigma_space=sigma_space,
        border_type="replicate",
        color_distance_type="l1"
    ).squeeze()
    return filtered_torch

