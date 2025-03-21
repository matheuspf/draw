import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import torch.nn.functional as F


def apply_median_filter_torch(image: torch.Tensor, size=9) -> torch.Tensor:
    """
    Apply median filter to a tensor image in a fully differentiable manner.
    
    Args:
        image: A tensor of shape (C, H, W)
        size: Size of the median filter window
    
    Returns:
        Filtered tensor of the same shape
    """
    # Check if dimensions are correct
    if len(image.shape) != 3:
        raise ValueError("Image should be a 3D tensor (C, H, W)")
    
    # Get dimensions
    channels, height, width = image.shape
    
    # Padding size
    padding = size // 2
    
    # Process each channel separately
    result = torch.zeros_like(image)
    
    # Create a properly padded version for the full image
    padded_image = F.pad(image, (padding, padding, padding, padding), mode='replicate')
    
    for c in range(channels):
        # Extract the padded channel
        padded_channel = padded_image[c]  # Shape: (H+2*padding, W+2*padding)
        
        # Use unfold to extract patches - shape: (H, W, size, size)
        patches = padded_channel.unfold(0, size, 1).unfold(1, size, 1)
        
        # Reshape to (H, W, size*size)
        flat_patches = patches.reshape(height, width, -1)
        
        # Sort each patch
        sorted_patches, _ = torch.sort(flat_patches, dim=2)
        
        # Extract the median (middle value)
        median_idx = size * size // 2
        medians = sorted_patches[:, :, median_idx]
        
        # Assign to result
        result[c] = medians
    
    return result
