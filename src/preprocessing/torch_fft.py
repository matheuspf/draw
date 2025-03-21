import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import torch.nn.functional as F


def apply_fft_torch(image: torch.Tensor, cutoff_frequency: float = 0.5) -> torch.Tensor:
    """Differentiable FFT low-pass filter that matches original numpy implementation.
    
    Args:
        image: Tensor in [0,255] range (B,C,H,W) format
        cutoff_frequency: Normalized cutoff frequency (0-1)
        
    Returns:
        Filtered image tensor with same shape as input
    """
    # FFT calculations
    fft = torch.fft.fft2(image, dim=(-2, -1))
    fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))
    
    # Create low-pass mask
    H, W = image.shape[-2], image.shape[-1]
    center = (H//2, W//2)
    radius = int(min(center) * cutoff_frequency)
    
    # Create circular mask using tensor operations
    y = torch.arange(H, device=image.device)
    x = torch.arange(W, device=image.device)
    Y, X = torch.meshgrid(y, x, indexing='ij')
    mask = ((Y - center[0])**2 + (X - center[1])**2) <= radius**2
    mask = mask.to(image.dtype).unsqueeze(0).unsqueeze(0)  # Add batch/channel dims
    
    # Apply mask and inverse transform
    fft_filtered = fft_shifted * mask
    fft_ishift = torch.fft.ifftshift(fft_filtered, dim=(-2, -1))
    img_filtered = torch.fft.ifft2(fft_ishift, dim=(-2, -1))
    
    # Get real components and maintain valid range
    img_real = img_filtered.real
    img_real = torch.clamp(img_real, min=0.0, max=1.0)
    img_real = img_real.squeeze()

    return img_real
    