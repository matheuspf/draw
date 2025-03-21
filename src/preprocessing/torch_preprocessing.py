import torch
from .torch_jpeg_compression import apply_jpeg_compression_torch
from .torch_bilateral_filter import apply_bilateral_filter_torch
from .torch_median_filter import apply_median_filter_torch
from .torch_fft import apply_fft_torch


def apply_preprocessing_torch(image: torch.Tensor) -> torch.Tensor:
    image = apply_jpeg_compression_torch(image, quality=95)
    image = apply_median_filter_torch(image)
    image = apply_fft_torch(image)
    image = apply_bilateral_filter_torch(image)
    image = apply_jpeg_compression_torch(image, quality=92)
    return image


