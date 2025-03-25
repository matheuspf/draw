import torch
from .torch_jpeg_compression import apply_jpeg_compression_torch
from .torch_bilateral_filter import apply_bilateral_filter_torch
from .torch_median_filter import apply_median_filter_torch
from .torch_fft import apply_fft_torch

from src.segmentation.train import LitSegmentation

# seg_model = LitSegmentation.load_from_checkpoint("/home/mpf/code/kaggle/draw/checkpoints/segmentation-epoch=06-val_loss=0.0073.ckpt")
# seg_model.eval()
# seg_model.requires_grad_(False)
# seg_model.to("cuda")

def apply_preprocessing_torch(image: torch.Tensor) -> torch.Tensor:
    image = apply_jpeg_compression_torch(image, quality=95)
    image = apply_median_filter_torch(image)
    image = apply_fft_torch(image)
    image = apply_bilateral_filter_torch(image)
    image = apply_jpeg_compression_torch(image, quality=92).clamp(0, 1)
    # image = seg_model(image.unsqueeze(0))[0]

    return image.clamp(0, 1)


