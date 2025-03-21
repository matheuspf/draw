import torch
from kornia.enhance import jpeg_codec_differentiable

def apply_jpeg_compression_torch(image: torch.Tensor, quality: int = 85) -> torch.Tensor:
    return jpeg_codec_differentiable(image.unsqueeze(0), jpeg_quality=torch.tensor([quality])).squeeze(0)
