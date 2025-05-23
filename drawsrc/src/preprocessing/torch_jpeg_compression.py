import torch
from kornia.enhance import jpeg_codec_differentiable

def apply_jpeg_compression_torch(image: torch.Tensor, quality: int = 85) -> torch.Tensor:
    quality = torch.tensor([quality], dtype=torch.float32, device=image.device)
    return jpeg_codec_differentiable(image.unsqueeze(0), jpeg_quality=quality).squeeze(0)
