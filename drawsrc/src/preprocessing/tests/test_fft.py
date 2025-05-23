import torch
import cv2
import numpy as np
from PIL import Image
from src.preprocessing.torch_fft import apply_fft_torch
from src.score_original import ImageProcessor


def test_fft():
    # Create a random image tensor with shape (C, H, W)
    np.random.seed(0)
    torch.manual_seed(0)
    # image_np = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

    image_np = cv2.imread("/home/mpf/Downloads/tt1.png")
    
    # Convert BGR to RGB for consistent processing
    image_np_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    
    # Create PIL image from RGB numpy array
    image_processor = ImageProcessor(image=Image.fromarray(image_np_rgb))
    image_processor.apply_fft_low_pass(cutoff_frequency=0.5)
    filtered_np = np.array(image_processor.image)
    
    # Prepare torch tensor (CHW format, normalized to 0-1)
    image_torch = torch.tensor(image_np_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
    
    # Apply torch FFT filter
    filtered_torch = apply_fft_torch(image_torch)
    
    # Convert back to numpy array in HWC format with 0-255 range
    filtered_torch = (filtered_torch * 255.0).permute(1, 2, 0).numpy().round().clip(0, 255).astype(np.uint8)

    # Compare results
    print(filtered_np[:2, :2])
    print("-"*20)
    print(filtered_torch[:2, :2])
    print("="*50)
    print(filtered_np[30:32, 30:32])
    print("-"*20)
    print(filtered_torch[30:32, 30:32])

    diff = np.abs(filtered_np.astype(np.float32) - filtered_torch.astype(np.float32))

    print(diff.mean(), diff.max(), diff.min(), np.quantile(diff, [0.01, 0.1, 0.5, 0.9, 0.99]))

    # # Compare the results
    # assert np.allclose(filtered_torch, filtered_np, atol=10.0), "The filters do not match!"


test_fft()
