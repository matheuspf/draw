import torch
import cv2
import numpy as np
from PIL import Image
from src.preprocessing.torch_jpeg_compression import apply_jpeg_compression_torch
from src.score_original import ImageProcessor
from kornia.enhance import jpeg_codec_differentiable



def test_jpeg_compression():
    # Create a random image tensor with shape (C, H, W)
    np.random.seed(0)
    torch.manual_seed(0)
    # image_np = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

    image_np = cv2.imread("/home/mpf/Downloads/tt1.png")
    # image_np = cv2.imread("/home/mpf/code/kaggle/draw_bkp/C.png")

    # Kornia expects a BCHW tensor with normalized values
    image_torch = torch.tensor(image_np, dtype=torch.float32).permute(2, 0, 1) / 255.0

    # filtered_np = cv2.bilateralFilter(image_np, 5, 75, 75)
    image_processor = ImageProcessor(image=Image.fromarray(image_np))
    image_processor.apply_jpeg_compression(quality=95)
    filtered_np = np.array(image_processor.image)

    filtered_torch = apply_jpeg_compression_torch(image_torch, quality=95)

    filtered_torch = (filtered_torch * 255.0).permute(1, 2, 0).numpy().round().astype(np.uint8)

    print(filtered_np[:2, :2])
    print("-"*20)
    print(filtered_torch[:2, :2])
    print("="*50)
    print(filtered_np[30:32, 30:32])
    print("-"*20)
    print(filtered_torch[30:32, 30:32])

    # import pdb; pdb.set_trace()

    diff = np.abs(filtered_np.astype(np.float32) - filtered_torch.astype(np.float32))

    print(diff.mean(), diff.max(), diff.min(), np.quantile(diff, [0.01, 0.1, 0.5, 0.9, 0.99]))

    # # Compare the results
    # assert np.allclose(filtered_torch, filtered_np, atol=10.0), "The filters do not match!"


test_jpeg_compression()
