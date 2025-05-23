import torch
import cv2
import numpy as np
from PIL import Image
from src.preprocessing.torch_bilateral_filter import apply_bilateral_filter_torch
from src.score_original import ImageProcessor



def test_bilateral_filter_kornia():
    # Create a random image tensor with shape (C, H, W)
    np.random.seed(0)
    torch.manual_seed(0)
    # image_np = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

    image_np = cv2.imread("/home/mpf/Downloads/tt1.png")

    # Kornia expects a BCHW tensor with normalized values
    image_torch = torch.tensor(image_np, dtype=torch.float32).permute(2, 0, 1) / 255.0

    # filtered_np = cv2.bilateralFilter(image_np, 5, 75, 75)
    image_processor = ImageProcessor(image=Image.fromarray(image_np))
    image_processor.apply_bilateral_filter(d=5, sigma_color=75, sigma_space=75)
    filtered_np = np.array(image_processor.image)

    filtered_torch = apply_bilateral_filter_torch(image_torch)
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

    # Compare the results
    assert np.allclose(filtered_torch, filtered_np, atol=10.0), "The filters do not match!"




def opt_bilateral_filter_kornia():
    # Create a random image tensor with shape (C, H, W)
    np.random.seed(0)
    torch.manual_seed(0)
    image_np = cv2.imread("/home/mpf/Downloads/tt1.png")
    image_np = cv2.resize(image_np, (640, 480))
    
    # image_np = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    # Kornia expects a BCHW tensor with normalized values
    image_torch = torch.tensor(image_np, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
    filtered_np = cv2.bilateralFilter(image_np, 5, 75, 75)

    best_res = (1e8, 1e8, 1e8, 1e8)

    for k in [5]:
        for sigma_color in [75.0/255]:
            for sigma_space in np.arange(1.0, 2.0, 0.01):
                # Use the correct parameters to match OpenCV
                filtered_torch = (bilateral_blur(
                    image_torch,
                    kernel_size=(k, k),
                    # sigma_color=75.0/255.0,  # OpenCV uses 0-255 scale, Kornia uses 0-1
                    # sigma_space=(25.0, 25.0),  # Need to adjust sigma_space scaling
                    sigma_color=sigma_color,
                    sigma_space=(sigma_space, sigma_space),
                    border_type="reflect",
                    color_distance_type="l1"  # OpenCV uses L1 distance by default
                )[0] * 255.0).permute(1, 2, 0).numpy().astype(np.uint8)


                # print(filtered_np[:2, :2])
                # print("-"*20)
                # print(filtered_torch[:2, :2])
                # print("="*50)
                # print(filtered_np[30:32, 30:32])
                # print("-"*20)
                # print(filtered_torch[30:32, 30:32])

                diff = np.abs(filtered_np.astype(np.float32) - filtered_torch.astype(np.float32)).mean()

                # print(f"sigma_space: {sigma_space}, sigma_color: {sigma_color}, diff: {diff}")

                if diff < best_res[0]:
                    best_res = (diff, k, sigma_color, sigma_space)

    print(f"Best result: {best_res}")



def test_bilateral_filter_random_image():
    # Create a random image tensor with shape (C, H, W)
    np.random.seed(0)
    torch.manual_seed(0)
    image_np = np.random.randint(0, 256, (3, 100, 100), dtype=np.uint8)
    image_torch = torch.tensor(image_np, dtype=torch.float32)

    filtered_torch = apply_bilateral_filter_torch(image_torch, d=5, sigma_color=75, sigma_space=75)

    # image_processor = ImageProcessor(image=Image.fromarray(image_np.transpose(1, 2, 0)))
    # image_processor.apply_bilateral_filter(d=5, sigma_color=75, sigma_space=75)
    # filtered_np = np.array(image_processor.image).transpose(2, 0, 1)

    image_np = image_np.transpose(1, 2, 0)
    filtered_np = cv2.bilateralFilter(image_np, 5, 75, 75)
    filtered_np = filtered_np.transpose(2, 0, 1)

    print(filtered_torch[:, :2, :2])
    print("-"*20)
    print(filtered_np[:, :2, :2])
    print("="*50)
    print(filtered_torch[:, 30:32, 30:32])
    print("-"*20)
    print(filtered_np[:, 30:32, 30:32])

    import pdb; pdb.set_trace()

    # Compare the results
    assert np.allclose(filtered_torch.numpy(), filtered_np, atol=1e-5), "The filters do not match!"

    print("Test passed: PyTorch median filter matches the original implementation.")


test_bilateral_filter_kornia()
# opt_bilateral_filter_kornia()
