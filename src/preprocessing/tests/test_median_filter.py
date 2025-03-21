import unittest
import torch
import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from src.preprocessing.torch_median_filter import apply_median_filter_torch
from src.score_original import ImageProcessor
import os
import random

class TestMedianFilter(unittest.TestCase):
    
    def setUp(self):
        # Create a directory for test outputs if it doesn't exist
        self.output_dir = "test_outputs"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set random seed for reproducibility
        random.seed(42)
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create test images
        self.create_test_images()
    
    def create_test_images(self):
        # Create a random noise image
        self.noise_img = Image.fromarray(
            (np.random.rand(100, 100, 3) * 255).astype(np.uint8)
        )
        
        # Create a simple geometric image
        img_array = np.zeros((100, 100, 3), dtype=np.uint8)
        # Add a red square
        img_array[20:50, 20:50, 0] = 255
        # Add a blue circle
        y, x = np.ogrid[:100, :100]
        mask = ((x - 70) ** 2 + (y - 70) ** 2) <= 20 ** 2
        img_array[mask, 2] = 255
        self.geometric_img = Image.fromarray(img_array)
        
        # Save original test images
        self.noise_img.save(os.path.join(self.output_dir, "original_noise.png"))
        self.geometric_img.save(os.path.join(self.output_dir, "original_geometric.png"))
    
    def test_median_filter_random_image(self):
        # Create a random image tensor with shape (C, H, W)
        np.random.seed(0)
        torch.manual_seed(0)
        image_np = np.random.randint(0, 256, (3, 100, 100), dtype=np.uint8)
        image_torch = torch.tensor(image_np, dtype=torch.float32)

        # Apply the PyTorch median filter
        filtered_torch = apply_median_filter_torch(image_torch, size=9)

        # Convert the numpy image to a PIL image and apply the original median filter
        image_processor = ImageProcessor(image=Image.fromarray(image_np.transpose(1, 2, 0)))
        image_processor.apply_median_filter(size=9)
        filtered_np = np.array(image_processor.image).transpose(2, 0, 1)

        # Compare the results
        assert np.allclose(filtered_torch.numpy(), filtered_np, atol=1e-5), "The filters do not match!"

        print("Test passed: PyTorch median filter matches the original implementation.")

    
    def test_median_filter_comparison(self):
        """Test that PyTorch median filter produces similar results to PIL's median filter."""
        
        # Test with different kernel sizes
        kernel_sizes = [3, 5, 9]
        
        for img, img_name in [(self.noise_img, "noise"), (self.geometric_img, "geometric")]:
            for size in kernel_sizes:
                # Apply PIL's median filter

                image_processor = ImageProcessor(image=img)
                image_processor.apply_median_filter(size=size)
                pil_filtered = image_processor.image

                # pil_filtered = img.filter(ImageFilter.MedianFilter(size=size))
                # pil_filtered.save(os.path.join(self.output_dir, f"pil_{img_name}_median_{size}.png"))
                
                # Convert PIL image to PyTorch tensor
                img_tensor = torch.from_numpy(np.array(img).transpose(2, 0, 1) / 255.0).float()
                
                # Apply PyTorch median filter
                torch_filtered_tensor = apply_median_filter_torch(img_tensor, size=size)
                
                # Convert back to PIL for visualization
                torch_filtered_np = (torch_filtered_tensor.numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
                torch_filtered = Image.fromarray(torch_filtered_np)
                torch_filtered.save(os.path.join(self.output_dir, f"torch_{img_name}_median_{size}.png"))
                
                # Compare the results
                pil_np = np.array(pil_filtered)
                torch_np = torch_filtered_np
                
                # Calculate mean absolute difference
                mae = np.mean(np.abs(pil_np.astype(float) - torch_np.astype(float)))
                print(f"Mean absolute difference for {img_name} with kernel size {size}: {mae}")
                
                # Visualize the difference
                diff = np.abs(pil_np.astype(int) - torch_np.astype(int))
                diff_img = Image.fromarray(diff.astype(np.uint8))
                diff_img.save(os.path.join(self.output_dir, f"diff_{img_name}_median_{size}.png"))
                
                # Create comparison plot
                fig, axes = plt.subplots(1, 3, figsize=(15, 7))
                axes[0].imshow(pil_np)
                axes[0].set_title(f"PIL Median (size={size})")
                axes[0].axis('off')
                
                axes[1].imshow(torch_np)
                axes[1].set_title(f"PyTorch Median (size={size})")
                axes[1].axis('off')
                
                axes[2].imshow(diff, cmap='hot')
                axes[2].set_title(f"Difference (MAE={mae:.2f})")
                axes[2].axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, f"comparison_{img_name}_median_{size}.png"))
                plt.close()
                
                # Assert that the difference is within an acceptable range
                # The implementations are different, so we expect some differences
                self.assertLess(mae, 30.0, f"Difference too large for {img_name} with kernel size {size}")
    
    def test_edge_cases(self):
        """Test edge cases for the PyTorch median filter."""
        
        # Test with a single-channel image
        gray_img = self.geometric_img.convert('L')
        gray_tensor = torch.from_numpy(np.array(gray_img) / 255.0).float().unsqueeze(0)
        
        # Apply PyTorch median filter
        torch_filtered_tensor = apply_median_filter_torch(gray_tensor, size=5)
        
        # Check output shape
        self.assertEqual(torch_filtered_tensor.shape, gray_tensor.shape, 
                         "Output shape should match input shape for grayscale image")
        
    def test_median_filter_gradient(self):
        """Test that gradients can flow through the median filter operation."""
        # Create a random image tensor with requires_grad
        input_tensor = torch.randn(3, 10, 10, requires_grad=True)
        
        # Apply median filter
        filtered = apply_median_filter_torch(input_tensor, size=3)
        
        # Compute a dummy loss (sum of filtered pixels)
        loss = filtered.sum()
        
        # Backpropagate to compute gradients
        loss.backward()

        print(f"Gradient info: {input_tensor.grad.mean()}, {input_tensor.grad.std()}, {input_tensor.grad.min()}, {input_tensor.grad.max()}")
        
        # Verify gradients exist
        self.assertIsNotNone(input_tensor.grad, "Gradients should be available for input tensor")
        
        # Verify gradients are not all zero (basic sanity check)
        self.assertTrue(torch.any(input_tensor.grad != 0), "Gradients should not be all zero")

if __name__ == '__main__':
    unittest.main()
