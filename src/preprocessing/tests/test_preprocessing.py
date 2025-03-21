from src.preprocessing.torch_preprocessing import apply_preprocessing_torch
from PIL import Image
import cv2
import torch
import numpy as np
from src.score_original import ImageProcessor


def test_preprocessing():

    img_org = cv2.imread("/home/mpf/Downloads/tt1.png")
    # img_org = cv2.imread("/home/mpf/code/kaggle/draw_bkp/C.png")

    
    processor = ImageProcessor(image=Image.fromarray(img_org))
    processor.apply_jpeg_compression(quality=95) \
        .apply_median_filter(size=9) \
        .apply_fft_low_pass(cutoff_frequency=0.5) \
        .apply_bilateral_filter(d=5, sigma_color=75, sigma_space=75) \
        .apply_jpeg_compression(quality=92)

    img_org = np.array(processor.image)

    cv2.imwrite("org.png", img_org)

    img = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)
    img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0

    img = apply_preprocessing_torch(img)


    img = (img * 255).permute(1, 2, 0).numpy().round().clip(0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imwrite("processed.png", img)

    diff = np.abs(img_org.astype(np.float32) - img.astype(np.float32))

    print(diff.mean(), diff.max(), diff.min(), np.quantile(diff, [0.01, 0.1, 0.5, 0.9, 0.99]))

    diff = diff.mean(-1).round().clip(0, 255).astype(np.uint8)

    cv2.imwrite("diff.png", diff)


test_preprocessing()
