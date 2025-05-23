import torch
import copy
from PIL import Image
from tqdm import tqdm
import numpy as np
from pathlib import Path
from torchvision import transforms, models
from torch.utils.data import Dataset
import lightning as L
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import io
# from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from src.segmentation.data import prepare_datasets

torch.set_float32_matmul_precision('medium')


def get_transform(size: int = 224, split: str = "train"):
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((size, size)),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class PreprocessDataset(Dataset):
    def __init__(
        self,
        data_path: str | Path = "./segmentation_data",
        split: str = "train",
        transform: transforms.Compose | None = None,
    ):
        self.data_path = Path(data_path).resolve() / split
        self.input_images = sorted(list((self.data_path / "input").glob("*.png")))
        self.output_images = sorted(list((self.data_path / "output").glob("*.png")))
        self.transform = transform
        assert [inp.name == out.name for inp, out in zip(self.input_images, self.output_images)]

    def __getitem__(self, idx):
        input_image = Image.open(self.input_images[idx]).convert("RGB")
        output_image = Image.open(self.output_images[idx]).convert("RGB")

        if self.transform:
            input_image = self.transform(input_image)
            output_image = self.transform(output_image)

        return input_image, output_image

    def __len__(self):
        return len(self.input_images)


class LitSegmentation(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = models.segmentation.fcn_resnet50(num_classes=3)
        self.loss_fn = torch.nn.MSELoss()
        # self.loss_fn = torch.nn.L1Loss()
    
    def forward(self, x):
        return (self.model(x)["out"] + x).clamp(0, 1)

    def training_step(self, batch):
        images, targets = batch

        outputs = self.model(images)["out"]
        outputs = (outputs + images).clamp(0, 1)
        
        loss = self.loss_fn(outputs, targets)
        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch

        outputs = self.model(images)["out"]
        outputs = (outputs + images).clamp(0, 1)

        loss = self.loss_fn(outputs, targets)
        self.log("val_loss", loss, prog_bar=True)
        
        # Log images for visualization every few batches
        if batch_idx % 5 == 0:
            # Take first few images from batch
            num_images = min(4, images.size(0))
            self.visualize_results(
                images[:num_images], 
                targets[:num_images], 
                outputs[:num_images], 
                batch_idx
            )
            
        # # Calculate image quality metrics
        # psnr = peak_signal_noise_ratio(outputs, targets)
        # ssim = structural_similarity_index_measure(outputs, targets)
        # self.log("val_psnr", psnr, prog_bar=True)
        # self.log("val_ssim", ssim, prog_bar=True)
        
        return loss

    def visualize_results(self, inputs, targets, predictions, batch_idx):
        """
        Visualize input, target and prediction images side by side in TensorBoard.
        
        Args:
            inputs: Input images (B, C, H, W)
            targets: Target images (B, C, H, W)
            predictions: Predicted images (B, C, H, W)
            batch_idx: Batch index for logging
        """
        # Denormalize the images to [0, 1] range
        # mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        # std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        
        # inputs_denorm = inputs * std + mean
        # targets_denorm = targets * std + mean
        # predictions_denorm = predictions * std + mean

        inputs_denorm = inputs
        targets_denorm = targets
        predictions_denorm = predictions
        
        # Clamp values to [0, 1] to ensure valid image range
        inputs_denorm = torch.clamp(inputs_denorm, 0, 1)
        targets_denorm = torch.clamp(targets_denorm, 0, 1)
        predictions_denorm = torch.clamp(predictions_denorm, 0, 1)
        
        # Create a grid with original, target, and predicted images
        batch_size = inputs.size(0)
        all_images = []
        
        for i in range(batch_size):
            all_images.extend([
                inputs_denorm[i], 
                targets_denorm[i], 
                predictions_denorm[i]
            ])
        
        # Make a grid with 3 columns (input, target, prediction)
        grid = make_grid(all_images, nrow=3, padding=2, normalize=False)
        
        # Add text labels to the grid
        fig = plt.figure(figsize=(12, 4 * batch_size))
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
        
        # for i in range(batch_size):
        #     plt.text(0.15, (i * 3) + 0.5, "Input", fontsize=12)
        #     plt.text(0.45, (i * 3) + 0.5, "Target", fontsize=12)
        #     plt.text(0.75, (i * 3) + 0.5, "Prediction", fontsize=12)
        
        plt.axis('off')
        
        # Convert the plot to an image
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        
        # Log to TensorBoard
        self.logger.experiment.add_image(
            f'validation_samples/batch_{batch_idx}', 
            torch.tensor(np.array(Image.open(buf))).permute(2, 0, 1),
            self.global_step
        )
        
        # Also log the grid directly for simpler visualization
        self.logger.experiment.add_image(
            f'validation_grid/batch_{batch_idx}', 
            grid,
            self.global_step
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]


class SegmentationData(L.LightningDataModule):
    def prepare_data(self):
        prepare_datasets()
    
    def train_dataloader(self):
        dataset = PreprocessDataset(
            data_path="./segmentation_data", split="train", transform=get_transform(224, "train")
        )
        return torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    def val_dataloader(self):
        dataset = PreprocessDataset(
            data_path="./segmentation_data", split="val", transform=get_transform(224, "val")
        )
        return torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)


if __name__ == "__main__":
    model = LitSegmentation()
    data = SegmentationData()
    
    # Add checkpoint callback to save the best model based on validation loss
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints/',
        filename='segmentation-{epoch:02d}-{val_loss:.4f}',
        save_top_k=1,
        mode='min',
    )
    
    trainer = L.Trainer(
        max_epochs=10, 
        log_every_n_steps=50,
        callbacks=[checkpoint_callback],
    )
    
    trainer.fit(model, data)
    
    # Print the path to the best model
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")
    print(f"Best validation loss: {checkpoint_callback.best_model_score:.4f}")
