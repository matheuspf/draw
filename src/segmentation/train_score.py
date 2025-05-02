import torch
import torch.nn as nn
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
from src.score_original import AestheticEvaluator
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
import json
import clip
import multiprocessing as mp
from functools import partial
from src.segmentation.model import CLIP
from src.preprocessing import apply_preprocessing_torch


def process_image(evaluator, image, output_path, split, out_name=None):
    from src.score_original import ImageProcessor

    """Process a single image for dataset preparation"""
    if isinstance(image, (str, Path)):
        out_name = image.stem
        image = Image.open(image).convert("RGB").resize((384, 384))

    output_image_org = ImageProcessor(copy.deepcopy(image)).apply().image
    score = evaluator.score(output_image_org)


    torch_image = torch.from_numpy(np.array(image)).permute(2, 0, 1)
    torch_image = torch_image.float().cuda() / 255.0
    torch_image = apply_preprocessing_torch(torch_image)

    torch_image = (torch_image * 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    output_image = Image.fromarray(torch_image)

    output_image.save(output_path / split / "images" / f"{out_name}.png")
    return score


def generate_random_images(evaluator, output_path, split, num_samples=100):
    std_value: float = 0.1
    mean_value: float = 0.5
    low: float = 0.0
    high: float = 1.0

    data = {}

    for idx in tqdm(range(num_samples), desc=f"Generating random images for {split}"):
        random_image = np.random.randn(224, 224, 3)
        random_image = random_image * std_value
        random_image = random_image + mean_value
        random_image = np.clip(random_image, 0.0, 1.0)
        random_image = (random_image * 255).astype(np.uint8)
        random_image = Image.fromarray(random_image).convert("RGB")
        score = process_image(evaluator, random_image, output_path, split, f"random_{idx}")
        data[f"random_{idx}"] = score
    
    return data


def prepare_datasets(
    data_path: str | Path = "/home/mpf/code/yolov9/coco/images/val2017",
    max_samples: int = 1000,
    val_split: float = 0.05,
    output_path: str | Path = "./score_data",
    num_workers: int = 24
):
    dataset_path = Path(data_path).resolve()
    output_path = Path(output_path).resolve()

    if output_path.exists():
        return
    
    evaluator = AestheticEvaluator()

    # Create directories
    (output_path / "train" / "images").mkdir(parents=True, exist_ok=True)
    (output_path / "val" / "images").mkdir(parents=True, exist_ok=True)

    images_list = sorted(list(dataset_path.glob("*.jpg")))[:max_samples]
    np.random.shuffle(images_list)
    val_size = int(len(images_list) * val_split)

    train_images = images_list[val_size:]
    val_images = images_list[:val_size]

    train_data = {}
    val_data = {}

    for image in tqdm(train_images, desc="Processing training images"):
        score = process_image(evaluator, image, output_path, "train")
        train_data[image.stem] = score

    for image in tqdm(val_images, desc="Processing validation images"):
        score = process_image(evaluator, image, output_path, "val")
        val_data[image.stem] = score
   
    train_data.update(generate_random_images(evaluator, output_path, "train", num_samples=100))
    val_data.update(generate_random_images(evaluator, output_path, "val", num_samples=100))
        
    with open(output_path / "train" / "scores.json", "w") as f:
        json.dump(train_data, f)

    with open(output_path / "val" / "scores.json", "w") as f:
        json.dump(val_data, f)
        


torch.set_float32_matmul_precision('medium')


def get_transform(size: int = 224, split: str = "train"):
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ]
    )


class ScoreDataset(Dataset):
    def __init__(
        self,
        data_path: str | Path = "./score_data",
        split: str = "train",
        transform: transforms.Compose | None = None,
    ):
        self.data_path = Path(data_path).resolve() / split
        self.images = sorted(list((self.data_path / "images").glob("*.png")))
        self.scores = json.load(open(self.data_path / "scores.json"))
        self.transform = transform

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        score = self.scores[self.images[idx].stem]

        if self.transform:
            image = self.transform(image)

        return image, score

    def __len__(self):
        return len(self.images)



class AestheticPredictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)


class LitScore(L.LightningModule):
    def __init__(self):
        super().__init__()
        # self.loss_fn = torch.nn.MSELoss()
        self.loss_fn = torch.nn.L1Loss()
    
        self.predictor_model_path = '/kaggle/input/sac-logos-ava1-l14-linearmse/sac+logos+ava1-l14-linearMSE.pth'
        self.clip_model_path = '/kaggle/input/openai-clip-vit-large-patch14/ViT-L-14.pt'

        scripted_model = torch.jit.load(self.clip_model_path)
        
        state_dict = {}
        for name, param in scripted_model.named_parameters():
            state_dict[name] = param
        
        for name, buffer in scripted_model.named_buffers():
            state_dict[name] = buffer
        
        self.clip_model = CLIP(
            embed_dim=768,
            image_resolution=224,
            vision_layers=24,
            vision_width=1024,
            vision_patch_size=14,
            context_length=77,
            vocab_size=49408,
            transformer_width=768,
            transformer_layers=12,
            transformer_heads=12
        )
        self.clip_model.load_state_dict(state_dict, strict=False)
        
        self.predictor_model = AestheticPredictor(input_size=768)
        self.predictor_model.load_state_dict(torch.load(self.predictor_model_path, weights_only=False))
        self.predictor_model.eval()
        self.predictor_model.requires_grad = False


    def forward(self, x):
        image_features = self.clip_model.encode_image(x)
        norm = image_features.norm(dim=-1, keepdim=True)
        image_features_norm = (image_features / norm).float()
        score = self.predictor_model(image_features_norm) / 10.0
        return score

    def training_step(self, batch):
        images, targets = batch
        
        outputs = self(images).squeeze()
        loss = self.loss_fn(outputs, targets.float().squeeze())
        self.log("train_loss", loss, prog_bar=True)
        self.log('lr', self.optimizers().param_groups[0]['lr'], prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch

        outputs = self(images).squeeze()
        loss = self.loss_fn(outputs, targets.float().squeeze())
        self.log("val_loss", loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        clip_params = list(self.clip_model.parameters())
        predictor_params = list(self.predictor_model.parameters())

        lr = 1e-4
        optimizer = torch.optim.AdamW([
            {'params': clip_params, 'lr': lr},
            {'params': predictor_params, 'lr': lr}
        ])
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.1 * lr)
        return [optimizer], [scheduler]


class ScoreData(L.LightningDataModule):
    def prepare_data(self):
        prepare_datasets()
    
    def train_dataloader(self):
        dataset = ScoreDataset(
            data_path="./score_data", split="train", transform=get_transform(224, "train")
        )
        return torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    def val_dataloader(self):
        dataset = ScoreDataset(
            data_path="./score_data", split="val", transform=get_transform(224, "val")
        )
        return torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)


if __name__ == "__main__":
    model = LitScore()
    data = ScoreData()
    
    # Add checkpoint callback to save the best model based on validation loss
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints_score/',
        filename='score-{epoch:02d}-{val_loss:.4f}',
        save_top_k=1,
        mode='min',
    )
    
    trainer = L.Trainer(
        max_epochs=10, 
        log_every_n_steps=50,
        callbacks=[checkpoint_callback],
        gradient_clip_val=1.0,
        # accumulate_grad_batches=4
    )
    
    trainer.fit(model, data)
    
    # Print the path to the best model
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")
    print(f"Best validation loss: {checkpoint_callback.best_model_score:.4f}")
