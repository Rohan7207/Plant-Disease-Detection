
import csv
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = (
    PROJECT_ROOT / "datasets" / "metadata" / "master_image_manifest.csv"
)

IMAGE_SIZE = 224

# Initial preprocessing only.
# Training augmentation will be added separately.

# Training transformations
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.10,
        hue=0.02
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# Validation and testing transformations
eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])



class PlantDiseaseDataset(Dataset):

    def __init__(self, split, transform=None):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split: {split}")

        self.transform = transform or (
            train_transform if split == "train" else eval_transform
        )

        with MANIFEST_PATH.open(
            "r", encoding="utf-8-sig", newline=""
        ) as file:
            rows = list(csv.DictReader(file))

        self.samples = [
            row for row in rows
            if row["Classification_Action"] == "KEEP"
            and row["Split"] == split
        ]

        if not self.samples:
            raise ValueError(f"No images found for split: {split}")

        # One consistent label mapping for all splits.
        self.classes = sorted({
            row["Class"] for row in rows
            if row["Classification_Action"] == "KEEP"
        })

        self.class_to_idx = {
            class_name: index
            for index, class_name in enumerate(self.classes)
        }

        self.idx_to_class = {
            index: class_name
            for class_name, index in self.class_to_idx.items()
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        row = self.samples[index]

        image_path = PROJECT_ROOT / row["Image"]

        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image)
            image = image.convert("RGB")

            if self.transform:
                image = self.transform(image)

        label = self.class_to_idx[row["Class"]]

        return image, torch.tensor(label, dtype=torch.long)
