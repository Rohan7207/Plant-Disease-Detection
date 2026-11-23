import csv
import torch
from pathlib import Path

from dataset import PlantDiseaseDataset

ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT = ROOT / "server/ai/mobilenetv2_phase2_best.pth"
MANIFEST = ROOT / "datasets/metadata/master_image_manifest.csv"

# Load checkpoint classes
checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=True
)

checkpoint_classes = checkpoint["classes"]

# Load the existing dataset mapping
dataset = PlantDiseaseDataset("test")

print("Checkpoint classes:", len(checkpoint_classes))
print("Dataset classes:", len(dataset.classes))

print(
    "Class mappings identical:",
    checkpoint_classes == dataset.classes
)

with open(MANIFEST, encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

eligible = [
    row for row in rows
    if row["Dataset"] == "PlantSeg"
    and row["Split"] == "test"
    and row["Classification_Action"] == "KEEP"
    and row["Segmentation_Action"] == "AVAILABLE"
]

missing_images = []
missing_masks = []
missing_classes = []

for row in eligible:
    if not (ROOT / row["Image"]).exists():
        missing_images.append(row["Image"])

    if not (ROOT / row["Mask"]).exists():
        missing_masks.append(row["Mask"])

    if row["Class"] not in checkpoint_classes:
        missing_classes.append(row["Class"])

print("\nEligible images:", len(eligible))
print("Missing images:", len(missing_images))
print("Missing masks:", len(missing_masks))
print("Missing classes:", len(missing_classes))

if missing_images:
    print("Example missing image:", missing_images[0])

if missing_masks:
    print("Example missing mask:", missing_masks[0])

if missing_classes:
    print("Example missing class:", missing_classes[0])