
import csv
from pathlib import Path

import torch
from PIL import Image, ImageOps

from model import PlantDiseaseModel
from dataset import eval_transform

ROOT = Path(__file__).resolve().parents[2]

MANIFEST = ROOT / "datasets/metadata/master_image_manifest.csv"
CHECKPOINT = ROOT / "server/ai/mobilenetv2_phase2_best.pth"

ACTUAL_CLASS = "garlic__garlic_leaf_blight"
TARGET_PREDICTION = ACTUAL_CLASS
device = torch.device("cpu")


def main():
    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=True
    )

    classes = checkpoint["classes"]

    model = PlantDiseaseModel(num_classes=len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    samples = [
        row for row in rows
        if row["Classification_Action"] == "KEEP"
        and row["Split"] == "test"
        and row["Class"] == ACTUAL_CLASS
    ]

    print("Checking", len(samples), "test images...")

    for row in samples:
        image_path = ROOT / row["Image"]

        with Image.open(image_path) as original:
            image = ImageOps.exif_transpose(original).convert("RGB")
            tensor = eval_transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            probabilities = torch.softmax(model(tensor), dim=1)[0]

        predicted_idx = probabilities.argmax().item()
        predicted_class = classes[predicted_idx]

        if predicted_class == TARGET_PREDICTION:
            confidence = probabilities[predicted_idx].item() * 100

            print("\n========== MATCH FOUND ==========")
            print("Image:", row["Image"])
            print("Actual:", ACTUAL_CLASS)
            print("Predicted:", predicted_class)
            print("Confidence:", round(confidence, 2), "%")
            return

    print("No matching image found.")


if __name__ == "__main__":
    main()
