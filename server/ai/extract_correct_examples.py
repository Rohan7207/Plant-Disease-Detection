
import csv
from pathlib import Path

import torch
from PIL import Image, ImageOps

from model import PlantDiseaseModel
from dataset import eval_transform

ROOT = Path(__file__).resolve().parents[2]

MANIFEST = ROOT / "datasets/metadata/master_image_manifest.csv"
CHECKPOINT = ROOT / "server/ai/mobilenetv2_phase2_best.pth"
OUTPUT = ROOT / "datasets/metadata/correct_crop_examples.csv"

TARGET_CLASSES = {
    "squash__squash_powdery_mildew",
    "zucchini__zucchini_powdery_mildew",
}

DEVICE = torch.device("cpu")


def main():
    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=True
    )

    classes = checkpoint["classes"]
    class_to_idx = {name: i for i, name in enumerate(classes)}

    model = PlantDiseaseModel(num_classes=len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    results = []

    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    samples = [
        row for row in rows
        if row["Classification_Action"] == "KEEP"
        and row["Split"] == "test"
        and row["Class"] in TARGET_CLASSES
    ]

    print("Images to inspect:", len(samples))

    for row in samples:
        image_path = ROOT / row["Image"]

        try:
            with Image.open(image_path) as original:
                image = ImageOps.exif_transpose(original).convert("RGB")
                tensor = eval_transform(image).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                probabilities = torch.softmax(model(tensor), dim=1)[0]

            confidence, predicted_idx = probabilities.max(dim=0)
            predicted_class = classes[predicted_idx.item()]

            if predicted_class == row["Class"]:
                results.append({
                    "Image": row["Image"],
                    "Actual_Class": row["Class"],
                    "Predicted_Class": predicted_class,
                    "Confidence": round(confidence.item() * 100, 2),
                })

        except Exception as error:
            print("Skipped:", row["Image"], error)

    results.sort(key=lambda item: (
        item["Actual_Class"],
        -item["Confidence"]
    ))

    with OUTPUT.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "Image", "Actual_Class",
            "Predicted_Class", "Confidence"
        ])
        writer.writeheader()
        writer.writerows(results)

    print("\n========== CORRECT EXAMPLES ==========")

    for class_name in sorted(TARGET_CLASSES):
        matching = [
            item for item in results
            if item["Actual_Class"] == class_name
        ]
        print(f"\n{class_name}: {len(matching)} correctly classified")
        for item in matching[:6]:
            print(item["Image"], "|", item["Confidence"], "%")

    print("\nSaved:", OUTPUT)


if __name__ == "__main__":
    main()
