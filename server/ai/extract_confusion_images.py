import csv
from pathlib import Path

import torch
from PIL import Image, ImageOps

from model import PlantDiseaseModel
from dataset import eval_transform

ROOT = Path(__file__).resolve().parents[2]

MANIFEST = ROOT / "datasets/metadata/master_image_manifest.csv"
CHECKPOINT = ROOT / "server/ai/mobilenetv2_phase2_best.pth"
OUTPUT = ROOT / "datasets/metadata/squash_zucchini_errors.csv"

ACTUAL = "squash__squash_powdery_mildew"
PREDICTED = "zucchini__zucchini_powdery_mildew"


def main():
    device = torch.device("cpu")

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

    predicted_index = classes.index(PREDICTED)

    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as file:
        rows = [
            row for row in csv.DictReader(file)
            if row["Class"] == ACTUAL
            and row["Split"] == "test"
            and row["Classification_Action"] == "KEEP"
        ]

    results = []

    for row in rows:
        image_path = ROOT / row["Image"]

        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            tensor = eval_transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            probabilities = torch.softmax(model(tensor), dim=1)[0]

        prediction = probabilities.argmax().item()

        if prediction == predicted_index:
            results.append({
                "Image": row["Image"],
                "Actual_Class": ACTUAL,
                "Predicted_Class": PREDICTED,
                "Confidence": round(
                    probabilities[prediction].item() * 100, 2
                )
            })

    with OUTPUT.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "Image",
                "Actual_Class",
                "Predicted_Class",
                "Confidence"
            ]
        )
        writer.writeheader()
        writer.writerows(results)

    print("Matching errors:", len(results))
    print("Saved:", OUTPUT)

    for result in results:
        print(result)


if __name__ == "__main__":
    main()