
import csv
import torch
from pathlib import Path
from collections import defaultdict
from torch.utils.data import DataLoader

from model import PlantDiseaseModel
from dataset import PlantDiseaseDataset, eval_transform, PROJECT_ROOT


DEVICE = torch.device("cpu")

CHECKPOINT_PATH = (
    PROJECT_ROOT / "server" / "ai" / "mobilenetv2_phase2_best.pth"
)

OUTPUT_PATH = (
    PROJECT_ROOT / "datasets" / "metadata" / "phase2_test_confusions.csv"
)


def main():

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    dataset = PlantDiseaseDataset(
        split="test",
        transform=eval_transform
    )

    classes = checkpoint["classes"]

    assert dataset.classes == classes, "Class mapping mismatch!"

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0
    )

    model = PlantDiseaseModel(num_classes=len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    confusion = defaultdict(int)
    true_support = defaultdict(int)

    total = 0
    correct = 0

    print("Analyzing test-set confusions...")
    print("Test images:", len(dataset))

    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(loader):

            outputs = model(images.to(DEVICE))
            predictions = outputs.argmax(dim=1)

            for actual, predicted in zip(
                labels.tolist(),
                predictions.tolist()
            ):
                confusion[(actual, predicted)] += 1
                true_support[actual] += 1

                total += 1

                if actual == predicted:
                    correct += 1

            if (batch_index + 1) % 25 == 0:
                print(f"Processed {total}/{len(dataset)}")

    # Collect incorrect class pairs.
    errors = []

    for (actual, predicted), count in confusion.items():

        if actual == predicted:
            continue

        actual_class = classes[actual]
        predicted_class = classes[predicted]

        actual_crop = actual_class.split("__", 1)[0]
        predicted_crop = predicted_class.split("__", 1)[0]

        errors.append({
            "Actual_Class": actual_class,
            "Predicted_Class": predicted_class,
            "Count": count,
            "Actual_Class_Support": true_support[actual],
            "Error_Rate_Per_Actual_Class": round(
                count / true_support[actual], 4
            ),
            "Same_Crop": actual_crop == predicted_crop
        })

    errors.sort(
        key=lambda row: row["Count"],
        reverse=True
    )

    # Save all observed confusion pairs.
    with OUTPUT_PATH.open(
        "w", newline="", encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "Actual_Class",
                "Predicted_Class",
                "Count",
                "Actual_Class_Support",
                "Error_Rate_Per_Actual_Class",
                "Same_Crop"
            ]
        )

        writer.writeheader()
        writer.writerows(errors)

    same_crop_errors = sum(
        row["Count"] for row in errors if row["Same_Crop"]
    )

    different_crop_errors = sum(
        row["Count"] for row in errors if not row["Same_Crop"]
    )

    print("\n========== TEST CONFUSION SUMMARY ==========")
    print("Total images:", total)
    print("Correct predictions:", correct)
    print("Incorrect predictions:", total - correct)
    print("Within-crop errors:", same_crop_errors)
    print("Cross-crop errors:", different_crop_errors)

    print("\n========== TOP 30 CONFUSIONS ==========")

    for row in errors[:30]:
        print(
            f'{row["Actual_Class"]} -> '
            f'{row["Predicted_Class"]} | '
            f'Count: {row["Count"]} | '
            f'Class error rate: '
            f'{row["Error_Rate_Per_Actual_Class"]:.2%} | '
            f'Same crop: {row["Same_Crop"]}'
        )

    print("\nFull report saved to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()

