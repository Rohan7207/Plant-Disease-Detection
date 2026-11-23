
import csv
import torch
from pathlib import Path
from collections import defaultdict

from model import PlantDiseaseModel
from dataset import PlantDiseaseDataset, eval_transform, PROJECT_ROOT
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report
)


DEVICE = torch.device("cpu")
CHECKPOINT_PATH = PROJECT_ROOT / "server" / "ai" / "mobilenetv2_phase2_best.pth"
REPORT_PATH = PROJECT_ROOT / "datasets" / "metadata" / "phase2_test_report.csv"


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

    assert dataset.classes == checkpoint["classes"], "Class mapping mismatch!"

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0
    )

    model = PlantDiseaseModel(num_classes=len(dataset.classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    y_true = []
    y_pred = []

    print("Evaluating Phase 2 on test set...")
    print("Test images:", len(dataset))
    print("Classes:", len(dataset.classes))

    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(loader):

            images = images.to(DEVICE)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            y_true.extend(labels.tolist())
            y_pred.extend(predictions.tolist())

            if (batch_index + 1) % 25 == 0:
                print(
                    f"Processed {(batch_index + 1) * 16}/"
                    f"{len(dataset)} images"
                )

    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(dataset.classes))),
        zero_division=0
    )

    represented = [i for i, count in enumerate(support) if count > 0]

    macro_precision = precision[represented].mean()
    macro_recall = recall[represented].mean()
    macro_f1 = f1[represented].mean()

    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )
    )

    print("\n========== PHASE 2 TEST RESULTS ==========")
    print("Test images:", len(y_true))
    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("Macro precision:", round(macro_precision, 4))
    print("Macro recall:", round(macro_recall, 4))
    print("Macro F1:", round(macro_f1, 4))
    print("Weighted precision:", round(weighted_precision, 4))
    print("Weighted recall:", round(weighted_recall, 4))
    print("Weighted F1:", round(weighted_f1, 4))
    print("Classes represented:", len(represented))
    print("Classes without test samples:", len(dataset.classes) - len(represented))

    # Save per-class report.
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(dataset.classes))),
        target_names=dataset.classes,
        output_dict=True,
        zero_division=0
    )

    with REPORT_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Class", "Precision", "Recall", "F1", "Support"
        ])

        for class_name in dataset.classes:
            row = report[class_name]
            writer.writerow([
                class_name,
                round(row["precision"], 4),
                round(row["recall"], 4),
                round(row["f1-score"], 4),
                int(row["support"])
            ])

    print("\nPer-class report saved to:")
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
