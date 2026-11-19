
import csv
import torch
from torch.utils.data import DataLoader
from collections import Counter

from dataset import PlantDiseaseDataset
from model import PlantDiseaseModel


DEVICE = torch.device("cpu")
BATCH_SIZE = 16

CHECKPOINT_PATH = "server/ai/mobilenetv2_phase1.pth"

# Load validation dataset
val_dataset = PlantDiseaseDataset("val")

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# Load checkpoint
checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=True
)

classes = checkpoint["classes"]

assert classes == val_dataset.classes

# Initialize model
model = PlantDiseaseModel(num_classes=len(classes)).to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Evaluating Phase 1 model...")
print("Validation images:", len(val_dataset))
print("Number of classes:", len(classes))

# Predictions
actual = []
predicted = []

with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images.to(DEVICE))
        predictions = outputs.argmax(dim=1)

        actual.extend(labels.tolist())
        predicted.extend(predictions.tolist())

# Confusion matrix
num_classes = len(classes)

matrix = [
    [0] * num_classes
    for _ in range(num_classes)
]

for true_label, predicted_label in zip(actual, predicted):
    matrix[true_label][predicted_label] += 1

# Per-class metrics
report_rows = []

for class_idx, class_name in enumerate(classes):

    true_positive = matrix[class_idx][class_idx]

    false_positive = sum(
        matrix[row][class_idx]
        for row in range(num_classes)
        if row != class_idx
    )

    false_negative = sum(matrix[class_idx]) - true_positive

    support = sum(matrix[class_idx])

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive else 0
    )

    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative else 0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    report_rows.append({
        "Class": class_name,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "Support": support
    })

# Overall metrics
correct = sum(
    matrix[i][i] for i in range(num_classes)
)

accuracy = correct / len(actual)

macro_precision = sum(
    row["Precision"] for row in report_rows
) / num_classes

macro_recall = sum(
    row["Recall"] for row in report_rows
) / num_classes

macro_f1 = sum(
    row["F1_Score"] for row in report_rows
) / num_classes

total_support = sum(row["Support"] for row in report_rows)

weighted_precision = sum(
    row["Precision"] * row["Support"]
    for row in report_rows
) / total_support

weighted_recall = sum(
    row["Recall"] * row["Support"]
    for row in report_rows
) / total_support

weighted_f1 = sum(
    row["F1_Score"] * row["Support"]
    for row in report_rows
) / total_support

print("\n========== OVERALL RESULTS ==========")
print(f"Accuracy:           {accuracy * 100:.2f}%")
print(f"Macro Precision:    {macro_precision:.4f}")
print(f"Macro Recall:       {macro_recall:.4f}")
print(f"Macro F1-score:     {macro_f1:.4f}")
print(f"Weighted Precision: {weighted_precision:.4f}")
print(f"Weighted Recall:    {weighted_recall:.4f}")
print(f"Weighted F1-score:  {weighted_f1:.4f}")

# Save per-class report
report_path = "datasets/metadata/phase1_validation_report.csv"

with open(report_path, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "Class",
            "Precision",
            "Recall",
            "F1_Score",
            "Support"
        ]
    )

    writer.writeheader()
    writer.writerows(report_rows)

print("\nPer-class report saved:", report_path)

# Find top confusions
confusions = []

for actual_idx in range(num_classes):
    for predicted_idx in range(num_classes):

        if actual_idx == predicted_idx:
            continue

        count = matrix[actual_idx][predicted_idx]

        if count > 0:
            confusions.append((
                count,
                classes[actual_idx],
                classes[predicted_idx]
            ))

confusions.sort(reverse=True)

confusion_path = "datasets/metadata/phase1_top_confusions.csv"

with open(confusion_path, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)

    writer.writerow([
        "Actual_Class",
        "Predicted_Class",
        "Incorrect_Predictions"
    ])

    for count, actual_class, predicted_class in confusions[:50]:
        writer.writerow([
            actual_class,
            predicted_class,
            count
        ])

print("\n========== TOP 10 CONFUSIONS ==========")

for count, actual_class, predicted_class in confusions[:10]:
    print(f"{actual_class} -> {predicted_class}: {count}")

print("\nPhase 1 evaluation completed!")
