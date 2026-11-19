
import csv
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import PlantDiseaseDataset
from model import PlantDiseaseModel


# -------------------- CONFIGURATION --------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PHASE1_CHECKPOINT = (
    PROJECT_ROOT / "server" / "ai" / "mobilenetv2_phase1.pth"
)

PHASE2_CHECKPOINT = (
    PROJECT_ROOT / "server" / "ai" / "mobilenetv2_phase2_best.pth"
)

HISTORY_PATH = (
    PROJECT_ROOT / "datasets" / "metadata" / "phase2_training_history.csv"
)

BATCH_SIZE = 16
EPOCHS = 3

FEATURES_LR = 0.00001
CLASSIFIER_LR = 0.0001

UNFREEZE_FROM = 16

DEVICE = torch.device("cpu")


# -------------------- DATASETS --------------------

print("Loading datasets...")

train_dataset = PlantDiseaseDataset(split="train")
val_dataset = PlantDiseaseDataset(split="val")

assert train_dataset.classes == val_dataset.classes

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

num_classes = len(train_dataset.classes)

print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Classes:", num_classes)


# -------------------- MODEL --------------------

print("\nLoading Phase 1 checkpoint...")

checkpoint = torch.load(
    PHASE1_CHECKPOINT,
    map_location=DEVICE,
    weights_only=False
)

assert checkpoint["classes"] == train_dataset.classes, (
    "Class ordering does not match Phase 1 checkpoint!"
)

model = PlantDiseaseModel(num_classes=num_classes).to(DEVICE)

model.load_state_dict(checkpoint["model_state_dict"])


# Freeze every parameter first.
for parameter in model.parameters():
    parameter.requires_grad = False

# Unfreeze the final MobileNetV2 feature blocks.
for block in model.model.features[UNFREEZE_FROM:]:
    for parameter in block.parameters():
        parameter.requires_grad = True

# Unfreeze the classification head.
for parameter in model.model.classifier.parameters():
    parameter.requires_grad = True

trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print("\n========== FINE-TUNING CONFIGURATION ==========")
print("Unfrozen feature blocks:", f"{UNFREEZE_FROM} to 18")
print("Trainable parameters:", f"{trainable_parameters:,}")
print("Total parameters:", f"{total_parameters:,}")
print("Feature learning rate:", FEATURES_LR)
print("Classifier learning rate:", CLASSIFIER_LR)


# -------------------- CLASS WEIGHTS --------------------

class_counts = Counter(
    row["Class"] for row in train_dataset.samples
)

counts = torch.tensor(
    [class_counts[class_name] for class_name in train_dataset.classes],
    dtype=torch.float32
)

class_weights = counts.sum() / (num_classes * counts)
class_weights = class_weights.to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights)


# -------------------- OPTIMIZER --------------------

feature_parameters = [
    parameter
    for parameter in model.model.features.parameters()
    if parameter.requires_grad
]

classifier_parameters = [
    parameter
    for parameter in model.model.classifier.parameters()
    if parameter.requires_grad
]

optimizer = torch.optim.Adam([
    {
        "params": feature_parameters,
        "lr": FEATURES_LR
    },
    {
        "params": classifier_parameters,
        "lr": CLASSIFIER_LR
    }
])


# -------------------- TRAINING --------------------

history = []
best_val_loss = float("inf")

print("\nStarting Phase 2 fine-tuning...\n")

for epoch in range(EPOCHS):

    model.train()

    # Keep all frozen feature blocks and their BatchNorm
    # running statistics unchanged.
    model.model.features[:UNFREEZE_FROM].eval()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        predictions = outputs.argmax(dim=1)

        train_correct += (predictions == labels).sum().item()
        train_total += labels.size(0)

    train_loss /= len(train_loader)
    train_accuracy = 100 * train_correct / train_total

    # -------------------- VALIDATION --------------------

    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            val_correct += (predictions == labels).sum().item()
            val_total += labels.size(0)

    val_loss /= len(val_loader)
    val_accuracy = 100 * val_correct / val_total

    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Accuracy: {train_accuracy:.2f}%")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.2f}%")
    print("-" * 45)

    history.append({
        "Epoch": epoch + 1,
        "Train_Loss": train_loss,
        "Train_Accuracy": train_accuracy,
        "Validation_Loss": val_loss,
        "Validation_Accuracy": val_accuracy
    })

    # Save the best checkpoint based on validation loss.
    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save({
            "model_state_dict": model.state_dict(),
            "classes": train_dataset.classes,
            "epoch": epoch + 1,
            "validation_loss": val_loss,
            "validation_accuracy": val_accuracy,
            "phase": 2
        }, PHASE2_CHECKPOINT)

        print("Best Phase 2 checkpoint saved!")


# -------------------- SAVE TRAINING HISTORY --------------------

with open(HISTORY_PATH, "w", newline="", encoding="utf-8") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "Epoch",
            "Train_Loss",
            "Train_Accuracy",
            "Validation_Loss",
            "Validation_Accuracy"
        ]
    )

    writer.writeheader()
    writer.writerows(history)


print("\n========== PHASE 2 COMPLETED ==========")
print("Best validation loss:", round(best_val_loss, 4))
print("Checkpoint:", PHASE2_CHECKPOINT)
print("Training history:", HISTORY_PATH)
