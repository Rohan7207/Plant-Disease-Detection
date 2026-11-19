
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import Counter

from dataset import PlantDiseaseDataset
from model import PlantDiseaseModel


# Configuration
DEVICE = torch.device("cpu")
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 0.001

# Load datasets
train_dataset = PlantDiseaseDataset("train")
val_dataset = PlantDiseaseDataset("val")

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

# Calculate class weights using training data only.
class_counts = Counter(
    row["Class"] for row in train_dataset.samples
)

counts = torch.tensor(
    [class_counts[class_name] for class_name in train_dataset.classes],
    dtype=torch.float32
)

class_weights = counts.sum() / (num_classes * counts)
class_weights = class_weights.to(DEVICE)

# Initialize model
model = PlantDiseaseModel(num_classes=num_classes).to(DEVICE)

# Freeze feature extractor
for parameter in model.model.features.parameters():
    parameter.requires_grad = False

# Loss and optimizer
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = torch.optim.Adam(
    model.model.classifier.parameters(),
    lr=LEARNING_RATE
)

# Training loop
for epoch in range(EPOCHS):

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_accuracy = 100 * correct / total

    # Validation
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

            val_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)
            val_correct += (predictions == labels).sum().item()
            val_total += labels.size(0)

    val_loss /= val_total
    val_accuracy = 100 * val_correct / val_total

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Accuracy: {train_accuracy:.2f}%")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.2f}%")

# Save trained model
save_path = "server/ai/mobilenetv2_phase1.pth"

torch.save({
    "model_state_dict": model.state_dict(),
    "classes": train_dataset.classes
}, save_path)

print(f"\nModel saved: {save_path}")
