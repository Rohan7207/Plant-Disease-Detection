
import csv
from collections import defaultdict, Counter

import torch
from torch.utils.data import DataLoader

from dataset import PlantDiseaseDataset, eval_transform
from model import PlantDiseaseModel

REPORT_PATH = "datasets/metadata/phase1_validation_report.csv"
CHECKPOINT_PATH = "server/ai/mobilenetv2_phase1.pth"

device = torch.device("cpu")

# Get the 20 weakest classes with validation samples.
with open(REPORT_PATH, newline="", encoding="utf-8") as file:
    report = list(csv.DictReader(file))

weak_classes = {
    row["Class"]
    for row in sorted(
        [r for r in report if int(r["Support"]) > 0],
        key=lambda r: float(r["F1_Score"])
    )[:20]
}

dataset = PlantDiseaseDataset("val", transform=eval_transform)
loader = DataLoader(dataset, batch_size=16, shuffle=False)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=False
)

model = PlantDiseaseModel(num_classes=len(checkpoint["classes"]))
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

assert dataset.classes == checkpoint["classes"]

confusions = defaultdict(Counter)

with torch.no_grad():
    for images, labels in loader:
        images = images.to(device)
        predictions = model(images).argmax(dim=1)

        for true_idx, pred_idx in zip(labels.tolist(), predictions.tolist()):
            true_class = dataset.idx_to_class[true_idx]
            pred_class = dataset.idx_to_class[pred_idx]

            if true_class in weak_classes and true_class != pred_class:
                confusions[true_class][pred_class] += 1

print("========== WEAK CLASS CONFUSIONS ==========")

for true_class in sorted(weak_classes):
    predictions = confusions[true_class]

    print(f"\n{true_class}")
    if not predictions:
        print("No incorrect predictions recorded.")
        continue

    for predicted_class, count in predictions.most_common(3):
        print(f"  Predicted as {predicted_class}: {count}")
