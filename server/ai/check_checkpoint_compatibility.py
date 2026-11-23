
import csv
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

MANIFEST = ROOT / "datasets/metadata/master_image_manifest.csv"
CHECKPOINT = ROOT / "server/ai/mobilenetv2_phase2_best.pth"

with MANIFEST.open("r", newline="", encoding="utf-8") as file:
    rows = list(csv.DictReader(file))

manifest_classes = sorted({
    row["Class"]
    for row in rows
    if row["Classification_Action"] == "KEEP"
})

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=False
)

checkpoint_classes = checkpoint["classes"]

print("========== CHECKPOINT COMPATIBILITY ==========")
print("Manifest classes:", len(manifest_classes))
print("Checkpoint classes:", len(checkpoint_classes))
print("Same class order:", manifest_classes == checkpoint_classes)

missing = set(checkpoint_classes) - set(manifest_classes)
new = set(manifest_classes) - set(checkpoint_classes)

print("Missing from manifest:", sorted(missing))
print("New in manifest:", sorted(new))

if manifest_classes == checkpoint_classes:
    print("\nRESULT: COMPATIBLE")
else:
    print("\nRESULT: CLASS MAPPING MISMATCH")

