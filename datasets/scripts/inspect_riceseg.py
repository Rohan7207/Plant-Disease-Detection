from pathlib import Path
import csv
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]

images_root = ROOT / "datasets/raw/RiceSeg/images/Rice Leaf Disease Images"
masks_root = ROOT / "datasets/raw/RiceSeg/masks/mask-vesrion1"
manifest = ROOT / "datasets/metadata/master_image_manifest.csv"

print("=== RICESEG IMAGE COUNTS ===")
for folder in sorted(images_root.iterdir()):
    if folder.is_dir():
        count = sum(1 for p in folder.iterdir() if p.is_file())
        print(f"{folder.name}: {count}")

print("\n=== RICESEG MASK COUNTS ===")
for folder in sorted(masks_root.iterdir()):
    if folder.is_dir():
        count = sum(1 for p in folder.iterdir() if p.is_file())
        print(f"{folder.name}: {count}")

print("\n=== RICESEG CLASSES IN MASTER MANIFEST ===")

classes = Counter()

with open(manifest, newline="", encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        if row["Dataset"].strip().lower() == "riceseg":
            classes[row["Class"]] += 1

for name, count in sorted(classes.items()):
    print(f"{name}: {count}")

print("\nTotal manifest RiceSeg records:", sum(classes.values()))