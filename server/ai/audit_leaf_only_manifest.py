
import csv
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "datasets/metadata/master_image_manifest.csv"

with MANIFEST.open("r", newline="", encoding="utf-8") as file:
    rows = list(csv.DictReader(file))

kept = [r for r in rows if r["Classification_Action"] == "KEEP"]

split_counts = Counter(r["Split"] for r in kept)
class_counts = defaultdict(Counter)

for row in kept:
    class_counts[row["Class"]][row["Split"]] += 1

print("========== LEAF-ONLY DATASET AUDIT ==========")
print("Total retained images:", len(kept))
print("Classes:", len(class_counts))

print("\n========== SPLIT COUNTS ==========")
for split in ("train", "val", "test"):
    print(f"{split}: {split_counts[split]}")

print("\n========== CLASSES WITH LOW TRAINING COUNTS ==========")
for class_name, counts in sorted(class_counts.items()):
    if counts["train"] < 10:
        print(
            f'{class_name}: train={counts["train"]}, '
            f'val={counts["val"]}, test={counts["test"]}'
        )

print("\n========== EMPTY SPLITS ==========")
empty = [
    name for name, counts in class_counts.items()
    if not counts["train"] or not counts["val"] or not counts["test"]
]
print("Classes missing a split:", len(empty))
for name in empty:
    print(name)

