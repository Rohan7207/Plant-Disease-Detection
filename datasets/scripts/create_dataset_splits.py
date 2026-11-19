
import csv
import random
from collections import defaultdict, Counter
from pathlib import Path

ROOT = Path.cwd()
manifest_path = ROOT / "datasets" / "metadata" / "master_image_manifest.csv"

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

with manifest_path.open("r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    rows = list(reader)

# Group only retained images from datasets without official splits.
groups = defaultdict(list)

for row in rows:
    if (
        row["Classification_Action"] == "KEEP"
        and row["Dataset"] in {"RiceSeg", "Cotton", "Chilli"}
    ):
        groups[(row["Dataset"], row["Class"])].append(row)

rng = random.Random(SEED)
split_counts = Counter()

for (dataset, class_name), images in sorted(groups.items()):
    rng.shuffle(images)

    total = len(images)
    train_count = int(total * TRAIN_RATIO)
    val_count = int(total * VAL_RATIO)

    # Ensure all three splits contain at least one image.
    train_count = max(1, train_count)
    val_count = max(1, val_count)

    if train_count + val_count >= total:
        raise ValueError(
            f"Not enough images for three splits: {class_name} ({total})"
        )

    for i, row in enumerate(images):
        if i < train_count:
            row["Split"] = "train"
        elif i < train_count + val_count:
            row["Split"] = "val"
        else:
            row["Split"] = "test"

        split_counts[(dataset, row["Split"])] += 1

# Preserve PlantSeg official splits and leave excluded records unchanged.
with manifest_path.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print("Dataset splits created successfully.")
print(f"Random seed: {SEED}")
print("Split ratios: 70% / 15% / 15%")

print("\nSplit counts:")
for (dataset, split), count in sorted(split_counts.items()):
    print(f"{dataset} | {split}: {count}")

print("\nClass-level split verification:")
for (dataset, class_name), images in sorted(groups.items()):
    counts = Counter(row["Split"] for row in images)
    print(
        f"{class_name}: "
        f"train={counts['train']}, "
        f"val={counts['val']}, "
        f"test={counts['test']}"
    )
