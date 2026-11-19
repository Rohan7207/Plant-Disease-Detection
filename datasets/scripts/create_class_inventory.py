
import csv
from collections import Counter
from pathlib import Path

BASE = Path("datasets/raw")
MANIFEST = Path("datasets/metadata/deduplication_manifest.csv")
OUTPUT = Path("datasets/metadata/class_inventory.csv")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

excluded = set()

with MANIFEST.open("r", encoding="utf-8-sig", newline="") as f:
    for row in csv.DictReader(f):
        if row["Action"] != "KEEP":
            excluded.add((row["Dataset"], row["File"].replace("\\", "/")))

counts = Counter()

def count_folder(dataset, folder):
    for image in folder.rglob("*"):
        if not image.is_file() or image.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        relative = image.as_posix()

        if (dataset, relative) not in excluded:
            counts[(dataset, folder.name)] += 1

# Cotton
for folder in (BASE / "Cotton" / "Cotton_Original_Dataset").iterdir():
    if folder.is_dir():
        count_folder("Cotton", folder)

# Chilli
for folder in (BASE / "Chilli" / "Chili Leaf Disease Original Dataset").iterdir():
    if folder.is_dir():
        count_folder("Chilli", folder)

# RiceSeg
for folder in (BASE / "RiceSeg" / "images" / "Rice Leaf Disease Images").iterdir():
    if folder.is_dir():
        count_folder("RiceSeg", folder)

# PlantSeg: use Metadata.csv for crop and disease labels
plantseg_metadata = BASE / "PlantSeg" / "plantseg" / "Metadata.csv"

with plantseg_metadata.open("r", encoding="utf-8-sig", newline="") as f:
    for row in csv.DictReader(f):
        split = {
            "Training": "train",
            "Validation": "val",
            "Test": "test"
            }[row["Split"]]
        
        filename = row["Name"]

        relative = f"datasets/raw/PlantSeg/plantseg/images/{split}/{filename}"

        if ("PlantSeg", relative) not in excluded:
            counts[("PlantSeg", f"{row['Plant']} | {row['Disease']}")] += 1

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT.open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Dataset", "Class", "Image_Count"])

    for (dataset, class_name), count in sorted(counts.items()):
        writer.writerow([dataset, class_name, count])

print("Inventory saved:", OUTPUT.resolve())
print("Total classes:", len(counts))
print("\nDataset summaries:")

totals = Counter()

for (dataset, class_name), count in counts.items():
    totals[dataset] += count

for dataset, total in sorted(totals.items()):
    print(f"{dataset}: {total} images, {sum(1 for d, _ in counts if d == dataset)} classes")
