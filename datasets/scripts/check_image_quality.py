
from pathlib import Path
from collections import Counter
from PIL import Image, UnidentifiedImageError
import csv

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ROOT / "datasets" / "raw"
REPORT_DIR = ROOT / "datasets" / "metadata"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Each path points to a folder containing class folders or images.
SOURCES = {
    "PlantSeg": DATASETS / "PlantSeg" / "plantseg" / "images",
    "RiceSeg": DATASETS / "RiceSeg" / "images" / "Rice Leaf Disease Images",
    "Cotton": DATASETS / "Cotton" / "Cotton_Original_Dataset",
    "Chilli": DATASETS / "Chilli" / "Chili Leaf Disease Original Dataset",
}

summary = []
bad_images = []
dimension_counts = {}

for dataset_name, folder in SOURCES.items():
    print(f"\nChecking {dataset_name}...")

    if not folder.exists():
        print(f"Folder not found: {folder}")
        continue

    total = 0
    valid = 0
    corrupted = 0
    dimensions = Counter()

    for path in folder.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue

        total += 1

        try:
            with Image.open(path) as img:
                img.verify()

            # Reopen after verify to read dimensions.
            with Image.open(path) as img:
                dimensions[f"{img.width}x{img.height}"] += 1

            valid += 1

        except (UnidentifiedImageError, OSError, ValueError) as error:
            corrupted += 1
            bad_images.append([
                dataset_name,
                str(path.relative_to(ROOT)),
                str(error)
            ])

    summary.append([
        dataset_name,
        total,
        valid,
        corrupted,
        len(dimensions)
    ])

    dimension_counts[dataset_name] = dimensions

    print(f"Total: {total}")
    print(f"Valid: {valid}")
    print(f"Corrupted: {corrupted}")
    print(f"Unique dimensions: {len(dimensions)}")

# Save summary report.
with open(REPORT_DIR / "image_quality_summary.csv",
          "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow([
        "Dataset", "Total Images", "Valid Images",
        "Corrupted Images", "Unique Dimensions"
    ])
    writer.writerows(summary)

# Save corrupted image report.
with open(REPORT_DIR / "corrupted_images.csv",
          "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Dataset", "File", "Error"])
    writer.writerows(bad_images)

# Save dimension report.
with open(REPORT_DIR / "image_dimensions.csv",
          "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Dataset", "Dimensions", "Image Count"])

    for dataset_name, dimensions in dimension_counts.items():
        for dimension, count in dimensions.most_common():
            writer.writerow([dataset_name, dimension, count])

print("\n========== QUALITY CHECK COMPLETE ==========")
print(f"Reports saved in: {REPORT_DIR}")
