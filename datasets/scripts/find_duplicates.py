
from pathlib import Path
from collections import defaultdict
from PIL import Image
import hashlib
import csv

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ROOT / "datasets" / "raw"
REPORT = ROOT / "datasets" / "metadata" / "duplicate_images.csv"

SOURCES = {
    "PlantSeg": DATASETS / "PlantSeg" / "plantseg" / "images",
    "RiceSeg": DATASETS / "RiceSeg" / "images" / "Rice Leaf Disease Images",
    "Cotton": DATASETS / "Cotton" / "Cotton_Original_Dataset",
    "Chilli": DATASETS / "Chilli" / "Chili Leaf Disease Original Dataset",
}

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

hashes = defaultdict(list)

for dataset_name, folder in SOURCES.items():
    print(f"\nScanning {dataset_name}...")

    for path in folder.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue

        # Hash the original file bytes.
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes[(dataset_name, digest)].append(path)

duplicates = []

for (dataset_name, digest), paths in hashes.items():
    if len(paths) > 1:
        for path in paths:
            duplicates.append([
                dataset_name,
                digest,
                str(path.relative_to(ROOT))
            ])

with open(REPORT, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Dataset", "SHA256", "File"])
    writer.writerows(duplicates)

print("\n========== DUPLICATE CHECK COMPLETE ==========")
print(f"Duplicate files found: {len(duplicates)}")
print(f"Report saved: {REPORT}")
