
import csv
import shutil
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]

MANIFEST = ROOT / "datasets/metadata/master_image_manifest.csv"
BACKUP = ROOT / "datasets/metadata/master_image_manifest_before_leaf_review.csv"

# Confirmed non-leaf images
EXCLUDE = {
    "datasets/raw/PlantSeg/plantseg/images/test/blueberry_anthracnose_Bing_0145.jpg",
    "datasets/raw/PlantSeg/plantseg/images/test/carrot_cavity_spot_google_0140.jpg",
    "datasets/raw/PlantSeg/plantseg/images/test/grape_black_rot_google_0320.jpg",
    "datasets/raw/PlantSeg/plantseg/images/train/peach_brown_rot_Bing_0166.jpg",
    "datasets/raw/PlantSeg/plantseg/images/val/grape_black_rot_141.jpg",
    "datasets/raw/PlantSeg/plantseg/images/val/wheat_head_scab_Baidu_0394.jpg",
}

# Uncertain image: exclude from training until reviewed
REVIEW = {
    "datasets/raw/PlantSeg/plantseg/images/train/grapevine_leafroll_disease_google_0184.jpg",
}

with MANIFEST.open("r", newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    fieldnames = reader.fieldnames
    rows = list(reader)

if "Classification_Action" not in fieldnames:
    raise ValueError("Classification_Action column not found")

found = Counter(row["Image"] for row in rows)

missing = (EXCLUDE | REVIEW) - set(found)
if missing:
    raise ValueError(f"Images not found in manifest: {sorted(missing)}")

if any(found[path] != 1 for path in EXCLUDE | REVIEW):
    raise ValueError("An image path occurs more than once in the manifest")

# Create a backup before modifying the manifest
if not BACKUP.exists():
    shutil.copy2(MANIFEST, BACKUP)

for row in rows:
    if row["Image"] in EXCLUDE:
        row["Classification_Action"] = "EXCLUDE_NON_LEAF"
    elif row["Image"] in REVIEW:
        row["Classification_Action"] = "REVIEW"

# Save updated manifest
with MANIFEST.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

counts = Counter(row["Classification_Action"] for row in rows)

print("========== LEAF-ONLY MANIFEST UPDATE ==========")
print("Confirmed non-leaf exclusions:", len(EXCLUDE))
print("Images marked for review:", len(REVIEW))
print("Backup:", BACKUP)
print("\nClassification action counts:")

for action, count in sorted(counts.items()):
    print(f"{action}: {count}")

print("\nUpdated manifest:", MANIFEST)
