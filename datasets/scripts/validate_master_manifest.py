
import csv
from collections import Counter, defaultdict
from pathlib import Path

path = Path("datasets/metadata/master_image_manifest.csv")

with path.open("r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

retained = [
    r for r in rows
    if r["Classification_Action"] == "KEEP"
]

excluded = [
    r for r in rows
    if r["Classification_Action"] == "EXCLUDE"
]

print("Total records:", len(rows))
print("Retained:", len(retained))
print("Excluded:", len(excluded))

# Check missing splits.
missing_splits = [
    r for r in retained
    if r["Split"] not in {"train", "val", "test"}
]

print("\nRetained records with missing/invalid splits:", len(missing_splits))

# Check PlantSeg official split distribution.
plantseg = [
    r for r in retained
    if r["Dataset"] == "PlantSeg"
]

print("\nPlantSeg split distribution:")
for split, count in sorted(Counter(r["Split"] for r in plantseg).items()):
    print(split, count)

# Check duplicate image paths.
path_counts = Counter(r["Image"] for r in retained)
duplicate_paths = [
    image for image, count in path_counts.items()
    if count > 1
]

print("\nDuplicate retained image paths:", len(duplicate_paths))

# Check that each class has all three splits where expected.
groups = defaultdict(set)

for r in retained:
    groups[(r["Dataset"], r["Class"])].add(r["Split"])

invalid_classes = [
    key for key, splits in groups.items()
    if key[0] != "PlantSeg"
    and splits != {"train", "val", "test"}
]

print("Classes missing one or more splits:", len(invalid_classes))

if invalid_classes:
    for item in invalid_classes:
        print(item, groups[item])

if missing_splits or duplicate_paths or invalid_classes:
    print("\nVALIDATION FAILED")
else:
    print("\nVALIDATION PASSED")

