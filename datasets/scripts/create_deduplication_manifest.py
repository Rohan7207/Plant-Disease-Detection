
import csv
from collections import defaultdict, Counter
from pathlib import Path

DUPLICATES = Path("datasets/metadata/duplicate_images.csv")
CONFLICTS = Path("datasets/metadata/conflicting_duplicates.csv")
PLANTSEG_CONFLICTS = Path("datasets/metadata/PlantSeg/label_conflicts.csv")
OUTPUT = Path("datasets/metadata/deduplication_manifest.csv")

groups = defaultdict(list)
conflicting_files = set()

# Load Cotton and Chilli conflicts
with CONFLICTS.open("r", encoding="utf-8-sig", newline="") as f:
    for row in csv.DictReader(f):
        conflicting_files.add((row["Dataset"], row["File"]))

# Load PlantSeg conflicts
with PLANTSEG_CONFLICTS.open("r", encoding="utf-8-sig", newline="") as f:
    for row in csv.DictReader(f):
        conflicting_files.add(("PlantSeg", row["File"]))

# Load exact duplicate groups
with DUPLICATES.open("r", encoding="utf-8-sig", newline="") as f:
    for row in csv.DictReader(f):
        groups[(row["Dataset"], row["SHA256"])].append(row["File"])

manifest = []

split_priority = {
    "test": 0,
    "val": 1,
    "train": 2
}

for (dataset, digest), files in groups.items():

    # Exclude every file in a conflicting group
    if any((dataset, file_path) in conflicting_files for file_path in files):
        for file_path in files:
            manifest.append({
                "Dataset": dataset,
                "SHA256": digest,
                "File": file_path,
                "Action": "EXCLUDE_CONFLICT"
            })
        continue

    # Determine the preferred copy
    if dataset == "PlantSeg":
        def get_split(file_path):
            parts = file_path.replace("\\", "/").split("/")
            return next(
                (part for part in parts if part in split_priority),
                None
            )

        selected = min(
            files,
            key=lambda path: (
                split_priority.get(get_split(path), 99),
                path
            )
        )
    else:
        selected = sorted(files)[0]

    # Keep one copy, exclude the remaining copies
    for file_path in files:
        action = "KEEP" if file_path == selected else "EXCLUDE_DUPLICATE"

        manifest.append({
            "Dataset": dataset,
            "SHA256": digest,
            "File": file_path,
            "Action": action
        })

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["Dataset", "SHA256", "File", "Action"]
    )
    writer.writeheader()
    writer.writerows(manifest)

counts = Counter(row["Action"] for row in manifest)

print("Manifest updated:", OUTPUT.resolve())
print("Files in duplicate groups:", len(manifest))
print("Keep:", counts["KEEP"])
print("Exclude duplicates:", counts["EXCLUDE_DUPLICATE"])
print("Exclude conflicts:", counts["EXCLUDE_CONFLICT"])
