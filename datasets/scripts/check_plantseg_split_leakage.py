
import csv
from collections import defaultdict, Counter
from pathlib import Path

MANIFEST = Path("datasets/metadata/deduplication_manifest.csv")
OUTPUT = Path("datasets/metadata/PlantSeg/remaining_split_leakage.csv")

groups = defaultdict(list)
valid_splits = {"train", "val", "test"}

with MANIFEST.open("r", encoding="utf-8-sig", newline="") as f:
    for row in csv.DictReader(f):
        if row["Dataset"] != "PlantSeg":
            continue

        if row["Action"] == "EXCLUDE_CONFLICT":
            continue

        normalized = row["File"].replace("\\", "/")
        parts = normalized.split("/")

        split = next(
            (part for part in parts if part in valid_splits),
            None
        )

        if split is None:
            print("WARNING: Could not identify split:", row["File"])
            continue

        groups[row["SHA256"]].append({
            "SHA256": row["SHA256"],
            "Split": split,
            "File": row["File"],
            "Action": row["Action"]
        })

leakage_groups = {
    digest: files
    for digest, files in groups.items()
    if len({item["Split"] for item in files}) > 1
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["SHA256", "Split", "File", "Action"]
    )
    writer.writeheader()

    for files in leakage_groups.values():
        writer.writerows(files)

combinations = Counter()
total_files = 0

for files in leakage_groups.values():
    splits = sorted({item["Split"] for item in files})
    combinations[";".join(splits)] += 1
    total_files += len(files)

print("Remaining cross-split duplicate groups:", len(leakage_groups))
print("Files involved:", total_files)
print("Split combinations:")

for combination, count in combinations.most_common():
    print(f"{combination}: {count} groups")

print("Report saved:", OUTPUT.resolve())
