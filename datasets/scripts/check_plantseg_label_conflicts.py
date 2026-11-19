
import csv
from collections import defaultdict
from pathlib import Path

DUPLICATES = Path("datasets/metadata/duplicate_images.csv")
METADATA = Path("datasets/metadata/PlantSeg/Metadata.csv")
OUTPUT = Path("datasets/metadata/PlantSeg/label_conflicts.csv")

labels = {}

with METADATA.open("r", encoding="utf-8-sig", newline="") as f:
    for row in csv.DictReader(f):
        labels[row["Name"]] = {
            "Plant": row["Plant"],
            "Disease": row["Disease"],
            "Split": row["Split"]
        }

groups = defaultdict(list)

with DUPLICATES.open("r", encoding="utf-8-sig", newline="") as f:
    for row in csv.DictReader(f):
        if row["Dataset"] != "PlantSeg":
            continue

        filename = Path(row["File"]).name

        if filename in labels:
            groups[row["SHA256"]].append({
                "File": row["File"],
                **labels[filename]
            })

conflicts = []

for digest, files in groups.items():
    unique_labels = {
        (item["Plant"], item["Disease"])
        for item in files
    }

    if len(unique_labels) > 1:
        for item in files:
            conflicts.append({
                "SHA256": digest,
                **item
            })

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "SHA256", "File", "Plant", "Disease", "Split"
        ]
    )
    writer.writeheader()
    writer.writerows(conflicts)

print("Conflicting groups:", len({
    row["SHA256"] for row in conflicts
}))
print("Files involved:", len(conflicts))
print("Report saved:", OUTPUT.resolve())
