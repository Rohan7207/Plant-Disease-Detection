
import csv
import hashlib
from collections import defaultdict
from pathlib import Path

BASE = Path("datasets/raw/PlantSeg/plantseg")
METADATA = Path("datasets/metadata/PlantSeg/Metadata.csv")
OUTPUT = Path("datasets/metadata/PlantSeg/label_conflicts.csv")

groups = defaultdict(list)

with METADATA.open("r", encoding="utf-8-sig", newline="") as f:
    for row in csv.DictReader(f):
        image_path = BASE / row["Name"]

        if not image_path.is_file():
            continue

        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()

        groups[digest].append({
            "Name": row["Name"],
            "Plant": row["Plant"],
            "Disease": row["Disease"],
            "Split": row["Split"]
        })

conflicts = []

for digest, files in groups.items():
    labels = {(item["Plant"], item["Disease"]) for item in files}

    if len(labels) > 1:
        for item in files:
            conflicts.append({
                "SHA256": digest,
                **item
            })

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["SHA256", "Name", "Plant", "Disease", "Split"]
    )
    writer.writeheader()
    writer.writerows(conflicts)

print("Conflicting duplicate groups:", len({
    item["SHA256"] for item in conflicts
}))
print("Files involved:", len(conflicts))
print("Report saved:", OUTPUT.resolve())
