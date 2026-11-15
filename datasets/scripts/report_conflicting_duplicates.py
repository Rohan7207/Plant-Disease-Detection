
from pathlib import Path
from collections import defaultdict
import hashlib
import csv

ROOT = Path(__file__).resolve().parents[2]

SOURCES = {
    "Cotton": ROOT / "datasets/raw/Cotton/Cotton_Original_Dataset",
    "Chilli": ROOT / "datasets/raw/Chilli/Chili Leaf Disease Original Dataset",
}

report = ROOT / "datasets/metadata/conflicting_duplicates.csv"
rows = []

for dataset, folder in SOURCES.items():
    groups = defaultdict(list)

    for path in folder.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {
            ".jpg", ".jpeg", ".png"
        }:
            continue

        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        groups[digest].append(path)

    for digest, paths in groups.items():
        classes = sorted(set(path.parent.name for path in paths))

        if len(classes) > 1:
            for path in paths:
                rows.append([
                    dataset,
                    digest,
                    path.parent.name,
                    str(path.relative_to(ROOT))
                ])

with open(report, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Dataset", "SHA256", "Class", "File"])
    writer.writerows(rows)

print("Conflicting duplicate files:", len(rows))
print("Report:", report)

for dataset in SOURCES:
    count = sum(row[0] == dataset for row in rows)
    print(f"{dataset}: {count} files")
