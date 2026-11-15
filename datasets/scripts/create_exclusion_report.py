
from pathlib import Path
import csv
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "datasets/metadata/conflicting_duplicates.csv"
OUTPUT = ROOT / "datasets/metadata/excluded_conflicting_images.csv"

groups = defaultdict(list)

with open(SOURCE, newline="", encoding="utf-8-sig") as file:
    for row in csv.DictReader(file):
        key = (row["Dataset"], row["SHA256"])
        groups[key].append(row)

with open(OUTPUT, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow([
        "Dataset", "SHA256", "Conflicting_Classes", "File"
    ])

    for (dataset, digest), rows in groups.items():
        classes = sorted(set(row["Class"] for row in rows))
        class_names = " | ".join(classes)

        for row in rows:
            writer.writerow([
                dataset,
                digest,
                class_names,
                row["File"]
            ])

print("Conflicting groups:", len(groups))
print("Files excluded:", sum(map(len, groups.values())))
print("Report saved:", OUTPUT)
