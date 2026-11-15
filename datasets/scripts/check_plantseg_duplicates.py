
from pathlib import Path
from collections import defaultdict, Counter
import hashlib
import csv

ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIR = ROOT / "datasets/raw/PlantSeg/plantseg/images"
REPORT = ROOT / "datasets/metadata/plantseg_duplicate_split_report.csv"

hash_groups = defaultdict(list)

for path in IMAGE_DIR.rglob("*"):
    if not path.is_file() or path.suffix.lower() not in {
        ".jpg", ".jpeg", ".png", ".bmp", ".webp"
    }:
        continue

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    relative = path.relative_to(IMAGE_DIR)
    split = relative.parts[0]

    hash_groups[digest].append((split, path))

cross_split = []
same_split = []

for digest, files in hash_groups.items():
    if len(files) < 2:
        continue

    splits = sorted(set(split for split, _ in files))

    record = [
        digest,
        ";".join(splits),
        len(files),
        ";".join(str(path.relative_to(ROOT)) for _, path in files)
    ]

    if len(splits) > 1:
        cross_split.append(record)
    else:
        same_split.append(record)

with open(REPORT, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["SHA256", "Splits", "File Count", "Files"])
    writer.writerows(cross_split)

print("\n========== PLANTSEG DUPLICATE SPLIT CHECK ==========")
print("Duplicate groups within the same split:", len(same_split))
print("Duplicate groups across different splits:", len(cross_split))

split_pairs = Counter()

for record in cross_split:
    split_pairs[record[1]] += 1

print("\nCross-split combinations:")
for combination, count in split_pairs.items():
    print(f"{combination}: {count}")

print("\nCross-split report saved:", REPORT)
