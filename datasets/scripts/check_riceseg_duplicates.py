
from pathlib import Path
from collections import defaultdict, Counter
import hashlib

ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIR = ROOT / "datasets/raw/RiceSeg/images/Rice Leaf Disease Images"

groups = defaultdict(list)

for path in IMAGE_DIR.rglob("*"):
    if not path.is_file() or path.suffix.lower() not in {
        ".jpg", ".jpeg", ".png"
    }:
        continue

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    disease = path.parent.name
    groups[digest].append((disease, path))

same_class = 0
cross_class = 0
cross_class_details = []

for digest, files in groups.items():
    if len(files) < 2:
        continue

    classes = set(disease for disease, _ in files)

    if len(classes) > 1:
        cross_class += 1
        cross_class_details.append(
            (digest, sorted(classes), len(files))
        )
    else:
        same_class += 1

print("\n========== RICESEG DUPLICATE ANALYSIS ==========")
print("Duplicate groups within same disease:", same_class)
print("Duplicate groups across diseases:", cross_class)

print("\nCross-disease duplicate examples:")
for digest, classes, count in cross_class_details[:20]:
    print(f"{classes} | Copies: {count}")
