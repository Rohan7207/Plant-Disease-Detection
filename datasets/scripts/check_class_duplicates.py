
from pathlib import Path
from collections import defaultdict
import hashlib

ROOT = Path(__file__).resolve().parents[2]

SOURCES = {
    "Cotton": ROOT / "datasets/raw/Cotton/Cotton_Original_Dataset",
    "Chilli": ROOT / "datasets/raw/Chilli/Chili Leaf Disease Original Dataset",
}

for dataset, folder in SOURCES.items():
    groups = defaultdict(list)

    print(f"\nChecking {dataset}...")

    for path in folder.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {
            ".jpg", ".jpeg", ".png"
        }:
            continue

        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        groups[digest].append((path.parent.name, path))

    same_class = 0
    cross_class = 0

    for files in groups.values():
        if len(files) < 2:
            continue

        classes = {class_name for class_name, _ in files}

        if len(classes) > 1:
            cross_class += 1
        else:
            same_class += 1

    print("Duplicate groups within same class:", same_class)
    print("Duplicate groups across classes:", cross_class)

print("\n========== CHECK COMPLETE ==========")
