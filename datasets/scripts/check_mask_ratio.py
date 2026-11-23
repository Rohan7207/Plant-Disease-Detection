import csv
import os
import numpy as np
from PIL import Image

base = "datasets/raw/PlantSeg/plantseg"
metadata = os.path.join(base, "Metadata.csv")

with open(metadata, encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

split_map = {
    "Training": "train",
    "Validation": "val",
    "Testing": "test"
}

for row in rows[:10]:
    mask_path = os.path.join(
        base, "annotations",
        split_map[row["Split"]],
        row["Label file"]
    )

    mask = np.array(Image.open(mask_path))

    actual_ratio = np.mean(mask == 1)
    metadata_ratio = float(row["Mask ratio"])

    print(f"\nImage: {row['Name']}")
    print(f"Metadata ratio: {metadata_ratio:.6f}")
    print(f"Actual white ratio: {actual_ratio:.6f}")
    print(f"Difference: {abs(metadata_ratio - actual_ratio):.6f}")