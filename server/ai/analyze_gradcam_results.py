
import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "datasets/metadata/riceseg_layercam_mask_evaluation.csv"

groups = defaultdict(list)

with open(CSV_PATH, encoding="utf-8-sig", newline="") as f:
    for row in csv.DictReader(f):
        if row["Mask_Usable"] != "True":
            continue
# tree datasets\metadata /F
        key = (
            row["Crop"],
            row["Actual_Class"],
            row["Target_Type"],
            row["Threshold"],
            row["Prediction_Correct"]
        )

        groups[key].append(row)

print("========== CLASS-WISE LAYERCAM RESULTS ==========")

for key, values in sorted(groups.items()):
    crop, actual, target, threshold, correct = key

    if "layercam" not in target:
        continue

    iou = sum(float(x["IoU"]) for x in values) / len(values)
    dice = sum(float(x["Dice"]) for x in values) / len(values)

    print(
        f"{crop} | {actual} | {target} | "
        f"threshold={threshold} | correct={correct} | "
        f"n={len(values)} | IoU={iou:.4f} | Dice={dice:.4f}"
    )
