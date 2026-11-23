import csv
from collections import defaultdict

path = "datasets/metadata/gradcam_mask_evaluation.csv"

with open(path, encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

grouped = defaultdict(dict)

for row in rows:
    key = (row["Image"], row["Threshold"])
    grouped[key][row["Target_Type"]] = row

print("========== IMAGE-WISE COMPARISON ==========")

for (image, threshold), targets in grouped.items():
    actual = targets.get("actual_class")
    predicted = targets.get("predicted_class")

    if actual and predicted:
        same = (
            actual["IoU"] == predicted["IoU"]
            and actual["Dice"] == predicted["Dice"]
        )

        print(f"\nImage: {image}")
        print("Threshold:", threshold)
        print("Prediction correct:", actual["Prediction_Correct"])
        print("Actual IoU:", actual["IoU"])
        print("Predicted IoU:", predicted["IoU"])
        print("Same metrics:", same)

print("\n========== EMPTY MASKS ==========")

for image in sorted(set(row["Image"] for row in rows)):
    row = next(r for r in rows if r["Image"] == image)

    if float(row["GroundTruth_Area_Percent"]) == 0:
        print(image)