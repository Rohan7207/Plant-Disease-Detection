
import csv
from collections import defaultdict

REPORT_PATH = "datasets/metadata/phase1_validation_report.csv"

with open(REPORT_PATH, newline="", encoding="utf-8") as file:
    rows = list(csv.DictReader(file))

for row in rows:
    row["F1_Score"] = float(row["F1_Score"])
    row["Support"] = int(row["Support"])

groups = {
    "1-2 images": [],
    "3-5 images": [],
    "6-10 images": [],
    "11-20 images": [],
    "21-50 images": [],
    "51+ images": []
}

for row in rows:
    support = row["Support"]

    if support == 0:
        continue
    elif support <= 2:
        groups["1-2 images"].append(row)
    elif support <= 5:
        groups["3-5 images"].append(row)
    elif support <= 10:
        groups["6-10 images"].append(row)
    elif support <= 20:
        groups["11-20 images"].append(row)
    elif support <= 50:
        groups["21-50 images"].append(row)
    else:
        groups["51+ images"].append(row)

print("========== PERFORMANCE BY CLASS FREQUENCY ==========")

for group_name, group_rows in groups.items():
    if not group_rows:
        continue

    macro_f1 = sum(
        row["F1_Score"] for row in group_rows
    ) / len(group_rows)

    total_images = sum(
        row["Support"] for row in group_rows
    )

    print(f"\n{group_name}")
    print("Number of classes:", len(group_rows))
    print("Validation images:", total_images)
    print(f"Macro F1: {macro_f1:.4f}")

# Crop-level macro F1
crop_groups = defaultdict(list)

for row in rows:
    if row["Support"] > 0:
        crop = row["Class"].split("__")[0]
        crop_groups[crop].append(row)

crop_results = []

for crop, crop_rows in crop_groups.items():
    macro_f1 = sum(
        row["F1_Score"] for row in crop_rows
    ) / len(crop_rows)

    crop_results.append((macro_f1, crop, len(crop_rows)))

crop_results.sort()

print("\n========== 10 LOWEST CROP-LEVEL MACRO F1 ==========")

for f1, crop, class_count in crop_results[:10]:
    print(
        f"{crop}: F1={f1:.4f}, "
        f"Classes={class_count}"
    )

print("\n========== 10 HIGHEST CROP-LEVEL MACRO F1 ==========")

for f1, crop, class_count in crop_results[-10:][::-1]:
    print(
        f"{crop}: F1={f1:.4f}, "
        f"Classes={class_count}"
    )
