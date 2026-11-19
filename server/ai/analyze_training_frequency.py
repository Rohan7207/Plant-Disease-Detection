
import csv
from collections import Counter

MANIFEST_PATH = "datasets/metadata/master_image_manifest.csv"
REPORT_PATH = "datasets/metadata/phase1_validation_report.csv"

# Count retained training images per class
train_counts = Counter()

with open(MANIFEST_PATH, newline="", encoding="utf-8") as file:
    for row in csv.DictReader(file):
        if (
            row["Classification_Action"] == "KEEP"
            and row["Split"] == "train"
        ):
            train_counts[row["Class"]] += 1

# Load validation metrics
with open(REPORT_PATH, newline="", encoding="utf-8") as file:
    report = list(csv.DictReader(file))

for row in report:
    row["F1_Score"] = float(row["F1_Score"])
    row["Support"] = int(row["Support"])
    row["Train_Count"] = train_counts[row["Class"]]

# Group by training frequency
groups = {
    "1-5": [],
    "6-10": [],
    "11-20": [],
    "21-50": [],
    "51-100": [],
    "101+": []
}

for row in report:
    count = row["Train_Count"]

    if count <= 5:
        groups["1-5"].append(row)
    elif count <= 10:
        groups["6-10"].append(row)
    elif count <= 20:
        groups["11-20"].append(row)
    elif count <= 50:
        groups["21-50"].append(row)
    elif count <= 100:
        groups["51-100"].append(row)
    else:
        groups["101+"].append(row)

print("========== PERFORMANCE BY TRAINING FREQUENCY ==========")

for name, rows in groups.items():
    if not rows:
        continue

    evaluated = [r for r in rows if r["Support"] > 0]

    macro_f1 = (
        sum(r["F1_Score"] for r in evaluated) / len(evaluated)
        if evaluated else 0
    )

    print(f"\nTraining images per class: {name}")
    print("Total classes:", len(rows))
    print("Classes evaluated:", len(evaluated))
    print(f"Macro F1: {macro_f1:.4f}")

# Weakest classes with both counts
print("\n========== 20 WEAKEST EVALUATED CLASSES ==========")

evaluated = [r for r in report if r["Support"] > 0]

for row in sorted(evaluated, key=lambda r: r["F1_Score"])[:20]:
    print(
        f'{row["Class"]} | '
        f'Train: {row["Train_Count"]} | '
        f'Val: {row["Support"]} | '
        f'F1: {row["F1_Score"]:.4f}'
    )

print("\n========== CLASSES WITHOUT VALIDATION DATA ==========")

for row in report:
    if row["Support"] == 0:
        print(
            f'{row["Class"]} | '
            f'Train: {row["Train_Count"]}'
        )
