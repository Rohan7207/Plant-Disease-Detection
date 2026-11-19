
import csv

REPORT_PATH = "datasets/metadata/phase1_validation_report.csv"

with open(REPORT_PATH, newline="", encoding="utf-8") as file:
    rows = list(csv.DictReader(file))

for row in rows:
    row["Precision"] = float(row["Precision"])
    row["Recall"] = float(row["Recall"])
    row["F1_Score"] = float(row["F1_Score"])
    row["Support"] = int(row["Support"])

evaluated = [row for row in rows if row["Support"] > 0]
not_evaluated = [row for row in rows if row["Support"] == 0]

macro_f1_evaluated = sum(
    row["F1_Score"] for row in evaluated
) / len(evaluated)

print("Total classes:", len(rows))
print("Classes with validation samples:", len(evaluated))
print("Classes without validation samples:", len(not_evaluated))

print(
    "\nMacro F1 (classes with validation samples):",
    round(macro_f1_evaluated, 4)
)

print("\n========== 15 WEAKEST EVALUATED CLASSES ==========")

weakest = sorted(
    evaluated,
    key=lambda row: (row["F1_Score"], row["Support"])
)

for row in weakest[:15]:
    print(
        f'{row["Class"]} | '
        f'F1: {row["F1_Score"]:.4f} | '
        f'Recall: {row["Recall"]:.4f} | '
        f'Precision: {row["Precision"]:.4f} | '
        f'Support: {row["Support"]}'
    )

print("\n========== CLASSES WITHOUT VALIDATION SAMPLES ==========")

for row in not_evaluated:
    print(row["Class"])
