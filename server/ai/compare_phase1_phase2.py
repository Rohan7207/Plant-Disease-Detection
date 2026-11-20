
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PHASE1_REPORT = PROJECT_ROOT / "datasets/metadata/phase1_validation_report.csv"
PHASE2_REPORT = PROJECT_ROOT / "datasets/metadata/phase2_validation_report.csv"

OUTPUT_REPORT = PROJECT_ROOT / "datasets/metadata/phase1_phase2_class_comparison.csv"


def load_report(path):
    with open(path, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    return {row["Class"]: row for row in rows}


phase1 = load_report(PHASE1_REPORT)
phase2 = load_report(PHASE2_REPORT)

if set(phase1) != set(phase2):
    raise ValueError("Phase 1 and Phase 2 class lists do not match!")

results = []

for class_name in sorted(phase1):

    p1 = phase1[class_name]
    p2 = phase2[class_name]

    f1_before = float(p1["F1_Score"])
    f1_after = float(p2["F1_Score"])

    support_before = int(p1["Support"])
    support_after = int(p2["Support"])

    if support_before != support_after:
        raise ValueError(f"Validation support mismatch: {class_name}")

    if f1_after > f1_before:
        status = "IMPROVED"
    elif f1_after < f1_before:
        status = "DECLINED"
    else:
        status = "UNCHANGED"

    results.append({
        "Class": class_name,
        "Train_F1": f1_before,
        "Phase2_F1": f1_after,
        "F1_Change": f1_after - f1_before,
        "Validation_Support": support_before,
        "Status": status
    })


with open(OUTPUT_REPORT, "w", newline="", encoding="utf-8") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "Class",
            "Train_F1",
            "Phase2_F1",
            "F1_Change",
            "Validation_Support",
            "Status"
        ]
    )

    writer.writeheader()
    writer.writerows(results)


improved = [r for r in results if r["Status"] == "IMPROVED"]
declined = [r for r in results if r["Status"] == "DECLINED"]
unchanged = [r for r in results if r["Status"] == "UNCHANGED"]

evaluated = [r for r in results if r["Validation_Support"] > 0]

print("========== PHASE 1 vs PHASE 2 ==========")

print("Total classes:", len(results))
print("Classes with validation samples:", len(evaluated))
print("Improved classes:", len(improved))
print("Declined classes:", len(declined))
print("Unchanged classes:", len(unchanged))

if evaluated:
    p1_macro = sum(r["Train_F1"] for r in evaluated) / len(evaluated)
    p2_macro = sum(r["Phase2_F1"] for r in evaluated) / len(evaluated)

    print(f"\nPhase 1 macro F1: {p1_macro:.4f}")
    print(f"Phase 2 macro F1: {p2_macro:.4f}")
    print(f"Macro F1 improvement: {p2_macro - p1_macro:+.4f}")

print("\n========== 15 BIGGEST IMPROVEMENTS ==========")

for row in sorted(results, key=lambda r: r["F1_Change"], reverse=True)[:15]:
    print(
        f'{row["Class"]} | '
        f'{row["Train_F1"]:.4f} -> {row["Phase2_F1"]:.4f} | '
        f'{row["F1_Change"]:+.4f} | '
        f'Val: {row["Validation_Support"]}'
    )

print("\n========== 15 BIGGEST DECLINES ==========")

for row in sorted(results, key=lambda r: r["F1_Change"])[:15]:
    print(
        f'{row["Class"]} | '
        f'{row["Train_F1"]:.4f} -> {row["Phase2_F1"]:.4f} | '
        f'{row["F1_Change"]:+.4f} | '
        f'Val: {row["Validation_Support"]}'
    )

print("\nComparison report saved:", OUTPUT_REPORT)
