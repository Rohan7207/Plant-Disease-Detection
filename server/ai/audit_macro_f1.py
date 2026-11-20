
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORTS = {
    "Phase 1": ROOT / "datasets/metadata/phase1_validation_report.csv",
    "Phase 2": ROOT / "datasets/metadata/phase2_validation_report.csv",
}

for phase, path in REPORTS.items():
    with path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    total_classes = len(rows)
    evaluated = [r for r in rows if int(r["Support"]) > 0]
    zero_support = total_classes - len(evaluated)

    macro_all = sum(float(r["F1_Score"]) for r in rows) / total_classes
    macro_evaluated = (
        sum(float(r["F1_Score"]) for r in evaluated) / len(evaluated)
    )

    print(f"\n========== {phase} ==========")
    print("Total classes:", total_classes)
    print("Classes with validation support:", len(evaluated))
    print("Classes without validation support:", zero_support)
    print(f"Macro F1 (all classes): {macro_all:.4f}")
    print(f"Macro F1 (evaluated classes): {macro_evaluated:.4f}")
