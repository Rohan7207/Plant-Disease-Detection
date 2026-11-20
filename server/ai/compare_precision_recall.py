
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
META = ROOT / "datasets" / "metadata"

def load_report(filename):
    with (META / filename).open(
        "r", newline="", encoding="utf-8"
    ) as file:
        return {
            row["Class"]: row
            for row in csv.DictReader(file)
        }

p1 = load_report("phase1_validation_report.csv")
p2 = load_report("phase2_validation_report.csv")

if set(p1) != set(p2):
    raise ValueError("Class lists do not match!")

results = []

for name in sorted(p1):
    a, b = p1[name], p2[name]

    if int(a["Support"]) != int(b["Support"]):
        raise ValueError(f"Support mismatch: {name}")

    if int(a["Support"]) == 0:
        continue

    precision_delta = float(b["Precision"]) - float(a["Precision"])
    recall_delta = float(b["Recall"]) - float(a["Recall"])

    if precision_delta > 0 and recall_delta > 0:
        category = "Both improved"
    elif precision_delta < 0 and recall_delta < 0:
        category = "Both declined"
    elif precision_delta > 0 and recall_delta < 0:
        category = "Precision improved, recall declined"
    elif precision_delta < 0 and recall_delta > 0:
        category = "Recall improved, precision declined"
    else:
        category = "Mixed or unchanged"

    results.append({
        "Class": name,
        "Support": int(a["Support"]),
        "Phase1_Precision": float(a["Precision"]),
        "Phase2_Precision": float(b["Precision"]),
        "Precision_Delta": precision_delta,
        "Phase1_Recall": float(a["Recall"]),
        "Phase2_Recall": float(b["Recall"]),
        "Recall_Delta": recall_delta,
        "Category": category,
    })

output = META / "phase1_phase2_precision_recall.csv"

with output.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

from collections import Counter

counts = Counter(row["Category"] for row in results)

print("Classes evaluated:", len(results))
print("\n========== COMPARISON ==========")

for category, count in counts.items():
    print(f"{category}: {count}")

print("\n========== LARGEST RECALL IMPROVEMENTS ==========")
for row in sorted(
    results, key=lambda x: x["Recall_Delta"], reverse=True
)[:10]:
    print(
        f'{row["Class"]} | '
        f'Recall: {row["Phase1_Recall"]:.3f} -> '
        f'{row["Phase2_Recall"]:.3f} | '
        f'Support: {row["Support"]}'
    )

print("\n========== LARGEST RECALL DECLINES ==========")
for row in sorted(results, key=lambda x: x["Recall_Delta"])[:10]:
    print(
        f'{row["Class"]} | '
        f'Recall: {row["Phase1_Recall"]:.3f} -> '
        f'{row["Phase2_Recall"]:.3f} | '
        f'Support: {row["Support"]}'
    )

print("\nSaved:", output)
