
import csv
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "datasets/metadata/image_quality_test_report.csv"

warning_counts = Counter()
error_counts = Counter()
warning_examples = {}
error_examples = {}

with REPORT.open("r", newline="", encoding="utf-8") as file:
    rows = list(csv.DictReader(file))

for row in rows:
    for warning in filter(None, row["Warnings"].split("; ")):
        warning_counts[warning] += 1
        warning_examples.setdefault(warning, row["Image"])

    for error in filter(None, row["Errors"].split("; ")):
        error_counts[error] += 1
        error_examples.setdefault(error, row["Image"])

print("========== WARNING BREAKDOWN ==========")
for reason, count in warning_counts.most_common():
    print(f"{reason}: {count}")
    print("Example:", warning_examples[reason])

print("\n========== ERROR BREAKDOWN ==========")
for reason, count in error_counts.most_common():
    print(f"{reason}: {count}")
    print("Example:", error_examples[reason])

print("\n========== SUMMARY ==========")
print("Total images:", len(rows))
print("Images with warnings:", sum(bool(r["Warnings"]) for r in rows))
print("Images with errors:", sum(bool(r["Errors"]) for r in rows))
