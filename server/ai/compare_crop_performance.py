
import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

P1_PATH = ROOT / "datasets/metadata/phase1_validation_report.csv"
P2_PATH = ROOT / "datasets/metadata/phase2_validation_report.csv"
OUTPUT_PATH = ROOT / "datasets/metadata/phase1_phase2_crop_comparison.csv"


def load_report(path):
    with open(path, newline="", encoding="utf-8") as file:
        return {row["Class"]: row for row in csv.DictReader(file)}


p1 = load_report(P1_PATH)
p2 = load_report(P2_PATH)

if set(p1) != set(p2):
    raise ValueError("Class lists do not match!")


crop_data = defaultdict(lambda: {
    "p1_f1": [],
    "p2_f1": [],
    "classes": 0
})

for class_name in p1:

    crop = class_name.split("__", 1)[0]

    support = int(p1[class_name]["Support"])

    if support == 0:
        continue

    f1_before = float(p1[class_name]["F1_Score"])
    f1_after = float(p2[class_name]["F1_Score"])

    crop_data[crop]["p1_f1"].append(f1_before)
    crop_data[crop]["p2_f1"].append(f1_after)
    crop_data[crop]["classes"] += 1


results = []

for crop, data in crop_data.items():

    p1_macro = sum(data["p1_f1"]) / len(data["p1_f1"])
    p2_macro = sum(data["p2_f1"]) / len(data["p2_f1"])

    results.append({
        "Crop": crop,
        "Classes": data["classes"],
        "Phase1_Macro_F1": p1_macro,
        "Phase2_Macro_F1": p2_macro,
        "Change": p2_macro - p1_macro
    })


results.sort(key=lambda row: row["Change"])

with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "Crop",
            "Classes",
            "Phase1_Macro_F1",
            "Phase2_Macro_F1",
            "Change"
        ]
    )

    writer.writeheader()
    writer.writerows(results)


print("========== CROP-LEVEL COMPARISON ==========")

print("\n10 crops with largest declines:")

for row in results[:10]:
    print(
        f'{row["Crop"]} | '
        f'{row["Phase1_Macro_F1"]:.4f} -> '
        f'{row["Phase2_Macro_F1"]:.4f} | '
        f'{row["Change"]:+.4f} | '
        f'Classes: {row["Classes"]}'
    )


print("\n10 crops with largest improvements:")

for row in results[-10:][::-1]:
    print(
        f'{row["Crop"]} | '
        f'{row["Phase1_Macro_F1"]:.4f} -> '
        f'{row["Phase2_Macro_F1"]:.4f} | '
        f'{row["Change"]:+.4f} | '
        f'Classes: {row["Classes"]}'
    )


print("\nComparison saved:", OUTPUT_PATH)
