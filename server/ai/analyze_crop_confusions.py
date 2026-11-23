
import csv
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]

REPORT_INPUT = ROOT / "datasets/metadata/phase2_test_report.csv"
CONFUSION_INPUT = ROOT / "datasets/metadata/phase2_test_confusions.csv"
OUTPUT = ROOT / "datasets/metadata/phase2_crop_confusions.csv"


def main():
    crop_support = defaultdict(int)
    crop_errors = defaultdict(int)
    cross_crop_errors = defaultdict(int)

    # Read support for EVERY class, including perfectly classified classes.
    with REPORT_INPUT.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            class_name = row["Class"]
            support = int(float(row["Support"]))

            crop = class_name.split("__", 1)[0]
            crop_support[crop] += support

    # Count only cross-crop misclassifications.
    with CONFUSION_INPUT.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            actual = row["Actual_Class"]
            predicted = row["Predicted_Class"]
            count = int(row["Count"])

            actual_crop = actual.split("__", 1)[0]
            predicted_crop = predicted.split("__", 1)[0]

            if actual_crop != predicted_crop:
                crop_errors[actual_crop] += count
                cross_crop_errors[(actual_crop, predicted_crop)] += count

    results = []

    for crop, support in crop_support.items():
        errors = crop_errors[crop]
        accuracy = (support - errors) / support if support else 0

        results.append({
            "Crop": crop,
            "Test_Images": support,
            "Cross_Crop_Errors": errors,
            "Crop_Accuracy": round(accuracy, 4)
        })

    results.sort(key=lambda row: row["Crop_Accuracy"])

    with OUTPUT.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "Crop",
                "Test_Images",
                "Cross_Crop_Errors",
                "Crop_Accuracy"
            ]
        )
        writer.writeheader()
        writer.writerows(results)

    total_images = sum(crop_support.values())
    total_cross_errors = sum(crop_errors.values())

    print("\n========== CORRECTED CROP-LEVEL ACCURACY ==========")

    for row in results:
        print(
            f'{row["Crop"]}: {row["Crop_Accuracy"]:.2%} '
            f'| Errors: {row["Cross_Crop_Errors"]}/{row["Test_Images"]}'
        )

    print("\nOverall crop accuracy:",
          f"{(total_images - total_cross_errors) / total_images:.2%}")

    print("\n========== TOP CROSS-CROP CONFUSIONS ==========")

    for (actual, predicted), count in sorted(
        cross_crop_errors.items(),
        key=lambda item: item[1],
        reverse=True
    )[:20]:
        print(f"{actual} -> {predicted}: {count}")

    print("\nSaved:", OUTPUT)


if __name__ == "__main__":
    main()
