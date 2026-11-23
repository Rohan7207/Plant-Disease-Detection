
import csv
from pathlib import Path

from image_quality import check_image_quality

ROOT = Path(__file__).resolve().parents[2]

MANIFEST = ROOT / "datasets/metadata/master_image_manifest.csv"
OUTPUT = ROOT / "datasets/metadata/image_quality_test_report.csv"

with MANIFEST.open("r", newline="", encoding="utf-8") as file:
    manifest = list(csv.DictReader(file))

image_paths = sorted({
    ROOT / row["Image"]
    for row in manifest
    if row["Classification_Action"] == "KEEP"
})

print("Retained classification images:", len(image_paths))

results = []

for index, image_path in enumerate(image_paths, start=1):
    result = check_image_quality(image_path)

    results.append({
        "Image": str(image_path.relative_to(ROOT)),
        "Valid": result["valid"],
        "Format": result["format"],
        "Width": result["width"],
        "Height": result["height"],
        "Brightness": result["brightness"],
        "Contrast": result["contrast"],
        "Blur_Score": result["blur_score"],
        "Warnings": "; ".join(result["warnings"]),
        "Errors": "; ".join(result["errors"]),
    })

    if index % 500 == 0:
        print(f"Processed {index}/{len(image_paths)}")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("\n========== CLASSIFICATION IMAGE RESULTS ==========")
print("Total images:", len(results))
print("Valid images:", sum(r["Valid"] for r in results))
print("Images with warnings:", sum(bool(r["Warnings"]) for r in results))
print("Images with errors:", sum(bool(r["Errors"]) for r in results))
print("Report saved:", OUTPUT)
