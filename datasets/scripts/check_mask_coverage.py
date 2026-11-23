import csv
from collections import Counter

path = "datasets/metadata/master_image_manifest.csv"

with open(path, encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

plantseg_test = [
    r for r in rows
    if r["Dataset"] == "PlantSeg"
    and r["Split"].lower() == "test"
]

print("PlantSeg test images:", len(plantseg_test))

print("\nClassification actions:")
print(Counter(r["Classification_Action"] for r in plantseg_test))

print("\nSegmentation actions:")
print(Counter(r["Segmentation_Action"] for r in plantseg_test))

eligible = [
    r for r in plantseg_test
    if r["Classification_Action"] == "KEEP"
    and r["Segmentation_Action"] == "AVAILABLE"
]

print("\nEligible for evaluation:", len(eligible))
print("\nEligible images by crop:")
for crop, count in Counter(r["Crop"] for r in eligible).most_common():
    print(f"{crop}: {count}")