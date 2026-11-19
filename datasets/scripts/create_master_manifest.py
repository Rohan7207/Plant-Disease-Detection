
import csv
from pathlib import Path

ROOT = Path.cwd()
META = ROOT / "datasets" / "metadata"
RAW = ROOT / "datasets" / "raw"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def normalize(path):
    return str(path).replace("\\", "/").lower()

# Load deduplication decisions.
dedup_path = META / "deduplication_manifest.csv"
dedup_rows = read_csv(dedup_path)

dedup_actions = {
    normalize(row["File"]): row["Action"]
    for row in dedup_rows
}

# Load segmentation-only exclusions.
seg_exclusion_path = META / "plantseg_segmentation_exclusions.csv"
seg_exclusions = set()

if seg_exclusion_path.exists():
    for row in read_csv(seg_exclusion_path):
        seg_exclusions.add(normalize(row["Image"]))

records = []
errors = []

def add_record(dataset, crop, disease, image, mask=None, split=""):
    image_rel = image.relative_to(ROOT)
    image_key = normalize(image_rel)

    dedup_action = dedup_actions.get(image_key, "KEEP")
    classification_action = (
        "EXCLUDE" if dedup_action != "KEEP" else "KEEP"
    )

    segmentation_action = "NO_MASK"

    if mask is not None:
        segmentation_action = (
            "EXCLUDE"
            if image_key in seg_exclusions
            else "AVAILABLE"
        )

    crop_label = crop.strip().lower().replace(" ", "_")
    disease_label = disease.strip().lower().replace(" ", "_")
    class_name = f"{crop_label}__{disease_label}"

    records.append({
        "Dataset": dataset,
        "Crop": crop,
        "Disease": disease,
        "Class": class_name,
        "Image": image_rel.as_posix(),
        "Mask": mask.relative_to(ROOT).as_posix() if mask else "",
        "Split": split,
        "Dedup_Action": dedup_action,
        "Classification_Action": classification_action,
        "Segmentation_Action": segmentation_action,
    })


# 1. PlantSeg
plantseg_meta = read_csv(META / "PlantSeg" / "Metadata.csv")

split_map = {
    "Training": "train",
    "Validation": "val",
    "Test": "test",
}

plantseg_images = RAW / "PlantSeg" / "plantseg" / "images"
plantseg_masks = RAW / "PlantSeg" / "plantseg" / "annotations"

for row in plantseg_meta:
    split = split_map.get(row["Split"])

    if split is None:
        errors.append(f"Unknown PlantSeg split: {row['Split']}")
        continue

    image = plantseg_images / split / row["Name"]
    mask = plantseg_masks / split / row["Label file"]

    if not image.is_file():
        errors.append(f"Missing PlantSeg image: {image}")
        continue

    if not mask.is_file():
        errors.append(f"Missing PlantSeg mask: {mask}")
        continue

    add_record(
        "PlantSeg",
        row["Plant"],
        row["Disease"],
        image,
        mask,
        split,
    )


# 2. RiceSeg, Cotton, and Chilli
folder_datasets = {
    "RiceSeg": {
        "images": RAW / "RiceSeg" / "images" / "Rice Leaf Disease Images",
        "masks": RAW / "RiceSeg" / "masks" / "mask-vesrion1",
        "crop": "Rice",
    },
    "Cotton": {
        "images": RAW / "Cotton" / "Cotton_Original_Dataset",
        "masks": None,
        "crop": "Cotton",
    },
    "Chilli": {
        "images": RAW / "Chilli" / "Chili Leaf Disease Original Dataset",
        "masks": None,
        "crop": "Chilli",
    },
}

for dataset, config in folder_datasets.items():
    image_root = config["images"]
    mask_root = config["masks"]
    crop = config["crop"]

    if not image_root.is_dir():
        errors.append(f"Missing image directory: {image_root}")
        continue

    # Index RiceSeg masks by class folder and filename stem.
    mask_index = {}

    if mask_root:
        if not mask_root.is_dir():
            errors.append(f"Missing mask directory: {mask_root}")
            continue

        for mask in mask_root.rglob("*"):
            if mask.is_file() and mask.suffix.lower() in IMAGE_EXTENSIONS:
                key = (mask.parent.name.lower(), mask.stem.lower())
                mask_index[key] = mask

    for class_dir in sorted(p for p in image_root.iterdir() if p.is_dir()):
        disease = class_dir.name

        for image in class_dir.rglob("*"):
            if not image.is_file() or image.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            mask = None

            if mask_root:
                key = (class_dir.name.lower(), image.stem.lower())
                mask = mask_index.get(key)

                if mask is None:
                    errors.append(f"Missing {dataset} mask for: {image}")
                    continue

            add_record(dataset, crop, disease, image, mask)


# Write master manifest.
output_path = META / "master_image_manifest.csv"
fieldnames = [
    "Dataset",
    "Crop",
    "Disease",
    "Class",
    "Image",
    "Mask",
    "Split",
    "Dedup_Action",
    "Classification_Action",
    "Segmentation_Action",
]

with output_path.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(records)

# Print validation summary.
print(f"\nMaster manifest saved: {output_path}")
print(f"Total image records: {len(records)}")

print("\nDataset counts:")
for dataset in folder_datasets:
    count = sum(r["Dataset"] == dataset for r in records)
    print(f"{dataset}: {count}")

plantseg_count = sum(r["Dataset"] == "PlantSeg" for r in records)
print(f"PlantSeg: {plantseg_count}")

print("\nClassification actions:")
for action in ("KEEP", "EXCLUDE"):
    count = sum(r["Classification_Action"] == action for r in records)
    print(f"{action}: {count}")

print("\nSegmentation actions:")
for action in ("AVAILABLE", "NO_MASK", "EXCLUDE"):
    count = sum(r["Segmentation_Action"] == action for r in records)
    print(f"{action}: {count}")

print(f"\nValidation errors: {len(errors)}")
for error in errors[:20]:
    print(error)

if len(errors) > 20:
    print(f"... and {len(errors) - 20} more errors")
