
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from PIL import Image, ImageOps

from model import PlantDiseaseModel
from dataset import eval_transform
from gradcam import GradCAM

from collections import defaultdict, deque, Counter
import random


ROOT = Path(__file__).resolve().parents[2]

MANIFEST = ROOT / "datasets/metadata/master_image_manifest.csv"
CHECKPOINT = ROOT / "server/ai/mobilenetv2_phase2_best.pth"
OUTPUT = ROOT / "datasets/metadata/riceseg_layercam_mask_evaluation.csv"

DEVICE = torch.device("cpu")
LIMIT = 100
THRESHOLDS = [0.3, 0.5, 0.7]
MIN_MASK_AREA_PERCENT = 0.01
RANDOM_SEED = 42


def calculate_metrics(heatmap, mask, threshold):
    # Resize Grad-CAM to the original mask dimensions.
    heatmap_tensor = torch.tensor(
        heatmap, dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0)

    heatmap = torch.nn.functional.interpolate(
        heatmap_tensor,
        size=mask.shape,
        mode="bilinear",
        align_corners=False
    )[0, 0].numpy()

    predicted_mask = heatmap >= threshold

    intersection = np.logical_and(
        predicted_mask, mask
    ).sum()

    union = np.logical_or(
        predicted_mask, mask
    ).sum()

    pred_pixels = predicted_mask.sum()
    actual_pixels = mask.sum()

    iou = intersection / union if union else 0.0

    dice_denominator = pred_pixels + actual_pixels

    dice = (
        2 * intersection / dice_denominator
        if dice_denominator else 0.0
    )

    return {
        "IoU": float(iou),
        "Dice": float(dice),
        "GradCAM_Area_Percent": float(
            pred_pixels / mask.size * 100
        ),
        "GroundTruth_Area_Percent": float(
            actual_pixels / mask.size * 100
        )
    }
    
def select_representative_images(rows, limit):
    rng = random.Random(RANDOM_SEED)

    # Group images by crop and disease class.
    grouped = defaultdict(lambda: defaultdict(list))

    for row in rows:
        grouped[row["Crop"]][row["Class"]].append(row)

    # Create a shuffled queue of disease classes for each crop.
    crop_queues = {}

    for crop, class_groups in grouped.items():
        class_names = list(class_groups.keys())
        rng.shuffle(class_names)

        class_queue = deque()

        for class_name in class_names:
            images = class_groups[class_name]
            rng.shuffle(images)

            class_queue.append((class_name, images))

        crop_queues[crop] = class_queue

    # Randomize crop order.
    crop_order = list(crop_queues.keys())
    rng.shuffle(crop_order)

    selected = []

    # Round-robin across crops and disease classes.
    while len(selected) < limit:
        progress = False

        for crop in crop_order:
            queue = crop_queues[crop]

            if not queue:
                continue

            class_name, images = queue.popleft()

            selected.append(images.pop())
            progress = True

            if images:
                queue.append((class_name, images))

            if len(selected) >= limit:
                break

        if not progress:
            break

    return selected


def main():
    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=True
    )

    classes = checkpoint["classes"]

    model = PlantDiseaseModel(num_classes=len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    class_to_idx = {
        name: index
        for index, name in enumerate(classes)
    }

    with open(MANIFEST, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    eligible_all = [
        row for row in rows
        if row["Dataset"] == "RiceSeg"
        and row["Split"] == "test"
        and row["Classification_Action"] == "KEEP"
        and row["Segmentation_Action"] == "AVAILABLE"
    ]

    eligible = select_representative_images(
        eligible_all,
        LIMIT
    )

    print("Total eligible images:", len(eligible_all))
    print("Selected images:", len(eligible))

    print("\nSelected images per crop:")

    for crop, count in sorted(
        Counter(row["Crop"] for row in eligible).items()
    ):
        print(f"{crop}: {count}")

    print("\nSelected images per disease class:")
    print("Unique classes:", len(set(row["Class"] for row in eligible)))

    target_layer = model.model.features[17]
    gradcam = GradCAM(model, target_layer)

    results = []
    correct_predictions = 0

    try:
        for i, row in enumerate(eligible, 1):
            image_path = ROOT / row["Image"]
            mask_path = ROOT / row["Mask"]

            with Image.open(image_path) as img:
                image = ImageOps.exif_transpose(img).convert("RGB")

            with Image.open(mask_path) as img:
                mask_image = img.convert("L")

            mask = np.array(mask_image) >= 128
            
            mask_area_percent = float(mask.mean() * 100)

            mask_usable = (
                mask_area_percent >= MIN_MASK_AREA_PERCENT
            )

            input_tensor = eval_transform(image).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits = model(input_tensor)
                probabilities = torch.softmax(logits, dim=1)[0]

            predicted_idx = probabilities.argmax().item()
            predicted_class = classes[predicted_idx]
            actual_class = row["Class"]

            is_correct = predicted_class == actual_class
            correct_predictions += int(is_correct)

            targets = {
                "actual_class": class_to_idx[actual_class],
                "predicted_class": predicted_idx
            }

            print(
                f"\n[{i}/{len(eligible)}] {row['Image']}"
            )
            print("Actual:", actual_class)
            print("Predicted:", predicted_class)
            print("Correct:", is_correct)

            heatmaps = {}

            for target_name, target_idx in targets.items():
                if (
                    target_name == "predicted_class"
                    and predicted_class == actual_class
                ):
                    heatmaps[target_name] = heatmaps["actual_class"]
                else:
                    heatmaps[target_name] = gradcam.generate(
                        input_tensor,
                        target_idx
                    )

                # Generate LayerCAM using the same target class.
                layercam_name = f"{target_name}_layercam"

                if (
                    target_name == "predicted_class"
                    and predicted_class == actual_class
                ):
                    heatmaps[layercam_name] = heatmaps["actual_class_layercam"]
                else:
                    heatmaps[layercam_name] = gradcam.generate_layercam(
                        input_tensor,
                        target_idx
                    )
                    
            actual_cam = heatmaps["actual_class"]
            predicted_cam = heatmaps["predicted_class"]

            print("\nGrad-CAM comparison:")
            print(
                "Mean absolute difference:",
                round(float(np.mean(
                    np.abs(actual_cam - predicted_cam)
                )), 6)
            )
            
            print("\nLayerCAM comparison:")
            print(
                "Mean absolute difference:",
                round(float(np.mean(
                    np.abs(
                        heatmaps["actual_class_layercam"]
                        - heatmaps["predicted_class_layercam"]
                    )
                )), 6)
            )

            for name, cam in heatmaps.items():
                print(
                    name,
                    "| min:", round(float(cam.min()), 4),
                    "| max:", round(float(cam.max()), 4),
                    "| mean:", round(float(cam.mean()), 4),
                    "| pixels above 0.3:", int((cam >= 0.3).sum())
                )

            print(
                "Mask pixels at original resolution:",
                int(mask.sum()),
                "/",
                mask.size
            )

            for target_name, heatmap in heatmaps.items():
                for threshold in THRESHOLDS:
                    metrics = calculate_metrics(
                        heatmap, mask, threshold
                    )

                    results.append({
                        "Image": row["Image"],
                        "Crop": row["Crop"],
                        "Actual_Class": actual_class,
                        "Predicted_Class": predicted_class,
                        "Target_Type": target_name,
                        "Threshold": threshold,
                        "Prediction_Correct": is_correct,
                        "Mask_Usable": mask_usable,
                        **metrics
                    })

                print(
                    f"{target_name}: "
                    f"GT area = "
                    f"{mask.mean() * 100:.2f}%"
                )

    finally:
        gradcam.remove_hooks()

    fieldnames = list(results[0].keys())

    with open(OUTPUT, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("\n========== SUMMARY ==========")

    total_images = len(eligible)

    print("Images evaluated:", total_images)
    print(
        "Classification accuracy:",
        round(correct_predictions / total_images * 100, 2),
        "%"
    )

    usable_images = {
        row["Image"]
        for row in results
        if row["Mask_Usable"]
    }

    print("Images with usable masks:", len(usable_images))
    print("Images with tiny masks:", total_images - len(usable_images))

    print("\n========== LOCALIZATION RESULTS ==========")

    grouped = defaultdict(list)

    for result in results:
        if not result["Mask_Usable"]:
            continue

        key = (
            result["Target_Type"],
            result["Threshold"],
            result["Prediction_Correct"]
        )

        grouped[key].append(result)

    for (target, threshold, is_correct), values in sorted(grouped.items()):
        mean_iou = np.mean([v["IoU"] for v in values])
        mean_dice = np.mean([v["Dice"] for v in values])

        prediction_status = (
            "Correct predictions" if is_correct
            else "Incorrect predictions"
        )

        print(
            f"{target} | "
            f"threshold={threshold} | "
            f"{prediction_status} | "
            f"Images={len(values)} | "
            f"IoU={mean_iou:.4f} | "
            f"Dice={mean_dice:.4f}"
        )

    print("\nResults saved:", OUTPUT)


if __name__ == "__main__":
    main()
