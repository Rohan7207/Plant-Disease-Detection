
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import PlantDiseaseDataset, eval_transform
from model import PlantDiseaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = PROJECT_ROOT / "datasets" / "metadata"

CHECKPOINTS = {
    "Phase 1": PROJECT_ROOT / "server" / "ai" / "mobilenetv2_phase1.pth",
    "Phase 2": PROJECT_ROOT / "server" / "ai" / "mobilenetv2_phase2_best.pth",
}


def main():
    device = torch.device("cpu")

    dataset = PlantDiseaseDataset(
        split="val",
        transform=eval_transform
    )

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0
    )

    results = []

    print("Validation images:", len(dataset))
    print("Classes:", len(dataset.classes))

    for phase, checkpoint_path in CHECKPOINTS.items():
        print(f"\nAnalyzing {phase}...")

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False
        )

        if checkpoint["classes"] != dataset.classes:
            raise ValueError(f"{phase}: class ordering mismatch!")

        model = PlantDiseaseModel(
            num_classes=len(dataset.classes)
        ).to(device)

        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        phase_results = []

        with torch.no_grad():
            for batch_index, (images, labels) in enumerate(loader):
                images = images.to(device)
                labels = labels.to(device)

                logits = model(images)
                probabilities = torch.softmax(logits, dim=1)

                confidence, predictions = probabilities.max(dim=1)
                top_probabilities, top_indices = probabilities.topk(3, dim=1)

                for i in range(len(labels)):
                    sample_index = batch_index * loader.batch_size + i

                    actual_idx = labels[i].item()
                    predicted_idx = predictions[i].item()

                    top3 = " | ".join(
                        f"{dataset.idx_to_class[idx.item()]}:"
                        f"{prob.item():.4f}"
                        for idx, prob in zip(
                            top_indices[i],
                            top_probabilities[i]
                        )
                    )

                    row = {
                        "Phase": phase,
                        "Image": dataset.samples[sample_index]["Image"],
                        "Actual_Class": dataset.idx_to_class[actual_idx],
                        "Predicted_Class": dataset.idx_to_class[predicted_idx],
                        "Confidence": round(confidence[i].item(), 6),
                        "Correct": actual_idx == predicted_idx,
                        "Top_3_Predictions": top3,
                    }

                    results.append(row)
                    phase_results.append(row)

        correct = [r for r in phase_results if r["Correct"]]
        incorrect = [r for r in phase_results if not r["Correct"]]

        mean_correct = (
            sum(r["Confidence"] for r in correct) / len(correct)
            if correct else 0
        )

        mean_incorrect = (
            sum(r["Confidence"] for r in incorrect) / len(incorrect)
            if incorrect else 0
        )

        accuracy = len(correct) / len(phase_results) * 100

        print("Accuracy:", f"{accuracy:.2f}%")
        print("Correct predictions:", len(correct))
        print("Incorrect predictions:", len(incorrect))
        print("Mean confidence (correct):", f"{mean_correct:.4f}")
        print("Mean confidence (incorrect):", f"{mean_incorrect:.4f}")

        print("\nConfidence bins:")

        bins = [
            (0.0, 0.5),
            (0.5, 0.7),
            (0.7, 0.9),
            (0.9, 1.01),
        ]

        for lower, upper in bins:
            group = [
                r for r in phase_results
                if lower <= r["Confidence"] < upper
            ]

            if group:
                bin_accuracy = (
                    sum(r["Correct"] for r in group) / len(group) * 100
                )
                print(
                    f"{lower:.0%}-{min(upper, 1.0):.0%}: "
                    f"{len(group)} images, "
                    f"accuracy {bin_accuracy:.2f}%"
                )

    output_path = REPORT_DIR / "prediction_confidence_comparison.csv"

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "Phase",
                "Image",
                "Actual_Class",
                "Predicted_Class",
                "Confidence",
                "Correct",
                "Top_3_Predictions",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print("\nDetailed results saved to:")
    print(output_path)


if __name__ == "__main__":
    main()
