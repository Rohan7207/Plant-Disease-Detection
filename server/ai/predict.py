
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from model import PlantDiseaseModel
from dataset import eval_transform, PROJECT_ROOT
from image_quality import check_image_quality


CHECKPOINT_PATH = PROJECT_ROOT / "server" / "ai" / "mobilenetv2_phase2_best.pth"
DEVICE = torch.device("cpu")


class PlantDiseasePredictor:

    def __init__(self):
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=DEVICE,
            weights_only=True
        )

        self.classes = checkpoint["classes"]

        self.model = PlantDiseaseModel(
            num_classes=len(self.classes)
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(DEVICE)
        self.model.eval()

        self.transform = eval_transform

        print("Model loaded successfully!")
        print("Phase:", checkpoint["phase"])
        print("Epoch:", checkpoint["epoch"])
        print("Validation accuracy:", checkpoint["validation_accuracy"])
        print("Number of classes:", len(self.classes))

    def predict(self, image_path, top_k=3):

        image_path = Path(image_path)

        if not image_path.is_absolute():
            image_path = PROJECT_ROOT / image_path

        quality = check_image_quality(image_path)

        if not quality["valid"]:
            return {
                "success": False,
                "error": "Image failed technical quality checks.",
                "quality": quality
            }

        with Image.open(image_path) as original:
            image = ImageOps.exif_transpose(original).convert("RGB")
            image_tensor = self.transform(image).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(image_tensor.to(DEVICE))
            probabilities = F.softmax(logits, dim=1)[0]

        top_k = min(max(1, top_k), len(self.classes))
        scores, indices = torch.topk(probabilities, top_k)

        predictions = []

        for score, index in zip(scores.tolist(), indices.tolist()):
            class_name = self.classes[index]
            crop, disease = class_name.split("__", 1)

            predictions.append({
                "crop": crop,
                "disease": disease,
                "class": class_name,
                "confidence": round(score * 100, 2)
            })

        best = predictions[0]

        return {
            "success": True,
            "crop": best["crop"],
            "disease": best["disease"],
            "confidence": best["confidence"],
            "top_predictions": predictions,
            "quality": quality
        }


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python server/ai/predict.py <image_path>")
        raise SystemExit(1)

    predictor = PlantDiseasePredictor()
    result = predictor.predict(sys.argv[1])

    print("\n========== PREDICTION ==========")

    for key, value in result.items():
        print(f"{key}: {value}")
