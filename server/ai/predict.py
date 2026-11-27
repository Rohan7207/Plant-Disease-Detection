
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from .model import PlantDiseaseModel
from .dataset import eval_transform, PROJECT_ROOT
from .image_quality import check_image_quality


CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "server"
    / "ai"
    / "mobilenetv2_phase2_best.pth"
)

DEVICE = torch.device("cpu")


class PlantDiseasePredictor:

    def __init__(self):

        if not CHECKPOINT_PATH.is_file():
            raise FileNotFoundError(
                f"Model checkpoint not found: {CHECKPOINT_PATH}"
            )

        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=DEVICE,
            weights_only=True
        )

        required_keys = {"classes", "model_state_dict"}

        if not required_keys.issubset(checkpoint):
            raise ValueError("Invalid model checkpoint structure")

        self.classes = checkpoint["classes"]

        if not isinstance(self.classes, list) or not self.classes:
            raise ValueError("Invalid class mapping in checkpoint")

        if len(set(self.classes)) != len(self.classes):
            raise ValueError("Duplicate classes found in checkpoint")

        self.model = PlantDiseaseModel(
            num_classes=len(self.classes),
            pretrained=False
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True
        )

        self.model.to(DEVICE)
        self.model.eval()

        self.transform = eval_transform

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

        top_k = max(1, min(int(top_k), len(self.classes)))

        with torch.inference_mode():
            logits = self.model(image_tensor.to(DEVICE))
            probabilities = F.softmax(logits, dim=1)[0]

            scores, indices = torch.topk(
                probabilities,
                top_k
            )

        predictions = []

        for score, index in zip(scores.tolist(), indices.tolist()):

            class_name = self.classes[index]

            if "__" not in class_name:
                raise ValueError(
                    f"Invalid class name in checkpoint: {class_name}"
                )

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
