
from pathlib import Path
from uuid import uuid4

from flask import Blueprint, jsonify, request

from server.services.prediction_service import PredictionService
from server.utils.exceptions import ValidationError


prediction_bp = Blueprint("prediction", __name__)

UPLOAD_FOLDER = Path(__file__).resolve().parents[1] / "temp_uploads"

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Initialize the prediction service once.
prediction_service = PredictionService()


@prediction_bp.route("/predict", methods=["POST"])
def predict():

    # Validate uploaded file.
    if "image" not in request.files:
        raise ValidationError(
            "No image uploaded. Use the 'image' field."
        )

    image = request.files["image"]

    if not image.filename:
        raise ValidationError("No file selected.")

    extension = Path(image.filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise ValidationError("Unsupported image format.")

    UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

    # Generate a unique server-side filename.
    image_path = UPLOAD_FOLDER / f"{uuid4().hex}{extension}"

    try:
        image.save(image_path)

        result = prediction_service.predict_image(image_path)

        # Handle image-quality validation failures.
        if not result["success"]:
            return jsonify({
                "success": False,
                "error": {
                    "code": "IMAGE_QUALITY_VALIDATION_FAILED",
                    "message": result.get(
                        "error",
                        "Image validation failed."
                    ),
                    "details": {
                        "quality": result.get("quality")
                    }
                }
            }), 422

        return jsonify(result), 200

    finally:
        # Always remove the temporary uploaded image.
        image_path.unlink(missing_ok=True)
