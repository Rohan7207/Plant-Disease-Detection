
import json
from pathlib import Path

from server.utils.exceptions import NotFoundError, ValidationError


DISEASE_DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "diseases.json"
)

SUPPORTED_LANGUAGES = {"en", "kn", "hi"}


class DiseaseService:
    def __init__(self):
        if not DISEASE_DATA_PATH.is_file():
            raise FileNotFoundError(
                f"Disease data file not found: {DISEASE_DATA_PATH}"
            )

        with DISEASE_DATA_PATH.open("r", encoding="utf-8") as file:
            self.diseases = json.load(file)

    def get_disease_info(self, crop, disease, language="en"):
        crop = crop.strip().lower()
        disease = disease.strip().lower()
        language = language.strip().lower()

        if language not in SUPPORTED_LANGUAGES:
            raise ValidationError(
                "Unsupported language. Supported languages: en, kn, hi."
            )

        crop_data = self.diseases.get(crop)

        if crop_data is None:
            raise NotFoundError(f"No information found for crop: {crop}")

        disease_data = crop_data.get(disease)

        if disease_data is None:
            raise NotFoundError(
                f"No information found for disease: {disease}"
            )

        return {
            "crop": crop,
            "disease": disease,
            "language": language,
            "name": disease_data["name"][language],
            "scientific_name": disease_data.get("scientific_name"),
            "description": disease_data["description"][language],
            "symptoms": disease_data["symptoms"][language],
            "affected_parts": disease_data["affected_parts"][language],
            "prevention": disease_data["prevention"][language],
            "management": disease_data["management"][language]
        }
