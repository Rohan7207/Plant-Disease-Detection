
from pathlib import Path


class PredictionService:

    def __init__(self):
        self._predictor = None

    @property
    def predictor(self):
        if self._predictor is None:
            from server.ai.predict import PlantDiseasePredictor
            self._predictor = PlantDiseasePredictor()

        return self._predictor

    def predict_image(self, image_path: Path):
        return self.predictor.predict(image_path)
