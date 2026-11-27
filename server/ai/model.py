
import torch
import torch.nn as nn
from torchvision import models

class PlantDiseaseModel(nn.Module):

    def __init__(self, num_classes=130, pretrained=True):
        super().__init__()

        weights = (
            models.MobileNet_V2_Weights.DEFAULT
            if pretrained
            else None
        )

        self.model = models.mobilenet_v2(weights=weights)

        input_features = self.model.classifier[1].in_features

        self.model.classifier[1] = nn.Linear(
            input_features,
            num_classes
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    device = torch.device("cpu")

    model = PlantDiseaseModel(num_classes=130)
    model = model.to(device)

    print("Model: MobileNetV2")
    print("Classes:", 130)
    print("Device:", device)

    # Test one batch.
    dummy_input = torch.randn(4, 3, 224, 224)

    with torch.no_grad():
        output = model(dummy_input)

    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)
    print("Model setup successful!")
