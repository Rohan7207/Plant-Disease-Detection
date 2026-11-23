
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image, ImageOps

from model import PlantDiseaseModel
from dataset import eval_transform
from gradcam import GradCAM


ROOT = Path(__file__).resolve().parents[2]

IMAGE_PATH = "datasets/raw/PlantSeg/plantseg/images/test/garlic_leaf_blight_140.jpg"

CHECKPOINT = ROOT / "server/ai/mobilenetv2_phase2_best.pth"

OUTPUT = "datasets/metadata/gradcam_correct_garlic.png"
ACTUAL_CLASS = "garlic__garlic_leaf_blight"
TARGET_CLASS = "garlic__garlic_leaf_blight"

DEVICE = torch.device("cpu")


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

    if TARGET_CLASS not in classes:
        raise ValueError(f"Class not found: {TARGET_CLASS}")

    with Image.open(IMAGE_PATH) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")

    input_tensor = eval_transform(image).unsqueeze(0).to(DEVICE)

    # Get the model's prediction.
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    predicted_idx = probabilities.argmax().item()

    print("Actual class:", ACTUAL_CLASS)
    print("Predicted class:", classes[predicted_idx])
    print(
        "Prediction confidence:",
        round(probabilities[predicted_idx].item() * 100, 2),
        "%"
    )

    print("\nTop 5 predictions:")

    top_values, top_indices = torch.topk(probabilities, 5)

    for value, index in zip(top_values, top_indices):
        print(
            classes[index.item()],
            ":", round(value.item() * 100, 2), "%"
        )

    # Generate Grad-CAM for both classes.
    target_layer = model.model.features[-1]
    gradcam = GradCAM(model, target_layer)

    try:
        garlic_heatmap = gradcam.generate(
            input_tensor,
            classes.index(TARGET_CLASS)
        )

        wheat_heatmap = gradcam.generate(
            input_tensor,
            classes.index(ACTUAL_CLASS)
        )
    finally:
        gradcam.remove_hooks()

    
    # Prepare an image with exactly the same dimensions as model input.
    resized_image = image.resize(
        (224, 224),
        Image.Resampling.BILINEAR
    )

    image_array = np.array(resized_image)

    fig, axes = plt.subplots(1, 5, figsize=(20, 5))

    axes[0].imshow(image)
    axes[0].set_title("Original Image")

    axes[1].imshow(garlic_heatmap, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Garlic Heatmap")

    axes[2].imshow(image_array)
    axes[2].imshow(
        garlic_heatmap,
        cmap="jet",
        alpha=0.45,
        vmin=0,
        vmax=1
    )
    axes[2].set_title("Garlic Overlay")

    axes[3].imshow(wheat_heatmap, cmap="jet", vmin=0, vmax=1)
    axes[3].set_title("Wheat Heatmap")

    axes[4].imshow(image_array)
    axes[4].imshow(
        wheat_heatmap,
        cmap="jet",
        alpha=0.45,
        vmin=0,
        vmax=1
    )
    axes[4].set_title("Wheat Overlay")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT, dpi=200, bbox_inches="tight")
    plt.close()


    print("\nGrad-CAM saved:", OUTPUT)


if __name__ == "__main__":
    main()
