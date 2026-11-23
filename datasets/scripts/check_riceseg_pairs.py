# from pathlib import Path
# from PIL import Image
# from collections import Counter
# import random

# ROOT = Path(__file__).resolve().parents[2]
# MASK_ROOT = ROOT / "datasets/raw/RiceSeg/masks/mask-vesrion1"

# random.seed(42)

# for disease_dir in sorted(MASK_ROOT.iterdir()):
#     if not disease_dir.is_dir():
#         continue

#     files = list(disease_dir.glob("*"))
#     samples = random.sample(files, min(5, len(files)))

#     print(f"\n=== {disease_dir.name} ===")

#     for path in samples:
#         with Image.open(path) as img:
#             colors = Counter(img.convert("RGB").getdata())
#             print(path.name)
#             print("Unique colors:", len(colors))
#             print("Most common colors:", colors.most_common(5))

from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]

images_root = ROOT / "datasets/raw/RiceSeg/images/Rice Leaf Disease Images"
masks_root = ROOT / "datasets/raw/RiceSeg/masks/mask-vesrion1"

classes = ["Bacterialblight", "Blast", "Brownspot", "Tungro"]

fig, axes = plt.subplots(4, 2, figsize=(10, 14))

for i, cls in enumerate(classes):
    image_path = next((images_root / cls).iterdir())
    mask_path = masks_root / cls / image_path.name

    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    axes[i, 0].imshow(image)
    axes[i, 0].set_title(f"{cls} - Image")

    axes[i, 1].imshow(mask, cmap="gray")
    axes[i, 1].set_title(f"{cls} - Mask")

    axes[i, 0].axis("off")
    axes[i, 1].axis("off")

plt.tight_layout()
plt.show()