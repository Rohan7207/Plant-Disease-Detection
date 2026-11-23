from PIL import Image
import matplotlib.pyplot as plt
import os

base = "datasets/raw/PlantSeg/plantseg"

image_name = "apple_black_rot_1.jpg"
mask_name = "apple_black_rot_1.png"

image = Image.open(
    os.path.join(base, "images", "train", image_name)
).convert("RGB")

mask = Image.open(
    os.path.join(base, "annotations", "train", mask_name)
)

print("Image size:", image.size)
print("Mask size:", mask.size)
print("Mask values:", mask.getextrema())

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(mask, cmap="gray", vmin=0, vmax=1)
plt.title("Ground Truth Mask")
plt.axis("off")

plt.tight_layout()
plt.show()