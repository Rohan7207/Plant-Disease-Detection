
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP", "MPO"}

MIN_WIDTH = 128
MIN_HEIGHT = 128

# Initial heuristic thresholds. We'll calibrate these using real images.
MIN_BRIGHTNESS = 70
MAX_BRIGHTNESS = 225
MIN_CONTRAST = 22
MIN_BLUR_SCORE = 100

def check_image_quality(image_path):
    image_path = Path(image_path)

    result = {
        "valid": False,
        "format": None,
        "width": None,
        "height": None,
        "brightness": None,
        "contrast": None,
        "blur_score": None,
        "warnings": [],
        "errors": [],
    }

    if not image_path.is_file():
        result["errors"].append("Image file not found")
        return result

    try:
        with Image.open(image_path) as original:
            image_format = original.format

            if image_format not in SUPPORTED_FORMATS:
                result["errors"].append("Unsupported image format")
                return result

            image = ImageOps.exif_transpose(original).convert("RGB")

            width, height = image.size
            result["format"] = image_format
            result["width"] = width
            result["height"] = height

            if width < MIN_WIDTH or height < MIN_HEIGHT:
                result["errors"].append("Image dimensions are too small")

            gray = image.convert("L").resize((256, 256))
            pixels = np.asarray(gray, dtype=np.float32)

            brightness = float(np.mean(pixels))
            contrast = float(np.std(pixels))

            # Laplacian variance: a heuristic blur indicator.
            laplacian = (
                -4 * pixels
                + np.roll(pixels, 1, axis=0)
                + np.roll(pixels, -1, axis=0)
                + np.roll(pixels, 1, axis=1)
                + np.roll(pixels, -1, axis=1)
            )
            blur_score = float(np.var(laplacian))

            result["brightness"] = round(brightness, 2)
            result["contrast"] = round(contrast, 2)
            result["blur_score"] = round(blur_score, 2)

            if brightness < MIN_BRIGHTNESS:
                result["warnings"].append("Image may be too dark")
            elif brightness > MAX_BRIGHTNESS:
                result["warnings"].append("Image may be overexposed")

            if contrast < MIN_CONTRAST:
                result["warnings"].append("Image may have low contrast")

            if blur_score < MIN_BLUR_SCORE:
                result["warnings"].append("Image may be blurry")

            result["valid"] = len(result["errors"]) == 0

    except (UnidentifiedImageError, OSError, ValueError):
        result["errors"].append("Image is corrupted or unreadable")

    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python server/ai/image_quality.py <image_path>")
        raise SystemExit(1)

    result = check_image_quality(sys.argv[1])

    print("\n========== IMAGE QUALITY REPORT ==========")

    for key, value in result.items():
        print(f"{key}: {value}")
