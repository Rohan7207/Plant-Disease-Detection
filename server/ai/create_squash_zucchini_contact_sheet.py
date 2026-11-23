
import csv
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw

ROOT = Path(__file__).resolve().parents[2]

INPUT = ROOT / "datasets/metadata/squash_zucchini_errors.csv"
OUTPUT = ROOT / "datasets/metadata/squash_zucchini_contact_sheet.jpg"

COLUMNS = 3
CELL_WIDTH = 360
CELL_HEIGHT = 290
THUMB_WIDTH = 340
THUMB_HEIGHT = 235


def main():
    with INPUT.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    rows.sort(key=lambda row: float(row["Confidence"]), reverse=True)

    image_count = len(rows)
    sheet = Image.new(
        "RGB",
        (COLUMNS * CELL_WIDTH,
         ((image_count + COLUMNS - 1) // COLUMNS) * CELL_HEIGHT),
        "white"
    )

    draw = ImageDraw.Draw(sheet)

    for index, row in enumerate(rows):
        path = ROOT / row["Image"]

        with Image.open(path) as original:
            image = ImageOps.exif_transpose(original).convert("RGB")
            image.thumbnail((THUMB_WIDTH, THUMB_HEIGHT))

        x = (index % COLUMNS) * CELL_WIDTH
        y = (index // COLUMNS) * CELL_HEIGHT

        image_x = x + (CELL_WIDTH - image.width) // 2
        sheet.paste(image, (image_x, y + 5))

        filename = Path(row["Image"]).name
        confidence = row["Confidence"]

        draw.text(
            (x + 8, y + 245),
            f"{index + 1}. {filename[:43]}",
            fill="black"
        )
        draw.text(
            (x + 8, y + 265),
            f"Actual: Squash | Predicted: Zucchini | {confidence}%",
            fill="black"
        )

    sheet.save(OUTPUT, quality=95)

    print("Images:", image_count)
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()
