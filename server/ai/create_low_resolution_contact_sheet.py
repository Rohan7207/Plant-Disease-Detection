
import csv
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "datasets/metadata/image_quality_test_report.csv"
OUTPUT = ROOT / "datasets/metadata/low_resolution_contact_sheet.jpg"

with REPORT.open("r", newline="", encoding="utf-8") as file:
    rows = list(csv.DictReader(file))

failed = [
    row for row in rows
    if "Image dimensions are too small" in row["Errors"]
]

columns = 3
cell_width = 420
cell_height = 300
thumb_width = 390
thumb_height = 235

sheet = Image.new(
    "RGB",
    (columns * cell_width, ((len(failed) + columns - 1) // columns) * cell_height),
    "white"
)
draw = ImageDraw.Draw(sheet)

for index, row in enumerate(failed):
    image_path = ROOT / row["Image"]
    x = (index % columns) * cell_width
    y = (index // columns) * cell_height

    try:
        with Image.open(image_path) as original:
            image = ImageOps.exif_transpose(original).convert("RGB")
            image.thumbnail((thumb_width, thumb_height))
            sheet.paste(image, (x + 10, y + 10))

        draw.text(
            (x + 10, y + 250),
            f'{image_path.name}',
            fill="black"
        )
        draw.text(
            (x + 10, y + 270),
            f'{row["Width"]} x {row["Height"]}',
            fill="black"
        )
    except Exception as error:
        draw.text((x + 10, y + 10), f"Could not open: {error}", fill="red")

sheet.save(OUTPUT, quality=95)
print("Images inspected:", len(failed))
print("Contact sheet saved:", OUTPUT)
