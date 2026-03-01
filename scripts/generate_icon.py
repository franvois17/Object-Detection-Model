"""Generate assets/icon.ico using Pillow.

Run once before building with PyInstaller:
    python scripts/generate_icon.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def generate_icon(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sizes = [256, 128, 64, 48, 32, 16]
    images: list[Image.Image] = []

    for size in sizes:
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Dark blue rounded-ish square background
        margin = max(1, size // 16)
        draw.rounded_rectangle(
            [margin, margin, size - margin - 1, size - margin - 1],
            radius=max(2, size // 8),
            fill=(30, 60, 120, 255),
        )

        # "DI" label — use a truetype font if available, else the default
        text = "DI"
        font_size = max(6, int(size * 0.40))
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except OSError:
            font = ImageFont.load_default()

        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        x = (size - text_w) // 2 - bbox[0]
        y = (size - text_h) // 2 - bbox[1]
        draw.text((x, y), text, fill=(255, 255, 255, 255), font=font)

        images.append(img)

    images[0].save(
        output_path,
        format="ICO",
        sizes=[(s, s) for s in sizes],
        append_images=images[1:],
    )
    print(f"Icon saved: {output_path}")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    generate_icon(repo_root / "assets" / "icon.ico")
