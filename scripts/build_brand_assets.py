"""앱 아이콘 원본(assets/brand/icon_source.png)에서 배포용 자산을 만든다.

    python scripts/build_brand_assets.py

만드는 것: hana.ico(여러 크기), assets/brand/icon_*.png, 설치 프로그램 그림(bmp).
"""
from __future__ import annotations

import os
import sys

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRAND = os.path.join(ROOT, "assets", "brand")
COBALT = (36, 86, 230)
INK = (27, 36, 48)
ICO_SIZES = [16, 20, 24, 32, 40, 48, 64, 128, 256]


def rounded_master(source: Image.Image, size: int = 1024, radius_ratio: float = 0.235) -> Image.Image:
    """바깥 모서리를 투명하게 한 정사각 아이콘. 원본의 흰 모서리를 잘라 낸다."""
    image = source.convert("RGB").resize((size, size), Image.LANCZOS)
    scale = 4
    mask = Image.new("L", (size * scale, size * scale), 0)
    inset = 3 * scale
    ImageDraw.Draw(mask).rounded_rectangle(
        (inset, inset, size * scale - inset - 1, size * scale - inset - 1),
        radius=int(size * scale * radius_ratio), fill=255)
    mask = mask.resize((size, size), Image.LANCZOS)
    # 원본 모서리의 흰 테두리가 비치지 않도록 가장자리를 바탕색으로 덮은 뒤 자른다.
    base = Image.new("RGB", (size, size), image.getpixel((size // 2, 20)))
    core = Image.new("L", (size, size), 0)
    ImageDraw.Draw(core).rounded_rectangle((14, 14, size - 15, size - 15), radius=int(size * radius_ratio) - 14, fill=255)
    base.paste(image, (0, 0), core)
    result = base.convert("RGBA")
    result.putalpha(mask)
    return result


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    windir = os.environ.get("WINDIR", r"C:\Windows")
    for name in (("malgunbd.ttf",) if bold else ()) + ("malgun.ttf",):
        path = os.path.join(windir, "Fonts", name)
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def ribbon(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    left, top, right, bottom = box
    colors = [(245, 196, 0), (224, 51, 138), (26, 167, 224), INK]
    step = (right - left) / len(colors)
    for index, color in enumerate(colors):
        draw.rectangle((round(left + index * step), top, round(left + (index + 1) * step), bottom), fill=color)


def wizard_image(icon: Image.Image, size: tuple[int, int]) -> Image.Image:
    """설치 마법사 왼쪽 그림: 코발트 바탕, 아이콘, 이름, 아래에 리본 띠."""
    width, height = size
    background = icon.convert("RGB").getpixel((icon.width // 2, icon.height // 12))   # 아이콘 바탕과 같은 파랑
    image = Image.new("RGB", size, background)
    draw = ImageDraw.Draw(image)
    mark = int(width * 0.56)
    logo = icon.resize((mark, mark), Image.LANCZOS)
    # 아이콘의 흰 카드가 돋보이도록 조금 밝은 판 위에 올린다.
    image.paste(logo, ((width - mark) // 2, int(height * 0.16)), logo)
    title = font(max(12, width // 9), bold=True)
    text = "하나스튜디오"
    box = draw.textbbox((0, 0), text, font=title)
    draw.text(((width - (box[2] - box[0])) // 2, int(height * 0.16) + mark + height // 22), text, font=title, fill="white")
    sub = font(max(9, width // 15))
    text2 = "카드 디자인과 인쇄"
    box2 = draw.textbbox((0, 0), text2, font=sub)
    draw.text(((width - (box2[2] - box2[0])) // 2, int(height * 0.16) + mark + height // 22 + (box[3] - box[1]) + height // 40),
              text2, font=sub, fill=(214, 224, 252))
    ribbon(draw, (0, height - max(6, height // 40), width, height))
    return image


def main() -> int:
    source_path = os.path.join(BRAND, "icon_source.png")
    if not os.path.exists(source_path):
        print(f"원본이 없습니다: {source_path}")
        return 1
    master = rounded_master(Image.open(source_path))
    master.save(os.path.join(BRAND, "icon_1024.png"))
    for size in (512, 256, 128, 64, 48, 32):
        master.resize((size, size), Image.LANCZOS).save(os.path.join(BRAND, f"icon_{size}.png"))
    master.save(os.path.join(ROOT, "hana.ico"), format="ICO", sizes=[(s, s) for s in ICO_SIZES])
    for scale, suffix in ((1, ""), (2, "_2x")):
        wizard_image(master, (164 * scale, 314 * scale)).save(os.path.join(BRAND, f"installer_side{suffix}.bmp"))
        small = Image.new("RGB", (55 * scale, 58 * scale), "white")
        mark = master.resize((46 * scale, 46 * scale), Image.LANCZOS)
        small.paste(mark, ((55 * scale - 46 * scale) // 2, (58 * scale - 46 * scale) // 2), mark)
        small.save(os.path.join(BRAND, f"installer_small{suffix}.bmp"))
    print("만든 파일:", ", ".join(sorted(os.listdir(BRAND))), "+ hana.ico")
    return 0


if __name__ == "__main__":
    sys.exit(main())
