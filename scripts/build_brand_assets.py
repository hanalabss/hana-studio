"""1.0 로고(assets/brand/logo_v1.png: 손글씨 hana + STUDIO + 사각 틀)에서 배포용 자산을 만든다.

    python scripts/build_brand_assets.py

만드는 것: assets/brand/icon_source.png, hana.ico(여러 크기), assets/brand/icon_*.png, 설치 프로그램 그림(bmp).

아이콘은 크기에 따라 세 가지 모양을 쓴다(작게 줄이면 가는 손글씨가 뭉개지기 때문이다).
    큰 것(128 이상)   틀 + hana + STUDIO, 틀의 아랫변은 인쇄 리본 4색 띠
    중간(40~64)       hana 손글씨와 리본 띠만
    작은 것(32 이하)  손글씨 h 한 글자
"""
from __future__ import annotations

import os
import sys

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRAND = os.path.join(ROOT, "assets", "brand")
COBALT = (36, 86, 230)
INK = (27, 36, 48)
ICO_SIZES = [16, 20, 24, 32, 40, 48, 64, 128, 256]
RIBBON = [(245, 196, 0), (224, 51, 138), (26, 167, 224), INK]
LOGO_V1 = os.path.join(BRAND, "logo_v1.png")     # 256×256, 흰 바탕에 먹색
SCRIPT_BOX = (20, 52, 250, 150)                  # logo_v1 안에서 손글씨 hana가 있는 자리
LETTER_H_BOX = (22, 54, 79, 148)                 # 그중 첫 글자 h


def ink_mask(box: tuple[int, int, int, int], scale: float, grow: int = 0) -> Image.Image:
    """logo_v1의 box 부분을 scale배로 키운 먹 자리(L). grow(픽셀)만큼 획을 굵게 한다.

    원본이 256픽셀뿐이라 그대로 키우면 흐리다. 키운 뒤 살짝 흐리게 하고 경계를 다시 세워 또렷하게 만든다.
    """
    ink = ImageOps.invert(Image.open(LOGO_V1).convert("L"))
    wipe = ImageDraw.Draw(ink)                      # 오른쪽 틀 선 조각은 뺀다(글씨 꼬리만 남긴다)
    wipe.rectangle((238, 0, 255, 110), fill=0)
    wipe.rectangle((238, 150, 255, 255), fill=0)
    ink = ink.crop(box)
    big = ink.resize((round(ink.width * scale), round(ink.height * scale)), Image.BICUBIC)
    if grow > 0:
        big = big.filter(ImageFilter.MaxFilter(2 * grow + 1))
    big = big.filter(ImageFilter.GaussianBlur(max(1.0, scale * 0.3)))
    low, high = 70, 120
    return big.point(lambda v: 0 if v <= low else 255 if v >= high else round((v - low) * 255 / (high - low)))


def serif(size: int) -> ImageFont.FreeTypeFont:
    windir = os.environ.get("WINDIR", r"C:\Windows")
    for name in ("georgia.ttf", "cambria.ttc", "times.ttf"):
        path = os.path.join(windir, "Fonts", name)
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def ribbon_bar(image: Image.Image, box: tuple[float, float, float, float]) -> None:
    """인쇄 리본 4색(Y·M·C·K) 띠. 양 끝을 둥글게 한다."""
    left, top, right, bottom = (round(v) for v in box)
    bar = Image.new("RGB", (right - left, bottom - top))
    ribbon(ImageDraw.Draw(bar), (0, 0, bar.width, bar.height))
    mask = Image.new("L", bar.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, bar.width - 1, bar.height - 1), radius=bar.height // 2, fill=255)
    image.paste(bar, (left, top), mask)


def full_art(size: int = 1024) -> Image.Image:
    """큰 아이콘: 1.0 로고의 짜임(틀·hana·STUDIO)을 코발트 바탕에 흰색으로 옮기고, 틀 아랫변을 리본 띠로 바꾼다."""
    image = Image.new("RGB", (size, size), COBALT)
    draw = ImageDraw.Draw(image)
    scale = size * 3.02 / 1024
    ox, oy = size / 2 - 128.5 * scale, size / 2 - 123.0 * scale

    def at(x: float, y: float) -> tuple[float, float]:
        return ox + x * scale, oy + y * scale

    line = max(2, round(size * 0.0125))
    left, right, top, bottom = 16, 241, 25, 228
    for (x1, y1), (x2, y2) in (((left, top), (right, top)),
                               ((left, top), (left, 70)), ((left, 183), (left, bottom)),
                               ((right, top), (right, 70)), ((right, 183), (right, bottom))):
        (ax, ay), (bx, by) = at(x1, y1), at(x2, y2)
        draw.rounded_rectangle((ax - line / 2, ay - line / 2, bx + line / 2, by + line / 2), radius=line / 2, fill="white")
    (ax, ay), (bx, _) = at(left, bottom), at(right, bottom)
    ribbon_bar(image, (ax - line / 2, ay - line * 1.1, bx + line / 2, ay + line * 1.1))

    script = ink_mask(SCRIPT_BOX, scale, grow=max(1, round(size * 0.002)))
    image.paste("white", tuple(round(v) for v in at(SCRIPT_BOX[0], SCRIPT_BOX[1])), script)

    letters, face = "STUDIO", serif(round(size * 0.052))
    tracking = size * 0.021
    widths = [draw.textlength(ch, font=face) for ch in letters]
    x = at(240, 0)[0] - (sum(widths) + tracking * (len(letters) - 1))
    baseline = at(0, 181)[1]
    for ch, width in zip(letters, widths):
        draw.text((x, baseline), ch, font=face, fill="white", anchor="ls")
        x += width + tracking
    return image


def medium_art(size: int = 1024) -> Image.Image:
    """중간 아이콘: hana 손글씨를 크게, 그 아래 리본 띠."""
    image = Image.new("RGB", (size, size), COBALT)
    scale = size * 0.83 / (SCRIPT_BOX[2] - SCRIPT_BOX[0])
    script = ink_mask(SCRIPT_BOX, scale, grow=round(size * 0.009))
    left, top = (size - script.width) // 2, round(size * 0.24)
    image.paste("white", (left, top), script)
    bar_top = top + script.height + size * 0.06
    ribbon_bar(image, (size * 0.2, bar_top, size * 0.8, bar_top + size * 0.07))
    return image


def small_art(size: int = 1024) -> Image.Image:
    """작은 아이콘: 손글씨 h 한 글자."""
    image = Image.new("RGB", (size, size), COBALT)
    scale = size * 0.66 / (LETTER_H_BOX[3] - LETTER_H_BOX[1])
    letter = ink_mask(LETTER_H_BOX, scale, grow=round(size * 0.03))
    image.paste("white", ((size - letter.width) // 2 + round(size * 0.02), (size - letter.height) // 2), letter)
    return image


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
    step = (right - left) / len(RIBBON)
    for index, color in enumerate(RIBBON):
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
    text = "hanaStudio"
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
    if not os.path.exists(LOGO_V1):
        print(f"원본이 없습니다: {LOGO_V1}")
        return 1
    full_art().save(os.path.join(BRAND, "icon_source.png"))
    masters = {name: rounded_master(art()) for name, art in
               (("small", small_art), ("medium", medium_art), ("full", full_art))}

    def sized(size: int) -> Image.Image:
        kind = "small" if size <= 32 else "medium" if size <= 64 else "full"
        return masters[kind].resize((size, size), Image.LANCZOS)

    master = masters["full"]
    master.save(os.path.join(BRAND, "icon_1024.png"))
    for size in (512, 256, 128, 64, 48, 32):
        sized(size).save(os.path.join(BRAND, f"icon_{size}.png"))
    sized(256).save(os.path.join(ROOT, "hana.ico"), format="ICO", sizes=[(s, s) for s in ICO_SIZES],
                    append_images=[sized(s) for s in ICO_SIZES if s != 256])
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
