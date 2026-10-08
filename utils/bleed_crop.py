"""
utils/bleed_crop.py
도련(bleed)이 포함된 디자인 이미지에서 칼선 안쪽 영역만 잘라낸다.

예) 58x90mm 디자인 (사방 2mm 도련) -> 54x86mm 칼선 영역
이미지 비율이 디자인 규격(세로 58:90 또는 가로 90:58)과 일치할 때만 자른다.
이미 카드 크기로 만든 이미지는 그대로 둔다.
"""

from typing import Optional, Tuple
from PIL import Image

from config import config

# 비율 일치 판정 허용 오차 (상대값). 리사이즈 반올림 오차 정도만 허용
_RATIO_TOLERANCE = 0.005


def _bleed_settings() -> Tuple[bool, float, float, float]:
    enabled = bool(config.get("printer.bleed_crop_enabled", True))
    design_w = float(config.get("printer.design_width_mm", 58.0))
    design_h = float(config.get("printer.design_height_mm", 90.0))
    bleed = float(config.get("printer.bleed_mm", 2.0))
    return enabled, design_w, design_h, bleed


def _matches(ratio: float, target: float) -> bool:
    return abs(ratio - target) / target <= _RATIO_TOLERANCE


def get_bleed_crop_box(width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    """도련 제거용 crop box (left, top, right, bottom). 대상이 아니면 None."""
    enabled, design_w, design_h, bleed = _bleed_settings()
    if not enabled or width <= 0 or height <= 0 or bleed <= 0:
        return None

    ratio = width / height
    if _matches(ratio, design_w / design_h):
        mm_w, mm_h = design_w, design_h          # 세로형 디자인
    elif _matches(ratio, design_h / design_w):
        mm_w, mm_h = design_h, design_w          # 가로형 디자인
    else:
        return None

    bleed_x = round(width * bleed / mm_w)
    bleed_y = round(height * bleed / mm_h)
    if width - 2 * bleed_x <= 0 or height - 2 * bleed_y <= 0:
        return None
    return bleed_x, bleed_y, width - bleed_x, height - bleed_y


def crop_bleed(image: Image.Image) -> Image.Image:
    """도련 규격 이미지면 칼선 안쪽만 잘라서 반환, 아니면 원본 반환."""
    box = get_bleed_crop_box(image.width, image.height)
    if box is None:
        return image
    print(f"[BLEED] 도련 제거: {image.size} -> crop {box}")
    return image.crop(box)
