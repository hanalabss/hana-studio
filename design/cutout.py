"""배경을 지운 그림 만들기(Qt 없이 numpy·Pillow만 쓴다).

두 가지 방법이 같은 결과물(투명한 배경의 RGBA 그림)을 만든다.

- "ai"    : 모델이 준 알파(255 = 사람·물건)에 '지우는 정도'를 적용한다.
- "color" : 바탕색과 비슷한 곳만 지운다. 로고·도장·서명처럼 바탕이 한 가지 색인 그림용.

가장자리는 반투명으로 남기고, 그 자리에 섞여 있던 옛 배경색은 걷어 낸다.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageFilter

METHOD_AI = "ai"
METHOD_COLOR = "color"

DEFAULT_THRESHOLD = 200
MIN_THRESHOLD, MAX_THRESHOLD = 5, 250
DEFAULT_TOLERANCE = 30
MIN_TOLERANCE, MAX_TOLERANCE = 2, 150
# 배경을 지운 그림의 긴 변 최대 픽셀. 카드(86mm)를 600dpi로 찍어도 남는 크기다.
MAX_CUTOUT_PX = 2400
# AI가 이보다 적게(또는 많이) 남기면 주인공을 못 찾은 것으로 본다.
WIPEOUT_BELOW = 0.15
UNTOUCHED_ABOVE = 0.97
# 바탕이 한 가지 색인 그림에서 그림 부분이 이보다 많이 지워지면 사용자에게 물어본다.
LOST_ABOVE = 0.2

# 사진 항목의 extra에 적어 두는 이름. 파일에 함께 저장된다.
THRESHOLD_KEY = "cutout_threshold"
METHOD_KEY = "cutout_method"
COLOR_KEY = "cutout_color"
TOLERANCE_KEY = "cutout_tolerance"
_KEYS = (THRESHOLD_KEY, METHOD_KEY, COLOR_KEY, TOLERANCE_KEY)

Color = tuple[int, int, int]


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _clamp(value: float, low: int, high: int) -> int:
    return int(min(high, max(low, round(value))))


def parse_color(value: Any) -> Color | None:
    if not isinstance(value, str) or len(value) != 7 or not value.startswith("#"):
        return None
    try:
        return (int(value[1:3], 16), int(value[3:5], 16), int(value[5:7], 16))
    except ValueError:
        return None


def format_color(color: Color) -> str:
    return "#{:02x}{:02x}{:02x}".format(*(int(c) for c in color))


@dataclass(frozen=True)
class CutoutSettings:
    """배경을 지우는 방법과 정도. 사진 항목의 extra에 적어 둔다."""

    method: str = METHOD_AI
    threshold: int = DEFAULT_THRESHOLD     # ai: 클수록 더 지운다
    color: Color | None = None             # color: 지울 색. None이면 그림 모서리 색
    tolerance: int = DEFAULT_TOLERANCE     # color: 클수록 비슷한 색까지 지운다

    def with_changes(self, **changes) -> "CutoutSettings":
        return replace(self, **changes)

    @classmethod
    def from_extra(cls, extra: Mapping[str, Any], default_threshold: int = DEFAULT_THRESHOLD) -> "CutoutSettings":
        method = METHOD_COLOR if extra.get(METHOD_KEY) == METHOD_COLOR else METHOD_AI
        threshold = _number(extra.get(THRESHOLD_KEY))
        tolerance = _number(extra.get(TOLERANCE_KEY))
        return cls(
            method=method,
            threshold=_clamp(default_threshold if threshold is None else threshold, MIN_THRESHOLD, MAX_THRESHOLD),
            color=parse_color(extra.get(COLOR_KEY)),
            tolerance=_clamp(DEFAULT_TOLERANCE if tolerance is None else tolerance, MIN_TOLERANCE, MAX_TOLERANCE),
        )

    def to_extra(self, extra: Mapping[str, Any]) -> dict:
        """이 설정을 적어 넣은 새 extra. 배경 지우기와 상관없는 값은 그대로 둔다."""
        out = without_settings(extra)
        if self.method == METHOD_COLOR:
            out[METHOD_KEY] = METHOD_COLOR
            out[TOLERANCE_KEY] = int(self.tolerance)
            if self.color is not None:
                out[COLOR_KEY] = format_color(self.color)
        else:
            out[THRESHOLD_KEY] = int(self.threshold)
        return out


def without_settings(extra: Mapping[str, Any]) -> dict:
    return {key: value for key, value in extra.items() if key not in _KEYS}


# ---------------------------------------------------------------- 알파

def flat_background(rgb: np.ndarray, alpha: np.ndarray) -> Color | None:
    """모델이 확실히 배경이라고 본 곳이 한 가지 색이면 그 색. 여러 색이면 None."""
    sure = np.asarray(alpha) < 8
    if sure.sum() < max(16, sure.size // 100):
        return None
    colors = np.asarray(rgb)[sure].astype(np.int16)
    middle = np.median(colors, axis=0)
    spread = np.abs(colors - middle).max(axis=1)
    # 지워진 그림 조각(다른 색)이 섞여 있어도 바탕색을 알아보도록, 대부분(60%)이 같은 색이면 한 가지 색으로 본다.
    if float((spread <= 14).mean()) < 0.6:
        return None
    return (int(middle[0]), int(middle[1]), int(middle[2]))


def _fill_pockets(keep: np.ndarray, alpha: np.ndarray, threshold: float, sure_bg: np.ndarray) -> np.ndarray:
    """남는 부분에 둘러싸인 '애매한' 구멍을 메운다.

    모델은 배경과 색이 비슷한 옷 같은 곳을 반쯤만 사람으로 보곤 한다. 그런 곳은 테두리 안쪽에 갇혀 있고
    확실한 배경(알파가 아주 낮은 곳)이 섞여 있지 않다. 팔과 몸 사이처럼 진짜 배경인 구멍은 그대로 둔다.
    """
    try:
        import cv2
    except ImportError:
        return keep
    count, labels = cv2.connectedComponents((~keep).astype(np.uint8), connectivity=4)
    if count <= 1:
        return keep
    flat = labels.ravel()
    sizes = np.bincount(flat, minlength=count)
    sure = np.bincount(flat, weights=sure_bg.ravel(), minlength=count)
    border = np.zeros(count, dtype=bool)
    for edge in (labels[0], labels[-1], labels[:, 0], labels[:, -1]):
        border[edge] = True
    pocket = ~border & (sure < np.maximum(4.0, sizes * 0.02))
    pocket[0] = False                      # 0번은 남는 부분 자체
    return keep | pocket[labels]


def soft_alpha(alpha: np.ndarray, threshold: int, rgb: np.ndarray | None = None) -> np.ndarray:
    """'지우는 정도'를 적용한 알파. 안쪽은 또렷하게 남기고 가장자리만 부드럽게 한다.

    모델이 준 알파를 그대로 투명도로 쓰면 모델이 망설인 곳(배경과 비슷한 색의 옷 등)이 반투명해진다.
    그래서 기준값보다 높은 곳은 모두 남기고, 갇힌 구멍을 메운 뒤, 경계선만 한두 픽셀 폭으로 누그러뜨린다.
    기준값을 올리면 흐릿한 곳부터 지워지고 내리면 더 많이 남는다.
    rgb를 주면, 배경이 한 가지 색일 때 기준값에 못 미친 곳 가운데 그 색과 똑같은 곳은 확실한 배경으로 본다
    (머리카락과 목 사이처럼 갇힌 배경이 메워지지 않게).
    """
    alpha = np.asarray(alpha, dtype=np.uint8)
    t = float(_clamp(threshold, MIN_THRESHOLD, MAX_THRESHOLD))
    keep = alpha > t
    sure_bg = alpha < t / 4.0
    key = flat_background(rgb, alpha) if rgb is not None else None
    if key is not None:
        distance = np.abs(np.asarray(rgb, dtype=np.int16) - np.array(key, dtype=np.int16)).max(axis=2)
        sure_bg |= ~keep & (distance <= 12)
    keep = _fill_pockets(keep, alpha, t, sure_bg)
    scale = max(1.0, max(alpha.shape) / 1000.0)
    mask = Image.fromarray(np.where(keep, 255, 0).astype(np.uint8), "L")
    return np.asarray(mask.filter(ImageFilter.GaussianBlur(0.7 * scale)), dtype=np.uint8)


def white_mask(alpha: np.ndarray) -> np.ndarray:
    """화이트 층 마스크: 흰색(255) = 흰 잉크를 까는 곳 = 그림이 절반 넘게 남은 곳."""
    return np.where(np.asarray(alpha) > 127, 255, 0).astype(np.uint8)


def kept_ratio(alpha: np.ndarray) -> float:
    """그림에서 남는 부분의 비율(0..1)."""
    alpha = np.asarray(alpha)
    return float(alpha.mean()) / 255.0 if alpha.size else 0.0


def verdict(ratio: float, lost: float = 0.0) -> str:
    """AI 결과를 그대로 써도 되는지.

    "ok" | "wipeout"(거의 다 지워짐) | "untouched"(거의 그대로) | "partial"(그림의 일부가 함께 지워짐)
    """
    if ratio < WIPEOUT_BELOW:
        return "wipeout"
    if ratio > UNTOUCHED_ABOVE:
        return "untouched"
    if lost > LOST_ABOVE:
        return "partial"
    return "ok"


def lost_ratio(rgb: np.ndarray, base_alpha: np.ndarray, final_alpha: np.ndarray) -> float:
    """바탕이 한 가지 색인 그림에서, 바탕색이 아닌 곳(=그림) 가운데 지워진 비율. 알 수 없으면 0.

    도장·로고를 AI에 맡기면 테두리만 남기고 속을 지우곤 한다. 그런 결과를 가려낸다.
    """
    key = flat_background(rgb, base_alpha)
    if key is None:
        return 0.0
    distance = np.abs(np.asarray(rgb, dtype=np.int16) - np.array(key, dtype=np.int16)).max(axis=2)
    drawn = distance > 48
    total = int(drawn.sum())
    if total < drawn.size // 100:
        return 0.0
    return float((drawn & (np.asarray(final_alpha) < 128)).sum()) / total


def corner_color(rgb: np.ndarray) -> Color:
    """그림 네 모서리에서 가장 흔한 색. 바탕색을 짐작할 때 쓴다."""
    rgb = np.asarray(rgb)
    height, width = rgb.shape[:2]
    size = max(1, min(height, width) // 40)
    patches = [rgb[:size, :size], rgb[:size, -size:], rgb[-size:, :size], rgb[-size:, -size:]]
    medians = [np.median(patch.reshape(-1, 3), axis=0) for patch in patches]
    # 다른 모서리들과 가장 가까운 것을 고른다(한쪽 모서리에 그림이 걸쳐 있어도 흔들리지 않게).
    best = min(medians, key=lambda color: sorted(float(np.abs(color - other).max()) for other in medians)[2])
    return (int(best[0]), int(best[1]), int(best[2]))


def color_key_alpha(rgb: np.ndarray, color: Color, tolerance: int) -> np.ndarray:
    """바탕색과 비슷한 곳을 지운 알파. 색이 섞인 가장자리는 섞인 만큼만 남긴다."""
    tolerance = float(_clamp(tolerance, MIN_TOLERANCE, MAX_TOLERANCE))
    key = np.array(color, dtype=np.int16)
    distance = np.abs(np.asarray(rgb, dtype=np.int16) - key).max(axis=2).astype(np.uint8)
    # 가까운 곳에서 바탕색과 가장 다른 값을 '그림 본래 색'의 거리로 본다. 가장자리의 섞인 정도 = 거리 ÷ 그 값.
    nearby = np.asarray(Image.fromarray(distance, "L").filter(ImageFilter.MaxFilter(5)), dtype=np.float32)
    span = np.maximum(nearby - tolerance, 1.0)
    values = (distance.astype(np.float32) - tolerance) / span
    return (np.clip(values, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


# ---------------------------------------------------------------- 가장자리 색 정리

def _box_blur(values: np.ndarray, radius: int) -> np.ndarray:
    """가로세로 (2r+1) 평균. values: float32 (H, W, C)."""
    try:
        import cv2
    except ImportError:
        cv2 = None
    if cv2 is not None:
        size = 2 * radius + 1
        blurred = cv2.blur(values, (size, size), borderType=cv2.BORDER_REPLICATE)
        return blurred.reshape(values.shape)
    for axis in (0, 1):
        pad = [(0, 0)] * values.ndim
        pad[axis] = (radius + 1, radius)
        summed = np.cumsum(np.pad(values, pad, mode="edge"), axis=axis, dtype=np.float64)
        count = values.shape[axis]
        upper = np.take(summed, range(2 * radius + 1, 2 * radius + 1 + count), axis=axis)
        lower = np.take(summed, range(0, count), axis=axis)
        values = ((upper - lower) / (2 * radius + 1)).astype(np.float32)
    return values


def unmix_key_color(rgb: np.ndarray, alpha: np.ndarray, color: Color) -> np.ndarray:
    """반투명 가장자리에 섞인 바탕색을 덜어 낸다: 본래 색 = 바탕색 + (보이는 색 - 바탕색) ÷ 알파."""
    a = np.asarray(alpha, dtype=np.float32)[..., None] / 255.0
    key = np.array(color, dtype=np.float32)
    source = np.asarray(rgb, dtype=np.float32)
    mixed = (a > 0.0) & (a < 1.0)
    restored = key + (source - key) / np.maximum(a, 1.0 / 255.0)
    return np.clip(np.where(mixed, restored, source), 0.0, 255.0).astype(np.uint8)


def spread_solid_color(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """가장자리의 색을 바로 안쪽의 색으로 바꾼다. 옛 배경색이 섞인 테두리가 사라진다.

    경계선에서 한두 픽셀 안쪽까지는 배경색이 섞여 있다고 보고, 그보다 안쪽 색을 번지게 해 덮는다.
    안쪽이 없는 가는 선(머리카락 등)은 그대로 둔다.
    """
    alpha = np.asarray(alpha, dtype=np.uint8)
    source = np.asarray(rgb, dtype=np.float32)
    reach = max(1, round(max(alpha.shape) / 1000.0))
    solid_mask = Image.fromarray(np.where(alpha >= 250, 255, 0).astype(np.uint8), "L")
    inner = np.asarray(solid_mask.filter(ImageFilter.MinFilter(2 * reach + 1))) > 0
    edge = (alpha > 0) & ~inner
    if not edge.any() or not inner.any():
        return np.asarray(rgb, dtype=np.uint8)
    solid = inner.astype(np.float32)[..., None]
    out = source.copy()
    todo = edge
    radius = reach + 1
    for _ in range(2):                      # 가까운 색부터, 못 채운 바깥쪽은 조금 더 멀리서
        weight = _box_blur(solid, radius)
        colors = _box_blur(source * solid, radius)
        found = todo & (weight[..., 0] > 0.02)
        out[found] = (colors / np.maximum(weight, 1e-6))[found]
        todo = todo & ~found
        if not todo.any():
            break
        radius *= 3
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


# ---------------------------------------------------------------- 결과물

def cut(rgb: np.ndarray, base_alpha: np.ndarray | None, settings: CutoutSettings) -> tuple[np.ndarray, np.ndarray]:
    """(정리된 RGB, 알파)를 만든다. base_alpha는 "ai"일 때 모델이 준 알파."""
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    if settings.method == METHOD_COLOR:
        color = settings.color if settings.color is not None else corner_color(rgb)
        alpha = color_key_alpha(rgb, color, settings.tolerance)
        return unmix_key_color(rgb, alpha, color), alpha
    if base_alpha is None:
        raise ValueError("배경을 가려낸 결과가 없습니다.")
    alpha = soft_alpha(base_alpha, settings.threshold, rgb)
    return spread_solid_color(rgb, alpha), alpha


def compose(rgb: np.ndarray, alpha: np.ndarray) -> Image.Image:
    return Image.fromarray(np.dstack([rgb, alpha]), "RGBA")


def work_size(size: tuple[int, int], limit: int = MAX_CUTOUT_PX) -> tuple[int, int]:
    """배경을 지운 그림의 크기. 너무 큰 그림만 비율을 지켜 줄인다."""
    width, height = size
    scale = min(1.0, limit / max(width, height, 1))
    return (max(1, round(width * scale)), max(1, round(height * scale)))
