"""회전한 상자의 좌표 계산. Qt에 의존하지 않고 단위는 mm."""

from __future__ import annotations

import math

Point = tuple[float, float]
Box = tuple[float, float, float, float, float]  # x, y, w, h, rotation

HANDLES: dict[str, tuple[float, float]] = {
    "nw": (0.0, 0.0), "n": (0.5, 0.0), "ne": (1.0, 0.0), "e": (1.0, 0.5),
    "se": (1.0, 1.0), "s": (0.5, 1.0), "sw": (0.0, 1.0), "w": (0.0, 0.5),
}
CORNER_HANDLES = ("nw", "ne", "se", "sw")
MIN_SIZE_MM = 0.5


def rotate_vec(x: float, y: float, degrees: float) -> Point:
    if not degrees:
        return (x, y)
    angle = math.radians(degrees)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    return (x * cos_a - y * sin_a, x * sin_a + y * cos_a)


def box_of(item) -> Box:
    return (item.x, item.y, item.w, item.h, item.rotation)


def box_center(box: Box) -> Point:
    x, y, w, h, _ = box
    return (x + w / 2.0, y + h / 2.0)


def to_scene(box: Box, fx: float, fy: float) -> Point:
    """상자 안 비율 좌표(0..1)를 회전을 반영한 면 좌표로."""
    x, y, w, h, rotation = box
    cx, cy = box_center(box)
    dx, dy = rotate_vec((fx - 0.5) * w, (fy - 0.5) * h, rotation)
    return (cx + dx, cy + dy)


def to_local(box: Box, px: float, py: float) -> Point:
    """면 좌표를 상자의 회전 전 좌표(왼쪽 위가 0,0)로."""
    x, y, w, h, rotation = box
    cx, cy = box_center(box)
    dx, dy = rotate_vec(px - cx, py - cy, -rotation)
    return (dx + w / 2.0, dy + h / 2.0)


def corners(box: Box) -> list[Point]:
    return [to_scene(box, fx, fy) for fx, fy in ((0, 0), (1, 0), (1, 1), (0, 1))]


def handle_point(box: Box, handle: str) -> Point:
    fx, fy = HANDLES[handle]
    return to_scene(box, fx, fy)


def segment_distance(px: float, py: float, a: Point, b: Point) -> float:
    ax, ay = a
    bx, by = b
    vx, vy = bx - ax, by - ay
    length_sq = vx * vx + vy * vy
    if length_sq <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / length_sq))
    return math.hypot(px - (ax + t * vx), py - (ay + t * vy))


def line_endpoints(item) -> tuple[Point, Point]:
    """선 항목의 양 끝점(면 좌표, 회전 반영)."""
    box = box_of(item)
    if getattr(item, "flip", False):
        return (to_scene(box, 0.0, 1.0), to_scene(box, 1.0, 0.0))
    return (to_scene(box, 0.0, 0.0), to_scene(box, 1.0, 1.0))


def hit_item(item, px: float, py: float, tolerance: float = 0.0) -> bool:
    """점이 항목 위에 있는가. 선은 굵기와 무관하게 tolerance만큼 여유를 둔다."""
    if item.kind == "line":
        a, b = line_endpoints(item)
        reach = max(getattr(item, "width_mm", 0.0) / 2.0, tolerance)
        return segment_distance(px, py, a, b) <= reach
    lx, ly = to_local(box_of(item), px, py)
    pad_x = max(0.0, tolerance - item.w / 2.0) if item.w < 2 * tolerance else 0.0
    pad_y = max(0.0, tolerance - item.h / 2.0) if item.h < 2 * tolerance else 0.0
    return -pad_x <= lx <= item.w + pad_x and -pad_y <= ly <= item.h + pad_y


def resize_box(
    start: Box, handle: str, px: float, py: float, *,
    keep_aspect: bool = False, min_size: float = MIN_SIZE_MM,
) -> tuple[float, float, float, float]:
    """handle을 (px, py)로 끌었을 때의 새 (x, y, w, h). 맞은편 점은 제자리에 둔다."""
    x0, y0, w0, h0, rotation = start
    fx, fy = HANDLES[handle]
    sx = 0 if fx == 0.5 else (1 if fx > 0.5 else -1)
    sy = 0 if fy == 0.5 else (1 if fy > 0.5 else -1)
    cx0, cy0 = x0 + w0 / 2.0, y0 + h0 / 2.0
    anchor = ((1.0 - fx - 0.5) * w0, (1.0 - fy - 0.5) * h0)
    lx, ly = rotate_vec(px - cx0, py - cy0, -rotation)
    vx, vy = lx - anchor[0], ly - anchor[1]

    new_w = max(min_size, sx * vx) if sx else w0
    new_h = max(min_size, sy * vy) if sy else h0
    if keep_aspect and w0 > 0 and h0 > 0:
        if sx and sy:
            scale = max(new_w / w0, new_h / h0)
        elif sx:
            scale = new_w / w0
        else:
            scale = new_h / h0
        scale = max(scale, min_size / min(w0, h0))
        new_w, new_h = w0 * scale, h0 * scale

    center_local = (anchor[0] + sx * new_w / 2.0, anchor[1] + sy * new_h / 2.0)
    dx, dy = rotate_vec(center_local[0], center_local[1], rotation)
    cx, cy = cx0 + dx, cy0 + dy
    return (cx - new_w / 2.0, cy - new_h / 2.0, new_w, new_h)


def normalize_angle(degrees: float) -> float:
    value = degrees % 360.0
    if abs(value - 360.0) < 1e-9:
        return 0.0
    return value


def angle_to(cx: float, cy: float, px: float, py: float) -> float:
    """중심에서 본 점의 방향. 위쪽이 0°, 시계 방향으로 증가."""
    return normalize_angle(math.degrees(math.atan2(py - cy, px - cx)) + 90.0)


def snap_angle(degrees: float, step: float | None = None, soft: float = 0.0) -> float:
    """step이 있으면 그 배수로, 없으면 90° 배수 근처(soft 이내)에서만 달라붙는다."""
    if step:
        return normalize_angle(round(degrees / step) * step)
    if soft > 0:
        nearest = round(degrees / 90.0) * 90.0
        if abs(degrees - nearest) <= soft:
            return normalize_angle(nearest)
    return normalize_angle(degrees)


def line_from_points(a: Point, b: Point) -> dict:
    """두 끝점으로 선 항목의 x, y, w, h, flip, rotation."""
    (x1, y1), (x2, y2) = a, b
    if x1 > x2:
        x1, y1, x2, y2 = x2, y2, x1, y1
    return {
        "x": x1, "y": min(y1, y2), "w": x2 - x1, "h": abs(y2 - y1),
        "flip": y2 < y1, "rotation": 0.0,
    }


def union_bounds(bounds_list) -> tuple[float, float, float, float]:
    boxes = list(bounds_list)
    return (
        min(b[0] for b in boxes), min(b[1] for b in boxes),
        max(b[2] for b in boxes), max(b[3] for b in boxes),
    )
