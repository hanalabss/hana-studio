"""선택한 항목에 붙는 꼬리표("명단: 부서" 등)를 어디에 그릴지 정한다. 좌표는 뷰포트 픽셀."""

from __future__ import annotations

from typing import Iterable

Rect = tuple[float, float, float, float]   # (left, top, right, bottom)

ABOVE, BELOW, INSIDE = "above", "below", "inside"
GAP_ABOVE_PX = 7.0
GAP_BELOW_PX = 9.0
INSIDE_PAD_PX = 4.0
# 상자 높이가 꼬리표 높이의 이 배수는 되어야 안쪽 귀퉁이에 넣는다(짧은 글자 칸은 내용이 가려진다).
INSIDE_MIN_HEIGHTS = 3.0
_MIN_SIDE_PX = 2.0
_BACKDROP_SLACK_PX = 1.0


def _intersects(a: Rect, b: Rect) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _within(inner: Rect, outer: Rect) -> bool:
    return inner[0] >= outer[0] and inner[1] >= outer[1] and inner[2] <= outer[2] and inner[3] <= outer[3]


def overlap_area(tag: Rect, obstacle: Rect) -> float:
    """꼬리표가 obstacle을 가리는 넓이(px²). 선처럼 납작한 것은 조금 부풀려서 본다."""
    left, top, right, bottom = obstacle
    if right - left < _MIN_SIDE_PX:
        middle = (left + right) / 2.0
        left, right = middle - _MIN_SIDE_PX / 2.0, middle + _MIN_SIDE_PX / 2.0
    if bottom - top < _MIN_SIDE_PX:
        middle = (top + bottom) / 2.0
        top, bottom = middle - _MIN_SIDE_PX / 2.0, middle + _MIN_SIDE_PX / 2.0
    overlap_w = min(tag[2], right) - max(tag[0], left)
    overlap_h = min(tag[3], bottom) - max(tag[1], top)
    return overlap_w * overlap_h if overlap_w > 0 and overlap_h > 0 else 0.0


def _is_backdrop(obstacle: Rect, box: Rect) -> bool:
    """선택 상자를 통째로 품은 항목(카드 배경, 넓은 띠). 어디에 두든 가려지므로 걸림돌로 치지 않는다."""
    return (obstacle[0] <= box[0] + _BACKDROP_SLACK_PX and obstacle[1] <= box[1] + _BACKDROP_SLACK_PX
            and obstacle[2] >= box[2] - _BACKDROP_SLACK_PX and obstacle[3] >= box[3] - _BACKDROP_SLACK_PX)


def _shift_into(rect: Rect, viewport: Rect | None) -> Rect:
    """가로로만 화면 안으로 민다(세로로 밀면 선택 상자와 겹친다)."""
    if viewport is None:
        return rect
    left, top, right, bottom = rect
    shift = 0.0
    if right > viewport[2]:
        shift = viewport[2] - right
    if left + shift < viewport[0]:
        shift = viewport[0] - left
    return (left + shift, top, right + shift, bottom)


def place_tag(
    box: Rect, tag_size: tuple[float, float], obstacles: Iterable[Rect] = (), *,
    viewport: Rect | None = None, keep_clear: Iterable[Rect] = (), allow_inside: bool = True,
) -> tuple[str, Rect]:
    """꼬리표 자리를 고른다. ("above" | "below" | "inside", 꼬리표 사각형)을 돌려준다.

    box: 선택 상자(회전했으면 경계 상자). obstacles: 같은 면의 다른 항목들(선택 상자를 통째로 품은
    배경은 알아서 뺀다).
    keep_clear: 조금도 가리면 안 되는 곳(회전 손잡이 등). viewport: 보이는 영역.

    위 → 아래 → 안쪽 왼쪽 위 순서로, 화면 안에 들어오고 다른 항목을 가리지 않는 첫 자리를 쓴다.
    안쪽은 상자가 충분히 클 때만 후보가 된다. 모두 걸리면 가장 덜 가리는 자리를 쓴다.
    """
    width, height = tag_size
    left, top, right, bottom = box
    obstacles = [obstacle for obstacle in obstacles if not _is_backdrop(obstacle, box)]
    keep_clear = list(keep_clear)

    candidates: list[tuple[str, Rect]] = [
        (ABOVE, (left, top - GAP_ABOVE_PX - height, left + width, top - GAP_ABOVE_PX)),
        (BELOW, (left, bottom + GAP_BELOW_PX, left + width, bottom + GAP_BELOW_PX + height)),
    ]
    fits_inside = (
        allow_inside
        and bottom - top >= height * INSIDE_MIN_HEIGHTS
        and right - left >= width + 2 * INSIDE_PAD_PX
    )
    if fits_inside:
        x, y = left + INSIDE_PAD_PX, top + INSIDE_PAD_PX
        candidates.append((INSIDE, (x, y, x + width, y + height)))

    scored: list[tuple[float, int, str, Rect]] = []
    for order, (where, rect) in enumerate(candidates):
        if where != INSIDE:
            rect = _shift_into(rect, viewport)
        visible = viewport is None or _within(rect, viewport)
        blocked = any(_intersects(rect, area) for area in keep_clear)
        # 안쪽 자리는 선택한 항목 자신을 가릴 뿐이라 다른 항목과 견주지 않는다.
        covered = 0.0 if where == INSIDE else sum(overlap_area(rect, o) for o in obstacles)
        if visible and not blocked and covered <= 0.0:
            return where, rect
        penalty = covered + (1e9 if not visible else 0.0) + (1e8 if blocked else 0.0)
        scored.append((penalty, order, where, rect))
    _, _, where, rect = min(scored)
    return where, rect
