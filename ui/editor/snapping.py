"""끌 때 달라붙는 기준선 계산. Qt에 의존하지 않고 단위는 mm."""

from __future__ import annotations

from dataclasses import dataclass, field

Bounds = tuple[float, float, float, float]
Target = tuple[float, float, float]  # 위치, 기준이 된 대상의 시작, 끝(안내선 길이용)

_EPS = 1e-6


@dataclass(frozen=True)
class Guide:
    """orientation "v"는 x=pos인 세로선, "h"는 y=pos인 가로선."""

    orientation: str
    pos: float
    start: float
    end: float


@dataclass
class SnapTargets:
    xs: list[Target] = field(default_factory=list)
    ys: list[Target] = field(default_factory=list)


def collect_targets(document, side_key: str, exclude_ids=()) -> SnapTargets:
    """카드 가장자리·가운데, 칼선, 다른 항목의 가장자리·가운데."""
    excluded = set(exclude_ids)
    width, height = document.side_size_mm(side_key)
    targets = SnapTargets()
    for x in (0.0, width / 2.0, width):
        targets.xs.append((x, 0.0, height))
    for y in (0.0, height / 2.0, height):
        targets.ys.append((y, 0.0, width))
    if document.bleed_mm > 0:
        left, top, right, bottom = document.side_bounds(side_key, inside_bleed=True)
        targets.xs += [(left, 0.0, height), (right, 0.0, height)]
        targets.ys += [(top, 0.0, width), (bottom, 0.0, width)]
    for item in document.side(side_key).items:
        if item.id in excluded or not item.visible:
            continue
        left, top, right, bottom = item.bounds()
        for x in (left, (left + right) / 2.0, right):
            targets.xs.append((x, top, bottom))
        for y in (top, (top + bottom) / 2.0, bottom):
            targets.ys.append((y, left, right))
    return targets


def _best_shift(lines: tuple[float, ...], targets: list[Target], threshold: float) -> float | None:
    best: float | None = None
    for line in lines:
        for pos, _, _ in targets:
            delta = pos - line
            if abs(delta) <= threshold and (best is None or abs(delta) < abs(best) - _EPS):
                best = delta
    return best


def _guides_for(
    orientation: str, lines: tuple[float, ...], shift: float, targets: list[Target],
    cross_start: float, cross_end: float,
) -> list[Guide]:
    merged: dict[float, list[float]] = {}
    for line in lines:
        moved = line + shift
        for pos, start, end in targets:
            if abs(pos - moved) <= 1e-4:
                span = merged.setdefault(round(pos, 4), [cross_start, cross_end])
                span[0] = min(span[0], start)
                span[1] = max(span[1], end)
    return [Guide(orientation, pos, span[0], span[1]) for pos, span in sorted(merged.items())]


def _grid_shift(value: float, step: float, threshold: float) -> float | None:
    if step <= 0:
        return None
    delta = round(value / step) * step - value
    return delta if abs(delta) <= threshold else None


def snap_bounds(
    bounds: Bounds, targets: SnapTargets, threshold: float, grid_step: float | None = None,
) -> tuple[float, float, list[Guide]]:
    """옮기는 중인 경계를 가장 가까운 기준에 붙이는 (dx, dy, 안내선)."""
    left, top, right, bottom = bounds
    x_lines = (left, (left + right) / 2.0, right)
    y_lines = (top, (top + bottom) / 2.0, bottom)
    guides: list[Guide] = []

    dx = _best_shift(x_lines, targets.xs, threshold)
    dy = _best_shift(y_lines, targets.ys, threshold)
    if dx is None and grid_step:
        dx = _grid_shift(left, grid_step, threshold)
    elif dx is not None:
        guides += _guides_for("v", x_lines, dx, targets.xs, top + (dy or 0.0), bottom + (dy or 0.0))
    if dy is None and grid_step:
        dy = _grid_shift(top, grid_step, threshold)
    elif dy is not None:
        guides += _guides_for("h", y_lines, dy, targets.ys, left + (dx or 0.0), right + (dx or 0.0))
    return (dx or 0.0, dy or 0.0, guides)


def snap_value(
    value: float, targets: list[Target], threshold: float, orientation: str,
    cross: tuple[float, float], grid_step: float | None = None,
) -> tuple[float, list[Guide]]:
    """값 하나(끌고 있는 가장자리나 끝점)를 기준에 붙인다."""
    shift = _best_shift((value,), targets, threshold)
    if shift is not None:
        return value + shift, _guides_for(orientation, (value,), shift, targets, cross[0], cross[1])
    if grid_step:
        shift = _grid_shift(value, grid_step, threshold)
        if shift is not None:
            return value + shift, []
    return value, []
