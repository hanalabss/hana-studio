import math

import pytest

from design.model import LineItem, RectItem
from ui.editor import geometry, snapping


def test_resize_east_handle_keeps_left_edge():
    x, y, w, h = geometry.resize_box((10, 10, 20, 10, 0), "e", 40, 12)
    assert (x, y, w, h) == pytest.approx((10, 10, 30, 10))


def test_resize_northwest_handle_keeps_bottom_right():
    x, y, w, h = geometry.resize_box((10, 10, 20, 10, 0), "nw", 5, 4)
    assert (x + w, y + h) == pytest.approx((30, 20))
    assert (x, y) == pytest.approx((5, 4))


def test_resize_never_flips_or_collapses():
    x, y, w, h = geometry.resize_box((10, 10, 20, 10, 0), "e", -50, 0)
    assert w == pytest.approx(geometry.MIN_SIZE_MM)
    assert x == pytest.approx(10)


def test_resize_keep_aspect_on_corner():
    x, y, w, h = geometry.resize_box((0, 0, 20, 10, 0), "se", 40, 12, keep_aspect=True)
    assert w / h == pytest.approx(2.0)
    assert (x, y, w) == pytest.approx((0, 0, 40))


@pytest.mark.parametrize("rotation", [30.0, 90.0, 215.0])
def test_resize_rotated_box_keeps_opposite_corner_fixed(rotation):
    start = (10.0, 20.0, 30.0, 12.0, rotation)
    anchor_before = geometry.handle_point(start, "nw")
    grabbed = geometry.handle_point(start, "se")
    dx, dy = geometry.rotate_vec(6.0, 3.0, rotation)
    x, y, w, h = geometry.resize_box(start, "se", grabbed[0] + dx, grabbed[1] + dy)
    assert (w, h) == pytest.approx((36.0, 15.0))
    anchor_after = geometry.handle_point((x, y, w, h, rotation), "nw")
    assert anchor_after == pytest.approx(anchor_before)


def test_hit_thin_horizontal_line_with_tolerance():
    line = LineItem(x=10, y=20, w=30, h=0, width_mm=0.1)
    assert geometry.hit_item(line, 25, 20.4, tolerance=0.5)
    assert not geometry.hit_item(line, 25, 21.0, tolerance=0.5)
    assert not geometry.hit_item(line, 45, 20, tolerance=0.5)


def test_hit_diagonal_line_only_near_the_stroke():
    line = LineItem(x=0, y=0, w=20, h=20, width_mm=0.3)
    assert geometry.hit_item(line, 10, 10.2, tolerance=0.5)
    assert not geometry.hit_item(line, 18, 2, tolerance=0.5)
    flipped = LineItem(x=0, y=0, w=20, h=20, width_mm=0.3, flip=True)
    assert geometry.hit_item(flipped, 18, 2.2, tolerance=0.5)


def test_hit_rotated_rect_uses_rotated_shape():
    rect = RectItem(x=0, y=0, w=40, h=4, rotation=90)
    assert geometry.hit_item(rect, 20, -15)
    assert not geometry.hit_item(rect, 5, 2)


def test_angle_and_snapping():
    assert geometry.angle_to(0, 0, 10, 0) == pytest.approx(90)
    assert geometry.angle_to(0, 0, 0, -10) == pytest.approx(0)
    assert geometry.snap_angle(37, step=15) == pytest.approx(30)
    assert geometry.snap_angle(88.5, soft=3) == pytest.approx(90)
    assert geometry.snap_angle(80, soft=3) == pytest.approx(80)
    assert geometry.snap_angle(359.2, soft=3) == pytest.approx(0)


def test_line_from_points_orders_and_flips():
    values = geometry.line_from_points((30, 5), (10, 15))
    assert values == {"x": 10, "y": 5, "w": 20, "h": 10, "flip": True, "rotation": 0.0}


def test_snap_bounds_to_edge_and_center():
    targets = snapping.SnapTargets(xs=[(0, 0, 50), (40, 0, 50), (80, 0, 50)], ys=[(25, 0, 80)])
    dx, dy, guides = snapping.snap_bounds((0.4, 20.3, 10.4, 30.3), targets, 0.6)
    assert dx == pytest.approx(-0.4)
    assert dy == pytest.approx(-0.3)
    assert {(g.orientation, g.pos) for g in guides} == {("v", 0), ("h", 25)}


def test_snap_bounds_ignores_far_targets_and_uses_grid():
    targets = snapping.SnapTargets(xs=[(50, 0, 10)], ys=[])
    dx, dy, guides = snapping.snap_bounds((10.2, 10.3, 20.2, 20.3), targets, 0.5)
    assert (dx, dy, guides) == (0.0, 0.0, [])
    dx, dy, guides = snapping.snap_bounds((10.2, 10.3, 20.2, 20.3), targets, 0.5, grid_step=1.0)
    assert (dx, dy) == pytest.approx((-0.2, -0.3))
    assert guides == []


def test_collect_targets_includes_cut_line_and_other_items():
    from design import presets

    document = presets.new_document(presets.CUT_54X86_BLEED)
    side = document.side("front")
    a = side.add_item(RectItem(x=10, y=10, w=20, h=10))
    b = side.add_item(RectItem(x=50, y=30, w=10, h=10))
    targets = snapping.collect_targets(document, "front", [a.id])
    xs = {round(pos, 3) for pos, _, _ in targets.xs}
    assert {0.0, 45.0, 90.0, 2.0, 88.0, 50.0, 55.0, 60.0} <= xs
    assert 10.0 not in xs and 20.0 not in xs
    assert math.isclose(min(pos for pos, _, _ in targets.ys), 0.0)
    assert b.id != a.id
