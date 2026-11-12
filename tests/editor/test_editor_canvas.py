import json

import pytest
from PySide6.QtCore import QMimeData, QPoint, QPointF, Qt, QUrl
from PySide6.QtGui import QDragEnterEvent, QDragMoveEvent, QDropEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from design import presets
from design.binding import ColumnSource
from ui.editor import MIME_FIELD, EditorPalette, field_mime
from ui.editor.view import Ruler

SHIFT = Qt.KeyboardModifier.ShiftModifier
CTRL = Qt.KeyboardModifier.ControlModifier
ZOOM_LEVELS = [3.0, 6.5, 14.0]  # 뷰포트 px / mm


def _rect(controller, x, y, w=10.0, h=10.0, **extra):
    item = controller.add_item("rect", w=w, h=h, **extra)
    controller.move_items_to({item.id: (x, y)})
    controller.clear_selection()
    return item


def _handle(view, item, name) -> QPointF:
    return view._handle_points(item)[name]


def _drop(view, mime, pos: QPointF) -> bool:
    """끌어 들어오기 → 움직이기 → 놓기 순서로 보낸다. 받아들였으면 True."""
    left, none = Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier
    enter = QDragEnterEvent(pos.toPoint(), Qt.DropAction.CopyAction, mime, left, none)
    QApplication.sendEvent(view.viewport(), enter)
    if not enter.isAccepted():
        return False
    QApplication.sendEvent(view.viewport(), QDragMoveEvent(pos.toPoint(), Qt.DropAction.CopyAction, mime, left, none))
    QApplication.sendEvent(view.viewport(), QDropEvent(pos, Qt.DropAction.CopyAction, mime, left, none))
    return True


# ---------------------------------------------------------------- 보기

def test_view_fits_card_on_show_and_tracks_resize(editor):
    view = editor.view
    viewport = view.viewport().rect()
    card_px = (85.6 * view.px_per_mm, 53.98 * view.px_per_mm)
    assert card_px[0] <= viewport.width() and card_px[1] <= viewport.height()
    assert card_px[0] >= viewport.width() * 0.7
    center = view.mm_to_viewport(42.8, 26.99)
    assert center.x() == pytest.approx(viewport.width() / 2, abs=2)
    before = view.zoom
    view.resize(500, 400)
    editor.app.processEvents()
    assert view.zoom < before


def test_zoom_actual_fit_and_selection(editor):
    view, controller = editor.view, editor.controller
    seen = []
    view.zoom_changed.connect(seen.append)
    view.zoom_actual()
    assert view.zoom == pytest.approx(1.0)
    assert view.px_per_mm == pytest.approx(view.logicalDpiX() / 25.4)
    item = _rect(controller, 10.0, 10.0, w=5.0, h=5.0)
    controller.select([item.id])
    view.zoom_to_selection()
    assert view.zoom > 3.0
    assert view.visible_center_mm() == pytest.approx((12.5, 12.5), abs=0.5)
    controller.zoom_fit()
    assert 85.6 * view.px_per_mm <= view.viewport().width()
    assert len(seen) == 3


def test_ctrl_wheel_zooms_around_cursor(editor):
    from PySide6.QtGui import QWheelEvent

    view = editor.view
    anchor = QPointF(200, 150)
    before_mm = view.viewport_to_mm(anchor)
    before_zoom = view.zoom
    event = QWheelEvent(
        anchor, QPointF(view.viewport().mapToGlobal(anchor.toPoint())), QPoint(0, 0), QPoint(0, 240),
        Qt.MouseButton.NoButton, CTRL, Qt.ScrollPhase.NoScrollPhase, False,
    )
    QApplication.sendEvent(view.viewport(), event)
    assert view.zoom == pytest.approx(before_zoom * 1.15 ** 2)
    assert view.viewport_to_mm(anchor) == pytest.approx(before_mm, abs=0.2)


def test_space_drag_pans(editor):
    view = editor.view
    editor.set_px_per_mm(14.0)
    before = view.viewport_to_mm(QPointF(100, 100))
    QTest.keyPress(view, Qt.Key.Key_Space)
    editor.mouse.drag(QPointF(300, 300), QPointF(240, 270))
    QTest.keyRelease(view, Qt.Key.Key_Space)
    after = view.viewport_to_mm(QPointF(100, 100))
    assert after[0] - before[0] == pytest.approx(60 / 14.0, abs=0.1)
    assert after[1] - before[1] == pytest.approx(30 / 14.0, abs=0.1)
    assert editor.controller.undo_stack.count() == 0


def test_rulers_follow_zoom_and_side_switch(editor):
    assert Ruler.steps(3.0) == (20.0, 2.0)
    assert Ruler.steps(10.0) == (5.0, 1.0)
    label, minor = Ruler.steps(40.0)
    assert label == 2.0 and minor == pytest.approx(0.2)
    view = editor.view
    assert view.viewport().x() == view._h_ruler.height() == view._v_ruler.width()
    editor.controller.new_document(presets.CR80, orientation="portrait")
    editor.app.processEvents()
    assert editor.controller.scene.card_rect().height() == pytest.approx(85.6)
    assert 85.6 * view.px_per_mm <= view.viewport().height()
    view.set_rulers_visible(False)
    assert view.viewport().x() == 0


def test_palette_switch_changes_background(editor):
    view = editor.view
    view.set_palette(EditorPalette.dark())
    assert view.backgroundBrush().color().name().upper() == "#15171A"
    corner = view.viewport().grab().toImage().pixelColor(3, 3)
    assert corner.name().upper() == "#15171A"


# ---------------------------------------------------------------- 선택

def test_click_selects_topmost_and_empty_click_clears(editor):
    controller, mouse = editor.controller, editor.mouse
    below = _rect(controller, 10.0, 10.0, w=20.0, h=20.0)
    above = _rect(controller, 20.0, 20.0, w=20.0, h=20.0)
    QTest.mouseClick(editor.view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                     mouse.at(25.0, 25.0).toPoint())
    assert controller.selection == [above.id]
    mouse.click(mouse.at(12.0, 12.0))
    assert controller.selection == [below.id]
    mouse.click(mouse.at(70.0, 45.0))
    assert controller.selection == []
    assert controller.undo_stack.count() == 4


def test_shift_click_toggles(editor):
    controller, mouse = editor.controller, editor.mouse
    a = _rect(controller, 5.0, 5.0)
    b = _rect(controller, 30.0, 5.0)
    mouse.click(mouse.at(10.0, 10.0))
    mouse.click(mouse.at(35.0, 10.0), SHIFT)
    assert controller.selection == [a.id, b.id]
    mouse.click(mouse.at(10.0, 10.0), SHIFT)
    assert controller.selection == [b.id]


def test_rubber_band_selects_touched_items_only(editor):
    controller, mouse = editor.controller, editor.mouse
    a = _rect(controller, 5.0, 5.0)
    b = _rect(controller, 30.0, 5.0)
    c = _rect(controller, 60.0, 35.0)
    locked = _rect(controller, 18.0, 20.0)
    controller.set_locked([locked.id], True)
    mouse.drag(mouse.at(2.0, 2.0), mouse.at(45.0, 32.0))
    assert controller.selection == [a.id, b.id]
    mouse.drag(mouse.at(55.0, 30.0), mouse.at(75.0, 50.0), SHIFT)
    assert controller.selection == [a.id, b.id, c.id]


def test_thin_line_and_rotated_item_are_clickable(editor):
    controller, mouse = editor.controller, editor.mouse
    line = controller.add_item("line", width_mm=0.1, w=30.0, pos=(30.0, 10.0))
    bar = _rect(controller, 40.0, 28.0, w=30.0, h=3.0)
    controller.set_rotation([bar.id], 90.0)
    controller.clear_selection()
    mouse.click(mouse.at(30.0, 10.0) + QPointF(0, 3))
    assert controller.selection == [line.id]
    mouse.click(mouse.at(55.0, 18.0))
    assert controller.selection == [bar.id]
    mouse.click(mouse.at(42.0, 29.5))
    assert controller.selection == []


# ---------------------------------------------------------------- 이동

@pytest.mark.parametrize("px_per_mm", ZOOM_LEVELS)
def test_drag_moves_item_in_mm_at_any_zoom(editor, px_per_mm):
    controller, mouse = editor.controller, editor.mouse
    item = _rect(controller, 20.0, 15.0)
    editor.set_px_per_mm(px_per_mm)
    count = controller.undo_stack.count()
    start = mouse.at(25.0, 20.0)
    mouse.drag(start, start + QPointF(42, -21), steps=7)
    assert item.x == pytest.approx(20.0 + 42 / px_per_mm, abs=0.011)
    assert item.y == pytest.approx(15.0 - 21 / px_per_mm, abs=0.011)
    assert controller.selection == [item.id]
    assert controller.undo_stack.count() == count + 1
    canvas_item = controller.scene.canvas_item(item.id)
    assert (canvas_item.pos().x(), canvas_item.pos().y()) == (item.x, item.y)
    moved = (item.x, item.y)
    controller.undo_stack.undo()
    assert (item.x, item.y) == pytest.approx((20.0, 15.0))
    assert (canvas_item.pos().x(), canvas_item.pos().y()) == pytest.approx((20.0, 15.0))
    controller.undo_stack.redo()
    assert (item.x, item.y) == pytest.approx(moved)


def test_click_without_drag_adds_no_undo_step(editor):
    controller, mouse = editor.controller, editor.mouse
    item = _rect(controller, 20.0, 15.0)
    count = controller.undo_stack.count()
    mouse.click(mouse.at(25.0, 20.0))
    mouse.drag(mouse.at(25.0, 20.0), mouse.at(25.0, 20.0) + QPointF(1, 1))
    assert controller.undo_stack.count() == count
    assert (item.x, item.y) == (20.0, 15.0)


def test_drag_moves_whole_selection_and_shift_constrains_axis(editor):
    controller, mouse = editor.controller, editor.mouse
    a = _rect(controller, 5.0, 5.0)
    b = _rect(controller, 30.0, 5.0)
    controller.select([a.id, b.id])
    editor.set_px_per_mm(8.0)
    start = mouse.at(10.0, 10.0)
    mouse.drag(start, start + QPointF(80, 16))
    assert (a.x, b.x) == pytest.approx((15.0, 40.0), abs=0.011)
    assert (a.y, b.y) == pytest.approx((7.0, 7.0), abs=0.011)
    controller.select([a.id])
    start = mouse.at(a.x + 5, a.y + 5)
    mouse.drag(start, start + QPointF(40, 12), SHIFT)
    assert a.x == pytest.approx(20.0, abs=0.011)
    assert a.y == pytest.approx(7.0, abs=0.011)


def test_drag_snaps_to_other_item_and_shows_guides(editor):
    controller, mouse = editor.controller, editor.mouse
    anchor = _rect(controller, 30.0, 5.0, w=20.0)
    item = _rect(controller, 10.0, 30.0)
    controller.set_snap_enabled(True)
    editor.set_px_per_mm(8.0)
    start = mouse.at(15.0, 35.0)
    end = start + QPointF((30.3 - 10.0) * 8.0, 0)
    mouse.drag(start, end, release=False)
    assert item.x == pytest.approx(anchor.x)
    guides = editor.view.active_guides
    assert any(g.orientation == "v" and g.pos == pytest.approx(30.0) for g in guides)
    vertical = next(g for g in guides if g.orientation == "v")
    assert vertical.start <= 5.0 and vertical.end >= 40.0
    mouse.release(end)
    assert editor.view.active_guides == []
    assert item.x == pytest.approx(30.0)


def test_drag_snaps_to_card_center_and_alt_disables(editor):
    controller, mouse = editor.controller, editor.mouse
    item = _rect(controller, 10.0, 12.0)
    controller.set_snap_enabled(True)
    editor.set_px_per_mm(8.0)
    start = mouse.at(15.0, 17.0)
    mouse.drag(start, start + QPointF((37.5 - 10.0) * 8.0, (22.3 - 12.0) * 8.0))
    assert item.center() == pytest.approx((42.8, 26.99))
    start = mouse.at(*item.center())
    mouse.drag(start, start + QPointF(3, 3), Qt.KeyboardModifier.AltModifier)
    assert item.center()[0] == pytest.approx(42.8 + 3 / 8.0, abs=0.011)


def test_drag_snaps_to_cut_line_and_grid(editor):
    controller, mouse = editor.controller, editor.mouse
    controller.new_document(presets.CUT_54X86_BLEED)
    item = _rect(controller, 20.0, 20.0)
    controller.set_snap_enabled(True)
    editor.set_px_per_mm(8.0)
    start = mouse.at(25.0, 25.0)
    mouse.drag(start, start + QPointF((2.4 - 20.0) * 8.0, 0))
    assert item.x == pytest.approx(2.0)

    controller.set_grid_visible(True)
    controller.set_grid_step(5.0)
    start = mouse.at(item.x + 5, item.y + 5)
    mouse.drag(start, start + QPointF((14.6 - 2.0) * 8.0, (34.6 - 20.0) * 8.0))
    assert (item.x, item.y) == pytest.approx((15.0, 35.0))


def test_locked_item_cannot_be_clicked_or_dragged(editor):
    controller, mouse = editor.controller, editor.mouse
    below = _rect(controller, 10.0, 10.0, w=30.0, h=30.0)
    locked = _rect(controller, 15.0, 15.0, w=10.0, h=10.0)
    controller.set_locked([locked.id], True)
    controller.clear_selection()
    mouse.click(mouse.at(20.0, 20.0))
    assert controller.selection == [below.id]

    controller.select([locked.id])
    assert controller.selection == [locked.id]
    assert editor.view._handle_at(_handle_point_guess(editor.view, locked)) is None
    count = controller.undo_stack.count()
    QTest.keyClick(editor.view, Qt.Key.Key_Right)
    QTest.keyClick(editor.view, Qt.Key.Key_Delete)
    assert (locked.x, locked.y) == (15.0, 15.0)
    assert controller.side().get(locked.id) is locked
    assert controller.undo_stack.count() == count

    controller.set_locked([below.id], True)
    controller.select([locked.id])
    start = mouse.at(20.0, 20.0)
    mouse.drag(start, start + QPointF(50, 30))
    assert (locked.x, locked.y) == (15.0, 15.0)
    assert (below.x, below.y) == (10.0, 10.0)


def _handle_point_guess(view, item) -> QPointF:
    return view.mm_to_viewport(item.x + item.w, item.y + item.h)


# ---------------------------------------------------------------- 크기 / 회전

@pytest.mark.parametrize("px_per_mm", ZOOM_LEVELS)
def test_resize_handle_updates_model_at_any_zoom(editor, px_per_mm):
    controller, mouse = editor.controller, editor.mouse
    item = _rect(controller, 20.0, 15.0, w=20.0, h=12.0)
    controller.select([item.id])
    editor.set_px_per_mm(px_per_mm)
    count = controller.undo_stack.count()
    start = _handle(editor.view, item, "se")
    mouse.drag(start, start + QPointF(28, 14), steps=6)
    assert (item.x, item.y) == pytest.approx((20.0, 15.0))
    assert item.w == pytest.approx(20.0 + 28 / px_per_mm, abs=1e-6)
    assert item.h == pytest.approx(12.0 + 14 / px_per_mm, abs=1e-6)
    assert controller.undo_stack.count() == count + 1

    start = _handle(editor.view, item, "nw")
    right, bottom = item.x + item.w, item.y + item.h
    mouse.drag(start, start + QPointF(-14, 7))
    assert (item.x + item.w, item.y + item.h) == pytest.approx((right, bottom))
    assert item.x == pytest.approx(20.0 - 14 / px_per_mm, abs=1e-6)
    assert item.y == pytest.approx(15.0 + 7 / px_per_mm, abs=1e-6)

    controller.undo_stack.undo()
    controller.undo_stack.undo()
    assert (item.x, item.y, item.w, item.h) == pytest.approx((20.0, 15.0, 20.0, 12.0))
    bounding = controller.scene.canvas_item(item.id).boundingRect()
    assert (bounding.width(), bounding.height()) == pytest.approx((20.0, 12.0))


def test_edge_handle_changes_one_dimension_and_shift_keeps_aspect(editor):
    controller, mouse = editor.controller, editor.mouse
    item = _rect(controller, 20.0, 15.0, w=20.0, h=10.0)
    controller.select([item.id])
    editor.set_px_per_mm(8.0)
    start = _handle(editor.view, item, "e")
    mouse.drag(start, start + QPointF(40, 25))
    assert (item.w, item.h) == pytest.approx((25.0, 10.0))
    start = _handle(editor.view, item, "se")
    mouse.drag(start, start + QPointF(40, 0), SHIFT)
    assert item.w == pytest.approx(30.0)
    assert item.w / item.h == pytest.approx(2.5)


def test_image_and_qr_keep_aspect_by_default(editor, make_png):
    controller, mouse = editor.controller, editor.mouse
    image = controller.add_image_from_file(make_png(size=(300, 100)))
    ratio = image.w / image.h
    editor.set_px_per_mm(8.0)
    start = _handle(editor.view, image, "se")
    mouse.drag(start, start + QPointF(40, 0))
    assert image.w / image.h == pytest.approx(ratio)
    start = _handle(editor.view, image, "se")
    mouse.drag(start, start + QPointF(0, 40), SHIFT)
    assert image.w / image.h != pytest.approx(ratio)

    qr = controller.add_item("qr", pos=(20.0, 20.0))
    start = _handle(editor.view, qr, "e")
    mouse.drag(start, start + QPointF(32, 0))
    assert (qr.w, qr.h) == pytest.approx((22.0, 22.0))


def test_resize_rotated_item_keeps_opposite_corner(editor):
    from ui.editor import geometry

    controller, mouse = editor.controller, editor.mouse
    item = _rect(controller, 30.0, 20.0, w=20.0, h=10.0)
    controller.set_rotation([item.id], 90.0)
    editor.set_px_per_mm(8.0)
    anchor = geometry.handle_point(geometry.box_of(item), "nw")
    start = _handle(editor.view, item, "se")
    mouse.drag(start, start + QPointF(0, 40))
    assert item.rotation == 90.0
    assert (item.w, item.h) == pytest.approx((25.0, 10.0))
    assert geometry.handle_point(geometry.box_of(item), "nw") == pytest.approx(anchor)


def test_resize_edge_snaps_to_other_item(editor):
    controller, mouse = editor.controller, editor.mouse
    _rect(controller, 50.0, 5.0, w=10.0)
    item = _rect(controller, 10.0, 30.0, w=20.0)
    controller.select([item.id])
    controller.set_snap_enabled(True)
    editor.set_px_per_mm(8.0)
    start = _handle(editor.view, item, "e")
    mouse.drag(start, start + QPointF((49.6 - 30.0) * 8.0, 0))
    assert item.x + item.w == pytest.approx(50.0)


def test_rotate_handle_and_shift_steps(editor):
    controller, mouse = editor.controller, editor.mouse
    item = _rect(controller, 30.0, 20.0, w=20.0, h=10.0)
    controller.select([item.id])
    editor.set_px_per_mm(8.0)
    count = controller.undo_stack.count()
    center = mouse.at(40.0, 25.0)
    mouse.drag(_handle(editor.view, item, "rotate"), center + QPointF(100, 0), steps=8)
    assert item.rotation == pytest.approx(90.0)
    assert (item.x, item.y, item.w, item.h) == (30.0, 20.0, 20.0, 10.0)
    assert controller.scene.canvas_item(item.id).rotation() == pytest.approx(90.0)
    assert controller.undo_stack.count() == count + 1

    mouse.drag(_handle(editor.view, item, "rotate"), center + QPointF(100, 62), SHIFT)
    assert item.rotation == pytest.approx(120.0)
    mouse.drag(_handle(editor.view, item, "rotate"), center + QPointF(-60, 80))
    assert item.rotation == pytest.approx(216.9, abs=0.11)
    controller.undo_stack.undo()
    controller.undo_stack.undo()
    controller.undo_stack.undo()
    assert item.rotation == 0.0
    assert controller.selection == [item.id]


def test_line_endpoint_drag(editor):
    controller, mouse = editor.controller, editor.mouse
    line = controller.add_item("line", w=20.0, pos=(30.0, 20.0))
    editor.set_px_per_mm(8.0)
    assert set(editor.view._handle_points(line)) == {"p1", "p2"}
    start = _handle(editor.view, line, "p2")
    mouse.drag(start, start + QPointF(40, -80))
    assert (line.x, line.y, line.w, line.h, line.flip) == pytest.approx((20.0, 10.0, 25.0, 10.0, True))
    start = _handle(editor.view, line, "p2")
    mouse.drag(start, mouse.at(50.0, 21.0), SHIFT)
    assert line.h == pytest.approx(0.0, abs=1e-6)
    assert line.w == pytest.approx((30.0 ** 2 + 1.0) ** 0.5)


# ---------------------------------------------------------------- 키보드

def test_arrow_keys_nudge(editor):
    controller = editor.controller
    item = _rect(controller, 20.0, 15.0)
    controller.select([item.id])
    QTest.keyClick(editor.view, Qt.Key.Key_Right)
    QTest.keyClick(editor.view, Qt.Key.Key_Down, SHIFT)
    assert (item.x, item.y) == pytest.approx((20.1, 16.0))
    QTest.keyClick(editor.view, Qt.Key.Key_Left, SHIFT)
    QTest.keyClick(editor.view, Qt.Key.Key_Up)
    assert (item.x, item.y) == pytest.approx((19.1, 15.9))
    controller.undo_stack.undo()
    assert (item.x, item.y) == pytest.approx((19.1, 16.0))


def test_delete_duplicate_and_select_all_keys(editor):
    controller = editor.controller
    a = _rect(controller, 5.0, 5.0)
    b = _rect(controller, 30.0, 5.0)
    controller.select([a.id])
    QTest.keyClick(editor.view, Qt.Key.Key_D, CTRL)
    assert len(controller.side().items) == 3
    clone_id = controller.selection[0]
    assert clone_id not in (a.id, b.id)
    QTest.keyClick(editor.view, Qt.Key.Key_Delete)
    assert controller.side().get(clone_id) is None
    QTest.keyClick(editor.view, Qt.Key.Key_A, CTRL)
    assert controller.selection == [a.id, b.id]
    QTest.keyClick(editor.view, Qt.Key.Key_Escape)
    assert controller.selection == []
    QTest.keyClick(editor.view, Qt.Key.Key_Z, CTRL)
    assert controller.side().get(clone_id) is not None
    QTest.keyClick(editor.view, Qt.Key.Key_Z, CTRL)
    assert controller.side().get(clone_id) is None
    QTest.keyClick(editor.view, Qt.Key.Key_Y, CTRL)
    assert controller.side().get(clone_id) is not None


def test_copy_cut_paste_keys_across_sides(editor):
    controller = editor.controller
    item = _rect(controller, 5.0, 5.0)
    controller.select([item.id])
    QTest.keyClick(editor.view, Qt.Key.Key_C, CTRL)
    QTest.keyClick(editor.view, Qt.Key.Key_V, CTRL)
    assert len(controller.side().items) == 2
    QTest.keyClick(editor.view, Qt.Key.Key_X, CTRL)
    assert controller.side().items == [item]
    controller.set_side("back")
    QTest.keyClick(editor.view, Qt.Key.Key_V, CTRL)
    back = controller.document.side("back").items
    assert len(back) == 1 and (back[0].x, back[0].y) == pytest.approx((7.0, 7.0))
    assert controller.scene.canvas_item(back[0].id) is not None
    assert controller.scene.canvas_item(item.id) is None


# ---------------------------------------------------------------- 글자 바로 고치기

def test_double_click_edits_text_inline(editor):
    controller, mouse = editor.controller, editor.mouse
    item = controller.add_item("text", text="처음", pos=(40.0, 20.0))
    count = controller.undo_stack.count()
    QTest.mouseDClick(editor.view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                      mouse.at(40.0, 20.0).toPoint())
    field = editor.view.text_editor()
    assert field is not None and field.toPlainText() == "처음"
    assert controller.scene.editing_id == item.id
    field.setPlainText("홍길동")
    QTest.keyClick(field, Qt.Key.Key_Return)
    assert item.text == "홍길동"
    assert not editor.view.is_editing_text() and controller.scene.editing_id is None
    assert controller.undo_stack.count() == count + 1
    controller.undo_stack.undo()
    assert item.text == "처음"


def test_inline_edit_escape_cancels_and_click_elsewhere_commits(editor):
    controller, mouse = editor.controller, editor.mouse
    item = controller.add_item("text", text="처음", pos=(40.0, 20.0))
    assert editor.view.begin_text_edit(item.id)
    editor.view.text_editor().setPlainText("버릴 글자")
    QTest.keyClick(editor.view.text_editor(), Qt.Key.Key_Escape)
    assert item.text == "처음" and not editor.view.is_editing_text()

    assert editor.view.begin_text_edit(item.id)
    editor.view.text_editor().setPlainText("남길 글자")
    mouse.click(mouse.at(75.0, 48.0))
    assert item.text == "남길 글자"


def test_double_click_on_bound_text_asks_for_binding_instead(editor):
    controller, mouse = editor.controller, editor.mouse
    item = controller.add_bound_field("이름", "text", pos=(40.0, 20.0))
    asked = []
    controller.bound_edit_requested.connect(asked.append)
    QTest.mouseDClick(editor.view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                      mouse.at(40.0, 20.0).toPoint())
    assert asked == [item.id]
    assert not editor.view.is_editing_text()


# ---------------------------------------------------------------- 끌어다 놓기

def test_field_drop_creates_bound_item_at_drop_point(editor):
    controller, mouse = editor.controller, editor.mouse
    mime = QMimeData()
    mime.setData(MIME_FIELD, json.dumps({"column": "이름", "kind": "text"}).encode("utf-8"))
    assert _drop(editor.view, mime, mouse.at(30.0, 20.0))
    item = controller.selected_items()[0]
    assert item.kind == "text"
    assert isinstance(item.source, ColumnSource) and item.source.column == "이름"
    assert item.center() == pytest.approx((30.0, 20.0), abs=0.2)
    assert controller.scene.value_of(item)[0] == "«이름»"

    _drop(editor.view, field_mime("사진", "image"), mouse.at(60.0, 30.0))
    photo = controller.selected_items()[0]
    assert photo.kind == "image" and photo.source.column == "사진"
    controller.undo_stack.undo()
    assert controller.side().get(photo.id) is None


def test_bad_field_payload_is_ignored(editor):
    mime = QMimeData()
    mime.setData(MIME_FIELD, b"not json")
    assert not _drop(editor.view, mime, editor.mouse.at(30.0, 20.0))
    plain = QMimeData()
    plain.setText("이름")
    assert not _drop(editor.view, plain, editor.mouse.at(30.0, 20.0))
    assert editor.controller.side().items == []


def test_image_file_drop_imports_asset(editor, make_png, tmp_path):
    controller, mouse = editor.controller, editor.mouse
    text_file = tmp_path / "notes.txt"
    text_file.write_text("x")
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(make_png()), QUrl.fromLocalFile(str(text_file))])
    _drop(editor.view, mime, mouse.at(40.0, 25.0))
    items = controller.side().items
    assert len(items) == 1 and items[0].kind == "image" and items[0].asset
    assert items[0].center() == pytest.approx((40.0, 25.0), abs=0.2)


# ---------------------------------------------------------------- 모델 → 화면

def test_programmatic_changes_update_scene(editor):
    controller = editor.controller
    item = controller.add_item("rect")
    canvas_item = controller.scene.canvas_item(item.id)
    assert canvas_item is not None and canvas_item.scene() is controller.scene
    controller.set_properties([item.id], {"x": 11.0, "y": 7.0, "w": 33.0, "rotation": 45.0})
    assert (canvas_item.pos().x(), canvas_item.pos().y()) == (11.0, 7.0)
    assert canvas_item.boundingRect().width() == 33.0
    assert canvas_item.rotation() == 45.0
    assert canvas_item.transformOriginPoint().x() == pytest.approx(16.5)
    controller.set_visible([item.id], False)
    assert not canvas_item.isVisible()
    controller.undo_stack.undo()
    assert canvas_item.isVisible()
    controller.set_layer([item.id], "uv")
    assert canvas_item.opacity() < 1.0
    controller.delete([item.id])
    assert controller.scene.canvas_item(item.id) is None
    controller.undo_stack.undo()
    assert controller.scene.canvas_item(item.id).model is item


def test_selection_overlay_is_painted_in_palette_color(editor):
    controller, view = editor.controller, editor.view
    item = _rect(controller, 20.0, 15.0, w=20.0, h=12.0, fill="#ffffff")
    editor.set_px_per_mm(8.0)
    corner = view.mm_to_viewport(20.0, 15.0).toPoint()
    plain = view.viewport().grab().toImage().pixelColor(corner + QPoint(40, 0))
    controller.select([item.id])
    editor.app.processEvents()
    image = view.viewport().grab().toImage()
    edge = image.pixelColor(corner + QPoint(40, 0))
    selection = EditorPalette.light().selection
    assert plain.name() == "#ffffff"
    assert abs(edge.red() - int(selection[1:3], 16)) < 60 and edge.blue() > 200
    handle_center = image.pixelColor(corner)
    assert handle_center.name() == "#ffffff"


def test_warning_state_shows_for_item_crossing_card_edge(editor):
    controller, mouse = editor.controller, editor.mouse
    item = _rect(controller, 60.0, 20.0, w=20.0)
    editor.set_px_per_mm(6.0)
    assert controller.warnings == {}
    start = mouse.at(70.0, 25.0)
    mouse.drag(start, start + QPointF(72, 0))
    assert item.x == pytest.approx(72.0, abs=0.011)
    assert list(controller.warnings) == [item.id]
    assert controller.layers()[0].warning


def test_escape_during_drag_puts_item_back(editor):
    controller, mouse = editor.controller, editor.mouse
    item = _rect(controller, 20.0, 15.0)
    editor.set_px_per_mm(8.0)
    start = mouse.at(25.0, 20.0)
    mouse.drag(start, start + QPointF(80, 40), release=False)
    assert item.x == pytest.approx(30.0, abs=0.011)
    QTest.keyClick(editor.view, Qt.Key.Key_Escape)
    assert (item.x, item.y) == pytest.approx((20.0, 15.0))
    mouse.move(start + QPointF(120, 40))
    mouse.release(start + QPointF(120, 40))
    assert (item.x, item.y) == pytest.approx((20.0, 15.0))
    assert controller.selection == [item.id]


def test_tab_cycles_items_then_leaves_canvas(editor):
    controller, view = editor.controller, editor.view
    a = _rect(controller, 5.0, 5.0)
    b = _rect(controller, 30.0, 5.0)
    locked = _rect(controller, 50.0, 5.0)
    controller.set_locked([locked.id], True)
    controller.clear_selection()
    view.activateWindow()
    view.setFocus()
    editor.app.processEvents()
    assert view.hasFocus()
    assert view.focusNextPrevChild(True) and controller.selection == [b.id]
    assert view.focusNextPrevChild(True) and controller.selection == [a.id]
    view.focusNextPrevChild(True)
    assert controller.selection == []
    assert view.focusNextPrevChild(False) and controller.selection == [a.id]
