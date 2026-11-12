import json

import pytest
from pages_design_helpers import *  # noqa: F401,F403

from design.binding import ColumnSource
from ui.editor import MIME_FIELD
from ui.widgets import FIELD_MIME


@pytest.fixture
def editor(page):
    page.start_panel.choose("blank")
    return page


def _drop(view, mime, x_mm: float, y_mm: float) -> None:
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtGui import QDropEvent

    pos = view.mm_to_viewport(x_mm, y_mm)
    event = QDropEvent(QPoint(round(pos.x()), round(pos.y())), Qt.CopyAction, mime,
                       Qt.LeftButton, Qt.NoModifier)
    view.dropEvent(event)


@pytest.mark.parametrize("tile,kind", [("text", "text"), ("line", "line"), ("barcode", "barcode"), ("qr", "qr")])
def test_insert_tiles_add_items(editor, tile, kind):
    editor.insert_panel.tiles[tile].click()
    items = editor.controller.side().items
    assert [item.kind for item in items] == [kind]
    assert editor.controller.selection == [items[0].id]
    editor.undo()
    assert editor.controller.side().items == []


def test_shape_and_image_menus_add_items(editor, tmp_path):
    panel = editor.insert_panel
    panel.rect_action.trigger()
    panel.ellipse_action.trigger()
    panel.image_frame_action.trigger()
    assert [item.kind for item in editor.controller.side().sorted_items()] == ["rect", "ellipse", "image"]

    editor.dialogs.image_path = make_image(tmp_path / "logo.png")
    panel.image_file_action.trigger()
    added = editor.controller.side().sorted_items()[-1]
    assert added.kind == "image" and added.asset
    assert editor.inspector.current_section() == "image"


def test_without_list_shows_disabled_examples_and_link(editor):
    panel = editor.insert_panel
    assert not panel.has_list()
    assert [chip.name() for chip in panel.chips] == ["이름", "부서", "사진"]
    assert all(not chip.isEnabled() for chip in panel.chips)
    assert panel.list_link.isVisible()

    visited = []
    editor.navigateRequested.connect(visited.append)
    panel.list_link.click()
    assert visited == ["list"]


def test_list_columns_become_chips(editor, workspace, people_store):
    workspace.set_store(people_store)
    panel = editor.insert_panel
    assert [chip.name() for chip in panel.chips] == ["이름", "부서", "직급", "사번", "사진"]
    assert all(chip.isEnabled() for chip in panel.chips)
    assert {chip.name(): chip.kind for chip in panel.chips}["사진"] == "image"
    assert not panel.list_link.isVisible()

    people_store.add_column("입사일", "date")
    workspace.notify_columns_changed()
    assert editor.insert_panel.chips[-1].name() == "입사일"

    workspace.set_store(None)
    assert not editor.insert_panel.has_list()


def test_chip_click_adds_bound_text_at_centre(editor, workspace, people_store):
    workspace.set_store(people_store)
    chip = next(chip for chip in editor.insert_panel.chips if chip.name() == "이름")
    chip.clicked.emit(chip.name())

    item = editor.controller.side().items[0]
    assert item.kind == "text"
    assert item.source == ColumnSource(column="이름")
    width, height = editor.controller.side_size_mm()
    assert abs(item.center()[0] - width / 2) < 1 and abs(item.center()[1] - height / 2) < 1
    assert editor.inspector.current_section() == "text"
    editor.undo()
    assert editor.controller.side().items == []


def test_photo_chip_click_adds_bound_image(editor, workspace, people_store):
    workspace.set_store(people_store)
    chip = next(chip for chip in editor.insert_panel.chips if chip.name() == "사진")
    chip.clicked.emit(chip.name())
    item = editor.controller.side().items[0]
    assert item.kind == "image" and item.source == ColumnSource(column="사진")


def test_chip_mime_carries_both_formats(editor, workspace, people_store):
    workspace.set_store(people_store)
    chip = next(chip for chip in editor.insert_panel.chips if chip.name() == "사진")
    mime = chip.mime_data()
    assert bytes(mime.data(FIELD_MIME)).decode("utf-8") == "사진"
    assert json.loads(bytes(mime.data(MIME_FIELD)).decode("utf-8")) == {"column": "사진", "kind": "image"}


def test_chip_drop_on_canvas_creates_bound_item_where_dropped(editor, workspace, people_store, qapp):
    workspace.set_store(people_store)
    pump(qapp)
    chips = {chip.name(): chip for chip in editor.insert_panel.chips}

    _drop(editor.view, chips["부서"].mime_data(), 30.0, 20.0)
    _drop(editor.view, chips["사진"].mime_data(), 60.0, 30.0)

    text, photo = editor.controller.side().sorted_items()
    assert (text.kind, text.source) == ("text", ColumnSource(column="부서"))
    assert abs(text.center()[0] - 30.0) < 0.6 and abs(text.center()[1] - 20.0) < 0.6
    assert (photo.kind, photo.source) == ("image", ColumnSource(column="사진"))
    assert abs(photo.center()[0] - 60.0) < 0.6


def test_add_back_side_button_makes_card_dual_sided(editor):
    bar = editor.canvas_bar
    bar.add_back_button.click()
    assert editor.controller.document.dual_side is True
    assert editor.controller.current_side == "back"
    assert bar.side_control.isVisible() and not bar.add_back_button.isVisible()
    assert bar.side_control.current() == "back"

    bar.side_control.set_current("front")
    assert editor.controller.current_side == "front"


def test_view_menu_toggles_follow_and_drive_controller(editor):
    bar, controller = editor.canvas_bar, editor.controller
    assert bar.snap_toggle.isChecked() and not bar.grid_toggle.isChecked()
    bar.grid_toggle.trigger()
    assert controller.grid_visible is True
    bar.snap_toggle.trigger()
    assert controller.snap_enabled is False
    controller.set_grid_visible(False)
    assert not bar.grid_toggle.isChecked()
    assert not bar.cut_toggle.isEnabled()          # 재단 여백이 없는 카드
    controller.set_card_setup(preset="cut_54x86_bleed")
    assert bar.cut_toggle.isEnabled()
    bar.cut_toggle.trigger()
    assert controller.cut_line_visible is False


def test_zoom_controls_show_percent(editor, qapp):
    bar, view = editor.canvas_bar, editor.view
    bar.actual_action.trigger()
    assert bar.zoom_button.text() == "100%"
    bar.zoom_in_button.click()
    assert bar.zoom_button.text() == "125%"
    bar.zoom_out_button.click()
    assert bar.zoom_button.text() == "100%"
    bar.fit_action.trigger()
    assert bar.zoom_button.text() == f"{round(view.zoom * 100)}%"


def test_align_menu_aligns_single_item_to_card(editor):
    item = editor.add("rect")
    editor.controller.set_properties(item.id, {"x": 3.0, "y": 4.0})
    index = editor.controller.undo_stack.index()

    editor.canvas_bar._sync_align_menu()
    assert editor.canvas_bar.align_note.text() == "카드에 맞춥니다"
    assert not editor.canvas_bar.distribute_actions["horizontal"].isEnabled()
    editor.canvas_bar.align_actions["center"].trigger()

    width, _height = editor.controller.side_size_mm()
    assert abs(item.center()[0] - width / 2) < 0.01
    assert editor.controller.undo_stack.index() == index + 1
    editor.undo()
    assert item.x == 3.0
