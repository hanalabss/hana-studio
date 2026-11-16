"""다눌에 있던 기능을 화면에서 쓰는 흐름: 카메라 사진, 글자 꾸미기, 사진 모서리, 바코드 종류."""

import pytest
from pages_design_helpers import *  # noqa: F401,F403
from PySide6.QtGui import QColor, QImage

from ui import camera


@pytest.fixture
def editor(page, tmp_path):
    page.start_panel.choose("blank")
    page.photo = page.add_image_file(make_image(tmp_path / "portrait.png", size=(80, 100)))
    return page


def _shot(w=640, h=480, color="#c04040"):
    image = QImage(w, h, QImage.Format.Format_RGB32)
    image.fill(QColor(color))
    return image


def test_camera_photo_replaces_image_cropped_to_box(editor, monkeypatch):
    seen = {}

    def fake_take(parent, aspect=None):
        seen["aspect"] = aspect
        return camera.crop_to_aspect(_shot(), aspect)

    monkeypatch.setattr(camera, "take_photo", fake_take)
    section = editor.inspector.image
    old_asset = editor.photo.asset
    assert section.take_photo()
    assert seen["aspect"] == pytest.approx(editor.photo.w / editor.photo.h)
    assert editor.photo.asset != old_asset and editor.photo.asset.startswith("assets/")


def test_camera_cancel_changes_nothing(editor, monkeypatch):
    monkeypatch.setattr(camera, "take_photo", lambda parent, aspect=None: None)
    old_asset = editor.photo.asset
    assert not editor.inspector.image.take_photo()
    assert editor.photo.asset == old_asset


def test_image_corner_radius_field_edits_item(editor):
    section = editor.inspector.image
    section.radius_field.set_value(3.0)
    assert editor.photo.corner_radius_mm == pytest.approx(3.0)
    editor.undo()
    assert editor.photo.corner_radius_mm == 0.0


def test_text_strikeout_and_background_controls(editor):
    item = editor.add("text")
    section = editor.inspector.text
    assert section.background_picker.isHidden()
    section.strikeout_toggle.click()
    assert item.strikeout is True
    section.background_check.setChecked(True)
    assert item.background and not section.background_picker.isHidden()
    section.background_picker.set_current(section.background_picker.colors()[3])
    assert item.background == section.background_picker.colors()[3]
    section.background_check.setChecked(False)
    assert item.background is None
    editor.undo()
    assert item.background == section.background_picker.colors()[3]


@pytest.mark.parametrize("key, data, ok", [
    ("code39ext", "emp-42a", True), ("code93", "EMP-0042", True),
    ("code128a", "AB12", True), ("code128b", "Ab12", True), ("code128c", "1234", True), ("code128c", "123", False),
])
def test_new_barcode_kinds_are_selectable(editor, key, data, ok):
    item = editor.add("barcode")
    section = editor.inspector.code
    pick(section.symbology_combo, key)
    section.data_edit.setText(data)
    section.data_edit.textEdited.emit(data)
    assert item.symbology == key
    assert section.validation.isVisible() is (not ok)


# ---------------------------------------------------------------- IC 카드(Mifare) 인쇄 설정

def test_mifare_section_only_for_capable_printer_and_saves_choices(qapp):
    from design.model import Document
    from printer.base import CAP_MIFARE
    from ui.pages.printing.options_sheet import OptionsSheet, PrintChoices

    columns = ["이름", "사번"]
    assert OptionsSheet(PrintChoices(), Document(), set(), columns, people=True).mifare_column is None
    sheet = OptionsSheet(PrintChoices(), Document(), {CAP_MIFARE}, columns, people=True)
    assert sheet.mifare_column is not None
    sheet.mifare_column.setCurrentText("사번")
    sheet.mifare_block.set_value(8)
    sheet.mifare_key_type.set_current("B")
    sheet.mifare_key.setText("a0a1a2a3a4a5")
    options = sheet.choices().options
    assert options.mifare == {"column": "사번", "block": 8, "key": "A0A1A2A3A4A5", "key_type": "B",
                              "format": "text"}


def test_mifare_bad_key_blocks_accept(qapp):
    from design.model import Document
    from printer.base import CAP_MIFARE
    from ui.pages.printing.options_sheet import OptionsSheet, PrintChoices

    sheet = OptionsSheet(PrintChoices(), Document(), {CAP_MIFARE}, ["사번"], people=True)
    sheet.mifare_column.setCurrentText("사번")
    sheet.mifare_key.setText("123")
    sheet.mifare_block.set_value(7)
    assert "키가 들어 있는" in sheet.mifare_problem()
    sheet.accept()
    assert not sheet.mifare_error.isHidden() and sheet.result() == 0


# ---------------------------------------------------------------- 캔버스 오른쪽 클릭 메뉴

def _texts(menu):
    return [a.text().split("\t")[0] for a in menu.actions() if not a.isSeparator()]


def test_canvas_menu_for_image_offers_camera_and_rotation(editor, monkeypatch):
    editor.controller.select([editor.photo.id])
    menu = editor.build_canvas_menu()
    texts = _texts(menu)
    assert "카메라로 사진 찍기…" in texts and "오른쪽으로 90° 돌리기" in texts and "복사" in texts
    action = next(a for a in menu.actions() if a.text().startswith("오른쪽으로 90°"))
    action.trigger()
    assert editor.photo.rotation == 90.0
    editor.undo()
    assert editor.photo.rotation == 0.0


def test_canvas_menu_on_empty_space_offers_paste_and_select_all(editor):
    editor.controller.clear_selection()
    texts = _texts(editor.build_canvas_menu())
    assert texts == ["붙여넣기", "전체 선택"]


def test_right_click_selects_item_under_cursor(editor, qapp, monkeypatch):
    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QContextMenuEvent

    shown = []
    monkeypatch.setattr(editor, "show_canvas_menu", lambda pos: shown.append(pos))
    editor.view.contextMenuAt.disconnect()
    editor.view.contextMenuAt.connect(editor.show_canvas_menu)
    editor.controller.clear_selection()
    item = editor.photo
    point = editor.view._to_view(item.x + item.w / 2, item.y + item.h / 2).toPoint()
    event = QContextMenuEvent(QContextMenuEvent.Reason.Mouse, point, editor.view.viewport().mapToGlobal(point))
    editor.view.contextMenuEvent(event)
    assert editor.controller.selection == [item.id] and shown


def test_encoding_link_explains_when_no_list(editor):
    messages = []
    editor.notify = lambda text, *a, **k: messages.append(text)
    editor.insert_panel.encoding_link.click()
    assert messages and "명단" in messages[0]
