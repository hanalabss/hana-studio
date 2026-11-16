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
