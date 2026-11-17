"""카메라로 사진 찍기(다눌의 '사진 캡쳐'). 실제 카메라 없이 확인할 수 있는 부분."""

from PySide6.QtGui import QColor, QImage

from ui import camera
from ui.camera import CameraDialog, crop_to_aspect, unique_path


def _image(w, h):
    image = QImage(w, h, QImage.Format.Format_RGB32)
    image.fill(QColor("#336699"))
    return image


def test_crop_to_aspect_keeps_center():
    assert (crop_to_aspect(_image(640, 480), 3 / 4).width(), crop_to_aspect(_image(640, 480), 3 / 4).height()) == (360, 480)
    wide = crop_to_aspect(_image(300, 400), 2.0)
    assert (wide.width(), wide.height()) == (300, 150)
    same = crop_to_aspect(_image(10, 10), None)
    assert (same.width(), same.height()) == (10, 10)


def test_unique_path_does_not_overwrite(tmp_path):
    first = unique_path(tmp_path, "HN:001")
    assert first.name == "HN001.jpg"
    first.write_bytes(b"x")
    assert unique_path(tmp_path, "HN:001").name == "HN001-2.jpg"


def test_dialog_without_camera_explains_and_offers_retry(qapp, monkeypatch):
    monkeypatch.setattr(CameraDialog, "devices", staticmethod(lambda: []))
    dialog = CameraDialog(aspect=3 / 4)
    assert "카메라가 없습니다" in dialog.still.text()
    assert not dialog.retry_button.isHidden()
    assert not dialog.shoot_button.isEnabled()
    dialog.set_captured(_image(640, 480))
    image = dialog.captured_image()
    assert (image.width(), image.height()) == (360, 480)
    assert not dialog.use_button.isHidden()
    dialog.reject()


def test_take_photo_returns_none_when_cancelled(qapp, monkeypatch):
    monkeypatch.setattr(CameraDialog, "devices", staticmethod(lambda: []))
    monkeypatch.setattr(CameraDialog, "exec", lambda self: CameraDialog.Rejected)
    assert camera.take_photo(None) is None
