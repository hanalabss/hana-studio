import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = str(Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class Mouse:
    """뷰포트에 마우스 이벤트를 직접 보낸다. 좌표는 뷰포트 픽셀(QPointF)."""

    def __init__(self, view, app) -> None:
        self.view = view
        self.app = app

    def at(self, x_mm: float, y_mm: float):
        return self.view.mm_to_viewport(x_mm, y_mm)

    def _send(self, kind, pos, button, buttons, modifiers) -> None:
        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QMouseEvent
        from PySide6.QtWidgets import QApplication

        viewport = self.view.viewport()
        point = QPointF(pos)
        event = QMouseEvent(kind, point, QPointF(viewport.mapToGlobal(point.toPoint())), button, buttons, modifiers)
        QApplication.sendEvent(viewport, event)

    def press(self, pos, modifiers=None) -> None:
        from PySide6.QtCore import QEvent, Qt

        left = Qt.MouseButton.LeftButton
        self._send(QEvent.Type.MouseButtonPress, pos, left, left, modifiers or Qt.KeyboardModifier.NoModifier)

    def move(self, pos, modifiers=None, pressed: bool = True) -> None:
        from PySide6.QtCore import QEvent, Qt

        buttons = Qt.MouseButton.LeftButton if pressed else Qt.MouseButton.NoButton
        self._send(
            QEvent.Type.MouseMove, pos, Qt.MouseButton.NoButton, buttons,
            modifiers or Qt.KeyboardModifier.NoModifier,
        )

    def release(self, pos, modifiers=None) -> None:
        from PySide6.QtCore import QEvent, Qt

        self._send(
            QEvent.Type.MouseButtonRelease, pos, Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton,
            modifiers or Qt.KeyboardModifier.NoModifier,
        )

    def drag(self, start, end, modifiers=None, steps: int = 5, release: bool = True) -> None:
        self.press(start, modifiers)
        for index in range(1, steps + 1):
            self.move(start + (end - start) * (index / steps), modifiers)
        if release:
            self.release(end, modifiers)

    def click(self, pos, modifiers=None) -> None:
        self.press(pos, modifiers)
        self.release(pos, modifiers)


class Editor:
    def __init__(self, controller, view, mouse, app) -> None:
        self.controller = controller
        self.view = view
        self.mouse = mouse
        self.app = app

    def set_px_per_mm(self, value: float) -> None:
        self.view.set_zoom(value / (self.view.px_per_mm / self.view.zoom))
        self.app.processEvents()


@pytest.fixture(autouse=True)
def _gui(qapp):
    from design.renderer import ensure_gui

    ensure_gui()


@pytest.fixture
def controller(qapp):
    from ui.editor import EditorController

    ctl = EditorController()
    ctl.set_snap_enabled(False)
    yield ctl
    ctl.close()


@pytest.fixture
def editor(qapp, controller):
    """컨트롤러 + 화면에 띄운 캔버스. 달라붙기는 꺼 둔다(필요한 테스트에서 켠다)."""
    from ui.editor import EditorView

    view = EditorView(controller)
    view.resize(900, 640)
    view.show()
    qapp.processEvents()
    yield Editor(controller, view, Mouse(view, qapp), qapp)
    view.close()
    view.deleteLater()
    qapp.processEvents()


@pytest.fixture
def make_png(tmp_path):
    counter = {"n": 0}

    def factory(size=(200, 100), color=(200, 30, 30)):
        from PIL import Image, ImageDraw

        image = Image.new("RGB", size, color)
        draw = ImageDraw.Draw(image)
        draw.rectangle([size[0] // 4, size[1] // 4, size[0] // 2, size[1] // 2], fill=(20, 60, 200))
        counter["n"] += 1
        path = tmp_path / f"img{counter['n']}.png"
        image.save(path)
        return str(path)

    return factory


@pytest.fixture
def qimage_array():
    def convert(image) -> np.ndarray:
        from PySide6.QtGui import QImage

        converted = image.convertToFormat(QImage.Format.Format_RGB888)
        stride = converted.bytesPerLine()
        raw = np.frombuffer(bytes(converted.constBits()), dtype=np.uint8)
        return raw.reshape(converted.height(), stride)[:, : converted.width() * 3].reshape(
            converted.height(), converted.width(), 3
        ).copy()

    return convert
