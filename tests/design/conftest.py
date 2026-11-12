import numpy as np
import pytest
from PIL import Image


def _qimage_to_array(image) -> np.ndarray:
    from PySide6.QtGui import QImage

    converted = image.convertToFormat(QImage.Format.Format_RGBA8888)
    stride = converted.bytesPerLine()
    raw = np.frombuffer(bytes(converted.constBits()), dtype=np.uint8)
    return raw.reshape(converted.height(), stride)[:, : converted.width() * 4].reshape(
        converted.height(), converted.width(), 4
    ).copy()


@pytest.fixture(autouse=True)
def _gui(qapp):
    from design.renderer import ensure_gui

    ensure_gui()


@pytest.fixture
def qimage_array():
    """QImage -> (h, w, 4) RGBA 배열."""
    return _qimage_to_array


@pytest.fixture
def make_png(tmp_path):
    """작은 PNG를 만들어 경로를 돌려준다. painter(draw, image)로 내용을 그린다."""
    counter = {"n": 0}

    def factory(size=(200, 100), color=(255, 0, 0), mode="RGB", painter=None, name=None):
        from PIL import ImageDraw

        image = Image.new(mode, size, color)
        if painter is not None:
            painter(ImageDraw.Draw(image), image)
        counter["n"] += 1
        path = tmp_path / (name or f"img{counter['n']}.png")
        image.save(path)
        return str(path)

    return factory
