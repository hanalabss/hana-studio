import sys
import time
from pathlib import Path

import pytest
from PIL import Image

ROOT = str(Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from printer.base import PrinterDevice  # noqa: E402
from printer.fakes import FakeR600Library, FakeSmartLibrary  # noqa: E402


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """인쇄 흐름 안의 안정화 대기(time.sleep)를 없앤다."""
    monkeypatch.setattr(time, "sleep", lambda seconds: None)


@pytest.fixture
def make_image(tmp_path):
    counter = {"n": 0}

    def factory(size=(540, 860), color=(200, 30, 30), suffix=".jpg", name=None):
        counter["n"] += 1
        path = tmp_path / (name or f"image{counter['n']}{suffix}")
        Image.new("RGB", size, color).save(path)
        return str(path)

    return factory


@pytest.fixture
def r600_lib():
    return FakeR600Library()


@pytest.fixture
def smart_lib():
    return FakeSmartLibrary()


@pytest.fixture
def rtai_device():
    return PrinterDevice("rtai", "RTAI R600 FAKE", "usb", "RTAI R600 FAKE")


@pytest.fixture
def smart_device(smart_lib):
    item = smart_lib.devices[0]
    return PrinterDevice("smart", item.name, "usb", item.desc,
                         {"id": item.id, "dev": item.dev, "pid": item.pid})


@pytest.fixture
def card_size():
    """config.json의 인쇄 영역 크기 (짧은 변, 긴 변)."""
    from config import config
    return float(config.get("printer.card_width", 55)), float(config.get("printer.card_height", 86.6))
