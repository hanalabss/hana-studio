"""디자인 화면 테스트가 함께 쓰는 픽스처. 각 테스트 파일이 `from pages_design_helpers import *`로 가져온다."""

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

ROOT = str(Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

__all__ = [
    "FakeDialogs", "FakeEngine", "find_item", "make_image", "page", "people_store", "pick", "pump",
    "settings", "theme_manager", "workspace",
]

PEOPLE = [
    ("강민서", "영업1팀", "대리", "HN240102"),
    ("고은채", "디자인팀", "사원", "HN250311"),
    ("김하늘", "경영지원팀", "사원", "HN260417"),
    ("남지호", "개발팀", "과장", "HN190705"),
]


class FakeEngine:
    """모델 없이 도는 배경 분리. 가운데 절반을 사람으로 본다."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.fail_with: str | None = None
        self.ready = True

    def prepare(self, parent=None) -> bool:
        return self.ready

    def default_threshold(self) -> int:
        return 128

    def compute_alpha(self, image_path, progress):
        from PIL import Image

        self.calls.append(image_path)
        progress("가려내는 중")
        if self.fail_with:
            raise RuntimeError(self.fail_with)
        with Image.open(image_path) as image:
            width, height = image.size
        alpha = np.zeros((height, width), dtype=np.uint8)
        alpha[height // 4: height * 3 // 4, width // 4: width * 3 // 4] = 255
        alpha[height // 4: height * 3 // 4, width // 8: width // 4] = 100   # 흐릿한 가장자리
        return alpha


class FakeDialogs:
    def __init__(self) -> None:
        self.design_path: str | None = None
        self.save_path: str | None = None
        self.image_path: str | None = None
        self.export_path: str | None = None
        self.unsaved_answer = "discard"
        self.unsaved_asked = 0

    def open_design(self, parent):
        return self.design_path

    def save_design(self, parent, suggested):
        return self.save_path

    def open_image(self, parent, title):
        return self.image_path

    def save_image(self, parent, suggested):
        return self.export_path

    def ask_unsaved(self, parent, name):
        self.unsaved_asked += 1
        return self.unsaved_answer


def make_image(path, size=(80, 100), color=(40, 90, 200)) -> str:
    from PIL import Image

    Image.new("RGB", size, color).save(str(path))
    return str(path)


def pump(qapp, rounds: int = 4) -> None:
    for _ in range(rounds):
        qapp.processEvents()


def pick(combo, data) -> None:
    """콤보에서 사용자가 항목을 고른 것처럼 한다."""
    index = combo.findData(data)
    assert index >= 0, f"콤보에 없는 값: {data!r}"
    combo.setCurrentIndex(index)
    combo.activated.emit(index)


def find_item(controller, name: str):
    return next(item for item in controller.side().items if item.name == name)


@pytest.fixture
def settings(tmp_path):
    from PySide6.QtCore import QSettings

    return QSettings(str(tmp_path / "design-test.ini"), QSettings.IniFormat)


@pytest.fixture(autouse=True)
def theme_manager(qapp, settings):
    from ui.theme import ThemeManager

    manager = ThemeManager(settings)
    ThemeManager.set_instance(manager)
    manager.set_mode("light", persist=False)
    manager.apply(qapp)
    yield manager
    from PySide6.QtCore import QCoreApplication, QEvent

    for widget in qapp.topLevelWidgets():
        widget.close()
        widget.deleteLater()
    qapp.processEvents()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    ThemeManager.set_instance(None)
    qapp.setStyleSheet("")


@pytest.fixture
def workspace(qapp, theme_manager):
    from ui.pages.workspace import Workspace

    ws = Workspace()
    yield ws
    ws.close()


@pytest.fixture
def page(qapp, workspace):
    from ui.pages.design import DesignPage

    widget = DesignPage(workspace, background_engine=FakeEngine())
    widget.dialogs = FakeDialogs()
    widget.resize(1196, 612)
    widget.show()
    pump(qapp)
    yield widget
    widget.inspector.image.wait_for_removal(5000)
    widget.dialogs.unsaved_answer = "discard"
    widget.close()


@pytest.fixture
def people_store(tmp_path):
    """네 사람짜리 명단. 사진 열이 있고, 마지막 사람은 사진 파일이 없다."""
    from data.store import DataStore

    photos = tmp_path / "photos"
    photos.mkdir()
    store = DataStore(str(tmp_path / "people.db"))
    for name in ("이름", "부서", "직급", "사번"):
        store.add_column(name)
    store.add_column("사진", "photo", {"folder": str(photos)})
    rows = []
    for index, (name, team, rank, number) in enumerate(PEOPLE):
        if index != len(PEOPLE) - 1:
            make_image(photos / f"{number}.png", color=(200, 60 * index, 90))
        rows.append({"이름": name, "부서": team, "직급": rank, "사번": number, "사진": number})
    store.insert_many(rows)
    return store
