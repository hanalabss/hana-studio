"""인쇄·기록·프린터·설정 화면 테스트가 함께 쓰는 준비물. 각 테스트 파일이 `from print_helpers import *`로 가져간다."""
import sys
import time
from pathlib import Path

import pytest
from PIL import Image
from PySide6.QtCore import QEventLoop, QThread

ROOT = str(Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

__all__ = [
    "PEOPLE_CSV", "env", "history", "queue", "services", "photo_dir", "store", "workspace", "document",
    "rtai_lib", "smart_lib", "wait_until", "pump", "make_page", "finish_job", "Gate", "discover",
]

PEOPLE_CSV = (
    "이름,사번,부서,직급,사진\n"
    "김철수,A0001,영업부,과장,kim\n"
    "이영희,A0002,개발부,대리,lee\n"
    "박민수,A0003,개발부,사원,park\n"
    "최지우,A0004,인사부,부장,choi\n"
    "정하나,A0005,영업부,사원,jung\n"
)
_monotonic = time.monotonic


def wait_until(app, predicate, timeout=20.0):
    deadline = _monotonic() + timeout
    while not predicate():
        app.processEvents(QEventLoop.AllEvents, 20)
        QThread.msleep(2)
        assert _monotonic() < deadline, "시간 안에 끝나지 않았습니다"
    app.processEvents(QEventLoop.AllEvents, 20)


def pump(app, milliseconds=60):
    deadline = _monotonic() + milliseconds / 1000.0
    while _monotonic() < deadline:
        app.processEvents(QEventLoop.AllEvents, 10)
        QThread.msleep(2)


@pytest.fixture(autouse=True)
def env(qapp, tmp_path, monkeypatch):
    """가짜 프린터, 사용자 폴더와 설정 파일 격리, 테스트마다 새 테마와 설정."""
    from PySide6.QtCore import QCoreApplication, QEvent, QSettings

    from design.renderer import ensure_gui
    from printer import registry
    from ui.pages.settings.prefs import Prefs
    from ui.theme import ThemeManager

    ensure_gui()
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "appdata"))
    monkeypatch.setenv("HANA_STUDIO_SETTINGS_FILE", str(tmp_path / "settings.ini"))
    monkeypatch.setenv("HANA_STUDIO_LEGACY_CONFIG", str(tmp_path / "no-config.json"))
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    registry.use_fakes(True)
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat)
    manager = ThemeManager(settings)
    ThemeManager.set_instance(manager)
    manager.set_mode("light", persist=False)
    manager.apply(qapp)
    Prefs.set_instance(Prefs(settings))
    yield qapp
    for widget in qapp.topLevelWidgets():
        widget.close()
        widget.deleteLater()
    qapp.processEvents()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    Prefs.set_instance(None)
    ThemeManager.set_instance(None)
    qapp.setStyleSheet("")
    registry.use_fakes(False)
    registry._use_fakes = None
    for family in ("rtai", "smart"):
        registry.register_adapter(family, None)


@pytest.fixture
def history(tmp_path):
    from printer.history import JobHistory

    return JobHistory(str(tmp_path / "history.sqlite"))


@pytest.fixture
def queue(qapp, history):
    from printer.print_queue import PrintQueueManager

    manager = PrintQueueManager(history=history)
    yield manager
    manager.cancel_all()


@pytest.fixture
def services(qapp, queue, history, tmp_path):
    from services.print_service import PrintService
    from services.printers import PrinterService
    from ui.pages.services import AppServices

    printers = PrinterService(queue, settings_path=str(tmp_path / "printer.json"), poll_interval_ms=200)
    bundle = AppServices(printers=printers, printing=PrintService(queue), history=history)
    yield bundle
    bundle.shutdown()
    wait_until(qapp, lambda: not printers.is_busy)


@pytest.fixture
def photo_dir(tmp_path):
    folder = tmp_path / "photos"
    folder.mkdir()
    for index, name in enumerate(("kim", "lee", "park", "choi", "jung")):
        Image.new("RGB", (120, 160), (40 * index + 30, 80, 200 - 30 * index)).save(folder / f"{name}.png")
    return folder


@pytest.fixture
def store(tmp_path, photo_dir):
    from data.importers import import_into, preview_csv
    from data.store import DataStore

    csv_path = tmp_path / "people.csv"
    csv_path.write_text(PEOPLE_CSV, encoding="utf-8-sig")
    opened = DataStore(tmp_path / "people.hsdata")
    result = import_into(opened, preview_csv(csv_path))
    assert result.inserted == 5
    opened.set_column_type("사진", "photo", {"folder": str(photo_dir)})
    return opened


@pytest.fixture
def document():
    from design import presets

    return presets.build_template("employee")


@pytest.fixture
def workspace(qapp, store, document):
    """사원증 디자인과 5명 명단이 열린 작업 상태."""
    from ui.pages.workspace import Workspace

    ws = Workspace()
    ws.editor.set_document(document)
    ws.set_store(store)
    yield ws
    ws.close()


@pytest.fixture
def rtai_lib():
    from printer import registry

    registry.discover(["rtai"])
    return registry._fake_library("rtai")


@pytest.fixture
def smart_lib():
    from printer import registry

    registry.discover(["smart"])
    return registry._fake_library("smart")


def discover(qapp, services):
    """가짜 프린터 두 대를 찾고 첫 번째(RTAI)의 상태까지 읽어 둔다."""
    printers = services.printers
    printers.refresh()
    wait_until(qapp, lambda: not printers.is_busy and printers.devices())
    printers.poll_now()
    wait_until(qapp, lambda: not printers.is_busy and printers.status() is not None)


def make_page(qapp, workspace, services, find_printers=True):
    """프린터를 찾아 둔 인쇄 화면."""
    from ui.pages.printing import PrintPage

    if find_printers:
        discover(qapp, services)
    page = PrintPage(workspace, services)
    page.resize(1200, 640)
    page.refresh_now()
    return page


def finish_job(qapp, page):
    from ui.pages.printing import STATE_DONE

    wait_until(qapp, lambda: page.state() == STATE_DONE)


class Gate:
    """n번째 카드에서 인쇄 스레드를 붙잡아 두는 가짜 프린터 덧씌우개."""

    def __init__(self, registry, hold_on):
        import threading

        from printer.rtai_adapter import RtaiAdapter

        self.entered = threading.Event()
        self.release = threading.Event()
        self.count = 0
        gate = self

        class Held(RtaiAdapter):
            def print_card(self, job, progress=None):
                gate.count += 1
                if gate.count == hold_on:
                    job.front        # 이미지를 먼저 받아 둔다(실기에서 카드가 들어간 상태)
                    gate.entered.set()
                    gate.release.wait(15)
                return super().print_card(job, progress)

        registry.register_adapter(
            "rtai", lambda device: Held(lib=registry._fake_library("rtai"), settle_seconds=0.0))
