"""조립된 앱 전체를 처음부터 끝까지 모는 테스트의 준비물. `from app_helpers import *`로 가져다 쓴다."""
import sys
import time
from pathlib import Path

import pytest
from PIL import Image
from PySide6.QtCore import QEventLoop, Qt, QThread
from PySide6.QtTest import QTest

ROOT = str(Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

__all__ = [
    "PEOPLE", "MISSING_PHOTO", "app_env", "app_window", "people_csv", "photo_folder", "wait_until", "pump",
    "click", "click_footer", "pick_template", "import_people", "find_printers", "select_printer",
    "print_page_ready", "wait_done", "names_in", "id_of", "journey_to_print", "fake_library",
]

# 파일의 머리글은 카드의 칸 이름과 다르다: 성명/소속부서/직급/사원번호
PEOPLE = [
    ("강민서", "영업1팀", "대리", "HN260415"),
    ("고은채", "디자인팀", "사원", "HN260416"),
    ("김하늘", "경영지원팀", "사원", "HN260417"),
    ("남지호", "개발팀", "과장", "HN230102"),
    ("문서윤", "개발팀", "사원", "HN260418"),
    ("박도윤", "영업2팀", "차장", "HN190233"),
    ("서지안", "경영지원팀", "사원", "HN260419"),
    ("양수아", "디자인팀", "대리", "HN260420"),
]
MISSING_PHOTO = ("문서윤", "서지안")
_monotonic = time.monotonic


def wait_until(app, predicate, timeout=30.0):
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


def click(widget) -> None:
    """사람이 누르듯 누른다. 꺼져 있거나 안 보이는 것을 누르려 하면 테스트가 틀린 것이다."""
    assert widget is not None, "누를 것이 없습니다"
    assert widget.isEnabled(), f"꺼져 있어 누를 수 없습니다: {getattr(widget, 'text', lambda: widget)()}"
    assert widget.isVisible(), f"보이지 않아 누를 수 없습니다: {getattr(widget, 'text', lambda: widget)()}"
    QTest.mouseClick(widget, Qt.LeftButton)


def click_footer(window, starts_with: str) -> None:
    """아래 줄의 "다음: ..." 같은 단추를 누른다."""
    from PySide6.QtWidgets import QApplication

    QApplication.processEvents()        # 아래 줄 단추는 화면이 바뀐 다음 차례에 그려진다
    buttons = [b for b in window.footer_buttons() if b.text().startswith(starts_with)]
    assert buttons, f"아래 줄에 '{starts_with}…' 단추가 없습니다: {[b.text() for b in window.footer_buttons()]}"
    click(buttons[0])


@pytest.fixture(autouse=True)
def app_env(qapp, tmp_path, monkeypatch):
    """가짜 프린터, 사용자 폴더·설정 파일 격리, 테스트마다 새 테마와 설정."""
    from PySide6.QtCore import QCoreApplication, QEvent, QSettings

    from design.renderer import ensure_gui
    from printer import registry
    from ui.pages.settings.prefs import Prefs
    from ui.theme import ThemeManager

    ensure_gui()
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setenv("APPDATA", str(tmp_path / "roaming"))
    monkeypatch.setenv("HANA_STUDIO_DATA_DIR", str(tmp_path / "lists"))
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


def build_window(tmp_path, name: str = "history.sqlite"):
    """가짜 프린터와 임시 기록 파일로 조립한 메인 창. (창, 서비스 묶음)을 돌려준다."""
    from printer.history import JobHistory
    from printer.print_queue import PrintQueueManager
    from services.print_service import PrintService
    from services.printers import PrinterService
    from ui.app import build_main_window
    from ui.pages.services import AppServices

    history = JobHistory(str(tmp_path / name))
    queue = PrintQueueManager(history=history)
    services = AppServices(
        printers=PrinterService(queue=queue, settings_path=str(tmp_path / "printer.json"), poll_interval_ms=200),
        printing=PrintService(queue=queue, history=history), history=history, queue=queue)
    window = build_main_window(services=services, start_services=False)
    # 창을 닫을 때 "저장하지 않았습니다" 창이 떠서 기다리지 않게 한다(아무도 눌러 줄 수 없다).
    window.page("design").dialogs.ask_unsaved = lambda parent, name: "discard"
    window.page("print").confirm = lambda *args, **kwargs: True
    return window, services


@pytest.fixture
def app_window(app_env, tmp_path):
    window, services = build_window(tmp_path)
    window.resize(1280, 800)
    window.show()
    pump(app_env, 30)
    yield window
    services.shutdown()
    wait_until(app_env, lambda: not services.printers.is_busy and not services.printing.is_printing)
    window.workspace.close()
    window.close()
    window.deleteLater()


@pytest.fixture
def people_csv(tmp_path):
    """cp949로 저장한, 머리글 이름이 카드와 다른 CSV."""
    path = tmp_path / "신입사원_명단.csv"
    lines = ["성명,소속부서,직급,사원번호"] + [",".join(row) for row in PEOPLE]
    path.write_bytes("\r\n".join(lines).encode("cp949"))
    return path


@pytest.fixture
def photo_folder(tmp_path):
    """사번으로 이름 붙인 사진 폴더. 두 명은 사진이 없다."""
    folder = tmp_path / "사진"
    folder.mkdir()
    for index, (name, _dept, _rank, number) in enumerate(PEOPLE):
        if name in MISSING_PHOTO:
            continue
        # 위아래 색이 달라 사진을 옮기면 그림이 달라진다.
        image = Image.new("RGB", (120, 200), (30 + index * 20, 90, 200 - index * 15))
        image.paste((250, 240, 40), (0, 0, 120, 50))
        image.save(folder / f"{number}.png")
    return folder


# ---------------------------------------------------------------- 여정의 단계들

def pick_template(window, key: str = "employee") -> None:
    """시작 화면에서 모양을 눌러 꾸미는 화면으로 들어간다."""
    design = window.page("design")
    assert window.current_page_id() == "design" and not design.is_started()
    click(design.start_panel.cards[key])
    assert design.is_started()


def import_people(app, window, csv_path, folder):
    """명단 화면에서 파일을 가져온다(가져오기 창을 단계대로 넘긴다). 가져오기 창을 돌려준다."""
    people = window.page("list")
    assert window.current_page_id() == "list"
    wizard = people.start_import(csv_path, run=False)
    assert wizard.wait_idle()
    wizard.go_next()
    if folder is not None:
        wizard.set_photo_folder(folder)
    wizard.start_import()
    assert wizard.wait_idle()
    wizard.go_next()
    pump(app)
    return wizard


def find_printers(app, window) -> None:
    printers = window.services.printers
    printers.refresh()
    wait_until(app, lambda: not printers.is_busy and printers.devices())
    printers.poll_now()
    wait_until(app, lambda: not printers.is_busy and printers.status() is not None)


def select_printer(app, window, family: str):
    printers = window.services.printers
    device = next(d for d in printers.devices() if d.family == family)
    printers.select(device)
    printers.poll_now()
    wait_until(app, lambda: not printers.is_busy and printers.status() is not None
               and printers.current() == device)
    return device


def print_page_ready(app, window):
    """인쇄 화면으로 가서 미뤄 둔 계산을 끝낸다."""
    if window.current_page_id() != "print":
        window.navigate("print")
    page = window.page("print")
    page.refresh_now()
    pump(app, 30)
    return page


def wait_done(app, page) -> None:
    from ui.pages.printing import STATE_DONE

    wait_until(app, lambda: page.state() == STATE_DONE)
    pump(app, 30)


def names_in(people_page) -> list[str]:
    model = people_page.model
    return [model.record_at(row)["이름"] for row in range(model.rowCount())]


def id_of(store, name: str) -> int:
    return next(record["_id"] for record in store.rows() if record["이름"] == name)


def fake_library(family: str = "rtai"):
    from printer import registry

    registry.discover([family])
    return registry._fake_library(family)


def journey_to_print(app, window, csv_path, folder):
    """시작 화면 → 사원증 → 명단 가져오기 → 인쇄 화면까지 사람이 가는 길 그대로. 인쇄 화면을 돌려준다."""
    find_printers(app, window)
    pick_template(window, "employee")
    click_footer(window, "다음: 명단 넣기")
    import_people(app, window, csv_path, folder)
    click_footer(window, "다음:")
    assert window.current_page_id() == "print"
    return print_page_ready(app, window)
