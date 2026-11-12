"""틀 전체: 빠른 인쇄(기존 화면), 밝게·어둡게 바꾸기, 여러 번 켜고 끄기."""
import os
import subprocess
import sys
import textwrap
from pathlib import Path

from app_helpers import *  # noqa: F401,F403
from app_helpers import click_footer, find_printers, import_people, pick_template, pump, wait_until

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QCheckBox, QPushButton, QSpinBox

ROOT = Path(__file__).resolve().parents[2]
PAGES = ("design", "list", "print", "history", "printer", "settings")


def test_quick_print_legacy_screen_opens_inside_the_shell(app_env, app_window, monkeypatch):
    import hana_studio

    window, app = app_window, app_env
    # 기존 화면이 나중에 하는 무거운 일(AI 모델, 실제 프린터 DLL)과 저장소의 config.json 쓰기는 막는다.
    monkeypatch.setattr(hana_studio.HanaStudio, "_lazy_initialize", lambda self: None)
    monkeypatch.setattr(hana_studio.config, "save_settings", lambda *args, **kwargs: None)

    assert "quick" in window.page_ids()
    assert window.navigate("quick")
    host = window.page("quick")
    assert host.load_now(), host.error()
    legacy = host.legacy_window()
    # 기존 화면은 패널과 뷰어를 조금씩 나눠 만든다. 다 만들 때까지 기다린다.
    wait_until(app, lambda: getattr(legacy.ui, "_panels_initialized", True)
               and getattr(legacy.ui, "_viewers_initialized", True))
    pump(app, 50)
    assert isinstance(legacy, hana_studio.HanaStudio) and host.is_loaded() and host.error() is None
    assert not legacy.isWindow() and legacy.isVisibleTo(window)       # 따로 뜬 창이 아니라 틀 안에 있다
    assert window.windowTitle().endswith("빠른 인쇄 - Hana Studio") or "빠른 인쇄" in window.windowTitle()

    # 기존 화면의 자기 위젯이 살아 있다
    buttons = [b for b in legacy.findChildren(QPushButton) if b.isVisibleTo(legacy)]
    assert buttons and all(button.isEnabled() or button.text() for button in buttons)
    spin = next((s for s in legacy.findChildren(QSpinBox) if s.isVisibleTo(legacy)), None)
    if spin is not None:
        start = spin.value()
        spin.setFocus()
        QTest.keyClick(spin, Qt.Key_Up)
        assert spin.value() == start + 1
    check = next((c for c in legacy.findChildren(QCheckBox) if c.isVisibleTo(legacy) and c.isEnabled()), None)
    if check is not None:
        was = check.isChecked()
        check.click()
        assert check.isChecked() != was
        check.click()
    image = window.grab().toImage()
    assert not image.isNull() and image.width() >= 1000

    # 다른 화면으로 갔다 와도 다시 만들지 않는다
    window.navigate("design")
    window.navigate("quick")
    assert host.legacy_window() is legacy


def test_theme_switch_on_every_page_updates_canvas_palette(app_env, app_window, people_csv, photo_folder):
    from ui.theme import ThemeManager

    window, app = app_window, app_env
    find_printers(app, window)
    pick_template(window, "employee")
    click_footer(window, "다음: 명단 넣기")
    import_people(app, window, people_csv, photo_folder)
    manager = ThemeManager.instance()
    view = window.page("design").view

    def canvas_background() -> str:
        window.navigate("design")
        pump(app, 30)
        image = view.viewport().grab().toImage()
        return image.pixelColor(3, 3).name()

    light_canvas = canvas_background()
    for page_id in PAGES:
        window.navigate(page_id)
        pump(app, 20)
        for mode in ("dark", "light", "dark"):
            window.top_bar.themeToggleRequested.emit()
            pump(app, 20)
            assert manager.theme_name == mode, page_id
            assert manager.theme.is_dark == (mode == "dark")
            assert not window.grab().toImage().isNull()
        dark_canvas = canvas_background()
        assert dark_canvas != light_canvas, f"{page_id}에서 어둡게 바꿨는데 캔버스 색이 그대로입니다"
        window.navigate(page_id)
        window.top_bar.themeToggleRequested.emit()
        pump(app, 20)
        assert manager.theme_name == "light"
        assert canvas_background() == light_canvas, f"{page_id}에서 밝게 되돌렸는데 캔버스 색이 다릅니다"


_LIFECYCLE = r'''
import gc, os, sys, threading, time
sys.path.insert(0, r"{root}")
os.environ["QT_QPA_PLATFORM"] = "offscreen"
work = r"{work}"
for name, value in (("LOCALAPPDATA", "local"), ("APPDATA", "roaming"), ("HANA_STUDIO_DATA_DIR", "lists")):
    os.environ[name] = os.path.join(work, value)
os.environ["HANA_STUDIO_SETTINGS_FILE"] = os.path.join(work, "settings.ini")
os.environ["HANA_STUDIO_LEGACY_CONFIG"] = os.path.join(work, "no-config.json")
time.sleep = lambda seconds: None

from PySide6.QtCore import QCoreApplication, QEvent, QEventLoop, QSettings, QThread, qInstallMessageHandler
from PySide6.QtWidgets import QApplication

messages = []
qInstallMessageHandler(lambda kind, context, text: messages.append(text))
app = QApplication([])

from data.store import DataStore
from design import presets
from printer import registry
from printer.history import JobHistory
from printer.print_queue import PrintQueueManager
from services.print_service import PrintService
from services.printers import PrinterService
from ui.app import build_main_window
from ui.pages.printing import STATE_DONE
from ui.pages.services import AppServices
from ui.pages.settings.prefs import Prefs
from ui.theme import ThemeManager

registry.use_fakes(True)
settings = QSettings(os.environ["HANA_STUDIO_SETTINGS_FILE"], QSettings.IniFormat)
manager = ThemeManager(settings)
ThemeManager.set_instance(manager)
manager.apply(app)
Prefs.set_instance(Prefs(settings))


def wait(predicate, seconds=30):
    deadline = time.monotonic() + seconds
    while not predicate():
        app.processEvents(QEventLoop.AllEvents, 20)
        QThread.msleep(2)
        assert time.monotonic() < deadline, "시간 안에 끝나지 않았습니다"


def settle():
    for _ in range(5):
        app.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    gc.collect()


threads = []
for round_number in range(5):
    list_path = os.path.join(work, f"people{{round_number}}.hsdata")
    history_path = os.path.join(work, f"history{{round_number}}.sqlite")
    store = DataStore(list_path)
    store.add_column("이름")
    store.add_column("사번")
    store.insert_many([{{"이름": "김하나", "사번": "A1"}}, {{"이름": "이두리", "사번": "A2"}}])
    store.close()

    history = JobHistory(history_path)
    queue = PrintQueueManager(history=history)
    printers = PrinterService(queue=queue, settings_path=os.path.join(work, "printer.json"), poll_interval_ms=100)
    services = AppServices(printers=printers, printing=PrintService(queue=queue, history=history),
                           history=history, queue=queue)
    window = build_main_window(services=services, start_services=False)
    window.page("design").dialogs.ask_unsaved = lambda parent, name: "discard"
    window.resize(1280, 800)
    window.show()
    services.start()
    wait(lambda: not printers.is_busy and printers.devices() and printers.status() is not None)

    window.page("design").start_with_template("employee")
    window.navigate("list")
    assert window.page("list").open_list(list_path)
    window.page("list").toggle_all_in_view()
    for page_id in ("print", "history", "printer", "settings", "print"):
        window.navigate(page_id)
        app.processEvents()
    page = window.page("print")
    # 이 명단에 없는 항목(사진, 부서 …)을 쓰는 칸은 지워 인쇄할 수 있게 한다.
    document = window.workspace.editor.document
    for _side, item in list(document.iter_items()):
        if item.source is not None and not item.source.columns() <= {{"이름", "사번"}}:
            window.workspace.editor.delete([item.id])
    page.refresh_now()
    page.acknowledge_warnings(True)
    assert page.start_printing() is not None, [p.text for p in page.problems()]
    wait(lambda: page.state() == STATE_DONE)
    assert history.list_jobs()[0].success_cards == 2

    assert window.close()
    window.shutdown()
    wait(lambda: not printers.is_busy and not services.printing.is_printing)
    queue.wait_idle(5000)
    window.deleteLater()
    del window, page, services, printers, queue, history, document, store
    settle()

    # 윈도우에서는 열려 있는 파일을 지울 수 없다: 지워지면 잠금이 남지 않은 것이다.
    for path in (list_path, history_path):
        os.remove(path)
        for suffix in ("-wal", "-shm", "-journal"):
            assert not os.path.exists(path + suffix), path + suffix
    threads.append(threading.active_count())

assert max(threads[1:]) <= threads[0], threads          # 켜고 끌 때마다 스레드가 늘지 않는다
bad = [m for m in messages if "QThread" in m or "QObject" in m or "QTimer" in m or "QSqlDatabase" in m
       or "destroyed" in m.lower() or "leak" in m.lower()]
assert not bad, bad
print("WARNINGS", len(messages), messages[:5])
print("LIFECYCLE-OK", threads)
'''


def test_build_and_destroy_main_window_five_times_then_exit_cleanly(tmp_path):
    script = tmp_path / "lifecycle.py"
    work = tmp_path / "work"
    work.mkdir()
    script.write_text(textwrap.dedent(_LIFECYCLE).format(root=str(ROOT), work=str(work)), encoding="utf-8")
    # 이 테스트 파일은 APPDATA를 임시 폴더로 돌려 두므로, 새 파이썬이 사용자 site-packages를 찾도록 길을 넘긴다.
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONIOENCODING="utf-8", PYTHONFAULTHANDLER="1",
               PYTHONPATH=os.pathsep.join(path for path in sys.path if path))
    result = subprocess.run([sys.executable, str(script)], cwd=str(tmp_path), env=env, capture_output=True,
                            timeout=240)
    stdout = result.stdout.decode("utf-8", "replace")
    stderr = result.stderr.decode("utf-8", "replace")
    assert result.returncode == 0, f"종료 코드 {result.returncode}\n{stdout[-3000:]}\n{stderr[-3000:]}"
    assert "LIFECYCLE-OK" in stdout, stdout[-2000:]
    for sign in ("Traceback", "QThread: Destroyed", "Fatal Python error", "Windows fatal exception"):
        assert sign not in stderr, stderr[-3000:]
    assert not list(work.glob("*.hsdata")) and not list(work.glob("history*.sqlite"))
