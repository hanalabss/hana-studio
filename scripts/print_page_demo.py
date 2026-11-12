"""인쇄·기록·프린터·설정 화면 미리 보기.

    python scripts/print_page_demo.py --fake                  가짜 프린터와 예시 명단으로 띄운다
    python scripts/print_page_demo.py --fake --theme dark     어두운 화면으로
    python scripts/print_page_demo.py --fake --fail-at 3      세 번째 카드에서 인쇄가 실패한다
    python scripts/print_page_demo.py --fake --slow 0.3       카드 한 장을 더 빨리(단계마다 0.3초)
    python scripts/print_page_demo.py --shot <폴더>            주요 장면을 PNG로 저장 (화면 없이, 가짜 프린터)
    python scripts/print_page_demo.py                         실제 프린터를 찾는다(예시 명단)
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PEOPLE = (
    ("강민서", "HN260415", "영업1팀", "대리"), ("고은채", "HN260416", "디자인팀", "사원"),
    ("김하늘", "HN260417", "경영지원팀", "사원"), ("문서윤", "HN260418", "개발팀", "사원"),
    ("서지안", "HN260419", "경영지원팀", "사원"), ("양수아", "HN260420", "디자인팀", "대리"),
    ("이준우", "HN260421", "개발팀", "사원"), ("장하린", "HN260422", "영업2팀", "사원"),
    ("최유나", "HN260423", "개발팀", "사원"), ("한지민", "HN260424", "디자인팀", "사원"),
    ("남지호", "HN230102", "개발팀", "과장"), ("박도윤", "HN190233", "영업2팀", "차장"),
)
_real_sleep = time.sleep


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="하나스튜디오 인쇄 화면 미리 보기")
    parser.add_argument("--theme", choices=("light", "dark"), default=None)
    parser.add_argument("--shot", metavar="DIR", help="주요 장면을 PNG로 저장하고 끝낸다")
    parser.add_argument("--fake", action="store_true", help="가짜 프린터를 쓴다")
    parser.add_argument("--fail-at", type=int, default=0, metavar="N", help="N번째 카드에서 실패하게 한다(--fake)")
    parser.add_argument("--slow", type=float, default=0.8, metavar="초", help="가짜 프린터의 단계별 시간")
    parser.add_argument("--page", default="print", help="처음 열 화면 id")
    return parser.parse_args(argv)


def _prepare_environment(args: argparse.Namespace, folder: Path) -> None:
    if args.shot:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    if args.shot or args.fake:
        os.environ["LOCALAPPDATA"] = str(folder / "appdata")
        os.environ["HANA_STUDIO_SETTINGS_FILE"] = str(folder / "settings.ini")
        os.environ["HANA_STUDIO_LEGACY_CONFIG"] = str(folder / "no-config.json")


# --------------------------------------------------------------------------- 예시 자료

def build_document():
    from design import presets
    from design.model import RectItem, TextItem

    document = presets.build_template("employee", name="2026 신입 사원증")
    document.dual_side = True
    back = document.side("back")
    width, height = document.side_size_mm("back")
    back.add_item(RectItem(name="아래 띠", x=0, y=height - 9, w=width, h=9, fill="#1f4e79", stroke_color=None))
    back.add_item(TextItem(name="안내", text="이 카드를 주우신 분은 가까운 안내 데스크에 맡겨 주세요.",
                           x=6, y=10, w=width - 12, h=16, font_size_pt=8, wrap=True, color="#333333"))
    back.add_item(TextItem(name="회사", text="하나상사 인사팀  02-0000-0000", x=6, y=30, w=width - 12, h=8,
                           font_size_pt=8, bold=True, color="#1f4e79"))
    return document


def build_store(folder: Path, missing_photos: tuple[str, ...] = (), printed: int = 0):
    """12명 명단과 사진 폴더. missing_photos의 사번은 사진 파일을 만들지 않는다."""
    from PIL import Image, ImageDraw

    from data.importers import import_into, preview_csv
    from data.store import DataStore

    photos = folder / "photos"
    photos.mkdir(parents=True, exist_ok=True)
    lines = ["이름,사번,부서,직급,사진"]
    for index, (name, number, team, rank) in enumerate(PEOPLE):
        lines.append(f"{name},{number},{team},{rank},{number}")
        if number in missing_photos:
            continue
        image = Image.new("RGB", (300, 400), (188 - index * 7, 204 - index * 4, 224 - index * 3))
        draw = ImageDraw.Draw(image)
        draw.ellipse((95, 70, 205, 180), fill=(236, 222, 208))
        draw.rounded_rectangle((55, 205, 245, 420), 60, fill=(52 + index * 9, 70 + index * 5, 104))
        image.save(photos / f"{number}.png")
    csv_path = folder / "people.csv"
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    path = folder / "people.hsdata"
    if path.exists():
        path.unlink()
    store = DataStore(path)
    import_into(store, preview_csv(csv_path))
    store.set_column_type("사진", "photo", {"folder": str(photos)})
    if printed:
        store.mark_printed(store.ids()[-printed:])
    return store


# --------------------------------------------------------------------------- 가짜 프린터 조절

class FakeControl:
    """가짜 RTAI 프린터를 느리게 하거나 n번째 카드에서 붙잡아 둔다."""

    def __init__(self) -> None:
        self.delay = 0.0
        self.hold_on = 0
        self.count = 0
        self.entered = threading.Event()
        self.release = threading.Event()

    def install(self) -> None:
        from printer import registry
        from printer.rtai_adapter import RtaiAdapter

        control = self

        class Slow(RtaiAdapter):
            def print_card(self, job, progress=None):
                control.count += 1
                if control.hold_on and control.count == control.hold_on:
                    job.front
                    if progress is not None:
                        progress(45, "앞면 인쇄 중...")
                    control.entered.set()
                    control.release.wait(30)
                if control.delay:
                    for percent, text in ((10, "카드 넣는 중..."), (45, "앞면 인쇄 중..."), (80, "뒷면 인쇄 중...")):
                        if progress is not None:
                            progress(percent, text)
                        _real_sleep(control.delay)
                return super().print_card(job, progress)

        registry.register_adapter(
            "rtai", lambda device: Slow(lib=registry._fake_library("rtai"), settle_seconds=0.0))

    def reset(self) -> None:
        self.hold_on = 0
        self.count = 0
        self.entered.clear()
        self.release.clear()


# --------------------------------------------------------------------------- 조립

class Demo:
    def __init__(self, folder: Path, fake: bool, missing_photos: tuple[str, ...] = (), printed: int = 0):
        from printer import registry
        from printer.history import JobHistory
        from printer.print_queue import PrintQueueManager
        from services.print_service import PrintService
        from services.printers import PrinterService
        from ui.pages.history import HistoryPage
        from ui.pages.printer import PrinterPage
        from ui.pages.printing import PrintPage
        from ui.pages.services import AppServices
        from ui.pages.settings import Prefs, SettingsPage
        from ui.pages.workspace import Workspace
        from ui.shell import PAGE_HISTORY, PAGE_PRINT, PAGE_PRINTER, PAGE_SETTINGS, AppShell

        self.folder = folder
        folder.mkdir(parents=True, exist_ok=True)
        self.control = FakeControl()
        if fake:
            registry.use_fakes(True)
            self.control.install()
            history = JobHistory(str(folder / "history.sqlite"))
            queue = PrintQueueManager(history=history)
            printers = PrinterService(queue, settings_path=str(folder / "printer.json"))
            self.services = AppServices(printers=printers, printing=PrintService(queue), history=history)
        else:
            self.services = AppServices()
        self.workspace = Workspace()
        self.workspace.editor.set_document(build_document())
        self.workspace.set_store(build_store(folder, missing_photos, printed))

        self.shell = AppShell()
        self.shell.set_document("2026 신입 사원증", True)
        self.shell.set_step_done("design", True)
        self.shell.set_step_done("list", True)
        self.print_page = PrintPage(self.workspace, self.services)
        self.history_page = HistoryPage(self.workspace, self.services)
        self.printer_page = PrinterPage(self.services)
        self.settings_page = SettingsPage(services=self.services)
        self.shell.register_page(PAGE_PRINT, self.print_page, "인쇄", "print")
        self.shell.register_page(PAGE_HISTORY, self.history_page, "기록", "history")
        self.shell.register_page(PAGE_PRINTER, self.printer_page, "프린터", "printer")
        self.shell.register_page(PAGE_SETTINGS, self.settings_page, "설정", "settings")
        self.services.start(Prefs.instance().retention_days())

    def close(self) -> None:
        self.control.release.set()
        self.services.shutdown()
        self.shell.close()
        self.workspace.close()
        self.shell.deleteLater()


# --------------------------------------------------------------------------- PNG 저장

def _wait(app, predicate, timeout: float = 30.0) -> None:
    from PySide6.QtCore import QEventLoop

    deadline = time.monotonic() + timeout
    while not predicate():
        app.processEvents(QEventLoop.AllEvents, 20)
        _real_sleep(0.003)
        if time.monotonic() > deadline:
            raise RuntimeError("장면을 준비하다가 시간이 넘었습니다.")
    for _ in range(4):
        app.processEvents()


def _pump(app, seconds: float = 0.15) -> None:
    deadline = time.monotonic() + seconds
    _wait(app, lambda: time.monotonic() > deadline)


def export_screenshots(app, out_dir: str | os.PathLike, themes=("light", "dark")) -> list[Path]:
    from PySide6.QtWidgets import QApplication, QScrollArea

    from printer import registry
    from printer.fakes import R600_ERR_PRINT
    from ui.pages.printing import STATE_DONE
    from ui.pages.settings import Prefs
    from ui.shell import PAGE_HISTORY, PAGE_PRINT, PAGE_PRINTER, PAGE_SETTINGS
    from ui.theme import ThemeManager

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    manager = ThemeManager.instance()
    manager.apply(app)

    def save(widget, name: str) -> None:
        path = out / name
        if not widget.grab().save(str(path)):
            raise RuntimeError(f"그림을 저장하지 못했습니다: {path}")
        written.append(path)

    for theme in themes:
        manager.set_mode(theme, persist=False)
        Prefs.set_instance(None)
        work = Path(tempfile.mkdtemp(prefix=f"hana_demo_{theme}_"))

        def ready(demo: Demo) -> None:
            printers = demo.services.printers
            _wait(app, lambda: not printers.is_busy and printers.status() is not None)
            demo.shell.resize(1280, 800)
            demo.shell.show()
            demo.shell.navigate(PAGE_PRINT)
            demo.print_page.refresh_now()
            _pump(app)

        # 1. 준비: 문제 없음 (12명 중 2명은 이미 뽑음 → 아직 안 뽑은 10명)
        demo = Demo(work / "clean", fake=True, printed=2)
        ready(demo)
        demo.print_page.show_position(2)
        save(demo.shell, f"print_1_ready_{theme}.png")

        # 3. 인쇄 설정
        sheet = demo.print_page.make_options_sheet()
        sheet.show()
        _pump(app)
        save(sheet, f"print_3_options_{theme}.png")
        sheet.close()

        # 9. 프린터 알약을 누르면 뜨는 창
        demo.shell.show_printer_popover()
        _pump(app)
        save(demo.shell.printer_popover, f"printer_3_popover_{theme}.png")
        demo.shell.printer_popover.hide()

        # 4. 인쇄 중 (10장 중 4장 끝, 다섯 번째를 찍는 중)
        demo.control.hold_on = 5
        demo.print_page.start_printing()
        _wait(app, demo.control.entered.is_set)
        _pump(app, 0.25)
        save(demo.shell, f"print_4_printing_{theme}.png")
        demo.print_page.confirm = lambda *a, **k: True
        demo.print_page.stop_after_current()
        demo.control.release.set()
        _wait(app, lambda: demo.print_page.state() == STATE_DONE)
        _pump(app)
        save(demo.shell, f"print_6_stopped_{theme}.png")
        demo.print_page.back_to_ready()

        # 5. 끝남: 한 장 실패
        demo.control.reset()
        choices = demo.print_page.choices().clone()
        choices.options.stop_on_error = False
        demo.print_page.apply_choices(choices)
        demo.print_page.refresh_now()
        demo.print_page.acknowledge_warnings(True)
        library = registry._fake_library("rtai")
        calls = library._counts.get("R600PrintDraw", 0)
        library.fail("R600PrintDraw", R600_ERR_PRINT, on_call=calls + 2)
        demo.print_page.start_printing()
        _wait(app, lambda: demo.print_page.state() == STATE_DONE)
        _pump(app)
        save(demo.shell, f"print_5_finished_{theme}.png")

        # 6. 기록: 목록과 자세히
        demo.shell.navigate(PAGE_HISTORY)
        demo.history_page.show_latest()
        _pump(app)
        save(demo.shell, f"history_1_detail_{theme}.png")

        # 8. 프린터: 가짜 두 대, 한 대는 점검 결과까지
        demo.shell.navigate(PAGE_PRINTER)
        _wait(app, lambda: not demo.services.printers.is_busy)
        device = demo.services.printers.current()
        demo.printer_page.probe_device(device)
        _wait(app, lambda: not demo.printer_page.is_probing() and not demo.services.printers.is_busy)
        _pump(app)
        save(demo.shell, f"printer_2_found_{theme}.png")

        # 7. 프린터: 못 찾음
        original = registry.discover
        registry.discover = lambda families=registry.ALL_FAMILIES: registry.DiscoveryResult([], [
            "RTAI: 연결된 프린터가 없습니다.", "SMART: 라이브러리(SmartComm2.dll)를 찾을 수 없습니다."])
        try:
            demo.printer_page.refresh_devices()
            _wait(app, lambda: not demo.services.printers.is_busy and not demo.services.printers.devices())
            demo.printer_page.set_detail_visible(True)
            _pump(app)
            save(demo.shell, f"printer_1_none_{theme}.png")
        finally:
            registry.discover = original

        # 10. 설정 (위쪽과 아래쪽)
        demo.shell.navigate(PAGE_SETTINGS)
        _pump(app)
        save(demo.shell, f"settings_1_{theme}.png")
        bar = demo.settings_page.findChild(QScrollArea).verticalScrollBar()
        bar.setValue(bar.maximum())
        _pump(app)
        save(demo.shell, f"settings_2_{theme}.png")
        demo.close()
        _pump(app)

        # 2. 준비: 막는 문제 (사진 없는 사람 2명) + 알아 둘 것
        blocked = Demo(work / "blocked", fake=True, missing_photos=("HN260418", "HN260422"), printed=2)
        ready(blocked)
        blocked.print_page.set_mode("all")
        blocked.print_page.refresh_now()
        _pump(app)
        save(blocked.shell, f"print_2_blocked_{theme}.png")
        blocked.print_page.exclude(blocked.print_page.report().error_record_ids())
        blocked.print_page.refresh_now()
        _pump(app)
        save(blocked.shell, f"print_7_fixed_{theme}.png")

        # 기록: 빈 화면
        empty_history = blocked.history_page
        blocked.shell.navigate(PAGE_HISTORY)
        _pump(app)
        save(blocked.shell, f"history_2_empty_{theme}.png")
        del empty_history
        blocked.close()
        _pump(app)
        registry.use_fakes(True)
        QApplication.processEvents()
    return written


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    folder = Path(tempfile.mkdtemp(prefix="hana_print_demo_"))
    _prepare_environment(args, folder)

    from PySide6.QtWidgets import QApplication

    from ui.theme import ThemeManager, configure_hidpi

    configure_hidpi()
    app = QApplication.instance() or QApplication(sys.argv[:1])
    manager = ThemeManager.instance()
    manager.apply(app)

    if args.shot:
        from design.renderer import ensure_gui

        ensure_gui()
        files = export_screenshots(app, args.shot, (args.theme,) if args.theme else ("light", "dark"))
        print(f"{len(files)}개 저장: {Path(args.shot).resolve()}")
        for path in files:
            print(" ", path.name)
        return 0

    if args.theme:
        manager.set_mode(args.theme, persist=False)
    demo = Demo(folder / "work", fake=args.fake, printed=2)
    if args.fake:
        demo.control.delay = max(0.0, args.slow)
        if args.fail_at > 0:
            from printer import registry
            from printer.fakes import R600_ERR_PRINT

            registry._fake_library("rtai").fail("R600PrintDraw", R600_ERR_PRINT, on_call=args.fail_at)
    app.aboutToQuit.connect(demo.services.shutdown)
    demo.shell.navigate(args.page)
    demo.shell.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
