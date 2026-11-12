"""조립된 앱 전체를 가짜 프린터와 예시 자료로 띄운다. (라이선스 확인 없이 메인 창만 만든다.)

    python scripts/app_demo.py                          빈 상태로 띄운다(가짜 프린터 두 대)
    python scripts/app_demo.py --sample                 사원증 디자인과 12명 명단을 미리 넣어 띄운다
    python scripts/app_demo.py --theme dark             어두운 화면으로
    python scripts/app_demo.py --shot <폴더>             처음부터 끝까지 가는 길을 PNG로 저장(화면 없이)
    python scripts/app_demo.py --shot <폴더> --size 1100x700     가장 작은 창 크기로

예시 명단 파일(cp949 CSV, 머리글은 성명/소속부서/직급/사원번호)과 사진 폴더는 임시 폴더에 만들고,
--shot이 아닐 때는 그 위치를 콘솔에 알려 준다. 가져오기 연습에 쓸 수 있다.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PEOPLE = (
    ("강민서", "영업1팀", "대리", "HN260415"), ("고은채", "디자인팀", "사원", "HN260416"),
    ("김하늘", "경영지원팀", "사원", "HN260417"), ("남지호", "개발팀", "과장", "HN230102"),
    ("문서윤", "개발팀", "사원", "HN260418"), ("박도윤", "영업2팀", "차장", "HN190233"),
    ("서지안", "경영지원팀", "사원", "HN260419"), ("양수아", "디자인팀", "대리", "HN260420"),
    ("이준우", "개발팀", "사원", "HN260421"), ("장하린", "영업2팀", "사원", "HN260422"),
    ("최유나", "개발팀", "사원", "HN260423"), ("한지민", "디자인팀", "사원", "HN260424"),
)
MISSING_PHOTO = ("HN260418", "HN260419")
_real_sleep = time.sleep


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="하나스튜디오 전체 앱 미리 보기(가짜 프린터)")
    parser.add_argument("--theme", choices=("light", "dark"), default=None)
    parser.add_argument("--shot", metavar="DIR", help="가는 길 전체를 PNG로 저장하고 끝낸다")
    parser.add_argument("--size", default="1280x800", help="창 크기 (예: 1100x700)")
    parser.add_argument("--sample", action="store_true", help="디자인과 명단을 미리 넣어 띄운다")
    parser.add_argument("--page", default="design", help="처음 열 화면 id")
    return parser.parse_args(argv)


def _size(text: str) -> tuple[int, int]:
    width, _, height = text.lower().partition("x")
    return max(1100, int(width)), max(700, int(height))


def prepare_environment(folder: Path, offscreen: bool) -> None:
    """사용자 설정·기록·명단 폴더를 임시 폴더로 돌린다(실제 설정을 건드리지 않는다)."""
    if offscreen:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ["LOCALAPPDATA"] = str(folder / "appdata")
    os.environ["HANA_STUDIO_DATA_DIR"] = str(folder / "lists")
    os.environ["HANA_STUDIO_SETTINGS_FILE"] = str(folder / "settings.ini")
    os.environ["HANA_STUDIO_LEGACY_CONFIG"] = str(folder / "no-config.json")


def write_sample_files(folder: Path) -> tuple[Path, Path]:
    """(명단 CSV, 사진 폴더). 두 명은 사진이 없다."""
    from PIL import Image, ImageDraw

    folder.mkdir(parents=True, exist_ok=True)
    photos = folder / "사진"
    photos.mkdir(exist_ok=True)
    lines = ["성명,소속부서,직급,사원번호"]
    for index, (name, team, rank, number) in enumerate(PEOPLE):
        lines.append(f"{name},{team},{rank},{number}")
        if number in MISSING_PHOTO:
            continue
        image = Image.new("RGB", (300, 400), (188 - index * 7, 204 - index * 4, 224 - index * 3))
        draw = ImageDraw.Draw(image)
        draw.ellipse((95, 70, 205, 180), fill=(236, 222, 208))
        draw.rounded_rectangle((55, 205, 245, 420), 60, fill=(52 + index * 9, 70 + index * 5, 104))
        image.save(photos / f"{number}.png")
    csv_path = folder / "신입사원_명단.csv"
    csv_path.write_bytes("\r\n".join(lines).encode("cp949"))
    return csv_path, photos


class Demo:
    """가짜 프린터로 조립한 메인 창 한 벌."""

    def __init__(self, folder: Path) -> None:
        from printer import registry
        from printer.history import JobHistory
        from printer.print_queue import PrintQueueManager
        from services.print_service import PrintService
        from services.printers import PrinterService
        from ui.app import build_main_window
        from ui.pages.services import AppServices

        folder.mkdir(parents=True, exist_ok=True)
        registry.use_fakes(True)
        self.history = JobHistory(str(folder / "history.sqlite"))
        self.queue = PrintQueueManager(history=self.history)
        self.services = AppServices(
            printers=PrinterService(queue=self.queue, settings_path=str(folder / "printer.json")),
            printing=PrintService(queue=self.queue, history=self.history), history=self.history, queue=self.queue)
        self.window = build_main_window(services=self.services, start_services=False)

    def page(self, page_id: str):
        return self.window.page(page_id)

    def find_printers(self, app) -> None:
        printers = self.services.printers
        printers.refresh()
        wait(app, lambda: not printers.is_busy and printers.devices())
        printers.poll_now()
        wait(app, lambda: not printers.is_busy and printers.status() is not None)

    def import_people(self, app, csv_path: Path, photos: Path, shots=None):
        people = self.page("list")
        wizard = people.start_import(csv_path, run=False)
        wizard.wait_idle()
        if shots:
            shots(wizard, "04_가져오기_파일")
        wizard.go_next()
        wizard.set_photo_folder(photos)
        pump(app)
        if shots:
            shots(wizard, "05_가져오기_맞추기")
        wizard.start_import()
        wizard.wait_idle()
        if shots:
            shots(wizard, "06_가져오기_끝")
        wizard.go_next()
        pump(app)

    def close(self, app) -> None:
        self.page("design").dialogs.ask_unsaved = lambda parent, name: "discard"
        self.page("print").confirm = lambda *args, **kwargs: True
        self.window.close()
        self.window.shutdown()
        wait(app, lambda: not self.services.printers.is_busy)
        self.window.deleteLater()
        pump(app)


def pump(app, seconds: float = 0.12) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        app.processEvents()
        _real_sleep(0.004)


def wait(app, predicate, seconds: float = 30.0) -> None:
    deadline = time.monotonic() + seconds
    while not predicate() and time.monotonic() < deadline:
        app.processEvents()
        _real_sleep(0.004)
    app.processEvents()


def export_journey(app, out_dir: str, themes: tuple[str, ...], size: tuple[int, int]) -> list[Path]:
    """처음 쓰는 사람이 가는 길을 따라가며 장면마다 PNG를 남긴다."""
    from ui.pages.printing import STATE_DONE
    from ui.theme import ThemeManager

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    time.sleep = lambda seconds: None           # 가짜 프린터가 기다리는 시간을 없앤다
    tag = f"{size[0]}x{size[1]}"

    for theme in themes:
        ThemeManager.instance().set_mode(theme, persist=False)
        work = Path(tempfile.mkdtemp(prefix="hana_app_demo_"))
        csv_path, photos = write_sample_files(work / "예시")

        def save(widget, name: str) -> None:
            pump(app)
            path = out / f"{name}_{theme}_{tag}.png"
            widget.grab().save(str(path))
            written.append(path)

        # ---- 막다른 길 확인: 디자인도 프린터도 없이 인쇄·명단 화면에 왔을 때
        bare = Demo(work / "bare")
        bare.window.resize(*size)
        bare.window.show()
        pump(app)
        save(bare.window, "01_시작")
        bare.window.navigate("print")
        bare.page("print").refresh_now()
        save(bare.window, "20_인쇄_디자인과_프린터_없음")
        bare.window.navigate("list")
        save(bare.window, "02_명단_비어있음")
        bare.window.navigate("history")
        save(bare.window, "21_기록_비어있음")
        bare.window.navigate("printer")
        wait(app, lambda: not bare.services.printers.is_busy)
        save(bare.window, "22_프린터_찾기전")
        bare.close(app)

        # ---- 가는 길
        demo = Demo(work / "main")
        window = demo.window
        window.resize(*size)
        window.show()
        demo.find_printers(app)
        design = demo.page("design")
        design.start_panel.choose("employee")
        pump(app)
        save(window, "03_디자인_사원증")
        editor = window.workspace.editor
        bound = [item for _side, item in editor.document.iter_items() if item.source is not None
                 and item.kind == "text"]
        if bound:
            editor.select([bound[-1].id])
            save(window, "03b_디자인_명단칸_고름")
            editor.clear_selection()

        window.navigate("list")
        demo.import_people(app, csv_path, photos, shots=save)
        save(window, "07_명단_가져온뒤")

        window.navigate("design")
        pump(app, 0.5)
        save(window, "08_디자인_명단으로_미리보기")

        window.navigate("print")
        page = demo.page("print")
        page.refresh_now()
        save(window, "09_인쇄_사진없는사람")
        page.exclude(page.report().error_record_ids())
        page.refresh_now()
        page.acknowledge_warnings(True)
        save(window, "10_인쇄_준비됨")
        page.start_printing()
        wait(app, lambda: page.state() == STATE_DONE)
        save(window, "11_인쇄_끝")

        window.navigate("history")
        demo.page("history").show_job(page.job().history_id)
        save(window, "12_기록")

        window.navigate("list")
        save(window, "13_명단_인쇄한뒤")
        missing = [r["_id"] for r in window.workspace.store.rows() if not r["_printed"]]
        demo.page("list").show_people(missing)
        save(window, "14_명단_확인할사람만")
        demo.page("list").show_everyone()

        window.navigate("print")
        page.back_to_ready()
        page.refresh_now()
        save(window, "15_인쇄_다시왔을때")

        window.navigate("printer")
        wait(app, lambda: not demo.services.printers.is_busy)
        save(window, "16_프린터")
        window.navigate("settings")
        save(window, "17_설정")
        demo.close(app)
    return written


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    folder = Path(tempfile.mkdtemp(prefix="hana_app_demo_"))
    prepare_environment(folder, offscreen=bool(args.shot))

    from PySide6.QtWidgets import QApplication

    from ui.theme import ThemeManager, configure_hidpi
    from utils.console import ensure_utf8_console

    ensure_utf8_console()
    configure_hidpi()
    app = QApplication.instance() or QApplication(sys.argv[:1])
    manager = ThemeManager.instance()
    manager.apply(app)
    size = _size(args.size)

    if args.shot:
        from design.renderer import ensure_gui

        ensure_gui()
        files = export_journey(app, args.shot, (args.theme,) if args.theme else ("light", "dark"), size)
        print(f"{len(files)}개 저장: {Path(args.shot).resolve()}")
        return 0

    if args.theme:
        manager.set_mode(args.theme, persist=False)
    csv_path, photos = write_sample_files(folder / "예시")
    demo = Demo(folder / "work")
    demo.window.resize(*size)
    demo.window.show()
    demo.services.start()
    if args.sample:
        demo.page("design").start_with_template("employee")
        demo.window.navigate("list")
        demo.import_people(app, csv_path, photos)
    print(f"예시 명단: {csv_path}\n예시 사진 폴더: {photos}")
    demo.window.navigate(args.page)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
