"""디자인 화면 데모. 라이선스 확인도 프린터도 없이 돈다.

    python scripts/design_page_demo.py                      빈 시작 화면부터
    python scripts/design_page_demo.py --sample             사원증 템플릿 + 작은 예시 명단
    python scripts/design_page_demo.py --theme dark
    python scripts/design_page_demo.py --shot <폴더>         장면별 PNG 저장 (--theme을 주지 않으면 두 테마 모두)
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PEOPLE = [
    ("강민서", "영업1팀", "대리", "HN240102"), ("고은채", "디자인팀", "사원", "HN250311"),
    ("김하늘", "경영지원팀", "사원", "HN260417"), ("남지호", "개발팀", "과장", "HN190705"),
    ("문서윤", "개발팀", "사원", "HN260420"), ("박도현", "영업2팀", "차장", "HN150218"),
    ("서지안", "인사팀", "대리", "HN230904"), ("알렉산드라 이바노브나", "해외사업팀", "선임연구원", "HN221130"),
    ("오세훈", "재무팀", "부장", "HN110301"), ("윤채원", "디자인팀", "대리", "HN240615"),
    ("이준우", "개발팀", "사원", "HN260421"), ("정다은", "경영지원팀", "과장", "HN180822"),
]
_PHOTO_TONES = (("#C9D6E8", "#3F5A8A"), ("#E8DCC9", "#8A5A3F"), ("#D3E8C9", "#3F8A5A"), ("#E3C9E8", "#6A3F8A"))


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="하나스튜디오 디자인 화면 데모")
    parser.add_argument("--theme", choices=("light", "dark"), default=None)
    parser.add_argument("--shot", metavar="DIR", help="장면별 PNG를 저장하고 끝낸다")
    parser.add_argument("--sample", action="store_true", help="템플릿과 예시 명단을 넣고 시작한다")
    parser.add_argument("--size", default="1280x800", help="창 크기 (예: 1600x900)")
    return parser.parse_args(argv)


def _prepare_environment(args: argparse.Namespace) -> None:
    if args.shot:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    if "HANA_STUDIO_SETTINGS_FILE" not in os.environ:
        folder = Path(tempfile.gettempdir()) / "hana_studio_ui"
        folder.mkdir(parents=True, exist_ok=True)
        os.environ["HANA_STUDIO_SETTINGS_FILE"] = str(folder / "design_demo.ini")


def make_photo(path: str, tone: int = 0) -> str:
    from PySide6.QtCore import QPointF, QRectF, Qt
    from PySide6.QtGui import QColor, QImage, QPainter

    background, jacket = _PHOTO_TONES[tone % len(_PHOTO_TONES)]
    image = QImage(360, 450, QImage.Format.Format_RGB32)
    image.fill(QColor(background))
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor("#F1D3BC"))
    painter.drawEllipse(QPointF(180, 170), 78, 90)
    painter.setBrush(QColor("#2B3A55"))
    painter.drawChord(QRectF(96, 70, 168, 150), 0, 180 * 16)
    painter.setBrush(QColor(jacket))
    painter.drawRoundedRect(QRectF(50, 290, 260, 220), 90, 90)
    painter.end()
    image.save(path)
    return path


def make_sample_store(directory: str):
    """작은 예시 명단. 두 사람은 일부러 사진이 없다."""
    from data.store import DataStore

    photos = os.path.join(directory, "photos")
    os.makedirs(photos, exist_ok=True)
    store = DataStore(os.path.join(directory, "sample_list.db"))
    for name in ("이름", "부서", "직급", "사번"):
        store.add_column(name)
    store.add_column("사진", "photo", {"folder": photos})
    rows = []
    for index, (name, team, rank, number) in enumerate(PEOPLE):
        if index not in (4, 9):
            make_photo(os.path.join(photos, f"{number}.png"), index)
        rows.append({"이름": name, "부서": team, "직급": rank, "사번": number, "사진": number})
    store.insert_many(rows)
    return store


class DemoEngine:
    """모델 없이 도는 배경 분리 흉내: 사진의 테두리 색과 다른 곳을 사람으로 본다."""

    def prepare(self, parent=None) -> bool:
        return True

    def default_threshold(self) -> int:
        return 120

    def compute_alpha(self, image_path, progress):
        import numpy as np
        from PIL import Image

        progress("사진에서 사람과 배경을 가려내고 있습니다.")
        rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.int16)
        distance = np.abs(rgb - rgb[2, 2]).sum(axis=2)
        return np.clip(distance * 4, 0, 255).astype(np.uint8)


def build(sample: bool = False, workdir: str | None = None, real_engine: bool = False):
    from ui.pages.design import DesignPage
    from ui.pages.workspace import Workspace
    from ui.shell import PAGE_DESIGN, AppShell

    workdir = workdir or tempfile.mkdtemp(prefix="hsdesign_")
    shell = AppShell(legacy_factory=lambda: __import__("PySide6.QtWidgets").QtWidgets.QWidget())
    shell.set_printer_status("ready", "RTAI R600 준비됨")
    workspace = Workspace(shell)
    page = DesignPage(workspace, background_engine=None if real_engine else DemoEngine())
    shell.register_page(PAGE_DESIGN, page, "디자인", "design", step_label="카드 꾸미기")
    shell.workspace, shell.design_page, shell.demo_dir = workspace, page, workdir
    if sample:
        load_sample(shell)
    return shell


def load_sample(shell, with_list: bool = True) -> None:
    page, workspace = shell.design_page, shell.workspace
    page.start_panel.size_control.set_current("cut_54x86_bleed")
    page.start_with_template("employee")
    workspace.editor.set_card_setup(name="2026 신입 사원증")
    workspace.editor.undo_stack.clear()
    if with_list:
        workspace.set_store(make_sample_store(shell.demo_dir))
        page.preview_bar.step(1)
        page.preview_bar.step(1)
        page.preview_bar.step(1)


def _settle(app, rounds: int = 6) -> None:
    for _ in range(rounds):
        app.processEvents()


def _find(controller, name: str):
    return next(item for item in controller.side().items if item.name == name)


def export_screenshots(app, out_dir: str, themes: tuple[str, ...], size: tuple[int, int]) -> list[Path]:
    import time

    from ui.theme import ThemeManager

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manager = ThemeManager.instance()
    manager.apply(app)
    written: list[Path] = []

    for theme in themes:
        manager.set_mode(theme, persist=False)
        shell = build()
        shell.resize(*size)
        shell.show()
        _settle(app)
        page, workspace = shell.design_page, shell.workspace
        controller = workspace.editor
        page.dialogs.ask_unsaved = lambda *_args: "discard"

        def save(name: str, widget=None) -> None:
            _settle(app)
            path = out / f"{theme}_{name}.png"
            if not (widget or shell).grab().save(str(path)):
                raise RuntimeError(f"그림을 저장하지 못했습니다: {path}")
            written.append(path)

        save("1_start")

        load_sample(shell, with_list=False)
        _settle(app)
        controller.select(_find(controller, "이름").id)
        save("2_text_selected")
        page.canvas_bar._sync_align_menu()
        for name, menu in (("view", page.canvas_bar.view_menu), ("align", page.canvas_bar.align_menu),
                           ("file", page.file_menu)):
            menu.ensurePolished()
            menu.adjustSize()
            save(f"menu_{name}", menu)

        logo = controller.add_image_from_file(make_photo(os.path.join(shell.demo_dir, "logo_photo.png"), 2))
        controller.set_properties(logo.id, {"x": 62.0, "y": 14.0, "w": 20.0, "h": 25.0, "name": "대표 사진"})
        controller.select(logo.id)
        save("3_image_selected")

        page.inspector.image.start_removal()
        deadline = time.time() + 20
        while page.inspector.image.is_busy() and time.time() < deadline:
            page.inspector.image.wait_for_removal(200)
            _settle(app)
        page.inspector.scroll.verticalScrollBar().setValue(page.inspector.scroll.verticalScrollBar().maximum())
        save("3b_background_removed")

        controller.delete([logo.id])
        controller.set_properties(_find(controller, "회사명").id, {"x": 70.0})
        controller.clear_selection()
        save("4_nothing_selected")

        workspace.set_store(make_sample_store(shell.demo_dir))
        for _ in range(3):
            page.preview_bar.step(1)
        controller.select(_find(controller, "부서 직급").id)
        save("5_with_list")

        controller.select(_find(controller, "사번 바코드").id)
        save("6_barcode_selected")

        controller.select([_find(controller, "이름").id, _find(controller, "부서 직급").id])
        save("7_multi_selected")

        rect = controller.add_item("rect")
        page.inspector.shape.stroke_check.setChecked(True)
        page.inspector.detail.box.set_open(True)
        save("8_shape_detail")
        controller.delete([rect.id])
        page.inspector.detail.box.set_open(False)

        workspace.set_current_record(page.preview_bar.record_ids()[7])
        controller.clear_selection()
        save("9_long_name")

        from ui.pages.design.inspector.concat import ConcatDialog

        controller.select(_find(controller, "부서 직급").id)
        text_section = page.inspector.text
        dialog = ConcatDialog(["이름", "부서", "직급", "사번"], text_section.concat_seed(),
                              text_section.preview_text, shell)
        dialog.show()
        save("dialog_concat", dialog)
        dialog.close()

        text = controller.add_item("text")
        text_section.mode_control.set_current("auto")
        save("10_date_text")
        text_section.auto_control.set_current("counter")
        save("11_counter_text")
        controller.delete([text.id])

        shell.close()
        workspace.close()
        shell.deleteLater()
        _settle(app)
    return written


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    _prepare_environment(args)

    from PySide6.QtWidgets import QApplication

    from ui.theme import ThemeManager, configure_hidpi

    configure_hidpi()
    app = QApplication.instance() or QApplication(sys.argv[:1])
    width, height = (int(v) for v in args.size.lower().split("x"))

    if args.shot:
        themes = (args.theme,) if args.theme else ("light", "dark")
        for path in export_screenshots(app, args.shot, themes, (width, height)):
            print(path)
        return 0

    manager = ThemeManager.instance()
    if args.theme:
        manager.set_mode(args.theme, persist=False)
    manager.apply(app)
    shell = build(sample=args.sample, real_engine=True)
    shell.resize(width, height)
    shell.show()
    code = app.exec()
    shell.workspace.close()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
