"""명단 화면 데모.

    python scripts/people_page_demo.py                    빈 화면에서 시작
    python scripts/people_page_demo.py --sample           128명 예시 명단(사진 4명 빠짐)을 넣은 채로 시작
    python scripts/people_page_demo.py --theme dark
    python scripts/people_page_demo.py --shot <폴더>       화면 없이 PNG 저장 (--theme을 주면 그 테마만)
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SURNAMES = "김이박최정강조윤장임한오서신권황안송류전홍고문양손배백허유남"
GIVEN = ("민서", "은채", "하늘", "지호", "서윤", "도윤", "지안", "수아", "준우", "예린", "시우", "하준", "지유", "서연",
         "주원", "유진", "태윤", "소율", "건우", "채원", "현우", "다은", "선우", "가은", "연우", "나윤", "승현", "수빈",
         "정민", "보람", "재희", "미래")
TEAMS = ("경영지원팀", "영업1팀", "영업2팀", "개발팀", "디자인팀", "생산관리팀", "품질관리팀", "고객지원팀")
RANKS = ("사원", "사원", "사원", "대리", "대리", "과장", "차장", "부장")
SKIN = ("#F1D3BC", "#E8C3A6", "#F6DEC9", "#DDB392")
HAIR = ("#2B2A33", "#3A2A20", "#1F2A3A", "#4A3B2E", "#2E2E2E")
SHIRT = ("#3F5A8A", "#5A6B7A", "#7A4E5A", "#3E6B5C", "#6A5A8A", "#8A6A3E")
BACK = ("#C9D6E8", "#DCE5D8", "#E8DCCB", "#D8D3E6", "#D3E3E6")
MISSING_ROWS = (4, 37, 71, 102)


def sample_people(count: int = 128, seed: int = 26) -> list[dict]:
    rng = random.Random(seed)
    people, used = [], set()
    while len(people) < count:
        name = rng.choice(SURNAMES) + rng.choice(GIVEN)
        if name in used:
            continue
        used.add(name)
        index = len(people)
        year = rng.choice((26, 26, 26, 26, 25, 24, 23, 19, 15))
        people.append({
            "성명": name,
            "소속부서": rng.choice(TEAMS),
            "직급": rng.choice(RANKS) if year < 26 else "사원",
            "사원번호": f"HN{year:02d}{415 + index:04d}",
            "입사일": f"20{year:02d}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
        })
    people.sort(key=lambda p: p["성명"])
    return people


def make_photo(path: Path, seed: int) -> None:
    from PySide6.QtCore import QPointF, QRectF, Qt
    from PySide6.QtGui import QColor, QImage, QPainter

    rng = random.Random(seed)
    image = QImage(360, 450, QImage.Format.Format_RGB32)
    image.fill(QColor(rng.choice(BACK)))
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(rng.choice(SHIRT)))
    painter.drawRoundedRect(QRectF(46, 300, 268, 220), 96, 96)
    painter.setBrush(QColor(rng.choice(SKIN)))
    painter.drawRoundedRect(QRectF(154, 250, 52, 70), 20, 20)
    painter.drawEllipse(QPointF(180, 178), 76 + rng.randint(-4, 4), 90 + rng.randint(-4, 6))
    painter.setBrush(QColor(rng.choice(HAIR)))
    if rng.random() < 0.5:
        painter.drawChord(QRectF(98, 76, 164, 150), 0, 180 * 16)
    else:
        painter.drawChord(QRectF(92, 72, 176, 170), -10 * 16, 200 * 16)
        painter.drawRoundedRect(QRectF(92, 150, 30, 150), 14, 14)
        painter.drawRoundedRect(QRectF(238, 150, 30, 150), 14, 14)
    painter.end()
    image.save(str(path), "JPG", 88)


def make_sample(folder: str | os.PathLike, count: int = 128) -> dict:
    """예시 파일을 만든다: 신입사원_명단.xlsx(시트 2개), 같은 내용의 CP949 CSV, 사진 폴더(4명 빠짐)."""
    from openpyxl import Workbook

    root = Path(folder)
    photos = root / "사진"
    photos.mkdir(parents=True, exist_ok=True)
    people = sample_people(count)
    headers = list(people[0])

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "신입사원"
    sheet.append(headers)
    for person in people:
        sheet.append([person[h] for h in headers])
    other = workbook.create_sheet("퇴사자")
    other.append(headers)
    for person in sample_people(6, seed=7):
        other.append([person[h] for h in headers])
    xlsx = root / "신입사원_명단.xlsx"
    workbook.save(xlsx)

    csv_path = root / "신입사원_명단.csv"
    lines = [",".join(headers)] + [",".join(person[h] for h in headers) for person in people]
    csv_path.write_bytes("\r\n".join(lines).encode("cp949"))

    missing = []
    for index, person in enumerate(people):
        if index in MISSING_ROWS:
            missing.append(person["성명"])
            continue
        make_photo(photos / f"{person['사원번호']}.jpg", index)
    return {"folder": str(root), "xlsx": str(xlsx), "csv": str(csv_path), "photos": str(photos),
            "people": people, "missing": missing}


def _prepare_environment(shot: bool, data_dir: str) -> None:
    if shot:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("HANA_STUDIO_DATA_DIR", data_dir)
    if "HANA_STUDIO_SETTINGS_FILE" not in os.environ:
        os.environ["HANA_STUDIO_SETTINGS_FILE"] = str(Path(data_dir) / "demo.ini")


def build(sample: dict | None = None, with_list: bool = False):
    """(틀, 명단 화면, workspace). with_list면 예시 명단을 미리 넣는다."""
    from design import presets
    from ui.pages.people import PeoplePage
    from ui.pages.workspace import Workspace
    from ui.shell import PAGE_LIST, AppShell

    shell = AppShell(legacy_factory=lambda: None)
    workspace = Workspace(shell)
    workspace.editor.new_document(presets.CUT_54X86_BLEED, "employee", name="2026 신입 사원증")
    shell.set_document("2026 신입 사원증", True)
    shell.set_printer_status("ready", "RTAI R600 준비됨")
    page = PeoplePage(workspace)
    shell.register_page(PAGE_LIST, page, "명단", "people", step_label="명단 넣기")
    shell.navigate(PAGE_LIST)
    if sample is not None and with_list:
        import_sample(page, sample)
    return shell, page, workspace


def import_sample(page, sample: dict, printed: int = 87) -> None:
    wizard = page.start_import(sample["xlsx"], run=False)
    wizard.wait_idle()
    wizard.go_next()
    wizard.set_photo_folder(sample["photos"])
    wizard.start_import()
    wizard.wait_idle()
    wizard.accept()
    store = page.store
    ids = store.ids()
    rng = random.Random(3)
    store.mark_printed(rng.sample(ids, min(printed, len(ids))))
    page.workspace.notify_records_changed()
    page.clear_checked()


def _settle(app, rounds: int = 4) -> None:
    for _ in range(rounds):
        app.processEvents()


def export_screenshots(app, out_dir: str, themes: tuple[str, ...] = ("light", "dark")) -> list[Path]:
    from PySide6.QtWidgets import QApplication

    from data.filters import Condition, Group
    from data.store import PhotoAdjustment
    from ui.pages.people.dialogs import ConditionDialog
    from ui.pages.people.thumbs import ThumbLoader
    from ui.theme import ThemeManager

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manager = ThemeManager.instance()
    written: list[Path] = []

    def save(widget, name: str) -> None:
        _settle(app)
        widget.grab()
        for loader in widget.findChildren(ThumbLoader):
            loader.flush()
        _settle(app)
        path = out / name
        if not widget.grab().save(str(path)):
            raise RuntimeError(f"그림을 저장하지 못했습니다: {path}")
        written.append(path)

    for theme in themes:
        manager.set_mode(theme, persist=False)
        work = tempfile.mkdtemp(prefix="hs_people_shot_")
        os.environ["HANA_STUDIO_DATA_DIR"] = str(Path(work) / "lists")
        sample = make_sample(work)
        shell, page, workspace = build()
        shell.resize(1280, 800)
        shell.show()
        _settle(app)
        save(shell, f"{theme}_01_empty.png")

        wizard = page.start_import(sample["xlsx"], run=False)
        wizard.show()
        wizard.wait_idle()
        save(wizard, f"{theme}_02_wizard_file.png")
        wizard.detail_toggle.setChecked(True)
        save(wizard, f"{theme}_03_wizard_file_details.png")
        wizard.detail_toggle.setChecked(False)
        wizard.go_next()
        save(wizard, f"{theme}_04_wizard_match.png")
        wizard.set_photo_folder(sample["photos"])
        wizard._render_sample()
        save(wizard, f"{theme}_05_wizard_match_photos.png")
        wizard.start_import()
        wizard.wait_idle()
        save(wizard, f"{theme}_06_wizard_done.png")
        wizard.accept()
        _settle(app)
        store = page.store
        ids = store.ids()
        store.mark_printed(random.Random(3).sample(ids, 87))
        workspace.notify_records_changed()
        save(shell, f"{theme}_07_table_after_import.png")

        page.clear_checked()
        page.set_chip("unprinted")
        page.model.set_checked(page.model.ids()[:12], True)
        page.table.go_to_row(2)
        save(shell, f"{theme}_08_table_filter_ticked.png")

        page.toggle_all_in_view()
        save(shell, f"{theme}_09_table_select_all.png")

        page.clear_checked()
        page.set_chip("missing")
        save(shell, f"{theme}_10_table_missing_photo.png")

        page.set_chip("all")
        page.model.sort(page.model.column_of("부서"), __import__("PySide6.QtCore").QtCore.Qt.AscendingOrder)
        page.table.go_to_row(1)
        dialog = page.edit_person_at(1, run=False)
        dialog.show()
        dialog.refresh_preview()
        save(dialog, f"{theme}_11_person_form.png")
        dialog.reject()

        dialog = page.add_person(run=False)
        dialog.show()
        save(dialog, f"{theme}_12_person_form_new.png")
        dialog.reject()

        page.model.set_order(None)
        page.table.go_to_row(0)
        page.preview.adjust_toggle.setChecked(True)
        page.preview.set_adjustment(PhotoAdjustment(1.6, 0.0, 0.12))
        save(shell, f"{theme}_13_photo_adjust.png")
        page.preview.reset_adjustment()
        page.preview.adjust_toggle.setChecked(False)

        condition = ConditionDialog(store, Group("and", [Condition("부서", "contains", "영업"),
                                                         Condition("입사일", "ge", "2026-01-01")]), page)
        condition.show()
        save(condition, f"{theme}_14_condition.png")
        condition.reject()

        page.set_search("없는사람")
        save(shell, f"{theme}_15_no_match.png")
        page.set_search("")

        again = page.start_import(sample["csv"], run=False)
        again.show()
        again.wait_idle()
        again.go_next()
        again.skip_check.setChecked(True)
        save(again, f"{theme}_16_wizard_reimport.png")
        again.reject()

        _settle(app)
        shell.close()
        workspace.close()
        shell.deleteLater()
        _settle(app)
        QApplication.sendPostedEvents(None, 52)
        shutil.rmtree(work, ignore_errors=True)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="하나스튜디오 명단 화면 데모")
    parser.add_argument("--theme", choices=("light", "dark"), default=None)
    parser.add_argument("--shot", metavar="DIR", help="PNG로 저장하고 끝낸다")
    parser.add_argument("--sample", action="store_true", help="128명 예시 명단을 만들어 넣는다")
    args = parser.parse_args(argv)

    work = tempfile.mkdtemp(prefix="hs_people_demo_")
    _prepare_environment(bool(args.shot), str(Path(work) / "lists"))

    from PySide6.QtWidgets import QApplication

    from ui.theme import ThemeManager, configure_hidpi

    configure_hidpi()
    app = QApplication.instance() or QApplication(sys.argv[:1])
    manager = ThemeManager.instance()
    manager.apply(app)

    if args.shot:
        themes = (args.theme,) if args.theme else ("light", "dark")
        files = export_screenshots(app, args.shot, themes)
        print(f"{len(files)}개 저장: {Path(args.shot).resolve()}")
        return 0

    if args.theme:
        manager.set_mode(args.theme, persist=False)
    sample = make_sample(work) if args.sample else None
    shell, page, workspace = build(sample, with_list=args.sample)
    if sample is not None:
        print(f"예시 파일: {sample['folder']}")
    shell.show()
    code = app.exec()
    workspace.close()
    return code


if __name__ == "__main__":
    sys.exit(main())
