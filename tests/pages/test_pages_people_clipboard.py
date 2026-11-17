"""명단 화면의 Ctrl+C / Ctrl+V (다눌과 같은 엑셀 붙여 넣기)."""

from people_helpers import *  # noqa: F401,F403
from people_helpers import PEOPLE, pump

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from data.clipboard import parse_tsv


def names(page):
    return [page.model.record_at(row)["이름"] for row in range(page.model.rowCount())]


def test_ctrl_v_overwrites_from_current_cell_and_adds_new_people(page, qapp):
    page.table.go_to_row(len(PEOPLE) - 1, page.model.column_of("이름"))
    page.table.setFocus()
    QApplication.clipboard().setText("양수아2\t디자인1팀\r\n새사람\t총무팀\r\n")
    QTest.keyClick(page.table, Qt.Key_V, Qt.ControlModifier)
    pump(qapp)
    rows = names(page)
    assert len(rows) == len(PEOPLE) + 1
    assert "양수아2" in rows and "새사람" in rows and "양수아" not in rows
    new = next(r for r in page.store.rows() if r["이름"] == "새사람")
    assert new["부서"] == "총무팀"
    assert page.last_toast() is not None


def test_paste_can_be_undone(page, qapp):
    page.table.go_to_row(0, page.model.column_of("이름"))
    before = sorted(r["이름"] for r in page.store.rows())
    result = page.paste_people("가\n나\n다\n")
    assert result is not None and len(result.updated) == 3
    page.undo_paste(result)
    pump(qapp)
    assert sorted(r["이름"] for r in page.store.rows()) == before


def test_append_paste_skips_excel_header(page, qapp):
    result = page.paste_people("이름\t부서\n홍길동\t인사팀\n", append=True)
    assert result.skipped_header and len(result.inserted) == 1
    assert any(r["이름"] == "홍길동" and r["부서"] == "인사팀" for r in page.store.rows())


def test_ctrl_c_copies_selected_people_as_tab_separated_text(page, qapp):
    page.table.go_to_row(0, page.model.column_of("이름"))
    page.table.setFocus()
    QTest.keyClick(page.table, Qt.Key_C, Qt.ControlModifier)
    rows = parse_tsv(QApplication.clipboard().text())
    first = page.model.record_at(0)
    assert rows[0][page._field_names().index("이름")] == first["이름"]


def test_empty_clipboard_warns(page):
    assert page.paste_people("") is None


def test_camera_photo_is_saved_in_photo_folder_and_linked(page, qapp, monkeypatch, photo_folder):
    from PySide6.QtGui import QColor, QImage

    from ui import camera

    def fake_take(parent, aspect=None):
        image = QImage(300, 400, QImage.Format.Format_RGB32)
        image.fill(QColor("#a05030"))
        return image

    monkeypatch.setattr(camera, "take_photo", fake_take)
    row = 0
    record = page.model.record_at(row)
    path = page.take_photo_for(row, "사진")
    assert path is not None
    from pathlib import Path
    saved = Path(path)
    assert saved.parent.resolve() == Path(photo_folder).resolve()
    assert saved.name.startswith(record["사번"]) and saved.name != f"{record['사번']}.png"
    fresh = page.store.get(record["_id"])
    assert fresh["사진"] == saved.name
