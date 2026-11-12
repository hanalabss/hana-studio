import csv
from pathlib import Path

from people_helpers import *  # noqa: F401,F403
from people_helpers import PEOPLE, pump

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QLineEdit

from data.filters import Condition, Group
from ui.pages.people import PeoplePage, storage
from ui.shell import PAGE_DESIGN, PAGE_LIST, PAGE_PRINT, AppShell


class Notices:
    """workspace 신호가 몇 번 났는지 센다."""

    def __init__(self, workspace):
        self.columns = 0
        self.records = 0
        workspace.columns_changed.connect(self._columns)
        workspace.records_changed.connect(self._records)

    def _columns(self):
        self.columns += 1

    def _records(self):
        self.records += 1


def names(page):
    return [page.model.record_at(row)["이름"] for row in range(page.model.rowCount())]


def id_of(page, name):
    return next(r["_id"] for r in page.store.rows() if r["이름"] == name)


# ---------------------------------------------------------------- 빈 화면

def test_empty_state_invites_import(qapp, workspace):
    page = PeoplePage(workspace)
    page.show()
    pump(qapp)
    assert page.empty_view.isVisible()
    assert page.empty_view.heading.text() == "엑셀이나 CSV 파일을 가져오세요"
    assert page.empty_view.import_button.text() == "파일 가져오기"
    assert not page.empty_view.recent_box.isVisible()
    assert "lists" in page.empty_view.details.text()
    forward = page.footer_actions()[-1]
    assert forward.text == "다음: 인쇄하기" and not forward.enabled and forward.tooltip
    assert page.footer_actions()[0].text == "이전: 카드 꾸미기"


def test_manual_start_creates_columns_the_design_needs(qapp, workspace, tmp_path):
    page = PeoplePage(workspace)
    page.start_manual(open_form=False)
    store = workspace.store
    assert store is not None
    assert {c.name: c.type for c in store.columns()} == {
        "사진": "photo", "이름": "text", "부서": "text", "직급": "text", "사번": "text"}
    assert Path(store.path).parent == tmp_path / "lists"
    assert Path(store.path).suffix == ".hsdata"
    assert not page.empty_view.isVisibleTo(page)
    assert page.blank_title.text() == "아직 아무도 없습니다"


def test_manual_start_without_design_fields_uses_name_and_team(qapp, tmp_path):
    from ui.pages.workspace import Workspace

    ws = Workspace()
    page = PeoplePage(ws)
    page.start_manual(open_form=False)
    assert ws.store.column_names() == ["이름", "부서"]
    ws.close()


def test_list_file_goes_next_to_a_saved_design(qapp, workspace, tmp_path):
    design = tmp_path / "designs" / "사원증.hsd"
    design.parent.mkdir()
    workspace.editor.save(design)
    page = PeoplePage(workspace)
    page.start_manual(open_form=False)
    assert Path(workspace.store.path) == design.with_suffix(".hsdata")
    assert storage.new_list_path(design) == design.parent / "사원증 (2).hsdata"


def test_recent_lists_are_offered_and_open(qapp, workspace, people_store, tmp_path):
    path = people_store.path
    people_store.close()
    storage.remember_list(path)
    storage.remember_list(tmp_path / "없어진 명단.hsdata")
    page = PeoplePage(workspace)
    page.show()
    pump(qapp)
    assert [Path(r.path).name for r in storage.recent_lists()] == [Path(path).name]
    assert len(page.empty_view.recent_buttons) == 1
    page.empty_view.recent_buttons[0].click()
    assert workspace.store is not None and workspace.store.count() == len(PEOPLE)
    assert page.model.rowCount() == len(PEOPLE)


def test_dropping_a_file_is_accepted_only_for_list_files(qapp, workspace, tmp_path):
    from PySide6.QtCore import QMimeData, QPoint, QUrl
    from PySide6.QtGui import QDragEnterEvent

    page = PeoplePage(workspace)

    def enter(file_name):
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(str(tmp_path / file_name))])
        event = QDragEnterEvent(QPoint(10, 10), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
        page.dragEnterEvent(event)
        return event.isAccepted()

    assert enter("명단.xlsx") and enter("명단.csv")
    assert not enter("사진.png")


# ---------------------------------------------------------------- 찾기와 걸러 보기

def test_chips_show_live_counts(page):
    assert {key: chip.count() for key, chip in page.chips.items()} == {
        "all": 8, "unprinted": 6, "printed": 2, "missing": 2}
    assert page.stats_label.text() == "8명"


def test_each_chip_filters(page, workspace):
    page.set_chip("unprinted")
    assert len(names(page)) == 6 and "남지호" not in names(page)
    assert page.stats_label.text() == "8명 중 6명 보는 중"
    page.set_chip("printed")
    assert names(page) == ["남지호", "박도윤"]
    page.set_chip("missing")
    assert names(page) == ["문서윤", "서지안"]
    assert workspace.store.ids(workspace.view_filter) == [id_of(page, "문서윤"), id_of(page, "서지안")]
    page.set_chip("all")
    assert len(names(page)) == 8 and workspace.view_filter is None


def test_clicking_a_chip_filters(page):
    page.chips["printed"].click()
    assert page.chip() == "printed"
    assert page.chips["printed"].isChecked() and not page.chips["all"].isChecked()
    assert names(page) == ["남지호", "박도윤"]


def test_quick_search(page, workspace, qapp):
    page.set_search("개발팀")
    assert names(page) == ["남지호", "문서윤"]
    assert page.stats_label.text() == "8명 중 2명 보는 중"
    assert workspace.store.count(workspace.view_filter) == 2
    page.set_chip("unprinted")                       # 찾기와 칩은 함께 걸린다
    assert names(page) == ["문서윤"]
    page.set_search("")
    assert len(names(page)) == 6


def test_typing_in_the_search_field_filters(page, qapp):
    QTest.keyClicks(page.search_field, "HN26041")        # 사번으로도 찾는다
    QTest.keyClick(page.search_field, Qt.Key_Return)
    assert names(page) == ["강민서", "고은채", "김하늘", "문서윤", "서지안"]
    QTest.keyClick(page.search_field, Qt.Key_Escape)
    assert page.search_field.text() == "" and len(names(page)) == 8


def test_no_match_shows_guidance(page):
    page.set_search("없는사람")
    assert page.model.rowCount() == 0
    assert page.blank.isVisibleTo(page)
    assert page.blank_title.text() == "맞는 사람이 없습니다"
    page.blank_button.click()
    assert page.model.rowCount() == 8 and not page.blank.isVisibleTo(page)


def test_condition_builder_produces_filter(page, workspace):
    from ui.pages.people.dialogs import ConditionDialog

    dialog = ConditionDialog(workspace.store, None, page)
    row = dialog.rows()[0]
    row.column_combo.setCurrentText("부서")
    assert row.op_combo.currentData() == "contains"
    assert "들어 있다" in [row.op_combo.itemText(i) for i in range(row.op_combo.count())]
    row.value_edit.setText("영업")
    assert dialog.count_label.text() == "2명이 맞습니다."
    second = dialog.add_row()
    second.column_combo.setCurrentText("입사일")
    second.op_combo.setCurrentIndex(second.op_combo.findData("ge"))
    assert second.op_combo.currentText() == "부터"
    second.value_edit.setText("2026-01-01")
    assert dialog.count_label.text() == "1명이 맞습니다."
    dialog.mode_control.set_current("or")
    assert dialog.count_label.text() == "7명이 맞습니다."

    group = dialog.result_filter()
    assert group == Group("or", [Condition("부서", "contains", "영업"), Condition("입사일", "ge", "2026-01-01")])
    page.set_condition(group)
    assert len(names(page)) == 7 and "남지호" not in names(page)
    assert page.condition_button.text() == "조건 2개로 찾는 중"
    assert page.condition_clear.isVisibleTo(page)
    assert workspace.store.count(workspace.view_filter) == 7
    page.condition_clear.click()
    assert page.condition() is None and len(names(page)) == 8


def test_condition_builder_handles_between_empty_and_bad_values(page, workspace):
    from ui.pages.people.dialogs import ConditionDialog

    dialog = ConditionDialog(workspace.store, Group("and", [Condition("입사일", "between", ("2026-03-01", "2026-03-05"))]), page)
    row = dialog.rows()[0]
    assert row.value_edit2.isVisibleTo(dialog) and row.value_edit2.text() == "2026-03-05"
    assert dialog.count_label.text() == "3명이 맞습니다."
    row.op_combo.setCurrentIndex(row.op_combo.findData("empty"))
    assert not row.value_edit.isVisibleTo(dialog)
    assert dialog.result_filter() == Group("and", [Condition("입사일", "empty")])
    row.op_combo.setCurrentIndex(row.op_combo.findData("gt"))
    row.value_edit.setText("작년")                       # 날짜가 아닌 값
    assert not dialog.apply_button.isEnabled()
    assert "날짜" in dialog.count_label.text()
    row.value_edit.setText("")
    assert dialog.result_filter() is None and dialog.apply_button.isEnabled()


# ---------------------------------------------------------------- 체크와 workspace

def test_checked_ids_and_view_state_stay_in_sync(page, workspace, qapp):
    page.model.set_checked([id_of(page, "양수아"), id_of(page, "강민서")])
    pump(qapp)
    assert workspace.checked_ids == [id_of(page, "강민서"), id_of(page, "양수아")]
    assert page.stats_label.text() == "8명 · 2명 선택"

    page.model.sort(page.model.column_of("이름"), Qt.DescendingOrder)
    pump(qapp)
    assert workspace.view_order == [("이름", "desc")]
    assert workspace.checked_ids == [id_of(page, "양수아"), id_of(page, "강민서")]

    page.set_chip("printed")
    assert workspace.view_filter is page.model.view_filter is not None
    assert len(workspace.checked_ids) == 2             # 안 보이게 걸러도 체크는 남는다
    assert page.stats_label.text() == "8명 중 2명 보는 중 · 2명 선택"

    page.clear_checked()
    assert workspace.checked_ids == []


def test_select_all_in_filter_shows_banner_and_can_be_cleared(page, workspace, qapp):
    page.set_chip("unprinted")
    page.table.header().checkToggled.emit()
    pump(qapp)
    assert len(workspace.checked_ids) == 6
    assert page.select_banner.isVisibleTo(page)
    assert page.select_banner.text() == "현재 조건의 6명을 모두 선택했습니다."
    assert page.table.header().check_state() == Qt.Checked

    page.set_chip("all")
    assert page.table.header().check_state() == Qt.PartiallyChecked
    assert not page.select_banner.isVisibleTo(page)

    page.toggle_all_in_view()
    assert len(workspace.checked_ids) == 8
    assert page.select_banner.text().startswith("8명을 모두 선택했습니다.")
    page.select_banner.action_button().click()
    assert workspace.checked_ids == [] and not page.select_banner.isVisibleTo(page)


def test_footer_reflects_selection(page, workspace):
    forward = page.footer_actions()[-1]
    assert forward.text == "다음: 인쇄하기" and not forward.enabled
    assert "체크" in forward.tooltip and "체크" in page.footer_hint()
    page.model.set_checked([id_of(page, "강민서"), id_of(page, "문서윤"), id_of(page, "서지안")])
    forward = page.footer_actions()[-1]
    assert forward.text == "다음: 3장 인쇄하기" and forward.enabled
    assert page.footer_hint() == "3명을 골랐습니다. 그중 2명은 사진이 없어 사진 칸이 비어 나옵니다."


def test_footer_buttons_navigate_in_the_shell(qapp, workspace, people_store, tmp_path):
    from PySide6.QtCore import QSettings

    shell = AppShell(QSettings(str(tmp_path / "shell.ini"), QSettings.IniFormat), legacy_factory=lambda: None)
    page = PeoplePage(workspace)
    shell.register_page(PAGE_LIST, page, "명단", "people", step_label="명단 넣기")
    shell.navigate(PAGE_LIST)
    workspace.set_store(people_store)
    shell.show()
    pump(qapp)
    assert shell.footer_buttons()[-1].text() == "다음: 인쇄하기" and not shell.footer_buttons()[-1].isEnabled()
    page.model.set_checked(people_store.ids()[:3])
    pump(qapp)
    assert shell.footer_buttons()[-1].text() == "다음: 3장 인쇄하기"
    shell.footer_buttons()[-1].click()
    assert shell.current_page_id() == PAGE_PRINT
    shell.navigate(PAGE_LIST)
    shell.footer_buttons()[0].click()
    assert shell.current_page_id() == PAGE_DESIGN
    shell.close()


def test_clicking_a_row_sets_the_current_record(page, workspace, qapp):
    assert workspace.current_record_id == id_of(page, "강민서")      # 처음에는 첫 사람
    rect = page.table.visualRect(page.model.index(2, page.model.column_of("이름")))
    QTest.mouseClick(page.table.viewport(), Qt.LeftButton, Qt.NoModifier, rect.center())
    assert workspace.current_record_id == id_of(page, "김하늘")
    assert page.preview.record()["이름"] == "김하늘"
    assert page.preview.position_label.text() == "3 / 8"
    page.preview.next_button.click()
    assert workspace.current_record_id == id_of(page, "남지호")
    page.preview.prev_button.click()
    page.preview.prev_button.click()
    assert workspace.current_record_id == id_of(page, "고은채")


def test_clicking_the_check_cell_ticks_without_changing_the_row(page, workspace, qapp):
    rect = page.table.visualRect(page.model.index(3, 0))
    QTest.mouseClick(page.table.viewport(), Qt.LeftButton, Qt.NoModifier, rect.center())
    pump(qapp)
    assert workspace.checked_ids == [id_of(page, "남지호")]
    assert workspace.current_record_id == id_of(page, "강민서")
    rect = page.table.visualRect(page.model.index(6, 0))
    QTest.mouseClick(page.table.viewport(), Qt.LeftButton, Qt.ShiftModifier, rect.center())
    pump(qapp)
    assert len(workspace.checked_ids) == 4              # Shift로 사이 줄까지


# ---------------------------------------------------------------- 고치기, 넣기, 지우기

def test_keyboard_enter_edits_space_ticks(page, workspace, qapp):
    table = page.table
    table.setFocus()
    table.go_to_row(1, page.model.column_of("부서"))
    QTest.keyClick(table, Qt.Key_Return)
    editor = table.findChild(QLineEdit)
    assert editor is not None and editor.text() == "디자인팀"
    editor.setText("브랜드팀")
    QTest.keyClick(editor, Qt.Key_Return)
    pump(qapp)
    assert workspace.store.get(id_of(page, "고은채"))["부서"] == "브랜드팀"

    QTest.keyClick(table, Qt.Key_Down)
    assert workspace.current_record_id == id_of(page, "김하늘")
    QTest.keyClick(table, Qt.Key_Space)
    pump(qapp)
    assert workspace.checked_ids == [id_of(page, "김하늘")]
    QTest.keyClick(table, Qt.Key_Space)
    pump(qapp)
    assert workspace.checked_ids == []


def test_ctrl_f_focuses_search(page, qapp):
    page.activateWindow()
    page.table.setFocus()
    pump(qapp)
    QTest.keyClick(page.table, Qt.Key_F, Qt.ControlModifier)
    pump(qapp)
    assert page.search_field.hasFocus()


def test_type_aware_editors(page, workspace, qapp):
    from PySide6.QtWidgets import QDateEdit

    from ui.widgets import ComboBox

    workspace.store.set_column_type("직급", "list", {"choices": ["사원", "대리", "과장"]})
    workspace.notify_columns_changed()
    table = page.table

    table.go_to_row(0, page.model.column_of("입사일"))
    table.edit_current()
    date_editor = table.findChild(QDateEdit)
    assert date_editor is not None and date_editor.date().toString("yyyy-MM-dd") == "2026-03-02"
    date_editor.setDate(date_editor.date().addDays(1))
    table.commitData(date_editor)
    table.closeEditor(date_editor, table.itemDelegate().EndEditHint.NoHint)
    assert workspace.store.get(id_of(page, "강민서"))["입사일"] == "2026-03-03"

    table.go_to_row(0, page.model.column_of("직급"))
    table.edit_current()
    combo = table.findChild(ComboBox)
    assert [combo.itemText(i) for i in range(combo.count())] == ["", "사원", "대리", "과장"]
    assert combo.currentText() == "대리"
    combo.setCurrentText("과장")
    table.commitData(combo)
    table.closeEditor(combo, table.itemDelegate().EndEditHint.NoHint)
    assert workspace.store.get(id_of(page, "강민서"))["직급"] == "과장"


def test_photo_cell_takes_a_picked_file(page, workspace, photo_folder, tmp_path):
    from people_helpers import write_png

    picked = []
    page.table.photoPickRequested.connect(picked.append)
    row = names(page).index("서지안")
    index = page.model.index(row, page.model.column_of("사진"))
    page.table.setCurrentIndex(index)
    page.table.photoPickRequested.disconnect(page._pick_photo_for)
    page.table.edit_current()
    assert len(picked) == 1                              # 사진 칸은 글상자 대신 파일 고르기를 부른다

    inside = write_png(photo_folder / "HN260419.png")
    page.set_photo(index, inside)
    assert workspace.store.get(id_of(page, "서지안"))["사진"] == "HN260419.png"     # 폴더 안이면 이름만
    assert page.chips["missing"].count() == 1

    outside = write_png(tmp_path / "다른곳" / "문서윤.png")
    row = names(page).index("문서윤")
    page.set_photo(page.model.index(row, page.model.column_of("사진")), outside)
    assert workspace.store.get(id_of(page, "문서윤"))["사진"] == outside
    assert page.chips["missing"].count() == 0
    assert not page.chips["missing"].isChecked()


def test_inline_edit_refreshes_counts_and_notifies(page, workspace):
    notices = Notices(workspace)
    index = page.model.index(names(page).index("서지안"), page.model.column_of("사진"))
    page.model.setData(index, "HN260420", Qt.EditRole)
    assert notices.records == 1
    assert page.chips["missing"].count() == 1
    assert page.model.rowCount() == 8                    # 표는 그대로, 자리도 그대로
    assert page.table.current_row() == 0


def test_add_person_with_the_form(page, workspace, qapp):
    notices = Notices(workspace)
    dialog = page.add_person(run=False)
    assert dialog.title() == "한 명 추가"
    assert dialog.card.image() is not None
    dialog.input("이름").setText("한지민")
    dialog.input("부서").setText("개발팀")
    dialog.input("사번").setText("HN260499")
    dialog.input("사진").edit.setText("HN260420")
    dialog.refresh_preview()
    assert dialog._rows["사진"].hint() == "사진을 찾았습니다."
    dialog.save_button.click()
    pump(qapp)
    assert dialog.result() == dialog.DialogCode.Accepted
    new_id = id_of(page, "한지민")
    assert workspace.store.get(new_id)["사번"] == "HN260499"
    assert notices.records >= 1
    assert page.model.rowCount() == 9 and page.chips["all"].count() == 9
    assert workspace.current_record_id == new_id         # 넣은 사람 줄로 간다


def test_add_several_people_in_a_row(page, workspace, qapp):
    dialog = page.add_person(run=False)
    dialog.input("이름").setText("가나다")
    dialog.more_button.click()
    assert dialog.input("이름").text() == "" and dialog.result() != dialog.DialogCode.Accepted
    dialog.input("이름").setText("라마바")
    dialog.save_button.click()
    assert dialog.saved_ids == [id_of(page, "가나다"), id_of(page, "라마바")]
    assert page.model.rowCount() == 10


def test_empty_form_is_not_saved(page, workspace):
    dialog = page.add_person(run=False)
    dialog.save_button.click()
    assert dialog.banner.isVisibleTo(dialog) and "적지 않았습니다" in dialog.banner.text()
    assert workspace.store.count() == 8
    dialog.reject()


def test_double_click_opens_the_form_for_editing(page, workspace, qapp):
    opened = []
    page.table.rowActivated.disconnect(page.edit_person_at)
    page.table.rowActivated.connect(opened.append)
    rect = page.table.visualRect(page.model.index(4, page.model.column_of("이름")))
    QTest.mouseClick(page.table.viewport(), Qt.LeftButton, Qt.NoModifier, rect.center())
    QTest.mouseDClick(page.table.viewport(), Qt.LeftButton, Qt.NoModifier, rect.center())
    assert opened == [4]

    dialog = page.edit_person_at(4, run=False)
    assert dialog.title() == "문서윤 고치기"
    assert dialog.input("부서").text() == "개발팀"
    assert dialog.input("입사일").date().toString("yyyy-MM-dd") == "2026-03-09"
    assert dialog._rows["사진"].hint().startswith("이 이름의 사진 파일을 찾지 못했습니다")
    dialog.input("직급").setText("대리")
    dialog.save_button.click()
    pump(qapp)
    assert workspace.store.get(id_of(page, "문서윤"))["직급"] == "대리"
    assert workspace.current_record_id == id_of(page, "문서윤")


def test_delete_can_be_undone_from_the_toast(page, workspace, qapp):
    target = id_of(page, "고은채")
    page.model.set_checked([target])
    page.table.go_to_row(1)
    QTest.keyClick(page.table, Qt.Key_Delete)
    pump(qapp)
    assert "고은채" not in names(page) and page.model.rowCount() == 7
    assert page.chips["all"].count() == 7 and workspace.checked_ids == []
    assert workspace.store.count(workspace.view_filter) == 7
    toast = page.last_toast()
    assert toast.text() == "고은채 님을 명단에서 지웠습니다."
    assert toast.action_button().text() == "되돌리기"

    toast.action_button().click()
    pump(qapp)
    assert names(page)[1] == "고은채" and page.model.rowCount() == 8
    assert workspace.store.get(target)["이름"] == "고은채"
    assert workspace.current_record_id == target
    assert workspace.view_filter is None


def test_delete_becomes_final_when_the_toast_goes_away(page, workspace, qapp):
    notices = Notices(workspace)
    targets = [id_of(page, "강민서"), id_of(page, "남지호")]
    assert page.delete_people(targets) == 2
    assert page.last_toast().text() == "2명을 명단에서 지웠습니다."
    assert workspace.store.count() == 8                  # 아직 파일에는 남아 있다
    assert page.chips["printed"].count() == 1
    page.last_toast().dismiss()
    pump(qapp)
    assert workspace.store.count() == 6
    assert workspace.store.get(targets[0]) is None
    assert notices.records >= 1
    assert page.undo_delete() is False
    assert page.model.rowCount() == 6 and workspace.view_filter is None


def test_leaving_the_page_finishes_pending_delete(page, workspace):
    page.delete_people([id_of(page, "양수아")])
    page.on_leave()
    assert workspace.store.count() == 7


def test_duplicate_copies_values_and_photo_adjustment(page, workspace, qapp):
    from data.store import PhotoAdjustment

    source = id_of(page, "김하늘")
    workspace.store.set_photo_adjustment(source, "사진", PhotoAdjustment(1.5, 0.1, 0.0))
    created = page.duplicate_people([source])
    assert len(created) == 1
    copy = workspace.store.get(created[0])
    assert copy["이름"] == "김하늘" and copy["사번"] == "HN260417" and not copy["_printed"]
    assert workspace.store.get_photo_adjustment(created[0], "사진") == PhotoAdjustment(1.5, 0.1, 0.0)
    assert page.model.rowCount() == 9
    assert workspace.current_record_id == created[0]


# ---------------------------------------------------------------- 항목(열)

def test_rename_column_is_reflected_and_announced(page, workspace):
    notices = Notices(workspace)
    page.set_condition(Group("and", [Condition("부서", "contains", "개발")]))
    page.model.sort(page.model.column_of("부서"), Qt.AscendingOrder)
    assert page.rename_column("부서", "소속")
    assert notices.columns == 1
    assert "소속" in workspace.store.column_names() and "부서" not in workspace.store.column_names()
    headers = [page.model.headerData(i, Qt.Horizontal) for i in range(page.model.columnCount())]
    assert "소속" in headers and "부서" not in headers
    assert page.condition() == Group("and", [Condition("소속", "contains", "개발")])
    assert workspace.view_order == [("소속", "asc")]
    assert names(page) == ["남지호", "문서윤"]
    # 카드의 '부서' 칸과 끊어졌으니 확인할 것에 나온다
    assert any(i.kind == "unmatched_field" and i.payload == "부서" for i in page.issues())


def test_rename_to_an_existing_name_explains(page, workspace):
    assert not page.rename_column("부서", "이름")
    assert "이미 있습니다" in page.last_toast().text() and "항목" in page.last_toast().text()
    assert "부서" in workspace.store.column_names()


def test_change_type_converts_and_reports(page, workspace):
    notices = Notices(workspace)
    assert page.change_column_type("입사일", "text")
    assert workspace.store.column("입사일").type == "text" and notices.columns == 1
    assert page.last_toast().text() == "'입사일' 항목을 글자로 바꿨습니다."

    assert page.change_column_type("사번", "number")       # 숫자가 아닌 값은 그대로 둔다
    assert workspace.store.column("사번").type == "number"
    assert "8칸은 숫자로 읽을 수 없어" in page.last_toast().text()
    assert workspace.store.get(id_of(page, "강민서"))["사번"] == "HN260415"

    assert page.change_column_type("직급", "list")
    assert workspace.store.column("직급").options["choices"] == ["대리", "사원", "과장", "차장"]


def test_remove_column(page, workspace):
    notices = Notices(workspace)
    assert page.remove_column("입사일", ask=False)
    assert notices.columns == 1
    assert "입사일" not in workspace.store.column_names()
    assert page.model.column_of("입사일") == -1
    assert page.model.columnCount() == 7


def test_remove_column_asks_first(page, workspace, monkeypatch):
    asked = []

    def fake_confirm(parent, title, message, confirm_text="확인", cancel_text="취소", danger=False):
        asked.append((title, message, confirm_text, danger))
        return False

    monkeypatch.setattr("ui.pages.people.page.confirm", fake_confirm)
    assert not page.remove_column("부서")
    assert asked[0][0] == "'부서' 항목을 지울까요?"
    assert "8명" in asked[0][1] and "되돌릴 수 없습니다" in asked[0][1] and asked[0][3] is True
    assert "카드의 '부서' 칸" in asked[0][1]
    assert "부서" in workspace.store.column_names()


def test_hide_and_show_column(page, workspace):
    notices = Notices(workspace)
    page.hide_column("직급")
    assert page.model.column_of("직급") == -1 and notices.columns == 1
    assert workspace.store.get(id_of(page, "강민서"))["직급"] == "대리"       # 값은 그대로
    assert page.model.hidden_columns() == ["직급"]
    page.last_toast().action_button().click()                                # 다시 보이기
    assert page.model.column_of("직급") > 0 and page.model.hidden_columns() == []


def test_add_column(page, workspace):
    notices = Notices(workspace)
    assert page.add_column("연락처")
    assert page.add_column("등급", "list", {"choices": ["A", "B"]})
    assert notices.columns == 2
    assert workspace.store.column("등급").options == {"choices": ["A", "B"]}
    assert page.model.column_of("연락처") > 0
    assert not page.add_column("연락처")                  # 같은 이름
    assert "항목이 이미 있습니다" in page.last_toast().text()


def test_add_column_dialog_uses_plain_type_names(page):
    from ui.pages.people.dialogs import AddColumnDialog

    dialog = AddColumnDialog(page)
    labels = [dialog.type_combo.itemText(i) for i in range(dialog.type_combo.count())]
    assert labels == ["글자", "숫자", "날짜", "시간", "고르는 목록", "사진"]
    assert not dialog.confirm_button.isEnabled()
    dialog.name_edit.setText("직책")
    dialog.type_combo.setCurrentIndex(labels.index("고르는 목록"))
    dialog.choices_edit.setText("팀장, 파트장 ,")
    assert dialog.confirm_button.isEnabled()
    assert (dialog.name(), dialog.column_type(), dialog.options()) == ("직책", "list", {"choices": ["팀장", "파트장"]})


def test_column_menu_reaches_the_header(page, qapp):
    asked = []
    page.table.columnMenuRequested.disconnect(page._show_column_menu)
    page.table.columnMenuRequested.connect(lambda section, pos: asked.append(section))
    header = page.table.header()
    section = page.model.column_of("부서")
    x = header.sectionViewportPosition(section) + header.sectionSize(section) - 10
    from PySide6.QtCore import QPoint

    QTest.mouseClick(header.viewport(), Qt.LeftButton, Qt.NoModifier, QPoint(x, 20))
    assert asked == [section]
    assert page.model.view_order is None                 # 메뉴 자리를 눌렀을 때는 정렬하지 않는다
    QTest.mouseClick(header.viewport(), Qt.LeftButton, Qt.NoModifier,
                     QPoint(header.sectionViewportPosition(section) + 20, 20))
    assert page.model.view_order == [("부서", "asc")]
    QTest.mouseClick(header.viewport(), Qt.LeftButton, Qt.NoModifier,
                     QPoint(header.sectionViewportPosition(section) + 20, 20))
    assert page.model.view_order == [("부서", "desc")]
    QTest.mouseClick(header.viewport(), Qt.LeftButton, Qt.NoModifier,
                     QPoint(header.sectionViewportPosition(section) + 20, 20))
    assert page.model.view_order is None


def test_photo_folder_change_recounts_missing(page, workspace, tmp_path):
    empty = tmp_path / "빈폴더"
    empty.mkdir()
    assert page.change_photo_folder("사진", str(empty))
    assert page.chips["missing"].count() == 8
    assert "8명은 못 찾았습니다" in page.last_toast().text() or "0명의 사진" in page.last_toast().text()


# ---------------------------------------------------------------- 확인할 것, 내보내기, 인쇄 기록

def test_issue_list_and_actions(page, workspace):
    record = id_of(page, "양수아")
    workspace.store.update(record, {"이름": None})
    workspace.store.rename_column("직급", "직위")
    workspace.notify_columns_changed()
    workspace.notify_records_changed()
    issues = {i.kind: i for i in page.issues()}
    assert issues["missing_photo"].text == "사진이 없는 사람이 2명 있습니다."
    assert issues["empty_field"].text == "이름이 비어 있는 사람이 1명 있습니다."
    assert issues["unmatched_field"].payload == "직급"
    assert len(page.preview.issue_rows) == 3

    page._on_issue_action(issues["empty_field"])
    assert page.model.rowCount() == 1 and page.model.id_at(0) == record
    page.set_condition(None)
    page._on_issue_action(issues["missing_photo"])
    assert page.chip() == "missing"

    page.match_field("직급", "직위")                      # 명단의 '직위'를 카드의 '직급' 칸으로
    assert "직급" in workspace.store.column_names()
    assert "unmatched_field" not in {i.kind for i in page.issues()}


def test_no_issues_says_so(qapp, workspace, make_store, photo_folder):
    from people_helpers import write_png

    for person in PEOPLE:
        write_png(photo_folder / f"{person['사번']}.png")
    people = [dict(p, 사진=p["사번"]) for p in PEOPLE]
    page = PeoplePage(workspace)
    workspace.set_store(make_store(people))
    assert page.issues() == []
    assert not page.chips["missing"].count()
    assert "확인할 것이 없습니다" in page.preview.issue_rows[0].findChildren(type(page.stats_label))[0].text()


def test_export_csv_and_xlsx(page, workspace, tmp_path):
    from openpyxl import load_workbook

    out = tmp_path / "out" / "명단.csv"
    assert page.export_to(out, "all", include_status=True) == 8
    with open(out, encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))
    assert rows[0] == ["이름", "부서", "직급", "사번", "입사일", "사진", "인쇄 여부", "인쇄 횟수", "마지막 인쇄 시각"]
    assert rows[4][0] == "남지호" and rows[4][6] == "인쇄됨"

    page.set_chip("unprinted")
    page.model.sort(page.model.column_of("이름"), Qt.DescendingOrder)
    xlsx = tmp_path / "보이는사람.xlsx"
    assert page.export_to(xlsx, "view") == 6
    sheet = load_workbook(xlsx).active
    values = [row[0].value for row in sheet.iter_rows(min_row=2)]
    assert values[0] == "양수아" and "남지호" not in values and sheet.max_column == 6

    page.model.set_checked([id_of(page, "강민서"), id_of(page, "남지호")])
    picked = tmp_path / "고른사람.csv"
    assert page.export_to(picked, "checked") == 2


def test_export_dialog_offers_scopes(page):
    from ui.pages.people.dialogs import ExportDialog

    dialog = ExportDialog(128, 41, 12, True, page)
    assert list(dialog.scope_buttons) == ["all", "view", "checked"]
    assert dialog.scope_buttons["view"].text() == "지금 보이는 41명"
    assert dialog.scope() == "view" and dialog.file_format() == "xlsx" and not dialog.include_status()
    plain = ExportDialog(128, 128, 0, False, page)
    assert list(plain.scope_buttons) == ["all"] and plain.scope() == "all"


def test_reset_print_status_for_selection_then_all(page, workspace, monkeypatch):
    asked = []
    monkeypatch.setattr("ui.pages.people.page.confirm",
                        lambda parent, title, message, *a, **k: asked.append(message) or True)
    page.model.set_checked([id_of(page, "남지호")])
    page.reset_print_status()
    assert "선택한 1명" in asked[-1]
    assert not workspace.store.get(id_of(page, "남지호"))["_printed"]
    assert workspace.store.get(id_of(page, "박도윤"))["_printed"]
    assert page.chips["printed"].count() == 1

    page.clear_checked()
    page.reset_print_status()
    assert "모든 사람(8명)" in asked[-1]
    assert page.chips["printed"].count() == 0 and page.chips["unprinted"].count() == 8


def test_print_status_changes_from_outside_show_up(page, workspace):
    workspace.store.mark_printed([id_of(page, "강민서")])
    workspace.notify_records_changed()
    assert page.chips["printed"].count() == 3
    status = page.model.data(page.model.index(0, page.model.columnCount() - 1))
    assert status.endswith("1장")
    assert workspace.current_record_id == id_of(page, "강민서")


def test_closing_the_list_returns_to_the_empty_state(page, workspace, qapp):
    path = workspace.store.path
    page.close_list()
    pump(qapp)
    assert workspace.store is None
    assert page.empty_view.isVisibleTo(page)
    assert [r.path for r in storage.recent_lists()] == [str(Path(path))]
    assert page.footer_actions()[-1].enabled is False


def test_page_created_after_the_list_was_opened_still_hears_changes(qapp, workspace, people_store):
    workspace.set_store(people_store)
    page = PeoplePage(workspace)
    assert page.model.rowCount() == 8 and not page.empty_view.isVisibleTo(page)
    people_store.insert({"이름": "한지민"})
    workspace.notify_records_changed()
    assert page.model.rowCount() == 9
    people_store.add_column("연락처")
    workspace.notify_columns_changed()
    assert page.model.column_of("연락처") > 0


def test_opening_a_list_reads_it_once(qapp, workspace, people_store):
    from people_helpers import spy_rows

    page = PeoplePage(workspace)
    calls = spy_rows(people_store)
    workspace.set_store(people_store)
    scans = [c for c in calls if c[1] == 2000]            # 사진 없는 사람을 세는 훑기
    assert len(scans) == 1
    people_store.insert({"이름": "한지민"})
    workspace.notify_records_changed()                    # 그 뒤의 알림은 놓치지 않는다
    assert page.model.rowCount() == 9


def test_hiding_the_page_finishes_pending_delete(page, workspace, qapp):
    page.delete_people([id_of(page, "양수아")])
    page.hide()
    pump(qapp)
    assert workspace.store.count() == 7


def test_changing_a_column_to_photo_asks_for_the_folder(page, workspace):
    page.add_column("증명사진")
    assert page.change_column_type("증명사진", "photo")
    toast = page.last_toast()
    assert "사진이 들어 있는 폴더" in toast.text() and toast.action_button().text() == "사진 폴더 고르기"


def test_select_everything_then_space_on_a_big_list_stays_light(qapp, workspace, tmp_path):
    from people_helpers import spy_rows

    from data.store import DataStore

    store = DataStore(tmp_path / "big.hsdata")
    store.add_column("이름")
    store.insert_many({"이름": f"사람{i:05d}"} for i in range(10_000))
    page = PeoplePage(workspace)
    page.resize(1196, 620)
    page.show()
    workspace.set_store(store)
    pump(qapp)
    calls = spy_rows(store)
    page.table.selectAll()
    QTest.keyClick(page.table, Qt.Key_Space)
    pump(qapp)
    assert len(workspace.checked_ids) == 10_000
    assert page.footer_actions()[-1].text == "다음: 10,000장 인쇄하기"
    assert sum(count for _, _, count in calls) <= 1_000     # 줄을 전부 읽어 오지 않는다
    page.close()


def test_clicking_an_empty_photo_cell_asks_for_a_file(page, qapp):
    asked = []
    page.table.photoPickRequested.disconnect(page._pick_photo_for)
    page.table.photoPickRequested.connect(lambda index: asked.append(index.row()))
    photo = page.model.column_of("사진")
    QTest.mouseClick(page.table.viewport(), Qt.LeftButton, Qt.NoModifier,
                     page.table.visualRect(page.model.index(4, photo)).center())
    assert asked == [4]                                  # 문서윤: 사진 없음
    QTest.mouseClick(page.table.viewport(), Qt.LeftButton, Qt.NoModifier,
                     page.table.visualRect(page.model.index(0, photo)).center())
    assert asked == [4]                                  # 사진이 있는 칸은 그냥 줄만 고른다
