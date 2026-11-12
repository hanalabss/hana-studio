from pathlib import Path

import pytest
from people_helpers import *  # noqa: F401,F403
from people_helpers import PEOPLE, pump, write_png

from data.errors import DataError
from data.importers import preview_file
from data.store import DataStore
from ui.pages.people import PeoplePage
from ui.pages.people.importing import (
    PAGE_DONE,
    PAGE_FILE,
    PAGE_MATCH,
    DesignField,
    ImportPlan,
    ImportWizard,
    count_photos,
    design_fields,
    header_is_uncertain,
    resolve_plan,
    run_import,
    suggest_field_map,
    suggest_photo_source,
    text_is_uncertain,
)

HEADERS = ["성명", "소속부서", "직급", "사원번호", "입사일"]
ROWS = [[p["이름"], p["부서"], p["직급"], p["사번"], p["입사일"]] for p in PEOPLE]


@pytest.fixture
def cp949_csv(tmp_path):
    path = tmp_path / "신입사원_명단.csv"
    lines = [",".join(HEADERS)] + [",".join(row) for row in ROWS]
    path.write_bytes("\r\n".join(lines).encode("cp949"))
    return path


@pytest.fixture
def xlsx(tmp_path):
    from openpyxl import Workbook

    workbook = Workbook()
    first = workbook.active
    first.title = "안내"
    first.append(["2026년 신입사원 명단입니다"])
    sheet = workbook.create_sheet("신입사원")
    sheet.append(HEADERS)
    for row in ROWS:
        sheet.append(row)
    old = workbook.create_sheet("퇴사자")
    old.append(HEADERS)
    old.append(["한퇴사", "영업1팀", "과장", "HN100001", "2010-01-04"])
    path = tmp_path / "신입사원_명단.xlsx"
    workbook.save(path)
    return path


@pytest.fixture
def empty_page(qapp, workspace):
    page = PeoplePage(workspace)
    page.resize(1196, 620)
    page.show()
    pump(qapp)
    yield page
    page.close()


def open_wizard(page, path) -> ImportWizard:
    wizard = page.start_import(path, run=False)
    assert wizard.wait_idle()
    return wizard


# ---------------------------------------------------------------- 계획 세우기 (화면 없이)

def test_design_fields_come_from_the_card(workspace):
    fields = design_fields(workspace.editor.document)
    assert [(f.name, f.is_photo) for f in fields] == [
        ("사진", True), ("이름", False), ("부서", False), ("직급", False), ("사번", False)]


def test_mapping_suggestions_use_synonyms_and_containment(cp949_csv, workspace):
    preview = preview_file(cp949_csv)
    fields = design_fields(workspace.editor.document)
    assert suggest_field_map(fields, preview) == {
        "사진": None, "이름": "성명", "부서": "소속부서", "직급": "직급", "사번": "사원번호"}
    # 헷갈릴 열이 둘이면 넘겨짚지 않는다
    twin = preview_file(cp949_csv)
    twin.columns[2].name = "부서코드"
    assert suggest_field_map([DesignField("부서")], twin)["부서"] is None


def test_photo_source_is_the_column_that_matches_files(cp949_csv, photo_folder):
    preview = preview_file(cp949_csv)
    assert suggest_photo_source(preview, None) == "사원번호"          # 폴더를 모르면 이름으로 짐작
    assert suggest_photo_source(preview, photo_folder) == "사원번호"
    for name in ("강민서", "고은채", "김하늘", "남지호", "문서윤", "박도윤", "서지안"):
        write_png(photo_folder.parent / "이름사진" / f"{name}.jpg")
    assert suggest_photo_source(preview, photo_folder.parent / "이름사진") == "성명"
    found, missing = count_photos(preview, "사원번호", photo_folder)
    assert (found, missing) == (6, [4, 6])


def test_resolve_plan_maps_fields_rest_and_photo_copy(cp949_csv, photo_folder):
    preview = preview_file(cp949_csv)
    plan = ImportPlan(preview, {"이름": "성명", "부서": "소속부서", "직급": "직급", "사번": "사원번호"},
                      photo_field="사진", photo_source="사원번호", photo_folder=str(photo_folder))
    resolved, column_map = resolve_plan(plan)
    assert column_map == {"성명": "이름", "소속부서": "부서", "직급": "직급", "사원번호": "사번",
                          "입사일": "입사일", "사진 파일 이름": "사진"}
    assert resolved.columns[-1].type == "photo" and resolved.rows[0][-1] == "HN260415"
    assert len(preview.columns) == 5                     # 원래 미리보기는 건드리지 않는다

    plan.include_rest = False
    assert resolve_plan(plan)[1]["입사일"] is None


def test_resolve_plan_rejects_one_column_for_two_fields(cp949_csv):
    preview = preview_file(cp949_csv)
    with pytest.raises(DataError, match="한 열은 한 칸에만"):
        resolve_plan(ImportPlan(preview, {"이름": "성명", "부서": "성명"}))
    with pytest.raises(DataError, match="가져올 열이 없습니다"):
        resolve_plan(ImportPlan(preview, {"이름": None}, include_rest=False))


def test_rest_column_named_like_a_field_gets_a_new_name(tmp_path):
    path = tmp_path / "a.csv"
    path.write_text("이름,별명\n김하늘,하늘이\n", encoding="utf-8")
    preview = preview_file(path)
    _, column_map = resolve_plan(ImportPlan(preview, {"이름": "별명"}))
    assert column_map == {"별명": "이름", "이름": "이름 (2)"}


def test_run_import_reports_progress_and_can_cancel(tmp_path):
    path = tmp_path / "many.csv"
    path.write_text("이름,번호\n" + "\n".join(f"사람{i},{i}" for i in range(1000)), encoding="utf-8")
    store = DataStore(tmp_path / "many.hsdata")
    store.add_column("예전")
    store.insert({"예전": "남아야 함"})
    seen = []
    with pytest.raises(Exception) as caught:
        run_import(store, ImportPlan(preview_file(path), mode="replace"),
                   report=lambda done, total: seen.append((done, total)), cancelled=lambda: len(seen) >= 5)
    assert type(caught.value).__name__ == "ImportCancelled"
    assert seen[0] == (0, 1000) and seen[-1][0] == 400
    assert store.count() == 1 and store.column_names() == ["예전"]      # 전부 되돌려졌다

    seen.clear()
    outcome = run_import(store, ImportPlan(preview_file(path)), report=lambda done, total: seen.append(done))
    assert outcome.result.inserted == 1000 and outcome.total == 1001
    assert seen[-1] == 1000 and seen == sorted(seen)
    store.close()


def test_detection_doubts(tmp_path, cp949_csv):
    sure = preview_file(cp949_csv)
    assert not header_is_uncertain(sure) and not text_is_uncertain(sure)
    no_header = tmp_path / "noheader.csv"
    no_header.write_text("김하늘,1001\n이바다,1002\n", encoding="utf-8")
    assert header_is_uncertain(preview_file(no_header))
    one_column = tmp_path / "one.csv"
    one_column.write_text("이름 부서\n김하늘 개발팀\n이바다 영업팀\n", encoding="utf-8")
    assert text_is_uncertain(preview_file(one_column))


# ---------------------------------------------------------------- 가져오기 창: CP949 CSV

def test_wizard_imports_a_cp949_csv_into_a_new_list(empty_page, workspace, cp949_csv, photo_folder, tmp_path, qapp):
    original = cp949_csv.read_bytes()
    wizard = open_wizard(empty_page, cp949_csv)
    assert wizard.page_index() == PAGE_FILE
    assert wizard.title() == "신입사원_명단.csv에서 8명을 찾았습니다"
    assert wizard.preview.encoding == "cp949" and wizard.preview.delimiter == ","
    assert wizard.preview_table.rowCount() == 8
    assert [wizard.preview_table.horizontalHeaderItem(i).text() for i in range(5)] == HEADERS
    assert wizard.preview_table.item(0, 0).text() == "강민서"        # 한글이 깨지지 않았다
    assert wizard.header_check.isChecked()
    assert not wizard.detail_box.isVisibleTo(wizard)                 # 확실하면 접어 둔다
    assert not wizard.file_banner.isVisibleTo(wizard)
    assert not wizard.sheet_row.isVisibleTo(wizard)
    assert wizard.next_button.text() == "다음" and wizard.next_button.isEnabled()
    assert workspace.store is None                                   # 아직 아무것도 만들지 않았다

    wizard.go_next()
    assert wizard.page_index() == PAGE_MATCH
    assert wizard.title() == "신입사원_명단.csv에서 8명을 가져옵니다"
    assert wizard.mapping() == {"이름": "성명", "부서": "소속부서", "직급": "직급", "사번": "사원번호"}
    assert "카드의 칸 4개를 파일의 열과 맞췄습니다" in wizard.description()
    assert wizard.rest_check.isChecked() and "입사일" in wizard.rest_hint.text()
    assert not wizard.mode_card.isVisibleTo(wizard)                  # 새 명단이라 물을 것이 없다
    assert wizard.photo_card.isVisibleTo(wizard)
    assert wizard.photo_source_combo.currentData() == "사원번호"
    assert wizard.next_button.text() == "8명 가져오기"
    assert wizard.sample_card.image() is not None
    assert wizard.sample_record()["이름"] == "강민서"

    wizard.set_photo_folder(photo_folder)
    assert wizard.photo_counts() == (6, 2)
    assert "8명 중 6명은 찾았고, 2명은 못 찾았습니다 (문서윤, 서지안)" in wizard.photo_status.text()
    assert wizard.sample_record()["사진"] == str(photo_folder / "HN260415.png")

    wizard.start_import()
    assert wizard.wait_idle()
    assert wizard.page_index() == PAGE_DONE
    assert wizard.title() == "신입사원_명단.csv에서 8명을 가져왔습니다"
    assert wizard.outcome.result.inserted == 8
    assert (wizard.outcome.photo_found, wizard.outcome.photo_missing) == (6, 2)
    assert wizard.next_button.text() == "명단 보기"
    assert wizard.done_cards[0].image() is not None

    wizard.go_next()
    pump(qapp)
    store = workspace.store
    assert store is not None and store is wizard.store
    assert Path(store.path) == tmp_path / "lists" / "신입사원_명단.hsdata"
    assert store.column_names() == ["이름", "부서", "직급", "사번", "입사일", "사진"]
    assert store.column("사진").type == "photo" and store.column("입사일").type == "date"
    assert store.column("사진").options["folder"] == str(photo_folder)
    first = store.rows()[0]
    assert (first["이름"], first["부서"], first["사번"], first["사진"]) == ("강민서", "영업1팀", "HN260415", "HN260415")
    assert cp949_csv.read_bytes() == original                        # 원본은 그대로

    # 화면: 표가 채워지고, 사진 없는 두 명이 보이고, 모두 체크돼 바로 인쇄로 갈 수 있다
    assert empty_page.model.rowCount() == 8
    assert empty_page.chips["missing"].count() == 2
    assert len(workspace.checked_ids) == 8
    assert empty_page.select_banner.isVisibleTo(empty_page)
    assert empty_page.footer_actions()[-1].text == "다음: 8장 인쇄하기"


def test_wizard_shows_details_when_detection_is_uncertain(empty_page, tmp_path):
    path = tmp_path / "제목없음.csv"
    path.write_bytes("김하늘;1001\r\n이바다;1002\r\n".encode("cp949"))
    wizard = open_wizard(empty_page, path)
    assert wizard.preview.delimiter == ";"
    assert not wizard.header_check.isChecked()
    assert wizard.detail_box.isVisibleTo(wizard)                     # 확실하지 않으면 펼쳐서 묻는다
    assert "첫 줄이 제목인지" in wizard.file_banner.text()
    assert wizard.title().endswith("2명을 찾았습니다")

    wizard.header_check.click()                                     # 첫 줄을 제목으로
    assert wizard.wait_idle()
    assert wizard.preview.has_header and wizard.preview.total_rows == 1
    assert [c.name for c in wizard.preview.columns] == ["김하늘", "1001"]

    wizard.header_check.click()
    wizard.wait_idle()
    index = wizard.delimiter_combo.findData(",")
    wizard.delimiter_combo.setCurrentIndex(index)
    wizard.delimiter_combo.activated.emit(index)
    assert wizard.wait_idle()
    assert len(wizard.preview.columns) == 1                          # 쉼표로 나누면 한 칸
    wizard.reject()


def test_wizard_explains_unreadable_files(empty_page, tmp_path):
    bad = tmp_path / "옛날.xls"
    bad.write_bytes(b"not really excel")
    wizard = open_wizard(empty_page, bad)
    assert wizard.preview is None
    assert wizard.title() == "이 파일은 가져올 수 없습니다"
    assert ".xlsx" in wizard.file_banner.text() and wizard.file_banner.kind() == "error"
    assert not wizard.next_button.isEnabled()
    wizard.reject()
    assert empty_page.store is None


# ---------------------------------------------------------------- 가져오기 창: 시트가 여러 개인 엑셀

def test_wizard_imports_a_multi_sheet_xlsx(empty_page, workspace, xlsx, photo_folder, qapp):
    wizard = open_wizard(empty_page, xlsx)
    assert wizard.sheet_row.isVisibleTo(wizard)
    assert [wizard.sheet_combo.itemText(i) for i in range(wizard.sheet_combo.count())] == ["안내", "신입사원", "퇴사자"]
    assert wizard.sheet_combo.currentText() == "안내"                # 첫 시트는 안내문뿐이다
    assert wizard.preview.total_rows <= 1

    index = wizard.sheet_combo.findText("신입사원")
    wizard.sheet_combo.setCurrentIndex(index)
    wizard.sheet_combo.activated.emit(index)
    assert wizard.wait_idle()
    assert wizard.preview.sheet == "신입사원" and wizard.preview.total_rows == 8
    assert wizard.title() == "신입사원_명단.xlsx에서 8명을 찾았습니다"
    assert not wizard.text_row.isVisibleTo(wizard)                   # 엑셀에는 글자·구분 설정이 없다

    wizard.go_next()
    assert wizard.mapping() == {"이름": "성명", "부서": "소속부서", "직급": "직급", "사번": "사원번호"}
    wizard.set_mapping("직급", None)                                 # 직급은 가져오지 않기로
    assert "카드의 칸 4개 중 3개를 맞췄습니다" in wizard.description()
    wizard.rest_check.setChecked(False)
    wizard.set_photo_folder(photo_folder)
    assert wizard.photo_counts() == (6, 2)
    wizard.start_import()
    assert wizard.wait_idle()
    assert wizard.outcome.result.inserted == 8
    wizard.accept()
    pump(qapp)
    assert workspace.store.column_names() == ["이름", "부서", "사번", "사진"]
    assert any(i.kind == "unmatched_field" and i.payload == "직급" for i in empty_page.issues())


def test_sample_card_steps_through_people(empty_page, xlsx):
    wizard = open_wizard(empty_page, xlsx)
    index = wizard.sheet_combo.findText("신입사원")
    wizard.sheet_combo.setCurrentIndex(index)
    wizard.sheet_combo.activated.emit(index)
    wizard.wait_idle()
    wizard.go_next()
    assert wizard.sample_position.text() == "1 / 8" and not wizard.sample_prev.isEnabled()
    wizard.sample_next.click()
    assert wizard.sample_position.text() == "2 / 8" and wizard.sample_record()["이름"] == "고은채"
    wizard.reject()


# ---------------------------------------------------------------- 이미 있는 명단에 다시 가져오기

def test_reimport_append_skips_people_already_there(page, workspace, cp949_csv, tmp_path, qapp):
    more = tmp_path / "추가.csv"
    lines = [",".join(HEADERS), "강민서,영업1팀,대리,HN260415,2026-03-02", "한지민,개발팀,사원,HN260499,2026-04-01"]
    more.write_text("\n".join(lines), encoding="utf-8-sig")
    wizard = open_wizard(page, more)
    wizard.go_next()
    assert wizard.mode_card.isVisibleTo(wizard)
    assert wizard.mode_control.current() == "append"
    assert wizard.mode_hint.text() == "지금 있는 8명은 그대로 두고, 그 뒤에 2명을 더합니다."
    assert wizard.mapping()["이름"] == "성명" and wizard.mapping()["입사일"] == "입사일"
    assert wizard.key_combo.currentData() == "사번"
    assert not wizard.key_combo.isEnabled()
    wizard.skip_check.setChecked(True)
    assert wizard.key_combo.isEnabled()
    wizard.start_import()
    assert wizard.wait_idle()
    assert wizard.outcome.result.inserted == 1 and wizard.outcome.skipped_existing == 1
    assert wizard.created_store is False
    wizard.accept()
    pump(qapp)
    assert workspace.store.count() == 9
    assert page.model.rowCount() == 9 and page.chips["all"].count() == 9
    assert workspace.store.rows()[-1]["이름"] == "한지민"
    assert workspace.store.column("사진").options.get("folder")       # 사진 폴더 설정은 그대로


def test_reimport_replace_spells_out_the_consequence(page, workspace, cp949_csv, qapp):
    wizard = open_wizard(page, cp949_csv)
    wizard.go_next()
    wizard.mode_control.set_current("replace")
    assert "지금 있는 8명을 모두 지우고" in wizard.mode_hint.text() and "되돌릴 수 없습니다" in wizard.mode_hint.text()
    assert wizard.next_button.text() == "8명으로 바꾸기"
    assert not wizard.skip_check.isEnabled()
    wizard.start_import()
    assert wizard.wait_idle()
    wizard.accept()
    pump(qapp)
    assert workspace.store.count() == 8
    assert page.chips["printed"].count() == 0                         # 인쇄 기록도 새로 시작


def test_cancelled_wizard_leaves_nothing_behind(empty_page, workspace, cp949_csv, tmp_path):
    wizard = open_wizard(empty_page, cp949_csv)
    wizard.go_next()
    wizard.cancel_button.click()
    assert workspace.store is None
    assert not (tmp_path / "lists").exists() or not list((tmp_path / "lists").glob("*.hsdata"))


def test_failed_import_removes_the_new_file_and_says_why(empty_page, workspace, cp949_csv, tmp_path):
    wizard = open_wizard(empty_page, cp949_csv)
    wizard.go_next()
    wizard.set_mapping("부서", "성명")                                # 한 열을 두 칸에
    wizard.start_import()
    wizard.wait_idle()
    assert wizard.page_index() == PAGE_MATCH
    assert wizard.match_banner.isVisibleTo(wizard) and "한 열은 한 칸에만" in wizard.match_banner.text()
    assert wizard.store is None
    assert not list((tmp_path / "lists").glob("*.hsdata")) if (tmp_path / "lists").exists() else True
    wizard.reject()


def test_import_runs_off_the_gui_thread_and_freezes_the_table(empty_page, cp949_csv):
    from PySide6.QtCore import QThread

    wizard = open_wizard(empty_page, cp949_csv)
    wizard.go_next()
    busy = []
    wizard.busyChanged.connect(busy.append)
    wizard.start_import()
    assert wizard.is_busy() and wizard._worker is not None
    assert isinstance(wizard._worker, QThread) and wizard._worker is not QThread.currentThread()
    assert empty_page.model._frozen is True
    wizard.wait_idle()
    assert busy == [True, False]
    assert empty_page.model._frozen is False
    wizard.accept()


def test_blank_design_imports_every_column(qapp, tmp_path, cp949_csv):
    from ui.pages.workspace import Workspace

    ws = Workspace()
    page = PeoplePage(ws)
    wizard = open_wizard(page, cp949_csv)
    wizard.go_next()
    assert wizard.mapping() == {}
    assert "파일의 열을 그대로 모두 가져옵니다" in wizard.description()
    assert not wizard.photo_card.isVisibleTo(wizard) and wizard.photo_link.isVisibleTo(wizard)
    wizard.start_import()
    wizard.wait_idle()
    wizard.accept()
    assert ws.store.column_names() == HEADERS
    ws.close()
