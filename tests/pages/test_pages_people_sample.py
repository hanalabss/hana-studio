from pathlib import Path

import pytest
from people_helpers import *  # noqa: F401,F403
from people_helpers import pump

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QFileDialog

from data.photos import PhotoResolver
from data.samples import MISSING_PHOTO_ID, write_sample
from ui.pages.people import PeoplePage
from ui.pages.people import sample as sample_ui
from ui.pages.people.importing import PAGE_DONE, PAGE_FILE, PAGE_MATCH
from ui.pages.people.sample import SampleSavedDialog, default_sample_folder, sample_headers_for
from ui.widgets import Dialog


@pytest.fixture
def empty_page(qapp, workspace):
    page = PeoplePage(workspace)
    page.resize(1196, 620)
    page.show()
    pump(qapp)
    yield page
    page.close()


@pytest.fixture
def folder_dialog(monkeypatch, tmp_path):
    """폴더 고르기 창 대신 정해 둔 폴더를 돌려준다. 무엇을 물었는지 남긴다."""
    asked = {"calls": [], "answer": str(tmp_path / "문서")}
    (tmp_path / "문서").mkdir()

    def fake(parent=None, caption="", directory="", *args, **kwargs):
        asked["calls"].append((caption, directory))
        return asked["answer"]

    monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(fake))
    return asked


@pytest.fixture
def saved_dialog(monkeypatch):
    """'예시 파일을 저장했습니다' 창을 띄우지 않고 정해 둔 대답을 한다. 뜬 창을 남긴다."""
    shown = {"dialogs": [], "answer": QDialog.Rejected}

    def fake_exec(self):
        shown["dialogs"].append(self)
        return shown["answer"]

    monkeypatch.setattr(SampleSavedDialog, "exec", fake_exec)
    return shown


def finish(wizard, qapp):
    assert wizard.wait_idle()
    wizard.go_next()
    wizard.start_import()
    assert wizard.wait_idle()
    assert wizard.page_index() == PAGE_DONE
    wizard.go_next()
    pump(qapp)


# ---------------------------------------------------------------- 빈 화면

def test_empty_state_offers_the_sample_as_a_secondary_action(empty_page, qapp):
    view = empty_page.empty_view
    button = view.sample_button
    assert button.isVisibleTo(empty_page) and button.isEnabled()
    assert button.text() == "예시 파일 받기" and button.accessibleName() == "예시 파일 받기"
    assert "예시 명단" in button.toolTip()
    assert button.variant() == "quiet" and view.import_button.variant() == "primary"   # 주 버튼은 그대로
    assert view.import_button.text() == "파일 가져오기"
    assert button.focusPolicy() & Qt.TabFocus
    # Tab 순서: 파일 가져오기 -> 예시 파일 받기 -> 직접 입력
    def next_stop(widget):
        widget = widget.nextInFocusChain()
        while not (widget.focusPolicy() & Qt.TabFocus and widget.isVisibleTo(empty_page)):
            widget = widget.nextInFocusChain()
        return widget

    assert next_stop(view.import_button) is button
    assert next_stop(button) is view.manual_button


def test_sample_button_asks_for_a_folder_and_writes_the_files(empty_page, workspace, folder_dialog, saved_dialog,
                                                              tmp_path, qapp):
    empty_page.empty_view.sample_button.click()
    pump(qapp)
    assert folder_dialog["calls"] == [("예시 파일을 저장할 폴더 고르기", default_sample_folder())]
    target = tmp_path / "문서"
    assert (target / "하나스튜디오 명단 예시.xlsx").is_file()
    assert len(list((target / "사진 예시").glob("*.jpg"))) == 9

    dialog = saved_dialog["dialogs"][0]
    assert dialog.title() == "예시 파일을 저장했습니다"
    assert dialog.folder_label.full_text() == str(target)
    told = " ".join(text for _icon, text in dialog.lines)
    assert "하나스튜디오 명단 예시.xlsx" in told and "사진 예시 폴더" in told and "서이슬" in told
    assert dialog.try_button.text() == "이 파일로 바로 해 보기" and dialog.try_button.isDefault()
    assert dialog.open_button.accessibleName() == "예시 파일을 저장한 폴더 열기"
    assert workspace.store is None and empty_page.wizard is None          # '닫기'를 눌렀으니 저장만 했다


def test_default_folder_is_the_users_documents():
    from PySide6.QtCore import QStandardPaths

    documents = QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation)
    assert default_sample_folder() == (documents or str(Path.home()))
    assert default_sample_folder()


def test_cancelling_the_folder_chooser_does_nothing(empty_page, folder_dialog, saved_dialog, tmp_path):
    folder_dialog["answer"] = ""
    assert empty_page.get_sample() is None
    assert saved_dialog["dialogs"] == [] and list((tmp_path / "문서").iterdir()) == []


def test_sample_matches_the_cards_fields(empty_page, workspace, tmp_path):
    headers, photos = sample_headers_for(workspace.editor.document)
    assert headers == ["사진", "이름", "부서", "직급", "사번"] and photos == ["사진"]
    sample = empty_page.get_sample(tmp_path, ask=False)
    assert sample.headers == ["이름", "부서", "직급", "사번"]                # 사진 칸은 열로 만들지 않는다
    assert sample.photo_headers == ["사진"] and sample.id_header == "사번"


def test_sample_for_a_card_without_fields_uses_the_default_headers(qapp, tmp_path):
    from ui.pages.workspace import Workspace

    ws = Workspace()
    page = PeoplePage(ws)
    assert sample_headers_for(ws.editor.document) == (None, [])
    sample = page.get_sample(tmp_path, ask=False)
    assert sample.headers == ["이름", "부서", "직급", "사번"]

    wizard = page.try_sample(sample, run=False)                             # 사진 칸이 없는 카드에서도 사진 폴더를 쓴다
    finish(wizard, qapp)
    store = ws.store
    assert store.column_names() == ["이름", "부서", "직급", "사번", "사진"]
    assert store.column("사진").type == "photo" and store.column("사진").options["folder"] == str(sample.photo_dir)
    assert (wizard.outcome.photo_found, wizard.outcome.photo_missing) == (9, 1)
    page.close()
    ws.close()


def test_write_failure_is_explained_without_a_traceback(empty_page, workspace, saved_dialog, tmp_path, qapp):
    blocker = tmp_path / "파일입니다"
    blocker.write_text("x", encoding="utf-8")
    assert empty_page.get_sample(blocker) is None
    pump(qapp)
    toast = empty_page.last_toast()
    assert toast is not None and "저장할 수 없습니다" in toast.text()
    assert "Traceback" not in toast.text() and "Errno" not in toast.text()
    assert saved_dialog["dialogs"] == [] and workspace.store is None


def test_try_it_now_runs_the_normal_import_with_the_photo_folder_set(empty_page, workspace, folder_dialog,
                                                                     saved_dialog, tmp_path, qapp):
    saved_dialog["answer"] = QDialog.Accepted                               # '이 파일로 바로 해 보기'
    sample = empty_page.get_sample(run=False)
    wizard = empty_page.wizard
    assert wizard is not None and wizard.wait_idle()
    assert wizard.page_index() == PAGE_FILE
    assert wizard.title() == "하나스튜디오 명단 예시.xlsx에서 10명을 찾았습니다"
    assert wizard.preview.sheet == "명단" and not wizard.file_banner.isVisibleTo(wizard)

    wizard.go_next()
    assert wizard.page_index() == PAGE_MATCH
    assert wizard.mapping() == {"이름": "이름", "부서": "부서", "직급": "직급", "사번": "사번"}
    assert "카드의 칸 4개를 파일의 열과 맞췄습니다" in wizard.description()
    assert wizard.photo_card.isVisibleTo(wizard)
    assert wizard.folder_label.full_text() == str(sample.photo_dir)         # 사진 폴더가 미리 골라져 있다
    assert wizard.photo_source() == "사번" and wizard.photo_counts() == (9, 1)
    assert "서이슬" in wizard.photo_status.text()
    assert wizard.sample_record()["사진"] == str(sample.photo_dir / "HN260411.jpg")

    wizard.start_import()
    assert wizard.wait_idle()
    assert wizard.page_index() == PAGE_DONE
    assert wizard.outcome.result.inserted == 10 and not wizard.outcome.result.warnings
    assert (wizard.outcome.photo_found, wizard.outcome.photo_missing) == (9, 1)
    wizard.go_next()
    pump(qapp)

    store = workspace.store
    assert store is not None and Path(store.path) == tmp_path / "lists" / "하나스튜디오 명단 예시.hsdata"
    assert store.count() == 10 and store.column_names() == ["이름", "부서", "직급", "사번", "사진"]
    assert store.column("사진").type == "photo" and store.column("사진").options["folder"] == str(sample.photo_dir)
    resolver = PhotoResolver(store)
    without = [r["사번"] for r in store.rows() if resolver.path(r, "사진") is None]
    assert without == [MISSING_PHOTO_ID]
    # 화면: 사람들이 표에 있고, 사진 없는 한 명이 보인다
    assert not empty_page.empty_view.isVisibleTo(empty_page)
    assert empty_page.model.rowCount() == 10
    assert empty_page.chips["missing"].count() == 1
    assert len(workspace.checked_ids) == 10


def test_saving_twice_keeps_the_first_files(empty_page, tmp_path):
    first = empty_page.get_sample(tmp_path, ask=False)
    second = empty_page.get_sample(tmp_path, ask=False)
    assert first.workbook_path.is_file() and second.workbook_path.name == "하나스튜디오 명단 예시 (2).xlsx"
    assert second.photo_dir.name == "사진 예시 (2)"


# ---------------------------------------------------------------- 가져오기 창 첫 화면

def test_wizard_first_step_offers_the_sample(empty_page, workspace, folder_dialog, saved_dialog, tmp_path, qapp):
    csv = tmp_path / "아무거나.csv"
    csv.write_text("이름,메모\n김하늘,없음\n", encoding="utf-8")
    wizard = empty_page.start_import(csv, run=False)
    assert wizard.wait_idle() and wizard.page_index() == PAGE_FILE
    button = wizard.sample_button
    assert button.isVisibleTo(wizard) and button.text() == "예시 파일 받기"
    assert button.accessibleName() == "예시 파일 받기" and button.focusPolicy() & Qt.TabFocus
    assert button.variant() == "quiet" and wizard.next_button.variant() == "primary"

    saved_dialog["answer"] = QDialog.Accepted
    button.click()
    assert wizard.wait_idle()
    target = tmp_path / "문서"
    assert (target / "하나스튜디오 명단 예시.xlsx").is_file() and (target / "사진 예시").is_dir()
    assert saved_dialog["dialogs"][0].parent() is wizard
    assert wizard.title() == "하나스튜디오 명단 예시.xlsx에서 10명을 찾았습니다"     # 같은 창에서 그 파일로 넘어갔다
    assert wizard.page_index() == PAGE_FILE

    wizard.go_next()
    assert wizard.photo_counts() == (9, 1) and wizard.photo_source() == "사번"
    wizard.start_import()
    assert wizard.wait_idle()
    wizard.go_next()
    pump(qapp)
    assert workspace.store.count() == 10
    assert workspace.store.column("사진").options["folder"] == str(target / "사진 예시")


def test_wizard_sample_can_be_saved_without_switching_files(empty_page, folder_dialog, saved_dialog, tmp_path):
    csv = tmp_path / "아무거나.csv"
    csv.write_text("이름,메모\n김하늘,없음\n", encoding="utf-8")
    wizard = empty_page.start_import(csv, run=False)
    assert wizard.wait_idle()
    sample = wizard.get_sample()                                            # 창에서는 '닫기'
    assert sample.workbook_path.is_file()
    assert wizard.title() == "아무거나.csv에서 1명을 찾았습니다"
    wizard.reject()


def test_wizard_explains_a_write_failure(empty_page, saved_dialog, tmp_path):
    csv = tmp_path / "아무거나.csv"
    csv.write_text("이름,메모\n김하늘,없음\n", encoding="utf-8")
    blocker = tmp_path / "파일입니다"
    blocker.write_text("x", encoding="utf-8")
    wizard = empty_page.start_import(csv, run=False)
    assert wizard.wait_idle()
    assert wizard.get_sample(blocker) is None
    assert wizard.file_banner.isVisibleTo(wizard) and wizard.file_banner.kind() == "error"
    assert "저장할 수 없습니다" in wizard.file_banner.text()
    assert wizard.next_button.isEnabled()                                   # 원래 파일은 그대로 가져올 수 있다
    wizard.reject()


def test_saved_dialog_is_built_from_shared_widgets(qapp, tmp_path):
    sample = write_sample(tmp_path)
    dialog = SampleSavedDialog(sample)
    assert isinstance(dialog, Dialog) and dialog.isModal()
    assert "같은 모양으로 우리 명단" in dialog.description()
    assert [dialog.open_button.text(), dialog.close_button.text(), dialog.try_button.text()] == [
        "폴더 열기", "닫기", "이 파일로 바로 해 보기"]
    assert sample_ui.TRY_BUTTON_TEXT == "이 파일로 바로 해 보기"
    dialog.deleteLater()
