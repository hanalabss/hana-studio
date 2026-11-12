"""인쇄 화면: 인쇄 전 확인 결과, 고치는 버튼, 알아 둘 것 확인, 인쇄 설정."""
import os

from print_helpers import *  # noqa: F401,F403
from print_helpers import discover, make_page, wait_until

from design.model import RectItem, TextItem
from ui.shell import PAGE_DESIGN, PAGE_LIST, PAGE_PRINT, PAGE_PRINTER, AppShell, BasePage


class FakeListPage(BasePage):
    def __init__(self):
        super().__init__("명단")
        self.shown = []

    def show_people(self, ids):
        self.shown.append(list(ids))


def in_shell(page):
    shell = AppShell()
    people = FakeListPage()
    shell.register_page(PAGE_LIST, people, "명단", "people")
    shell.register_page(PAGE_PRINT, page, "인쇄", "print")
    shell.navigate(PAGE_PRINT)
    return shell, people


def test_clean_list_shows_ok_and_enables_the_button(env, workspace, services):
    page = make_page(env, workspace, services)
    assert page.problems() == []
    assert [row.problem.code for row in page.ready.finding_rows] == ["ok"]
    assert page.ready.finding_rows[0].property("tone") == "ok"
    assert page.ready.start_button.isEnabled() and not page.ready.reason.isVisibleTo(page.ready)
    assert not page.ready.acknowledge.isVisibleTo(page.ready)
    assert page.ready.printer_name.text() == "RTAI R600 FAKE"
    assert page.ready.printer_pill.state() == "ready" and page.ready.printer_pill.text() == "준비됨"
    assert page.ready.supplies.text().startswith("리본 ")


def test_missing_photo_blocks_until_those_people_are_left_out(env, workspace, services, photo_dir):
    os.remove(photo_dir / "park.png")
    os.remove(photo_dir / "jung.png")
    page = make_page(env, workspace, services)
    ids = workspace.store.ids()
    row = page.ready.finding("missing_photo")
    assert row is not None and row.property("tone") == "error"
    assert "사진이 없는 사람이 2명" in row.text() and "박민수" in row.text()
    assert "'사진' 항목" in row.text() and " 열" not in row.text()
    assert not page.ready.start_button.isEnabled()
    assert page.ready.start_button.text() == "5장 인쇄 시작"
    assert page.ready.reason.text() == "고쳐야 할 것 1개를 먼저 해결해야 인쇄할 수 있습니다."
    assert page.ready.reason.property("tone") == "error"
    assert page.ready.findings_title.text() == "인쇄 전 확인 (고쳐야 할 것 1개)"
    assert page.ready.stat_photo.number.text() == "2명"
    from ui.pages.printing.ready_view import ROLE_TAG

    assert page.ready.people.item(2).data(ROLE_TAG) == "사진 없음"
    assert page.start_printing() is None, "막힌 채로는 시작되지 않는다"

    button = row.button("exclude")
    assert button.text() == "사진 없는 2명 빼고 인쇄"
    button.click()
    assert page.target_ids() == [ids[0], ids[1], ids[3]]
    assert page.ready.finding("missing_photo") is None
    assert page.ready.start_button.isEnabled() and page.ready.start_button.text() == "3장 인쇄 시작"
    assert page.ready.excluded_note.text() == "직접 뺀 사람 2명"
    assert page.ready.stat_photo.number.text() == "0명"


def test_fix_in_list_goes_to_the_list_with_those_people(env, workspace, services, photo_dir):
    os.remove(photo_dir / "lee.png")
    page = make_page(env, workspace, services)
    shell, people = in_shell(page)
    asked = []
    page.showPeopleRequested.connect(asked.append)
    ids = workspace.store.ids()
    page.ready.finding("missing_photo").button("show_people").click()
    assert asked == [[ids[1]]] and people.shown == [[ids[1]]]
    assert shell.current_page_id() == PAGE_LIST
    assert workspace.current_record_id == ids[1]


def test_fix_in_design_selects_the_item(env, workspace, services):
    editor = workspace.editor
    stray = TextItem(name="밖으로 나간 글자", text="밖", x=200, y=200, w=20, h=8)
    editor.document.side("front").add_item(stray)
    editor.document_changed.emit()
    page = make_page(env, workspace, services)
    shell, _people = in_shell(page)
    row = page.ready.finding("item_outside")
    assert row is not None and row.property("tone") == "warn"
    assert "카드 밖에" in row.text()
    row.button("fix_design").click()
    assert editor.selection == [stray.id]
    assert shell.current_page_id() == PAGE_DESIGN


def test_warnings_must_be_acknowledged(env, workspace, services):
    editor = workspace.editor
    editor.document.side("front").add_item(RectItem(name="밖의 상자", x=300, y=10, w=10, h=10))
    editor.document_changed.emit()
    page = make_page(env, workspace, services)
    assert [p.tone for p in page.problems()] == ["warn"]
    box = page.ready.acknowledge
    assert box.isVisibleTo(page.ready) and box.text() == "알아 둘 것 1개를 읽었습니다. 그대로 인쇄합니다."
    assert not page.ready.start_button.isEnabled()
    assert "체크하면 인쇄할 수 있습니다" in page.ready.reason.text()
    assert page.ready.reason.property("tone") == "warn"

    box.click()
    assert page.ready.start_button.isEnabled() and page.ready.reason.text() == ""

    # 알아 둘 것이 달라지면 다시 확인받는다.
    workspace.store.mark_printed(workspace.store.ids()[:1])
    workspace.notify_records_changed()
    page.set_mode("all")
    page.refresh_now()
    assert page.ready.finding("already_printed") is not None
    assert not box.isChecked() and not page.ready.start_button.isEnabled()
    assert box.text().startswith("알아 둘 것 2개")
    page.ready.finding("already_printed").button("exclude").click()
    assert len(page.target_ids()) == 4 and page.ready.finding("already_printed") is None


def test_no_printer_blocks_with_a_way_out(env, workspace, services):
    page = make_page(env, workspace, services, find_printers=False)
    shell, _people = in_shell(page)
    row = page.ready.finding("no_printer")
    assert row is not None and "프린터 전원과 USB 선" in row.text()
    assert not page.ready.start_button.isEnabled()
    assert page.ready.printer_name.text() == "쓸 프린터가 없습니다"
    assert page.ready.printer_pill.state() == "offline"
    row.button("open_printer").click()
    assert shell.current_page_id() == PAGE_PRINTER

    discover(env, services)
    wait_until(env, lambda: page.ready.start_button.isEnabled())
    assert page.ready.finding("no_printer") is None
    assert shell.top_bar.printer_pill.text() == "RTAI R600 FAKE 준비됨"


def test_printer_problem_blocks_and_recheck_clears_it(env, workspace, services, rtai_lib):
    page = make_page(env, workspace, services)
    rtai_lib.feeder_empty = True
    services.printers.poll_now()
    wait_until(env, lambda: page.ready.finding("feeder_empty") is not None)
    row = page.ready.finding("feeder_empty")
    assert "카드를 넣은 뒤" in row.text() and row.property("tone") == "error"
    assert not page.ready.start_button.isEnabled()
    assert page.ready.printer_pill.state() == "error"
    assert "카드가 없습니다" in page.ready.printer_detail.text()

    rtai_lib.feeder_empty = False
    row.button("recheck").click()
    wait_until(env, lambda: page.ready.start_button.isEnabled())
    assert page.ready.finding("feeder_empty") is None and page.ready.printer_pill.state() == "ready"


def test_options_sheet_shows_only_what_the_printer_can_do(env, workspace, services):
    document = workspace.editor.document
    document.dual_side = True
    document.side("back").add_item(TextItem(text="뒷면", x=5, y=5, w=30, h=8))
    document.side("front").add_item(RectItem(name="화이트 바탕", x=5, y=40, w=20, h=8, layer="white"))
    workspace.editor.document_changed.emit()
    page = make_page(env, workspace, services)

    sheet = page.make_options_sheet()
    assert sheet.white is not None and sheet.uv is None, "UV로 꾸민 것이 없으면 UV 칸은 없다"
    assert set(sheet.magnetic) == {"track1", "track2", "track3"}
    assert sheet.sides is not None and sheet.sides.current() == "both"
    assert page.ready.options_summary.text() == "한 사람에 1장, 양면, 화이트 포함"

    sheet.copies.set_value(2)
    sheet.sides.set_current("front")
    sheet.white.setChecked(False)
    sheet.magnetic["track2"].setCurrentText("사번")
    sheet.offset_x.set_value(0.5)
    sheet.offset_y.set_value(-0.3)
    sheet.stop_on_error.setChecked(False)
    page.apply_choices(sheet.choices())
    page.refresh_now()
    assert page.ready.options_summary.text() == (
        "한 사람에 2장, 앞면만, 화이트 빼고, 마그네틱 기록, 위치 가로 +0.5 세로 -0.3mm, 실패해도 계속")
    assert page.ready.start_button.text() == "10장 인쇄 시작"
    assert page.prefs.offsets() == (0.5, -0.3), "위치 맞춤은 다음에도 쓰도록 저장한다"
    assert page.report().by_code("no_dual_side") == [] and page.report().by_code("no_white") == []

    # SMART로 바꾸면 화이트 칸이 사라지고, 화이트는 빠진다고 알려 준다.
    smart = next(d for d in services.printers.devices() if d.family == "smart")
    services.printers.select(smart)
    services.printers.poll_now()
    wait_until(env, lambda: not services.printers.is_busy and services.printers.status() is not None)
    page.refresh_now()
    sheet = page.make_options_sheet()
    assert sheet.white is None and set(sheet.magnetic) == {"track1", "track2", "track3"}
    assert page.ready.printer_name.text() == "IDP SMART-51 Card Printer"
    assert page.ready.finding("no_white") is not None
    assert "화이트 빼고" in page.ready.options_summary.text()


def test_choices_reach_the_print_job(env, workspace, services, rtai_lib):
    from print_helpers import finish_job

    document = workspace.editor.document
    document.dual_side = True
    document.side("back").add_item(TextItem(text="뒷면", x=5, y=5, w=30, h=8))
    workspace.editor.document_changed.emit()
    page = make_page(env, workspace, services)
    choices = page.choices().clone()
    choices.front_only = True
    choices.options.magnetic = {"track1": "사번"}
    choices.options.offset_x_mm = 1.0
    page.apply_choices(choices)
    page.refresh_now()
    workspace.set_checked_ids(workspace.store.ids()[:1])
    page.refresh_now()
    handle = page.start_printing()
    assert handle is not None and handle.total == 1
    finish_job(env, page)
    assert handle._render_job.document.dual_side is False, "앞면만 뽑기로 했으면 뒷면은 보내지 않는다"
    assert handle._render_job.options.offset_x_mm == 1.0
    assert rtai_lib.mag_tracks[0] == b"A0001"


def test_preflight_runs_again_when_the_design_changes(env, workspace, services):
    page = make_page(env, workspace, services)
    assert page.ready.start_button.isEnabled()
    name_item = next(item for _, item in workspace.editor.document.iter_items() if item.name == "이름")
    workspace.editor.set_geometry(name_item.id, {"x": 400.0})
    assert not page.ready.start_button.isEnabled(), "다시 확인이 끝날 때까지는 누를 수 없다"
    assert page.ready.reason.text() == "인쇄해도 되는지 확인하고 있습니다."
    wait_until(env, lambda: page.ready.finding("item_outside") is not None)
    assert page.ready.acknowledge.isVisibleTo(page.ready)
