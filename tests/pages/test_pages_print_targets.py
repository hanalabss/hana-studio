"""인쇄 화면: 누구를 뽑을지 고르는 방식, 개수, 순서, 미리 보기."""
from print_helpers import *  # noqa: F401,F403
from print_helpers import make_page, wait_until

from data.filters import Condition
from ui.pages.printing.targets import (
    MODE_ALL, MODE_CHECKED, MODE_COPIES, MODE_CURRENT, MODE_PICKED, MODE_RANGES, MODE_UNPRINTED,
)


def names(page):
    return [page.ready.people.item(i).text() for i in range(page.ready.people.count())]


def test_default_is_everyone_not_printed_yet(env, workspace, services):
    page = make_page(env, workspace, services)
    ids = workspace.store.ids()
    assert page.mode() == MODE_UNPRINTED and page.target_ids() == ids
    assert page.ready.mode_buttons[MODE_UNPRINTED].text() == "아직 안 뽑은 5명 모두"
    assert page.ready.mode_buttons[MODE_UNPRINTED].isChecked()
    assert names(page) == ["김철수", "이영희", "박민수", "최지우", "정하나"]
    assert page.ready.stat_targets.number.text() == "5명"
    assert page.ready.start_button.text() == "5장 인쇄 시작"
    # 체크한 사람도, 보고 있는 사람도 없으면 그 방식은 고를 수 없고 이유가 적혀 있다.
    assert not page.ready.mode_buttons[MODE_CHECKED].isEnabled()
    assert "체크" in page.ready.mode_buttons[MODE_CHECKED].text()
    assert not page.ready.mode_buttons[MODE_CURRENT].isEnabled()
    # 명단 값을 쓰는 디자인이라 '명단 없이'는 보이지 않는다.
    assert page.ready.mode_buttons[MODE_COPIES].isHidden()


def test_unprinted_counts_follow_print_status(env, workspace, services):
    ids = workspace.store.ids()
    workspace.store.mark_printed(ids[:2])
    page = make_page(env, workspace, services)
    assert page.target_ids() == ids[2:]
    assert page.ready.mode_buttons[MODE_UNPRINTED].text() == "아직 안 뽑은 3명 모두"
    assert page.ready.stat_printed.caption.text() == "이미 뽑아서 뺌"
    assert page.ready.stat_printed.number.text() == "2명"
    assert page.ready.start_button.text() == "3장 인쇄 시작"

    workspace.store.mark_printed(ids[2:3])
    workspace.notify_records_changed()
    page.refresh_now()
    assert page.target_ids() == ids[3:]


def test_checked_people_become_the_target_in_list_order(env, workspace, services):
    page = make_page(env, workspace, services)
    ids = workspace.store.ids()
    workspace.set_checked_ids([ids[3], ids[0]])
    page.refresh_now()
    assert page.mode() == MODE_CHECKED
    assert page.ready.mode_buttons[MODE_CHECKED].text() == "선택한 2명"
    assert page.target_ids() == [ids[0], ids[3]]
    assert page.ready.start_button.text() == "2장 인쇄 시작"

    workspace.set_checked_ids([])
    page.refresh_now()
    assert page.mode() == MODE_UNPRINTED and len(page.target_ids()) == 5


def test_current_person_mode(env, workspace, services):
    ids = workspace.store.ids()
    workspace.set_current_record(ids[2])
    page = make_page(env, workspace, services)
    assert page.ready.mode_buttons[MODE_CURRENT].text() == "지금 보는 한 사람 (박민수)"
    page.ready.mode_buttons[MODE_CURRENT].click()
    assert page.mode() == MODE_CURRENT and page.target_ids() == [ids[2]]
    assert page.ready.start_button.text() == "1장 인쇄 시작"
    assert page.ready.preview_name.text() == "박민수"


def test_ranges_resolve_and_errors_are_shown_inline(env, workspace, services):
    page = make_page(env, workspace, services)
    ids = workspace.store.ids()
    page.ready.mode_buttons[MODE_RANGES].click()
    assert page.mode() == MODE_RANGES
    assert page.ready.range_box.isVisibleTo(page.ready)
    page.refresh_now()
    assert not page.ready.start_button.isEnabled()
    assert "번호를 적어" in page.ready.people_empty.text()

    page.ready.range_edit.setText("1,3-4")
    page.refresh_now()
    assert page.target_ids() == [ids[0], ids[2], ids[3]]
    assert page.ready.range_edit.property("invalid") in (None, "false")
    assert page.ready.start_button.text() == "3장 인쇄 시작" and page.ready.start_button.isEnabled()

    page.ready.range_edit.setText("2,9")
    page.refresh_now()
    assert page.target_ids() == []
    assert page.ready.range_edit.property("invalid") == "true"
    hint = page.ready.range_hint.text()
    assert "9번은 범위를 벗어났습니다" in hint and "레코드" not in hint and "5명" in hint
    assert page.ready.range_hint.property("tone") == "error"
    assert not page.ready.start_button.isEnabled()
    assert page.ready.reason.text() and page.ready.reason.isVisibleTo(page.ready)

    page.ready.range_edit.setText("4-2")
    page.refresh_now()
    assert "작은 번호를 먼저" in page.ready.range_hint.text()


def test_ranges_and_all_follow_the_list_view(env, workspace, services):
    workspace.view_filter = Condition("부서", "eq", "개발부")
    page = make_page(env, workspace, services)
    developers = workspace.store.ids(workspace.view_filter)
    assert page.ready.mode_buttons[MODE_ALL].text() == "명단에 보이는 2명 모두"
    page.set_mode(MODE_ALL)
    assert page.target_ids() == developers
    page.set_mode(MODE_RANGES)
    page.set_range_text("2")
    assert page.target_ids() == [developers[1]]
    page.set_range_text("3")
    assert "모두 2명" in page.ready.range_hint.text()


def test_copies_without_a_list(env, services, document):
    from design import presets
    from design.model import TextItem
    from ui.pages.workspace import Workspace

    plain_card = presets.new_document(presets.CR80, name="방문증")
    plain_card.side("front").add_item(TextItem(text="방문", x=5, y=5, w=40, h=10))
    ws = Workspace()
    ws.editor.set_document(plain_card)
    page = make_page(env, ws, services)
    assert page.mode() == MODE_COPIES
    for mode in (MODE_CHECKED, MODE_UNPRINTED, MODE_CURRENT, MODE_RANGES, MODE_ALL, MODE_PICKED):
        assert page.ready.mode_buttons[mode].isHidden()
    assert page.ready.mode_buttons[MODE_COPIES].text() == "명단 없이 같은 카드 1장"
    page.ready.copies_field.set_value(3)
    page.refresh_now()
    assert page.card_count() == 3
    assert page.ready.start_button.text() == "3장 인쇄 시작" and page.ready.start_button.isEnabled()
    assert page.ready.stat_targets.caption.text() == "뽑을 카드"
    assert page.ready.preview.has_image()

    # 명단 값을 쓰는 디자인인데 명단이 없으면 막고, 명단을 열라고 알려 준다.
    ws.editor.set_document(document)
    page.refresh_now()
    row = page.ready.finding("no_data")
    assert row is not None and "명단을 먼저" in row.text()
    assert row.button("open_list").text() == "명단 열기"
    assert not page.ready.start_button.isEnabled()
    ws.close()


def test_copies_per_person_multiply_the_card_count(env, workspace, services):
    page = make_page(env, workspace, services)
    choices = page.choices().clone()
    choices.options.copies = 2
    page.apply_choices(choices)
    page.refresh_now()
    assert page.ready.start_button.text() == "10장 인쇄 시작"
    assert page.ready.options_summary.text().startswith("한 사람에 2장")
    assert page.ready.stat_targets.number.text() == "5명"


def test_remove_reorder_and_restore_people(env, workspace, services):
    page = make_page(env, workspace, services)
    ids = workspace.store.ids()
    page.ready.people.setCurrentRow(1)
    page.ready.remove_button.click()
    page.refresh_now()
    assert page.target_ids() == [ids[0], ids[2], ids[3], ids[4]]
    assert page.ready.excluded_note.text() == "직접 뺀 사람 1명"
    assert page.ready.excluded_row.isVisibleTo(page.ready)
    assert page.ready.start_button.text() == "4장 인쇄 시작"

    page.ready.people.setCurrentRow(3)
    page.ready.up_button.click()
    assert page.target_ids() == [ids[0], ids[2], ids[4], ids[3]]
    assert names(page) == ["김철수", "박민수", "정하나", "최지우"]
    assert page.ready.people.currentRow() == 2

    page.ready.restore_button.click()
    page.refresh_now()
    assert page.target_ids() == ids

    handle = None
    page.ready.people.setCurrentRow(0)
    page.ready.down_button.click()
    assert page.target_ids()[:2] == [ids[1], ids[0]]
    handle = page.start_printing()
    assert handle is not None and handle.record_ids[:2] == [ids[1], ids[0]], "바꾼 순서대로 인쇄한다"
    from print_helpers import finish_job

    finish_job(env, page)


def test_people_handed_over_from_another_page(env, workspace, services):
    page = make_page(env, workspace, services)
    ids = workspace.store.ids()
    assert page.ready.mode_buttons[MODE_PICKED].isHidden()
    page.print_people([ids[4], ids[1]])
    assert page.mode() == MODE_PICKED
    assert page.ready.mode_buttons[MODE_PICKED].text() == "기록에서 고른 2명"
    assert page.target_ids() == [ids[1], ids[4]]


def test_preview_follows_the_person_and_side(env, workspace, services):
    page = make_page(env, workspace, services)
    assert page.ready.preview.has_image()
    assert page.ready.preview_name.text() == "김철수"
    assert page.ready.preview_sub.text() == "영업부 과장", "바코드에 넣는 사번 말고 부서와 직급을 보여 준다"
    assert page.ready.position_label.text() == "1 / 5"
    assert not page.ready.prev_button.isEnabled() and page.ready.next_button.isEnabled()
    assert page.ready.side_control.isHidden(), "한 면짜리 디자인에는 앞·뒤 고르기가 없다"

    page.ready.next_button.click()
    assert page.ready.preview_name.text() == "이영희" and page.ready.position_label.text() == "2 / 5"
    assert page.ready.people.currentRow() == 1
    page.ready.people.setCurrentRow(4)
    assert page.ready.preview_name.text() == "정하나"
    assert not page.ready.next_button.isEnabled()

    workspace.editor.document.dual_side = True
    workspace.editor.document_changed.emit()
    page.refresh_now()
    assert not page.ready.side_control.isHidden()
    before = page.ready.preview._pixmap.toImage()
    page.ready.side_control.set_current("back")
    assert page.ready.preview.has_image()
    assert page.ready.preview._pixmap.toImage() != before


def test_list_changes_are_picked_up_after_a_short_delay(env, workspace, services):
    page = make_page(env, workspace, services)
    ids = workspace.store.ids()
    workspace.store.delete([ids[0]])
    workspace.notify_records_changed()
    assert len(page.target_ids()) == 5, "바로 다시 세지 않고 조금 모았다가 한다"
    wait_until(env, lambda: len(page.target_ids()) == 4)
    wait_until(env, lambda: page.ready.start_button.text() == "4장 인쇄 시작" and page.ready.start_button.isEnabled())
