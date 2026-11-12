"""인쇄 화면: 가짜 프린터로 처음부터 끝까지."""
from print_helpers import *  # noqa: F401,F403
from print_helpers import Gate, finish_job, make_page, pump, wait_until

from printer.fakes import R600_ERR_PRINT
from ui.pages.printing import STATE_DONE, STATE_READY, STATE_RUNNING
from ui.pages.printing.run_view import ROW_CANCELLED, ROW_FAILED, ROW_SKIPPED, ROW_SUCCESS


def test_full_print_updates_rows_list_summary_and_history(env, workspace, services, history):
    page = make_page(env, workspace, services)
    changes = []
    workspace.records_changed.connect(lambda: changes.append(1))
    assert page.state() == STATE_READY
    assert page.ready.start_button.text() == "5장 인쇄 시작"
    assert page.ready.start_button.isEnabled(), page.blocking_reason()

    handle = page.start_printing()
    assert handle is not None and page.state() == STATE_RUNNING
    assert page.stack.currentWidget() is page.run
    assert page.run.title.text() == "이번에 뽑는 5명"
    finish_job(env, page)

    assert [page.run.row_state(i) for i in range(5)] == [ROW_SUCCESS] * 5
    assert page.run.row_status(0) == "끝남"
    assert page.run.summary_headline.text() == "5장을 모두 뽑았습니다"
    assert not page.run.retry_button.isVisibleTo(page.run) and not page.run.continue_button.isVisibleTo(page.run)
    assert changes, "명단 화면이 인쇄 상태를 다시 읽도록 알려야 한다"
    assert all(record["_printed"] for record in workspace.store.rows())
    jobs = history.list_jobs()
    assert len(jobs) == 1 and jobs[0].status == "completed" and jobs[0].success_cards == 5
    assert len(history.list_cards(jobs[0].id)) == 5

    page.back_to_ready()
    assert page.state() == STATE_READY
    # 모두 뽑았으니 '아직 안 뽑은 사람'은 고를 수 없고 명단 전체가 기본이 된다.
    assert not page.ready.mode_buttons["unprinted"].isEnabled()
    assert page.mode() == "all"
    assert page.ready.finding("already_printed") is not None


def test_mid_job_failure_then_retry_prints_only_failed(env, workspace, services, rtai_lib, history):
    page = make_page(env, workspace, services)
    choices = page.choices().clone()
    choices.options.stop_on_error = False
    page.apply_choices(choices)
    page.refresh_now()
    rtai_lib.fail("R600PrintDraw", R600_ERR_PRINT, on_call=3)

    page.start_printing()
    finish_job(env, page)
    states = [page.run.row_state(i) for i in range(5)]
    assert states == [ROW_SUCCESS, ROW_SUCCESS, ROW_FAILED, ROW_SUCCESS, ROW_SUCCESS]
    assert page.run.row_status(2).startswith("실패: ")
    assert page.run.summary_headline.text() == "5장 중 4장을 뽑았습니다"
    assert len(page.run.failure_texts()) == 1 and page.run.failure_texts()[0].startswith("박민수")
    assert page.run.retry_button.text() == "실패한 1장 다시 인쇄"
    assert page.run.retry_button.isVisibleTo(page.run)
    assert not page.run.continue_button.isVisibleTo(page.run)
    printed_before = rtai_lib.printed_cards

    retry = page.retry_failed()
    assert retry is not None and retry.total == 1
    assert page.run.title.text() == "다시 뽑는 1장"
    assert page.run.model.rowCount() == 1 and page.run.model.row(0)["name"] == "박민수"
    finish_job(env, page)
    assert rtai_lib.printed_cards == printed_before + 1
    assert page.run.summary_headline.text() == "1장을 모두 뽑았습니다"
    assert all(record["_printed"] for record in workspace.store.rows())
    assert [job.status for job in history.list_jobs()] == ["completed", "partial"]


def test_failure_stops_by_default_and_remaining_can_continue(env, workspace, services, rtai_lib):
    page = make_page(env, workspace, services)
    rtai_lib.fail("R600PrintDraw", R600_ERR_PRINT, on_call=2)
    page.start_printing()
    finish_job(env, page)
    states = [page.run.row_state(i) for i in range(5)]
    assert states == [ROW_SUCCESS, ROW_FAILED, ROW_SKIPPED, ROW_SKIPPED, ROW_SKIPPED]
    assert page.run.row_status(3) == "뽑지 않음"
    assert page.run.summary_counts["left"].text() == "3장"
    assert page.run.continue_button.text() == "실패한 카드까지 4장 이어서 인쇄"

    again = page.continue_remaining()
    assert again is not None and again.total == 4
    finish_job(env, page)
    assert page.run.summary_headline.text() == "4장을 모두 뽑았습니다"
    assert all(record["_printed"] for record in workspace.store.rows())


def test_stop_after_current_card(env, workspace, services):
    from printer import registry

    gate = Gate(registry, hold_on=2)
    page = make_page(env, workspace, services)
    page.start_printing()
    wait_until(env, gate.entered.is_set)
    assert page.stop_after_current() is True
    assert not page.run.stop_after_button.isEnabled()
    assert "끝나면 멈춥니다" in page.run.stop_note.text()
    gate.release.set()
    finish_job(env, page)
    states = [page.run.row_state(i) for i in range(5)]
    assert states == [ROW_SUCCESS, ROW_SUCCESS, ROW_CANCELLED, ROW_CANCELLED, ROW_CANCELLED]
    assert page.run.summary_headline.text() == "인쇄를 멈췄습니다"
    assert page.run.continue_button.text() == "남은 3장 이어서 인쇄"
    assert not page.run.retry_button.isVisibleTo(page.run)
    assert [r["_printed"] for r in workspace.store.rows()] == [True, True, False, False, False]


def test_stop_now_asks_first_and_explains_card_inside(env, workspace, services):
    from printer import registry

    gate = Gate(registry, hold_on=2)
    page = make_page(env, workspace, services)
    asked = []
    answers = [False, True]

    def fake_confirm(parent, title, message, **kwargs):
        asked.append((title, message, kwargs))
        return answers.pop(0)

    page.confirm = fake_confirm
    page.start_printing()
    wait_until(env, gate.entered.is_set)
    assert page.stop_now() is False, "'계속 인쇄'를 고르면 멈추지 않는다"
    assert page.run.stop_now_button.isEnabled()
    assert page.stop_now() is True
    assert "끝까지 찍혀" in asked[0][1] and asked[0][2]["confirm_text"] == "지금 멈추기"
    gate.release.set()
    finish_job(env, page)
    states = [page.run.row_state(i) for i in range(5)]
    assert states[0] == ROW_SUCCESS and states[1] == ROW_SUCCESS, "이미 들어간 카드는 끝까지 찍힌다"
    assert set(states[2:]) == {ROW_CANCELLED}
    assert page.job().summary.state == "cancelled"


def test_printing_state_survives_navigating_away_and_back(env, workspace, services):
    from printer import registry
    from ui.shell import PAGE_HISTORY, PAGE_PRINT, AppShell

    gate = Gate(registry, hold_on=3)
    shell = AppShell()
    page = make_page(env, workspace, services)
    shell.register_page(PAGE_PRINT, page, "인쇄", "print")
    shell.navigate(PAGE_PRINT)
    page.start_printing()
    wait_until(env, gate.entered.is_set)
    assert shell.top_bar.printer_pill.state() == "busy"
    assert shell.top_bar.printer_pill.text().endswith("인쇄 중")

    shell.navigate(PAGE_HISTORY)
    pump(env, 50)
    shell.navigate(PAGE_PRINT)
    assert page.state() == STATE_RUNNING and page.stack.currentWidget() is page.run
    assert page.run.count_done.text() == "2"
    assert page.run.row_state(0) == ROW_SUCCESS and page.run.row_state(2) == "printing"

    asked = []
    page.confirm = lambda *args, **kwargs: asked.append(args[1]) or False
    assert page.can_close() is False and asked == ["인쇄 중인데 닫을까요?"]

    shell.navigate(PAGE_HISTORY)
    gate.release.set()
    finish_job(env, page)
    shell.navigate(PAGE_PRINT)
    assert page.state() == STATE_DONE and page.run.summary_headline.text() == "5장을 모두 뽑았습니다"
    assert shell.guide.is_done("print")
    assert shell.top_bar.printer_pill.state() != "busy"
    assert page.can_close() is True


def test_eta_appears_after_two_cards(env, workspace, services):
    from printer import registry

    gate = Gate(registry, hold_on=3)
    page = make_page(env, workspace, services)
    page.start_printing()
    wait_until(env, gate.entered.is_set)
    pump(env, 30)
    assert page.run.eta.text() in ("1분 안에 끝납니다",) or "남음" in page.run.eta.text()
    assert page.run.current_name.text() == "박민수"
    assert page.run.preview.has_image()
    assert page.run.supply_labels["ribbon"].text() == "남아 있음 (프린터 표시값 250)"
    gate.release.set()
    finish_job(env, page)
    assert page.run.eta.text() == ""


def test_next_job_can_be_queued_while_printing(env, workspace, services):
    """인쇄가 도는 동안 다음 작업을 걸어 두면, 앞 작업이 끝난 뒤 이어서 뽑는다(기존 화면의 대기열)."""
    from printer import registry
    from ui.pages.printing import STATE_DONE, STATE_RUNNING

    gate = Gate(registry, hold_on=2)
    page = make_page(env, workspace, services)
    ids = workspace.store.ids()
    workspace.set_checked_ids(ids[:3])
    page.refresh_now()
    first = page.start_printing()
    wait_until(env, gate.entered.is_set)
    assert page.queue_bar.isVisibleTo(page) and page.queue_add_button.isVisibleTo(page)

    assert page.compose_next() is True
    assert page.stack.currentWidget() is page.ready
    workspace.set_checked_ids(ids[3:])
    page.refresh_now()
    assert "다음 작업으로 걸기" in page.ready.start_button.text()
    second = page.start_printing()
    assert second is not None and second.id != first.id
    assert page.waiting_jobs() == [second]
    assert page.stack.currentWidget() is page.run and page.state() == STATE_RUNNING
    assert "1개" in page.queue_label.text()

    gate.release.set()
    wait_until(env, lambda: page.job() is second)
    wait_until(env, lambda: page.state() == STATE_DONE)
    assert page.waiting_jobs() == []
    assert not page.queue_bar.isVisibleTo(page)
    assert all(r["_printed"] for r in workspace.store.rows())
    assert first.success_count == 3 and second.success_count == len(ids) - 3


def test_queued_job_can_be_removed_before_it_starts(env, workspace, services):
    from printer import registry

    gate = Gate(registry, hold_on=1)
    page = make_page(env, workspace, services)
    ids = workspace.store.ids()
    workspace.set_checked_ids(ids[:2])
    page.refresh_now()
    page.start_printing()
    wait_until(env, gate.entered.is_set)
    page.compose_next()
    workspace.set_checked_ids(ids[2:])
    page.refresh_now()
    second = page.start_printing()
    assert page.cancel_waiting() is True and page.waiting_jobs() == []
    gate.release.set()
    finish_job(env, page)
    printed = [r["_printed"] for r in workspace.store.rows()]
    assert printed[:2] == [True, True] and not any(printed[2:])
    assert not second.is_active
