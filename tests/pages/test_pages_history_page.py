"""기록 화면: 목록, 걸러 보기, 쪽 넘기기, 사람별 결과, 지우기."""
import sqlite3
from datetime import datetime, timedelta

from print_helpers import *  # noqa: F401,F403
from print_helpers import finish_job, make_page

from printer.base import PrinterDevice
from ui.pages.history import HistoryPage
from ui.pages.history.page import period_start
from ui.shell import PAGE_HISTORY, PAGE_LIST, PAGE_PRINT, AppShell, BasePage

DEVICE = PrinterDevice("rtai", "RTAI R600", "usb", "RTAI R600")
SMART = PrinterDevice("smart", "SMART-51", "usb", "SMART-51")


def add_job(history, name, cards, device=DEVICE, days_ago=0, status=None):
    """cards: [(record_id, "success" | "failed" | "cancelled", 이유)]"""
    job_id = history.start_job(name, device, len(cards))
    history.mark_job_started(job_id)
    for index, (record_id, state, reason) in enumerate(cards, start=1):
        history.record_card(job_id, index, state, reason or None, record_id=record_id)
    history.finish_job(job_id, status)
    if days_ago:
        stamp = (datetime.now() - timedelta(days=days_ago)).isoformat(timespec="seconds")
        with sqlite3.connect(history.path) as connection:
            connection.execute("UPDATE jobs SET created_at = ? WHERE id = ?", (stamp, job_id))
    return job_id


def titles(page):
    return [row["name"] for row in page.job_rows()]


def test_empty_state_points_to_the_print_page(env, workspace, services):
    page = HistoryPage(workspace, services)
    assert page.stack.currentWidget() is page.empty
    assert page.empty.title() == "아직 인쇄한 기록이 없습니다"
    assert page.empty.action_button().text() == "인쇄 화면으로 가기"


def test_jobs_are_listed_newest_first_with_plain_results(env, workspace, services, history):
    ids = workspace.store.ids()
    add_job(history, "봄 사원증", [(ids[0], "success", ""), (ids[1], "success", "")])
    add_job(history, "방문증", [(ids[2], "success", ""), (ids[3], "failed", "리본을 다 썼습니다.")])
    add_job(history, "임시 출입증", [(ids[4], "failed", "카드가 걸렸습니다.")], device=SMART)
    add_job(history, "멈춘 인쇄", [(ids[0], "success", ""), (ids[1], "cancelled", "")], status="cancelled")
    page = HistoryPage(workspace, services)
    assert page.stack.currentIndex() == 1
    rows = page.job_rows()
    assert [r["name"] for r in rows] == ["멈춘 인쇄", "임시 출입증", "방문증", "봄 사원증"]
    assert [r["result"] for r in rows] == ["취소", "실패", "일부 실패", "완료"]
    assert [r["tone"] for r in rows] == ["idle", "error", "warn", "ok"]
    assert rows[3]["counts"] == "2장 모두 완료" and rows[2]["counts"] == "2장 중 1장 완료"
    assert rows[1]["printer"] == "SMART-51" and rows[0]["when"].startswith("오늘 ")
    assert page.page_text() == "1–4번째, 모두 4건"
    assert not page.prev_button.isEnabled() and not page.next_button.isEnabled()


def test_period_status_and_search_filters(env, workspace, services, history):
    ids = workspace.store.ids()
    add_job(history, "작년 사원증", [(ids[0], "success", "")], days_ago=400)
    add_job(history, "지난달 방문증", [(ids[1], "failed", "카드가 걸렸습니다.")], days_ago=40)
    add_job(history, "오늘 사원증", [(ids[2], "success", "")])
    add_job(history, "오늘 방문증", [(ids[3], "success", ""), (ids[4], "failed", "리본")])
    page = HistoryPage(workspace, services)
    assert len(page.job_rows()) == 4

    page.period_chips["today"].click()
    assert titles(page) == ["오늘 방문증", "오늘 사원증"]
    page.period_chips["month"].click()
    assert set(titles(page)) >= {"오늘 방문증", "오늘 사원증"} and "작년 사원증" not in titles(page)
    page.period_chips["all"].click()
    assert len(page.job_rows()) == 4

    page.status_combo.setCurrentIndex(2)
    assert page.status_combo.currentText() == "일부 실패" and titles(page) == ["오늘 방문증"]
    page.status_combo.setCurrentIndex(3)
    assert titles(page) == ["지난달 방문증"]
    page.status_combo.setCurrentIndex(0)

    page.set_search("방문증")
    assert titles(page) == ["오늘 방문증", "지난달 방문증"]
    # 사람 이름으로도 찾는다(지금 열린 명단에서 그 사람을 찾아 그 사람 카드가 든 인쇄를 보여 준다).
    page.set_search("박민수")
    assert titles(page) == ["오늘 사원증"]
    page.set_search("없는말")
    assert page.job_rows() == [] and page.no_match.isVisibleTo(page)
    assert page.page_text() == "모두 0건"
    page.clear_filters_button.click()
    assert len(page.job_rows()) == 4 and page.search.text() == ""

    assert period_start("week", datetime(2026, 11, 12, 15, 0)) == "2026-11-09T00:00:00"
    assert period_start("month", datetime(2026, 11, 12, 15, 0)) == "2026-11-01T00:00:00"
    assert period_start("all") is None


def test_paging(env, workspace, services, history):
    for number in range(1, 6):
        add_job(history, f"인쇄 {number}", [(None, "success", "")])
    page = HistoryPage(workspace, services, page_size=2)
    assert titles(page) == ["인쇄 5", "인쇄 4"] and page.page_text() == "1–2번째, 모두 5건"
    assert not page.prev_button.isEnabled() and page.next_button.isEnabled()
    page.next_button.click()
    assert titles(page) == ["인쇄 3", "인쇄 2"] and page.page_text() == "3–4번째, 모두 5건"
    page.next_button.click()
    assert titles(page) == ["인쇄 1"] and not page.next_button.isEnabled() and page.prev_button.isEnabled()
    page.prev_button.click()
    assert titles(page) == ["인쇄 3", "인쇄 2"]
    page.set_search("인쇄 1")
    assert titles(page) == ["인쇄 1"] and page.page_text() == "1–1번째, 모두 1건"


def test_detail_shows_each_person_and_offers_follow_ups(env, workspace, services, history):
    ids = workspace.store.ids()
    job_id = add_job(history, "봄 사원증", [
        (ids[0], "success", ""), (ids[1], "failed", "리본을 다 썼습니다. [프린터 메시지: Ribbon is used up]"),
        (ids[2], "cancelled", ""),
    ])
    page = HistoryPage(workspace, services)
    assert page.detail_stack.currentIndex() == 0
    assert page.select_job(job_id)
    assert page.detail_stack.currentIndex() == 1
    assert page.detail_title.text() == "봄 사원증" and page.detail_tag.text() == "일부 실패"
    assert page.detail_counts.text() == "3장 중 1장 완료, 1장 실패"
    assert "RTAI R600 (RTAI)" in page.detail_meta.text()
    cards = page.card_rows()
    assert [c["person"] for c in cards] == ["김철수", "이영희", "박민수"]
    assert [c["result"] for c in cards] == ["끝남", "실패", "취소됨"]
    assert cards[1]["reason"] == "리본을 다 썼습니다." and cards[0]["reason"] == ""
    assert cards[0]["time"]
    assert page.show_failed_button.text() == "실패한 2명만 명단에서 보기"
    assert page.reprint_button.text() == "실패한 2명 다시 인쇄"
    assert "지금 열려 있는 명단" in page.names_note.text()

    shown, reprint = [], []
    page.showPeopleRequested.connect(shown.append)
    page.reprintRequested.connect(reprint.append)
    page.show_failed_button.click()
    page.reprint_button.click()
    assert shown == [[ids[1], ids[2]]] and reprint == [[ids[1], ids[2]]]

    # 명단을 닫으면 이름을 알 수 없고 다시 인쇄도 할 수 없다.
    workspace.set_store(None)
    assert [c["person"] for c in page.card_rows()] == ["명단에 없음"] * 3
    assert not page.reprint_button.isVisibleTo(page) and not page.show_failed_button.isVisibleTo(page)
    assert "명단을 열면" in page.names_note.text()


def test_follow_ups_reach_the_list_and_print_pages(env, workspace, services, history):
    class ListPage(BasePage):
        def __init__(self):
            super().__init__("명단")
            self.shown = []

        def show_people(self, ids):
            self.shown.append(list(ids))

    ids = workspace.store.ids()
    job_id = add_job(history, "봄 사원증", [(ids[0], "success", ""), (ids[3], "failed", "카드 걸림")])
    shell = AppShell()
    people = ListPage()
    print_page = make_page(env, workspace, services)
    page = HistoryPage(workspace, services)
    shell.register_page(PAGE_LIST, people, "명단", "people")
    shell.register_page(PAGE_PRINT, print_page, "인쇄", "print")
    shell.register_page(PAGE_HISTORY, page, "기록", "history")
    shell.navigate(PAGE_HISTORY)
    page.select_job(job_id)

    page.show_failed_button.click()
    assert people.shown == [[ids[3]]] and shell.current_page_id() == PAGE_LIST
    shell.navigate(PAGE_HISTORY)
    page.reprint_button.click()
    assert shell.current_page_id() == PAGE_PRINT
    assert print_page.mode() == "picked" and print_page.target_ids() == [ids[3]]
    assert print_page.ready.start_button.text() == "1장 인쇄 시작"


def test_delete_and_purge_ask_first(env, workspace, services, history):
    old = add_job(history, "오래된 인쇄", [(None, "success", "")], days_ago=45)
    add_job(history, "더 오래된 인쇄", [(None, "success", "")], days_ago=90)
    recent = add_job(history, "어제 인쇄", [(None, "success", "")], days_ago=1)
    page = HistoryPage(workspace, services)
    asked = []
    answers = []

    def fake_confirm(parent, title, message, **kwargs):
        asked.append((title, message, kwargs))
        return answers.pop(0)

    page.confirm = fake_confirm
    page.select_job(recent)
    answers.append(False)
    assert page.delete_selected() is False and history.get_job(recent) is not None
    assert asked[-1][0] == "이 기록을 지울까요?" and asked[-1][2]["danger"] is True
    assert "되돌릴 수 없습니다" in asked[-1][1]
    answers.append(True)
    page.delete_button.click()
    assert history.get_job(recent) is None and page.selected_job_id() is None
    assert page.detail_stack.currentIndex() == 0

    answers.append(False)
    assert page.purge_old() == 0 and history.get_job(old) is not None
    assert "2건" in asked[-1][1] and asked[-1][2]["confirm_text"] == "2건 지우기"
    answers.append(True)
    page.purge_button.click()
    assert history.count_jobs() == 0
    assert page.stack.currentWidget() is page.empty

    asked.clear()
    assert page.purge_old() == 0 and asked == [], "지울 것이 없으면 묻지 않는다"


def test_a_real_print_shows_up_with_names(env, workspace, services, history):
    print_page = make_page(env, workspace, services)
    page = HistoryPage(workspace, services)
    assert page.stack.currentWidget() is page.empty
    workspace.set_checked_ids(workspace.store.ids()[:2])
    print_page.refresh_now()
    print_page.start_printing()
    finish_job(env, print_page)
    assert page.stack.currentIndex() == 1, "인쇄가 끝나면 목록이 스스로 새로 읽힌다"
    assert titles(page) == [workspace.editor.document.name]
    page.show_latest()
    assert [c["person"] for c in page.card_rows()] == ["김철수", "이영희"]
    assert page.reprint_button.text() == "이 2명 다시 인쇄"
    assert not page.show_failed_button.isVisibleTo(page)
