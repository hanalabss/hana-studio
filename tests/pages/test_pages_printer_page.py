"""프린터 화면과 위쪽 줄 프린터 알약."""
from print_helpers import *  # noqa: F401,F403
from print_helpers import Gate, discover, finish_job, make_page, wait_until

from printer import registry
from ui.pages.printer import PrinterPage
from ui.pages.printer.page import checklist_for
from ui.shell import PAGE_PRINT, PAGE_PRINTER, AppShell


def make_printer_page(env, services, find=True):
    if find:
        discover(env, services)
    page = PrinterPage(services)
    page.resize(1100, 700)
    return page


def test_nothing_found_shows_a_checklist_and_raw_detail(env, services, monkeypatch):
    lines = ["RTAI: 연결된 프린터가 없습니다.", "SMART: 라이브러리(SmartComm2.dll)를 찾을 수 없습니다."]
    monkeypatch.setattr(registry, "discover",
                        lambda families=registry.ALL_FAMILIES: registry.DiscoveryResult([], lines))
    page = make_printer_page(env, services, find=False)
    assert page.searching_panel.isHidden() is False or page.none_panel.isHidden() is False
    services.printers.refresh()
    wait_until(env, lambda: not services.printers.is_busy)
    assert page.cards == []
    assert page.none_panel.isVisibleTo(page) and not page.searching_panel.isVisibleTo(page)
    assert page.none_panel.title() == "프린터를 찾지 못했습니다"
    assert page.summary.text() == "찾은 프린터가 없습니다"
    texts = page.checklist_texts()
    assert "전원" in texts[0] and "USB 선" in texts[1]
    assert "제조사의 드라이버를 먼저 설치해 주세요" in texts[2]
    assert any("SMART 프린터를 쓰는 데 필요한 프로그램 파일" in t for t in texts)
    assert not any("RTAI 프린터를 쓰는 데" in t for t in texts)
    assert "다시 찾기" in texts[-1]

    assert not page.raw_detail.isVisibleTo(page)
    page.detail_toggle.click()
    assert page.raw_detail.isVisibleTo(page) and page.detail_toggle.text() == "자세한 내용 접기"
    assert page.raw_detail.toPlainText().splitlines() == lines
    page.detail_toggle.click()
    assert not page.raw_detail.isVisibleTo(page)
    assert page.refresh_button.isEnabled()


def test_checklist_wording():
    base = checklist_for([])
    assert len(base) == 4
    both = checklist_for(["RTAI: 라이브러리(libDSRetransfer600App.dll)를 찾을 수 없습니다.",
                          "SMART: 탐지 중 오류가 발생했습니다 - OSError: x"])
    assert any(t.startswith("RTAI 프린터를 쓰는 데") for t in both)
    assert any("껐다가" in t for t in both)


def test_fake_devices_list_both_families_with_plain_abilities(env, services):
    page = make_printer_page(env, services)
    assert [card.device.family for card in page.cards] == ["rtai", "smart"]
    assert page.summary.text() == "찾은 프린터 2대 (RTAI, SMART)"
    assert not page.none_panel.isVisibleTo(page)
    rtai, smart = page.cards
    assert rtai.family_tag.text() == "RTAI" and smart.family_tag.text() == "SMART"
    assert rtai.ability_words() == ["양면", "화이트", "UV", "마그네틱"]
    assert smart.ability_words() == ["양면", "마그네틱"]
    assert rtai.connection.text() == "연습용 가짜 프린터"
    assert rtai.current_tag.isVisibleTo(page) and not smart.current_tag.isVisibleTo(page)
    assert not rtai.use_button.isVisibleTo(page) and smart.use_button.isVisibleTo(page)
    assert rtai.state_tag.text() == "준비됨" and rtai.status_text.text() == "바로 인쇄할 수 있습니다."
    assert rtai.supplies.text().startswith("리본 ") and "필름 " in rtai.supplies.text()
    assert "쓰기로 하면" in smart.status_text.text()

    smart.use_button.click()
    wait_until(env, lambda: not services.printers.is_busy and services.printers.status() is not None)
    assert services.printers.current().family == "smart"
    assert smart.current_tag.isVisibleTo(page) and not rtai.current_tag.isVisibleTo(page)
    assert smart.supplies.text() == "리본 약 180장 남음", "장수로 알려 주는 프린터만 '약 N장'이라고 말한다"
    assert smart.property("current") == "true" and rtai.property("current") == "false"


def test_refresh_and_actions_are_disabled_while_printing(env, workspace, services):
    gate = Gate(registry, hold_on=2)
    print_page = make_page(env, workspace, services)
    page = make_printer_page(env, services, find=False)
    assert page.refresh_button.isEnabled() and page.cards[0].probe_button.isEnabled()

    print_page.start_printing()
    wait_until(env, gate.entered.is_set)
    assert not page.refresh_button.isEnabled()
    assert "인쇄 중에는" in page.refresh_button.toolTip()
    assert page.refresh_reason.isVisibleTo(page) and "인쇄가 끝나면" in page.refresh_reason.text()
    for card in page.cards:
        assert not card.probe_button.isEnabled() and not card.test_button.isEnabled()
        assert not card.use_button.isEnabled()
    assert not page.fake_check.isEnabled()
    assert page.cards[0].state_tag.text() == "인쇄 중"
    assert page.refresh_devices() is False and page.probe_device(page.cards[0].device) is False
    assert page.test_print(page.cards[0].device) is False

    gate.release.set()
    finish_job(env, print_page)
    wait_until(env, lambda: page.refresh_button.isEnabled())
    assert not page.refresh_reason.isVisibleTo(page)
    assert page.cards[0].probe_button.isEnabled() and page.fake_check.isEnabled()


def test_probe_reports_without_using_a_card(env, services, rtai_lib):
    page = make_printer_page(env, services)
    card = page.cards[0]
    before = rtai_lib.printed_cards
    assert page.probe_device(card.device) is True
    assert not page.refresh_button.isEnabled(), "점검하는 동안은 다시 찾지 않는다"
    wait_until(env, lambda: not page.is_probing())
    texts = card.result_texts()
    assert texts[0] == "연결: 프린터와 잘 이어져 있습니다."
    assert "상태: 바로 인쇄할 수 있습니다." in texts
    assert "카드 넣는 곳: 카드가 있습니다." in texts
    assert any(t.startswith("리본: ") for t in texts) and any(t.startswith("기기 번호: ") for t in texts)
    assert card.results.isVisibleTo(page)
    assert rtai_lib.printed_cards == before and "R600CardInject" not in rtai_lib.call_names()

    rtai_lib.feeder_empty = True
    wait_until(env, lambda: not services.printers.is_busy)
    page.probe_device(card.device)
    wait_until(env, lambda: not page.is_probing())
    assert "카드 넣는 곳: 비어 있습니다. 카드를 채워 주세요." in card.result_texts()
    assert [line.tone for line in card.result_lines if line.title == "카드 넣는 곳"] == ["error"]


def test_test_print_is_guarded_and_uses_one_card(env, services, rtai_lib, history):
    page = make_printer_page(env, services)
    card = page.cards[0]
    asked = []
    answers = [False, True]

    def fake_confirm(parent, title, message, **kwargs):
        asked.append((title, message, kwargs))
        return answers.pop(0)

    page.confirm = fake_confirm
    card.test_button.click()
    assert asked[0][0] == "시험 인쇄를 할까요?" and "카드 1장과 리본" in asked[0][1]
    assert asked[0][2]["confirm_text"] == "카드 1장 쓰고 시험 인쇄"
    assert rtai_lib.printed_cards == 0 and not services.printing.is_printing

    card.test_button.click()
    assert services.printing.is_printing
    wait_until(env, lambda: not services.printing.is_printing)
    assert rtai_lib.printed_cards == 1
    assert card.result_lines[0].tone == "ok" and "카드가 나왔습니다" in card.result_lines[0].text
    jobs = history.list_jobs()
    assert [(job.name, job.total_cards, job.status) for job in jobs] == [("시험 인쇄", 1, "completed")]


def test_test_print_failure_says_what_happened(env, services, rtai_lib):
    from printer.fakes import R600_ERR_PRINT

    page = make_printer_page(env, services)
    page.confirm = lambda *args, **kwargs: True
    rtai_lib.fail("R600PrintDraw", R600_ERR_PRINT)
    card = page.cards[0]
    assert page.test_print(card.device) is True
    wait_until(env, lambda: not services.printing.is_printing)
    assert card.result_lines[0].tone == "error"
    assert card.result_lines[0].text.startswith("카드를 뽑지 못했습니다. ")
    assert "Print failed" not in card.result_lines[0].text


def test_practice_mode_switch(env, services):
    page = make_printer_page(env, services)
    assert page.fake_check.isChecked() and page.fake_check.text() == "연습 모드 (가짜 프린터)"
    calls = []

    def switch(enabled):
        calls.append(enabled)
        registry.use_fakes(enabled)      # 실제 프린터를 찾으러 가지는 않게 깃발만 바꾼다

    services.printers.set_use_fakes = switch
    page.fake_check.click()
    assert calls == [False]
    page.fake_check.click()
    assert calls == [False, True]


def test_shell_pill_and_popover_follow_the_printer(env, workspace, services):
    shell = AppShell()
    page = PrinterPage(services)
    shell.register_page(PAGE_PRINTER, page, "프린터", "printer")
    assert shell.top_bar.printer_pill.state() == "offline"
    discover(env, services)
    pill = shell.top_bar.printer_pill
    assert pill.state() == "ready" and pill.text() == "RTAI R600 FAKE 준비됨"
    popover = shell.printer_popover
    assert popover.title() == "RTAI R600 FAKE 준비됨" and popover.detail() == "바로 인쇄할 수 있습니다."
    content = shell._hana_printer_binder.content
    assert content.supplies.text().startswith("리본 ")
    assert [b.text() for b in content.switch_buttons] == ["IDP SMART-51 Card Printer 쓰기"]

    content.switch_buttons[0].click()
    wait_until(env, lambda: not services.printers.is_busy and services.printers.status() is not None)
    assert pill.text() == "IDP SMART-51 Card Printer 준비됨"
    content = shell._hana_printer_binder.content
    assert [b.text() for b in content.switch_buttons] == ["RTAI R600 FAKE 쓰기"]
    assert content.supplies.text() == "리본 약 180장 남음"

    # 인쇄 화면이 함께 올라가도 알약은 한 번만 이어진다.
    binder = shell._hana_printer_binder
    shell.register_page(PAGE_PRINT, make_page(env, workspace, services, find_printers=False), "인쇄", "print")
    assert shell._hana_printer_binder is binder


def test_printer_error_reaches_the_pill(env, services, rtai_lib):
    shell = AppShell()
    shell.register_page(PAGE_PRINTER, PrinterPage(services), "프린터", "printer")
    discover(env, services)
    rtai_lib.feeder_empty = True
    services.printers.poll_now()
    pill = shell.top_bar.printer_pill
    wait_until(env, lambda: pill.state() == "error")
    assert pill.text() == "RTAI R600 FAKE 확인 필요"
    assert "카드가 없습니다" in shell.printer_popover.detail()
    page = shell.page(PAGE_PRINTER)
    assert page.cards[0].state_tag.text() == "확인 필요" and "카드가 없습니다" in page.cards[0].status_text.text()
