"""SMART 프린터: 화이트로 찍는 부분이 있는 양면 카드를, 한 면만 찍는 화이트 없는 장치로 뽑는다."""
from app_helpers import *  # noqa: F401,F403
from app_helpers import (
    click, fake_library, find_printers, import_people, pick_template, click_footer, print_page_ready, pump,
    select_printer, wait_done,
)

from printer.base import CAP_DUAL_SIDE, CAP_WHITE
from ui.pages.printing import STATE_RUNNING


def test_smart_single_side_device_warns_in_plain_korean_and_prints_colour_only(
        app_env, app_window, people_csv, photo_folder):
    window, app = app_window, app_env
    find_printers(app, window)
    library = fake_library("smart")
    library.has_flipper = False                      # 뒤집는 장치가 없는(한 면만 찍는) SMART
    device = select_printer(app, window, "smart")
    assert device.family == "smart"
    capabilities = window.services.printers.capabilities()
    assert CAP_DUAL_SIDE not in capabilities and CAP_WHITE not in capabilities

    # 사원증 → 뒷면을 더하고 글자 하나를 화이트로 찍게 한다
    pick_template(window, "employee")
    design = window.page("design")
    editor = window.workspace.editor
    design.add_back_side()
    assert editor.document.dual_side and editor.current_side == "back"
    back_text = editor.add_item("text", pos=(30.0, 20.0))
    editor.set_side("front")
    white = editor.add_item("rect", pos=(20.0, 10.0))
    assert editor.set_layer([white.id], "white")

    click_footer(window, "다음: 명단 넣기")
    import_people(app, window, people_csv, photo_folder)
    click_footer(window, "다음:")
    page = print_page_ready(app, window)

    # 알아 둘 것으로, 쉬운 말로 보인다
    white_row, dual_row = page.ready.finding("no_white"), page.ready.finding("no_dual_side")
    assert white_row is not None and dual_row is not None
    assert not white_row.problem.blocking and not dual_row.problem.blocking
    assert "화이트" in white_row.text() and "컬러만 인쇄" in white_row.text()
    assert "뒷면은 인쇄되지 않고 앞면만" in dual_row.text()
    for row in (white_row, dual_row):
        for word in ("레이어", "레코드", "capab", "dual", "white", "None"):
            assert word not in row.text(), (word, row.text())
    summary = page.ready.options_summary.text() if hasattr(page.ready, "options_summary") else ""
    assert "레이어" not in summary

    click(page.ready.finding("missing_photo").button("exclude"))
    pump(app, 30)
    assert not page.ready.start_button.isEnabled()                  # 알아 둘 것을 읽어야 한다
    assert "알아 둘 것" in page.blocking_reason()
    page.acknowledge_warnings(True)
    assert page.ready.start_button.isEnabled(), page.blocking_reason()
    library.drawn.clear()
    pages_before = library.printed_pages
    click(page.ready.start_button)
    assert page.state() == STATE_RUNNING
    wait_done(app, page)

    assert page.run.summary_headline.text() == "6장을 모두 뽑았습니다"
    # 컬러 한 면만: 카드마다 그림 한 장, 뒷면(page 1)이나 화이트 판은 보내지 않았다
    assert library.printed_pages - pages_before == 6
    assert len(library.drawn) == 6
    assert {page_index for page_index, _panel, _path in library.drawn} == {0}
    assert len({panel for _page, panel, _path in library.drawn}) == 1
    assert back_text.id in {item.id for _side, item in editor.document.iter_items()}   # 디자인은 그대로다
