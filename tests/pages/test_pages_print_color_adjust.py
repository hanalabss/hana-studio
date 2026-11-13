"""인쇄 설정의 '색 맞추기': 지원하는 프린터에서만 보이고, 고른 값이 저장되고 프린터까지 간다."""
from print_helpers import *  # noqa: F401,F403
from print_helpers import finish_job, make_page, wait_until

from services.render_job import PrintOptions
from ui.pages.settings import Prefs

VISUAL = "R600SetImgVisualParam"


def test_color_fields_show_for_rtai_and_carry_the_saved_values(env, workspace, services):
    page = make_page(env, workspace, services)
    sheet = page.make_options_sheet()
    assert list(sheet.color_fields) == ["brightness", "contrast", "saturation", "gamma"]
    assert [field.value() for field in sheet.color_fields.values()] == [0, 0, 0, 0]

    sheet.color_fields["brightness"].set_value(10)
    sheet.color_fields["contrast"].set_value(-5)
    sheet.color_fields["saturation"].set_value(20)
    page.apply_choices(sheet.choices())
    page.refresh_now()
    assert "색 조정 밝기 +10 명암 -5 채도 +20" in page.ready.options_summary.text()
    assert page.prefs.color_adjust() == (10, -5, 20), "다음 인쇄에도 쓰도록 저장한다"
    assert Prefs().print_options().saturation == 20

    again = page.make_options_sheet()
    assert again.color_fields["brightness"].value() == 10
    again.reset_color.click()
    page.apply_choices(again.choices())
    assert page.prefs.color_adjust() == (0, 0, 0)


def test_color_fields_show_for_smart_too(env, workspace, services):
    page = make_page(env, workspace, services)
    smart = next(d for d in services.printers.devices() if d.family == "smart")
    services.printers.select(smart)
    services.printers.poll_now()
    wait_until(env, lambda: not services.printers.is_busy and services.printers.status() is not None)
    page.refresh_now()
    sheet = page.make_options_sheet()
    assert list(sheet.color_fields) == ["brightness", "contrast", "saturation"], "중간 밝기는 RTAI에만 있다"
    sheet.color_fields["saturation"].set_value(15)
    page.apply_choices(sheet.choices())
    page.refresh_now()
    assert "색 조정 밝기 +0 명암 +0 채도 +15" in page.ready.options_summary.text()


def test_color_fields_are_hidden_when_the_printer_cannot_do_it(env):
    from design.model import Document
    from ui.pages.printing.options_sheet import OptionsSheet, PrintChoices, summary_text

    choices = PrintChoices(options=PrintOptions(brightness=10))
    sheet = OptionsSheet(choices, Document(), set(), [], people=False)
    assert sheet.color_fields == {} and sheet.density_fields == {}
    assert "색 조정" not in summary_text(choices, Document(), set(), False)


def test_stored_values_outside_the_range_are_clamped(env):
    prefs = Prefs()
    prefs.set_color_adjust(300, -300, 5)
    assert prefs.color_adjust() == (100, -100, 5)


def test_options_survive_a_save_and_load():
    options = PrintOptions(brightness=5, contrast=-15, saturation=30)
    loaded = PrintOptions.from_dict(options.to_dict())
    assert (loaded.brightness, loaded.contrast, loaded.saturation) == (5, -15, 30)
    assert PrintOptions.from_dict({}).saturation == 0, "예전 프로젝트 파일에는 값이 없다"


def test_chosen_values_reach_the_printer(env, workspace, services, rtai_lib):
    page = make_page(env, workspace, services)
    choices = page.choices().clone()
    choices.options.brightness, choices.options.saturation = 12, 34
    page.apply_choices(choices)
    page.refresh_now()
    workspace.set_checked_ids(workspace.store.ids()[:1])
    page.refresh_now()
    rtai_lib.reset_calls()
    handle = page.start_printing()
    assert handle is not None
    finish_job(env, page)
    sent = [call.args for call in rtai_lib.calls_named(VISUAL)]
    assert sent[0] == (12, 0, 34) and sent[-1] == (0, 0, 0)


def test_gamma_is_saved_and_shown_for_rtai(env, workspace, services):
    page = make_page(env, workspace, services)
    sheet = page.make_options_sheet()
    sheet.color_fields["gamma"].set_value(25)
    page.apply_choices(sheet.choices())
    page.refresh_now()
    assert "중간 밝기 +25" in page.ready.options_summary.text()
    assert page.prefs.gamma_adjust() == 25 and Prefs().print_options().gamma == 25
    assert PrintOptions.from_dict(PrintOptions(gamma=-30).to_dict()).gamma == -30
    again = page.make_options_sheet()
    again.reset_color.click()
    page.apply_choices(again.choices())
    assert page.prefs.gamma_adjust() == 0
