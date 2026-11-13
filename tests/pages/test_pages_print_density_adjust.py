"""인쇄 설정의 '진하기': SMART 프린터에서만 보이고, 고른 값이 저장되고 프린터까지 간다."""
from print_helpers import *  # noqa: F401,F403
from print_helpers import finish_job, make_page, wait_until

from services.render_job import PrintOptions
from ui.pages.settings import Prefs


def select_smart(env, services, page):
    smart = next(d for d in services.printers.devices() if d.family == "smart")
    services.printers.select(smart)
    services.printers.poll_now()
    wait_until(env, lambda: not services.printers.is_busy and services.printers.status() is not None)
    page.refresh_now()


def test_density_fields_are_hidden_for_rtai(env, workspace, services):
    page = make_page(env, workspace, services)
    sheet = page.make_options_sheet()
    assert sheet.density_fields == {}
    assert "진하기" not in page.ready.options_summary.text()


def test_density_fields_show_for_smart_and_carry_the_saved_values(env, workspace, services):
    page = make_page(env, workspace, services)
    select_smart(env, services, page)
    sheet = page.make_options_sheet()
    assert list(sheet.density_fields) == ["density", "density_yellow", "density_magenta", "density_cyan"]
    assert [field.value() for field in sheet.density_fields.values()] == [0, 0, 0, 0]

    sheet.density_fields["density"].set_value(20)
    sheet.density_fields["density_yellow"].set_value(-10)
    page.apply_choices(sheet.choices())
    page.refresh_now()
    assert "진하기 전체 +20 노랑 -10" in page.ready.options_summary.text()
    assert page.prefs.density_adjust() == (20, -10, 0, 0), "다음 인쇄에도 쓰도록 저장한다"
    assert Prefs().print_options().density == 20

    again = page.make_options_sheet()
    assert again.density_fields["density"].value() == 20
    again.reset_density.click()
    page.apply_choices(again.choices())
    assert page.prefs.density_adjust() == (0, 0, 0, 0)


def test_stored_values_outside_the_range_are_clamped(env):
    prefs = Prefs()
    prefs.set_density_adjust(300, -300, 5, 0)
    assert prefs.density_adjust() == (100, -100, 5, 0)


def test_options_survive_a_save_and_load():
    options = PrintOptions(density=15, density_yellow=-5, density_magenta=10, density_cyan=20)
    loaded = PrintOptions.from_dict(options.to_dict())
    assert (loaded.density, loaded.density_yellow, loaded.density_magenta, loaded.density_cyan) == (15, -5, 10, 20)
    assert PrintOptions.from_dict({}).density == 0, "예전 프로젝트 파일에는 값이 없다"


def test_chosen_values_reach_the_printer(env, workspace, services, smart_lib):
    page = make_page(env, workspace, services)
    select_smart(env, services, page)
    choices = page.choices().clone()
    choices.options.density, choices.options.density_cyan = 25, -15
    page.apply_choices(choices)
    page.refresh_now()
    workspace.set_checked_ids(workspace.store.ids()[:1])
    page.refresh_now()
    smart_lib.printed_density.clear()
    handle = page.start_printing()
    assert handle is not None
    finish_job(env, page)
    assert smart_lib.printed_density[-1] == (25, 0, 0, -15)
    assert smart_lib.density() == (0, 0, 0, 0)
