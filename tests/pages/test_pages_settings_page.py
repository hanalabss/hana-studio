"""설정 화면, 인쇄 기본값 저장, 예전 설정 옮기기, 서비스 한 벌."""
import json
import sqlite3
from datetime import datetime, timedelta

from PySide6.QtCore import QSettings

from print_helpers import *  # noqa: F401,F403
from print_helpers import make_page

from ui.pages.services import AppServices
from ui.pages.settings import Prefs, SettingsPage
from ui.pages.settings import prefs as prefs_module
from ui.shell import PAGE_SETTINGS, AppShell
from ui.theme import KEY_GUIDE_VISIBLE, KEY_THEME_MODE, ThemeManager, app_settings, read_bool


def reopened() -> Prefs:
    """같은 설정 파일을 새로 연 Prefs(프로그램을 껐다 켠 것처럼)."""
    return Prefs(app_settings())


def test_theme_choice_applies_and_persists(env):
    page = SettingsPage()
    assert page.theme_control.keys() == ["system", "light", "dark"]
    assert page.theme_control.button("system").text() == "시스템 따라가기"
    page.theme_control.button("dark").click()
    manager = ThemeManager.instance()
    assert manager.mode == "dark" and manager.theme_name == "dark"
    assert str(app_settings().value(KEY_THEME_MODE)) == "dark"
    manager.set_mode("light")
    assert page.theme_control.current() == "light", "다른 곳에서 바꿔도 따라온다"


def test_guide_switch_drives_the_shell(env):
    shell = AppShell()
    page = SettingsPage()
    assert not page.guide_check.isEnabled(), "틀에 올라가기 전에는 켜고 끌 대상이 없다"
    shell.register_page(PAGE_SETTINGS, page, "설정", "settings")
    assert page.guide_check.isEnabled() and page.guide_check.isChecked()
    page.guide_check.click()
    assert shell.guide_visible() is False
    assert read_bool(app_settings(), KEY_GUIDE_VISIBLE, True) is False
    shell.set_guide_visible(True)
    assert page.guide_check.isChecked()


def test_print_defaults_persist_and_reach_the_print_page(env, workspace, services):
    page = SettingsPage()
    page.copies.set_value(3)
    page.stop_check.click()
    page.offset_x.set_value(0.7)
    page.offset_y.set_value(-1.2)
    page.card_combo.setCurrentIndex(2)
    page.retention_combo.setCurrentIndex(0)

    fresh = reopened()
    assert fresh.copies() == 3 and fresh.stop_on_error() is False
    assert fresh.offsets() == (0.7, -1.2)
    assert fresh.card_preset_key() == "cut_54x86_bleed" and fresh.card_preset().bleed_mm == 2.0
    assert fresh.retention_days() == 30
    options = fresh.print_options()
    assert (options.copies, options.stop_on_error, options.offset_x_mm, options.offset_y_mm) == (3, False, 0.7, -1.2)

    print_page = make_page(env, workspace, services)
    assert print_page.choices().options.copies == 3
    assert print_page.ready.start_button.text() == "15장 인쇄 시작"
    assert "위치 가로 +0.7 세로 -1.2mm" in print_page.ready.options_summary.text()

    # 설정을 바꾸면 열려 있는 인쇄 화면도 따라온다.
    page.copies.set_value(1)
    page.offset_reset.click()
    print_page.refresh_now()
    assert print_page.ready.start_button.text() == "5장 인쇄 시작"
    assert "위치" not in print_page.ready.options_summary.text()
    assert page.offset_x.value() == 0.0 and not page.offset_reset.isEnabled()


def test_position_changed_on_the_print_page_shows_up_in_settings(env, workspace, services):
    print_page = make_page(env, workspace, services)
    choices = print_page.choices().clone()
    choices.options.offset_x_mm = 1.5
    print_page.apply_choices(choices)
    page = SettingsPage()
    assert page.offset_x.value() == 1.5 and page.offset_reset.isEnabled()


def test_legacy_calibration_is_migrated_once(env, tmp_path, monkeypatch):
    legacy = tmp_path / "config.json"
    legacy.write_text(json.dumps({"printer": {
        "offset_x_mm": 0.4, "offset_y_mm": -0.6, "bleed_crop_enabled": True,
        "design_width_mm": 58, "design_height_mm": 90, "bleed_mm": 2,
    }}), encoding="utf-8")
    monkeypatch.setenv("HANA_STUDIO_LEGACY_CONFIG", str(legacy))
    settings = QSettings(str(tmp_path / "fresh.ini"), QSettings.IniFormat)
    prefs = Prefs(settings)
    assert prefs.offsets() == (0.4, -0.6)
    assert prefs.card_preset_key() == "cut_54x86_bleed"
    assert set(prefs.migrated) == {prefs_module.KEY_OFFSET_X, prefs_module.KEY_OFFSET_Y, prefs_module.KEY_CARD_PRESET}
    page = SettingsPage(prefs=prefs)
    assert page.offset_x.value() == 0.4 and page.offset_y.value() == -0.6
    assert page.migrated_note.text() == "예전 프로그램에서 쓰던 값을 그대로 가져왔습니다."
    assert page.card_combo.currentIndex() == 2

    # 한 번 옮긴 뒤에는 사용자가 바꾼 값을 예전 값으로 덮어쓰지 않는다.
    prefs.set_offsets(0.0, 0.0)
    again = Prefs(QSettings(str(tmp_path / "fresh.ini"), QSettings.IniFormat))
    assert again.offsets() == (0.0, 0.0) and again.migrated == {}


def test_legacy_values_for_the_shipped_config(env):
    shipped = {"dll_path": "./libDSRetransfer600App.dll", "card_width": 55, "card_height": 86.6,
               "bleed_crop_enabled": True, "design_width_mm": 58, "design_height_mm": 90, "bleed_mm": 2}
    values = prefs_module.legacy_values(shipped)
    assert values == {prefs_module.KEY_CARD_PRESET: "cut_54x86_bleed"}, "예전 앱은 위치 값을 파일에 남기지 않았다"
    assert prefs_module.legacy_values({"adjusted_x": "1.5", "adjusted_y": 0}) == {
        prefs_module.KEY_OFFSET_X: 1.5, prefs_module.KEY_OFFSET_Y: 0.0}
    assert prefs_module.legacy_values({}) == {}
    assert prefs_module.read_legacy_printer(prefs_module.legacy_config_path()) == {}


def test_out_of_range_stored_values_are_clamped(env):
    settings = app_settings()
    settings.setValue(prefs_module.KEY_OFFSET_X, 99)
    settings.setValue(prefs_module.KEY_COPIES, "많이")
    settings.setValue(prefs_module.KEY_RETENTION, 7)
    settings.setValue(prefs_module.KEY_CARD_PRESET, "없는 크기")
    prefs = reopened()
    assert prefs.offsets()[0] == prefs_module.OFFSET_LIMIT_MM
    assert prefs.copies() == 1 and prefs.retention_days() == prefs_module.DEFAULT_RETENTION
    assert prefs.card_preset_key() == "cr80"


def test_folders_and_versions(env, tmp_path):
    page = SettingsPage()
    opened = []
    page.open_folder = opened.append
    path, button = page.folder_rows["data"]
    assert path == str(tmp_path / "appdata" / "HanaStudio")
    button.click()
    assert opened == [path]
    assert (tmp_path / "appdata" / "HanaStudio").is_dir(), "없는 폴더는 만들어서 연다"
    assert set(page.version_labels) >= {"hanaStudio", "RTAI 프린터용 파일", "SMART 프린터용 파일"}
    assert page.version_labels["hanaStudio"]


def test_app_services_bundle_starts_and_shuts_down(env, queue, history, tmp_path):
    from services.print_service import PrintService
    from services.printers import PrinterService

    interrupted = history.start_job("끊긴 인쇄", None, 3)
    history.mark_job_started(interrupted)
    old = history.start_job("오래된 인쇄", None, 1)
    history.finish_job(old, "completed")
    stamp = (datetime.now() - timedelta(days=45)).isoformat(timespec="seconds")
    with sqlite3.connect(history.path) as connection:
        connection.execute("UPDATE jobs SET created_at = ? WHERE id = ?", (stamp, old))

    printers = PrinterService(queue, settings_path=str(tmp_path / "printer.json"))
    printing = PrintService(queue)
    bundle = AppServices(printers=printers, printing=printing, history=history)
    assert (bundle.printers, bundle.printing, bundle.history) == (printers, printing, history)
    order = []
    printing.shutdown = lambda: order.append("printing")
    stop_printers = printers.shutdown
    printers.shutdown = lambda: (order.append("printers"), stop_printers())

    bundle.start(retention_days=30)
    from print_helpers import wait_until

    wait_until(env, lambda: not printers.is_busy and printers.devices())
    assert history.get_job(interrupted).status == "failed", "켤 때 끊긴 기록을 닫는다"
    assert history.get_job(old) is None, "보관 기간이 지난 기록을 지운다"
    assert len(printers.devices()) == 2 and printers._timer.isActive()

    bundle.shutdown()
    bundle.shutdown()
    assert order == ["printing", "printers"], "인쇄를 먼저 멈추고 프린터 조회를 멈춘다(한 번만)"
    assert not printers._timer.isActive()


def test_license_admin_entry_only_for_admin(qapp, tmp_path, monkeypatch):
    from ui.pages.settings import SettingsPage

    plain = SettingsPage()
    assert not hasattr(plain, "license_button")

    admin = SettingsPage(is_admin=True)
    opened = []

    class FakePanel:
        def __init__(self, parent=None): opened.append(parent)
        def exec(self): opened.append("exec")

    import licensing.admin_panel as admin_panel
    monkeypatch.setattr(admin_panel, "AdminPanel", FakePanel)
    admin.license_button.click()
    assert opened and opened[-1] == "exec"
