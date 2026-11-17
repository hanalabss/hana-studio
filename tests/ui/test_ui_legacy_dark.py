"""어두운 테마에서 예전 창(라이선스 인증·관리자 패널·모델 받기)에 밝은 배경이 남지 않는다."""
import re

import pytest
from PySide6.QtWidgets import QWidget

from ui.theme import DARK, LIGHT
from utils.legacy_qss import themed_qss

def test_colors_follow_the_property():
    qss = "QPushButton { background-color: #3B82F6; color: #FFFFFF; border: 1px solid #D1D5DB; }"
    dark = themed_qss(qss, DARK)
    assert f"background-color: {DARK.hex('primary_bg')}" in dark
    assert f"color: {DARK.hex('primary_text')}" in dark                     # 같은 #FFFFFF라도 글자는 글자 색
    assert f"border: 1px solid {DARK.hex('control_line')}" in dark
    assert themed_qss("color: #123456;", DARK) == "color: #123456;"          # 표에 없는 색은 그대로


def _backgrounds(widget: QWidget) -> list[str]:
    found = []
    for child in [widget, *widget.findChildren(QWidget)]:
        found += re.findall(r"background(?:-color)?\s*:\s*(#[0-9A-Fa-f]{6})", child.styleSheet())
    return [color.upper() for color in found]


@pytest.fixture
def dark(theme_manager):
    theme_manager.set_mode("dark")
    yield
    theme_manager.set_mode("light")


def test_license_and_admin_windows_use_dark_colors(qapp, dark, monkeypatch):
    from licensing import admin_panel
    from licensing.dialog import LicenseDialog

    monkeypatch.setattr(admin_panel.LoadLicensesThread, "start", lambda self: None)   # 서버에 묻지 않는다
    for window in (LicenseDialog(), admin_panel.AdminPanel(), admin_panel.CreateLicenseDialog()):
        backgrounds = _backgrounds(window)
        dark_colors = {value.upper() for value in DARK.colors.values()} if hasattr(DARK, "colors") else None
        if dark_colors is None:
            from ui.theme.tokens import DARK_COLORS

            dark_colors = {value.upper() for value in DARK_COLORS.values()}
        leftovers = set(backgrounds) - dark_colors              # 모든 배경이 어두운 테마 색이어야 한다
        assert backgrounds and not leftovers, (type(window).__name__, leftovers)
        assert DARK.hex("surface").upper() in backgrounds or DARK.hex("ground").upper() in backgrounds


def test_light_theme_keeps_the_original_look(qapp, theme_manager):
    from licensing.dialog import LicenseDialog

    theme_manager.set_mode("light")
    assert LIGHT.hex("surface").upper() in _backgrounds(LicenseDialog())
