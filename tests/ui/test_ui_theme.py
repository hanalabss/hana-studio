import re
from pathlib import Path

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtGui import QFontDatabase, QFontMetrics, QPalette

from ui.theme import (
    COLOR_ROLES,
    DARK,
    FONT_SIZES,
    KEY_THEME_MODE,
    LIGHT,
    MARK_PAIRS,
    TEXT_PAIRS,
    THEMES,
    ThemeManager,
    build_stylesheet,
    contrast_ratio,
    resolve_font,
)

ROOT = Path(__file__).resolve().parents[2]
HEX = re.compile(r"#[0-9A-Fa-f]{6}\b")


@pytest.mark.parametrize("theme", [LIGHT, DARK], ids=["light", "dark"])
def test_every_role_is_defined_with_a_valid_colour(theme):
    assert set(theme.colors) == set(COLOR_ROLES)
    for role in COLOR_ROLES:
        assert HEX.fullmatch(theme.hex(role)), f"{theme.name}.{role} = {theme.hex(role)!r}"
        assert theme.color(role).isValid()


def test_both_themes_define_the_same_roles():
    assert set(LIGHT.colors) == set(DARK.colors)
    assert len(COLOR_ROLES) > 60


def test_core_palette_matches_the_approved_direction():
    assert (LIGHT.hex("ground"), LIGHT.hex("surface"), LIGHT.hex("text")) == ("#F6F7F9", "#FFFFFF", "#1B2430")
    assert (LIGHT.hex("accent"), LIGHT.hex("line"), LIGHT.hex("field_bg")) == ("#2456E6", "#DDE1E7", "#FFE9A8")
    assert (DARK.hex("ground"), DARK.hex("surface"), DARK.hex("surface_raised")) == ("#15171A", "#1E2126", "#272B31")
    assert (DARK.hex("text"), DARK.hex("text_muted"), DARK.hex("accent")) == ("#E8EAED", "#B4BAC3", "#FF4D8D")


@pytest.mark.parametrize("theme_name", ["light", "dark"])
@pytest.mark.parametrize("pair", TEXT_PAIRS, ids=lambda p: f"{p[0]}-on-{p[1]}")
def test_text_contrast_is_at_least_4_5(theme_name, pair):
    theme = THEMES[theme_name]
    ratio = contrast_ratio(theme.hex(pair[0]), theme.hex(pair[1]))
    assert ratio >= 4.5, f"{theme_name}: {pair[0]} on {pair[1]} = {ratio:.2f}"


@pytest.mark.parametrize("theme_name", ["light", "dark"])
@pytest.mark.parametrize("pair", MARK_PAIRS, ids=lambda p: f"{p[0]}-on-{p[1]}")
def test_focus_and_indicator_contrast_is_at_least_3(theme_name, pair):
    theme = THEMES[theme_name]
    ratio = contrast_ratio(theme.hex(pair[0]), theme.hex(pair[1]))
    assert ratio >= 3.0, f"{theme_name}: {pair[0]} on {pair[1]} = {ratio:.2f}"


def test_contrast_ratio_known_values():
    assert contrast_ratio("#000000", "#FFFFFF") == pytest.approx(21.0)
    assert contrast_ratio("#FFFFFF", "#FFFFFF") == pytest.approx(1.0)


@pytest.mark.parametrize("theme", [LIGHT, DARK], ids=["light", "dark"])
def test_stylesheet_uses_only_token_colours(theme, theme_manager):
    from ui.theme.assets import ensure_assets

    qss = build_stylesheet(theme, theme_manager.font_spec, ensure_assets(theme))
    assert "$" not in qss
    allowed = {value.upper() for value in theme.colors.values()}
    used = {match.upper() for match in HEX.findall(qss)}
    assert used, "스타일시트에 색이 하나도 없습니다"
    assert used <= allowed, f"토큰에 없는 색: {sorted(used - allowed)}"
    for selector in ("QComboBox", "QScrollBar", "QCheckBox::indicator", "QAbstractSpinBox", "QMenu",
                     "QToolTip", "QHeaderView::section", ":disabled", ":focus"):
        assert selector in qss


def test_stylesheet_assets_exist_on_disk(theme_manager):
    from ui.theme.assets import ensure_assets

    for theme in (LIGHT, DARK):
        for name, path in ensure_assets(theme).items():
            assert Path(path).is_file() and Path(path).stat().st_size > 0, name


def test_no_hardcoded_hex_outside_tokens():
    offenders = []
    for folder in ("ui/theme", "ui/widgets", "ui/shell"):
        for path in (ROOT / folder).rglob("*.py"):
            if path.name == "tokens.py":
                continue
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                if HEX.search(line):
                    offenders.append(f"{path.relative_to(ROOT)}:{number}: {line.strip()}")
    assert not offenders, "\n".join(offenders)


def test_apply_sets_stylesheet_palette_and_font(qapp, theme_manager):
    assert theme_manager.theme_name == "light"
    assert LIGHT.hex("ground") in qapp.styleSheet()
    assert qapp.palette().color(QPalette.Window).name().upper() == LIGHT.hex("ground")
    assert qapp.font().pixelSize() == FONT_SIZES["body"]
    assert qapp.font().family() == theme_manager.font_spec.family


def test_switch_applies_live_and_emits(qapp, theme_manager, spy):
    changed = spy(theme_manager.themeChanged)
    theme_manager.set_mode("dark")
    assert changed.calls == [("dark",)]
    assert theme_manager.theme is DARK
    assert DARK.hex("ground") in qapp.styleSheet()
    assert LIGHT.hex("ground") not in qapp.styleSheet()
    assert qapp.palette().color(QPalette.Window).name().upper() == DARK.hex("ground")
    assert qapp.palette().color(QPalette.Text).name().upper() == DARK.hex("text")

    theme_manager.set_mode("dark")
    assert len(changed) == 1, "같은 테마로 다시 바꿀 때는 신호를 보내지 않는다"

    theme_manager.toggle()
    assert changed.last == ("light",)
    assert LIGHT.hex("ground") in qapp.styleSheet()


def test_choice_persists_across_managers(theme_manager, settings, tmp_path):
    theme_manager.set_mode("dark")
    settings.sync()
    reopened = QSettings(str(tmp_path / "ui-test.ini"), QSettings.IniFormat)
    assert reopened.value(KEY_THEME_MODE) == "dark"
    again = ThemeManager(reopened)
    assert again.mode == "dark"
    assert again.theme_name == "dark"


def test_persist_false_does_not_write(theme_manager, settings):
    theme_manager.set_mode("dark", persist=False)
    assert settings.value(KEY_THEME_MODE) is None


def test_default_mode_follows_windows_setting(tmp_path, monkeypatch, qapp, spy):
    monkeypatch.setattr(ThemeManager, "system_theme_name", staticmethod(lambda: "dark"))
    manager = ThemeManager(QSettings(str(tmp_path / "fresh.ini"), QSettings.IniFormat))
    assert manager.mode == "system"
    assert manager.theme_name == "dark"
    manager.apply(qapp)
    assert DARK.hex("ground") in qapp.styleSheet()

    changed = spy(manager.themeChanged)
    monkeypatch.setattr(ThemeManager, "system_theme_name", staticmethod(lambda: "light"))
    manager._on_system_scheme_changed()
    assert changed.calls == [("light",)]
    assert LIGHT.hex("ground") in qapp.styleSheet()

    manager.set_mode("dark")
    monkeypatch.setattr(ThemeManager, "system_theme_name", staticmethod(lambda: "light"))
    manager._on_system_scheme_changed()
    assert manager.theme_name == "dark", "직접 고른 테마는 Windows 설정이 바뀌어도 그대로다"


def test_unknown_mode_is_rejected(theme_manager):
    with pytest.raises(ValueError):
        theme_manager.set_mode("sepia")
    assert theme_manager.mode == "light"


def test_corrupt_stored_mode_falls_back_to_system(tmp_path):
    store = QSettings(str(tmp_path / "bad.ini"), QSettings.IniFormat)
    store.setValue(KEY_THEME_MODE, "neon")
    assert ThemeManager(store).mode == "system"


def test_font_is_installed_and_has_korean_glyphs(qapp):
    spec = resolve_font()
    assert spec.family in QFontDatabase.families()
    assert spec.strong_weight in (600, 700)
    metrics = QFontMetrics(qapp.font())
    for ch in "가힣명단":
        assert metrics.inFontUcs4(ord(ch)), f"글꼴에 '{ch}' 글자가 없습니다"


def test_preferred_font_wins_when_installed():
    assert resolve_font("Malgun Gothic").family == "Malgun Gothic"
    assert resolve_font("없는 글꼴 이름").family in QFontDatabase.families()


def test_type_scale_is_ordered():
    sizes = [FONT_SIZES[k] for k in ("caption", "small", "body", "title", "heading", "display")]
    assert sizes == sorted(sizes)
    assert sizes[0] >= 12


def test_native_rules_are_scoped_to_themed_widgets():
    from ui.theme import scope_stylesheet

    scoped = scope_stylesheet(
        "/* 설명 */ QPushButton:hover, QLabel[role=\"title\"] { color: red; }\n"
        "QToolTip { color: red; }\nQFrame#Card { color: red; }\n"
        "QScrollBar::handle:vertical { color: red; }"
    )
    lines = scoped.splitlines()
    assert lines[0] == ('QPushButton[themed="true"]:hover, *[themed="true"] QPushButton:hover, '
                        'QLabel[role="title"] {color: red;}')
    assert lines[1] == "QToolTip {color: red;}"
    assert lines[2] == "QFrame#Card {color: red;}"
    assert lines[3].startswith('QScrollBar[themed="true"]::handle:vertical, *[themed="true"] QScrollBar::handle')
    assert "설명" not in scoped


def test_mark_themed_styles_widget_and_children_even_after_show(qapp, theme_manager):
    from PySide6.QtWidgets import QPlainTextEdit, QPushButton, QVBoxLayout, QWidget

    from ui.theme import mark_themed

    box = QWidget()
    layout = QVBoxLayout(box)
    button = QPushButton("저장")
    memo = QPlainTextEdit("뒷면 문구")
    layout.addWidget(button)
    layout.addWidget(memo)
    box.show()
    qapp.processEvents()
    plain_height = button.sizeHint().height()

    assert mark_themed(box) is box
    qapp.processEvents()
    assert box.property("themed") == "true"
    assert button.sizeHint().height() == 40 > plain_height
    assert memo.viewport().geometry().topLeft().x() == 10, "글상자 안쪽 여백(테두리 1 + 여백 9)이 살아 있어야 한다"
    assert box.styleSheet() == ""

    mark_themed(box, False)
    qapp.processEvents()
    assert button.sizeHint().height() == plain_height
    box.close()
