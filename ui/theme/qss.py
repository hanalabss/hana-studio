"""토큰으로 애플리케이션 스타일시트 한 장을 만든다.

QPushButton, QLineEdit 같은 기본 위젯 규칙은 `themed="true"` 속성이 붙은 위젯과 그 안쪽에만
먹는다(mark_themed). 그래서 기존 '빠른 인쇄' 화면처럼 표시하지 않은 영역은 원래 모습 그대로다.
"""

from __future__ import annotations

import re
from string import Template

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget

from .fonts import FontSpec
from .tokens import FONT_SIZES, RADIUS, SIZES, Theme

_TEMPLATE = Template(r"""
/* ---------- 바탕 ---------- */
QWidget { color: $text; font-size: ${fs_body}px; }
QDialog, QMessageBox { background: $ground; }
QWidget#PageArea, QWidget#Page, QWidget#GalleryRoot { background: $ground; }
QToolTip {
    background: $tooltip_bg; color: $tooltip_text; border: 1px solid $tooltip_bg;
    padding: 5px 8px; font-size: ${fs_caption}px;
}

/* ---------- 글자 ---------- */
QLabel { background: transparent; color: $text; }
QLabel:disabled { color: $text_disabled; }
QLabel[role="display"] { font-size: ${fs_display}px; font-weight: $strong; }
QLabel[role="heading"] { font-size: ${fs_heading}px; font-weight: $strong; }
QLabel[role="title"] { font-size: ${fs_title}px; font-weight: $strong; }
QLabel[role="strong"] { font-weight: $strong; }
QLabel[role="secondary"] { color: $text_secondary; }
QLabel[role="muted"] { color: $text_muted; font-size: ${fs_small}px; }
QLabel[role="label"] { color: $text_secondary; font-size: ${fs_small}px; }
QLabel[role="caption"], QLabel[role="hint"] { color: $text_muted; font-size: ${fs_caption}px; }
QLabel[role="hint"][tone="error"] { color: $danger_text; }
QLabel[role="hint"][tone="warn"] { color: $warn_text; }
QLabel[role="link"] { color: $accent_text; }

/* ---------- 버튼 ---------- */
QPushButton {
    min-height: 38px; padding: 0 16px; border: 1px solid $control_line; border-radius: ${r_md}px;
    background: $surface_raised; color: $text; font-size: ${fs_body}px; outline: none;
}
QPushButton:hover { background: $surface_hover; }
QPushButton:pressed { background: $surface_pressed; }
QPushButton:focus { border-color: $focus; }
QPushButton:disabled { color: $text_disabled; background: transparent; border-color: $line; }

QPushButton[sizing="sm"] { min-height: 30px; padding: 0 12px; font-size: ${fs_small}px; border-radius: ${r_sm}px; }
QPushButton[sizing="md"] { min-height: 38px; padding: 0 16px; }
QPushButton[sizing="lg"] { min-height: 46px; padding: 0 24px; font-size: ${fs_title}px; border-radius: 10px; }

QPushButton[variant="secondary"]:focus { border-color: $control_line; }

QPushButton[variant="primary"] {
    background: $primary_bg; color: $primary_text; border-color: $primary_bg; font-weight: $strong;
}
QPushButton[variant="primary"]:hover { background: $primary_hover; border-color: $primary_hover; }
QPushButton[variant="primary"]:pressed { background: $primary_pressed; border-color: $primary_pressed; }
QPushButton[variant="primary"]:focus { border-color: $primary_bg; }
QPushButton[variant="primary"]:disabled {
    background: $primary_disabled_bg; color: $text_disabled; border-color: $primary_disabled_bg;
}

QPushButton[variant="quiet"] { background: transparent; border-color: transparent; color: $text; }
QPushButton[variant="quiet"]:hover { background: $surface_hover; }
QPushButton[variant="quiet"]:pressed { background: $surface_pressed; }
QPushButton[variant="quiet"]:focus { border-color: transparent; }
QPushButton[variant="quiet"]:disabled { background: transparent; border-color: transparent; color: $text_disabled; }
QPushButton[variant="quiet"][accent="true"] { color: $accent_text; font-weight: $strong; }
QPushButton[variant="quiet"][accent="true"]:disabled { color: $text_disabled; }

QPushButton[variant="danger"] {
    background: $danger; color: $danger_on; border-color: $danger; font-weight: $strong;
}
QPushButton[variant="danger"]:hover { background: $danger_hover; border-color: $danger_hover; }
QPushButton[variant="danger"]:focus { border-color: $danger; }
QPushButton[variant="danger"]:disabled {
    background: $primary_disabled_bg; color: $text_disabled; border-color: $primary_disabled_bg;
}

QPushButton[iconOnly="true"] { padding: 0; }

QToolButton {
    min-height: 30px; padding: 0 8px; border: 1px solid transparent; border-radius: ${r_sm}px;
    background: transparent; color: $text;
}
QToolButton:hover { background: $surface_hover; }
QToolButton:pressed, QToolButton:checked { background: $surface_pressed; }
QToolButton:focus { border-color: $focus; }
QToolButton:disabled { color: $text_disabled; }

/* ---------- 세그먼트 ---------- */
QFrame#Segmented { background: $surface_sunken; border: 1px solid $line; border-radius: ${r_md}px; }
QPushButton[segment="true"] {
    min-height: 30px; padding: 0 14px; border: 0; border-radius: ${r_sm}px;
    background: transparent; color: $text_secondary; font-size: ${fs_small}px;
}
QPushButton[segment="true"]:hover { background: $surface_hover; color: $text; }
QPushButton[segment="true"]:checked { background: $strong_bg; color: $strong_text; font-weight: $strong; }
QPushButton[segment="true"]:disabled { background: transparent; color: $text_disabled; }
QPushButton[segment="true"]:checked:disabled { background: $surface_pressed; color: $text_disabled; }

/* ---------- 칩 ---------- */
QPushButton[chip="filter"] {
    min-height: 30px; padding: 0 14px; border: 1px solid $control_line; border-radius: 16px;
    background: $surface_raised; color: $text; font-size: ${fs_small}px;
}
QPushButton[chip="filter"]:hover { background: $surface_hover; }
QPushButton[chip="filter"]:checked { background: $strong_bg; color: $strong_text; border-color: $strong_bg; }
QPushButton[chip="filter"]:focus { border-color: $control_line; }
QPushButton[chip="filter"]:checked:focus { border-color: $strong_bg; }
QPushButton[chip="filter"][tone="warn"] { background: $warn_bg; color: $warn_text; border-color: $warn_line; }
QPushButton[chip="filter"][tone="warn"]:checked { background: $warn_text; color: $warn_bg; border-color: $warn_text; }
QPushButton[chip="filter"]:disabled { background: $surface_sunken; color: $text_disabled; border-color: $line; }

QLabel#FieldChip {
    background: $field_bg; color: $field_text; border: 1px solid $field_bg; border-radius: 16px;
    padding: 0 14px; font-size: ${fs_body}px;
}
QLabel#FieldChip:hover { background: $field_hover; border-color: $field_line; }
QLabel#FieldChip:focus { border: 2px solid $field_text; padding: 0 13px; }
QLabel#FieldChip:disabled { background: $surface_sunken; color: $text_disabled; border-color: $line; }

QPushButton[pill="status"] {
    min-height: 30px; padding: 0 14px 0 10px; border: 1px solid transparent; border-radius: 16px;
    font-size: ${fs_small}px; background: $idle_bg; color: $idle_text;
}
QPushButton[pill="status"][state="ready"] { background: $ok_bg; color: $ok_text; }
QPushButton[pill="status"][state="busy"] { background: $info_bg; color: $info_text; }
QPushButton[pill="status"][state="error"] { background: $danger_bg; color: $danger_text; }
QPushButton[pill="status"][state="offline"] { background: $idle_bg; color: $idle_text; }
QPushButton[pill="status"]:hover { border-color: $control_line; }
QPushButton[pill="status"]:focus { border-color: transparent; }

/* ---------- 상자 ---------- */
QFrame#Card { background: $surface; border: 1px solid $line; border-radius: ${r_lg}px; }
QFrame#Card[sunken="true"] { background: $surface_sunken; border-color: $surface_sunken; }
QFrame#Divider { background: $line; border: 0; }
QFrame#Popover { background: $surface_raised; border: 1px solid $control_line; border-radius: ${r_lg}px; }

QFrame#Banner { border-radius: ${r_md}px; border: 1px solid $info_line; background: $info_bg; }
QFrame#Banner QLabel { color: $info_text; font-size: ${fs_small}px; }
QFrame#Banner QPushButton {
    min-height: 28px; padding: 0 10px; border: 1px solid transparent; border-radius: ${r_sm}px;
    background: transparent; color: $info_text; font-weight: $strong; font-size: ${fs_small}px;
}
QFrame#Banner QPushButton:hover { border-color: $info_text; }
QFrame#Banner QPushButton[quiet="true"] { font-weight: normal; }
QFrame#Banner[kind="ok"] { background: $ok_bg; border-color: $ok_bg; }
QFrame#Banner[kind="ok"] QLabel, QFrame#Banner[kind="ok"] QPushButton { color: $ok_text; }
QFrame#Banner[kind="ok"] QPushButton:hover { border-color: $ok_text; }
QFrame#Banner[kind="warn"] { background: $warn_bg; border-color: $warn_line; }
QFrame#Banner[kind="warn"] QLabel, QFrame#Banner[kind="warn"] QPushButton { color: $warn_text; }
QFrame#Banner[kind="warn"] QPushButton:hover { border-color: $warn_text; }
QFrame#Banner[kind="error"] { background: $danger_bg; border-color: $danger_line; }
QFrame#Banner[kind="error"] QLabel, QFrame#Banner[kind="error"] QPushButton { color: $danger_text; }
QFrame#Banner[kind="error"] QPushButton:hover { border-color: $danger_text; }

QFrame#Toast { background: $toast_bg; border: 0; border-radius: 10px; }
QFrame#Toast QLabel { color: $toast_text; font-size: ${fs_body}px; }
QFrame#Toast QPushButton {
    min-height: 30px; padding: 0 10px; border: 1px solid transparent; border-radius: ${r_sm}px;
    background: transparent; color: $toast_action; font-weight: $strong;
}
QFrame#Toast QPushButton:hover { border-color: $toast_action; }

/* ---------- 입력 ---------- */
QLineEdit, QPlainTextEdit, QTextEdit, QAbstractSpinBox, QComboBox {
    border: 1px solid $control_line; border-radius: ${r_md}px; background: $surface_raised; color: $text;
    selection-background-color: $accent; selection-color: $text_on_accent; font-size: ${fs_body}px;
}
QLineEdit, QAbstractSpinBox, QComboBox { min-height: 38px; max-height: 38px; padding: 0 11px; }
QPlainTextEdit, QTextEdit { padding: 7px 9px; }
QLineEdit:hover, QPlainTextEdit:hover, QTextEdit:hover, QAbstractSpinBox:hover, QComboBox:hover {
    border-color: $indicator_line;
}
QLineEdit:focus, QAbstractSpinBox:focus, QComboBox:focus {
    border: 2px solid $focus; padding: 0 10px;
}
QPlainTextEdit:focus, QTextEdit:focus { border: 2px solid $focus; padding: 6px 8px; }
QLineEdit:disabled, QPlainTextEdit:disabled, QTextEdit:disabled, QAbstractSpinBox:disabled, QComboBox:disabled {
    background: $surface_sunken; color: $text_disabled; border-color: $line;
}
QLineEdit[invalid="true"] { border-color: $danger; }
QLineEdit[readOnly="true"] { background: $surface_sunken; }
QLineEdit[sizing="sm"], QComboBox[sizing="sm"], QAbstractSpinBox[sizing="sm"] { min-height: 30px; max-height: 30px; font-size: ${fs_small}px; border-radius: ${r_sm}px; }

QComboBox { padding-right: 4px; }
QComboBox:focus { padding-right: 3px; }
QComboBox::drop-down { subcontrol-origin: padding; subcontrol-position: center right; width: 30px; border: 0; }
QComboBox::down-arrow { image: url($img_chevron_down); width: 14px; height: 14px; }
QComboBox::down-arrow:disabled { image: url($img_chevron_down_disabled); }
QComboBox QAbstractItemView {
    background: $surface_raised; color: $text; border: 1px solid $control_line; outline: 0; padding: 4px;
    selection-background-color: $surface_hover; selection-color: $text;
}
QComboBox QAbstractItemView::item { min-height: 32px; padding: 0 8px; border-radius: ${r_sm}px; color: $text; }
QComboBox QAbstractItemView::item:selected, QComboBox QAbstractItemView::item:hover { background: $surface_hover; color: $text; }

QAbstractSpinBox { padding-right: 26px; }
QAbstractSpinBox:focus { padding-right: 25px; }
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {
    subcontrol-origin: border; width: 24px; border: 0; background: transparent; margin-right: 2px;
}
QAbstractSpinBox::up-button { subcontrol-position: top right; margin-top: 3px; border-top-right-radius: 5px; }
QAbstractSpinBox::down-button { subcontrol-position: bottom right; margin-bottom: 3px; border-bottom-right-radius: 5px; }
QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover { background: $surface_hover; }
QAbstractSpinBox::up-button:pressed, QAbstractSpinBox::down-button:pressed { background: $surface_pressed; }
QAbstractSpinBox::up-arrow { image: url($img_spin_up); width: 10px; height: 10px; }
QAbstractSpinBox::down-arrow { image: url($img_spin_down); width: 10px; height: 10px; }
QAbstractSpinBox::up-arrow:disabled, QAbstractSpinBox::up-arrow:off { image: url($img_spin_up_disabled); }
QAbstractSpinBox::down-arrow:disabled, QAbstractSpinBox::down-arrow:off { image: url($img_spin_down_disabled); }

QLineEdit#StepperValue { font-weight: $strong; font-size: ${fs_body}px; }
QPushButton[stepper="true"] { padding: 0; }

QCheckBox, QRadioButton { spacing: 8px; color: $text; background: transparent; padding: 3px 5px 3px 2px; outline: none; }
QCheckBox:disabled, QRadioButton:disabled { color: $text_disabled; }
QCheckBox::indicator, QRadioButton::indicator {
    width: 18px; height: 18px; border: 1px solid $indicator_line; background: $surface_raised;
}
QCheckBox::indicator { border-radius: 4px; }
QRadioButton::indicator { border-radius: 10px; }
QCheckBox::indicator:hover, QRadioButton::indicator:hover { border-color: $accent; }
QCheckBox::indicator:checked { background: $accent; border-color: $accent; image: url($img_check); }
QCheckBox::indicator:indeterminate { background: $accent; border-color: $accent; }
QRadioButton::indicator:checked { border: 6px solid $accent; width: 8px; height: 8px; background: $text_on_accent; }
QCheckBox::indicator:disabled, QRadioButton::indicator:disabled { border-color: $line; background: $surface_sunken; }
QCheckBox::indicator:checked:disabled { background: $surface_pressed; border-color: $surface_pressed; image: url($img_check_disabled); }
QRadioButton::indicator:checked:disabled { border: 6px solid $surface_pressed; background: $text_disabled; }
QCheckBox:focus, QRadioButton:focus { border: 1px solid $focus; border-radius: ${r_sm}px; padding: 2px 4px 2px 1px; }

QSlider::groove:horizontal { height: 4px; background: $surface_pressed; border-radius: 2px; }
QSlider::sub-page:horizontal { background: $accent; border-radius: 2px; }
QSlider::handle:horizontal {
    width: 16px; height: 16px; margin: -7px 0; border-radius: 9px;
    background: $surface_raised; border: 1px solid $indicator_line;
}
QSlider::handle:horizontal:hover, QSlider::handle:horizontal:focus { border: 2px solid $accent; margin: -8px 0; }
QSlider::handle:horizontal:disabled { background: $surface_sunken; border-color: $line; }
QSlider::sub-page:horizontal:disabled { background: $text_disabled; }

/* ---------- 진행 ---------- */
QProgressBar {
    min-height: 8px; max-height: 8px; border: 0; border-radius: 4px; background: $ribbon_track;
    color: transparent;
}
QProgressBar::chunk { background: $accent; border-radius: 4px; }

/* ---------- 목록과 표 ---------- */
QAbstractItemView {
    background: $surface; alternate-background-color: $surface_sunken; color: $text;
    border: 1px solid $line; outline: 0;
    selection-background-color: $accent_soft; selection-color: $text;
}
QTableView { gridline-color: $line_soft; }
QTableView::item, QTreeView::item, QListView::item { padding: 6px 8px; color: $text; }
QTableView::item:selected, QTreeView::item:selected, QListView::item:selected {
    background: $accent_soft; color: $text;
}
QTableView::item:hover, QTreeView::item:hover, QListView::item:hover { background: $surface_hover; }
QTableView::item:selected:hover, QTreeView::item:selected:hover, QListView::item:selected:hover { background: $accent_soft; }
QTableView::item:disabled, QTreeView::item:disabled, QListView::item:disabled { color: $text_disabled; }
QTableView::indicator, QTreeView::indicator, QListView::indicator {
    width: 16px; height: 16px; border: 1px solid $indicator_line; border-radius: 4px; background: $surface_raised;
}
QTableView::indicator:checked, QTreeView::indicator:checked, QListView::indicator:checked {
    background: $accent; border-color: $accent; image: url($img_check);
}
QHeaderView { background: $surface_sunken; border: 0; }
QHeaderView::section {
    background: $surface_sunken; color: $text_secondary; border: 0; border-bottom: 1px solid $line;
    padding: 9px 8px 9px 11px; font-size: ${fs_small}px;
}
QHeaderView::section:vertical { border-bottom: 0; border-right: 1px solid $line; padding: 0 8px; }
QHeaderView::section:hover { color: $text; }
QTableCornerButton::section { background: $surface_sunken; border: 0; border-bottom: 1px solid $line; }

/* ---------- 스크롤 ---------- */
QScrollArea { border: 0; background: transparent; }
QScrollArea > QWidget > QWidget#ScrollBody { background: transparent; }
QScrollBar { background: transparent; border: 0; }
QScrollBar:vertical { width: 12px; margin: 0; }
QScrollBar:horizontal { height: 12px; margin: 0; }
QScrollBar::handle { background: $scrollbar; border-radius: 3px; }
QScrollBar::handle:vertical { min-height: 36px; margin: 2px 3px; }
QScrollBar::handle:horizontal { min-width: 36px; margin: 3px 2px; }
QScrollBar::handle:hover, QScrollBar::handle:pressed { background: $scrollbar_hover; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; border: 0; background: transparent; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }
QAbstractScrollArea::corner { background: transparent; }

/* ---------- 메뉴 ---------- */
QMenu { background: $surface_raised; color: $text; border: 1px solid $control_line; padding: 6px; }
QMenu::item { padding: 8px 28px 8px 12px; border-radius: ${r_sm}px; background: transparent; }
QMenu::item:selected { background: $surface_hover; color: $text; }
QMenu::item:disabled { color: $text_disabled; background: transparent; }
QMenu::separator { height: 1px; background: $line; margin: 6px 4px; }
QMenu::icon { padding-left: 8px; }
QMenu::indicator { width: 14px; height: 14px; left: 8px; }
QMenu::right-arrow { image: url($img_chevron_right); width: 14px; height: 14px; right: 6px; }
QMenuBar { background: $surface; color: $text; }
QMenuBar::item { padding: 6px 10px; background: transparent; }
QMenuBar::item:selected { background: $surface_hover; }

/* ---------- 탭, 묶음 ---------- */
QTabWidget::pane { border: 1px solid $line; border-radius: ${r_md}px; top: -1px; background: $surface; }
QTabBar::tab {
    padding: 9px 16px; border: 0; border-bottom: 2px solid transparent; background: transparent;
    color: $text_secondary;
}
QTabBar::tab:hover { color: $text; }
QTabBar::tab:selected { color: $text; font-weight: $strong; border-bottom-color: $accent; }
QTabBar::tab:disabled { color: $text_disabled; }
QGroupBox {
    border: 1px solid $line; border-radius: ${r_md}px; margin-top: 12px; padding: 16px 12px 12px 12px;
    background: transparent; font-weight: $strong;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: $text_secondary; }
QSplitter::handle { background: $line; }
QSplitter::handle:horizontal { width: 1px; }
QSplitter::handle:vertical { height: 1px; }
QStatusBar { background: $surface; color: $text_muted; border-top: 1px solid $line; }

/* ---------- 틀 ---------- */
QFrame#TopBar { background: $surface; border: 0; border-bottom: 1px solid $line; }
QLabel#AppName { font-size: ${fs_title}px; font-weight: $strong; }
QLabel#DocName { color: $text_secondary; font-size: ${fs_small}px; }
QLabel#DocState { color: $text_muted; font-size: ${fs_caption}px; }
QLabel#DocState[state="dirty"] { color: $warn_text; }
QFrame#NavRail { background: $surface; border: 0; border-right: 1px solid $line; }
QFrame#GuideStrip { background: $surface; border: 0; border-bottom: 1px solid $line; }
QPushButton#GuideHide { color: $text_muted; }
QPushButton#GuideHide:hover { color: $text; }
QFrame#PageFooter { background: $surface; border: 0; border-top: 1px solid $line; }
QFrame#DialogFooter { background: $surface_sunken; border: 0; border-top: 1px solid $line; }
QDialog#KitDialog { background: $surface; }
QFrame#SettingsRow { background: transparent; border: 0; border-bottom: 1px solid $line_soft; }
""")


THEMED_PROPERTY = "themed"
_SCOPE = f'[{THEMED_PROPERTY}="true"]'
# 어디서나 먹어도 되는 것: 따로 뜨는 작은 창들.
_GLOBAL_TYPES = ("QToolTip", "QMenu", "QMenuBar")
# 우리 위젯만 갖는 속성. 이 속성으로 고른 규칙은 범위를 따로 좁히지 않는다.
_KIT_PROPERTIES = ("role", "variant", "sizing", "segment", "chip", "pill", "stepper", "iconOnly",
                   "tone", "kind", "quiet", "state", "accent", "sunken", "invalid")
_RULE = re.compile(r"([^{}]+)\{([^{}]*)\}")
_COMMENT = re.compile(r"/\*.*?\*/", re.S)
_LEADING_TYPE = re.compile(r"^(\*|Q\w+)")


def _needs_scope(selector: str) -> bool:
    if "#" in selector or any(f"[{name}=" in selector for name in _KIT_PROPERTIES):
        return False
    return not selector.startswith(_GLOBAL_TYPES)


def scope_stylesheet(qss: str) -> str:
    """기본 위젯 규칙을 themed 위젯 자신과 그 자손으로 좁힌다."""
    out: list[str] = []
    for match in _RULE.finditer(_COMMENT.sub("", qss)):
        selectors = [s.strip() for s in match.group(1).split(",") if s.strip()]
        scoped: list[str] = []
        for selector in selectors:
            if _needs_scope(selector):
                scoped.append(_LEADING_TYPE.sub(lambda m: m.group(1) + _SCOPE, selector, count=1))
                scoped.append(f"*{_SCOPE} {selector}")
            else:
                scoped.append(selector)
        out.append(", ".join(scoped) + " {" + " ".join(match.group(2).split()) + "}")
    return "\n".join(out)


def mark_themed(widget: QWidget, themed: bool = True) -> QWidget:
    """이 위젯과 그 안쪽 전부에 테마를 입힌다. 따로 뜨는 창(팝업, 대화 상자)은 직접 불러 줘야 한다."""
    widget.setProperty(THEMED_PROPERTY, "true" if themed else "false")
    already_styled = any(w.testAttribute(Qt.WA_WState_Polished) for w in (widget, *widget.findChildren(QWidget)))
    if already_styled:
        # 이미 화면에 나온 위젯은 스타일을 통째로 다시 계산시킨다.
        # (unpolish/polish를 직접 부르면 글상자 안쪽 여백 같은 것이 빠진다.)
        sheet = widget.styleSheet()
        widget.setStyleSheet(sheet + "/**/")
        widget.setStyleSheet(sheet)
    return widget


def build_stylesheet(theme: Theme, font: FontSpec, assets: dict[str, str] | None = None) -> str:
    values: dict[str, object] = dict(theme.colors)
    values.update({f"fs_{k}": v for k, v in FONT_SIZES.items()})
    values.update({f"r_{k}": v for k, v in RADIUS.items()})
    values.update({f"size_{k}": v for k, v in SIZES.items()})
    values["strong"] = font.strong_weight
    for name, path in (assets or {}).items():
        values["img_" + name.replace("-", "_")] = path
    return scope_stylesheet(_TEMPLATE.substitute(values))

