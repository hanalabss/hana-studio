"""인쇄·기록·프린터·설정 화면 전용 스타일. 색은 모두 테마 토큰에서 가져온다."""
from __future__ import annotations

from string import Template

from PySide6.QtWidgets import QHBoxLayout, QWidget

from ui.theme import FONT_SIZES, Theme, ThemeManager

LEFT_WIDTH = 340
RIGHT_WIDTH = 368
RUN_WIDTH = 400
DETAIL_WIDTH = 460

_QSS = Template("""
QFrame#SidePanel { background: $surface; border: 0; }
QFrame#SidePanel[edge="right"] { border-right: 1px solid $line; }
QFrame#SidePanel[edge="left"] { border-left: 1px solid $line; }
QWidget#Stage, QWidget#PanelBody { background: transparent; }
QScrollArea#PanelScroll { background: transparent; border: 0; }

QLabel#BigCount { font-size: 64px; font-weight: $strong; color: $text; }
QLabel#BigCountRest { font-size: 22px; font-weight: $strong; color: $text_secondary; }
QLabel#Headline { font-size: 26px; font-weight: $strong; }
QLabel#StatNumber { font-size: ${fs_heading}px; font-weight: $strong; }
QLabel#StatNumber[tone="warn"] { color: $warn_text; }
QLabel#StatNumber[tone="muted"] { color: $text_muted; }
QLabel#Reason { color: $text_secondary; font-size: ${fs_small}px; }
QLabel#Reason[tone="error"] { color: $danger_text; }
QLabel#Reason[tone="warn"] { color: $warn_text; }
QLabel#Reason[tone="ok"] { color: $ok_text; }

QFrame#Finding { border-radius: 8px; border: 1px solid $info_line; background: $info_bg; }
QFrame#Finding QLabel { color: $info_text; font-size: ${fs_small}px; }
QFrame#Finding QPushButton {
    min-height: 28px; padding: 0 10px; border: 1px solid $info_text; border-radius: 6px;
    background: transparent; color: $info_text; font-weight: $strong; font-size: ${fs_small}px;
}
QFrame#Finding QPushButton:hover { background: $surface; }
QFrame#Finding[tone="error"] { background: $danger_bg; border-color: $danger_line; }
QFrame#Finding[tone="error"] QLabel { color: $danger_text; }
QFrame#Finding[tone="error"] QPushButton { color: $danger_text; border-color: $danger_text; }
QFrame#Finding[tone="warn"] { background: $warn_bg; border-color: $warn_line; }
QFrame#Finding[tone="warn"] QLabel { color: $warn_text; }
QFrame#Finding[tone="warn"] QPushButton { color: $warn_text; border-color: $warn_text; }
QFrame#Finding[tone="ok"] { background: $ok_bg; border-color: $ok_bg; }
QFrame#Finding[tone="ok"] QLabel { color: $ok_text; }
QFrame#Finding[tone="ok"] QPushButton { color: $ok_text; border-color: $ok_text; }
QFrame#Finding[tone="idle"] { background: $surface_sunken; border-color: $surface_sunken; }
QFrame#Finding[tone="idle"] QLabel { color: $text_secondary; }
QFrame#Finding[tone="idle"] QPushButton { color: $text; border-color: $control_line; }
QCheckBox#Acknowledge { color: $warn_text; font-size: ${fs_small}px; }

QFrame#Block { background: $surface_sunken; border: 0; border-radius: 10px; }
QFrame#TableCard { background: $surface; border: 1px solid $line; border-radius: 12px; }
QFrame#DeviceCard { background: $surface; border: 1px solid $line; border-radius: 12px; }
QFrame#DeviceCard[current="true"] { border: 2px solid $accent; }

QLabel#Tag {
    background: $idle_bg; color: $idle_text; border-radius: 10px; padding: 0 10px;
    min-height: 22px; max-height: 22px; font-size: ${fs_caption}px;
}
QLabel#Tag[tone="ok"] { background: $ok_bg; color: $ok_text; }
QLabel#Tag[tone="warn"] { background: $warn_bg; color: $warn_text; }
QLabel#Tag[tone="error"] { background: $danger_bg; color: $danger_text; }
QLabel#Tag[tone="busy"] { background: $info_bg; color: $info_text; }
QLabel#Tag[tone="accent"] { background: $accent_soft; color: $accent_text; }

QListView#People { background: transparent; border: 0; outline: 0; }
QListView#People::item { padding: 0; border: 0; }
QTableView#Rows { background: transparent; border: 0; outline: 0; gridline-color: transparent; }
QTableView#Rows QHeaderView { background: transparent; }
QTableView#Rows QHeaderView::section {
    background: transparent; color: $text_muted; border: 0; border-bottom: 1px solid $line;
    padding: 10px 8px 10px 10px; font-size: ${fs_small}px;
}

QRadioButton#Target { font-size: ${fs_body}px; padding: 5px 5px 5px 2px; }
QRadioButton#Target:focus { padding: 4px 4px 4px 1px; }
QPlainTextEdit#RawDetail {
    background: $surface_sunken; border: 1px solid $line; border-radius: 8px; color: $text_secondary;
    font-size: ${fs_caption}px;
}
QFrame#CheckItem { background: transparent; border: 0; border-bottom: 1px solid $line_soft; }
""")


def build_stylesheet(theme: Theme | None = None) -> str:
    manager = ThemeManager.instance()
    theme = theme or manager.theme
    values: dict[str, object] = dict(theme.colors)
    values.update({f"fs_{k}": v for k, v in FONT_SIZES.items()})
    values["strong"] = manager.font_spec.strong_weight
    return _QSS.substitute(values)


def hug_left(widget: QWidget) -> QWidget:
    """폭이 정해진 입력 칸을 줄의 왼쪽에 붙여 둔다(상자 배치가 가운데로 옮기지 않게)."""
    holder = QWidget()
    layout = QHBoxLayout(holder)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(widget)
    layout.addStretch(1)
    holder.setFocusProxy(widget)
    return holder


def apply_page_style(widget: QWidget) -> None:
    """화면(또는 따로 뜨는 창)에 전용 스타일을 입히고 테마가 바뀌면 다시 입힌다."""

    def refresh(*_args) -> None:
        try:
            widget.setStyleSheet(build_stylesheet())
        except RuntimeError:
            pass

    refresh()
    ThemeManager.instance().themeChanged.connect(refresh)
