"""디자인 화면 전용 스타일. 색은 모두 테마 토큰에서 가져온다."""

from __future__ import annotations

from string import Template

from ui.editor import EditorPalette
from ui.theme import FONT_SIZES, Theme, ThemeManager

SIDE_WIDTH = 240
INSPECTOR_WIDTH = 320
BAR_HEIGHT = 48

_QSS = Template("""
QFrame#DesignSide { background: $surface; border: 0; border-right: 1px solid $line; }
QFrame#DesignInspector { background: $surface; border: 0; border-left: 1px solid $line; }
QFrame#CanvasBar { background: $surface; border: 0; border-bottom: 1px solid $line; }
QFrame#PreviewBar { background: $surface; border: 0; border-top: 1px solid $line; }
QFrame#ThumbStrip { background: $surface_sunken; border: 0; border-top: 1px solid $line; }
QWidget#StartPanel, QWidget#StartBody { background: $ground; }
QWidget#InspectorBody { background: $surface; }

QPushButton[glyph="true"] {
    min-height: 38px; padding: 0; border: 1px solid $control_line; border-radius: 8px;
    background: $surface_raised; color: $text; font-size: ${fs_title}px;
}
QPushButton[glyph="true"]:hover { background: $surface_hover; }
QPushButton[glyph="true"]:checked { background: $strong_bg; color: $strong_text; border-color: $strong_bg; }

QPushButton[disclosure="true"] {
    min-height: 38px; padding: 0 4px; border: 0; border-radius: 6px; text-align: left;
    background: transparent; color: $accent_text; font-weight: $strong; font-size: ${fs_small}px;
}
QPushButton[disclosure="true"]:hover { background: $surface_hover; }

QPushButton[link="true"] {
    min-height: 26px; padding: 0 1px; border: 0; border-bottom: 1px solid transparent; border-radius: 0;
    background: transparent; color: $accent_text; font-weight: $strong; font-size: ${fs_small}px; text-align: left;
}
QPushButton[link="true"]:hover, QPushButton[link="true"]:focus { border-bottom-color: $accent_text; }

QPushButton[zoom="true"] {
    min-height: 30px; padding: 0 6px; border: 1px solid transparent; border-radius: 6px;
    background: transparent; color: $text; font-size: ${fs_small}px;
}
QPushButton[zoom="true"]:hover { background: $surface_hover; }
QPushButton[zoom="true"]::menu-indicator { image: none; width: 0; }
QPushButton[menu="true"]::menu-indicator { image: none; width: 0; }

QLabel#WarnChip {
    background: $warn_bg; color: $warn_text; border: 1px solid $warn_line; border-radius: 14px;
    padding: 0 12px; font-size: ${fs_small}px;
}
QLabel#PreviewName { font-size: ${fs_body}px; }

QFrame#ProblemRow { background: $warn_bg; border: 1px solid $warn_line; border-radius: 8px; }
QFrame#ProblemRow:hover { border-color: $warn_text; }
QFrame#ProblemRow QLabel { color: $warn_text; font-size: ${fs_small}px; }
QFrame#OkRow { background: $ok_bg; border: 0; border-radius: 8px; }
QFrame#OkRow QLabel { color: $ok_text; font-size: ${fs_small}px; }
QFrame#NoteRow { background: $surface_sunken; border: 0; border-radius: 8px; }
QFrame#NoteRow QLabel { color: $text_secondary; font-size: ${fs_small}px; }

QFrame#TemplateCard { background: $surface; border: 1px solid $line; border-radius: 12px; }
QFrame#TemplateCard:hover { border-color: $accent; }
QFrame#TemplateCard[focused="true"] { border: 2px solid $focus; }
QFrame#StartOptions { background: $surface; border: 1px solid $line; border-radius: 12px; }

QListWidget#Layers { background: transparent; border: 0; outline: 0; }
QListWidget#Layers::item { padding: 0; border: 0; background: transparent; }
QListWidget#Layers::item:selected, QListWidget#Layers::item:hover { background: transparent; }
QListWidget#Layers QLineEdit {
    min-height: 26px; max-height: 26px; padding: 0 6px; border-radius: 6px; font-size: ${fs_small}px;
}
QListView#Thumbs { background: transparent; border: 0; outline: 0; }
QListView#Thumbs::item { padding: 0; border: 0; background: transparent; }

QPlainTextEdit#ContentEdit { min-height: 56px; max-height: 76px; }
QScrollArea#InspectorScroll { background: $surface; border: 0; }
""")


def build_stylesheet(theme: Theme | None = None) -> str:
    manager = ThemeManager.instance()
    theme = theme or manager.theme
    values: dict[str, object] = dict(theme.colors)
    values.update({f"fs_{k}": v for k, v in FONT_SIZES.items()})
    values["strong"] = manager.font_spec.strong_weight
    return _QSS.substitute(values)


def _alpha(color: str, alpha: int) -> str:
    return f"#{alpha:02X}{color.lstrip('#')}"


def palette_for(theme: Theme | None = None) -> EditorPalette:
    """테마 토큰으로 채운 캔버스 색."""
    theme = theme or ThemeManager.instance().theme
    base = EditorPalette.dark() if theme.is_dark else EditorPalette.light()
    accent = theme.hex("accent")
    cut = theme.hex("cutline")
    return base.with_changes(
        name=theme.name,
        canvas_bg=theme.hex("ground"),
        selection=accent, hover=_alpha(accent, 0x80), rubber_fill=_alpha(accent, 0x22),
        cut_line=cut, bleed_tint=_alpha(cut, 0x1A),
        warning=theme.hex("warn_mark"), locked=theme.hex("indicator_line"),
        tag_bg=accent, tag_text=theme.hex("text_on_accent"),
        readout_bg=theme.hex("tooltip_bg"), readout_text=theme.hex("tooltip_text"),
        ruler_bg=theme.hex("surface"), ruler_text=theme.hex("text_muted"),
        ruler_tick=theme.hex("control_line"), ruler_border=theme.hex("line"),
        ruler_marker=accent, ruler_span=_alpha(accent, 0x26),
    )
