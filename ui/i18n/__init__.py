"""화면 언어(한국어·English·日本語·中文). 다눌의 Language 메뉴와 같은 일을 한다.

다눌처럼 메뉴·버튼·제목 같은 짧은 문구를 번역표(ui/i18n/<언어>.json, 한국어 → 그 언어)로 바꾼다.
번역표에 없는 문구(안내 문장, 사람 이름 같은 명단 값)는 한국어 그대로 보인다.

바꾸는 곳은 화면에 보이는 글자뿐이다: 글자 칸·버튼·메뉴 항목·창 제목·도움말·입력 칸 안내 글.
고르는 목록(ComboBox)의 항목과 사람이 입력한 글자는 값으로도 쓰이므로 건드리지 않는다.
언어는 프로그램을 다시 시작할 때 적용한다. 한국어면 아무것도 설치하지 않는다(동작이 전과 똑같다).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from PySide6.QtCore import QEvent, QObject
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QAbstractButton, QApplication, QGroupBox, QLabel, QLineEdit, QMenu, QPlainTextEdit, QWidget,
)

log = logging.getLogger(__name__)

KEY_LANGUAGE = "view/language"
DEFAULT_LANGUAGE = "ko"
LANGUAGES: tuple[tuple[str, str], ...] = (("ko", "한국어"), ("en", "English"), ("ja", "日本語"), ("zh", "中文"))
SKIP_PROPERTY = "i18n_skip"      # 이 속성이 참인 위젯은 번역하지 않는다(명단 값을 보여 주는 칩 등)
_DIR = Path(__file__).resolve().parent

# 일본어·중국어 글자가 □로 보이지 않게, 화면 글꼴 뒤에 붙일 그 언어의 Windows 기본 글꼴.
FALLBACK_FONTS = {
    "ja": (("Yu Gothic UI", "Yu Gothic", "Meiryo UI", "Meiryo", "MS UI Gothic"), ("YuGothM.ttc", "YuGothR.ttc", "meiryo.ttc")),
    "zh": (("Microsoft YaHei UI", "Microsoft YaHei", "SimSun"), ("msyh.ttc", "simsun.ttc")),
}

_table: dict[str, str] = {}
_language = DEFAULT_LANGUAGE
_installed = False


def stored_language(settings=None) -> str:
    if settings is None:
        from ui.theme import app_settings
        settings = app_settings()
    value = str(settings.value(KEY_LANGUAGE, DEFAULT_LANGUAGE) or DEFAULT_LANGUAGE)
    return value if value in dict(LANGUAGES) else DEFAULT_LANGUAGE


def save_language(code: str, settings=None) -> None:
    if code not in dict(LANGUAGES):
        return
    if settings is None:
        from ui.theme import app_settings
        settings = app_settings()
    settings.setValue(KEY_LANGUAGE, code)
    settings.sync()


def load_table(code: str) -> dict[str, str]:
    if code == DEFAULT_LANGUAGE:
        return {}
    try:
        data = json.loads((_DIR / f"{code}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        log.warning("번역표를 읽지 못했습니다: %s", code)
        return {}
    return {str(k): str(v) for k, v in data.items() if v}


def current_language() -> str:
    return _language


def translate(text: str) -> str:
    """번역표에 있으면 바꾼 글자, 없으면 그대로. '메뉴\\tCtrl+C'처럼 탭 뒤 단축키는 그대로 둔다."""
    if not _table or not text:
        return text
    found = _table.get(text)
    if found is not None:
        return found
    if "\t" in text:
        head, _, tail = text.partition("\t")
        found = _table.get(head)
        if found is not None:
            return f"{found}\t{tail}"
    return text


def _skipped(widget) -> bool:
    obj = widget
    while obj is not None:
        if isinstance(obj, QObject) and obj.property(SKIP_PROPERTY):
            return True
        obj = obj.parent() if isinstance(obj, QObject) else None
    return False


def _translate_widget(widget: QWidget) -> None:
    if _skipped(widget):
        return
    if isinstance(widget, (QLabel, QAbstractButton)):
        text = widget.text()
        new = translate(text)
        if new != text:
            _ORIGINAL["setText"](widget, new) if isinstance(widget, QLabel) else _ORIGINAL["btnSetText"](widget, new)
    if isinstance(widget, QGroupBox):
        widget.setTitle(translate(widget.title()))
    if isinstance(widget, (QLineEdit, QPlainTextEdit)):
        placeholder = widget.placeholderText()
        if placeholder:
            widget.setPlaceholderText(translate(placeholder))
    tip = widget.toolTip()
    if tip:
        _ORIGINAL["setToolTip"](widget, translate(tip))
    if widget.isWindow() and widget.windowTitle():
        _ORIGINAL["setWindowTitle"](widget, translate(widget.windowTitle()))
    accessible = widget.accessibleName()
    if accessible:
        widget.setAccessibleName(translate(accessible))


def _translate_menu(menu: QMenu) -> None:
    menu.setTitle(translate(menu.title()))
    for action in menu.actions():
        _ORIGINAL["actionSetText"](action, translate(action.text()))
        if action.toolTip():
            action.setToolTip(translate(action.toolTip()))
        if action.menu() is not None:
            action.menu().setTitle(translate(action.menu().title()))


class _Filter(QObject):
    """새로 보이는 위젯의 글자를 번역한다(생성자에서 넣은 글자는 setText를 거치지 않기 때문)."""

    def eventFilter(self, obj, event) -> bool:
        kind = event.type()
        if kind in (QEvent.Type.Polish, QEvent.Type.Show) and isinstance(obj, QWidget):
            try:
                if isinstance(obj, QMenu):
                    _translate_menu(obj)
                else:
                    _translate_widget(obj)
            except RuntimeError:
                pass
        return False


_ORIGINAL: dict = {}
_filter: _Filter | None = None


def _patch() -> None:
    """나중에 setText 등으로 바꾸는 글자도 번역되게 한다."""
    _ORIGINAL.update(
        setText=QLabel.setText, btnSetText=QAbstractButton.setText, actionSetText=QAction.setText,
        setToolTip=QWidget.setToolTip, setWindowTitle=QWidget.setWindowTitle,
    )

    def label_set_text(self, text):
        _ORIGINAL["setText"](self, text if _skipped(self) else translate(text))

    def button_set_text(self, text):
        _ORIGINAL["btnSetText"](self, text if _skipped(self) else translate(text))

    def action_set_text(self, text):
        _ORIGINAL["actionSetText"](self, translate(text))

    def set_tool_tip(self, text):
        _ORIGINAL["setToolTip"](self, translate(text))

    def set_window_title(self, text):
        _ORIGINAL["setWindowTitle"](self, translate(text))

    QLabel.setText = label_set_text
    QAbstractButton.setText = button_set_text
    QAction.setText = action_set_text
    QWidget.setToolTip = set_tool_tip
    QWidget.setWindowTitle = set_window_title


def _unpatch() -> None:
    if not _ORIGINAL:
        return
    QLabel.setText = _ORIGINAL["setText"]
    QAbstractButton.setText = _ORIGINAL["btnSetText"]
    QAction.setText = _ORIGINAL["actionSetText"]
    QWidget.setToolTip = _ORIGINAL["setToolTip"]
    QWidget.setWindowTitle = _ORIGINAL["setWindowTitle"]
    _ORIGINAL.clear()


def install(app: QApplication, code: str | None = None) -> str:
    """언어를 적용한다. 한국어면 아무것도 하지 않는다. 적용한 언어 코드를 돌려준다."""
    global _table, _language, _installed, _filter
    code = code or stored_language()
    uninstall(app)
    _language = code
    _table = load_table(code)
    if not _table:
        _language = DEFAULT_LANGUAGE
        return _language
    _add_fallback_fonts(app, code)
    _patch()
    _filter = _Filter(app)
    app.installEventFilter(_filter)
    _installed = True
    return _language


def _add_fallback_fonts(app: QApplication, code: str) -> None:
    families, files = FALLBACK_FONTS.get(code, ((), ()))
    if not families:
        return
    import os

    from PySide6.QtGui import QFontDatabase
    available = set(QFontDatabase.families())
    if not available & set(families):
        folder = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
        for name in files:
            if (folder / name).is_file():
                QFontDatabase.addApplicationFont(str(folder / name))
        available = set(QFontDatabase.families())
    extra = [name for name in families if name in available]
    if extra:
        font = app.font()
        font.setFamilies([*font.families(), *[n for n in extra if n not in font.families()]])
        app.setFont(font)


def uninstall(app: QApplication | None = None) -> None:
    global _table, _language, _installed, _filter
    if _filter is not None and app is not None:
        app.removeEventFilter(_filter)
    _filter = None
    _unpatch()
    _table = {}
    _language = DEFAULT_LANGUAGE
    _installed = False
