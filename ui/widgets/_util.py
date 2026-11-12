"""위젯 공통 도우미."""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QPainter, QPen
from PySide6.QtWidgets import QLabel, QWidget

from ui.theme import ThemeManager

_KEYBOARD_REASONS = (Qt.TabFocusReason, Qt.BacktabFocusReason, Qt.ShortcutFocusReason)
# 창을 오가거나 메뉴가 떴다 닫힐 때는 링 상태를 그대로 둔다.
_PASSIVE_REASONS = (Qt.ActiveWindowFocusReason, Qt.PopupFocusReason)


def set_prop(widget: QWidget, name: str, value) -> None:
    """동적 속성을 바꾸고 스타일을 다시 입힌다."""
    if widget.property(name) == value:
        return
    widget.setProperty(name, value)
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()


def label(text: str = "", role: str | None = None, wrap: bool = False, parent: QWidget | None = None) -> QLabel:
    """role: display, heading, title, strong, secondary, muted, label, caption, hint, link."""
    lb = QLabel(text, parent)
    if role:
        lb.setProperty("role", role)
    lb.setWordWrap(wrap)
    return lb


def theme_color(role: str):
    return ThemeManager.instance().theme.color(role)


def on_theme_changed(slot) -> None:
    ThemeManager.instance().themeChanged.connect(slot)


class FocusRingMixin:
    """키보드로 포커스가 왔거나 키를 누른 뒤에만 링을 그린다. 마우스로 누를 때는 그리지 않는다."""

    _keyboard_focus = False

    def focusInEvent(self, event) -> None:
        if event.reason() not in _PASSIVE_REASONS:
            self._keyboard_focus = event.reason() in _KEYBOARD_REASONS
        super().focusInEvent(event)
        self.update()

    def focusOutEvent(self, event) -> None:
        if event.reason() not in _PASSIVE_REASONS:
            self._keyboard_focus = False
        super().focusOutEvent(event)
        self.update()

    def keyPressEvent(self, event) -> None:
        if not self._keyboard_focus:
            self._keyboard_focus = True
            self.update()
        super().keyPressEvent(event)

    def shows_focus_ring(self) -> bool:
        return self._keyboard_focus and self.hasFocus()

    def paint_focus_ring(self, painter: QPainter, radius: float, role: str = "focus", inset: float = 0.0) -> None:
        if not self.shows_focus_ring():
            return
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(QPen(theme_color(role), 2))
        painter.setBrush(Qt.NoBrush)
        d = 1.0 + inset
        rect = QRectF(self.rect()).adjusted(d, d, -d, -d)
        r = max(0.0, radius - inset)
        painter.drawRoundedRect(rect, r, r)
        painter.restore()
