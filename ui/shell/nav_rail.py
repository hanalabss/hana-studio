"""왼쪽 메뉴."""

from __future__ import annotations

from PySide6.QtCore import QRectF, QSize, Qt, Signal
from PySide6.QtGui import QFont, QPainter
from PySide6.QtWidgets import QAbstractButton, QButtonGroup, QFrame, QVBoxLayout, QWidget

from ui.theme import FONT_SIZES, SIZES, ThemeManager
from ui.widgets import Divider, FocusRingMixin, paint_icon, theme_color


class RailButton(FocusRingMixin, QAbstractButton):
    def __init__(self, page_id: str, text: str, icon: str, compact: bool = False, parent: QWidget | None = None):
        super().__init__(parent)
        self.page_id = page_id
        self._icon = icon
        self._compact = compact
        self.setText(text)
        self.setCheckable(True)
        self.setAttribute(Qt.WA_Hover, True)
        self.setFixedHeight(52 if compact else SIZES["rail_item"])
        self.setAccessibleName(text)

    def sizeHint(self) -> QSize:
        return QSize(SIZES["rail"] - 16, self.height())

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        rect = QRectF(self.rect())
        checked = self.isChecked()
        if checked:
            painter.setPen(Qt.NoPen)
            painter.setBrush(theme_color("nav_selected_bg"))
            painter.drawRoundedRect(rect, 8, 8)
        elif self.underMouse() and self.isEnabled():
            painter.setPen(Qt.NoPen)
            painter.setBrush(theme_color("nav_hover_bg"))
            painter.drawRoundedRect(rect, 8, 8)

        if not self.isEnabled():
            text_role = icon_role = "text_disabled"
        elif checked:
            text_role, icon_role = "nav_selected_text", "nav_selected_icon"
        else:
            text_role = icon_role = "nav_text"

        icon_size = 20 if self._compact else 22
        text_height = FONT_SIZES["caption"] + 6
        gap = 4
        block = icon_size + gap + text_height
        top = (rect.height() - block) / 2
        paint_icon(painter, self._icon, QRectF((rect.width() - icon_size) / 2, top, icon_size, icon_size), icon_role)

        font = QFont(self.font())
        font.setPixelSize(FONT_SIZES["caption"] if self._compact else FONT_SIZES["small"])
        if checked:
            font.setWeight(QFont.Weight(ThemeManager.instance().font_spec.strong_weight))
        painter.setFont(font)
        painter.setPen(theme_color(text_role))
        painter.drawText(QRectF(0, top + icon_size + gap, rect.width(), text_height),
                         Qt.AlignHCenter | Qt.AlignVCenter, self.text())
        self.paint_focus_ring(painter, 8)
        painter.end()


class NavRail(QFrame):
    requested = Signal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("NavRail")
        self.setFixedWidth(SIZES["rail"])
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 12, 8, 12)
        layout.setSpacing(4)
        self._main = QVBoxLayout()
        self._main.setSpacing(4)
        self._extra = QVBoxLayout()
        self._extra.setSpacing(4)
        self._bottom = QVBoxLayout()
        self._bottom.setSpacing(2)
        self._divider = Divider()
        self._divider.setVisible(False)
        layout.addLayout(self._main)
        layout.addSpacing(4)
        layout.addWidget(self._divider)
        layout.addSpacing(4)
        layout.addLayout(self._extra)
        layout.addStretch(1)
        layout.addLayout(self._bottom)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._buttons: dict[str, RailButton] = {}

    def add_item(self, page_id: str, text: str, icon: str, section: str = "main", tooltip: str = "") -> RailButton:
        """section: "main"(작업 순서), "extra"(구분선 아래), "bottom"(맨 아래 작은 항목)."""
        button = RailButton(page_id, text, icon, compact=(section == "bottom"), parent=self)
        if tooltip:
            button.setToolTip(tooltip)
        button.clicked.connect(lambda _checked=False, pid=page_id: self.requested.emit(pid))
        self._group.addButton(button)
        self._buttons[page_id] = button
        {"main": self._main, "extra": self._extra, "bottom": self._bottom}[section].addWidget(button)
        if section == "extra":
            self._divider.setVisible(True)
        return button

    def button(self, page_id: str) -> RailButton:
        return self._buttons[page_id]

    def page_ids(self) -> list[str]:
        return list(self._buttons)

    def set_current(self, page_id: str) -> None:
        button = self._buttons.get(page_id)
        if button is not None:
            button.setChecked(True)

    def current(self) -> str | None:
        checked = self._group.checkedButton()
        return checked.page_id if checked else None
