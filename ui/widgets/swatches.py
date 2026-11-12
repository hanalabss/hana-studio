"""색 고르기: 자주 쓰는 색 몇 개와 '다른 색'."""

from __future__ import annotations

from typing import Sequence

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QAbstractButton, QButtonGroup, QColorDialog, QHBoxLayout, QSizePolicy, QWidget

from ui.theme import SWATCH_PRESETS, contrast_ratio

from ._util import FocusRingMixin, theme_color
from .icons import paint_icon

_SIZE = 36


def _normalize(color: str) -> str:
    return QColor(color).name().upper()


class _Swatch(FocusRingMixin, QAbstractButton):
    def __init__(self, color: str | None, name: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.color = _normalize(color) if color else None
        self.setCheckable(color is not None)
        self.setFixedSize(_SIZE, _SIZE)
        self.setToolTip(name)
        self.setAccessibleName(name)
        self.setAttribute(Qt.WA_Hover, True)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        center = QPointF(_SIZE / 2, _SIZE / 2)
        if self.color is None:
            pen = QPen(theme_color("indicator_line"), 1.2, Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(theme_color("surface_hover") if self.underMouse() else Qt.NoBrush)
            painter.drawEllipse(center, 12, 12)
            paint_icon(painter, "plus", QRectF(11, 11, 14, 14), "text_secondary")
        else:
            if self.isChecked():
                painter.setPen(QPen(theme_color("swatch_ring"), 2))
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(center, 16, 16)
            fill = QColor(self.color)
            surface = theme_color("surface").name()
            needs_edge = contrast_ratio(self.color, surface) < 1.5
            painter.setPen(QPen(theme_color("indicator_line"), 1) if needs_edge else Qt.NoPen)
            painter.setBrush(fill)
            radius = 12 if self.isChecked() or not self.underMouse() else 13
            painter.drawEllipse(center, radius, radius)
            if not self.isEnabled():
                veil = theme_color("surface")
                veil.setAlpha(150)
                painter.setPen(Qt.NoPen)
                painter.setBrush(veil)
                painter.drawEllipse(center, 13, 13)
        if self.shows_focus_ring():
            painter.setPen(QPen(theme_color("focus"), 2, Qt.DotLine))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(QRectF(1, 1, _SIZE - 2, _SIZE - 2), 6, 6)
        painter.end()


class SwatchPicker(QWidget):
    """colorChanged는 "#RRGGBB"를 보낸다. colors는 (이름, "#RRGGBB") 목록."""

    colorChanged = Signal(str)

    def __init__(self, colors: Sequence[tuple[str, str]] = SWATCH_PRESETS, current: str | None = None,
                 allow_custom: bool = True, parent: QWidget | None = None):
        super().__init__(parent)
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(4)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._swatches: dict[str, _Swatch] = {}
        self._current: str | None = None
        self._custom: _Swatch | None = None
        for name, color in colors:
            self._add(name, color)
        self._more: _Swatch | None = None
        if allow_custom:
            self._more = _Swatch(None, "다른 색 고르기", self)
            self._more.clicked.connect(self._pick_custom)
            self._layout.addWidget(self._more)
        self._layout.addStretch(1)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self._group.buttonToggled.connect(self._on_toggled)
        if current:
            self.set_current(current, emit=False)

    def _add(self, name: str, color: str) -> _Swatch:
        swatch = _Swatch(color, name, self)
        self._group.addButton(swatch)
        self._swatches[swatch.color] = swatch
        index = self._layout.indexOf(self._more) if getattr(self, "_more", None) else -1
        if index >= 0:
            self._layout.insertWidget(index, swatch)
        else:
            self._layout.addWidget(swatch)
        if self.isVisible():
            swatch.show()
        return swatch

    def colors(self) -> list[str]:
        return list(self._swatches)

    def current(self) -> str | None:
        return self._current

    def set_current(self, color: str, emit: bool = True) -> None:
        """목록에 없는 색이면 '고른 색' 칸을 하나 만들어 넣는다."""
        color = _normalize(color)
        if color not in self._swatches:
            if self._custom is not None:
                del self._swatches[self._custom.color]
                self._custom.color = color
                self._custom.setToolTip(f"고른 색 {color}")
                self._custom.setAccessibleName(f"고른 색 {color}")
                self._swatches[color] = self._custom
                self._custom.update()
            else:
                self._custom = self._add(f"고른 색 {color}", color)
        if color == self._current:
            return
        if not emit:
            self._group.blockSignals(True)
        self._swatches[color].setChecked(True)
        if not emit:
            self._group.blockSignals(False)
            self._current = color
        elif self._current != color:
            # 같은 버튼의 색만 바뀐 경우 toggled가 오지 않는다.
            self._current = color
            self.colorChanged.emit(color)

    def _on_toggled(self, swatch: _Swatch, checked: bool) -> None:
        if checked and swatch.color != self._current:
            self._current = swatch.color
            self.colorChanged.emit(swatch.color)

    def _pick_custom(self) -> None:
        initial = QColor(self._current) if self._current else QColor(Qt.black)
        chosen = QColorDialog.getColor(initial, self, "색 고르기")
        if chosen.isValid():
            self.set_current(chosen.name())
