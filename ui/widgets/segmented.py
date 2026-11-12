"""둘에서 넷 사이 선택지 중 하나를 고르는 세그먼트."""

from __future__ import annotations

from typing import Sequence

from PySide6.QtCore import QSize, Signal
from PySide6.QtGui import QFont, QFontMetrics, QPainter
from PySide6.QtWidgets import QButtonGroup, QFrame, QHBoxLayout, QPushButton, QSizePolicy, QWidget

from ui.theme import ThemeManager

from ._util import FocusRingMixin


class _Segment(FocusRingMixin, QPushButton):
    def __init__(self, key: str, text: str, parent: QWidget | None = None):
        super().__init__(text, parent)
        self.key = key
        self.setCheckable(True)
        self.setProperty("segment", "true")
        self.setFixedHeight(30)
        self.setAutoDefault(False)

    def _bold_extra(self) -> int:
        """고른 칸은 굵은 글씨로 그려진다. 보통 굵기로 잰 폭보다 얼마나 더 필요한지 돌려준다."""
        self.ensurePolished()
        bold = QFont(self.font())
        bold.setWeight(QFont.Weight(ThemeManager.instance().font_spec.strong_weight))
        text = self.text()
        extra = QFontMetrics(bold).horizontalAdvance(text) - self.fontMetrics().horizontalAdvance(text)
        return max(0, extra) + 2    # 2: 굵은 글씨 가장자리가 안쪽 여백에 닿지 않게 두는 여유

    def sizeHint(self) -> QSize:
        hint = super().sizeHint()
        return QSize(hint.width() + self._bold_extra(), hint.height())

    def minimumSizeHint(self) -> QSize:
        hint = super().minimumSizeHint()
        return QSize(max(hint.width(), self.sizeHint().width()), hint.height())

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if self.shows_focus_ring():
            painter = QPainter(self)
            if self.isChecked():
                self.paint_focus_ring(painter, 6, "strong_text", inset=2.0)
            else:
                self.paint_focus_ring(painter, 6)
            painter.end()


class SegmentedControl(QFrame):
    """options는 "글자" 또는 (key, "글자"). currentChanged는 key를 보낸다.

    칸 너비는 굵은 글씨(고른 상태) 기준으로 잡으므로, 어느 칸을 골라도 글자가 잘리지 않고
    고를 때마다 너비가 흔들리지도 않는다.
    """

    currentChanged = Signal(str)

    def __init__(self, options: Sequence[str | tuple[str, str]], current: str | None = None,
                 stretch: bool = False, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("Segmented")
        self.setFixedHeight(40)
        self.setSizePolicy(QSizePolicy.Expanding if stretch else QSizePolicy.Maximum, QSizePolicy.Fixed)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._segments: dict[str, _Segment] = {}
        for option in options:
            key, text = (option, option) if isinstance(option, str) else option
            segment = _Segment(key, text, self)
            if stretch:
                segment.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self._group.addButton(segment)
            self._segments[key] = segment
            layout.addWidget(segment)
        if not self._segments:
            raise ValueError("선택지가 하나 이상 필요합니다.")
        self._current = current if current in self._segments else next(iter(self._segments))
        self._segments[self._current].setChecked(True)
        self._group.buttonToggled.connect(self._on_toggled)

    def keys(self) -> list[str]:
        return list(self._segments)

    def current(self) -> str:
        return self._current

    def set_current(self, key: str, emit: bool = True) -> None:
        if key not in self._segments:
            raise KeyError(key)
        if key == self._current:
            return
        if emit:
            self._segments[key].setChecked(True)
            return
        self._group.blockSignals(True)
        self._segments[key].setChecked(True)
        self._group.blockSignals(False)
        self._current = key

    def set_option_enabled(self, key: str, enabled: bool) -> None:
        self._segments[key].setEnabled(enabled)

    def button(self, key: str) -> QPushButton:
        return self._segments[key]

    def _on_toggled(self, button: _Segment, checked: bool) -> None:
        if checked and button.key != self._current:
            self._current = button.key
            self.currentChanged.emit(button.key)
