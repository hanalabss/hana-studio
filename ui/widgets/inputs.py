"""입력: 콤보 상자, 찾기 칸, 날짜 칸, 숫자 스테퍼(mm)."""

from __future__ import annotations

import re

from PySide6.QtCore import QDate, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QTextCharFormat
from PySide6.QtWidgets import (
    QAbstractItemView, QCalendarWidget, QComboBox, QDateEdit, QFrame, QHBoxLayout, QLineEdit, QListView,
    QSizePolicy, QStyle, QStyledItemDelegate, QToolButton, QWidget,
)

from ui.theme import mark_themed

from ._util import on_theme_changed, theme_color
from .buttons import IconButton
from .icons import paint_icon

# 달력 칸이 손가락·마우스로 누르기 넉넉하도록 잡는 가장 작은 크기(날짜 한 칸이 약 40x36).
_CALENDAR_MIN_SIZE = (296, 300)
_NUMBER = re.compile(r"[-+]?\d+(?:[.,]\d+)?")


class ComboBox(QComboBox):
    """목록 줄 높이까지 테마를 따르는 콤보 상자."""

    def __init__(self, items: list[str] | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self.setView(QListView(self))
        self.setFixedHeight(40)
        if items:
            self.addItems(items)

    def wheelEvent(self, event) -> None:
        # 스크롤하다가 값이 바뀌는 사고를 막는다.
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class SearchField(QLineEdit):
    def __init__(self, placeholder: str = "", parent: QWidget | None = None):
        super().__init__(parent)
        self.setPlaceholderText(placeholder)
        self.setClearButtonEnabled(True)
        self.setFixedHeight(40)
        self.setTextMargins(24, 0, 0, 0)

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        role = "text_muted" if self.isEnabled() else "text_disabled"
        paint_icon(painter, "search", QRectF(12, (self.height() - 16) / 2, 16, 16), role)
        painter.end()


class _CalendarCellDelegate(QStyledItemDelegate):
    """달력 한 칸. 고른 날은 둥근 강조색, 다른 달 날짜는 흐리게 그린다.

    스타일시트의 ::item 규칙은 모델이 준 글자색(다른 달 = 흐린 색)을 덮어써 버려서 직접 그린다.
    색은 그릴 때마다 테마에서 읽는다.
    """

    def paint(self, painter: QPainter, option, index) -> None:
        text = str(index.data(Qt.DisplayRole) or "")
        is_day = text.isdigit()
        enabled = bool(option.state & QStyle.State_Enabled)
        cell = QRectF(option.rect).adjusted(2, 2, -2, -2)
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(Qt.NoPen)
        selected = is_day and enabled and bool(option.state & QStyle.State_Selected)
        if selected:
            painter.setBrush(theme_color("accent"))
            painter.drawRoundedRect(cell, 6, 6)
        elif is_day and enabled and option.state & QStyle.State_MouseOver:
            painter.setBrush(theme_color("surface_hover"))
            painter.drawRoundedRect(cell, 6, 6)
        foreground = index.data(Qt.ForegroundRole)
        if selected:
            color = theme_color("text_on_accent")
        elif not enabled:
            color = theme_color("text_disabled")
        elif isinstance(foreground, QBrush):
            color = foreground.color()
        elif isinstance(foreground, QColor):
            color = foreground
        else:
            color = theme_color("text")
        painter.setPen(color)
        painter.setFont(option.font)
        painter.drawText(option.rect, Qt.AlignCenter, text)
        painter.restore()


def style_calendar(calendar: QCalendarWidget) -> None:
    """달력을 테마에 맞춘다. 색과 모양은 스타일시트가 맡고, 여기서는 스타일시트로 못 하는 것만 한다.

    (주말 빨간 글씨와 요일 줄 색을 테마 색으로, 다른 달 날짜는 흐리게, 고른 날은 둥글게,
    격자와 주 번호 숨김, 이전·다음 달 풍선 도움말)
    테마가 바뀌면 다시 불러야 한다. DateField는 알아서 다시 부른다.
    """
    view = calendar.findChild(QAbstractItemView, "qt_calendar_calendarview")
    if view is not None and not isinstance(view.itemDelegate(), _CalendarCellDelegate):
        view.setItemDelegate(_CalendarCellDelegate(view))
        view.setMouseTracking(True)
        view.viewport().setAttribute(Qt.WA_Hover, True)
    mark_themed(calendar)
    # 바깥 테두리를 스타일시트로 그리게 하고, 안쪽 내용이 그 선을 덮지 않게 1px 물린다.
    calendar.setAttribute(Qt.WA_StyledBackground, True)
    if calendar.layout() is not None:
        calendar.layout().setContentsMargins(1, 1, 1, 1)
    calendar.setMinimumSize(*_CALENDAR_MIN_SIZE)
    calendar.setGridVisible(False)
    calendar.setVerticalHeaderFormat(QCalendarWidget.NoVerticalHeader)
    calendar.setHorizontalHeaderFormat(QCalendarWidget.ShortDayNames)
    day = QTextCharFormat()
    day.setForeground(theme_color("text"))
    for weekday in (Qt.Monday, Qt.Tuesday, Qt.Wednesday, Qt.Thursday, Qt.Friday, Qt.Saturday, Qt.Sunday):
        calendar.setWeekdayTextFormat(weekday, day)
    head = QTextCharFormat()
    head.setForeground(theme_color("text_muted"))
    head.setBackground(theme_color("surface_raised"))
    calendar.setHeaderTextFormat(head)
    for name, tip in (("qt_calendar_prevmonth", "이전 달"), ("qt_calendar_nextmonth", "다음 달")):
        button = calendar.findChild(QToolButton, name)
        if button is not None:
            button.setToolTip(tip)
            button.setAccessibleName(tip)


class DateField(QDateEdit):
    """달력에서 고르는 날짜 칸. 달력 팝업까지 테마를 따른다.

    empty_text를 주면 맨 앞 날짜(EMPTY_DATE)를 '값 없음'으로 쓰고 그 글자를 보여 준다.
    """

    EMPTY_DATE = QDate(1900, 1, 1)

    def __init__(self, date: QDate | None = None, empty_text: str | None = None,
                 display_format: str = "yyyy-MM-dd", parent: QWidget | None = None):
        super().__init__(parent)
        self.setCalendarPopup(True)
        self.setDisplayFormat(display_format)
        self.setFixedHeight(40)
        self._empty_text = empty_text
        if empty_text is not None:
            self.setMinimumDate(self.EMPTY_DATE)
            self.setSpecialValueText(empty_text)
        if date is not None and date.isValid():
            self.setDate(date)
        elif empty_text is not None:
            self.setDate(self.EMPTY_DATE)
        else:
            self.setDate(QDate.currentDate())
        self._restyle()
        on_theme_changed(self._restyle)

    def _restyle(self, *_args) -> None:
        calendar = self.calendarWidget()
        if calendar is not None:
            style_calendar(calendar)

    def is_empty(self) -> bool:
        return self._empty_text is not None and self.date() == self.EMPTY_DATE

    def clear_date(self) -> None:
        """값 없음으로 돌린다(empty_text를 준 칸만)."""
        if self._empty_text is not None:
            self.setDate(self.EMPTY_DATE)

    def wheelEvent(self, event) -> None:
        # 스크롤하다가 날짜가 바뀌는 사고를 막는다.
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class StepperField(QFrame):
    """[−] 값 [+]. 값을 직접 쳐 넣을 수도 있고, 위·아래 화살표 키로도 바꾼다."""

    valueChanged = Signal(float)

    def __init__(self, value: float = 0.0, minimum: float = 0.0, maximum: float = 999.0,
                 step: float = 1.0, decimals: int = 0, suffix: str = "", parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("Stepper")
        self._min, self._max = float(minimum), float(maximum)
        self._step = float(step)
        self._decimals = int(decimals)
        self._suffix = suffix
        self._value = self._clamp(value)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self._minus = IconButton("minus", "줄이기", variant="secondary")
        self._plus = IconButton("plus", "늘리기", variant="secondary")
        for button in (self._minus, self._plus):
            button.setProperty("stepper", "true")
            button.setAutoRepeat(True)
            button.setAutoRepeatDelay(350)
            button.setAutoRepeatInterval(60)
            button.setFocusPolicy(Qt.TabFocus)
        self._edit = QLineEdit(self)
        self._edit.setObjectName("StepperValue")
        self._edit.setAlignment(Qt.AlignCenter)
        self._edit.setFixedHeight(40)
        self._edit.setMinimumWidth(84)
        self._edit.installEventFilter(self)
        layout.addWidget(self._minus)
        layout.addWidget(self._edit, 1)
        layout.addWidget(self._plus)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.setFocusProxy(self._edit)

        self._minus.clicked.connect(lambda: self.step_by(-1))
        self._plus.clicked.connect(lambda: self.step_by(1))
        self._edit.textEdited.connect(self._on_text_edited)
        self._edit.editingFinished.connect(self._commit_text)
        self._render()

    # ---- 값 ----
    def value(self) -> float:
        return self._value

    def set_value(self, value: float, emit: bool = True) -> None:
        value = self._clamp(value)
        changed = value != self._value
        self._value = value
        self._render()
        if changed and emit:
            self.valueChanged.emit(value)

    def set_range(self, minimum: float, maximum: float) -> None:
        self._min, self._max = float(minimum), float(maximum)
        self.set_value(self._value)

    def set_step(self, step: float) -> None:
        self._step = float(step)

    def set_suffix(self, suffix: str) -> None:
        self._suffix = suffix
        self._render()

    def step_by(self, steps: int) -> None:
        self.set_value(self._value + steps * self._step)

    def text(self) -> str:
        return self._edit.text()

    def line_edit(self) -> QLineEdit:
        return self._edit

    def setAccessibleName(self, name: str) -> None:
        super().setAccessibleName(name)
        self._edit.setAccessibleName(name)
        self._minus.setAccessibleName(f"{name} 줄이기")
        self._plus.setAccessibleName(f"{name} 늘리기")

    # ---- 내부 ----
    def _clamp(self, value: float) -> float:
        return round(min(self._max, max(self._min, float(value))), self._decimals)

    def _render(self) -> None:
        number = f"{self._value:.{self._decimals}f}"
        self._edit.setText(f"{number} {self._suffix}".strip())
        self._minus.setEnabled(self.isEnabled() and self._value > self._min)
        self._plus.setEnabled(self.isEnabled() and self._value < self._max)

    def _on_text_edited(self, text: str) -> None:
        """치는 도중에도 값을 바로 반영한다. 글자는 건드리지 않는다(커서가 튀지 않게).

        Enter나 포커스 이동을 기다리면, 숫자만 치고 바로 인쇄를 누를 때 예전 값이 쓰인다.
        """
        match = _NUMBER.search(text)
        if not match:
            return
        value = self._clamp(float(match.group(0).replace(",", ".")))
        if value == self._value:
            return
        self._value = value
        self._minus.setEnabled(self.isEnabled() and value > self._min)
        self._plus.setEnabled(self.isEnabled() and value < self._max)
        self.valueChanged.emit(value)

    def _commit_text(self) -> None:
        match = _NUMBER.search(self._edit.text())
        if match:
            self.set_value(float(match.group(0).replace(",", ".")))
        else:
            self._render()

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() == event.Type.EnabledChange and hasattr(self, "_edit"):
            self._render()

    def eventFilter(self, obj, event) -> bool:
        if obj is self._edit and event.type() == event.Type.KeyPress:
            if event.key() == Qt.Key_Up:
                self.step_by(1)
                return True
            if event.key() == Qt.Key_Down:
                self.step_by(-1)
                return True
        return super().eventFilter(obj, event)


class MmField(StepperField):
    """mm 단위 숫자 칸. 기본 0.5mm씩 움직인다."""

    def __init__(self, value: float = 0.0, minimum: float = 0.0, maximum: float = 200.0,
                 step: float = 0.5, decimals: int = 1, parent: QWidget | None = None):
        super().__init__(value, minimum, maximum, step, decimals, "mm", parent)
