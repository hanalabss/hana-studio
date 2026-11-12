"""이 화면에서만 쓰는 입력 조각."""

from __future__ import annotations

from PySide6.QtCore import QDate, QRectF, Qt
from PySide6.QtGui import QFontMetrics, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QDateEdit, QFrame, QLabel, QVBoxLayout, QWidget

from ui.theme import ThemeManager
from ui.widgets import DateField, theme_color

EMPTY_DATE = DateField.EMPTY_DATE


class Rethemed:
    """테마가 바뀌면 다시 그린다. 위젯이 사라지면 연결도 함께 끊기도록 메서드로 잇는다."""

    def follow_theme(self) -> None:
        ThemeManager.instance().themeChanged.connect(self._theme_changed)

    def _theme_changed(self, *_args) -> None:
        target = self.viewport() if hasattr(self, "viewport") else self
        target.update()


def make_date_edit(parent: QWidget | None = None) -> QDateEdit:
    """달력에서 고르는 날짜 칸(공용 DateField). 맨 앞 값(날짜 없음)이 빈 값이다."""
    return DateField(empty_text="날짜 없음", parent=parent)


def date_value(edit: QDateEdit) -> str | None:
    return None if edit.date() == EMPTY_DATE else edit.date().toString("yyyy-MM-dd")


def set_date_value(edit: QDateEdit, value, fallback_today: bool = False) -> None:
    date = QDate.fromString(str(value or ""), "yyyy-MM-dd")
    if date.isValid():
        edit.setDate(date)
    else:
        edit.setDate(QDate.currentDate() if fallback_today else EMPTY_DATE)


class ElidedLabel(QLabel):
    """길면 가운데를 줄여 보여 주는 한 줄 글자(폴더 경로용). 전체는 툴팁으로."""

    def __init__(self, text: str = "", role: str | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self._full = text
        if role:
            self.setProperty("role", role)
        self.setMinimumWidth(40)
        self.set_full_text(text)

    def set_full_text(self, text: str) -> None:
        self._full = text
        self.setToolTip(text)
        self._apply()

    def full_text(self) -> str:
        return self._full

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._apply()

    def _apply(self) -> None:
        metrics = QFontMetrics(self.font())
        super().setText(metrics.elidedText(self._full, Qt.ElideMiddle, max(20, self.width() - 2)))

    def minimumSizeHint(self):
        hint = super().minimumSizeHint()
        hint.setWidth(40)
        return hint


class TableFrame(Rethemed, QFrame):
    """표를 감싸는 둥근 상자. 머리글 띠를 직접 그려서 스크롤 막대 위쪽까지 이어지게 한다."""

    def __init__(self, header_height: int, parent: QWidget | None = None):
        super().__init__(parent)
        self._header_height = header_height
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(1, 1, 1, 1)
        self.body.setSpacing(0)
        self.follow_theme()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        shape = QPainterPath()
        shape.addRoundedRect(rect, 12, 12)
        painter.fillPath(shape, theme_color("surface"))
        painter.save()
        painter.setClipPath(shape)
        painter.fillRect(QRectF(0, 0, self.width(), self._header_height + 1), theme_color("surface_sunken"))
        painter.setRenderHint(QPainter.Antialiasing, False)
        painter.setPen(QPen(theme_color("line"), 1))
        painter.drawLine(0, self._header_height, self.width(), self._header_height)
        painter.restore()
        painter.setPen(QPen(theme_color("line"), 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(shape)
        painter.end()
