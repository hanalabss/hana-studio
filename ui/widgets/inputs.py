"""입력: 콤보 상자, 찾기 칸, 숫자 스테퍼(mm)."""

from __future__ import annotations

import re

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import QComboBox, QFrame, QHBoxLayout, QLineEdit, QListView, QSizePolicy, QWidget

from .buttons import IconButton
from .icons import paint_icon

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
