"""3단계 안내: 카드 꾸미기 → 명단 넣기 → 인쇄하기."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QFont, QFontMetricsF, QPainter, QPen
from PySide6.QtWidgets import QAbstractButton, QFrame, QHBoxLayout, QWidget

from ui.theme import FONT_SIZES, SIZES, ThemeManager
from ui.widgets import FocusRingMixin, QuietButton, paint_icon, theme_color

STEP_STATES = ("current", "done", "upcoming")
_STATE_WORD = {"current": "지금 하는 일", "done": "끝남", "upcoming": "아직"}


class StepPill(FocusRingMixin, QAbstractButton):
    _HEIGHT = 40
    _MARK = 26

    def __init__(self, page_id: str, number: int, text: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.page_id = page_id
        self.number = number
        self._state = "upcoming"
        self.setText(text)
        self.setAttribute(Qt.WA_Hover, True)
        self.setFixedHeight(self._HEIGHT)
        self.setCursor(Qt.PointingHandCursor)
        self._describe()

    def state(self) -> str:
        return self._state

    def set_state(self, state: str) -> None:
        if state not in STEP_STATES:
            raise ValueError(state)
        if state != self._state:
            self._state = state
            self._describe()
            self.updateGeometry()
            self.update()

    def _describe(self) -> None:
        self.setAccessibleName(f"{self.number}단계 {self.text()}, {_STATE_WORD[self._state]}")
        self.setToolTip(f"{self.text()} 화면으로 가기")

    def _font(self, strong: bool) -> QFont:
        font = QFont(self.font())
        font.setPixelSize(FONT_SIZES["body"])
        if strong:
            font.setWeight(QFont.Weight(ThemeManager.instance().font_spec.strong_weight))
        return font

    def sizeHint(self) -> QSize:
        metrics = QFontMetricsF(self._font(True))
        return QSize(round(7 + self._MARK + 10 + metrics.horizontalAdvance(self.text()) + 18), self._HEIGHT)

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        radius = rect.height() / 2
        state = self._state
        hover = self.underMouse()

        if state == "current":
            painter.setPen(Qt.NoPen)
            painter.setBrush(theme_color("step_current_bg"))
            text_role, mark_bg, mark_text = "step_current_text", "step_current_mark_bg", "step_current_mark_text"
        elif state == "done":
            painter.setPen(QPen(theme_color("step_done_mark_bg"), 1) if hover else Qt.NoPen)
            painter.setBrush(theme_color("step_done_bg"))
            text_role, mark_bg, mark_text = "step_done_text", "step_done_mark_bg", "step_done_mark_text"
        else:
            painter.setPen(QPen(theme_color("indicator_line" if hover else "control_line"), 1))
            painter.setBrush(theme_color("surface_hover" if hover else "surface"))
            text_role, mark_bg, mark_text = "text_secondary", "surface_pressed", "text_secondary"
        painter.drawRoundedRect(rect, radius, radius)

        mark = QRectF(7, (self.height() - self._MARK) / 2, self._MARK, self._MARK)
        painter.setPen(Qt.NoPen)
        painter.setBrush(theme_color(mark_bg))
        painter.drawEllipse(mark)
        if state == "done":
            paint_icon(painter, "check", mark.adjusted(6, 6, -6, -6), mark_text, stroke=2.6)
        else:
            painter.setFont(self._font(True))
            painter.setPen(theme_color(mark_text))
            painter.drawText(mark, Qt.AlignCenter, str(self.number))

        painter.setFont(self._font(state == "current"))
        painter.setPen(theme_color(text_role))
        text_rect = QRectF(mark.right() + 10, 0, self.width() - mark.right() - 10 - 8, self.height())
        painter.drawText(text_rect, Qt.AlignLeft | Qt.AlignVCenter, self.text())

        if self.shows_focus_ring():
            if state == "current":
                self.paint_focus_ring(painter, radius, "step_current_text", inset=2.0)
            else:
                self.paint_focus_ring(painter, radius)
        painter.end()


class _Connector(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._done = False
        self.setFixedSize(36, 40)

    def set_done(self, done: bool) -> None:
        if done != self._done:
            self._done = done
            self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        pen = QPen(theme_color("step_done_mark_bg" if self._done else "control_line"), 2)
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        y = self.height() / 2
        painter.drawLine(QPointF(4, y), QPointF(self.width() - 4, y))
        painter.end()


class GuideStrip(QFrame):
    """항상 보이는 순서 안내. 단계를 누르면 그 화면으로 간다."""

    stepRequested = Signal(str)
    hideRequested = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("GuideStrip")
        self.setFixedHeight(SIZES["guide"])
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(24, 0, 12, 0)
        self._layout.setSpacing(0)
        self._steps_box = QHBoxLayout()
        self._steps_box.setSpacing(4)
        self._hide = QuietButton("안내 숨기기", size="sm")
        self._hide.setObjectName("GuideHide")
        self._hide.setToolTip("익숙해지면 숨겨 두세요. 설정에서 다시 켤 수 있습니다.")
        self._hide.clicked.connect(self.hideRequested)
        # 가운데 정렬을 맞추려고 왼쪽에도 숨기기 버튼만큼 자리를 둔다.
        self._balance = QWidget()
        self._layout.addWidget(self._balance)
        self._layout.addStretch(1)
        self._layout.addLayout(self._steps_box)
        self._layout.addStretch(1)
        self._layout.addWidget(self._hide)
        self._pills: dict[str, StepPill] = {}
        self._connectors: list[_Connector] = []
        self._done: dict[str, bool] = {}
        self._current: str | None = None

    def add_step(self, page_id: str, text: str) -> StepPill:
        if self._pills:
            connector = _Connector(self)
            self._connectors.append(connector)
            self._steps_box.addWidget(connector)
        pill = StepPill(page_id, len(self._pills) + 1, text, self)
        pill.clicked.connect(lambda _checked=False, pid=page_id: self.stepRequested.emit(pid))
        self._pills[page_id] = pill
        self._done[page_id] = False
        self._steps_box.addWidget(pill)
        self._balance.setFixedWidth(max(0, self._hide.sizeHint().width() - 12))
        self._refresh()
        return pill

    def step_ids(self) -> list[str]:
        return list(self._pills)

    def pill(self, page_id: str) -> StepPill:
        return self._pills[page_id]

    def current(self) -> str | None:
        return self._current

    def set_current(self, page_id: str | None) -> None:
        self._current = page_id if page_id in self._pills else None
        self._refresh()

    def set_done(self, page_id: str, done: bool = True) -> None:
        if page_id not in self._pills:
            raise KeyError(page_id)
        self._done[page_id] = bool(done)
        self._refresh()

    def is_done(self, page_id: str) -> bool:
        return self._done[page_id]

    def state(self, page_id: str) -> str:
        return self._pills[page_id].state()

    def states(self) -> dict[str, str]:
        return {pid: pill.state() for pid, pill in self._pills.items()}

    def _refresh(self) -> None:
        ids = list(self._pills)
        for pid, pill in self._pills.items():
            if pid == self._current:
                pill.set_state("current")
            elif self._done[pid]:
                pill.set_state("done")
            else:
                pill.set_state("upcoming")
        for i, connector in enumerate(self._connectors):
            connector.set_done(self._done[ids[i]])
