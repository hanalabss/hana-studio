"""알림: 화면 안 띠(Banner), 잠깐 뜨는 알림(Toast), 빈 화면 안내(EmptyState)."""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QEvent, QRectF, QTimer, Qt, Signal
from PySide6.QtGui import QPainter, QPen
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from ui.theme import SPACE, mark_themed

from ._util import label, set_prop, theme_color
from .buttons import PrimaryButton, QuietButton
from .icons import IconWidget, paint_icon

KINDS = ("info", "ok", "warn", "error")
_KIND_ICON = {"info": "info", "ok": "check", "warn": "warning", "error": "error"}
_KIND_ROLE = {"info": "info_text", "ok": "ok_text", "warn": "warn_text", "error": "danger_text"}


class Banner(QFrame):
    """화면 안에 자리 잡는 알림 띠. 무슨 일인지와 할 수 있는 일 하나를 함께 보여 준다."""

    actionTriggered = Signal()
    dismissed = Signal()

    def __init__(self, kind: str = "info", text: str = "", action_text: str | None = None,
                 closable: bool = False, parent: QWidget | None = None):
        super().__init__(parent)
        if kind not in KINDS:
            raise ValueError(f"kind는 {KINDS} 중 하나여야 합니다: {kind}")
        self.setObjectName("Banner")
        self._kind = kind
        self.setProperty("kind", kind)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 8, 8)
        layout.setSpacing(10)
        self._icon = IconWidget(_KIND_ICON[kind], 18, _KIND_ROLE[kind])
        icon_holder = QVBoxLayout()
        icon_holder.setContentsMargins(0, 5, 0, 0)
        icon_holder.addWidget(self._icon)
        icon_holder.addStretch(1)
        layout.addLayout(icon_holder)
        self._text = QLabel(text)
        self._text.setWordWrap(True)
        self._text.setMinimumHeight(28)
        layout.addWidget(self._text, 1)
        self._action = QPushButton(action_text or "")
        self._action.setAutoDefault(False)
        self._action.setVisible(bool(action_text))
        self._action.clicked.connect(self.actionTriggered)
        layout.addWidget(self._action, 0, Qt.AlignTop)
        self._close = QPushButton("닫기")
        self._close.setProperty("quiet", "true")
        self._close.setAutoDefault(False)
        self._close.setVisible(closable)
        self._close.clicked.connect(self._dismiss)
        layout.addWidget(self._close, 0, Qt.AlignTop)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

    def kind(self) -> str:
        return self._kind

    def set_kind(self, kind: str) -> None:
        if kind not in KINDS:
            raise ValueError(kind)
        self._kind = kind
        self._icon.set_icon(_KIND_ICON[kind], _KIND_ROLE[kind])
        set_prop(self, "kind", kind)
        for child in (self._text, self._action, self._close):
            child.style().unpolish(child)
            child.style().polish(child)

    def text(self) -> str:
        return self._text.text()

    def set_text(self, text: str) -> None:
        self._text.setText(text)

    def set_action(self, text: str | None) -> None:
        self._action.setText(text or "")
        self._action.setVisible(bool(text))

    def action_button(self) -> QPushButton:
        return self._action

    def _dismiss(self) -> None:
        self.hide()
        self.dismissed.emit()


class Toast(QFrame):
    """창 아래쪽에 잠깐 뜨는 알림. `Toast.show_message(host, "저장했습니다")`로 쓴다."""

    actionTriggered = Signal()
    closed = Signal()

    def __init__(self, host: QWidget, text: str, kind: str = "info", action_text: str | None = None,
                 timeout_ms: int = 4000, bottom_offset: int = 24):
        super().__init__(host)
        if kind not in KINDS:
            raise ValueError(kind)
        self.setObjectName("Toast")
        mark_themed(self)
        self._host = host
        self._kind = kind
        self._bottom_offset = bottom_offset
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 9, 8 if action_text else 16, 9)
        layout.setSpacing(10)
        if kind != "info":
            layout.addWidget(IconWidget(_KIND_ICON[kind], 18, "toast_text"))
        self._text = QLabel(text)
        self._text.setMinimumHeight(30)
        layout.addWidget(self._text)
        self._action = None
        if action_text:
            self._action = QPushButton(action_text)
            self._action.setAutoDefault(False)
            self._action.clicked.connect(self._on_action)
            layout.addWidget(self._action)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.dismiss)
        self._timeout = timeout_ms
        host.installEventFilter(self)
        self.setAccessibleDescription(text)

    @staticmethod
    def show_message(host: QWidget, text: str, kind: str = "info", action_text: str | None = None,
                     on_action: Callable[[], None] | None = None, timeout_ms: int | None = None,
                     bottom_offset: int = 24) -> "Toast":
        """timeout_ms=0이면 사용자가 누를 때까지 남는다. 문제 알림은 기본으로 더 오래 보인다."""
        previous = getattr(host, "_hana_toast", None)
        if previous is not None:
            try:
                previous.dismiss()
            except RuntimeError:
                pass
        if timeout_ms is None:
            timeout_ms = 8000 if kind in ("warn", "error") else 4000
        toast = Toast(host, text, kind, action_text, timeout_ms, bottom_offset)
        if on_action is not None:
            toast.actionTriggered.connect(on_action)
        host._hana_toast = toast
        toast.popup()
        return toast

    def kind(self) -> str:
        return self._kind

    def text(self) -> str:
        return self._text.text()

    def action_button(self) -> QPushButton | None:
        return self._action

    def popup(self) -> None:
        self.adjustSize()
        self._reposition()
        self.show()
        self.raise_()
        if self._timeout > 0:
            self._timer.start(self._timeout)

    def dismiss(self) -> None:
        self._timer.stop()
        if getattr(self._host, "_hana_toast", None) is self:
            self._host._hana_toast = None
        self._host.removeEventFilter(self)
        self.hide()
        self.closed.emit()
        self.deleteLater()

    def _on_action(self) -> None:
        self.actionTriggered.emit()
        self.dismiss()

    def _reposition(self) -> None:
        width = min(self.sizeHint().width(), max(200, self._host.width() - 48))
        self.resize(width, self.sizeHint().height())
        self.move((self._host.width() - width) // 2, self._host.height() - self.height() - self._bottom_offset)

    def eventFilter(self, obj, event) -> bool:
        if obj is self._host and event.type() == QEvent.Resize:
            self._reposition()
        return super().eventFilter(obj, event)


class _BlankCard(QWidget):
    """점선으로 그린 빈 카드. 아직 아무것도 없다는 뜻으로 쓴다."""

    def __init__(self, icon: str, parent: QWidget | None = None):
        super().__init__(parent)
        self._icon = icon
        self.setFixedSize(132, 86)

    def set_icon(self, icon: str) -> None:
        self._icon = icon
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        rect = QRectF(1.5, 1.5, self.width() - 3, self.height() - 3)
        painter.setBrush(theme_color("surface"))
        pen = QPen(theme_color("indicator_line"), 1.4, Qt.CustomDashLine)
        pen.setDashPattern([4, 3.5])
        painter.setPen(pen)
        painter.drawRoundedRect(rect, 9, 9)
        side = 30
        paint_icon(painter, self._icon,
                   QRectF((self.width() - side) / 2, (self.height() - side) / 2, side, side),
                   "text_muted", stroke=1.5)
        painter.end()


class EmptyState(QFrame):
    """내용이 없을 때: 제목, 한두 줄 안내, 할 일 버튼."""

    actionTriggered = Signal()
    secondaryTriggered = Signal()

    def __init__(self, title: str, guidance: str = "", action_text: str | None = None,
                 icon: str = "design", secondary_text: str | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("EmptyState")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(SPACE["xl"], SPACE["xl"], SPACE["xl"], SPACE["xl"])
        layout.setSpacing(0)
        layout.addStretch(4)
        self._art = _BlankCard(icon)
        layout.addWidget(self._art, 0, Qt.AlignHCenter)
        layout.addSpacing(SPACE["xl"])
        self._title = label(title, "heading")
        self._title.setAlignment(Qt.AlignHCenter)
        layout.addWidget(self._title)
        layout.addSpacing(SPACE["sm"])
        self._guidance = label(guidance, "secondary", wrap=True)
        self._guidance.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        self._guidance.setFixedWidth(440)
        self._guidance.setVisible(bool(guidance))
        guidance_row = QHBoxLayout()
        guidance_row.addStretch(1)
        guidance_row.addWidget(self._guidance)
        guidance_row.addStretch(1)
        layout.addLayout(guidance_row)
        layout.addSpacing(SPACE["xl"])
        buttons = QHBoxLayout()
        buttons.setSpacing(SPACE["sm"])
        buttons.addStretch(1)
        self._action = PrimaryButton(action_text or "")
        self._action.setVisible(bool(action_text))
        self._action.clicked.connect(self.actionTriggered)
        buttons.addWidget(self._action)
        self._secondary = QuietButton(secondary_text or "", accent=True)
        self._secondary.setVisible(bool(secondary_text))
        self._secondary.clicked.connect(self.secondaryTriggered)
        buttons.addWidget(self._secondary)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        layout.addStretch(5)

    def title(self) -> str:
        return self._title.text()

    def guidance(self) -> str:
        return self._guidance.text()

    def set_title(self, title: str) -> None:
        self._title.setText(title)

    def set_guidance(self, text: str) -> None:
        self._guidance.setText(text)
        self._guidance.setVisible(bool(text))

    def set_icon(self, icon: str) -> None:
        self._art.set_icon(icon)

    def set_action(self, text: str | None) -> None:
        self._action.setText(text or "")
        self._action.setVisible(bool(text))

    def action_button(self) -> PrimaryButton:
        return self._action

    def secondary_button(self) -> QuietButton:
        return self._secondary
