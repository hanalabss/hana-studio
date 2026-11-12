"""칩: 걸러 보기 칩, 사람마다 바뀌는 칸 칩, 프린터 상태 알약."""

from __future__ import annotations

import json

from PySide6.QtCore import QMimeData, QPoint, QSize, Qt, Signal
from PySide6.QtGui import QDrag, QPainter
from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QSizePolicy, QWidget

from ui.theme import SIZES

from ._util import FocusRingMixin, on_theme_changed, set_prop
from .icons import icon_pixmap, themed_icon

# 예전 형식: 항목 이름(UTF-8)만 담는다. 호환용으로 계속 싣는다.
FIELD_MIME = "application/x-hana-field"
# 카드 편집기 캔버스가 받는 형식(ui.editor.controller.MIME_FIELD와 같은 값). JSON {"column", "kind"}.
# 키트가 편집기를 import하지 않도록 값을 따로 적고, 같은지는 테스트가 지킨다.
CANVAS_FIELD_MIME = "application/x-hanastudio-field"
FIELD_CHIP_KINDS = ("text", "image", "barcode", "qr")

STATUS_STATES = ("ready", "busy", "error", "offline")
_STATUS_MARK = {"ready": "ok_mark", "busy": "busy_mark", "error": "danger_text", "offline": "idle_mark"}


class FilterChip(FocusRingMixin, QPushButton):
    """켜고 끄는 칩. 개수를 함께 보여 줄 수 있다. toggled(bool)을 쓴다."""

    def __init__(self, text: str, count: int | None = None, tone: str = "neutral",
                 checked: bool = False, parent: QWidget | None = None):
        super().__init__(parent)
        self._label = text
        self._count = count
        self.setCheckable(True)
        self.setChecked(checked)
        self.setProperty("chip", "filter")
        self.setProperty("tone", tone)
        self.setFixedHeight(SIZES["chip"])
        self.setAutoDefault(False)
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self._tone = tone
        self._sync_text()

    def label(self) -> str:
        return self._label

    def count(self) -> int | None:
        return self._count

    def set_count(self, count: int | None) -> None:
        self._count = count
        self._sync_text()

    def set_tone(self, tone: str) -> None:
        self._tone = tone
        set_prop(self, "tone", tone)

    def _sync_text(self) -> None:
        self.setText(self._label if self._count is None else f"{self._label} {self._count:,}")

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if self.shows_focus_ring():
            painter = QPainter(self)
            if self.isChecked():
                role = "warn_bg" if self._tone == "warn" else "strong_text"
                self.paint_focus_ring(painter, 16, role, inset=2.0)
            else:
                self.paint_focus_ring(painter, 16)
            painter.end()


class FieldChip(QLabel):
    """명단 항목 하나. 카드 위로 끌어다 놓는 출발점이고, 누르거나 Enter로도 넣을 수 있다.

    끌 때 캔버스가 받는 `application/x-hanastudio-field`(JSON: column, kind)를 싣고,
    예전 `application/x-hana-field`(항목 이름 UTF-8)와 보통 글자도 함께 싣는다.
    kind: "text"(글자 칸), "image"(사진 칸), "barcode", "qr".
    """

    clicked = Signal(str)
    dragStarted = Signal(str)

    def __init__(self, name: str, parent: QWidget | None = None, *, kind: str = "text"):
        super().__init__(name, parent)
        if kind not in FIELD_CHIP_KINDS:
            raise ValueError(f"kind는 {FIELD_CHIP_KINDS} 중 하나여야 합니다: {kind}")
        self._name = name
        self._kind = kind
        self._press_pos: QPoint | None = None
        self.setObjectName("FieldChip")
        self.setFixedHeight(SIZES["chip"])
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.setFocusPolicy(Qt.TabFocus)
        self.setCursor(Qt.OpenHandCursor)
        self.setAttribute(Qt.WA_Hover, True)
        self.setAccessibleName(f"명단 항목 {name}")
        self.setToolTip("카드 위로 끌어다 놓으세요")

    def name(self) -> str:
        return self._name

    def set_name(self, name: str) -> None:
        self._name = name
        self.setText(name)
        self.setAccessibleName(f"명단 항목 {name}")

    def field_kind(self) -> str:
        return self._kind

    def set_field_kind(self, kind: str) -> None:
        if kind not in FIELD_CHIP_KINDS:
            raise ValueError(f"kind는 {FIELD_CHIP_KINDS} 중 하나여야 합니다: {kind}")
        self._kind = kind

    def mime_data(self) -> QMimeData:
        mime = QMimeData()
        payload = json.dumps({"column": self._name, "kind": self._kind}, ensure_ascii=False)
        mime.setData(CANVAS_FIELD_MIME, payload.encode("utf-8"))
        mime.setData(FIELD_MIME, self._name.encode("utf-8"))
        mime.setText(self._name)
        return mime

    def start_drag(self) -> None:
        drag = QDrag(self)
        drag.setMimeData(self.mime_data())
        pixmap = self.grab()
        drag.setPixmap(pixmap)
        drag.setHotSpot(QPoint(pixmap.width() // 2, pixmap.height() // 2))
        self.dragStarted.emit(self._name)
        drag.exec(Qt.CopyAction)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._press_pos = event.position().toPoint()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._press_pos is not None and (event.buttons() & Qt.LeftButton):
            distance = (event.position().toPoint() - self._press_pos).manhattanLength()
            if distance >= QApplication.startDragDistance():
                self._press_pos = None
                self.start_drag()
                return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._press_pos is not None and event.button() == Qt.LeftButton:
            self._press_pos = None
            if self.rect().contains(event.position().toPoint()):
                self.clicked.emit(self._name)
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space):
            self.clicked.emit(self._name)
            return
        super().keyPressEvent(event)


class StatusPill(FocusRingMixin, QPushButton):
    """프린터 상태. state: ready(준비됨), busy(인쇄 중), error(문제 있음), offline(연결 안 됨)."""

    def __init__(self, text: str = "", state: str = "offline", parent: QWidget | None = None):
        super().__init__(text, parent)
        self._state = "offline"
        self.setProperty("pill", "status")
        self.setFixedHeight(SIZES["pill"])
        self.setAutoDefault(False)
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.setIconSize(QSize(10, 10))
        self.set_status(state, text)
        on_theme_changed(self._refresh_icon)

    def state(self) -> str:
        return self._state

    def set_status(self, state: str, text: str | None = None) -> None:
        if state not in STATUS_STATES:
            raise ValueError(f"state는 {STATUS_STATES} 중 하나여야 합니다: {state}")
        self._state = state
        if text is not None:
            self.setText(text)
        set_prop(self, "state", state)
        self._refresh_icon()

    def _refresh_icon(self, *_args) -> None:
        self.setIcon(themed_icon("dot", _STATUS_MARK[self._state], 10, _STATUS_MARK[self._state]))

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if self.shows_focus_ring():
            painter = QPainter(self)
            self.paint_focus_ring(painter, 16)
            painter.end()


def status_dot(state: str, size: int = 10):
    """상태 점 그림만 필요할 때."""
    return icon_pixmap("dot", size, _STATUS_MARK[state])
