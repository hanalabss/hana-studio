"""위쪽 줄: 앱 이름, 문서 이름과 저장 상태, 되돌리기, 화면 색, 프린터 상태."""

from __future__ import annotations

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from ui.theme import SIZES, SPACE, ThemeManager, mark_themed
from ui.widgets import (
    Divider,
    IconButton,
    SecondaryButton,
    StatusPill,
    label,
    set_prop,
)

APP_NAME = "하나스튜디오"


class PrinterPopover(QFrame):
    """프린터 상태 알약을 누르면 뜨는 작은 창. 프린터 담당 화면이 set_content로 내용을 바꾼다."""

    openPrinterPageRequested = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent, Qt.Popup | Qt.FramelessWindowHint | Qt.NoDropShadowWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setFixedWidth(320)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        frame = QFrame(self)
        frame.setObjectName("Popover")
        outer.addWidget(frame)
        self._layout = QVBoxLayout(frame)
        self._layout.setContentsMargins(SPACE["lg"], SPACE["lg"], SPACE["lg"], SPACE["lg"])
        self._layout.setSpacing(SPACE["sm"])
        self._title = label("", "title")
        self._detail = label("", "muted", wrap=True)
        self._layout.addWidget(self._title)
        self._layout.addWidget(self._detail)
        self._slot = QVBoxLayout()
        self._slot.setContentsMargins(0, 0, 0, 0)
        self._layout.addLayout(self._slot)
        self._content: QWidget | None = None
        self._layout.addSpacing(SPACE["xs"])
        self._open = SecondaryButton("프린터 화면 열기", icon="printer")
        self._open.clicked.connect(self._on_open)
        self._layout.addWidget(self._open)
        mark_themed(self)

    def set_status(self, title: str, detail: str) -> None:
        self._title.setText(title)
        self._detail.setText(detail)
        self._detail.setVisible(bool(detail))

    def title(self) -> str:
        return self._title.text()

    def detail(self) -> str:
        return self._detail.text()

    def set_content(self, widget: QWidget | None) -> None:
        if self._content is not None:
            self._slot.removeWidget(self._content)
            self._content.setParent(None)
        self._content = widget
        if widget is not None:
            self._slot.addWidget(widget)

    def open_button(self) -> SecondaryButton:
        return self._open

    def show_below(self, anchor: QWidget) -> None:
        self.adjustSize()
        corner = anchor.mapToGlobal(QPoint(anchor.width() - self.width(), anchor.height() + 6))
        self.move(corner)
        self.show()

    def _on_open(self) -> None:
        self.hide()
        self.openPrinterPageRequested.emit()


class TopBar(QFrame):
    undoRequested = Signal()
    redoRequested = Signal()
    themeToggleRequested = Signal()
    printerClicked = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("TopBar")
        self.setFixedHeight(SIZES["topbar"])
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 0, 16, 0)
        layout.setSpacing(SPACE["sm"])

        self._app_name = QLabel(APP_NAME)
        self._app_name.setObjectName("AppName")
        layout.addWidget(self._app_name)
        layout.addSpacing(SPACE["sm"])
        divider = Divider(Qt.Vertical)
        divider.setFixedHeight(20)
        layout.addWidget(divider)
        layout.addSpacing(SPACE["sm"])
        self._doc_name = QLabel("")
        self._doc_name.setObjectName("DocName")
        layout.addWidget(self._doc_name)
        self._doc_state = QLabel("")
        self._doc_state.setObjectName("DocState")
        layout.addWidget(self._doc_state)
        layout.addStretch(1)

        self.undo_button = SecondaryButton("되돌리기", size="sm", icon="undo")
        self.undo_button.setToolTip("방금 한 일을 되돌립니다 (Ctrl+Z)")
        self.undo_button.setEnabled(False)
        self.undo_button.clicked.connect(self.undoRequested)
        self.redo_button = SecondaryButton("다시 하기", size="sm", icon="redo")
        self.redo_button.setToolTip("되돌린 일을 다시 합니다 (Ctrl+Y)")
        self.redo_button.setEnabled(False)
        self.redo_button.clicked.connect(self.redoRequested)
        layout.addWidget(self.undo_button)
        layout.addWidget(self.redo_button)
        layout.addSpacing(SPACE["sm"])

        self.theme_button = IconButton("moon", "", size="sm")
        self.theme_button.clicked.connect(self.themeToggleRequested)
        layout.addWidget(self.theme_button)
        layout.addSpacing(SPACE["xs"])

        self.printer_pill = StatusPill("프린터 연결 안 됨", "offline")
        self.printer_pill.setToolTip("프린터 상태 자세히 보기")
        self.printer_pill.clicked.connect(self.printerClicked)
        layout.addWidget(self.printer_pill)

        ThemeManager.instance().themeChanged.connect(self._sync_theme_button)
        self._sync_theme_button()

    def _sync_theme_button(self, *_args) -> None:
        dark = ThemeManager.instance().theme.is_dark
        tip = "밝은 화면으로 바꾸기" if dark else "어두운 화면으로 바꾸기"
        self.theme_button.set_icon_name("sun" if dark else "moon")
        self.theme_button.setToolTip(f"{tip} (Ctrl+Shift+L)")
        self.theme_button.setAccessibleName(tip)

    def document_name(self) -> str:
        return self._doc_name.text()

    def document_state_text(self) -> str:
        return self._doc_state.text()

    def set_document(self, name: str, saved: bool | None) -> None:
        """saved: True(저장됨), False(저장 안 한 변경 있음), None(표시 안 함)."""
        self._doc_name.setText(name)
        if saved is None:
            self._doc_state.setText("")
            set_prop(self._doc_state, "state", "")
        elif saved:
            self._doc_state.setText("저장됨")
            set_prop(self._doc_state, "state", "saved")
        else:
            self._doc_state.setText("저장 안 한 변경이 있습니다")
            set_prop(self._doc_state, "state", "dirty")
