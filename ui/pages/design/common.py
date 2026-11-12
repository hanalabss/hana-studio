"""디자인 화면의 패널들이 함께 쓰는 작은 부품."""

from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QLayout, QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)

from ui.theme import SPACE, ThemeManager
from ui.widgets import label, set_prop

from . import icons

LABEL_WIDTH = 72


class EditSession:
    """잇따라 오는 같은 종류의 수정(끌기, 꾹 누르기, 타이핑)을 되돌리기 한 단계로 묶는 열쇠를 만든다."""

    IDLE_MS = 700

    def __init__(self, parent=None) -> None:
        self._serial = 0
        self._timer = QTimer(parent)
        self._timer.setSingleShot(True)
        self._timer.setInterval(self.IDLE_MS)
        self._timer.timeout.connect(self.close)

    def key(self, name: str) -> tuple[str, int]:
        self._timer.start()
        return (name, self._serial)

    def close(self) -> None:
        self._timer.stop()
        self._serial += 1


class FlowLayout(QLayout):
    """왼쪽부터 채우고 자리가 모자라면 다음 줄로 넘긴다."""

    def __init__(self, parent: QWidget | None = None, spacing: int = SPACE["sm"]) -> None:
        super().__init__(parent)
        self._items = []
        self._gap = spacing
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item) -> None:
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int):
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int):
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self._arrange(QRect(0, 0, width, 0), apply=False)

    def setGeometry(self, rect: QRect) -> None:
        super().setGeometry(rect)
        self._arrange(rect, apply=True)

    def sizeHint(self) -> QSize:
        return self.minimumSize()

    def minimumSize(self) -> QSize:
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        return size

    def _arrange(self, rect: QRect, apply: bool) -> int:
        x, y, row_height = rect.x(), rect.y(), 0
        for item in self._items:
            widget = item.widget()
            if widget is not None and widget.isHidden():
                continue
            hint = item.sizeHint()
            width = min(hint.width(), rect.width())
            if x > rect.x() and x + width > rect.right() + 1:
                x, y, row_height = rect.x(), y + row_height + self._gap, 0
            if apply:
                item.setGeometry(QRect(QPoint(x, y), QSize(width, hint.height())))
            x += width + self._gap
            row_height = max(row_height, hint.height())
        return y + row_height - rect.y()


def clear_layout(layout: QLayout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.hide()
            widget.setParent(None)
            widget.deleteLater()
        elif item.layout() is not None:
            clear_layout(item.layout())


class ClickFrame(QFrame):
    """통째로 누를 수 있는 상자. Enter와 Space로도 눌린다."""

    clicked = Signal()

    def __init__(self, object_name: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName(object_name)
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.TabFocus)
        self.setAttribute(Qt.WA_Hover, True)
        self._pressed = False

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._pressed = True
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._pressed and event.button() == Qt.LeftButton:
            self._pressed = False
            if self.rect().contains(event.position().toPoint()):
                self.clicked.emit()
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space):
            self.clicked.emit()
            return
        super().keyPressEvent(event)

    def focusInEvent(self, event) -> None:
        super().focusInEvent(event)
        if event.reason() in (Qt.TabFocusReason, Qt.BacktabFocusReason):
            set_prop(self, "focused", "true")

    def focusOutEvent(self, event) -> None:
        super().focusOutEvent(event)
        set_prop(self, "focused", "false")


class NoteRow(QFrame):
    """아이콘과 한두 줄 글. kind: problem(누를 수 있음) | ok | note."""

    clicked = Signal()

    _NAMES = {"problem": "ProblemRow", "ok": "OkRow", "note": "NoteRow"}
    _ICONS = {"problem": ("warning", "warn_text"), "ok": ("check", "ok_text"), "note": ("info", "text_muted")}

    def __init__(self, kind: str, text: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._kind = kind
        self.setObjectName(self._NAMES[kind])
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(8)
        self._icon = QLabel()
        self._icon.setFixedSize(16, 16)
        layout.addWidget(self._icon, 0, Qt.AlignTop)
        self._text = QLabel(text)
        self._text.setWordWrap(True)
        layout.addWidget(self._text, 1)
        if kind == "problem":
            self.setCursor(Qt.PointingHandCursor)
            self.setFocusPolicy(Qt.TabFocus)
            self.setAttribute(Qt.WA_Hover, True)
        self._refresh()
        ThemeManager.instance().themeChanged.connect(self._refresh)

    def text(self) -> str:
        return self._text.text()

    def set_text(self, text: str) -> None:
        self._text.setText(text)

    def _refresh(self, *_args) -> None:
        name, role = self._ICONS[self._kind]
        self._icon.setPixmap(icons.pixmap(name, 16, role))

    def mouseReleaseEvent(self, event) -> None:
        inside = self.rect().contains(event.position().toPoint())
        if self._kind == "problem" and event.button() == Qt.LeftButton and inside:
            self.clicked.emit()
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:
        if self._kind == "problem" and event.key() in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space):
            self.clicked.emit()
            return
        super().keyPressEvent(event)


class Collapsible(QWidget):
    """제목을 누르면 펼쳐지는 묶음. 내용은 body(QVBoxLayout)에 넣는다."""

    toggled = Signal(bool)

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._title = title
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(SPACE["sm"])
        self.header = QPushButton()
        self.header.setProperty("disclosure", "true")
        self.header.setCheckable(True)
        self.header.setAutoDefault(False)
        self.header.setCursor(Qt.PointingHandCursor)
        outer.addWidget(self.header)
        self._content = QWidget(self)
        self.body = QVBoxLayout(self._content)
        self.body.setContentsMargins(0, 0, 0, 0)
        self.body.setSpacing(SPACE["md"])
        self._content.setVisible(False)
        outer.addWidget(self._content)
        self.header.toggled.connect(self._on_toggled)
        self._refresh()
        ThemeManager.instance().themeChanged.connect(self._refresh)

    def is_open(self) -> bool:
        return self.header.isChecked()

    def set_open(self, opened: bool) -> None:
        self.header.setChecked(opened)

    def _on_toggled(self, opened: bool) -> None:
        self._content.setVisible(opened)
        self._refresh()
        self.toggled.emit(opened)

    def _refresh(self, *_args) -> None:
        opened = self.header.isChecked()
        self.header.setText(self._title)
        self.header.setIcon(icons.icon("chevron-up" if opened else "chevron-down", "accent_text", 16))
        self.header.setIconSize(QSize(16, 16))
        self.header.setAccessibleName(f"{self._title} {'접기' if opened else '펼치기'}")


def field_row(label_text: str, control: QWidget, label_width: int = LABEL_WIDTH) -> QWidget:
    """이름표가 왼쪽에 오는 한 줄. 이름표 폭이 같아 입력 칸의 왼쪽 끝이 맞는다."""
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(SPACE["sm"])
    name = label(label_text, "label")
    name.setFixedWidth(label_width)
    name.setBuddy(control)
    if not control.accessibleName():
        control.setAccessibleName(label_text)
    layout.addWidget(name)
    layout.addWidget(control, 1)
    return row


def group(label_text: str | None, *widgets: QWidget, hint: str | None = None) -> QWidget:
    """이름표 아래에 입력을 쌓은 묶음."""
    box = QWidget()
    layout = QVBoxLayout(box)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(6)
    if label_text:
        layout.addWidget(label(label_text, "label"))
        if widgets and not widgets[0].accessibleName():
            widgets[0].setAccessibleName(label_text)
    for widget in widgets:
        layout.addWidget(widget)
    if hint:
        layout.addWidget(label(hint, "hint", wrap=True))
    return box


def hbox(*widgets: QWidget | None, spacing: int = SPACE["sm"]) -> QWidget:
    """None 자리는 늘어나는 빈칸."""
    box = QWidget()
    layout = QHBoxLayout(box)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(spacing)
    for widget in widgets:
        if widget is None:
            layout.addStretch(1)
        else:
            layout.addWidget(widget)
    return box


def fixed_height_policy(widget: QWidget) -> QWidget:
    widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    return widget


def use_local_icon(button, name: str, role: str = "text", size: int = 18) -> None:
    """공용 버튼에 디자인 화면 아이콘을 얹고, 테마가 바뀌면 다시 그린다."""
    from shiboken6 import isValid

    def refresh(*_args) -> None:
        if isValid(button):
            button.setIcon(icons.icon(name, role, size))
            button.setIconSize(QSize(size, size))

    refresh()
    ThemeManager.instance().themeChanged.connect(refresh)


def as_menu_button(button, role: str = "text_secondary") -> None:
    """메뉴가 달린 버튼: 글자 오른쪽에 작은 화살표를 붙인다."""
    button.setProperty("menu", "true")
    button.setLayoutDirection(Qt.RightToLeft)
    use_local_icon(button, "chevron-down", role, 12)


def fit_segments(control) -> None:
    """고른 칸은 굵은 글씨가 되므로, 그때도 글자가 잘리지 않게 칸 너비를 잡아 둔다."""
    for key in control.keys():
        segment = control.button(key)
        font = segment.font()
        font.setBold(True)
        segment.setMinimumWidth(QFontMetrics(font).horizontalAdvance(segment.text()) + 34)


class LinkButton(QPushButton):
    """글 속 링크처럼 보이는 버튼. 왼쪽 끝이 위아래 글과 맞는다."""

    def __init__(self, text: str, parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setProperty("link", "true")
        self.setAutoDefault(False)
        self.setCursor(Qt.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)


def link_row(button) -> QWidget:
    box = QWidget()
    layout = QHBoxLayout(box)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    layout.addWidget(button)
    layout.addStretch(1)
    return box


class ChipBox(QWidget):
    """칩을 줄 바꿔 가며 담는다. 너비가 정해지면 필요한 높이를 스스로 잡는다."""

    def __init__(self, parent: QWidget | None = None, spacing: int = 6) -> None:
        super().__init__(parent)
        self.flow = FlowLayout(self, spacing=spacing)
        policy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self.flow.heightForWidth(width)
