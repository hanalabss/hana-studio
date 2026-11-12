"""상자와 배치: 카드, 제목 있는 패널, 폼 한 줄, 구분선."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from ui.theme import SPACE

from ._util import label, set_prop


class Card(QFrame):
    """테두리 있는 흰 상자. 내용은 card.body(QVBoxLayout)에 넣는다."""

    def __init__(self, parent: QWidget | None = None, padding: int = SPACE["lg"], sunken: bool = False,
                 spacing: int = SPACE["md"]):
        super().__init__(parent)
        self.setObjectName("Card")
        if sunken:
            self.setProperty("sunken", "true")
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(padding, padding, padding, padding)
        self.body.setSpacing(spacing)

    def set_sunken(self, sunken: bool) -> None:
        set_prop(self, "sunken", "true" if sunken else "false")


class Panel(Card):
    """제목과 한 줄 설명이 붙은 카드."""

    def __init__(self, title: str, description: str | None = None, parent: QWidget | None = None,
                 padding: int = SPACE["lg"]):
        super().__init__(parent, padding)
        self._header = QHBoxLayout()
        self._header.setContentsMargins(0, 0, 0, 0)
        self._header.setSpacing(SPACE["sm"])
        self._title = label(title, "title")
        self._header.addWidget(self._title)
        self._header.addStretch(1)
        self.body.addLayout(self._header)
        self._description = label(description or "", "muted", wrap=True)
        self._description.setVisible(bool(description))
        self.body.addWidget(self._description)

    def title(self) -> str:
        return self._title.text()

    def set_title(self, title: str) -> None:
        self._title.setText(title)

    def set_description(self, text: str | None) -> None:
        self._description.setText(text or "")
        self._description.setVisible(bool(text))

    def add_header_widget(self, widget: QWidget) -> None:
        self._header.addWidget(widget)


class FormRow(QWidget):
    """이름표 + 입력 + 도움말 한 묶음. orientation="horizontal"이면 이름표가 왼쪽에 온다."""

    def __init__(self, label_text: str, control: QWidget, hint: str | None = None,
                 orientation: str = "vertical", label_width: int = 120, parent: QWidget | None = None):
        super().__init__(parent)
        self.control = control
        self._label = label(label_text, "label")
        self._label.setBuddy(control)
        if not control.accessibleName():
            control.setAccessibleName(label_text)
        self._hint = label(hint or "", "hint", wrap=True)
        self._hint.setVisible(bool(hint))

        if orientation == "horizontal":
            outer = QHBoxLayout(self)
            outer.setContentsMargins(0, 0, 0, 0)
            outer.setSpacing(SPACE["md"])
            self._label.setFixedWidth(label_width)
            self._label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            self._label.setMinimumHeight(40)
            outer.addWidget(self._label, 0, Qt.AlignTop)
            column = QVBoxLayout()
            column.setContentsMargins(0, 0, 0, 0)
            column.setSpacing(6)
            column.addWidget(control)
            column.addWidget(self._hint)
            column.addStretch(1)
            outer.addLayout(column, 1)
        else:
            column = QVBoxLayout(self)
            column.setContentsMargins(0, 0, 0, 0)
            column.setSpacing(6)
            column.addWidget(self._label)
            column.addWidget(control)
            column.addWidget(self._hint)
            column.addStretch(1)

    def label_widget(self) -> QLabel:
        return self._label

    def hint(self) -> str:
        return self._hint.text()

    def set_hint(self, text: str | None, tone: str | None = None) -> None:
        """tone: None(보통), "warn", "error"."""
        self._hint.setText(text or "")
        self._hint.setVisible(bool(text))
        set_prop(self._hint, "tone", tone or "")

    def set_label(self, text: str) -> None:
        self._label.setText(text)


class Divider(QFrame):
    def __init__(self, orientation: Qt.Orientation = Qt.Horizontal, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("Divider")
        if orientation == Qt.Horizontal:
            self.setFixedHeight(1)
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        else:
            self.setFixedWidth(1)
            self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
