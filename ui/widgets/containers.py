"""상자와 배치: 카드, 제목 있는 패널, 폼 한 줄, 구분선."""

from __future__ import annotations

from typing import Iterable

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


FORM_ALIGNMENTS = ("fill", "left")


class FormRow(QWidget):
    """이름표 + 입력 + 도움말 한 묶음. orientation="horizontal"이면 이름표가 왼쪽에 온다.

    control_align: "fill"(입력이 줄 너비를 채움) 또는 "left"(입력을 제 크기로 왼쪽에 붙임.
    세그먼트, 스테퍼처럼 폭이 정해진 입력에 쓴다).
    가로 줄 여러 개의 이름표 폭을 맞추려면 align_form_rows(panel)을 부른다.
    """

    def __init__(self, label_text: str, control: QWidget, hint: str | None = None,
                 orientation: str = "vertical", label_width: int = 120, parent: QWidget | None = None,
                 *, control_align: str = "fill"):
        super().__init__(parent)
        if control_align not in FORM_ALIGNMENTS:
            raise ValueError(f"control_align은 {FORM_ALIGNMENTS} 중 하나여야 합니다: {control_align}")
        self.control = control
        self._orientation = "horizontal" if orientation == "horizontal" else "vertical"
        self._control_align = control_align
        align = Qt.AlignLeft if control_align == "left" else Qt.Alignment()
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
            column.addWidget(control, 0, align)
            column.addWidget(self._hint)
            column.addStretch(1)
            outer.addLayout(column, 1)
        else:
            column = QVBoxLayout(self)
            column.setContentsMargins(0, 0, 0, 0)
            column.setSpacing(6)
            column.addWidget(self._label)
            column.addWidget(control, 0, align)
            column.addWidget(self._hint)
            column.addStretch(1)

    def label_widget(self) -> QLabel:
        return self._label

    def orientation(self) -> str:
        return self._orientation

    def control_align(self) -> str:
        return self._control_align

    def label_width(self) -> int:
        """가로 줄에서 이름표가 차지하는 폭. 세로 줄은 0."""
        return self._label.width() if self._orientation == "horizontal" else 0

    def set_label_width(self, width: int) -> None:
        """가로 줄의 이름표 폭을 바꾼다. 세로 줄에서는 아무 일도 하지 않는다."""
        if self._orientation == "horizontal":
            self._label.setFixedWidth(max(0, int(width)))

    def label_width_hint(self) -> int:
        """이름표 글자가 잘리지 않는 가장 좁은 폭."""
        self._label.ensurePolished()
        return self._label.sizeHint().width()

    def hint(self) -> str:
        return self._hint.text()

    def set_hint(self, text: str | None, tone: str | None = None) -> None:
        """tone: None(보통), "warn", "error"."""
        self._hint.setText(text or "")
        self._hint.setVisible(bool(text))
        set_prop(self._hint, "tone", tone or "")

    def set_label(self, text: str) -> None:
        self._label.setText(text)


def align_form_rows(rows: QWidget | Iterable[FormRow], min_width: int = 0, max_width: int | None = None) -> int:
    """가로 FormRow들의 이름표 폭을 가장 긴 이름표에 맞춰, 입력 칸의 왼쪽 끝을 한 줄로 세운다.

    rows에는 FormRow 목록을 주거나, 그 줄들을 담은 상자(패널, 대화 상자)를 준다.
    맞춘 폭을 돌려준다. 이름표 글자를 나중에 바꾸면 다시 불러야 한다.
    """
    if isinstance(rows, QWidget):
        rows = rows.findChildren(FormRow)
    targets = [row for row in rows if row.orientation() == "horizontal"]
    if not targets:
        return 0
    width = max(min_width, max(row.label_width_hint() for row in targets))
    if max_width is not None:
        width = min(width, max_width)
    for row in targets:
        row.set_label_width(width)
    return width


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
