"""대화 상자 틀: 제목, 본문, 아래쪽 버튼 줄."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QFrame, QHBoxLayout, QVBoxLayout, QWidget

from ui.theme import SPACE, mark_themed

from ._util import label
from .buttons import Button


class Dialog(QDialog):
    """본문은 dialog.body(QVBoxLayout)에 넣고, 버튼은 add_button으로 붙인다.

    add_button의 role: "accept"(확인하고 닫기), "reject"(그만두고 닫기), None(닫지 않음).
    """

    def __init__(self, title: str, description: str | None = None, parent: QWidget | None = None,
                 width: int = 460):
        super().__init__(parent)
        self.setObjectName("KitDialog")
        mark_themed(self)
        self.setWindowTitle(title)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setMinimumWidth(width)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        content = QWidget(self)
        self.body = QVBoxLayout(content)
        self.body.setContentsMargins(SPACE["xl"], SPACE["xl"], SPACE["xl"], SPACE["xl"])
        self.body.setSpacing(SPACE["md"])
        self._title = label(title, "heading", wrap=True)
        self.body.addWidget(self._title)
        self._description = label(description or "", "secondary", wrap=True)
        self._description.setVisible(bool(description))
        self.body.addWidget(self._description)
        outer.addWidget(content, 1)

        self._footer = QFrame(self)
        self._footer.setObjectName("DialogFooter")
        self._buttons = QHBoxLayout(self._footer)
        self._buttons.setContentsMargins(SPACE["xl"], SPACE["md"], SPACE["xl"], SPACE["md"])
        self._buttons.setSpacing(SPACE["sm"])
        self._buttons.addStretch(1)
        self._footer.setVisible(False)
        outer.addWidget(self._footer)

    def title(self) -> str:
        return self._title.text()

    def set_description(self, text: str | None) -> None:
        self._description.setText(text or "")
        self._description.setVisible(bool(text))

    def set_body(self, widget: QWidget) -> None:
        self.body.addWidget(widget, 1)

    def add_button(self, text: str, variant: str = "secondary", role: str | None = "accept",
                   default: bool = False, left: bool = False) -> Button:
        button = Button(text, variant)
        button.setMinimumWidth(88)
        if role == "accept":
            button.clicked.connect(self.accept)
        elif role == "reject":
            button.clicked.connect(self.reject)
        if default:
            button.setAutoDefault(True)
            button.setDefault(True)
        if left:
            self._buttons.insertWidget(0, button)
        else:
            self._buttons.addWidget(button)
        self._footer.setVisible(True)
        return button


def confirm(parent: QWidget | None, title: str, message: str, confirm_text: str = "확인",
            cancel_text: str = "취소", danger: bool = False) -> bool:
    """예/아니오를 묻는다. 버튼 글자는 일어날 일을 그대로 쓴다(예: "12명 지우기")."""
    dialog = Dialog(title, message, parent)
    dialog.add_button(cancel_text, "secondary", "reject")
    dialog.add_button(confirm_text, "danger" if danger else "primary", "accept", default=not danger)
    return dialog.exec() == QDialog.Accepted


def inform(parent: QWidget | None, title: str, message: str, button_text: str = "확인") -> None:
    dialog = Dialog(title, message, parent)
    dialog.add_button(button_text, "primary", "accept", default=True)
    dialog.exec()
