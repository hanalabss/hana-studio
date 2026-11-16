"""작업자 확인(로그인)과 작업자 더하기 창."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox, QDialog, QLineEdit, QWidget

from services.operators import PIN_MAX, OperatorBook, OperatorError
from ui.widgets import ComboBox, Dialog, FormRow, label, set_prop


def _pin_field(placeholder: str = "") -> QLineEdit:
    field = QLineEdit()
    field.setEchoMode(QLineEdit.Password)
    field.setMaxLength(PIN_MAX)
    field.setPlaceholderText(placeholder)
    field.setInputMethodHints(Qt.ImhDigitsOnly)
    return field


class SignInDialog(Dialog):
    """누가 쓰는지 고르고 비밀번호를 넣는다. 맞으면 book의 지금 작업자가 바뀌고 accept된다."""

    def __init__(self, book: OperatorBook, parent: QWidget | None = None, closing_text: str = "닫기"):
        super().__init__("누가 작업하나요?", "인쇄 기록에 작업한 사람이 남습니다.", parent, width=400)
        self.book = book
        self.name_combo = ComboBox(book.names())
        self.name_combo.setAccessibleName("작업자")
        if book.current in book.names():
            self.name_combo.setCurrentIndex(book.names().index(book.current))
        self.pin_edit = _pin_field("숫자 비밀번호")
        self.pin_edit.setAccessibleName("비밀번호")
        self.pin_edit.returnPressed.connect(self.try_sign_in)
        self.error = label("", "hint", wrap=True)
        set_prop(self.error, "tone", "error")
        self.error.setVisible(False)
        self.body.addWidget(FormRow("작업자", self.name_combo))
        self.body.addWidget(FormRow("비밀번호", self.pin_edit))
        self.body.addWidget(self.error)
        self.add_button(closing_text, "secondary", "reject")
        self.ok_button = self.add_button("확인", "primary", None, default=True)
        self.ok_button.clicked.connect(self.try_sign_in)
        self.pin_edit.setFocus()

    def try_sign_in(self) -> bool:
        try:
            self.book.sign_in(self.name_combo.currentText(), self.pin_edit.text())
        except OperatorError as exc:
            self.error.setText(str(exc))
            self.error.setVisible(True)
            self.pin_edit.selectAll()
            self.pin_edit.setFocus()
            return False
        self.accept()
        return True


class AddOperatorDialog(Dialog):
    def __init__(self, book: OperatorBook, parent: QWidget | None = None):
        first = not book.enabled
        super().__init__(
            "작업자 더하기",
            "처음 만드는 작업자는 관리자가 됩니다. 관리자만 작업자를 더하고 지울 수 있습니다." if first
            else "비밀번호는 숫자 4~8자리입니다.", parent, width=420)
        self.book = book
        self.name_edit = QLineEdit()
        self.name_edit.setAccessibleName("작업자 이름")
        self.pin_edit = _pin_field("숫자 4~8자리")
        self.pin_again = _pin_field("한 번 더")
        self.admin_check = QCheckBox("관리자 (작업자를 더하고 지울 수 있음)")
        self.admin_check.setChecked(first)
        self.admin_check.setEnabled(not first)
        self.error = label("", "hint", wrap=True)
        set_prop(self.error, "tone", "error")
        self.error.setVisible(False)
        for title, widget in (("이름", self.name_edit), ("비밀번호", self.pin_edit), ("비밀번호 확인", self.pin_again)):
            self.body.addWidget(FormRow(title, widget))
        self.body.addWidget(self.admin_check)
        self.body.addWidget(self.error)
        self.add_button("그만두기", "secondary", "reject")
        self.ok_button = self.add_button("더하기", "primary", None, default=True)
        self.ok_button.clicked.connect(self.try_add)

    def try_add(self) -> bool:
        if self.pin_edit.text() != self.pin_again.text():
            return self._fail("두 비밀번호가 다릅니다.")
        try:
            self.book.add(self.name_edit.text(), self.pin_edit.text(), self.admin_check.isChecked())
        except OperatorError as exc:
            return self._fail(str(exc))
        self.accept()
        return True

    def _fail(self, message: str) -> bool:
        self.error.setText(message)
        self.error.setVisible(True)
        return False


def ask_operator(book: OperatorBook, parent: QWidget | None = None) -> bool:
    """작업자를 쓰면 누가 쓰는지 확인한다. 작업자가 없으면 묻지 않고 True. 그만두면 False."""
    if book is None or not book.enabled:
        return True
    return SignInDialog(book, parent, closing_text="프로그램 닫기").exec() == QDialog.Accepted
