"""위쪽 줄의 프린터 알약과 그 아래 작은 창을 서비스 상태에 맞춰 둔다."""
from __future__ import annotations

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget

from ui.pages.printing.text import pill_state, supplies_lines
from ui.theme import SPACE
from ui.widgets import QuietButton, label

_ATTRIBUTE = "_hana_printer_binder"


class PrinterPopoverContent(QWidget):
    """알약을 누르면 뜨는 창의 내용: 리본·필름 남은 양과 프린터 바꾸기."""

    def __init__(self, services, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._services = services
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, SPACE["xs"], 0, 0)
        layout.setSpacing(SPACE["xs"])
        self.supplies = label("", None, wrap=True)
        layout.addWidget(self.supplies)
        self.switch_title = label("프린터 바꾸기", "label")
        layout.addWidget(self.switch_title)
        self._buttons_box = QVBoxLayout()
        self._buttons_box.setSpacing(2)
        layout.addLayout(self._buttons_box)
        self.switch_buttons: list[QuietButton] = []
        self.note = label("", "hint", wrap=True)
        layout.addWidget(self.note)
        self.refresh()

    def refresh(self) -> None:
        printers = self._services.printers
        status = printers.status()
        current = printers.current()
        lines = supplies_lines(status) if current is not None else []
        self.supplies.setText("\n".join(lines))
        self.supplies.setVisible(bool(lines))

        for button in self.switch_buttons:
            self._buttons_box.removeWidget(button)
            button.setParent(None)
            button.deleteLater()
        self.switch_buttons = []
        others = [d for d in printers.devices()
                  if current is None or (d.family, d.address or d.name) != (current.family, current.address or current.name)]
        printing = self._services.printing.is_printing
        for device in others:
            button = QuietButton(f"{device.name} 쓰기", "sm", accent=True)
            button.setEnabled(not printing)
            button.clicked.connect(lambda _checked=False, d=device: self.use_device(d))
            self._buttons_box.addWidget(button, 0, Qt.AlignLeft)
            self.switch_buttons.append(button)
        self.switch_title.setVisible(bool(others))
        self.note.setText("인쇄가 끝나면 바꿀 수 있습니다." if others and printing else "")
        self.note.setVisible(bool(self.note.text()))

    def use_device(self, device) -> None:
        printers = self._services.printers
        printers.select(device)
        printers.poll_now()


class PrinterStatusBinder(QObject):
    changed = Signal()

    def __init__(self, shell, services) -> None:
        super().__init__(shell)
        self._shell = shell
        self._services = services
        self.content = PrinterPopoverContent(services)
        shell.printer_popover.set_content(self.content)
        printers, printing = services.printers, services.printing
        for signal in (printers.devices_changed, printers.current_changed, printers.status_changed,
                       printers.busy_changed, printing.job_queued, printing.job_started, printing.finished):
            signal.connect(self.refresh)
        self.refresh()

    def refresh(self, *_args) -> None:
        printers = self._services.printers
        searching = printers.is_busy and not printers.devices()
        state, text, detail = pill_state(printers.current(), printers.status(),
                                         self._services.printing.is_printing, searching)
        self._shell.set_printer_status(state, text, detail)
        self.content.refresh()
        self.changed.emit()


def bind_printer_status(shell, services) -> PrinterStatusBinder | None:
    """틀 하나에 한 번만 잇는다. 인쇄 화면과 프린터 화면 어느 쪽이 먼저 올라가도 된다."""
    if shell is None or not hasattr(shell, "set_printer_status"):
        return None
    binder = getattr(shell, _ATTRIBUTE, None)
    if binder is None or binder._services is not services:
        binder = PrinterStatusBinder(shell, services)
        setattr(shell, _ATTRIBUTE, binder)
    return binder
