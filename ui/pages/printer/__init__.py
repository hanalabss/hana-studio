"""프린터 화면과 위쪽 줄 프린터 알약."""
from .status import PrinterStatusBinder, bind_printer_status

__all__ = ["PrinterPage", "PrinterStatusBinder", "bind_printer_status"]


def __getattr__(name: str):
    if name == "PrinterPage":
        from .page import PrinterPage

        return PrinterPage
    raise AttributeError(name)
