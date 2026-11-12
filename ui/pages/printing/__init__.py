"""인쇄 화면. (기록·프린터 화면이 이 꾸러미의 text, style, tables를 함께 쓰므로 화면은 늦게 불러온다.)"""

STATE_READY = "ready"
STATE_RUNNING = "printing"
STATE_DONE = "finished"

__all__ = ["PrintPage", "STATE_DONE", "STATE_READY", "STATE_RUNNING"]


def __getattr__(name: str):
    if name == "PrintPage":
        from .page import PrintPage

        return PrintPage
    raise AttributeError(name)
