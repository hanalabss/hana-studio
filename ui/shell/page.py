"""화면(Page) 약속. AppShell에 올라가는 모든 화면이 따른다."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget


@dataclass
class FooterAction:
    """화면 아래 줄에 놓이는 버튼 하나. 글자는 누르면 일어날 일을 쓴다(예: "다음: 명단 넣기")."""

    text: str
    callback: Callable[[], None]
    variant: str = "primary"
    enabled: bool = True
    tooltip: str = ""


@runtime_checkable
class Page(Protocol):
    """QWidget이면서 아래 세 가지를 가지면 된다.

    선택 사항: footer_actions() -> list[FooterAction], footer_hint() -> str,
    footerChanged 시그널(아래 줄을 다시 그려야 할 때 보낸다).
    """

    title: str

    def on_enter(self) -> None: ...

    def on_leave(self) -> None: ...


class BasePage(QWidget):
    """Page 약속을 채운 기본 클래스. 대부분의 화면은 이것을 상속하면 된다."""

    footerChanged = Signal()

    def __init__(self, title: str = "", parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("Page")
        self.title = title
        self.shell = None  # AppShell이 등록할 때 채운다.
        self._footer_actions: list[FooterAction] = []
        self._footer_hint = ""

    def on_enter(self) -> None:
        pass

    def on_leave(self) -> None:
        pass

    def footer_actions(self) -> list[FooterAction]:
        return list(self._footer_actions)

    def set_footer_actions(self, actions: list[FooterAction]) -> None:
        self._footer_actions = list(actions)
        self.footerChanged.emit()

    def footer_hint(self) -> str:
        return self._footer_hint

    def set_footer_hint(self, text: str) -> None:
        self._footer_hint = text
        self.footerChanged.emit()
