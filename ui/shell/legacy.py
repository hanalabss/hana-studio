"""'빠른 인쇄' 자리. 기존 화면(hana_studio.HanaStudio)을 고치지 않고 그대로 안에 넣는다."""

from __future__ import annotations

import logging
from typing import Callable

from PySide6.QtCore import QEvent, QTimer, Qt
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import QApplication, QScrollArea, QStackedLayout, QWidget

from ui.theme import LIGHT, ThemeManager, mark_themed
from ui.widgets import EmptyState

from .page import BasePage

log = logging.getLogger(__name__)

LegacyFactory = Callable[[], QWidget]


def default_legacy_factory(is_admin: bool = False) -> QWidget:
    """기존 main.py가 하던 순서 그대로 화면을 만든다."""
    from hana_studio import HanaStudio

    window = HanaStudio(is_admin=is_admin)
    ui = getattr(window, "ui", None)
    for name in ("initialize_panels_deferred", "initialize_viewers_deferred"):
        step = getattr(ui, name, None)
        if callable(step):
            step()
    return window


class LegacyHostPage(BasePage):
    """처음 열 때 기존 화면을 만든다. 만들다 실패해도 앱은 계속 쓸 수 있다.

    기존 화면은 자기 색과 크기를 스스로 정하므로 새 테마를 입히지 않는다(themed = False).
    어두운 화면에서도 기존 화면은 밝은 모습 그대로다.
    """

    themed = False

    def __init__(self, factory: LegacyFactory | None = None, is_admin: bool = False,
                 parent: QWidget | None = None):
        super().__init__("빠른 인쇄", parent)
        self._factory = factory or (lambda: default_legacy_factory(is_admin))
        self._window: QWidget | None = None
        self._error: str | None = None
        self._loading = False
        self._creating = False

        self._stack = QStackedLayout(self)
        self._stack.setContentsMargins(0, 0, 0, 0)
        self._waiting = EmptyState(
            "빠른 인쇄 화면을 여는 중입니다",
            "사진 한 장을 골라 바로 뽑는 화면입니다. 처음 열 때는 몇 초 걸립니다.",
            icon="bolt",
        )
        self._failed = EmptyState(
            "빠른 인쇄 화면을 열지 못했습니다",
            "",
            action_text="다시 열기",
            icon="warning",
        )
        self._failed.actionTriggered.connect(self.load_now)
        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._stack.addWidget(self._waiting)
        self._stack.addWidget(self._failed)
        self._stack.addWidget(self._scroll)
        for part in (self._waiting, self._failed, self._scroll.verticalScrollBar(),
                     self._scroll.horizontalScrollBar()):
            mark_themed(part)

    # ---- 상태 ----
    def is_loaded(self) -> bool:
        return self._window is not None

    def legacy_window(self) -> QWidget | None:
        return self._window

    def error(self) -> str | None:
        return self._error

    # ---- Page ----
    def on_enter(self) -> None:
        if self._window is None and not self._loading:
            self._loading = True
            self._stack.setCurrentWidget(self._waiting)
            QTimer.singleShot(0, self.load_now)

    def load_now(self) -> bool:
        """기존 화면을 지금 만든다. 성공하면 True."""
        self._loading = False
        if self._window is not None:
            return True
        if self._creating:
            # 기존 화면은 만드는 도중 이벤트를 처리하므로 그 사이에 다시 불릴 수 있다.
            return False
        self._creating = True
        try:
            window = self._factory()
            if window is None:
                raise RuntimeError("화면을 만들지 못했습니다.")
            window.setParent(self._scroll)
            window.setWindowFlags(Qt.Widget)
            # 기존 화면이 혼자 뜰 때와 같은 팔레트와 글꼴을 준다.
            window.setPalette(ThemeManager.instance().palette(LIGHT))
            window.setFont(QFontDatabase.systemFont(QFontDatabase.GeneralFont))
            QApplication.instance().installEventFilter(self)
            self._scroll.setWidget(window)
            window.show()
        except Exception as exc:  # 기존 화면의 어떤 실패도 새 틀을 멈추면 안 된다.
            log.exception("빠른 인쇄 화면을 만들지 못했습니다")
            self._error = f"{type(exc).__name__}: {exc}"
            self._failed.set_guidance(
                "프로그램을 다시 켜도 같으면 아래 내용을 관리자에게 알려 주세요.\n" + self._error
            )
            self._stack.setCurrentWidget(self._failed)
            return False
        finally:
            self._creating = False
        self._window = window
        self._error = None
        self._stack.setCurrentWidget(self._scroll)
        return True

    def eventFilter(self, obj, event) -> bool:
        # 기존 화면이 띄우는 창(프린터 고르기 등)도 밝은 팔레트를 쓰게 한다.
        if (event.type() == QEvent.Show and self._window is not None and isinstance(obj, QWidget)
                and obj.isWindow() and obj.property("themed") is None
                and not obj.testAttribute(Qt.WA_SetPalette) and self._opened_by_legacy(obj)):
            obj.setPalette(ThemeManager.instance().palette(LIGHT))
        return super().eventFilter(obj, event)

    def _opened_by_legacy(self, widget: QWidget) -> bool:
        parent = widget.parentWidget()
        while parent is not None:
            if parent is self._window:
                return True
            if parent is self:
                return False
            parent = parent.parentWidget()
        return False

    def can_close(self) -> bool:
        """앱을 닫아도 되는지 기존 화면에 묻는다(인쇄 중이면 기존 화면이 확인 창을 띄운다)."""
        if self._window is None:
            return True
        try:
            return bool(self._window.close())
        except Exception:
            log.exception("빠른 인쇄 화면을 닫는 중 문제가 생겼습니다")
            return True
