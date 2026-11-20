"""인쇄가 멈췄을 때 자리를 비운 사람도 알 수 있게 소리와 작업 표시줄 깜빡임으로 알린다.

카드가 떨어지면 예전에는 프린터가 직접 소리를 냈지만, 지금은 앱이 먼저 알아채고 멈추므로
프린터가 조용하다. 그래서 앱이 대신 알린다.
- 작업 표시줄의 앱 아이콘을 창을 열 때까지 깜빡인다(QApplication.alert).
- 경고음을 REPEAT_MS마다 다시 낸다. 앱에서 마우스를 누르거나 키를 치면, 또는 MAX_MS가 지나면 그친다.
"""
from __future__ import annotations

import logging
import sys

from PySide6.QtCore import QEvent, QObject, QTimer
from PySide6.QtWidgets import QApplication, QWidget

log = logging.getLogger(__name__)

REPEAT_MS = 5_000
MAX_MS = 180_000
_STOP_EVENTS = (QEvent.MouseButtonPress, QEvent.KeyPress)


def play_alert_sound() -> None:
    """Windows 경고음(느낌표). 소리를 낼 수 없으면 기본 비프로 대신한다."""
    if sys.platform == "win32":
        try:
            import winsound

            winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS | winsound.SND_ASYNC)
            return
        except Exception:
            log.debug("Windows 경고음을 내지 못했습니다.", exc_info=True)
    QApplication.beep()


class AttentionAlert(QObject):
    """start()로 알림을 시작하고, 사람이 앱을 만지거나 stop()을 부르면 그친다."""

    def __init__(self, parent: QObject | None = None, sound=None,
                 repeat_ms: int = REPEAT_MS, max_ms: int = MAX_MS) -> None:
        super().__init__(parent)
        self._sound = sound
        self._repeat = QTimer(self)
        self._repeat.setInterval(repeat_ms)
        self._repeat.timeout.connect(self._ring)
        self._limit = QTimer(self)
        self._limit.setSingleShot(True)
        self._limit.setInterval(max_ms)
        self._limit.timeout.connect(self.stop)

    @property
    def active(self) -> bool:
        return self._repeat.isActive()

    def start(self, window: QWidget | None) -> None:
        app = QApplication.instance()
        if app is None:
            return
        if window is not None:
            QApplication.alert(window, 0)        # 0 = 창을 열 때까지 깜빡인다
        if not self.active:
            app.installEventFilter(self)
        self._ring()
        self._repeat.start()
        self._limit.start()

    def stop(self) -> None:
        if not self.active:
            return
        self._repeat.stop()
        self._limit.stop()
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)

    def _ring(self) -> None:
        try:
            (self._sound or play_alert_sound)()
        except Exception:
            log.debug("알림음을 내지 못했습니다.", exc_info=True)

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt 이름
        if event.type() in _STOP_EVENTS:
            self.stop()
        return False
