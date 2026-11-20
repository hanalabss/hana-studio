"""인쇄가 멈췄을 때 자리를 비운 사람도 알 수 있게 소리와 작업 표시줄 깜빡임으로 알린다.

드라이버로 인쇄하는 프로그램(다눌)은 카드가 떨어지면 프린터가 직접 오류음을 냈지만, SDK로 인쇄하면
공급함이 빈 순간 SDK가 명령을 바로 거절해 프린터는 Ready 그대로 조용하다(LUCA-40KM 실기 확인).
그래서 앱이 대신 알린다.
- 작업 표시줄의 앱 아이콘을 창을 열 때까지 깜빡인다(QApplication.alert).
- 앱에 넣은 '띠띠' 소리(assets/sounds/alert.wav)를 REPEAT_MS마다 다시 낸다. Windows 소리 설정의
  '느낌표'가 꺼져 있어도 들린다.
- 다른 프로그램 위에 알림 창을 띄우고, 그 창에서 '확인'을 누를 때까지 소리를 그치지 않는다.
"""
from __future__ import annotations

import logging
import os
import sys

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import QApplication, QWidget

from ui.widgets.dialogs import Dialog

log = logging.getLogger(__name__)

REPEAT_MS = 5_000


def alert_sound_path() -> str:
    """앱에 넣은 경고음 파일. 설치본(PyInstaller)에서는 _MEIPASS 아래에 있다."""
    base = getattr(sys, "_MEIPASS", None) or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "assets", "sounds", "alert.wav")


def play_alert_sound() -> None:
    """앱의 '띠띠' 소리. 파일이 없으면 Windows 느낌표 소리, 그것도 안 되면 기본 비프로 대신한다."""
    if sys.platform == "win32":
        try:
            import winsound

            path = alert_sound_path()
            if os.path.isfile(path):
                winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT)
            else:
                winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS | winsound.SND_ASYNC)
            return
        except Exception:
            log.debug("경고음을 내지 못했습니다.", exc_info=True)
    QApplication.beep()


class StopNotice(Dialog):
    """인쇄가 멈췄다는 알림 창. 다른 프로그램 위에 떠 있고, '확인'을 눌러야 닫힌다."""

    def __init__(self, title: str, message: str, parent: QWidget | None = None) -> None:
        super().__init__(title, message, parent)
        self.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        self.setWindowModality(Qt.NonModal)
        self.confirm_button = self.add_button("확인", "primary", "accept", default=True)


class AttentionAlert(QObject):
    """start()로 알림 창을 띄우고 경고음을 되풀이한다. 창에서 '확인'을 누르거나 stop()을 부르면 그친다."""

    confirmed = Signal()

    def __init__(self, parent: QObject | None = None, sound=None, repeat_ms: int = REPEAT_MS) -> None:
        super().__init__(parent)
        self._sound = sound
        self._repeat = QTimer(self)
        self._repeat.setInterval(repeat_ms)
        self._repeat.timeout.connect(self._ring)
        self.notice: StopNotice | None = None

    @property
    def active(self) -> bool:
        return self._repeat.isActive()

    def start(self, window: QWidget | None, title: str = "인쇄가 멈췄습니다", message: str = "") -> None:
        if QApplication.instance() is None:
            return
        self._close_notice()
        self.notice = StopNotice(title, message, window)
        self.notice.finished.connect(self._on_notice_closed)
        self.notice.show()
        self.notice.raise_()
        self.notice.activateWindow()
        if window is not None:
            QApplication.alert(window, 0)        # 0 = 창을 열 때까지 작업 표시줄 아이콘을 깜빡인다
        self._ring()
        self._repeat.start()

    def stop(self) -> None:
        """소리를 그치고 알림 창을 닫는다(새 인쇄를 시작할 때 등)."""
        self._repeat.stop()
        self._close_notice()

    def _close_notice(self) -> None:
        notice, self.notice = self.notice, None
        if notice is not None:
            notice.finished.disconnect(self._on_notice_closed)
            notice.close()
            notice.deleteLater()

    def _on_notice_closed(self, _result: int) -> None:
        self._repeat.stop()
        notice, self.notice = self.notice, None
        if notice is not None:
            notice.deleteLater()
        self.confirmed.emit()

    def _ring(self) -> None:
        try:
            (self._sound or play_alert_sound)()
        except Exception:
            log.debug("알림음을 내지 못했습니다.", exc_info=True)
