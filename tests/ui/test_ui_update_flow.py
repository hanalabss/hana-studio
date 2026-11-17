"""업데이트를 다 받은 뒤 프로그램이 바로 끝나는지(설치 프로그램이 60초 기다리다 강제 종료하지 않게)."""
from PySide6.QtWidgets import QMainWindow

from ui.update_flow import DownloadDialog, UpdateFlow


class _Updater:
    def __init__(self):
        self.applied = []

    def apply_and_restart(self, update):
        self.applied.append(update)


class _StubbornWindow(QMainWindow):
    """저장 안 한 디자인처럼 닫기를 거절하는 창."""

    def closeEvent(self, event):
        event.ignore()


def test_quit_after_download_does_not_wait_for_windows(qapp, monkeypatch):
    exits = []
    monkeypatch.setattr(qapp, "exit", lambda code=0: exits.append(code))
    window = _StubbornWindow()
    window.show()
    updater = _Updater()
    flow = UpdateFlow(window, updater=updater)
    flow._dialog = DownloadDialog("9.9.9", window)
    flow._dialog.open()
    dialog = flow._dialog

    flow._on_downloaded("update")
    assert updater.applied == ["update"]
    assert dialog.bar.maximum() == 0 and "설치" in dialog.percent.text()   # 100%에 멈춘 듯 보이지 않게
    dialog.reject()
    assert dialog.isVisible()                                            # 사용자는 여전히 닫을 수 없다

    flow._quit()
    assert not dialog.isVisible() and not window.isVisible()
    assert exits == [0]                                                  # 창에 묻지 않고 끝낸다
