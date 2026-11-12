import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import pytest


@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app
    # offscreen에서는 클립보드에 내용이 남은 채로 파이썬이 끝나면 Qt가 죽는다(종료 코드 139).
    # 복사·잘라내기 테스트가 남긴 것을 비워 둔다. 실제 Windows 화면에서는 생기지 않는 일이다.
    clipboard = app.clipboard()
    if clipboard is not None:
        clipboard.clear()
    app.processEvents()
