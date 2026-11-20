import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import pytest


@pytest.fixture(autouse=True)
def _isolated_recent_designs(tmp_path, monkeypatch):
    """테스트가 사용자의 '최근에 작업한 디자인' 목록을 건드리지 않게 한다."""
    monkeypatch.setenv("HANA_STUDIO_RECENT_FILE", str(tmp_path / "recent_projects.json"))


@pytest.fixture(autouse=True)
def _isolated_length_units(tmp_path):
    """테스트가 사용자가 고른 길이 단위(설정)를 읽거나 바꾸지 않게 한다. 언제나 mm로 시작한다."""
    from PySide6.QtCore import QSettings

    from ui.units import LengthUnits

    LengthUnits.set_instance(LengthUnits(QSettings(str(tmp_path / "units.ini"), QSettings.IniFormat)))
    yield
    LengthUnits.set_instance(None)


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


@pytest.fixture(autouse=True)
def _silent_attention(monkeypatch):
    """시험 중에는 인쇄 멈춤 경고음(ui.attention)을 실제로 내지 않는다."""
    monkeypatch.setattr("ui.attention.play_alert_sound", lambda: None)
