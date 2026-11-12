import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest


@pytest.fixture
def settings(tmp_path):
    from PySide6.QtCore import QSettings

    return QSettings(str(tmp_path / "ui-test.ini"), QSettings.IniFormat)


@pytest.fixture(autouse=True)
def theme_manager(qapp, settings):
    """테스트마다 새 ThemeManager를 쓰고, 끝나면 앱 스타일을 원래대로 돌린다."""
    from ui.theme import ThemeManager

    manager = ThemeManager(settings)
    ThemeManager.set_instance(manager)
    manager.set_mode("light", persist=False)
    manager.apply(qapp)
    yield manager
    from PySide6.QtCore import QCoreApplication, QEvent

    for widget in qapp.topLevelWidgets():
        widget.close()
        widget.deleteLater()
    qapp.processEvents()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    ThemeManager.set_instance(None)
    qapp.setStyleSheet("")


@pytest.fixture
def show(qapp):
    """위젯을 띄우고 활성 창으로 만든 뒤 돌려준다."""

    kept = []

    def _show(widget):
        kept.append(widget)
        widget.show()
        widget.activateWindow()
        for _ in range(3):
            qapp.processEvents()
        return widget

    return _show


class SignalSpy:
    def __init__(self, signal):
        self.calls = []
        signal.connect(self._record)

    def _record(self, *args):
        self.calls.append(args)

    def __len__(self):
        return len(self.calls)

    @property
    def last(self):
        return self.calls[-1]


@pytest.fixture
def spy():
    return SignalSpy
