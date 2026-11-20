"""인쇄가 멈췄을 때 소리로 알리는 AttentionAlert."""
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QWidget

from ui.attention import AttentionAlert


def make(rings, repeat_ms=30, max_ms=10_000):
    return AttentionAlert(sound=lambda: rings.append(1), repeat_ms=repeat_ms, max_ms=max_ms)


def test_rings_at_once_and_keeps_ringing(qapp):
    rings = []
    alert = make(rings)
    alert.start(QWidget())
    assert rings == [1] and alert.active
    QTest.qWait(120)
    assert len(rings) >= 3
    alert.stop()


def test_any_key_or_click_in_the_app_stops_it(qapp):
    rings = []
    alert = make(rings)
    alert.start(None)
    QApplication.sendEvent(qapp, QKeyEvent(QEvent.KeyPress, Qt.Key_A, Qt.NoModifier))
    assert not alert.active
    count = len(rings)
    QTest.qWait(100)
    assert len(rings) == count                       # 더 울리지 않는다


def test_gives_up_after_the_time_limit(qapp):
    rings = []
    alert = make(rings, repeat_ms=20, max_ms=80)
    alert.start(None)
    QTest.qWait(200)
    assert not alert.active


def test_a_broken_sound_does_not_raise(qapp):
    def boom():
        raise OSError("no audio")
    alert = AttentionAlert(sound=boom, repeat_ms=1000)
    alert.start(None)
    assert alert.active
    alert.stop()
