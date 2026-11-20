"""인쇄가 멈췄을 때 알림 창을 띄우고 '확인'까지 소리를 내는 AttentionAlert."""
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QWidget

from ui.attention import AttentionAlert


def make(rings, repeat_ms=30):
    return AttentionAlert(sound=lambda: rings.append(1), repeat_ms=repeat_ms)


def test_shows_a_notice_and_keeps_ringing_until_confirmed(qapp):
    rings, confirmed = [], []
    alert = make(rings)
    alert.confirmed.connect(lambda: confirmed.append(1))
    window = QWidget()
    alert.start(window, "인쇄가 멈췄습니다", "카드 공급함이 비어 있습니다.")
    assert rings == [1] and alert.active
    assert alert.notice.isVisible() and "카드 공급함" in alert.notice.description()
    QTest.qWait(150)
    assert len(rings) >= 3 and alert.active              # 다른 곳을 눌러도, 시간이 지나도 계속 운다
    alert.notice.confirm_button.click()
    assert not alert.active and alert.notice is None and confirmed == [1]
    count = len(rings)
    QTest.qWait(100)
    assert len(rings) == count


def test_closing_the_notice_window_also_stops_it(qapp):
    rings = []
    alert = make(rings)
    alert.start(None, "인쇄가 멈췄습니다", "")
    alert.notice.close()
    assert not alert.active


def test_stop_closes_the_notice_without_confirming(qapp):
    rings, confirmed = [], []
    alert = make(rings)
    alert.confirmed.connect(lambda: confirmed.append(1))
    alert.start(None, "인쇄가 멈췄습니다", "")
    alert.stop()
    assert not alert.active and alert.notice is None and confirmed == []


def test_a_second_stop_replaces_the_first_notice(qapp):
    rings = []
    alert = make(rings)
    alert.start(None, "첫 번째", "")
    first = alert.notice
    alert.start(None, "두 번째", "")
    assert alert.notice is not first and alert.notice.title() == "두 번째"
    alert.stop()


def test_a_broken_sound_does_not_raise(qapp):
    def boom():
        raise OSError("no audio")
    alert = AttentionAlert(sound=boom, repeat_ms=1000)
    alert.start(None)
    assert alert.active
    alert.stop()


def test_alert_sound_file_ships_with_the_app():
    import os
    import wave
    from ui.attention import alert_sound_path
    path = alert_sound_path()
    assert os.path.isfile(path)
    with wave.open(path) as sound:
        assert 0.5 < sound.getnframes() / sound.getframerate() < 2.0   # 짧은 '띠띠' 한 묶음


def test_every_dialog_and_the_stop_notice_carry_the_hana_studio_logo(qapp):
    from ui.widgets.dialogs import Dialog
    plain = Dialog("아무 알림", "내용")
    assert plain.logo.pixmap() is not None and not plain.logo.pixmap().isNull()
    assert not plain.logo.isHidden()
    alert = AttentionAlert(sound=lambda: None, repeat_ms=1000)
    alert.start(None, "인쇄가 멈췄습니다", "")
    assert not alert.notice.logo.pixmap().isNull()
    alert.stop()
