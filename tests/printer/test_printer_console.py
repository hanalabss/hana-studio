"""utils.console: 이모지가 섞인 print()가 cp949 콘솔·리다이렉트에서 죽지 않게 한다."""
import io
import sys

import pytest

from utils.console import ensure_utf8_console

EMOJI_LINE = "📄 1번째 카드 인쇄 중 ✅"


def cp949_stream():
    raw = io.BytesIO()
    return raw, io.TextIOWrapper(raw, encoding="cp949", errors="strict")


def test_cp949_stream_raises_on_emoji_until_it_is_made_safe():
    raw, stream = cp949_stream()
    with pytest.raises(UnicodeEncodeError):
        print(EMOJI_LINE, file=stream, flush=True)

    raw, stream = cp949_stream()
    assert ensure_utf8_console(streams=[stream]) == 1
    print(EMOJI_LINE, file=stream, flush=True)
    assert raw.getvalue().decode("utf-8").strip() == EMOJI_LINE
    assert (stream.encoding.lower(), stream.errors) == ("utf-8", "replace")


def test_process_streams_are_covered_and_the_call_is_idempotent(monkeypatch):
    out_raw, out = cp949_stream()
    err_raw, err = cp949_stream()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    assert ensure_utf8_console() >= 2
    assert ensure_utf8_console() >= 2
    print(EMOJI_LINE)
    print("⚠️ 경고", file=sys.stderr)
    out.flush()
    err.flush()
    assert EMOJI_LINE in out_raw.getvalue().decode("utf-8")
    assert "경고" in err_raw.getvalue().decode("utf-8")


def test_utf8_stream_keeps_its_encoding_but_stops_raising():
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="utf-8", errors="strict")
    assert ensure_utf8_console(streams=[stream]) == 1
    stream.write("\udc80 깨진 글자도 넘어간다")
    stream.flush()
    assert "깨진 글자도 넘어간다" in raw.getvalue().decode("utf-8")


def test_missing_replaced_or_closed_streams_never_raise(monkeypatch):
    closed_raw, closed = cp949_stream()
    closed.close()
    assert ensure_utf8_console(streams=[None, io.StringIO(), object(), closed]) == 0
    monkeypatch.setattr(sys, "stdout", None)            # 창 모드 빌드에는 콘솔이 없다
    monkeypatch.setattr(sys, "stderr", None)
    ensure_utf8_console()


def test_hw_scripts_setup_makes_the_console_safe(monkeypatch):
    import os

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    monkeypatch.chdir(root)
    monkeypatch.syspath_prepend(os.path.join(root, "scripts"))
    raw, stream = cp949_stream()
    monkeypatch.setattr(sys, "stdout", stream)
    monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
    try:
        import hw_common
        hw_common.setup()
        print(EMOJI_LINE, flush=True)
        assert EMOJI_LINE in raw.getvalue().decode("utf-8")
    finally:
        sys.modules.pop("hw_common", None)
