"""
utils/console.py
콘솔 출력(stdout/stderr)을 어떤 문자를 찍어도 죽지 않게 맞춘다.

기존 인쇄 코드는 진행 상황을 이모지가 섞인 print()로 남긴다. 출력이 파일·파이프로 돌려져 있거나
cp949 콘솔에 붙어 있으면 그 print()가 UnicodeEncodeError를 내고, 그 예외가 인쇄 흐름을 끊는다.
프로세스를 시작할 때 ensure_utf8_console()을 한 번 부르면 이후의 print()는 실패하지 않는다.
"""

from __future__ import annotations

import sys
from typing import Any, Iterable


def _is_utf8(encoding: Any) -> bool:
    return isinstance(encoding, str) and encoding.lower().replace("-", "").replace("_", "") == "utf8"


def _make_safe(stream: Any, errors: str) -> bool:
    """스트림 하나를 UTF-8 + 관대한 오류 처리로 바꾼다. 바꿨거나 이미 안전하면 True."""
    if stream is None:                      # 창 모드(pythonw, --windowed 빌드)에는 콘솔이 없다
        return False
    reconfigure = getattr(stream, "reconfigure", None)
    if not callable(reconfigure):           # StringIO 등 다른 객체로 바꿔 둔 경우는 건드리지 않는다
        return False
    try:
        if _is_utf8(getattr(stream, "encoding", None)):
            if getattr(stream, "errors", "strict") == "strict":
                reconfigure(errors=errors)
        else:
            reconfigure(encoding="utf-8", errors=errors)
    except (ValueError, OSError, LookupError, TypeError):
        # 이미 닫혔거나 인코딩을 바꿀 수 없는 스트림: 오류 처리만이라도 관대하게 해 본다.
        try:
            reconfigure(errors=errors)
        except Exception:
            return False
    return True


def ensure_utf8_console(errors: str = "replace", streams: Iterable[Any] | None = None) -> int:
    """stdout·stderr를 UTF-8로 맞추고, 찍을 수 없는 문자는 예외 대신 대체 문자로 내보내게 한다.

    여러 번 불러도 되고 예외를 던지지 않는다. 안전해진 스트림 수를 돌려준다.
    streams를 주지 않으면 sys.stdout, sys.stderr와 그 원본(sys.__stdout__, sys.__stderr__)을 다룬다.
    """
    if streams is None:
        streams = (sys.stdout, sys.stderr, sys.__stdout__, sys.__stderr__)
    seen: set[int] = set()
    count = 0
    for stream in streams:
        if stream is None or id(stream) in seen:
            continue
        seen.add(id(stream))
        if _make_safe(stream, errors):
            count += 1
    return count
