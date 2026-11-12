"""scripts/hw_*.py 공용: 경로 설정, 한국어 출력, SDK 잡음 가리기."""

from __future__ import annotations

import contextlib
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def setup() -> None:
    """프로젝트 루트를 작업 디렉터리와 임포트 경로로 삼고, 콘솔 출력을 UTF-8로 맞춘다."""
    os.chdir(ROOT)
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    # 기존 인쇄 코드의 이모지 섞인 print()가 cp949 콘솔·리다이렉트에서 죽지 않게 한다.
    from utils.console import ensure_utf8_console

    ensure_utf8_console()


def say(text: str = "") -> None:
    sys.__stdout__.write(text + "\n")
    sys.__stdout__.flush()


@contextlib.contextmanager
def quiet(verbose: bool):
    """기존 인쇄 코드가 찍는 디버그 문구를 가린다. verbose면 그대로 보여 준다."""
    if verbose:
        yield
        return
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        yield


def connection_label(connection: str) -> str:
    return {"usb": "USB", "tcp": "네트워크", "net": "네트워크", "fake": "가상"}.get(connection, connection or "-")


def family_label(family: str) -> str:
    return {"rtai": "RTAI R600", "smart": "IDP SMART"}.get(family, family)
