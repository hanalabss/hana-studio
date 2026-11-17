"""Mifare Classic(비접촉 IC 카드) 블록 쓰기 준비. 다눌의 RF 항목과 같은 일을 한다.

다눌은 블록 하나에 16바이트(16진 32글자)를 쓰고, 모자라면 뒤를 '0'으로 채운 뒤 다시 읽어 맞는지 본다.
여기서도 같게 하되, 카드를 못 쓰게 만들 수 있는 블록(0번 제조사 블록, 각 섹터의 마지막 '키 블록')에는
쓰지 않는다.

SDK 값 가운데 실기로 확인하지 못한 것: 비접촉 인코더로 카드를 옮기는 위치 번호(1~4 가운데 하나),
키 종류(A/B)를 SDK에 넘기는 숫자. 위치는 카드 번호(UID)가 읽히는 곳을 차례로 찾아 쓰고, 찾은 값은
기억해 두었다가 다음 카드부터 바로 쓴다.
"""

from __future__ import annotations

import string
from dataclasses import dataclass
from typing import Any, Mapping

BLOCK_BYTES = 16
MAX_BLOCK_1K = 63
DEFAULT_BLOCK = 4
DEFAULT_KEY = "FFFFFFFFFFFF"          # 공장 출하 기본 키
KEY_TYPES = ("A", "B")
FORMATS = ("text", "hex")
# 비접촉 인코더 위치 후보(SDK 문서 3.5: 일반 1·2, 3-in-1 3·4). 앞에서부터 시험한다.
PROBE_POSITIONS = (1, 2, 3, 4)
# 프린터(주소)별로 카드 번호가 읽힌 위치. 'IC 카드 시험'과 인쇄가 서로 다른 연결을 써도 함께 본다.
KNOWN_POSITIONS: dict[str, int] = {}
_HEX = set(string.hexdigits)


class MifareError(ValueError):
    """사용자에게 그대로 보여 줄 수 있는 한국어 메시지."""


def is_trailer(block: int) -> bool:
    """섹터의 마지막 블록(키와 접근 권한이 들어 있음). 1K 카드는 4블록이 한 섹터다."""
    return block % 4 == 3


def check_block(block: Any) -> int:
    try:
        number = int(block)
    except (TypeError, ValueError):
        raise MifareError("블록 번호는 숫자여야 합니다.") from None
    if not 1 <= number <= MAX_BLOCK_1K:
        raise MifareError(f"블록 번호는 1~{MAX_BLOCK_1K} 사이여야 합니다. (0번은 카드 번호가 든 블록입니다)")
    if is_trailer(number):
        raise MifareError(f"{number}번은 키가 들어 있는 블록이라 쓰면 카드를 못 쓰게 될 수 있습니다. "
                          f"{number - 1}번이나 {number + 1}번을 쓰세요.")
    return number


def check_key(key: Any) -> str:
    text = str(key or "").replace(" ", "").upper()
    if len(text) != 12 or not set(text) <= _HEX:
        raise MifareError("키는 16진수 12글자여야 합니다. (예: FFFFFFFFFFFF)")
    return text


def block_data(value: Any, fmt: str = "text") -> str:
    """블록에 쓸 16진 32글자. text는 글자(UTF-8)를, hex는 16진 글자를 그대로 넣고 모자라면 0으로 채운다."""
    text = "" if value is None else str(value)
    if fmt == "hex":
        cleaned = text.replace(" ", "").upper()
        if not set(cleaned) <= _HEX:
            raise MifareError("16진수(0~9, A~F)만 쓸 수 있습니다.")
        if len(cleaned) > BLOCK_BYTES * 2:
            raise MifareError(f"블록 하나에는 16진수 {BLOCK_BYTES * 2}글자까지 들어갑니다. (지금 {len(cleaned)}글자)")
        return cleaned.ljust(BLOCK_BYTES * 2, "0")
    raw = text.encode("utf-8")
    if len(raw) > BLOCK_BYTES:
        raise MifareError(f"블록 하나에는 {BLOCK_BYTES}바이트까지 들어갑니다. "
                          f"(영문·숫자 16자, 한글 5자 정도. 지금 {len(raw)}바이트)")
    return raw.hex().upper().ljust(BLOCK_BYTES * 2, "0")


@dataclass(frozen=True)
class MifareWrite:
    """카드 한 장에 쓸 것. 인쇄 옵션(card.options['mifare'])으로 넘긴다."""

    block: int
    data: str                  # 16진 32글자
    key: str = DEFAULT_KEY
    key_type: str = "A"
    position: int | None = None   # None이면 찾아서 쓴다

    def to_dict(self) -> dict[str, Any]:
        return {"block": self.block, "data": self.data, "key": self.key, "key_type": self.key_type,
                "position": self.position}

    @classmethod
    def from_options(cls, options: Mapping[str, Any] | None) -> "MifareWrite | None":
        if not options:
            return None
        return cls(block=check_block(options["block"]), data=str(options["data"]).upper(),
                   key=check_key(options.get("key", DEFAULT_KEY)),
                   key_type=str(options.get("key_type", "A")).upper(),
                   position=options.get("position"))


def sdk_key_type(key_type: str) -> int:
    """SDK의 ucPwdType. A=0, B=1로 넘긴다(실기 확인 전 값)."""
    return 1 if str(key_type).upper() == "B" else 0
