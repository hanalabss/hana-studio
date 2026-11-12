"""명단 화면에서 쓰는 말. 사용자가 읽는 낱말은 여기서만 정한다."""

from __future__ import annotations

from datetime import datetime
from typing import Any

TYPE_NAMES = {
    "text": "글자",
    "number": "숫자",
    "date": "날짜",
    "time": "시간",
    "list": "고르는 목록",
    "photo": "사진",
}
TYPE_HINTS = {
    "text": "이름, 부서처럼 아무 글자나 적습니다.",
    "number": "나이, 금액처럼 숫자만 적습니다. 사번처럼 0으로 시작하면 글자로 두세요.",
    "date": "입사일처럼 달력에서 고릅니다.",
    "time": "14:30처럼 시각을 적습니다.",
    "list": "직급처럼 정해 둔 것 중에서 고릅니다.",
    "photo": "사진 파일 이름을 적거나 파일을 고릅니다.",
}

_OPERATOR_LABELS = {
    "eq": "같다",
    "ne": "같지 않다",
    "contains": "들어 있다",
    "not_contains": "들어 있지 않다",
    "starts": "(으)로 시작한다",
    "ends": "(으)로 끝난다",
    "gt": "보다 크다",
    "ge": "이상이다",
    "lt": "보다 작다",
    "le": "이하이다",
    "between": "사이에 있다",
    "empty": "비어 있다",
    "not_empty": "비어 있지 않다",
}
_DATE_LABELS = {"gt": "보다 뒤다", "ge": "부터", "lt": "보다 앞이다", "le": "까지"}

ENCODING_NAMES = (
    ("", "알아서 고르기"),
    ("cp949", "한글 Windows (엑셀에서 저장한 CSV)"),
    ("utf-8", "UTF-8"),
    ("utf-16", "UTF-16"),
)
DELIMITER_NAMES = (
    ("", "알아서 고르기"),
    (",", "쉼표 ( , )"),
    ("\t", "탭"),
    (";", "세미콜론 ( ; )"),
    ("|", "세로줄 ( | )"),
)


def operator_label(op: str, column_type: str = "text") -> str:
    if column_type in ("date", "time") and op in _DATE_LABELS:
        return _DATE_LABELS[op]
    return _OPERATOR_LABELS[op]


def people(count: int) -> str:
    return f"{count:,}명"


def has_final_consonant(word: str) -> bool:
    for ch in reversed(word.strip()):
        if "가" <= ch <= "힣":
            return (ord(ch) - 0xAC00) % 28 != 0
        if ch.isdigit():
            return ch in "013678"
        if ch.isalpha():
            return ch.lower() in "lmnr"
    return False


def josa(word: str, with_final: str, without_final: str) -> str:
    """josa("사번", "을", "를") -> "사번을"."""
    return f"{word}{with_final if has_final_consonant(word) else without_final}"


def short_date(stamp: Any) -> str:
    """'2026-11-10 09:12:00' -> '11월 10일'. 올해가 아니면 해도 붙인다."""
    if not stamp:
        return ""
    try:
        moment = datetime.fromisoformat(str(stamp))
    except ValueError:
        return str(stamp)
    text = f"{moment.month}월 {moment.day}일"
    if moment.year != datetime.now().year:
        text = f"{moment.year}년 {text}"
    return text


def status_text(record: dict, photo_missing: bool) -> tuple[str, str]:
    """(보여 줄 글자, 색 역할)."""
    if record.get("_printed"):
        count = int(record.get("_print_count") or 1)
        return f"{short_date(record.get('_printed_at'))} {count}장".strip(), "ok_text"
    if photo_missing:
        return "사진 없음", "warn_text"
    return "아직", "text_muted"


def display_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)
