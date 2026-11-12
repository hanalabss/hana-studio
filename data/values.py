from __future__ import annotations

import math
import re
from datetime import date, datetime, time
from typing import Any, Iterable

COLUMN_TYPES = ("text", "number", "date", "time", "list", "photo")
TYPE_LABELS = {
    "text": "글자",
    "number": "숫자",
    "date": "날짜",
    "time": "시간",
    "list": "목록",
    "photo": "사진",
}
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp")

_NUM_RE = re.compile(r"^[+-]?(\d{1,3}(,\d{3})+|\d+)(\.\d+)?$")
_LEADING_ZERO_RE = re.compile(r"^[+-]?0\d")
_DATE_RES = (
    re.compile(r"^(\d{4})\s*[-./]\s*(\d{1,2})\s*[-./]\s*(\d{1,2})\s*\.?$"),
    re.compile(r"^(\d{2})\s*[-./]\s*(\d{1,2})\s*[-./]\s*(\d{1,2})\s*\.?$"),
    re.compile(r"^(\d{4}|\d{2})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일$"),
)
_MIDNIGHT_RE = re.compile(r"[ T]0{1,2}:00(:00(\.0+)?)?$")
_TIME_RE = re.compile(r"^(\d{1,2}):(\d{2})(?::(\d{2}))?$")
_AMPM_RE = re.compile(r"^(오전|오후|am|pm|a\.m\.|p\.m\.)\s*|\s*(오전|오후|am|pm|a\.m\.|p\.m\.)$", re.I)
_PHOTO_NAME_RE = re.compile(r"사진|이미지|photo|image|picture", re.I)
_INT64_LIMIT = 2**63


def is_empty(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def parse_number(value: Any) -> int | float | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value if abs(value) < _INT64_LIMIT else None
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    text = str(value).strip()
    if not _NUM_RE.match(text):
        return None
    text = text.replace(",", "")
    if "." in text:
        return float(text)
    number = int(text)
    return number if abs(number) < _INT64_LIMIT else None


def has_leading_zero(text: str) -> bool:
    """'007', '0101234' 처럼 숫자로 바꾸면 앞자리 0이 사라지는 값."""
    return bool(_LEADING_ZERO_RE.match(text.strip()))


def parse_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date() if value.time() == time(0) else None
    if isinstance(value, date):
        return value
    text = str(value).strip()
    text = _MIDNIGHT_RE.sub("", text)
    for pattern in _DATE_RES:
        match = pattern.match(text)
        if not match:
            continue
        year, month, day = (int(part) for part in match.groups())
        if year < 100:
            year += 1900 if year >= 70 else 2000
        try:
            return date(year, month, day)
        except ValueError:
            return None
    return None


def parse_time(value: Any) -> str | None:
    if isinstance(value, datetime):
        value = value.time()
    if isinstance(value, time):
        return value.strftime("%H:%M:%S" if value.second else "%H:%M")
    text = str(value).strip()
    marker = _AMPM_RE.search(text)
    if marker:
        text = _AMPM_RE.sub("", text).strip()
    match = _TIME_RE.match(text)
    if not match:
        return None
    hour, minute = int(match.group(1)), int(match.group(2))
    second = int(match.group(3)) if match.group(3) else None
    if marker:
        word = (marker.group(1) or marker.group(2)).lower().replace(".", "")
        if hour < 1 or hour > 12:
            return None
        if word in ("오후", "pm"):
            hour = hour % 12 + 12
        else:
            hour = hour % 12
    if hour > 23 or minute > 59 or (second or 0) > 59:
        return None
    if second is None:
        return f"{hour:02d}:{minute:02d}"
    return f"{hour:02d}:{minute:02d}:{second:02d}"


def _to_text(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, (date, time)):
        return value.isoformat()
    return str(value)


def normalize_value(column_type: str, value: Any) -> tuple[Any, bool]:
    """저장용 값으로 바꾼다. 형식에 맞지 않으면 (원래 글자, False)를 돌려주어 값을 잃지 않는다."""
    if value is None:
        return None, True
    if isinstance(value, str) and value == "":
        return None, True

    if column_type == "number":
        if isinstance(value, str):
            text = value.strip()
            if text == "":
                return None, True
            if has_leading_zero(text):
                return text, False
            number = parse_number(text)
            return (number, True) if number is not None else (text, False)
        number = parse_number(value)
        return (number, True) if number is not None else (_to_text(value), False)

    if column_type == "date":
        if isinstance(value, str) and value.strip() == "":
            return None, True
        parsed = parse_date(value)
        if parsed is not None:
            return parsed.isoformat(), True
        return _to_text(value).strip(), False

    if column_type == "time":
        if isinstance(value, str) and value.strip() == "":
            return None, True
        parsed_time = parse_time(value)
        if parsed_time is not None:
            return parsed_time, True
        return _to_text(value).strip(), False

    return _to_text(value), True


def guess_type(samples: Iterable[str], name: str = "") -> str:
    values = [s.strip() for s in samples if s is not None and s.strip() != ""]
    if not values:
        return "photo" if _PHOTO_NAME_RE.search(name or "") else "text"
    if all(v.lower().endswith(IMAGE_EXTENSIONS) for v in values):
        return "photo"
    if _PHOTO_NAME_RE.search(name or ""):
        return "photo"
    if all(parse_date(v) is not None for v in values):
        return "date"
    if all(parse_time(v) is not None for v in values):
        return "time"
    if all(_looks_like_number(v) for v in values):
        return "number"
    return "text"


def _looks_like_number(text: str) -> bool:
    if text.startswith("+"):
        return False
    if has_leading_zero(text):
        return False
    if len(re.sub(r"\D", "", text.split(".")[0])) > 15:
        return False
    return parse_number(text) is not None


def clean_column_name(name: Any, index: int = 0) -> str:
    """파일 머리글을 저장소 열 이름으로 쓸 수 있게 다듬는다."""
    text = "" if name is None else re.sub(r"\s+", " ", str(name)).strip()
    text = text.lstrip("_").strip()
    return text or f"열{index + 1}"


def unique_names(names: Iterable[str]) -> list[str]:
    seen: dict[str, int] = {}
    taken: set[str] = set()
    result = []
    for name in names:
        candidate = name
        while candidate in taken:
            seen[name] = seen.get(name, 1) + 1
            candidate = f"{name} ({seen[name]})"
        taken.add(candidate)
        result.append(candidate)
    return result
