"""데이터 바인딩: 항목의 `source`를 레코드 값으로 풀어낸다. Qt에 의존하지 않는다."""

from __future__ import annotations

import datetime as _dt
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Mapping

from design._serial import Serializable

_IMAGE_KIND = "image"
_CODE_KINDS = ("barcode", "qr")
_WEEKDAYS = "월화수목금토일"


def placeholder(column: str) -> str:
    return f"«{column}»"


@dataclass(kw_only=True)
class Source(Serializable):
    TYPE: ClassVar[str] = ""
    type: str = ""
    extra: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.TYPE:
            self.type = self.TYPE

    def columns(self) -> set[str]:
        return set()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]):
        if cls is Source:
            if not isinstance(data, Mapping):
                raise ValueError("source 형식이 올바르지 않습니다.")
            target = SOURCE_TYPES.get(data.get("type"), UnknownSource)
            return target.from_dict(data)
        return super().from_dict(data)


@dataclass(kw_only=True)
class FixedSource(Source):
    TYPE: ClassVar[str] = "fixed"
    value: str = ""


@dataclass(kw_only=True)
class ColumnSource(Source):
    TYPE: ClassVar[str] = "column"
    column: str = ""

    def columns(self) -> set[str]:
        return {self.column} if self.column else set()


@dataclass(kw_only=True)
class ConcatPart(Serializable):
    """column이 있으면 열 값, 없으면 text(고정 문구). 열 값이 비면 prefix/suffix도 숨는다."""

    column: str | None = None
    text: str = ""
    prefix: str = ""
    suffix: str = ""
    extra: dict = field(default_factory=dict, repr=False)


@dataclass(kw_only=True)
class ConcatSource(Source):
    TYPE: ClassVar[str] = "concat"
    parts: list[ConcatPart] = field(default_factory=list)
    separator: str = ""

    def columns(self) -> set[str]:
        return {p.column for p in self.parts if p.column}

    def _encode(self, name: str, value: Any) -> Any:
        if name == "parts":
            return [p.to_dict() for p in value]
        return super()._encode(name, value)

    @classmethod
    def _decode(cls, name: str, value: Any) -> Any:
        if name == "parts":
            return [ConcatPart.from_dict(p) for p in value]
        return super()._decode(name, value)


@dataclass(kw_only=True)
class CounterSource(Source):
    TYPE: ClassVar[str] = "counter"
    start: int = 1
    step: int = 1
    pad: int = 0
    prefix: str = ""
    suffix: str = ""


@dataclass(kw_only=True)
class DateSource(Source):
    TYPE: ClassVar[str] = "date"
    format: str = "%Y-%m-%d"
    offset_days: int = 0


@dataclass(kw_only=True)
class UnknownSource(Source):
    """이 버전이 모르는 source. 내용은 extra에 그대로 보존된다."""


SOURCE_TYPES: dict[str, type[Source]] = {
    c.TYPE: c for c in (FixedSource, ColumnSource, ConcatSource, CounterSource, DateSource)
}


@dataclass
class BindingContext:
    counter_index: int = 0
    now: Callable[[], _dt.datetime] = _dt.datetime.now
    photo_base_dir: str | None = None
    record_key: str | None = None


def to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "예" if value else "아니오"
    if isinstance(value, float):
        if value != value:
            return ""
        return str(int(value)) if value.is_integer() else str(value)
    if isinstance(value, _dt.datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, (_dt.date, _dt.time)):
        return value.isoformat()
    return str(value)


_DATE_DIRECTIVE = re.compile(r"%([-#]?)([A-Za-z%])")


def format_date(moment: _dt.datetime, fmt: str) -> str:
    """strftime 유사 형식. 로케일과 무관하게 동작하고 한글 문구를 그대로 둔다.

    지원: %Y %y %m %d %H %I %M %S %j %p %a %A %%, 그리고 0 없는 표기 %-m / %#m.
    """
    hour12 = moment.hour % 12 or 12
    table: dict[str, tuple[int, int]] = {
        "Y": (moment.year, 4), "y": (moment.year % 100, 2),
        "m": (moment.month, 2), "d": (moment.day, 2),
        "H": (moment.hour, 2), "I": (hour12, 2),
        "M": (moment.minute, 2), "S": (moment.second, 2),
        "j": (moment.timetuple().tm_yday, 3),
    }

    def repl(match: re.Match) -> str:
        flag, code = match.group(1), match.group(2)
        if code == "%":
            return "%"
        if code in table:
            number, width = table[code]
            return str(number) if flag else str(number).zfill(width)
        if code == "p":
            return "오전" if moment.hour < 12 else "오후"
        if code == "a":
            return _WEEKDAYS[moment.weekday()]
        if code == "A":
            return _WEEKDAYS[moment.weekday()] + "요일"
        return match.group(0)

    return _DATE_DIRECTIVE.sub(repl, fmt)


def _static_value(item: Any) -> Any:
    kind = getattr(item, "kind", "")
    if kind == "text":
        return getattr(item, "text", "")
    if kind in _CODE_KINDS:
        return getattr(item, "data", "")
    if kind == _IMAGE_KIND:
        return getattr(item, "asset", None)
    return None


def _concat(source: ConcatSource, record: Mapping[str, Any] | None) -> str:
    pieces: list[str] = []
    for part in source.parts:
        if part.column:
            value = placeholder(part.column) if record is None else to_text(record.get(part.column)).strip()
        else:
            value = part.text
        if value == "":
            continue
        pieces.append(f"{part.prefix}{value}{part.suffix}")
    return source.separator.join(pieces)


def _source_text(source: Source, record: Mapping[str, Any] | None, context: BindingContext) -> str:
    if isinstance(source, FixedSource):
        return source.value
    if isinstance(source, ColumnSource):
        if record is None:
            return placeholder(source.column)
        return to_text(record.get(source.column))
    if isinstance(source, ConcatSource):
        return _concat(source, record)
    if isinstance(source, CounterSource):
        number = source.start + source.step * context.counter_index
        sign = "-" if number < 0 else ""
        return f"{source.prefix}{sign}{str(abs(number)).zfill(max(0, source.pad))}{source.suffix}"
    if isinstance(source, DateSource):
        moment = context.now() + _dt.timedelta(days=source.offset_days)
        return format_date(moment, source.format)
    return ""


def resolve(item: Any, record: Mapping[str, Any] | None, context: BindingContext | None = None) -> Any:
    """항목에 찍힐 값. 글자·바코드는 str, 이미지는 파일 경로(str) 또는 None.

    record가 None이면 디자인 화면용 값: 글자는 `«열»`, 열에 묶인 이미지는 None.
    """
    context = context or BindingContext()
    source = getattr(item, "source", None)
    kind = getattr(item, "kind", "")
    if source is None:
        return _static_value(item)

    uses_columns = bool(source.columns())
    if kind == _IMAGE_KIND:
        if isinstance(source, FixedSource):
            return source.value or None
        if record is None and uses_columns:
            return None
        path = _source_text(source, record, context).strip()
        if not path:
            return None
        if context.photo_base_dir and not os.path.isabs(path):
            path = os.path.join(context.photo_base_dir, path)
        return path

    if record is None and uses_columns and kind in _CODE_KINDS:
        sample = getattr(item, "data", "")
        if sample:
            return sample
    return _source_text(source, record, context)


def columns_used(document: Any) -> set[str]:
    used: set[str] = set()
    for side in document.sides.values():
        for item in side.items:
            source = getattr(item, "source", None)
            if source is not None:
                used |= source.columns()
    return used
