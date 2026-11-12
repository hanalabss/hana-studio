"""dataclass <-> dict 변환 공통부. 모르는 키는 `extra`에 담아 다시 저장할 때 돌려준다."""

from __future__ import annotations

import copy
from dataclasses import fields
from typing import Any, Mapping


def _coerce(type_name: str, key: str, value: Any) -> Any:
    base = type_name.replace(" ", "")
    optional = base.endswith("|None")
    if optional:
        if value is None:
            return None
        base = base[: -len("|None")]
    is_number = isinstance(value, (int, float)) and not isinstance(value, bool)
    if base == "float":
        if not is_number:
            raise ValueError(f"'{key}' 값이 숫자가 아닙니다: {value!r}")
        return float(value)
    if base == "int":
        if not is_number or float(value) != int(value):
            raise ValueError(f"'{key}' 값이 정수가 아닙니다: {value!r}")
        return int(value)
    if base == "bool":
        if not isinstance(value, bool):
            raise ValueError(f"'{key}' 값이 참/거짓이 아닙니다: {value!r}")
        return value
    if base == "str":
        if not isinstance(value, str):
            raise ValueError(f"'{key}' 값이 문자열이 아닙니다: {value!r}")
        return value
    return value


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


class Serializable:
    """`extra: dict` 필드를 가진 dataclass용 믹스인."""

    def _floats_from_ints(self) -> None:
        """float 필드에 정수가 들어오면 float로 바꾼다(mm 값은 항상 float)."""
        for f in fields(self):
            if str(f.type) == "float" and _is_int(getattr(self, f.name)):
                setattr(self, f.name, float(getattr(self, f.name)))

    def _encode(self, name: str, value: Any) -> Any:
        return copy.deepcopy(value)

    @classmethod
    def _decode(cls, name: str, value: Any) -> Any:
        return copy.deepcopy(value)

    @classmethod
    def _saved_fields(cls):
        return [
            f for f in fields(cls)
            if f.name != "extra" and not f.metadata.get("transient")
        ]

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for f in self._saved_fields():
            value = getattr(self, f.name)
            if str(f.type) == "float" and _is_int(value):
                value = float(value)
            out[f.name] = self._encode(f.name, value)
        for key, value in self.extra.items():
            out.setdefault(key, copy.deepcopy(value))
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]):
        if not isinstance(data, Mapping):
            raise ValueError(f"사전 형식이 아닌 데이터입니다: {type(data).__name__}")
        known = {f.name: f for f in cls._saved_fields()}
        kwargs: dict[str, Any] = {}
        extra: dict[str, Any] = {}
        for key, value in data.items():
            f = known.get(key)
            if f is None:
                extra[key] = copy.deepcopy(value)
            else:
                kwargs[key] = cls._decode(key, _coerce(str(f.type), key, value))
        obj = cls(**kwargs)
        obj.extra = extra
        return obj
