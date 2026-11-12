from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .errors import DataError
from .values import normalize_value, parse_number

OPERATORS = {
    "eq": "같음",
    "ne": "같지 않음",
    "contains": "포함",
    "not_contains": "포함하지 않음",
    "starts": "시작 글자",
    "ends": "끝 글자",
    "gt": "보다 큼",
    "ge": "이상",
    "lt": "보다 작음",
    "le": "이하",
    "between": "사이",
    "empty": "비어 있음",
    "not_empty": "비어 있지 않음",
}
_ALIASES = {
    "equals": "eq", "=": "eq", "==": "eq",
    "not-equals": "ne", "not_equals": "ne", "!=": "ne", "<>": "ne",
    "not-contains": "not_contains",
    "startswith": "starts", "starts_with": "starts",
    "endswith": "ends", "ends_with": "ends",
    ">": "gt", ">=": "ge", "<": "lt", "<=": "le",
    "not-empty": "not_empty",
}
_COMPARE = {"gt": ">", "ge": ">=", "lt": "<", "le": "<="}
_TEXT_OPS = ["eq", "ne", "contains", "not_contains", "starts", "ends", "empty", "not_empty"]
_ORDERED_OPS = ["eq", "ne", "gt", "ge", "lt", "le", "between", "empty", "not_empty"]


def operators_for(column_type: str) -> list[str]:
    return list(_ORDERED_OPS if column_type in ("number", "date", "time") else _TEXT_OPS)


class SqlContext(Protocol):
    def sql_column(self, name: str) -> tuple[str, str]: ...
    def sql_columns(self) -> list[tuple[str, str]]: ...


class Filter:
    def to_sql(self, ctx: SqlContext) -> tuple[str, list]:
        raise NotImplementedError

    def __and__(self, other: "Filter") -> "Group":
        return Group("and", [self, other])

    def __or__(self, other: "Filter") -> "Group":
        return Group("or", [self, other])


def _like_pattern(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _canonical(op: str) -> str:
    key = str(op).strip().lower()
    key = _ALIASES.get(key, key)
    if key not in OPERATORS:
        raise DataError(f"지원하지 않는 조건입니다: {op}")
    return key


@dataclass(frozen=True)
class Condition(Filter):
    column: str
    op: str
    value: Any = None

    def to_sql(self, ctx: SqlContext) -> tuple[str, list]:
        op = _canonical(self.op)
        expr, column_type = ctx.sql_column(self.column)

        if op == "empty":
            return f"({expr} IS NULL OR {expr} = '')", []
        if op == "not_empty":
            return f"({expr} IS NOT NULL AND {expr} <> '')", []

        if op == "between":
            if not isinstance(self.value, (tuple, list)) or len(self.value) != 2:
                raise DataError(f"'{self.column}' 조건에는 시작 값과 끝 값을 모두 입력하세요.")
            low_sql, low_params = Condition(self.column, "ge", self.value[0]).to_sql(ctx)
            high_sql, high_params = Condition(self.column, "le", self.value[1]).to_sql(ctx)
            return f"({low_sql} AND {high_sql})", low_params + high_params

        if self.value is None or (isinstance(self.value, str) and self.value == ""):
            raise DataError(f"'{self.column}' 조건에 비교할 값을 입력하세요.")

        if op in ("contains", "not_contains", "starts", "ends"):
            pattern = _like_pattern(str(self.value))
            if op != "ends":
                pattern += "%"
            if op in ("contains", "not_contains", "ends"):
                pattern = "%" + pattern
            like = f"CAST({expr} AS TEXT) LIKE ? ESCAPE '\\'"
            if op == "not_contains":
                return f"({expr} IS NULL OR NOT {like})", [pattern]
            return f"({expr} IS NOT NULL AND {like})", [pattern]

        value, numeric = self._coerce(column_type, strict=op in _COMPARE)
        if op == "eq":
            if numeric:
                return f"({expr} = ?)", [value]
            return f"({expr} = ? COLLATE NOCASE)", [value]
        if op == "ne":
            if numeric:
                return f"({expr} IS NULL OR {expr} <> ?)", [value]
            return f"({expr} IS NULL OR {expr} <> ? COLLATE NOCASE)", [value]
        symbol = _COMPARE[op]
        if numeric:
            return f"(typeof({expr}) IN ('integer', 'real') AND {expr} {symbol} ?)", [value]
        return f"(typeof({expr}) = 'text' AND {expr} {symbol} ?)", [value]

    def _coerce(self, column_type: str, strict: bool) -> tuple[Any, bool]:
        value = self.value
        if column_type == "bool":
            if isinstance(value, str):
                return int(value.strip().lower() in ("1", "true", "y", "yes", "예", "인쇄됨")), True
            return int(bool(value)), True
        if column_type == "number":
            number = parse_number(value)
            if number is not None:
                return number, True
            if strict:
                raise DataError(f"'{self.column}' 항목에는 숫자만 넣을 수 있습니다. 숫자를 입력하세요: {value}")
            return str(value), False
        if column_type in ("date", "time"):
            converted, ok = normalize_value(column_type, value)
            if not ok and strict:
                label = "날짜" if column_type == "date" else "시간"
                raise DataError(f"'{self.column}' 항목은 {label} 항목입니다. 올바른 {label}를 입력하세요: {value}")
            return converted, False
        return normalize_value("text", value)[0], False


@dataclass(frozen=True)
class Group(Filter):
    mode: str = "and"
    items: list = field(default_factory=list)

    def to_sql(self, ctx: SqlContext) -> tuple[str, list]:
        mode = str(self.mode).strip().lower()
        if mode not in ("and", "or"):
            raise DataError(f"조건 묶음은 AND 또는 OR여야 합니다: {self.mode}")
        if not self.items:
            return "1 = 1", []
        parts, params = [], []
        for item in self.items:
            sql, item_params = item.to_sql(ctx)
            parts.append(f"({sql})")
            params.extend(item_params)
        return f" {mode.upper()} ".join(parts), params


@dataclass(frozen=True)
class QuickSearch(Filter):
    text: str = ""

    def to_sql(self, ctx: SqlContext) -> tuple[str, list]:
        terms = str(self.text or "").split()
        if not terms:
            return "1 = 1", []
        columns = ctx.sql_columns()
        if not columns:
            return "0 = 1", []
        parts, params = [], []
        for term in terms:
            pattern = f"%{_like_pattern(term)}%"
            parts.append(
                "(" + " OR ".join(f"CAST({expr} AS TEXT) LIKE ? ESCAPE '\\'" for expr, _ in columns) + ")"
            )
            params.extend([pattern] * len(columns))
        return " AND ".join(parts), params


@dataclass(frozen=True)
class PrintedFilter(Filter):
    printed: bool = True

    def to_sql(self, ctx: SqlContext) -> tuple[str, list]:
        return "printed = ?", [1 if self.printed else 0]


def all_of(*filters: Filter) -> Group:
    return Group("and", [f for f in filters if f is not None])


def any_of(*filters: Filter) -> Group:
    return Group("or", [f for f in filters if f is not None])


def quick_search(text: str) -> QuickSearch:
    """모든 열에서 글자를 찾는다. 띄어 쓴 단어는 모두 포함해야 한다."""
    return QuickSearch(text)


def printed_only() -> PrintedFilter:
    return PrintedFilter(True)


def unprinted_only() -> PrintedFilter:
    return PrintedFilter(False)


def compile_filter(filter: Filter | None, ctx: SqlContext) -> tuple[str, list]:
    if filter is None:
        return "1 = 1", []
    return filter.to_sql(ctx)
