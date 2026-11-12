from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from .errors import DataError, RangeError
from .filters import all_of, unprinted_only

_EXAMPLE = "예: 1,3,5-9"
_DASHES = str.maketrans({"~": "-", "～": "-", "〜": "-", "–": "-", "—": "-", "－": "-", "，": ",", "、": ","})


def parse_ranges(text: str, total: int) -> list[int]:
    """'1,3,4-9' → [1,3,4,5,6,7,8,9]. 번호는 1부터 total까지, 입력한 순서를 지키고 중복은 한 번만."""
    if total <= 0:
        raise RangeError("인쇄할 레코드가 없습니다.")
    cleaned = re.sub(r"\s*-\s*", "-", str(text or "").translate(_DASHES).strip())
    tokens = [t for t in re.split(r"[,;\s]+", cleaned) if t]
    if not tokens:
        raise RangeError(f"인쇄할 범위를 입력하세요. {_EXAMPLE}")

    def check(number: int) -> int:
        if number < 1:
            raise RangeError(f"번호는 1부터 시작합니다. {_EXAMPLE}")
        if number > total:
            raise RangeError(
                f"{number}번은 범위를 벗어났습니다. 전체 {total}건이므로 1부터 {total} 사이로 입력하세요."
            )
        return number

    result: dict[int, None] = {}
    for token in tokens:
        if re.fullmatch(r"\d+", token):
            result[check(int(token))] = None
            continue
        match = re.fullmatch(r"(\d*)-(\d*)", token)
        if not match or not (match.group(1) or match.group(2)):
            raise RangeError(f"'{token}'은(는) 올바른 범위가 아닙니다. {_EXAMPLE}")
        start = check(int(match.group(1))) if match.group(1) else 1
        end = check(int(match.group(2))) if match.group(2) else total
        if start > end:
            raise RangeError(
                f"'{token}': 시작 번호({start})가 끝 번호({end})보다 큽니다. '{end}-{start}'처럼 작은 번호를 먼저 입력하세요."
            )
        for number in range(start, end + 1):
            result[number] = None
    return list(result)


def format_ranges(numbers: Iterable[int]) -> str:
    """[1,3,4,5,9] → '1,3-5,9'"""
    ordered = sorted(set(int(n) for n in numbers))
    parts = []
    index = 0
    while index < len(ordered):
        end = index
        while end + 1 < len(ordered) and ordered[end + 1] == ordered[end] + 1:
            end += 1
        parts.append(str(ordered[index]) if end == index else f"{ordered[index]}-{ordered[end]}")
        index = end + 1
    return ",".join(parts)


@dataclass(frozen=True)
class PrintSelection:
    """인쇄 대상. resolve()의 filter/order는 지금 화면(그리드)에 적용된 필터와 정렬이다."""

    mode: str = "all"
    record_id: int | None = None
    range_text: str = ""
    record_ids: tuple[int, ...] = field(default_factory=tuple)
    filter: Any = None

    @classmethod
    def current(cls, record_id: int) -> "PrintSelection":
        return cls("current", record_id=int(record_id))

    @classmethod
    def all(cls) -> "PrintSelection":
        return cls("all")

    @classmethod
    def ranges(cls, text: str) -> "PrintSelection":
        return cls("ranges", range_text=str(text))

    @classmethod
    def ids(cls, ids: Iterable[int]) -> "PrintSelection":
        return cls("ids", record_ids=tuple(dict.fromkeys(int(i) for i in ids)))

    @classmethod
    def by_filter(cls, filter: Any) -> "PrintSelection":
        return cls("filter", filter=filter)

    @classmethod
    def unprinted(cls) -> "PrintSelection":
        return cls("unprinted")

    def resolve(self, store, filter: Any = None, order: Any = None) -> list[int]:
        """인쇄할 레코드 _id를 인쇄 순서대로 돌려준다."""
        if self.mode == "current":
            if self.record_id is None or store.get(self.record_id) is None:
                raise DataError("선택한 레코드를 찾을 수 없습니다.")
            return [self.record_id]
        if self.mode == "all":
            return store.ids(filter, order)
        if self.mode == "ranges":
            visible = store.ids(filter, order)
            return [visible[number - 1] for number in parse_ranges(self.range_text, len(visible))]
        if self.mode == "ids":
            wanted = set(self.record_ids)
            return [record_id for record_id in store.ids(None, order) if record_id in wanted]
        if self.mode == "filter":
            return store.ids(self.filter, order)
        if self.mode == "unprinted":
            return store.ids(all_of(filter, unprinted_only()), order)
        raise DataError(f"알 수 없는 인쇄 대상입니다: {self.mode}")

    def count(self, store, filter: Any = None, order: Any = None) -> int:
        return len(self.resolve(store, filter, order))

    def describe(self) -> str:
        if self.mode == "current":
            return "현재 레코드"
        if self.mode == "all":
            return "전체"
        if self.mode == "ranges":
            return f"범위 {self.range_text}"
        if self.mode == "ids":
            return f"선택한 {len(self.record_ids)}건"
        if self.mode == "filter":
            return "조건에 맞는 레코드"
        if self.mode == "unprinted":
            return "인쇄하지 않은 레코드"
        return self.mode
