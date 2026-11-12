"""누구 카드를 뽑을지 정한다: 고르는 방식 → 사람 번호(_id) 목록."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from data.errors import DataError
from data.filters import printed_only
from data.selection import PrintSelection
from data.store import DataStore
from design.model import Document
from services.render_job import needs_records

from .text import plain

MODE_CHECKED = "checked"
MODE_UNPRINTED = "unprinted"
MODE_CURRENT = "current"
MODE_RANGES = "ranges"
MODE_ALL = "all"
MODE_PICKED = "picked"      # 기록 화면 등에서 넘겨받은 사람들
MODE_COPIES = "copies"      # 명단 없이 같은 카드 여러 장
PEOPLE_MODES = (MODE_CHECKED, MODE_UNPRINTED, MODE_CURRENT, MODE_RANGES, MODE_ALL, MODE_PICKED)


@dataclass
class TargetCounts:
    """고르는 방식마다 몇 명인지. 고를 수 없으면 None."""

    checked: int | None = None
    unprinted: int | None = None
    current: int | None = None
    current_name: str = ""
    visible: int | None = None      # 명단 화면에 지금 보이는 사람 수(번호로 고르기의 기준)
    total: int = 0
    printed: int = 0
    picked: int | None = None
    copies_available: bool = True


@dataclass
class Resolution:
    ids: list[int] = field(default_factory=list)
    error: str = ""             # 번호를 잘못 적었을 때 등, 사용자에게 보여 줄 말


def copies_available(document: Document, store: DataStore | None) -> bool:
    return store is None or not needs_records(document)


def count_targets(store: DataStore | None, document: Document, checked_ids: list[int],
                  current_id: int | None, view_filter: Any = None, picked_ids: list[int] | None = None,
                  name_column: str | None = None) -> TargetCounts:
    counts = TargetCounts(copies_available=copies_available(document, store))
    if store is None:
        return counts
    counts.total = store.count()
    counts.printed = store.count(printed_only())
    counts.unprinted = counts.total - counts.printed
    existing = set(store.ids()) if (checked_ids or picked_ids) else set()
    counts.checked = sum(1 for i in dict.fromkeys(checked_ids) if i in existing)
    if picked_ids:
        counts.picked = sum(1 for i in dict.fromkeys(picked_ids) if i in existing)
    try:
        counts.visible = store.count(view_filter)
    except Exception:
        counts.visible = counts.total
    record = store.get(current_id) if current_id is not None else None
    if record is not None:
        counts.current = 1
        value = record.get(name_column) if name_column else None
        counts.current_name = str(value).strip() if value not in (None, "") else ""
    return counts


def resolve_targets(mode: str, store: DataStore | None, *, checked_ids: list[int] = (),
                    current_id: int | None = None, range_text: str = "", view_filter: Any = None,
                    view_order: Any = None, picked_ids: list[int] = ()) -> Resolution:
    """mode에 맞는 사람들을 인쇄할 순서대로 돌려준다."""
    if mode == MODE_COPIES or store is None:
        return Resolution()
    try:
        if mode == MODE_CHECKED:
            return Resolution(PrintSelection.ids(checked_ids).resolve(store, None, view_order))
        if mode == MODE_PICKED:
            return Resolution(PrintSelection.ids(picked_ids).resolve(store, None, view_order))
        if mode == MODE_UNPRINTED:
            return Resolution(PrintSelection.unprinted().resolve(store, None, view_order))
        if mode == MODE_CURRENT:
            if current_id is None:
                return Resolution(error="명단에서 한 사람을 먼저 눌러 주세요.")
            return Resolution(PrintSelection.current(current_id).resolve(store))
        if mode == MODE_RANGES:
            if not str(range_text).strip():
                return Resolution(error="뽑을 번호를 적어 주세요. 예: 1,3,5-9")
            return Resolution(PrintSelection.ranges(range_text).resolve(store, view_filter, view_order))
        if mode == MODE_ALL:
            return Resolution(PrintSelection.all().resolve(store, view_filter, view_order))
    except DataError as exc:
        return Resolution(error=plain(str(exc)))
    return Resolution(error="뽑을 사람을 고르는 방식을 알 수 없습니다.")


def default_mode(counts: TargetCounts, store: DataStore | None) -> str:
    if store is None:
        return MODE_COPIES
    if counts.picked:
        return MODE_PICKED
    if counts.checked:
        return MODE_CHECKED
    if counts.unprinted:
        return MODE_UNPRINTED
    if counts.total:
        return MODE_ALL
    return MODE_COPIES if counts.copies_available else MODE_UNPRINTED
