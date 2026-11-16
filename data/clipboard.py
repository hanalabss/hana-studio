"""명단 표와 엑셀 사이의 복사·붙여 넣기(다눌의 Ctrl+C / Ctrl+V와 같은 동작).

붙여 넣기는 고른 칸에서 시작해 오른쪽·아래로 덮어쓰고, 명단 끝을 넘는 줄은 새 사람으로 추가한다.
덮어쓴 값과 새로 넣은 사람을 기억해 두었다가 `undo_paste`로 되돌릴 수 있다.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from .store import DataStore


def _cell_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def to_tsv(rows: Sequence[Sequence[Any]]) -> str:
    """엑셀에 그대로 붙여 넣을 수 있는 탭 구분 글자. 줄바꿈·탭·따옴표가 든 칸은 따옴표로 감싼다."""
    out = io.StringIO()
    writer = csv.writer(out, delimiter="\t", lineterminator="\r\n", quoting=csv.QUOTE_MINIMAL)
    for row in rows:
        writer.writerow([_cell_text(v) for v in row])
    return out.getvalue()


def parse_tsv(text: str) -> list[list[str]]:
    """엑셀·구글 시트에서 복사한 글자를 칸 목록으로. 끝의 빈 줄은 버린다."""
    if not text:
        return []
    rows = [list(row) for row in csv.reader(io.StringIO(text.replace("\r\n", "\n")), delimiter="\t")]
    while rows and all(cell == "" for cell in rows[-1]):
        rows.pop()
    return rows


def looks_like_header(row: Sequence[str], names: Sequence[str]) -> bool:
    """첫 줄이 명단 항목 이름과 같으면(엑셀에서 머리글까지 복사한 경우) 값이 아니라 머리글이다."""
    cells = [c.strip() for c in row if c.strip()]
    return bool(cells) and all(c in set(names) for c in cells)


@dataclass
class PasteResult:
    updated: dict[int, dict[str, Any]] = field(default_factory=dict)   # 사람 id → 덮어쓰기 전 값
    inserted: list[int] = field(default_factory=list)
    skipped_header: bool = False
    cut_columns: int = 0          # 오른쪽에 항목이 모자라 버린 칸 수
    invalid: list[tuple[int, str, Any]] = field(default_factory=list)

    @property
    def changed_rows(self) -> int:
        return len(self.updated) + len(self.inserted)


def paste_grid(
    store: DataStore, grid: Sequence[Sequence[str]], field_names: Sequence[str],
    target_ids: Sequence[int], *, skip_header: bool = True,
) -> PasteResult:
    """grid를 field_names(시작 칸부터 오른쪽 순서) × target_ids(시작 줄부터 아래 순서)에 넣는다.

    target_ids를 다 쓰면 남은 줄은 새 사람으로 추가한다. 한 번의 트랜잭션으로 처리한다.
    """
    result = PasteResult()
    rows = [list(r) for r in grid]
    names: list[str | None] = list(field_names)
    if skip_header and rows and looks_like_header(rows[0], store.column_names()):
        # 머리글까지 복사했으면 칸 순서가 아니라 머리글 이름으로 항목을 맞춘다. 빈 머리글 칸은 버린다.
        names = [cell.strip() or None for cell in rows[0]]
        rows = rows[1:]
        result.skipped_header = True
    if result.skipped_header:
        width = max((len(r) for r in rows), default=0)
        result.cut_columns = sum(1 for i in range(width) if i >= len(names) or names[i] is None)
        rows = [[cell for i, cell in enumerate(r) if i < len(names) and names[i] is not None] for r in rows]
        names = [n for n in names if n is not None]
    if not rows or not names:
        return result
    if not result.skipped_header:
        result.cut_columns = max(0, max(len(r) for r in rows) - len(names))
    with store.transaction():
        new_rows: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            values = {name: row[i] for i, name in enumerate(names) if i < len(row)}
            if index < len(target_ids):
                record_id = target_ids[index]
                before = store.get(record_id)
                if before is None:
                    continue
                result.updated[record_id] = {name: before.get(name) for name in values}
                store.update(record_id, values)
            else:
                new_rows.append(values)
        if new_rows:
            start = max(store.ids() or [0])
            store.insert_many(new_rows, on_invalid=lambda i, n, v: result.invalid.append((i, n, v)))
            result.inserted = [i for i in store.ids() if i > start]
    return result


def undo_paste(store: DataStore, result: PasteResult) -> None:
    """paste_grid가 바꾼 것을 되돌린다."""
    with store.transaction():
        if result.inserted:
            store.delete(result.inserted)
        for record_id, values in result.updated.items():
            if store.get(record_id) is not None:
                store.update(record_id, values)


def copy_rows(records: Sequence[Mapping[str, Any]], field_names: Sequence[str],
              with_header: bool = False) -> str:
    rows: list[list[Any]] = [list(field_names)] if with_header else []
    rows += [[record.get(name) for name in field_names] for record in records]
    return to_tsv(rows)


PasteFn = Callable[[DataStore, Sequence[Sequence[str]], Sequence[str], Sequence[int]], PasteResult]
