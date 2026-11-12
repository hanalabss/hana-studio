from __future__ import annotations

import csv
import os
import tempfile
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Iterator

from .errors import DataError
from .store import Column, DataStore, Record
from .values import unique_names

STATUS_HEADERS = ("인쇄 여부", "인쇄 횟수", "마지막 인쇄 시각")
PRINTED_LABEL = "인쇄됨"


def _selected_columns(store: DataStore, columns: Iterable[str] | None) -> list[Column]:
    if columns is None:
        return store.columns()
    return [store.column(name) for name in columns]


def _headers(columns: list[Column], include_print_status: bool) -> list[str]:
    names = [c.name for c in columns]
    if include_print_status:
        names = unique_names(names + list(STATUS_HEADERS))
    return names


def _status_cells(record: Record) -> list[Any]:
    return [
        PRINTED_LABEL if record["_printed"] else "",
        record["_print_count"],
        record["_printed_at"] or "",
    ]


@contextmanager
def _atomic_target(path: str | Path) -> Iterator[str]:
    """임시 파일에 다 쓴 뒤 바꿔치기해서, 실패해도 기존 파일이 망가지지 않게 한다."""
    target = Path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        handle, temp_name = tempfile.mkstemp(prefix=".~export-", suffix=target.suffix, dir=target.parent)
        os.close(handle)
    except OSError as exc:
        raise DataError(f"이 위치에 파일을 저장할 수 없습니다: {target.parent}") from exc
    try:
        yield temp_name
        os.replace(temp_name, target)
    except PermissionError:
        raise DataError(
            f"파일을 저장할 수 없습니다. '{target.name}' 파일이 Excel 등 다른 프로그램에서 열려 있으면 닫고 다시 시도하세요."
        ) from None
    except OSError as exc:
        raise DataError(f"파일을 저장할 수 없습니다: {target}") from exc
    finally:
        if os.path.exists(temp_name):
            try:
                os.remove(temp_name)
            except OSError:
                pass


def export_csv(
    store: DataStore,
    path: str | Path,
    filter: Any = None,
    order: Any = None,
    include_print_status: bool = False,
    columns: Iterable[str] | None = None,
) -> int:
    """Excel에서 한글이 깨지지 않도록 BOM 있는 UTF-8로 쓴다. 내보낸 행 수를 돌려준다."""
    selected = _selected_columns(store, columns)
    records = store.rows(filter, order)
    with _atomic_target(path) as temp_name:
        with open(temp_name, "w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(_headers(selected, include_print_status))
            for record in records:
                row = ["" if record[c.name] is None else record[c.name] for c in selected]
                if include_print_status:
                    row += _status_cells(record)
                writer.writerow(row)
    return len(records)


def export_xlsx(
    store: DataStore,
    path: str | Path,
    filter: Any = None,
    order: Any = None,
    include_print_status: bool = False,
    columns: Iterable[str] | None = None,
    sheet_name: str = "데이터",
) -> int:
    from openpyxl import Workbook
    from openpyxl.cell import WriteOnlyCell

    selected = _selected_columns(store, columns)
    records = store.rows(filter, order)
    workbook = Workbook(write_only=True)
    sheet = workbook.create_sheet(sheet_name[:31] or "데이터")

    def text_cell(value: Any):
        # "=..."로 시작하는 값이 수식으로 실행되지 않도록 항상 문자열 칸으로 쓴다.
        cell = WriteOnlyCell(sheet, value=str(value))
        cell.data_type = "s"
        return cell

    def data_cell(column: Column, value: Any):
        if value is None:
            return None
        if column.type == "number" and isinstance(value, (int, float)):
            return value
        if column.type == "date" and isinstance(value, str):
            try:
                cell = WriteOnlyCell(sheet, value=date.fromisoformat(value))
                cell.number_format = "yyyy-mm-dd"
                return cell
            except ValueError:
                pass
        return text_cell(value)

    sheet.append([text_cell(name) for name in _headers(selected, include_print_status)])
    for record in records:
        row = [data_cell(c, record[c.name]) for c in selected]
        if include_print_status:
            printed, count, stamp = _status_cells(record)
            row += [text_cell(printed) if printed else None, count, text_cell(stamp) if stamp else None]
        sheet.append(row)

    with _atomic_target(path) as temp_name:
        workbook.save(temp_name)
        workbook.close()
    return len(records)
