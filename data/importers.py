from __future__ import annotations

import csv
import difflib
import io
import re
from dataclasses import dataclass, field
from datetime import date, datetime, time
from pathlib import Path
from typing import Any

from .errors import DataError
from .store import Column, DataStore
from .values import (
    TYPE_LABELS,
    clean_column_name,
    guess_type,
    has_leading_zero,
    parse_date,
    parse_number,
    parse_time,
    unique_names,
)

DELIMITERS = (",", "\t", ";", "|")
SAMPLE_ROWS = 20
MAX_WARNINGS = 100
_TYPE_SAMPLE = 300
_HEADER_WORDS = (
    "이름", "성명", "사번", "번호", "부서", "소속", "직급", "직위", "직책", "사진", "학번", "학년", "학과",
    "전화", "연락처", "휴대폰", "이메일", "메일", "주소", "생년월일", "입사일", "발급일", "성별", "비고", "회사",
)
_HEADER_WORDS_EN = {
    "name", "id", "no", "dept", "department", "title", "position", "photo", "image", "email", "phone",
    "mobile", "address", "birthday", "date", "company", "team", "grade", "class",
}
_SYNONYMS = (
    {"이름", "성명", "name", "fullname", "성함"},
    {"사번", "사원번호", "직원번호", "employeeid", "employeeno", "empno", "empid", "사원코드"},
    {"학번", "학생번호", "studentid", "studentno"},
    {"부서", "소속", "부서명", "department", "dept", "팀", "team"},
    {"직급", "직위", "직책", "position", "title", "jobtitle", "rank"},
    {"사진", "사진파일", "이미지", "photo", "image", "picture"},
    {"전화", "전화번호", "연락처", "휴대폰", "휴대전화", "핸드폰", "phone", "mobile", "tel"},
    {"이메일", "메일", "email", "mail"},
    {"생년월일", "생일", "birthday", "birthdate", "dob"},
    {"입사일", "입사일자", "joindate", "hiredate"},
    {"회사", "회사명", "company"},
    {"번호", "no", "순번", "연번"},
)


@dataclass
class ImportPreview:
    path: str
    kind: str  # "csv" | "excel"
    encoding: str | None
    delimiter: str | None
    has_header: bool
    columns: list[Column]
    sample_rows: list[list[str]]
    total_rows: int = 0
    sheet: str | None = None
    sheets: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    blank_rows: int = 0
    rows: list[list[str]] = field(default_factory=list, repr=False)
    row_numbers: list[int] = field(default_factory=list, repr=False)


@dataclass
class ImportResult:
    inserted: int = 0
    skipped: int = 0
    warnings: list[str] = field(default_factory=list)
    created_columns: list[str] = field(default_factory=list)


# ---- 공통: 표 → 미리보기 ----


def _looks_like_data(cell: str) -> bool:
    return (
        parse_number(cell) is not None
        or has_leading_zero(cell)
        or parse_date(cell) is not None
        or parse_time(cell) is not None
    )


def _detect_header(rows: list[list[str]]) -> bool:
    first = rows[0]
    filled = [c for c in first if c]
    if not filled:
        return False
    if any(_looks_like_data(c) for c in filled):
        return False
    for cell in filled:
        key = re.sub(r"[\s_\-./()]+", "", cell).lower()
        if key in _HEADER_WORDS_EN or any(word in key for word in _HEADER_WORDS):
            return True
    if len(set(filled)) != len(filled):
        return False
    for index, cell in enumerate(first):
        if cell and any(index < len(row) and row[index] == cell for row in rows[1:]):
            return False
    return True


def _build_preview(
    numbered: list[tuple[int, list[str]]],
    has_header: bool | None,
    header: list[str] | None = None,
    **info: Any,
) -> ImportPreview:
    """numbered: (원본 파일의 행 번호, 칸 목록). header가 주어지면 numbered는 데이터 행만 담는다."""
    while numbered and not any(numbered[-1][1]):
        numbered.pop()
    while numbered and header is None and not any(numbered[0][1]):
        numbered.pop(0)
    if not numbered and header is None:
        raise DataError("파일에 데이터가 없습니다.")

    width = max([len(cells) for _, cells in numbered] + [len(header or [])])
    for _, cells in numbered:
        cells.extend([""] * (width - len(cells)))

    if header is not None:
        has_header = True
        header = list(header) + [""] * (width - len(header))
    else:
        if has_header is None:
            has_header = _detect_header([cells for _, cells in numbered])
        if has_header:
            header = numbered.pop(0)[1]
        else:
            header = [""] * width

    data = [(n, cells) for n, cells in numbered if any(cells)]
    blank_rows = len(numbered) - len(data)

    keep = [i for i in range(width) if header[i].strip() or any(cells[i] for _, cells in data)]
    if not keep:
        raise DataError("파일에 데이터가 없습니다.")
    names = unique_names([clean_column_name(header[i], i) for i in keep])
    rows = [[cells[i] for i in keep] for _, cells in data]
    columns = [
        Column(name, guess_type((row[position] for row in rows[:_TYPE_SAMPLE]), name))
        for position, name in enumerate(names)
    ]
    return ImportPreview(
        has_header=bool(has_header),
        columns=columns,
        sample_rows=[list(row) for row in rows[:SAMPLE_ROWS]],
        total_rows=len(rows),
        blank_rows=blank_rows,
        rows=rows,
        row_numbers=[n for n, _ in data],
        **info,
    )


# ---- CSV ----


def detect_encoding(raw: bytes) -> str:
    if raw.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return "utf-16"
    head = raw[:4096]
    if head and head.count(b"\x00") > len(head) // 8:
        odd = head[1::2].count(b"\x00")
        even = head[0::2].count(b"\x00")
        return "utf-16-le" if odd >= even else "utf-16-be"
    for encoding in ("utf-8", "cp949"):
        try:
            raw.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue
    return "cp949"


def detect_delimiter(text: str) -> str:
    lines = [line for line in text.splitlines() if line.strip()][:30]
    if not lines:
        return ","
    sample = "\n".join(lines)
    best, best_key = ",", (False, 1, 0.0)
    for delimiter in DELIMITERS:
        counts = [len(row) for row in csv.reader(io.StringIO(sample, newline=""), delimiter=delimiter) if row]
        if not counts:
            continue
        mode = max(set(counts), key=counts.count)
        if mode < 2:
            continue
        share = counts.count(mode) / len(counts)
        key = (share >= 0.9, mode, share)
        if key > best_key:
            best, best_key = delimiter, key
    return best


def _read_bytes(path: str | Path) -> bytes:
    try:
        return Path(path).read_bytes()
    except FileNotFoundError:
        raise DataError(f"파일을 찾을 수 없습니다: {path}") from None
    except PermissionError:
        raise DataError(f"파일을 읽을 수 없습니다. 다른 프로그램에서 열려 있는지 확인하세요: {path}") from None
    except OSError as exc:
        raise DataError(f"파일을 읽을 수 없습니다: {path}") from exc


def preview_csv(
    path: str | Path,
    encoding: str | None = None,
    delimiter: str | None = None,
    has_header: bool | None = None,
) -> ImportPreview:
    raw = _read_bytes(path)
    warnings: list[str] = []
    encoding = encoding or detect_encoding(raw)
    try:
        text = raw.decode(encoding)
    except UnicodeDecodeError:
        text = raw.decode(encoding, errors="replace")
        warnings.append("일부 글자를 읽지 못했습니다. 글자가 깨져 보이면 인코딩을 바꿔서 다시 시도하세요.")
    except LookupError:
        raise DataError(f"지원하지 않는 인코딩입니다: {encoding}") from None
    text = text.lstrip("﻿")
    if not text.strip():
        raise DataError("파일에 데이터가 없습니다.")
    delimiter = delimiter or detect_delimiter(text)
    try:
        numbered = []
        reader = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter)
        for cells in reader:
            numbered.append((reader.line_num, [cell.strip() for cell in cells]))
    except csv.Error as exc:
        raise DataError(f"CSV 파일 형식이 올바르지 않습니다: {exc}") from exc
    return _build_preview(
        numbered,
        has_header,
        path=str(path),
        kind="csv",
        encoding=encoding,
        delimiter=delimiter,
        warnings=warnings,
    )


# ---- Excel ----


def _cell_text(cell) -> str:
    value = cell.value
    if value is None:
        return ""
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, datetime):
        if value.time() == time(0):
            return value.date().isoformat()
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, time):
        return value.strftime("%H:%M:%S" if value.second else "%H:%M")
    if isinstance(value, float):
        if not value.is_integer():
            return repr(value)
        value = int(value)
    if isinstance(value, int):
        number_format = cell.number_format or ""
        if re.fullmatch(r"0{2,}", number_format):
            return str(value).zfill(len(number_format))
        return str(value)
    return str(value).strip()


def _open_workbook(path: str | Path):
    suffix = Path(path).suffix.lower()
    if suffix == ".xls":
        raise DataError("예전 형식(.xls) 파일은 열 수 없습니다. Excel에서 '.xlsx'로 다시 저장한 뒤 가져오세요.")
    if not Path(path).exists():
        raise DataError(f"파일을 찾을 수 없습니다: {path}")
    try:
        from openpyxl import load_workbook

        return load_workbook(path, data_only=True)
    except PermissionError:
        raise DataError(f"파일을 읽을 수 없습니다. 다른 프로그램에서 열려 있는지 확인하세요: {path}") from None
    except Exception as exc:
        raise DataError(f"Excel 파일을 열 수 없습니다: {path}") from exc


def _sheet_has_data(sheet) -> bool:
    return any(cell.value is not None for row in sheet.iter_rows() for cell in row)


def list_sheets(path: str | Path) -> list[str]:
    workbook = _open_workbook(path)
    try:
        return [sheet.title for sheet in workbook.worksheets]
    finally:
        workbook.close()


def preview_excel(path: str | Path, sheet: str | int | None = None, has_header: bool | None = None) -> ImportPreview:
    workbook = _open_workbook(path)
    try:
        sheets = [ws.title for ws in workbook.worksheets]
        if sheet is None:
            worksheet = next((ws for ws in workbook.worksheets if _sheet_has_data(ws)), None)
            if worksheet is None:
                raise DataError("파일에 데이터가 없습니다.")
        elif isinstance(sheet, int) and not isinstance(sheet, bool):
            if not 0 <= sheet < len(sheets):
                raise DataError(f"{sheet + 1}번째 시트가 없습니다.")
            worksheet = workbook.worksheets[sheet]
        else:
            if sheet not in sheets:
                raise DataError(f"'{sheet}' 시트를 찾을 수 없습니다.")
            worksheet = workbook[sheet]

        grid = [[_cell_text(cell) for cell in row] for row in worksheet.iter_rows()]
        merges = [(m.min_row, m.max_row, m.min_col, m.max_col) for m in worksheet.merged_cells.ranges]
        first_row = first_col = 1  # iter_rows()는 항상 A1부터 시작한다
        title = worksheet.title
    finally:
        workbook.close()

    def at(row: int, col: int) -> tuple[int, int]:
        return row - first_row, col - first_col

    for min_row, max_row, min_col, max_col in merges:
        r0, c0 = at(min_row, min_col)
        if not (0 <= r0 < len(grid) and 0 <= c0 < len(grid[r0])):
            continue
        value = grid[r0][c0]
        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                r, c = at(row, col)
                if 0 <= r < len(grid) and 0 <= c < len(grid[r]):
                    grid[r][c] = value

    numbered = [(first_row + index, cells) for index, cells in enumerate(grid)]
    while numbered and not any(numbered[-1][1]):
        numbered.pop()
    while numbered and not any(numbered[0][1]):
        numbered.pop(0)
    if not numbered:
        raise DataError(f"'{title}' 시트에 데이터가 없습니다.")

    # 표 위의 제목 줄(값이 한 가지뿐인 줄)은 건너뛴다.
    table_width = max(sum(1 for cell in cells if cell) for _, cells in numbered)
    while len(numbered) > 1 and table_width > 1 and len({c for c in numbered[0][1] if c}) == 1:
        numbered.pop(0)
        while numbered and not any(numbered[0][1]):
            numbered.pop(0)

    header = None
    header_row = numbered[0][0]
    depth = 1
    for min_row, max_row, min_col, max_col in merges:
        if min_row != header_row:
            continue
        if max_row > min_row:
            depth = max(depth, max_row - min_row + 1)
        elif max_col > min_col:
            depth = max(depth, 2)
    depth = min(depth, len(numbered))
    if depth > 1 and has_header is not False:
        head_rows = [cells for _, cells in numbered[:depth]]
        width = max(len(cells) for cells in head_rows)
        parts_by_col = []
        for col in range(width):
            parts: list[str] = []
            for cells in head_rows:
                value = cells[col] if col < len(cells) else ""
                if value and value not in parts:
                    parts.append(value)
            parts_by_col.append(parts)
        leaves = [parts[-1] if parts else "" for parts in parts_by_col]
        filled = [leaf for leaf in leaves if leaf]
        if len(set(filled)) == len(filled):
            header = leaves
        else:
            header = [" ".join(parts) for parts in parts_by_col]
        numbered = numbered[depth:]

    return _build_preview(
        numbered,
        has_header,
        header,
        path=str(path),
        kind="excel",
        encoding=None,
        delimiter=None,
        sheet=title,
        sheets=sheets,
    )


def preview_file(path: str | Path, **options: Any) -> ImportPreview:
    """확장자를 보고 CSV 또는 Excel 미리보기를 고른다."""
    suffix = Path(path).suffix.lower()
    if suffix in (".xlsx", ".xlsm", ".xltx", ".xltm", ".xls"):
        return preview_excel(path, **options)
    if suffix in (".csv", ".txt", ".tsv", ".tab", ""):
        return preview_csv(path, **options)
    raise DataError(f"지원하지 않는 파일 형식입니다: {suffix} (CSV 또는 Excel(.xlsx) 파일을 선택하세요.)")


# ---- 열 연결 ----


def _key(name: str) -> str:
    return re.sub(r"[\s_\-./()\[\]:·]+", "", name).lower()


def _similarity(source: str, target: str) -> float:
    a, b = _key(source), _key(target)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if any(a in group and b in group for group in _SYNONYMS):
        return 0.95
    return difflib.SequenceMatcher(None, a, b).ratio()


def suggest_column_map(store: DataStore, preview: ImportPreview) -> dict[str, str | None]:
    """파일의 열 → 저장소 열. 비슷한 이름의 기존 열이 있으면 그 열에, 없으면 같은 이름의 새 열에 연결한다."""
    existing = store.column_names()
    candidates = sorted(
        (
            (_similarity(column.name, target), -position, column.name, target)
            for position, column in enumerate(preview.columns)
            for target in existing
        ),
        reverse=True,
    )
    matched: dict[str, str] = {}
    used: set[str] = set()
    for score, _, source, target in candidates:
        if score < 0.8:
            break
        if source in matched or target in used:
            continue
        matched[source] = target
        used.add(target)

    result: dict[str, str | None] = {}
    for column in preview.columns:
        if column.name in matched:
            result[column.name] = matched[column.name]
            continue
        name, suffix = column.name, 1
        while name in used:
            suffix += 1
            name = f"{column.name} ({suffix})"
        used.add(name)
        result[column.name] = name
    return result


# ---- 가져오기 ----


def import_into(
    store: DataStore,
    preview: ImportPreview,
    column_map: dict[str, str | None] | None = None,
    mode: str = "append",
) -> ImportResult:
    """column_map: {파일 열 이름: 저장소 열 이름 또는 None(가져오지 않음)}. 없는 저장소 열은 새로 만든다.
    mode="replace"는 기존 레코드를 모두 지우고 넣는다. 도중에 실패하면 아무것도 바뀌지 않는다."""
    if mode not in ("append", "replace"):
        raise DataError(f"가져오기 방식은 append 또는 replace여야 합니다: {mode}")
    if column_map is None:
        column_map = suggest_column_map(store, preview)

    positions = {column.name: index for index, column in enumerate(preview.columns)}
    pairs: list[tuple[int, str, Column]] = []
    targets: set[str] = set()
    for source, target in column_map.items():
        if source not in positions:
            raise DataError(f"가져올 파일에 '{source}' 열이 없습니다.")
        if target is None or str(target).strip() == "":
            continue
        target = str(target).strip()
        if target in targets:
            raise DataError(f"'{target}' 열에 파일의 열이 두 개 이상 연결되어 있습니다.")
        targets.add(target)
        pairs.append((positions[source], target, preview.columns[positions[source]]))
    if not pairs:
        raise DataError("가져올 열이 없습니다. 하나 이상의 열을 연결하세요.")
    pairs.sort(key=lambda pair: pair[0])

    result = ImportResult(skipped=preview.blank_rows)
    hidden = 0

    def warn(message: str) -> None:
        nonlocal hidden
        if len(result.warnings) < MAX_WARNINGS:
            result.warnings.append(message)
        else:
            hidden += 1

    with store.transaction():
        if mode == "replace":
            store.clear()
        for _, target, source_column in pairs:
            if not store.has_column(target):
                store.add_column(target, source_column.type, source_column.options)
                result.created_columns.append(target)
        types = {column.name: column.type for column in store.columns()}

        kept_numbers: list[int] = []

        def generate():
            for number, cells in zip(preview.row_numbers, preview.rows):
                values = {target: cells[index] for index, target, _ in pairs}
                if not any(values.values()):
                    result.skipped += 1
                    warn(f"{number}행: 가져올 값이 없어 건너뛰었습니다.")
                    continue
                kept_numbers.append(number)
                yield values

        def on_invalid(index: int, column: str, value: Any) -> None:
            label = TYPE_LABELS.get(types.get(column, "text"), "값")
            warn(
                f"{kept_numbers[index]}행: '{column}' 열의 값 '{value}'을(를) {label}(으)로 인식하지 못해 "
                "입력된 글자 그대로 저장했습니다."
            )

        result.inserted = store.insert_many(generate(), on_invalid=on_invalid)

    if hidden:
        result.warnings.append(f"그 밖에 {hidden}건의 알림이 더 있습니다.")
    return result
