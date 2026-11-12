"""명단 표의 모델. 보이는 만큼만 store.rows()로 읽어 온다."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, Signal

from data.errors import DataError
from data.filters import Filter
from data.photos import PhotoLocator, locator_for
from data.store import Column, DataStore
from data.values import is_empty

from .text import display_value, status_text

PAGE_SIZE = 200
MAX_PAGES = 12
SCAN_CHUNK = 2000

KIND_CHECK = "check"
KIND_FIELD = "field"
KIND_STATUS = "status"

RecordRole = Qt.UserRole + 1
SpecRole = Qt.UserRole + 2
PhotoPathRole = Qt.UserRole + 3
StatusRole = Qt.UserRole + 4

HIDDEN_OPTION = "hidden"


@dataclass(frozen=True)
class IdsFilter(Filter):
    """고른 사람들만(또는 그 사람들만 빼고)."""

    ids: frozenset
    negate: bool = False

    def to_sql(self, ctx) -> tuple[str, list]:
        expr = ctx.sql_column("_id")[0]
        if not self.ids:
            return ("1 = 1" if self.negate else "0 = 1"), []
        listed = ", ".join(str(int(i)) for i in sorted(self.ids))
        return f"{expr} {'NOT ' if self.negate else ''}IN ({listed})", []


@dataclass(frozen=True)
class ColumnSpec:
    kind: str
    column: Column | None = None

    @property
    def name(self) -> str:
        return self.column.name if self.column is not None else ""

    @property
    def type(self) -> str:
        return self.column.type if self.column is not None else ""

    @property
    def is_photo(self) -> bool:
        return self.kind == KIND_FIELD and self.type == "photo"


class PhotoIndex:
    """사진 항목마다 폴더 목록을 한 번 읽어 두고, 사진이 없는 사람을 가려낸다."""

    def __init__(self) -> None:
        self._store: DataStore | None = None
        self._columns: list[str] = []
        self._locators: dict[str, PhotoLocator] = {}
        self._missing: set[int] | None = None

    def configure(self, store: DataStore | None, design_columns: Iterable[str] = ()) -> None:
        self._store = store
        self._locators = {}
        self._missing = None
        self._columns = []
        if store is None:
            return
        photo_columns = [c.name for c in store.columns() if c.type == "photo"]
        used = [name for name in photo_columns if name in set(design_columns)]
        self._columns = used or photo_columns
        self._locators = {name: locator_for(store, name) for name in photo_columns}

    @property
    def columns(self) -> list[str]:
        """사진이 없으면 문제가 되는 항목들."""
        return list(self._columns)

    @property
    def primary(self) -> str | None:
        return self._columns[0] if self._columns else None

    def path(self, record: dict, column: str) -> Path | None:
        locator = self._locators.get(column)
        return locator.find(record.get(column)) if locator is not None else None

    def folder(self, column: str) -> Path | None:
        locator = self._locators.get(column)
        return locator.folder if locator is not None else None

    def is_missing(self, record: dict) -> bool:
        return any(self.path(record, name) is None for name in self._columns)

    def invalidate(self) -> None:
        self._missing = None

    def refresh_folders(self) -> None:
        for locator in self._locators.values():
            locator.refresh()
        self._missing = None

    def update_record(self, record: dict) -> None:
        if self._missing is None:
            return
        if self.is_missing(record):
            self._missing.add(record["_id"])
        else:
            self._missing.discard(record["_id"])

    def missing_ids(self) -> set[int]:
        if self._missing is None:
            found: set[int] = set()
            if self._store is not None and self._columns:
                offset = 0
                while True:
                    chunk = self._store.rows(None, None, offset, SCAN_CHUNK)
                    found.update(r["_id"] for r in chunk if self.is_missing(r))
                    if len(chunk) < SCAN_CHUNK:
                        break
                    offset += SCAN_CHUNK
            self._missing = found
        return self._missing


class PeopleTableModel(QAbstractTableModel):
    checkedChanged = Signal()
    orderChanged = Signal()
    recordEdited = Signal(int, str)     # 사람 id, 항목 이름
    editFailed = Signal(str)

    def __init__(self, photos: PhotoIndex | None = None, parent=None):
        super().__init__(parent)
        self.photos = photos or PhotoIndex()
        self._store: DataStore | None = None
        self._filter: Any = None
        self._order: list[tuple[str, str]] | None = None
        self._specs: list[ColumnSpec] = [ColumnSpec(KIND_CHECK), ColumnSpec(KIND_STATUS)]
        self._count = 0
        self._pages: OrderedDict[int, list[dict]] = OrderedDict()
        self._ids: list[int] | None = None
        self._checked: set[int] = set()
        self._frozen = False
        self.fetched_rows = 0
        self.fetch_calls = 0

    # ---- 구성 ----
    @property
    def store(self) -> DataStore | None:
        return self._store

    @property
    def specs(self) -> list[ColumnSpec]:
        return list(self._specs)

    @property
    def view_filter(self) -> Any:
        return self._filter

    @property
    def view_order(self) -> list[tuple[str, str]] | None:
        return list(self._order) if self._order else None

    def set_store(self, store: DataStore | None) -> None:
        self.beginResetModel()
        self._store = store
        self._filter = None
        self._order = None
        self._checked = set()
        self._rebuild_specs()
        self._reload()
        self.endResetModel()
        self.checkedChanged.emit()

    def rebuild_columns(self) -> None:
        self.beginResetModel()
        self._rebuild_specs()
        if self._order and self._store is not None:
            names = set(self._store.column_names())
            self._order = [(n, d) for n, d in self._order if n in names or n.startswith("_")] or None
        self._reload()
        self.endResetModel()

    def _rebuild_specs(self) -> None:
        specs = [ColumnSpec(KIND_CHECK)]
        if self._store is not None:
            columns = [c for c in self._store.columns() if not c.options.get(HIDDEN_OPTION)]
            primary = self.photos.primary
            columns.sort(key=lambda c: 0 if c.name == primary else 1)
            specs += [ColumnSpec(KIND_FIELD, c) for c in columns]
        specs.append(ColumnSpec(KIND_STATUS))
        self._specs = specs

    def hidden_columns(self) -> list[str]:
        if self._store is None:
            return []
        return [c.name for c in self._store.columns() if c.options.get(HIDDEN_OPTION)]

    def set_view(self, filter: Any, order: list[tuple[str, str]] | None) -> None:
        self.beginResetModel()
        self._filter = filter
        self._order = list(order) if order else None
        self._reload()
        self.endResetModel()

    def refresh(self) -> None:
        self.beginResetModel()
        self._reload()
        self.endResetModel()

    def set_frozen(self, frozen: bool) -> None:
        """가져오기처럼 다른 스레드가 명단을 쓰는 동안에는 새로 읽지 않는다."""
        self._frozen = frozen

    def _reload(self) -> None:
        self._pages.clear()
        self._ids = None
        self._count = 0
        if self._store is not None and not self._frozen:
            try:
                self._count = self._store.count(self._filter)
            except DataError as exc:
                self._filter = None
                self._count = self._store.count(None)
                self.editFailed.emit(str(exc))

    # ---- 읽기 ----
    def _page(self, number: int) -> list[dict]:
        page = self._pages.get(number)
        if page is not None:
            self._pages.move_to_end(number)
            return page
        if self._store is None or self._frozen:
            return []
        page = self._store.rows(self._filter, self._order, number * PAGE_SIZE, PAGE_SIZE)
        self.fetch_calls += 1
        self.fetched_rows += len(page)
        self._pages[number] = page
        while len(self._pages) > MAX_PAGES:
            self._pages.popitem(last=False)
        return page

    def record_at(self, row: int) -> dict | None:
        if not 0 <= row < self._count:
            return None
        page = self._page(row // PAGE_SIZE)
        offset = row % PAGE_SIZE
        return page[offset] if offset < len(page) else None

    def id_at(self, row: int) -> int | None:
        record = self.record_at(row)
        return record["_id"] if record is not None else None

    def ids(self) -> list[int]:
        """지금 조건·순서대로의 id 전부. 번호만 읽으므로 가볍다."""
        if self._ids is None:
            if self._store is None or self._frozen:
                return []
            self._ids = self._store.ids(self._filter, self._order)
        return self._ids

    def row_of(self, record_id: int | None) -> int:
        if record_id is None:
            return -1
        for number, page in self._pages.items():
            for offset, record in enumerate(page):
                if record["_id"] == record_id:
                    return number * PAGE_SIZE + offset
        try:
            return self.ids().index(record_id)
        except ValueError:
            return -1

    def column_of(self, name: str) -> int:
        for index, spec in enumerate(self._specs):
            if spec.kind == KIND_FIELD and spec.name == name:
                return index
        return -1

    # ---- Qt 모델 ----
    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else self._count

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._specs)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole):
        if orientation != Qt.Horizontal or not 0 <= section < len(self._specs):
            return None
        spec = self._specs[section]
        if role == Qt.DisplayRole:
            if spec.kind == KIND_CHECK:
                return ""
            return "인쇄" if spec.kind == KIND_STATUS else spec.name
        if role == SpecRole:
            return spec
        if role == Qt.AccessibleTextRole and spec.kind == KIND_CHECK:
            return "보이는 사람 모두 선택"
        return None

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):
        if not index.isValid():
            return None
        spec = self._specs[index.column()]
        if role == SpecRole:
            return spec
        record = self.record_at(index.row())
        if record is None:
            return None
        if role == RecordRole:
            return record
        if spec.kind == KIND_CHECK:
            if role == Qt.CheckStateRole:
                return Qt.Checked if record["_id"] in self._checked else Qt.Unchecked
            if role == Qt.AccessibleTextRole:
                return "인쇄할 사람으로 선택"
            return None
        if spec.kind == KIND_STATUS:
            if role in (Qt.DisplayRole, Qt.ToolTipRole):
                return status_text(record, self.photos.is_missing(record))[0]
            if role == StatusRole:
                return status_text(record, self.photos.is_missing(record))
            return None
        value = record.get(spec.name)
        if role == Qt.EditRole:
            return value
        if role == Qt.DisplayRole:
            return display_value(value)
        if spec.is_photo:
            if role == PhotoPathRole:
                found = self.photos.path(record, spec.name)
                return str(found) if found else ""
            if role == Qt.ToolTipRole:
                if is_empty(value):
                    return "사진이 없습니다. 눌러서 사진 파일을 고르세요."
                if self.photos.path(record, spec.name) is None:
                    return f"'{display_value(value)}' 사진 파일을 찾지 못했습니다. 눌러서 직접 고르세요."
                return display_value(value)
        return None

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        if not index.isValid():
            return Qt.NoItemFlags
        spec = self._specs[index.column()]
        flags = Qt.ItemIsEnabled | Qt.ItemIsSelectable
        if spec.kind == KIND_CHECK:
            flags |= Qt.ItemIsUserCheckable
        elif spec.kind == KIND_FIELD:
            flags |= Qt.ItemIsEditable
        return flags

    def setData(self, index: QModelIndex, value: Any, role: int = Qt.EditRole) -> bool:
        if not index.isValid() or self._store is None:
            return False
        spec = self._specs[index.column()]
        record = self.record_at(index.row())
        if record is None:
            return False
        if spec.kind == KIND_CHECK and role == Qt.CheckStateRole:
            checked = value in (Qt.Checked, Qt.CheckState.Checked, 2, True)
            self.set_checked([record["_id"]], checked)
            return True
        if spec.kind != KIND_FIELD or role != Qt.EditRole:
            return False
        if isinstance(value, str):
            value = value.strip()
        old = record.get(spec.name)
        if display_value(old) == display_value(value):
            return False
        try:
            self._store.update(record["_id"], {spec.name: value})
            fresh = self._store.get(record["_id"])
        except DataError as exc:
            self.editFailed.emit(str(exc))
            return False
        if fresh is not None:
            record.clear()
            record.update(fresh)
            self.photos.update_record(record)
        self.dataChanged.emit(self.index(index.row(), 0), self.index(index.row(), len(self._specs) - 1))
        self.recordEdited.emit(record["_id"], spec.name)
        return True

    def sort(self, column: int, order: Qt.SortOrder = Qt.AscendingOrder) -> None:
        if not 0 <= column < len(self._specs):
            return
        spec = self._specs[column]
        if spec.kind == KIND_CHECK:
            return
        direction = "asc" if order == Qt.AscendingOrder else "desc"
        if spec.kind == KIND_STATUS:
            self.set_order([("_printed", direction), ("_printed_at", direction)])
        else:
            self.set_order([(spec.name, direction)])

    def set_order(self, order: list[tuple[str, str]] | None) -> None:
        order = list(order) if order else None
        if order == self._order:
            return
        self.set_view(self._filter, order)
        self.orderChanged.emit()

    def sort_state(self) -> tuple[int, str] | None:
        """(열 번호, "asc"|"desc")."""
        if not self._order:
            return None
        name, direction = self._order[0]
        if name.startswith("_"):
            return len(self._specs) - 1, direction
        column = self.column_of(name)
        return (column, direction) if column >= 0 else None

    # ---- 체크 ----
    def checked_ids(self) -> set[int]:
        return set(self._checked)

    def is_checked(self, record_id: int) -> bool:
        return record_id in self._checked

    def set_checked(self, ids: Iterable[int], checked: bool = True) -> None:
        ids = set(ids)
        before = len(self._checked)
        if checked:
            changed = not ids <= self._checked
            self._checked |= ids
        else:
            changed = bool(ids & self._checked)
            self._checked -= ids
        if changed or before != len(self._checked):
            self._emit_check_column()
            self.checkedChanged.emit()

    def replace_checked(self, ids: Iterable[int]) -> None:
        ids = set(ids)
        if ids != self._checked:
            self._checked = ids
            self._emit_check_column()
            self.checkedChanged.emit()

    def toggle(self, record_id: int) -> None:
        self.set_checked([record_id], record_id not in self._checked)

    def _emit_check_column(self) -> None:
        if self._count:
            self.dataChanged.emit(self.index(0, 0), self.index(self._count - 1, 0), [Qt.CheckStateRole])

    def checked_in_view(self) -> int:
        if not self._checked:
            return 0
        if self._filter is None:
            return len(self._checked)
        return len(self._checked.intersection(self.ids()))

    def header_check_state(self) -> Qt.CheckState:
        inside = self.checked_in_view()
        if inside == 0 or self._count == 0:
            return Qt.Unchecked
        return Qt.Checked if inside >= self._count else Qt.PartiallyChecked

    def toggle_all_in_view(self) -> bool:
        """지금 조건에 맞는 사람을 모두 체크한다. 이미 모두 체크돼 있으면 푼다. 체크했으면 True."""
        ids = self.ids()
        if self.header_check_state() == Qt.Checked:
            self.set_checked(ids, False)
            return False
        self.set_checked(ids, True)
        return True
