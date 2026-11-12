"""화면들이 함께 쓰는 작업 상태: 디자인(편집기), 명단, 지금 보고 있는 사람."""
from __future__ import annotations

from typing import Any

from PySide6.QtCore import QObject, Signal

from data.photos import PhotoResolver
from data.store import DataStore
from design.binding import BindingContext
from ui.editor import EditorController


class Workspace(QObject):
    store_changed = Signal()            # 명단 파일이 열리거나 닫힘
    columns_changed = Signal()          # 열 추가·삭제·이름 변경
    records_changed = Signal()          # 행 추가·수정·삭제·인쇄 상태 변경
    current_record_changed = Signal(object)   # Record dict | None
    checked_ids_changed = Signal(list)  # 명단에서 체크한 사람들의 _id

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self.editor = EditorController(self)
        self._store: DataStore | None = None
        self._resolver: PhotoResolver | None = None
        self._current_id: int | None = None
        self._checked_ids: list[int] = []
        self.view_filter: Any = None    # 명단 화면의 현재 필터/정렬 (인쇄 대상 계산에 사용)
        self.view_order: Any = None

    # ---- 명단 ----
    @property
    def store(self) -> DataStore | None:
        return self._store

    def set_store(self, store: DataStore | None) -> None:
        if self._store is not None and self._store is not store:
            self._store.close()
        self._store = store
        self._resolver = PhotoResolver(store) if store is not None else None
        self._current_id = None
        self._checked_ids = []
        self.store_changed.emit()
        self.columns_changed.emit()
        self.records_changed.emit()
        self.checked_ids_changed.emit([])
        self._push_preview()

    def column_names(self) -> list[str]:
        return self._store.column_names() if self._store is not None else []

    def notify_columns_changed(self) -> None:
        if self._resolver is not None and self._store is not None:
            self._resolver = PhotoResolver(self._store)
        self.columns_changed.emit()
        self._push_preview()

    def notify_records_changed(self) -> None:
        self.records_changed.emit()
        self._push_preview()

    # ---- 지금 보고 있는 사람 ----
    @property
    def current_record_id(self) -> int | None:
        return self._current_id

    def current_record(self) -> dict | None:
        if self._store is None or self._current_id is None:
            return None
        return self._store.get(self._current_id)

    def set_current_record(self, record_id: int | None) -> None:
        if record_id == self._current_id:
            return
        self._current_id = record_id
        self._push_preview()

    def binding_record(self, record: dict | None) -> dict | None:
        if record is None or self._resolver is None:
            return None
        return self._resolver.binding_record(record)

    def _push_preview(self) -> None:
        record = self.current_record()
        if record is None:
            self.editor.set_preview_record(None)
        else:
            context = BindingContext(record_key=str(record["_id"]))
            self.editor.set_preview_record(self.binding_record(record), context)
        self.current_record_changed.emit(record)

    # ---- 체크한 사람들 ----
    @property
    def checked_ids(self) -> list[int]:
        return list(self._checked_ids)

    def set_checked_ids(self, ids: list[int]) -> None:
        ids = list(ids)
        if ids != self._checked_ids:
            self._checked_ids = ids
            self.checked_ids_changed.emit(list(ids))

    def close(self) -> None:
        if self._store is not None:
            self._store.close()
            self._store = None
        self.editor.close()
