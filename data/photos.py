from __future__ import annotations

import os
import unicodedata
from pathlib import Path
from typing import Any, Iterable

from .store import DataStore, PhotoAdjustment, Record
from .values import IMAGE_EXTENSIONS, is_empty

DEFAULT_EXTENSIONS = IMAGE_EXTENSIONS


def _norm(name: str) -> str:
    return unicodedata.normalize("NFC", name).lower()


def _extensions(extensions: Iterable[str] | None) -> tuple[str, ...]:
    result = []
    for ext in extensions or DEFAULT_EXTENSIONS:
        ext = str(ext).strip().lower()
        if not ext:
            continue
        result.append(ext if ext.startswith(".") else "." + ext)
    return tuple(dict.fromkeys(result)) or DEFAULT_EXTENSIONS


class PhotoLocator:
    """폴더 목록을 한 번만 읽어 두고 여러 레코드의 사진을 찾는다. 폴더 내용이 바뀌면 refresh()."""

    def __init__(self, folder: str | Path | None, extensions: Iterable[str] | None = None):
        self.folder = Path(folder) if folder else None
        self.extensions = _extensions(extensions)
        self._listings: dict[str, dict[str, Path]] = {}

    def refresh(self) -> None:
        self._listings.clear()

    def _listing(self, directory: Path) -> dict[str, Path]:
        key = str(directory)
        if key not in self._listings:
            files: dict[str, Path] = {}
            try:
                with os.scandir(directory) as entries:
                    for entry in entries:
                        if entry.is_file():
                            files.setdefault(_norm(entry.name), Path(entry.path))
            except OSError:
                pass
            self._listings[key] = files
        return self._listings[key]

    def find(self, value: Any) -> Path | None:
        if is_empty(value):
            return None
        text = str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)
        text = text.strip().strip('"').strip()
        if not text:
            return None
        path = Path(text)
        if not path.is_absolute():
            if self.folder is None:
                return None
            path = self.folder / path
        listing = self._listing(path.parent)
        name = _norm(path.name)
        if name in listing:
            return listing[name]
        for ext in self.extensions:
            if name + ext in listing:
                return listing[name + ext]
        stem, suffix = os.path.splitext(name)
        if suffix in self.extensions or suffix in DEFAULT_EXTENSIONS:
            for ext in self.extensions:
                if stem + ext in listing:
                    return listing[stem + ext]
        return None


def find_photo(value: Any, folder: str | Path | None, extensions: Iterable[str] | None = None) -> Path | None:
    return PhotoLocator(folder, extensions).find(value)


def set_photo_folder(
    store: DataStore, column: str, folder: str | Path, extensions: Iterable[str] | None = None
) -> None:
    options = dict(store.column(column).options)
    options["folder"] = str(folder)
    if extensions is not None:
        options["extensions"] = list(_extensions(extensions))
    store.set_column_options(column, options)


def locator_for(store: DataStore, column: str) -> PhotoLocator:
    """열 options의 folder/extensions로 탐색기를 만든다. 상대 폴더는 데이터 파일 위치 기준."""
    options = store.column(column).options
    folder = options.get("folder")
    if folder and not Path(folder).is_absolute() and store.path != ":memory:":
        folder = Path(store.path).resolve().parent / folder
    return PhotoLocator(folder, options.get("extensions"))


def photo_path(store: DataStore, record: Record, column: str, locator: PhotoLocator | None = None) -> Path | None:
    return (locator or locator_for(store, column)).find(record.get(column))


def find_missing(store: DataStore, column: str, filter: Any = None, include_empty: bool = True) -> list[Record]:
    """사진 파일을 찾지 못한 레코드. include_empty=False면 값이 비어 있는 레코드는 뺀다."""
    locator = locator_for(store, column)
    missing = []
    for record in store.rows(filter):
        value = record.get(column)
        if is_empty(value):
            if include_empty:
                missing.append(record)
        elif locator.find(value) is None:
            missing.append(record)
    return missing


class PhotoResolver:
    """인쇄 반복문에서 쓴다. 사진 열마다 폴더 목록을 한 번만 읽는다."""

    def __init__(self, store: DataStore):
        self.store = store
        self._locators = {c.name: locator_for(store, c.name) for c in store.columns() if c.type == "photo"}

    def path(self, record: Record, column: str) -> Path | None:
        locator = self._locators.get(column)
        if locator is None:
            locator = self._locators[column] = locator_for(self.store, column)
        return locator.find(record.get(column))

    def binding_record(self, record: Record) -> dict[str, Any]:
        """design.binding.resolve에 넘길 dict. 시스템 키를 빼고, 빈 값은 "", 사진 열은 실제 파일 경로("" = 없음)."""
        result: dict[str, Any] = {}
        for key, value in record.items():
            if key.startswith("_"):
                continue
            if key in self._locators:
                found = self._locators[key].find(value)
                result[key] = str(found) if found else ""
            else:
                result[key] = "" if value is None else value
        return result


def binding_record(store: DataStore, record: Record) -> dict[str, Any]:
    return PhotoResolver(store).binding_record(record)


def get_adjustment(
    store: DataStore, record_id: int, column: str, default: PhotoAdjustment | None = PhotoAdjustment()
) -> PhotoAdjustment | None:
    """저장된 사진 조정값. 없으면 default(None을 주면 "저장된 값 없음"을 구별할 수 있다)."""
    return store.get_photo_adjustment(record_id, column, default)


def has_adjustment(store: DataStore, record_id: int, column: str) -> bool:
    return store.has_photo_adjustment(record_id, column)


def set_adjustment(store: DataStore, record_id: int, column: str, adjustment: PhotoAdjustment) -> None:
    store.set_photo_adjustment(record_id, column, adjustment)


def apply_adjustment_to_all(
    store: DataStore, column: str, adjustment: PhotoAdjustment, filter: Any = None
) -> int:
    return store.apply_photo_adjustment_to_all(column, adjustment, filter)


def clear_adjustments(store: DataStore, column: str, ids: Iterable[int] | None = None) -> int:
    return store.clear_photo_adjustments(column, ids)
