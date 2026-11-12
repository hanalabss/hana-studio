"""프로젝트(.hsproj): 디자인(.hsd) · 명단(.hsdata) · 인쇄 옵션을 한데 묶는 작은 JSON 파일.

경로는 프로젝트 파일 기준 상대 경로로 적어, 폴더째 옮겨도 그대로 열린다.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from data.store import DataStore
from design import storage
from design.model import Document
from printer.base import PrinterDevice
from services.printers import app_data_dir, device_from_dict, device_to_dict
from services.render_job import PrintOptions

log = logging.getLogger(__name__)

FORMAT_NAME = "hana-studio-project"
CURRENT_VERSION = 1
EXTENSION = ".hsproj"
AUTOSAVE_SUFFIX = ".autosave"
RECENT_FILE = "recent_projects.json"
AUTOSAVE_DIR = "autosave"
RECENT_LIMIT = 12


class ProjectError(Exception):
    """사용자에게 그대로 보여줄 수 있는 한국어 메시지를 담는다."""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _write_json(path: str, payload: dict[str, Any]) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    temporary = f"{path}.{uuid.uuid4().hex[:8]}.tmp"
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


# ---------------------------------------------------------------- 최근 프로젝트

@dataclass
class RecentEntry:
    path: str
    name: str = ""
    opened_at: str = ""

    @property
    def exists(self) -> bool:
        return os.path.isfile(self.path)


class RecentProjects:
    """최근에 연 프로젝트 목록. 기본 위치는 %LOCALAPPDATA%\\HanaStudio\\recent_projects.json."""

    def __init__(self, path: str | None = None, limit: int = RECENT_LIMIT) -> None:
        self.path = (path or os.environ.get("HANA_STUDIO_RECENT_FILE")
                     or os.path.join(app_data_dir(), RECENT_FILE))
        self.limit = limit

    def entries(self, existing_only: bool = False) -> list[RecentEntry]:
        """최근 것부터. existing_only면 지워졌거나 옮겨진 파일은 뺀다."""
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                raw = json.load(handle).get("projects", [])
        except (OSError, ValueError, AttributeError):
            return []
        entries = []
        for item in raw if isinstance(raw, list) else []:
            if isinstance(item, dict) and isinstance(item.get("path"), str):
                entries.append(RecentEntry(item["path"], str(item.get("name") or ""),
                                           str(item.get("opened_at") or "")))
        return [e for e in entries if e.exists] if existing_only else entries

    def _save(self, entries: list[RecentEntry]) -> None:
        payload = {"version": 1, "projects": [
            {"path": e.path, "name": e.name, "opened_at": e.opened_at} for e in entries[: self.limit]
        ]}
        try:
            _write_json(self.path, payload)
        except OSError as exc:
            log.warning("최근 프로젝트 목록을 저장하지 못했습니다: %s", exc)

    @staticmethod
    def _key(path: str) -> str:
        return os.path.normcase(os.path.abspath(path))

    def add(self, path: str, name: str = "") -> None:
        path = os.path.abspath(path)
        entries = [e for e in self.entries() if self._key(e.path) != self._key(path)]
        entries.insert(0, RecentEntry(path, name or os.path.splitext(os.path.basename(path))[0], _now()))
        self._save(entries)

    def remove(self, path: str) -> None:
        self._save([e for e in self.entries() if self._key(e.path) != self._key(path)])

    def clear(self) -> None:
        self._save([])


# ---------------------------------------------------------------- 프로젝트

def _to_relative(target: str | None, base_dir: str | None) -> str | None:
    if not target:
        return None
    target = os.path.abspath(target)
    if base_dir:
        try:
            return os.path.relpath(target, base_dir).replace(os.sep, "/")
        except ValueError:
            pass   # 드라이브가 다르면 상대 경로를 만들 수 없다
    return target


def _to_absolute(stored: str | None, base_dir: str) -> str | None:
    if not stored:
        return None
    if os.path.isabs(stored):
        if os.path.exists(stored):
            return stored
        beside = os.path.join(base_dir, os.path.basename(stored))
        return beside if os.path.exists(beside) else stored
    return os.path.normpath(os.path.join(base_dir, stored))


@dataclass
class Project:
    """design_path / data_path는 절대 경로로 들고 있다가 저장할 때 상대 경로로 바꾼다."""

    path: str | None = None
    name: str = "새 프로젝트"
    design_path: str | None = None
    data_path: str | None = None
    options: PrintOptions = field(default_factory=PrintOptions)
    printer: PrinterDevice | None = None
    version: int = CURRENT_VERSION
    extra: dict[str, Any] = field(default_factory=dict, repr=False)
    _draft_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12], repr=False)

    # -- 만들기 · 열기 · 저장 ------------------------------------------------
    @classmethod
    def new(
        cls, path: str | os.PathLike | None = None, *, name: str | None = None,
        design_path: str | os.PathLike | None = None, data_path: str | os.PathLike | None = None,
        options: PrintOptions | None = None, remember: bool = True,
    ) -> "Project":
        """새 프로젝트. path를 주면 바로 저장한다(확장자가 없으면 .hsproj를 붙인다)."""
        project = cls(
            name=name or (os.path.splitext(os.path.basename(os.fspath(path)))[0] if path else "새 프로젝트"),
            design_path=os.path.abspath(os.fspath(design_path)) if design_path else None,
            data_path=os.path.abspath(os.fspath(data_path)) if data_path else None,
            options=options or PrintOptions(),
        )
        if path:
            project.save(path, remember=remember)
        return project

    @classmethod
    def open(cls, path: str | os.PathLike, *, remember: bool = True) -> "Project":
        source = os.path.abspath(os.fspath(path))
        if not os.path.isfile(source):
            raise ProjectError(f"프로젝트 파일을 찾을 수 없습니다: {source}")
        try:
            with open(source, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError) as exc:
            raise ProjectError(f"프로젝트 파일을 읽을 수 없습니다: {source}") from exc
        if not isinstance(data, dict) or data.get("format") != FORMAT_NAME:
            raise ProjectError("하나스튜디오 프로젝트 파일이 아닙니다.")
        version = data.get("version", 1)
        if isinstance(version, bool) or not isinstance(version, int) or version < 1:
            raise ProjectError(f"프로젝트 버전이 올바르지 않습니다: {version!r}")
        base_dir = os.path.dirname(source)
        known = {"format", "version", "name", "design", "data", "print_options", "printer", "saved_at"}
        project = cls(
            path=source,
            name=str(data.get("name") or os.path.splitext(os.path.basename(source))[0]),
            design_path=_to_absolute(data.get("design"), base_dir),
            data_path=_to_absolute(data.get("data"), base_dir),
            options=PrintOptions.from_dict(data.get("print_options")),
            printer=device_from_dict(data.get("printer")),
            version=max(version, CURRENT_VERSION),
            extra={k: copy.deepcopy(v) for k, v in data.items() if k not in known},
        )
        if remember:
            RecentProjects().add(source, project.name)
        return project

    def to_dict(self) -> dict[str, Any]:
        base_dir = os.path.dirname(self.path) if self.path else None
        payload: dict[str, Any] = {
            "format": FORMAT_NAME,
            "version": self.version,
            "name": self.name,
            "design": _to_relative(self.design_path, base_dir),
            "data": _to_relative(self.data_path, base_dir),
            "print_options": self.options.to_dict(),
            "printer": device_to_dict(self.printer) if self.printer else None,
            "saved_at": _now(),
        }
        for key, value in self.extra.items():
            payload.setdefault(key, copy.deepcopy(value))
        return payload

    def save(self, path: str | os.PathLike | None = None, *, remember: bool = True) -> str:
        target = os.fspath(path) if path else self.path
        if not target:
            raise ProjectError("프로젝트를 저장할 위치를 정하세요.")
        target = os.path.abspath(target)
        if not target.lower().endswith(EXTENSION):
            target += EXTENSION
        previous_autosave = self.autosave_path
        self.path = target
        try:
            _write_json(target, self.to_dict())
        except OSError as exc:
            raise ProjectError(f"프로젝트를 저장하지 못했습니다: {exc}") from exc
        if previous_autosave != self.autosave_path and os.path.isfile(previous_autosave):
            try:
                os.replace(previous_autosave, self.autosave_path)
            except OSError as exc:
                log.warning("자동 저장본을 옮기지 못했습니다: %s", exc)
        if remember:
            RecentProjects().add(target, self.name)
        return target

    @property
    def is_newer_format(self) -> bool:
        return self.version > CURRENT_VERSION

    @property
    def folder(self) -> str | None:
        return os.path.dirname(self.path) if self.path else None

    def missing_files(self) -> list[str]:
        """프로젝트가 가리키지만 지금은 없는 파일."""
        return [p for p in (self.design_path, self.data_path) if p and not os.path.exists(p)]

    # -- 디자인 · 명단 -------------------------------------------------------
    def load_document(self) -> Document:
        """디자인을 연다. 열린 문서의 작업 폴더는 쓰고 난 뒤 design.storage.cleanup_workdir로 지운다."""
        if not self.design_path:
            raise ProjectError("프로젝트에 디자인 파일이 지정되어 있지 않습니다.")
        try:
            return storage.load(self.design_path)
        except storage.StorageError as exc:
            raise ProjectError(str(exc)) from exc

    def save_document(self, document: Document, path: str | os.PathLike | None = None) -> str:
        """디자인을 저장하고 자동 저장본을 지운다. path를 주면 그 파일이 프로젝트의 디자인이 된다."""
        target = os.fspath(path) if path else self.design_path
        if not target:
            raise ProjectError("디자인을 저장할 위치를 정하세요.")
        target = os.path.abspath(target)
        if not target.lower().endswith(storage.EXTENSION):
            target += storage.EXTENSION
        try:
            storage.save(document, target)
        except storage.StorageError as exc:
            raise ProjectError(str(exc)) from exc
        self.discard_recovery()
        self.design_path = target
        return target

    def open_store(self) -> DataStore:
        """명단을 연다. 닫는 것은 부른 쪽 책임이다."""
        if not self.data_path:
            raise ProjectError("프로젝트에 명단 파일이 지정되어 있지 않습니다.")
        return DataStore(self.data_path)

    # -- 자동 저장 · 복구 ----------------------------------------------------
    @property
    def autosave_path(self) -> str:
        """디자인 자동 저장본 위치. 저장된 프로젝트는 프로젝트 파일 옆, 아니면 사용자 폴더."""
        if self.path:
            return self.path + AUTOSAVE_SUFFIX
        return os.path.join(app_data_dir(), AUTOSAVE_DIR, f"{self._draft_id}{storage.EXTENSION}")

    def autosave(self, document: Document) -> str:
        """편집 중인 디자인을 자동 저장본으로 남긴다(원본 디자인 파일은 건드리지 않는다)."""
        target = self.autosave_path
        try:
            storage.save(document, target, thumbnail=False)
        except (storage.StorageError, OSError) as exc:
            raise ProjectError(f"자동 저장에 실패했습니다: {exc}") from exc
        return target

    def has_recovery(self) -> bool:
        """마지막 저장 뒤에 남은 자동 저장본이 있는지(= 비정상 종료로 저장하지 못한 작업)."""
        sidecar = self.autosave_path
        if not os.path.isfile(sidecar):
            return False
        if self.design_path and os.path.isfile(self.design_path):
            return os.path.getmtime(sidecar) >= os.path.getmtime(self.design_path)
        return True

    def recover(self) -> Document:
        """자동 저장본을 연다. 사용자가 저장하면(save_document) 자동 저장본은 사라진다."""
        if not os.path.isfile(self.autosave_path):
            raise ProjectError("복구할 자동 저장본이 없습니다.")
        try:
            return storage.load(self.autosave_path)
        except storage.StorageError as exc:
            raise ProjectError(f"자동 저장본을 열 수 없습니다: {exc}") from exc

    def discard_recovery(self) -> None:
        try:
            os.remove(self.autosave_path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            log.warning("자동 저장본을 지우지 못했습니다: %s", exc)


def unsaved_recoveries() -> list[str]:
    """한 번도 저장하지 않은 프로젝트가 남긴 자동 저장본(.hsd) 목록, 최근 것부터."""
    directory = os.path.join(app_data_dir(), AUTOSAVE_DIR)
    try:
        files = [os.path.join(directory, name) for name in os.listdir(directory)
                 if name.lower().endswith(storage.EXTENSION)]
    except OSError:
        return []
    return sorted(files, key=os.path.getmtime, reverse=True)
