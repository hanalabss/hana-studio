"""폴더 자동 인쇄: 정해 둔 폴더에 명단 파일(CSV·Excel)이 들어오면 정해 둔 디자인으로 인쇄한다.

흐름
  1. interval_ms마다 폴더를 본다. 켤 때 이미 있던 파일은 인쇄하지 않는다.
  2. 새 파일은 크기·수정 시각이 두 번 연달아 같아야(복사가 끝나야) 처리한다.
  3. 파일을 임시 명단으로 읽는다. 열 이름이 디자인의 칸 이름(이름, 사번 …)과 같아야 한다.
     사진 칸 값은 파일 이름(사번 등)이나 경로이고, 사진은 감시 폴더에서 찾는다.
  4. 인쇄 서비스로 보낸다. 한 번에 파일 하나씩.
  5. 끝나면 파일을 '인쇄함'(모두 성공) 또는 '실패'(하나라도 실패·취소, 읽기 오류) 폴더로 옮긴다.
프린터가 연결돼 있지 않으면 파일을 그대로 두고 다음에 다시 본다.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QObject, QTimer, Signal

from data.errors import DataError
from data.importers import import_into, preview_file
from data.photos import set_photo_folder
from data.store import DataStore
from design import storage
from design.binding import columns_used
from services.render_job import PrintOptions, needs_records

log = logging.getLogger(__name__)

WATCH_SUFFIXES = (".csv", ".txt", ".xlsx", ".xlsm")
DONE_DIR = "인쇄함"
FAILED_DIR = "실패"
DEFAULT_INTERVAL_MS = 3000


class AutoPrintError(Exception):
    """사용자에게 그대로 보여 줄 한국어 문구."""


@dataclass
class AutoPrintEvent:
    """파일 하나를 처리한 결과."""
    name: str
    ok: bool
    message: str
    moved_to: str = ""
    printed: int = 0
    failed: int = 0
    finished_at: float = 0.0


def design_columns(document) -> dict[str, bool]:
    """디자인이 쓰는 명단 칸 {이름: 사진 칸인가}."""
    used = columns_used(document)
    found: dict[str, bool] = {}
    for side in document.sides.values():
        for item in side.sorted_items():
            source = getattr(item, "source", None)
            if source is None:
                continue
            for name in source.columns():
                if name in used:
                    found[name] = found.get(name, False) or item.kind == "image"
    return found


def _move(path: Path, folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / path.name
    number = 2
    while target.exists():
        target = folder / f"{path.stem}-{number}{path.suffix}"
        number += 1
    shutil.move(str(path), str(target))
    return target


class FolderAutoPrinter(QObject):
    """statusChanged(str): 지금 상태 문구. fileFinished(AutoPrintEvent): 파일 하나가 끝남. runningChanged(bool)."""

    statusChanged = Signal(str)
    fileFinished = Signal(object)
    runningChanged = Signal(bool)

    def __init__(self, printing, device: Callable[[], Any], options: Callable[[], PrintOptions] | None = None,
                 work_dir: str | os.PathLike | None = None, interval_ms: int = DEFAULT_INTERVAL_MS,
                 parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._printing = printing
        self._device = device
        self._options = options or PrintOptions
        self._work_dir = Path(work_dir) if work_dir else None
        self._timer = QTimer(self)
        self._timer.setInterval(int(interval_ms))
        self._timer.timeout.connect(self.poll)
        self._folder: Path | None = None
        self._design: Path | None = None
        self._ignored: set[str] = set()
        self._sizes: dict[str, tuple[int, int]] = {}
        self._current: dict[str, Any] | None = None
        self._status = "꺼져 있습니다."
        self.history: list[AutoPrintEvent] = []
        printing.finished.connect(self._on_finished)

    # ---- 상태 ----
    @property
    def running(self) -> bool:
        return self._timer.isActive()

    @property
    def folder(self) -> Path | None:
        return self._folder

    @property
    def design_path(self) -> Path | None:
        return self._design

    @property
    def status(self) -> str:
        return self._status

    @property
    def busy(self) -> bool:
        return self._current is not None

    def _set_status(self, text: str) -> None:
        if text != self._status:
            self._status = text
            self.statusChanged.emit(text)

    # ---- 켜기·끄기 ----
    def start(self, folder: str | os.PathLike, design_path: str | os.PathLike) -> None:
        folder = Path(folder)
        design_path = Path(design_path)
        if not folder.is_dir():
            raise AutoPrintError("감시할 폴더가 없습니다. 폴더를 다시 골라 주세요.")
        if not design_path.is_file():
            raise AutoPrintError("인쇄할 디자인 파일이 없습니다. 디자인을 다시 골라 주세요.")
        document = self._load_design(design_path)
        if not design_columns(document):
            raise AutoPrintError("이 디자인에는 명단과 이어진 칸이 없습니다. "
                                 "이름·사번처럼 명단 항목을 쓰는 디자인을 골라 주세요.")
        self._folder, self._design = folder, design_path
        self._ignored = {p.name for p in self._candidates()}
        self._sizes.clear()
        self._timer.start()
        self._set_status(f"'{folder.name}' 폴더를 보고 있습니다. 명단 파일을 넣으면 인쇄합니다.")
        self.runningChanged.emit(True)

    def stop(self) -> None:
        """감시를 멈춘다. 인쇄 중인 파일은 끝까지 인쇄하고 결과대로 옮긴다."""
        if not self._timer.isActive():
            return
        self._timer.stop()
        self._set_status("꺼져 있습니다." if self._current is None else "지금 파일까지만 인쇄하고 멈춥니다.")
        self.runningChanged.emit(False)

    # ---- 폴더 보기 ----
    def _candidates(self) -> list[Path]:
        if self._folder is None or not self._folder.is_dir():
            return []
        return sorted(p for p in self._folder.iterdir()
                      if p.is_file() and p.suffix.lower() in WATCH_SUFFIXES and not p.name.startswith(("~$", ".")))

    def poll(self) -> None:
        if self._folder is None or self._current is not None:
            return
        if not self._folder.is_dir():
            self._set_status("감시하던 폴더가 없어졌습니다. 폴더를 확인해 주세요.")
            return
        for path in self._candidates():
            if path.name in self._ignored:
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            signature = (stat.st_size, stat.st_mtime_ns)
            if self._sizes.get(path.name) != signature:
                self._sizes[path.name] = signature          # 다음에도 같으면 복사가 끝난 것
                continue
            if self._device() is None:
                self._set_status(f"프린터가 연결되면 '{path.name}'을(를) 인쇄합니다.")
                return
            self._sizes.pop(path.name, None)
            self._process(path)
            return

    # ---- 파일 하나 ----
    def _load_design(self, path: Path):
        try:
            return storage.load(path)
        except Exception as exc:
            raise AutoPrintError(f"디자인 파일을 열지 못했습니다: {exc}") from exc

    def _store_path(self) -> Path:
        base = self._work_dir or Path(os.environ.get("TEMP", "."))
        base.mkdir(parents=True, exist_ok=True)
        return base / f"auto_print_{os.getpid()}_{time.time_ns()}.hsdata"

    def _process(self, path: Path) -> None:
        store: DataStore | None = None
        try:
            document = self._load_design(self._design)
            fields = design_columns(document)
            store = DataStore(self._store_path())
            for name, is_photo in fields.items():
                store.add_column(name, "photo" if is_photo else "text")
                if is_photo:
                    set_photo_folder(store, name, self._folder)
            try:
                preview = preview_file(path)
            except (DataError, OSError) as exc:
                raise AutoPrintError(f"파일을 읽지 못했습니다: {exc}") from exc
            names = {column.name for column in preview.columns}
            matched = [name for name in fields if name in names]
            if not matched:
                raise AutoPrintError(
                    "파일의 열 이름이 디자인의 칸과 하나도 맞지 않습니다. "
                    f"첫 줄에 {', '.join(list(fields)[:4])} 같은 이름을 적어 주세요.")
            column_map = {column.name: column.name for column in preview.columns}
            result = import_into(store, preview, column_map)
            if result.inserted == 0:
                raise AutoPrintError("파일에 인쇄할 사람이 없습니다.")
            if not needs_records(document):
                raise AutoPrintError("이 디자인은 명단을 쓰지 않습니다.")
            options = dataclasses.replace(self._options(), job_name=f"자동 인쇄 · {path.name}")
            handle = self._printing.start(document, store, None, self._device(), options)
        except Exception as exc:
            if store is not None:
                self._discard(store)
            message = str(exc) if isinstance(exc, (AutoPrintError, DataError)) else f"인쇄를 시작하지 못했습니다: {exc}"
            log.warning("자동 인쇄 실패 %s: %s", path.name, message)
            self._finish_file(path, ok=False, message=message)
            return
        self._current = {"path": path, "store": store, "job_id": handle.id, "total": handle.total}
        self._set_status(f"'{path.name}' 인쇄 중 ({handle.total}장)")

    def _on_finished(self, summary) -> None:
        current = self._current
        if current is None or summary.job_id != current["job_id"]:
            return
        self._current = None
        self._discard(current["store"])
        ok = summary.ok and summary.failed == 0 and summary.cancelled == 0
        if ok:
            message = f"{summary.success}장 인쇄했습니다."
        else:
            message = (f"{summary.success}장 인쇄, {summary.failed}장 실패, {summary.cancelled}장 인쇄 안 함."
                       + (f" {summary.message}" if summary.message else ""))
        self._finish_file(current["path"], ok=ok, message=message,
                          printed=summary.success, failed=summary.failed + summary.cancelled)

    def _finish_file(self, path: Path, *, ok: bool, message: str, printed: int = 0, failed: int = 0) -> None:
        moved = ""
        try:
            moved = str(_move(path, self._folder / (DONE_DIR if ok else FAILED_DIR)))
        except OSError as exc:
            log.warning("자동 인쇄 파일을 옮기지 못했습니다 %s: %s", path, exc)
            self._ignored.add(path.name)                  # 옮기지 못하면 다시 인쇄하지 않게
        event = AutoPrintEvent(path.name, ok, message, moved, printed, failed, time.time())
        self.history.insert(0, event)
        del self.history[50:]
        self.fileFinished.emit(event)
        if self.running:
            self._set_status(f"'{path.name}': {message} 다음 파일을 기다립니다.")
        else:
            self._set_status("꺼져 있습니다.")

    @staticmethod
    def _discard(store: DataStore) -> None:
        path = getattr(store, "path", None)
        try:
            store.close()
        except Exception:
            pass
        for candidate in ([Path(path)] if path else []):
            for extra in (candidate, Path(str(candidate) + "-wal"), Path(str(candidate) + "-shm")):
                try:
                    extra.unlink()
                except OSError:
                    pass
