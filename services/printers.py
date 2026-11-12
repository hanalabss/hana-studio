"""프린터 탐지 · 선택 · 상태 조회.

SDK(특히 R600)는 프로세스 전체에서 하나만 쓸 수 있으므로, 탐지와 상태 조회는
인쇄 스레드가 도는 동안에는 하지 않고 작업 스레드 하나에서만 한다.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any, Callable

from PySide6.QtCore import QObject, QThread, QTimer, Signal

from printer import registry
from printer.base import (
    CAP_CARD_MOVE, CAP_DUAL_SIDE, CAP_K_SPLIT, CAP_MAGNETIC, CAP_MIFARE, CAP_POSITION_OFFSET,
    CAP_RIBBON_LEVEL, CAP_STATUS, CAP_UV, CAP_WHITE, FAMILY_RTAI, FAMILY_SMART, PrinterDevice,
    PrinterError, PrinterStatus,
)

log = logging.getLogger(__name__)

APP_DIR_NAME = "HanaStudio"
SETTINGS_FILE = "printer.json"
DEFAULT_POLL_INTERVAL_MS = 20000   # 상태 확인 주기. 프린터에 부담을 주지 않고 기록 파일이 빨리 커지지 않게 20초로 둔다.

_FAMILY_CAPABILITIES = {
    FAMILY_RTAI: frozenset({
        CAP_DUAL_SIDE, CAP_WHITE, CAP_UV, CAP_MAGNETIC, CAP_STATUS, CAP_RIBBON_LEVEL,
        CAP_K_SPLIT, CAP_MIFARE, CAP_CARD_MOVE, CAP_POSITION_OFFSET,
    }),
    FAMILY_SMART: frozenset({
        CAP_DUAL_SIDE, CAP_MAGNETIC, CAP_STATUS, CAP_RIBBON_LEVEL, CAP_CARD_MOVE,
        CAP_POSITION_OFFSET,
    }),
}


def app_data_dir() -> str:
    """사용자별 설정 폴더(%LOCALAPPDATA%\\HanaStudio)."""
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    return os.path.join(base, APP_DIR_NAME)


def default_capabilities(device: PrinterDevice | None) -> set[str]:
    """장치에 연결하지 않고 아는 범위의 지원 기능(제품군 기준)."""
    if device is None:
        return set()
    return set(_FAMILY_CAPABILITIES.get(device.family, ()))


def device_to_dict(device: PrinterDevice) -> dict[str, Any]:
    return {
        "family": device.family, "name": device.name, "connection": device.connection,
        "address": device.address, "extra": dict(device.extra or {}),
    }


def device_from_dict(data: Any) -> PrinterDevice | None:
    if not isinstance(data, dict) or not data.get("family") or not data.get("name"):
        return None
    extra = data.get("extra")
    return PrinterDevice(
        str(data["family"]), str(data["name"]), str(data.get("connection") or ""),
        str(data.get("address") or ""), extra if isinstance(extra, dict) else {},
    )


def _same_device(a: PrinterDevice | None, b: PrinterDevice | None) -> bool:
    if a is None or b is None:
        return False
    return (a.family, a.address or a.name) == (b.family, b.address or b.name)


class _Task(QThread):
    done = Signal(object)

    def __init__(self, work: Callable[[], Any], lock: threading.Lock, parent: QObject | None = None):
        super().__init__(parent)
        self._work = work
        self._lock = lock

    def run(self) -> None:
        with self._lock:
            try:
                result = self._work()
            except Exception as exc:
                log.exception("프린터 작업 스레드 오류")
                result = exc
        self.done.emit(result)


class PrinterService(QObject):
    """화면이 쓰는 프린터 창구.

    devices_changed(list[PrinterDevice])  탐지가 끝날 때마다
    current_changed(PrinterDevice | None) 선택한 프린터가 바뀔 때
    status_changed(PrinterStatus)         선택한 프린터의 상태가 바뀔 때
    busy_changed(bool)                    탐지·상태 조회 중인지
    """

    devices_changed = Signal(list)
    current_changed = Signal(object)
    status_changed = Signal(object)
    busy_changed = Signal(bool)

    def __init__(
        self,
        queue: Any = None,
        settings_path: str | None = None,
        *,
        poll_interval_ms: int = DEFAULT_POLL_INTERVAL_MS,
        guard_queue: bool = True,
        parent: QObject | None = None,
    ) -> None:
        """queue: 인쇄 큐(기본은 전역 print_queue). 큐가 일하는 동안에는 SDK를 건드리지 않는다.

        guard_queue: 큐의 프린터 생성 함수를 감싸, 탐지·상태 조회가 끝난 뒤에 인쇄 스레드가 SDK를 열게 한다.
        """
        super().__init__(parent)
        if queue is None:
            from printer.print_queue import print_queue as queue
        self._queue = queue
        self._settings_path = settings_path
        self._devices: list[PrinterDevice] = []
        self._diagnostics: list[str] = []
        self._current: PrinterDevice | None = None
        self._remembered = self._load_remembered()
        self._status: PrinterStatus | None = None
        self._capabilities: set[str] | None = None
        self._sdk_lock = threading.Lock()
        self._task: _Task | None = None
        self._task_kind = ""
        self._refresh_pending = False
        self._discovered_once = False
        self._timer = QTimer(self)
        self._timer.setInterval(max(200, int(poll_interval_ms)))
        self._timer.timeout.connect(self.poll_now)
        self._queue.all_jobs_completed.connect(self._on_queue_idle)
        if guard_queue:
            self._queue.set_printer_factory(self._guarded_open)
            self._queue.set_legacy_printer_factory(self._guarded_legacy)

    # -- 조회 --------------------------------------------------------------
    def devices(self) -> list[PrinterDevice]:
        return list(self._devices)

    def diagnostics(self) -> list[str]:
        """탐지 경과(장치 없음, 라이브러리 없음 등)의 한국어 설명."""
        return list(self._diagnostics)

    def current(self) -> PrinterDevice | None:
        return self._current

    def status(self) -> PrinterStatus | None:
        """마지막으로 조회한 상태. 아직 조회하지 않았으면 None."""
        return self._status

    def capabilities(self) -> set[str]:
        """선택한 프린터가 지원하는 기능. 상태를 조회한 뒤에는 장착 옵션(플리퍼 등)까지 반영된다."""
        if self._capabilities is not None:
            return set(self._capabilities)
        return default_capabilities(self._current)

    @property
    def is_busy(self) -> bool:
        return self._task is not None

    @property
    def printing(self) -> bool:
        """인쇄 스레드가 돌고 있거나 대기 작업이 있는지."""
        try:
            state = self._queue.get_queue_status()
            if state.get("is_processing") or state.get("queue_size"):
                return True
        except Exception:
            log.debug("인쇄 큐 상태를 읽지 못했습니다.", exc_info=True)
        try:
            from printer.printer_thread import print_manager

            return bool(print_manager.is_printing)
        except Exception:
            return False

    @property
    def use_fakes(self) -> bool:
        return registry.fakes_enabled()

    # -- 선택 --------------------------------------------------------------
    def select(self, device: PrinterDevice | None) -> None:
        if device is self._current or _same_device(device, self._current):
            return
        self._current = device
        self._status = None
        self._capabilities = None
        if device is not None:
            self._remembered = device
            self._save_remembered(device)
        self.current_changed.emit(device)
        if device is not None and self._timer.isActive():
            QTimer.singleShot(0, self.poll_now)

    def set_use_fakes(self, enabled: bool) -> None:
        """실기 대신 가상 프린터를 쓴다(시연·테스트). 목록을 다시 찾는다."""
        registry.use_fakes(bool(enabled))
        self._current = None
        self._status = None
        self._capabilities = None
        self.refresh()

    # -- 탐지 --------------------------------------------------------------
    def refresh(self) -> bool:
        """백그라운드에서 프린터를 찾는다. 인쇄 중이면 인쇄가 끝난 뒤로 미루고 False를 돌려준다."""
        if self.printing or self._task is not None:
            self._refresh_pending = True
            return False
        self._refresh_pending = False
        self._start_task("discover", registry.discover)
        return True

    def _on_discovered(self, result: Any) -> None:
        if isinstance(result, Exception):
            self._devices = []
            self._diagnostics = [f"프린터를 찾는 중 오류가 발생했습니다: {result}"]
        else:
            self._devices = list(result)
            self._diagnostics = list(getattr(result, "diagnostics", []))
        self._discovered_once = True
        self.devices_changed.emit(list(self._devices))
        keep = next((d for d in self._devices if _same_device(d, self._current)), None)
        if keep is None:
            keep = next((d for d in self._devices if _same_device(d, self._remembered)), None)
        if keep is None and self._devices:
            keep = self._devices[0]
        if keep is None:
            if self._current is not None:
                self._current = None
                self._status = None
                self._capabilities = None
                self.current_changed.emit(None)
            return
        if not _same_device(keep, self._current):
            remembered = self._remembered
            self._current = keep
            self._status = None
            self._capabilities = None
            if remembered is None:
                self._remembered = keep
                self._save_remembered(keep)
            self.current_changed.emit(keep)
        else:
            self._current = keep

    # -- 상태 조회 ----------------------------------------------------------
    def start_polling(self, interval_ms: int | None = None) -> None:
        """선택한 프린터의 상태를 주기적으로 본다. 인쇄 중에는 건너뛴다."""
        if interval_ms is not None:
            self._timer.setInterval(max(200, int(interval_ms)))
        self._timer.start()
        QTimer.singleShot(0, self.poll_now)

    def stop_polling(self) -> None:
        self._timer.stop()

    def poll_now(self) -> bool:
        """상태를 한 번 조회한다. 인쇄 중이거나 다른 조회가 진행 중이면 하지 않고 False."""
        device = self._current
        if device is None or self._task is not None or self.printing:
            return False

        def work():
            try:
                printer = registry.open(device)
            except PrinterError as exc:
                return device, PrinterStatus(ready=False, message=exc.message, code=exc.code), None
            try:
                return device, printer.status(), set(printer.capabilities())
            except PrinterError as exc:
                return device, PrinterStatus(ready=False, message=exc.message, code=exc.code), None
            finally:
                printer.close()

        self._start_task("status", work)
        return True

    def _on_status(self, result: Any) -> None:
        if isinstance(result, Exception):
            device, status, capabilities = self._current, PrinterStatus(
                ready=False, message=f"프린터 상태를 확인하지 못했습니다: {result}"), None
        else:
            device, status, capabilities = result
        if not _same_device(device, self._current):
            return
        if capabilities is not None:
            self._capabilities = capabilities
        if status != self._status:
            self._status = status
            self.status_changed.emit(status)

    # -- 작업 스레드 ---------------------------------------------------------
    def _start_task(self, kind: str, work: Callable[[], Any]) -> None:
        task = _Task(work, self._sdk_lock, self)
        self._task, self._task_kind = task, kind
        task.done.connect(self._on_task_done)
        self.busy_changed.emit(True)
        task.start()

    def _on_task_done(self, result: Any) -> None:
        task, kind = self._task, self._task_kind
        self._task, self._task_kind = None, ""
        if task is not None:
            task.wait(5000)
            task.deleteLater()
        if kind == "discover":
            self._on_discovered(result)
        elif kind == "status":
            self._on_status(result)
        self.busy_changed.emit(False)
        if self._refresh_pending and not self.printing:
            self.refresh()

    def _on_queue_idle(self, *args: Any) -> None:
        if self._refresh_pending:
            QTimer.singleShot(0, self.refresh)

    def _wait_sdk(self) -> None:
        with self._sdk_lock:
            pass

    def _guarded_open(self, device: PrinterDevice):
        self._wait_sdk()
        return registry.open(device)

    def _guarded_legacy(self, dll_path: str, selected: Any):
        from printer.r600_printer import R600Printer

        self._wait_sdk()
        if selected is not None:
            return R600Printer(dll_path, selected_printer=selected)
        return R600Printer(dll_path)

    def shutdown(self) -> None:
        """타이머를 멈추고 진행 중인 조회가 끝나기를 기다린다. 앱을 닫을 때 부른다."""
        self._timer.stop()
        self._refresh_pending = False
        task = self._task
        if task is not None:
            task.wait(10000)

    # -- 마지막으로 쓴 프린터 -------------------------------------------------
    def _settings_file(self) -> str:
        return self._settings_path or os.path.join(app_data_dir(), SETTINGS_FILE)

    def _load_remembered(self) -> PrinterDevice | None:
        try:
            with open(self._settings_file(), "r", encoding="utf-8") as handle:
                return device_from_dict(json.load(handle).get("last_device"))
        except (OSError, ValueError, AttributeError):
            return None

    def _save_remembered(self, device: PrinterDevice) -> None:
        path = self._settings_file()
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            temporary = path + ".tmp"
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump({"last_device": device_to_dict(device)}, handle,
                          ensure_ascii=False, indent=2, default=str)
            os.replace(temporary, path)
        except OSError as exc:
            log.warning("마지막으로 쓴 프린터를 저장하지 못했습니다: %s", exc)

    def remembered(self) -> PrinterDevice | None:
        return self._remembered
