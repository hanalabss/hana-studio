""""이 사람들을 이 디자인으로 이 프린터에 인쇄" 한 번의 호출.

스레드는 printer.print_queue가 맡는다. 이 서비스는 큐에 카드 작업을 넣고,
큐의 시그널을 작업 단위 진행 상황으로 바꿔 전하며, 성공한 카드만 명단에 인쇄 표시를 한다.
모든 슬롯은 서비스가 사는 스레드(보통 UI 스레드)에서 돌고, 명단(DataStore)도 그 스레드에서만 쓴다.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from PySide6.QtCore import QObject, Signal

from data.selection import PrintSelection
from data.store import DataStore
from design.model import Document
from printer.base import STAGE_UNKNOWN, PrinterDevice, ProgressEvent
from services.printers import default_capabilities
from services.render_job import (
    CANCEL_MESSAGE, CardFeeder, PrintOptions, RenderJob, magnetic_mapping, needs_records,
)

log = logging.getLogger(__name__)

STATE_QUEUED = "queued"
STATE_PRINTING = "printing"
STATE_COMPLETED = "completed"     # 전부 성공
STATE_PARTIAL = "partial"         # 일부만 성공
STATE_FAILED = "failed"           # 한 장도 성공하지 못함
STATE_CANCELLED = "cancelled"     # 사용자가 취소

CARD_SUCCESS = "success"
CARD_FAILED = "failed"
CARD_CANCELLED = "cancelled"

CANCEL_AFTER_CURRENT = "after_current"
CANCEL_NOW = "now"

STAGE_CONNECT = "connect"          # 프린터 연결 중(서비스가 알리는 단계)
STAGE_NEXT_CARD = "next_card"      # 카드 사이: 다음 카드 준비
CONNECT_FAILED_MESSAGE = "프린터에 연결하지 못했습니다. 전원과 케이블을 확인해 주세요."

_PANEL_PATTERN = re.compile(r"\b(YMCKO?|Y|M|C|K|O|W|UV|화이트|형광)\s*(?:패널|panel)", re.IGNORECASE)


class PrintServiceError(Exception):
    """사용자에게 그대로 보여줄 수 있는 한국어 메시지를 담는다."""


@dataclass
class CardResult:
    job_id: int
    index: int                 # 1부터
    record_id: int | None
    record_name: str
    ok: bool
    status: str                # success | failed | cancelled
    message: str = ""
    warnings: list[str] = field(default_factory=list)


@dataclass
class PrintProgress:
    """작업 진행 상황 한 건.

    stage      "카드 삽입 중..." 같은 한국어 단계 문구(화면에 그대로 보여 준다).
    stage_key  단계 키. printer.base의 STAGE_*("feed", "front_image", "print", "eject" ...)에
               서비스가 알리는 "connect"·"next_card"를 더한 값. 드라이버가 단계를 알려 주지 않으면 "".
    panel      지금 찍는 리본 패널("Y", "K", "UV" ...). 드라이버가 알려 줄 때만 채워지고 모르면 "".
               지금의 RTAI·SMART 드라이버는 인쇄가 SDK 호출 하나라 패널을 알 수 없어 항상 ""다.
    fraction   현재 단계 안에서의 진행 0.0~1.0. 장치가 알려 줄 때만(지금은 항상 None).
    percent    작업 전체 진행률(끝난 카드 기준). card_percent는 지금 카드 안에서의 진행률 0~100.
    side       그 단계가 한 면에만 해당하면 "front" | "back".
    """

    job_id: int
    done: int
    total: int
    index: int                 # 지금 인쇄 중인 카드 순번(1부터)
    record_id: int | None
    record_name: str
    stage: str                 # "카드 삽입 중..." 같은 단계 문구
    panel: str = ""            # 드라이버가 알려 줄 때만 채워진다
    percent: int = 0
    stage_key: str = STAGE_UNKNOWN
    fraction: float | None = None
    card_percent: int = 0
    side: str | None = None


@dataclass
class PrintSummary:
    job_id: int
    state: str
    total: int
    success: int
    failed: int
    cancelled: int             # 인쇄하지 않은 카드(취소했거나 오류로 멈춰 남은 것)
    failed_record_ids: list[int] = field(default_factory=list)
    unprinted_record_ids: list[int] = field(default_factory=list)
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.state == STATE_COMPLETED


class PrintJobHandle:
    """start()가 돌려주는 작업 손잡이."""

    def __init__(self, job_id: int, name: str, device: PrinterDevice, render_job: RenderJob,
                 store: DataStore | None) -> None:
        self.id = job_id
        self.name = name
        self.device = device
        self.total = render_job.total
        self.state = STATE_QUEUED
        self.queue_job_id: int | None = None
        self.history_id: int | None = None      # 인쇄 이력(JobHistory)의 작업 ID. 시작 전·이력 없음이면 None
        self.results: list[CardResult] = []
        self.summary: PrintSummary | None = None
        self.cancel_mode: str | None = None
        self.retry_of: int | None = None
        self._render_job = render_job
        self._store = store
        self._feeder: CardFeeder | None = None
        self._stage = ""
        self._queue_error = ""

    @property
    def record_ids(self) -> list[int | None]:
        return [spec.record_id for spec in self._render_job.specs]

    @property
    def is_active(self) -> bool:
        return self.state in (STATE_QUEUED, STATE_PRINTING)

    @property
    def done(self) -> int:
        return len(self.results)

    @property
    def success_count(self) -> int:
        return sum(1 for r in self.results if r.status == CARD_SUCCESS)

    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.results if r.status == CARD_FAILED)

    def _indices(self, status: str) -> list[int]:
        return [r.index - 1 for r in self.results if r.status == status]

    @property
    def failed_indices(self) -> list[int]:
        """실패한 카드의 0부터 센 순번."""
        return self._indices(CARD_FAILED)

    @property
    def unprinted_indices(self) -> list[int]:
        attempted = {r.index - 1 for r in self.results if r.status != CARD_CANCELLED}
        return [i for i in range(self.total) if i not in attempted]

    def __repr__(self) -> str:
        return f"PrintJobHandle(id={self.id}, state={self.state!r}, {self.done}/{self.total})"


class PrintService(QObject):
    """인쇄 작업 창구.

    job_queued(PrintJobHandle)     큐에 들어갔다
    job_started(PrintJobHandle)    프린터가 이 작업을 시작했다
    progress(PrintProgress)        전체 진행(done/total), 현재 레코드, 카드 단계 문구
    card_result(CardResult)        카드 한 장의 결과
    finished(PrintSummary)         작업이 끝났다(성공·실패·취소 장수, 실패한 레코드)
    error(PrintJobHandle, str)     사용자에게 알릴 오류(프린터 연결 실패, 인쇄 표시 저장 실패 등)
    """

    job_queued = Signal(object)
    job_started = Signal(object)
    progress = Signal(object)
    card_result = Signal(object)
    finished = Signal(object)
    error = Signal(object, str)

    def __init__(self, queue: Any = None, history: Any = None, parent: QObject | None = None) -> None:
        """queue: 인쇄 큐(기본은 전역 print_queue). history: 지정하면 큐의 인쇄 이력 저장소로 쓴다."""
        super().__init__(parent)
        if queue is None:
            from printer.print_queue import print_queue as queue
        self._queue = queue
        if history is not None:
            queue.set_history(history)
        self._counter = 0
        self._jobs: dict[int, PrintJobHandle] = {}
        self._by_queue_id: dict[int, PrintJobHandle] = {}
        self._queue_active: int | None = None
        queue.job_started.connect(self._on_queue_job_started)
        queue.card_progress.connect(self._on_queue_card_progress)
        queue.card_finished.connect(self._on_queue_card_finished)
        queue.job_error.connect(self._on_queue_job_error)
        queue.job_completed.connect(self._on_queue_job_completed)
        queue.job_cancelled.connect(self._on_queue_job_cancelled)

    # -- 조회 --------------------------------------------------------------
    def job(self, job_id: int) -> PrintJobHandle | None:
        return self._jobs.get(job_id)

    def active_jobs(self) -> list[PrintJobHandle]:
        return [job for job in self._jobs.values() if job.is_active]

    @property
    def is_printing(self) -> bool:
        return bool(self.active_jobs())

    # -- 시작 --------------------------------------------------------------
    def start(
        self,
        document: Document,
        store: DataStore | None,
        selection: PrintSelection | Iterable[int] | None,
        device: PrinterDevice,
        options: PrintOptions | None = None,
        *,
        filter: Any = None,
        order: Any = None,
    ) -> PrintJobHandle:
        """인쇄를 시작하고 작업 손잡이를 돌려준다. 렌더링·인쇄는 이후 비동기로 진행된다.

        selection: PrintSelection(전체·현재 레코드·범위·미인쇄 등), 레코드 _id 목록, 또는 None.
          None이면 디자인이 명단 열을 쓰는 경우 전체 레코드, 아니면 options.copies장을 인쇄한다.
        filter/order: 화면(그리드)에 걸려 있는 필터·정렬. PrintSelection을 풀 때 쓴다.
        시작할 수 없으면 PrintServiceError(한국어 문구).
        """
        if device is None:
            raise PrintServiceError("프린터를 선택하세요.")
        options = options or PrintOptions()
        capabilities = options.capabilities
        if capabilities is None:
            capabilities = default_capabilities(device) or None

        uses_data = needs_records(document) or bool(magnetic_mapping(document, options))
        if selection is None and (store is None or not uses_data):
            render_job = RenderJob.for_copies(document, options.copies, options, capabilities)
        else:
            if store is None:
                raise PrintServiceError("명단이 없어 인쇄할 대상을 정할 수 없습니다. 명단을 먼저 여세요.")
            record_ids = self._resolve(store, selection, filter, order)
            if not record_ids:
                raise PrintServiceError("인쇄할 레코드가 없습니다.")
            render_job = RenderJob.for_records(document, store, record_ids, options, capabilities)
        name = options.job_name or document.name
        return self._enqueue(render_job, store, device, name)

    @staticmethod
    def _resolve(store: DataStore, selection: Any, filter: Any, order: Any) -> list[int]:
        if selection is None:
            return store.ids(filter, order)
        if isinstance(selection, PrintSelection):
            try:
                return selection.resolve(store, filter, order)
            except Exception as exc:
                raise PrintServiceError(str(exc)) from exc
        if isinstance(selection, int):
            return [selection]
        return [int(record_id) for record_id in selection]

    def _enqueue(self, render_job: RenderJob, store: DataStore | None, device: PrinterDevice,
                 name: str, retry_of: int | None = None) -> PrintJobHandle:
        self._counter += 1
        handle = PrintJobHandle(self._counter, name, device, render_job, store)
        handle.retry_of = retry_of
        handle._feeder = CardFeeder(render_job, parent=self)
        self._jobs[handle.id] = handle
        try:
            queue_id = self._queue.add_card_job(
                device, handle._feeder.cards(), name=name,
                stop_on_error=render_job.options.stop_on_error,
            )
        except Exception as exc:
            self._jobs.pop(handle.id, None)
            self._dispose(handle)
            raise PrintServiceError(f"인쇄 작업을 시작하지 못했습니다: {exc}") from exc
        handle.queue_job_id = queue_id
        self._by_queue_id[queue_id] = handle
        self.job_queued.emit(handle)
        if self._queue_active == queue_id:
            self._mark_started(handle)
        return handle

    # -- 취소 · 재시도 -------------------------------------------------------
    def cancel(self, job: PrintJobHandle | int, mode: str = CANCEL_AFTER_CURRENT) -> bool:
        """작업을 취소한다.

        "after_current": 지금 진행 중인 카드는 끝까지 인쇄하고 다음 카드부터 하지 않는다.
        "now": 아직 프린터로 넘어가지 않은(이미지 준비 중인) 카드도 버린다. 이미 프린터 안에서
               인쇄되고 있는 카드는 SDK가 중단을 지원하지 않아 끝까지 인쇄된다.
        """
        if mode not in (CANCEL_AFTER_CURRENT, CANCEL_NOW):
            raise ValueError(f"알 수 없는 취소 방식입니다: {mode}")
        handle = self._handle(job)
        if handle is None or not handle.is_active or handle.queue_job_id is None:
            return False
        handle.cancel_mode = mode
        accepted = self._queue.cancel_job(handle.queue_job_id)
        if mode == CANCEL_NOW and handle._feeder is not None and handle.is_active:
            handle._feeder.abort(CANCEL_MESSAGE)
        return bool(accepted)

    def cancel_all(self, mode: str = CANCEL_AFTER_CURRENT) -> None:
        for handle in list(self.active_jobs()):
            self.cancel(handle, mode)

    def shutdown(self) -> None:
        """앱을 닫을 때 부른다. 이 서비스의 작업을 모두 멈춰 인쇄 스레드가 이미지를 기다리며 서 있지 않게 한다."""
        self.cancel_all(CANCEL_NOW)

    def retry_failed(self, job: PrintJobHandle | int, *, include_unprinted: bool = False,
                     device: PrinterDevice | None = None) -> PrintJobHandle:
        """끝난 작업에서 실패한 카드만 다시 인쇄한다. include_unprinted면 인쇄하지 못하고 남은 카드도 포함한다."""
        handle = self._handle(job)
        if handle is None:
            raise PrintServiceError("다시 인쇄할 작업을 찾을 수 없습니다.")
        if handle.is_active:
            raise PrintServiceError("작업이 끝난 뒤에 다시 인쇄할 수 있습니다.")
        indices = set(handle.failed_indices)
        if include_unprinted:
            indices |= set(handle.unprinted_indices)
        if not indices:
            raise PrintServiceError("다시 인쇄할 카드가 없습니다.")
        render_job = handle._render_job.subset(sorted(indices))
        return self._enqueue(render_job, handle._store, device or handle.device,
                             f"{handle.name} (재인쇄)", retry_of=handle.id)

    def _handle(self, job: PrintJobHandle | int) -> PrintJobHandle | None:
        return job if isinstance(job, PrintJobHandle) else self._jobs.get(job)

    # -- 큐 시그널 ----------------------------------------------------------
    def _on_queue_job_started(self, queue_id: int, index: int, total: int) -> None:
        self._queue_active = queue_id
        handle = self._by_queue_id.get(queue_id)
        if handle is not None:
            self._mark_started(handle)

    def _mark_started(self, handle: PrintJobHandle) -> None:
        if handle.state != STATE_QUEUED:
            return
        handle.state = STATE_PRINTING
        if handle.queue_job_id is not None:
            handle.history_id = self._queue.history_id_for(handle.queue_job_id)
        self.job_started.emit(handle)
        self._emit_progress(handle, "프린터 연결 중...", stage_key=STAGE_CONNECT)

    def _on_queue_card_progress(self, queue_id: int, card_index: int, event: ProgressEvent) -> None:
        handle = self._by_queue_id.get(queue_id)
        if handle is None or handle.state != STATE_PRINTING or not event.text:
            return
        self._emit_progress(handle, event.text.strip(), event)

    def _emit_progress(self, handle: PrintJobHandle, stage: str, event: ProgressEvent | None = None,
                       *, stage_key: str = STAGE_UNKNOWN) -> None:
        handle._stage = stage
        total = handle.total
        done = handle.done
        position = min(done, total - 1)
        spec = handle._render_job.specs[position]
        panel = event.panel if event is not None and event.panel else ""
        if not panel:
            # 단계 키를 주지 않는 드라이버가 문구에 패널 이름을 넣어 보내는 경우만 문구에서 읽는다.
            match = _PANEL_PATTERN.search(stage)
            panel = match.group(1) if match else ""
        self.progress.emit(PrintProgress(
            job_id=handle.id, done=done, total=total, index=position + 1,
            record_id=spec.record_id, record_name=spec.display_name, stage=stage,
            panel=panel,
            percent=int(done * 100 / total) if total else 0,
            stage_key=event.stage if event is not None else stage_key,
            fraction=event.fraction if event is not None else None,
            card_percent=event.percent if event is not None else 0,
            side=event.side if event is not None else None,
        ))

    def _on_queue_job_error(self, queue_id: int, message: str) -> None:
        handle = self._by_queue_id.get(queue_id)
        if handle is not None:
            handle._queue_error = message or ""

    def _on_queue_card_finished(self, queue_id: int, card_index: int, status: str, message: str) -> None:
        handle = self._by_queue_id.get(queue_id)
        if handle is None:
            return
        position = card_index - 1
        if not 0 <= position < handle.total:
            return
        spec = handle._render_job.specs[position]
        feeder = handle._feeder
        if status not in (CARD_SUCCESS, CARD_FAILED, CARD_CANCELLED):
            status = CARD_FAILED
        ok = status == CARD_SUCCESS
        warnings: list[str] = []
        if feeder is not None:
            warnings = feeder.warnings(position)
            feeder.release(position)
        result = CardResult(
            job_id=handle.id, index=card_index, record_id=spec.record_id,
            record_name=spec.display_name, ok=ok, status=status, message=message or "",
            warnings=warnings,
        )
        handle.results.append(result)
        if ok and spec.record_id is not None and handle._store is not None:
            try:
                handle._store.mark_printed([spec.record_id])
            except Exception as exc:
                log.exception("인쇄 표시 저장 실패 (record_id=%s)", spec.record_id)
                self.error.emit(handle, f"{spec.display_name}: 카드는 인쇄됐지만 명단에 인쇄 표시를 "
                                        f"남기지 못했습니다. ({exc})")
        self.card_result.emit(result)
        if handle.done < handle.total and status != CARD_CANCELLED:
            self._emit_progress(handle, "다음 카드 준비 중..." if ok else "카드 인쇄 실패",
                                stage_key=STAGE_NEXT_CARD)

    def _on_queue_job_cancelled(self, queue_id: int) -> None:
        handle = self._by_queue_id.get(queue_id)
        if handle is None:
            return
        if handle.state == STATE_QUEUED:
            self._finalize(handle, cancelled=True)
        elif handle.cancel_mode is None:
            # 큐에서 직접 취소한 경우(print_queue.cancel_all 등)도 취소로 마무리한다.
            handle.cancel_mode = CANCEL_AFTER_CURRENT

    def _on_queue_job_completed(self, queue_id: int, success: bool) -> None:
        if self._queue_active == queue_id:
            self._queue_active = None
        handle = self._by_queue_id.get(queue_id)
        if handle is not None and handle.is_active:
            self._finalize(handle, cancelled=handle.cancel_mode is not None)

    # -- 마무리 ------------------------------------------------------------
    def _finalize(self, handle: PrintJobHandle, cancelled: bool) -> None:
        specs = handle._render_job.specs
        success, failed = handle.success_count, handle.failed_count
        unprinted = handle.unprinted_indices
        message = ""
        if success == handle.total:
            state = STATE_COMPLETED
        elif cancelled:
            state = STATE_CANCELLED
            message = "인쇄를 취소했습니다."
        elif success:
            state = STATE_PARTIAL
        else:
            state = STATE_FAILED
        last_error = next((r.message for r in reversed(handle.results) if r.status == CARD_FAILED), "")
        connection_failed = not handle.results and not cancelled
        if connection_failed:
            last_error = handle._queue_error or CONNECT_FAILED_MESSAGE
        if state in (STATE_PARTIAL, STATE_FAILED):
            message = last_error

        def record_ids(indices: Iterable[int]) -> list[int]:
            ids = (specs[i].record_id for i in indices)
            return list(dict.fromkeys(i for i in ids if i is not None))

        if handle.queue_job_id is not None and handle.history_id is None:
            handle.history_id = self._queue.history_id_for(handle.queue_job_id)
        handle.state = state
        handle.summary = PrintSummary(
            job_id=handle.id, state=state, total=handle.total, success=success, failed=failed,
            cancelled=len(unprinted), failed_record_ids=record_ids(handle.failed_indices),
            unprinted_record_ids=record_ids(unprinted), message=message,
        )
        self._dispose(handle)
        if handle.queue_job_id is not None:
            self._by_queue_id.pop(handle.queue_job_id, None)
        if connection_failed:
            self.error.emit(handle, last_error)
        self.finished.emit(handle.summary)

    @staticmethod
    def _dispose(handle: PrintJobHandle) -> None:
        feeder = handle._feeder
        handle._feeder = None
        if feeder is not None:
            feeder.close()
            feeder.deleteLater()

    def forget(self, job: PrintJobHandle | int) -> None:
        """끝난 작업의 기록(재인쇄용 디자인 사본 포함)을 버린다."""
        handle = self._handle(job)
        if handle is not None and not handle.is_active:
            self._jobs.pop(handle.id, None)
