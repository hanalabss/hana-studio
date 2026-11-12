"""인쇄 화면: 준비(누구를, 어떻게, 인쇄 전 확인) → 인쇄 중 → 결과."""
from __future__ import annotations

import copy
import logging
import time
from typing import Any, Callable

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import QDialog, QHBoxLayout, QMenu, QStackedWidget, QVBoxLayout, QWidget

from printer.base import CAP_DUAL_SIDE, CAP_MAGNETIC, CAP_UV, CAP_WHITE
from services import preflight
from services.print_service import (
    CANCEL_AFTER_CURRENT, CANCEL_NOW, CARD_CANCELLED, CARD_FAILED, CARD_SUCCESS, STATE_CANCELLED,
    STATE_COMPLETED, STATE_FAILED, STATE_PRINTING, PrintJobHandle, PrintServiceError,
)
from services.render_job import (
    display_name, guess_name_column, prints_back, records_by_id, uses_uv, uses_white,
)
from ui.pages.printer.status import bind_printer_status
from ui.pages.settings.prefs import Prefs
from ui.shell import PAGE_DESIGN, PAGE_HISTORY, PAGE_LIST, PAGE_PRINT, PAGE_PRINTER, BasePage
from ui.theme import mark_themed
from ui.widgets import SecondaryButton, confirm, label

from . import STATE_DONE, STATE_READY, STATE_RUNNING
from .findings import (
    ACT_EXCLUDE, ACT_FIX_DESIGN, ACT_OPEN_LIST, ACT_OPEN_PRINTER, ACT_RECHECK, ACT_SHOW_PEOPLE, Action,
    Problem, problems_from, warning_signature,
)
from .options_sheet import OptionsSheet, PrintChoices, summary_text
from .preview import PreviewCache, render_card_image
from .ready_view import ReadyView
from .run_view import (
    ROW_CANCELLED, ROW_FAILED, ROW_SKIPPED, ROW_SUCCESS, RunView, SummaryView,
)
from .style import apply_page_style
from .targets import (
    MODE_ALL, MODE_CHECKED, MODE_COPIES, MODE_CURRENT, MODE_PICKED, MODE_RANGES, MODE_UNPRINTED,
    PEOPLE_MODES, Resolution, TargetCounts,
    count_targets, default_mode, resolve_targets,
)
from .text import (
    brief, duration_text, pill_state, plain, stage_sentence, status_sentence, supplies_lines, supply_text,
)

log = logging.getLogger(__name__)

ACT_EXCLUDE_PRINTED = "exclude_printed"
ACT_OPEN_DESIGN = "open_design"


def _is_blank(document) -> bool:
    """아무것도 놓지 않았고 바탕도 없는 카드."""
    return all(not document.side(key).items and not document.side(key).background
               for key in document.active_side_keys())
_RUN_PREVIEW_DPI = 110.0


class PrintPage(BasePage):
    """workspace: ui.pages.workspace.Workspace, services: ui.pages.services.AppServices."""

    showPeopleRequested = Signal(list)       # 명단 화면이 이 사람들만 보여 주면 된다
    stateChanged = Signal(str)

    def __init__(self, workspace, services, parent: QWidget | None = None, prefs: Prefs | None = None) -> None:
        super().__init__("인쇄", parent)
        self.workspace = workspace
        self.services = services
        self.prefs = prefs or Prefs.instance()
        self.confirm: Callable[..., bool] = confirm      # 테스트에서 바꿔 끼운다

        self._choices = PrintChoices(self.prefs.print_options())
        self._defaults = self.prefs.print_options()      # 마지막으로 본 설정 화면의 기본값
        self._copy_count = 1
        self._mode: str | None = None
        self._range_text = ""
        self._picked_ids: list[int] = []
        self._excluded: set[int] = set()
        self._custom_order: list[int] | None = None
        self._ids: list[int] = []
        self._resolution = Resolution()
        self._counts = TargetCounts()
        self._records: dict[int, dict] = {}
        self._name_column: str | None = None
        self._extra_columns: list[str] = []
        self._position = 0
        self._side = "front"
        self._report: preflight.PreflightReport | None = None
        self._problems: list[Problem] = []
        self._acknowledged: tuple | None = None
        self._cache = PreviewCache()
        self._entered = True
        self._targets_dirty = True

        self._state = STATE_READY
        self._job: PrintJobHandle | None = None
        self._job_document = None
        self._job_store = None
        self._job_started_at = 0.0
        self._last_card_at = 0.0
        self._durations: list[float] = []
        self._supplies_start = None        # 인쇄를 시작할 때의 프린터 상태
        self._job_sides = 1
        self._shown_index = 0
        self._last_error = ""

        self._waiting: list[dict] = []     # 인쇄 중에 걸어 둔 다음 작업들
        self._composing = False            # 인쇄 중에 다음 작업을 고르는 중

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.queue_bar = QWidget(self)
        self.queue_bar.setObjectName("QueueBar")
        bar = QHBoxLayout(self.queue_bar)
        bar.setContentsMargins(24, 10, 24, 10)
        bar.setSpacing(12)
        self.queue_label = label("", "secondary")
        self.queue_add_button = SecondaryButton("다음 작업 걸어 두기", size="sm")
        self.queue_add_button.setToolTip("지금 인쇄가 끝나면 이어서 뽑을 카드를 미리 골라 둡니다.")
        self.queue_add_button.clicked.connect(self.compose_next)
        self.queue_back_button = SecondaryButton("인쇄 중인 화면으로", size="sm")
        self.queue_back_button.clicked.connect(self.back_to_run)
        self.queue_cancel_button = SecondaryButton("걸어 둔 작업 빼기", size="sm")
        self.queue_cancel_button.clicked.connect(lambda: self.cancel_waiting())
        bar.addWidget(self.queue_label, 1)
        bar.addWidget(self.queue_cancel_button)
        bar.addWidget(self.queue_back_button)
        bar.addWidget(self.queue_add_button)
        layout.addWidget(self.queue_bar)
        self.queue_bar.hide()
        self.stack = QStackedWidget(self)
        layout.addWidget(self.stack)
        self.ready = ReadyView()
        self.run = RunView()
        self.stack.addWidget(self.ready)
        self.stack.addWidget(self.run)
        apply_page_style(self)

        self._targets_timer = self._timer(40, self._recompute)
        self._preflight_timer = self._timer(180, self._run_preflight)
        self._preview_timer = self._timer(30, self._update_preview)
        self._notify_timer = self._timer(900, self._notify_workspace)
        self._run_preview_timer = self._timer(0, self._update_run_preview)

        self._connect()
        self._recompute()
        self._refresh_printer()

    # ------------------------------------------------------------------ 다음 작업 걸어 두기
    def _is_ready(self) -> bool:
        """인쇄할 것을 고르는 중인가. 인쇄 도중에 다음 작업을 고르는 때도 포함한다."""
        return self._state == STATE_READY or self._composing

    def waiting_jobs(self) -> list[PrintJobHandle]:
        return [entry["handle"] for entry in self._waiting]

    def compose_next(self) -> bool:
        """인쇄가 도는 동안 다음에 뽑을 것을 고르는 화면으로 간다."""
        if self._state != STATE_RUNNING or self._job is None or not self._job.is_active:
            return False
        self._composing = True
        self.stack.setCurrentWidget(self.ready)
        self.refresh_now()
        self._refresh_queue_bar()
        return True

    def back_to_run(self) -> None:
        if self._composing:
            self._composing = False
            self.stack.setCurrentWidget(self.run)
            self._refresh_queue_bar()

    def cancel_waiting(self, handle: PrintJobHandle | None = None) -> bool:
        """걸어 둔 작업을 뺀다. 지정하지 않으면 마지막에 건 것."""
        if not self._waiting:
            return False
        entry = next((e for e in self._waiting if e["handle"] is handle), self._waiting[-1])
        self._waiting.remove(entry)
        self.services.printing.cancel(entry["handle"], "now")
        self._toast("걸어 둔 작업을 뺐습니다.", "info")
        self._refresh_queue_bar()
        return True

    def _refresh_queue_bar(self) -> None:
        active = self._state == STATE_RUNNING and self._job is not None and self._job.is_active
        self.queue_bar.setVisible(bool(active))
        if not active:
            return
        count = len(self._waiting)
        cards = sum(entry["handle"].total for entry in self._waiting)
        if self._composing:
            text = "지금 인쇄가 끝나면 이어서 뽑을 카드를 고르세요."
        elif count:
            text = f"다음에 뽑을 작업 {count:,}개({cards:,}장)가 기다리고 있습니다."
        else:
            text = "인쇄하는 동안 다음에 뽑을 카드를 미리 걸어 둘 수 있습니다."
        self.queue_label.setText(text)
        self.queue_add_button.setVisible(not self._composing)
        self.queue_back_button.setVisible(self._composing)
        self.queue_cancel_button.setVisible(count > 0 and not self._composing)

    def _timer(self, interval: int, slot) -> QTimer:
        timer = QTimer(self)
        timer.setSingleShot(True)
        timer.setInterval(interval)
        timer.timeout.connect(slot)
        return timer

    # ------------------------------------------------------------------ 틀과 잇기
    @property
    def shell(self):
        return self._shell

    @shell.setter
    def shell(self, value) -> None:
        self._shell = value
        if value is not None:
            bind_printer_status(value, self.services)

    def _connect(self) -> None:
        ws, view, run = self.workspace, self.ready, self.run
        for signal in (ws.store_changed, ws.columns_changed, ws.records_changed):
            signal.connect(self._on_list_changed)
        ws.current_record_changed.connect(self._on_list_changed)
        view_changed = getattr(ws, "view_changed", None)
        if view_changed is not None:
            view_changed.connect(self._on_list_changed)
        ws.checked_ids_changed.connect(self._on_checked_changed)
        editor = ws.editor
        for signal in (editor.document_changed, editor.items_changed, editor.properties_changed,
                       editor.card_changed):
            signal.connect(self._on_design_changed)

        printers, printing = self.services.printers, self.services.printing
        for signal in (printers.devices_changed, printers.current_changed, printers.status_changed):
            signal.connect(self._on_printer_changed)
        printing.job_queued.connect(self._on_printer_changed)
        printing.progress.connect(self._on_progress)
        printing.card_result.connect(self._on_card_result)
        printing.finished.connect(self._on_finished)
        printing.error.connect(self._on_error)
        self.prefs.changed.connect(self._on_prefs_changed)

        view.modeChanged.connect(self.set_mode)
        view.rangeEdited.connect(self._on_range_edited)
        view.copiesChanged.connect(self._on_copies_changed)
        view.peopleEdited.connect(self._on_people_edited)
        view.restoreRequested.connect(self.restore_excluded)
        view.personSelected.connect(self.show_position)
        view.stepRequested.connect(lambda delta: self.show_position(self._position + delta))
        view.sideChanged.connect(self._on_side_changed)
        view.optionsRequested.connect(self.open_options)
        view.printerChangeRequested.connect(self._change_printer)
        view.findingAction.connect(self._on_finding_action)
        view.acknowledgeToggled.connect(self._on_acknowledge)
        view.startRequested.connect(self.start_printing)

        run.stopAfterRequested.connect(self.stop_after_current)
        run.stopNowRequested.connect(self.stop_now)
        run.retryRequested.connect(self.retry_failed)
        run.continueRequested.connect(self.continue_remaining)
        run.historyRequested.connect(self._open_history)
        run.backRequested.connect(self.back_to_ready)

    def on_enter(self) -> None:
        self._entered = True
        if self._is_ready():
            self.refresh_now()
            self.services.printers.poll_now()

    def on_leave(self) -> None:
        self._entered = False

    def can_close(self) -> bool:
        if not self.services.printing.is_printing:
            return True
        return bool(self.confirm(
            self, "인쇄 중인데 닫을까요?",
            "지금 닫으면 남은 카드는 뽑지 않습니다. 프린터 안에 이미 들어간 카드는 끝까지 찍혀 나옵니다.",
            confirm_text="인쇄 멈추고 닫기", cancel_text="계속 인쇄", danger=True))

    def state(self) -> str:
        return self._state

    def job(self) -> PrintJobHandle | None:
        return self._job

    def _navigate(self, page_id: str) -> None:
        if self._shell is not None:
            self._shell.navigate(page_id)

    def _toast(self, text: str, kind: str = "info") -> None:
        if self._shell is not None:
            self._shell.show_toast(text, kind)

    # ------------------------------------------------------------------ 누구를 뽑을지
    def _on_list_changed(self, *_args) -> None:
        if self._is_ready():
            self._targets_dirty = True
            self._targets_timer.start()
            self._refresh_start()

    def _on_checked_changed(self, ids: list) -> None:
        if not self._is_ready():
            return
        if ids:
            self._mode = MODE_CHECKED
            self._reset_edits()
        self._on_list_changed()

    def _on_design_changed(self, *_args) -> None:
        self._cache.clear()
        if self._is_ready():
            self._targets_timer.start()
            self._refresh_start()

    def _on_prefs_changed(self) -> None:
        """설정 화면에서 기본값이 바뀌었다. 바뀐 값만, 이번 인쇄에서 따로 고치지 않은 것에만 옮긴다.

        (보관 기간처럼 인쇄와 상관없는 설정을 바꿔도, 이번 인쇄에 정해 둔 장수·위치는 그대로다.
        인쇄 중인 작업은 시작할 때 설정을 복사해 갔으므로 영향을 받지 않는다.)
        """
        defaults, before = self.prefs.print_options(), self._defaults
        self._defaults = defaults
        options = self._choices.options
        changed = False
        for name in ("copies", "stop_on_error", "offset_x_mm", "offset_y_mm"):
            old, new = getattr(before, name), getattr(defaults, name)
            if old != new and getattr(options, name) == old:
                setattr(options, name, new)
                changed = True
        if changed and self._is_ready():
            self._targets_timer.start()

    def _reset_edits(self) -> None:
        self._excluded.clear()
        self._custom_order = None
        self._position = 0

    def set_mode(self, mode: str) -> None:
        if mode == self._mode:
            return
        self._mode = mode
        self._reset_edits()
        self._recompute(True)

    def mode(self) -> str | None:
        return self._mode

    def set_range_text(self, text: str) -> None:
        self.ready.set_range_text(text)
        self._on_range_edited(text)

    def _on_range_edited(self, text: str) -> None:
        self._range_text = text
        self._reset_edits()
        self._recompute(True)

    def _on_copies_changed(self, copies: int) -> None:
        self._copy_count = max(1, int(copies))
        self._recompute(True)

    def print_people(self, record_ids: list[int]) -> None:
        """다른 화면(기록 등)이 뽑을 사람을 넘겨줄 때 부른다."""
        self._picked_ids = [int(i) for i in record_ids]
        self._mode = MODE_PICKED
        self._reset_edits()
        if self._state == STATE_DONE:
            self.back_to_ready()
        else:
            self._recompute(True)

    def exclude(self, record_ids) -> None:
        self._excluded |= {int(i) for i in record_ids}
        self._recompute(True)

    def restore_excluded(self) -> None:
        self._excluded.clear()
        self._recompute(True)

    def _on_people_edited(self, ids: list) -> None:
        ids = [int(i) for i in ids]
        removed = set(self._ids) - set(ids)
        self._excluded |= removed
        self._custom_order = ids
        if removed or ids != self.ready.people_ids():
            self._recompute(True)
        else:
            self._ids = ids
            self._preview_timer.start()
            self._preflight_timer.start()
            self._refresh_start()

    def target_ids(self) -> list[int]:
        return list(self._ids)

    def is_people_mode(self) -> bool:
        return self._mode in PEOPLE_MODES and self.workspace.store is not None

    def card_count(self) -> int:
        if self.is_people_mode():
            return len(self._ids) * max(1, self._choices.options.copies)
        return self._copy_count

    def _available_modes(self, counts: TargetCounts, has_list: bool) -> set[str]:
        modes: set[str] = set()
        if counts.copies_available:
            modes.add(MODE_COPIES)
        if has_list:
            if counts.checked:
                modes.add(MODE_CHECKED)
            if counts.unprinted:
                modes.add(MODE_UNPRINTED)
            if counts.current:
                modes.add(MODE_CURRENT)
            if counts.visible:
                modes |= {MODE_RANGES, MODE_ALL}
            if counts.picked:
                modes.add(MODE_PICKED)
        return modes

    def _recompute(self, user_action: bool = False) -> None:
        """대상을 다시 센다. user_action이면 미리 보기를 미루지 않고 바로 바꾼다."""
        self._targets_timer.stop()
        self._targets_dirty = False
        ws = self.workspace
        store, document = ws.store, ws.editor.document
        self._name_column = guess_name_column(store)
        self._extra_columns = self._pick_extras(store)
        counts = count_targets(store, document, ws.checked_ids, ws.current_record_id, ws.view_filter,
                               self._picked_ids, self._name_column)
        self._counts = counts
        has_list = store is not None
        available = self._available_modes(counts, has_list)
        if self._mode not in available:
            self._mode = default_mode(counts, store)
            if self._mode not in available and available:
                self._mode = MODE_COPIES if MODE_COPIES in available else sorted(available)[0]
            self._reset_edits()

        self._resolution = resolve_targets(
            self._mode, store, checked_ids=ws.checked_ids, current_id=ws.current_record_id,
            range_text=self._range_text, view_filter=ws.view_filter, view_order=ws.view_order,
            picked_ids=self._picked_ids)
        base = self._resolution.ids
        self._excluded &= set(base)
        ids = [i for i in base if i not in self._excluded]
        if self._custom_order is not None and set(self._custom_order) == set(ids):
            ids = list(self._custom_order)
        else:
            self._custom_order = None
        self._ids = ids
        self._records = records_by_id(store, ids) if (store is not None and ids) else {}
        self._position = max(0, min(self._position, len(ids) - 1)) if ids else 0

        view = self.ready
        view.set_mode(self._mode)
        view.set_counts(counts, has_list, self._copy_count)
        view.set_range_error(self._resolution.error if self._mode == MODE_RANGES and self._range_text.strip() else "")
        self._fill_people()
        self._refresh_stats()
        self._refresh_options_summary()
        if user_action:
            self._update_preview()
        else:
            self._preview_timer.start()
        self._preflight_timer.start()
        self._refresh_start()

    def _pick_extras(self, store) -> list[str]:
        """이름 옆에 덧붙여 보여 줄 항목 둘: 바코드에 넣는 번호 같은 것은 빼고 부서·직급 같은 글자."""
        if store is None:
            return []
        name = guess_name_column(store)
        codes: set[str] = set()
        for _, item in self.workspace.editor.document.iter_items():
            if item.kind in ("barcode", "qr") and item.source is not None:
                codes |= set(item.source.columns())
        candidates = [c.name for c in store.columns() if c.name != name and c.type in ("text", "list")]
        picked = [c for c in candidates if c not in codes] or candidates
        return picked[:2]

    def _person_name(self, record_id: int | None, position: int) -> str:
        record = self._records.get(record_id) if record_id is not None else None
        return display_name(record, self._name_column, f"{position + 1}번째 사람")

    def _person_sub(self, record_id: int | None) -> str:
        record = self._records.get(record_id) if record_id is not None else None
        if record is None:
            return ""
        values = [str(record.get(column) or "").strip() for column in self._extra_columns]
        return " ".join(v for v in values if v)

    def _tags(self) -> dict[int, tuple[str, str]]:
        tags: dict[int, tuple[str, str]] = {}
        for record_id in self._ids:
            record = self._records.get(record_id)
            if record is not None and record.get("_printed"):
                tags[record_id] = ("뽑은 적 있음", "muted")
        if self._report is not None:
            for finding in self._report.by_code("missing_photo"):
                for record_id in finding.record_ids:
                    tags[record_id] = ("사진 없음", "warn")
        return tags

    def _fill_people(self) -> None:
        tags = self._tags()
        rows = []
        for position, record_id in enumerate(self._ids):
            tag, tone = tags.get(record_id, ("", ""))
            rows.append({"id": record_id, "name": self._person_name(record_id, position),
                         "sub": self._person_sub(record_id), "tag": tag, "tone": tone})
        if self._mode == MODE_COPIES or self.workspace.store is None:
            empty = "명단 없이 같은 카드를 여러 장 뽑습니다."
        elif self._resolution.error:
            empty = self._resolution.error
        else:
            empty = "뽑을 사람이 없습니다. 위에서 다른 방법으로 골라 보세요."
        self.ready.set_people(rows, len(self._excluded), empty)
        self.ready.select_position(self._position)

    def _missing_photo_ids(self) -> set[int]:
        if self._report is None:
            return set()
        return {i for f in self._report.by_code("missing_photo") for i in f.record_ids}

    def _printed_target_ids(self) -> list[int]:
        return [i for i in self._ids if (self._records.get(i) or {}).get("_printed")]

    def _refresh_stats(self) -> None:
        people = self.is_people_mode()
        if not people:
            self.ready.set_stats(f"{self._copy_count:,}장", "–", "이미 뽑은 사람", "–", "muted", "muted",
                                 caption="뽑을 카드")
            return
        missing = len(self._missing_photo_ids() & set(self._ids))
        if self._mode == MODE_UNPRINTED:
            printed, caption, tone = self._counts.printed, "이미 뽑아서 뺌", "muted"
        else:
            printed = len(self._printed_target_ids())
            caption, tone = "이미 뽑은 사람", ("warn" if printed else "muted")
        self.ready.set_stats(f"{len(self._ids):,}명", f"{printed:,}명", caption, f"{missing:,}명",
                             tone, "warn" if missing else "muted")

    # ------------------------------------------------------------------ 미리 보기
    def show_position(self, position: int) -> None:
        if not self._ids:
            return
        position = max(0, min(int(position), len(self._ids) - 1))
        self._position = position
        self.ready.select_position(position)
        self._update_preview()

    def _on_side_changed(self, side: str) -> None:
        self._side = side
        self._update_preview()

    def _update_preview(self) -> None:
        self._preview_timer.stop()
        document, store = self.workspace.editor.document, self.workspace.store
        dual = bool(document.dual_side) and "back" in document.sides
        if not dual:
            self._side = "front"
        self.ready.set_sides(dual, self._side)
        people = self.is_people_mode()
        record_id = self._ids[self._position] if (people and self._ids) else None
        width, height = document.side_size_mm(self._side)
        self.ready.preview.set_aspect(width - 2 * document.bleed_mm, height - 2 * document.bleed_mm)
        self.ready.fit_preview()
        if people and not self._ids:
            self.ready.preview.set_image(None)
            self.ready.preview.set_placeholder("왼쪽에서 뽑을 사람을 고르면\n카드가 여기에 보입니다")
            self.ready.set_preview_caption("뽑을 사람을 골라 주세요", "", 0, 0)
            return
        key = (record_id, self._side, self._position if record_id is None else 0)
        image = self._cache.get(key)
        if image is None:
            image = render_card_image(document, store if people else None, record_id, self._side,
                                      position=self._position)
            if image is not None:
                self._cache.put(key, image)
        self.ready.preview.set_image(image)
        self.ready.preview.set_placeholder("이 카드를 그리지 못했습니다.\n디자인 화면에서 확인해 주세요")
        if people:
            self.ready.set_preview_caption(self._person_name(record_id, self._position),
                                           self._person_sub(record_id), self._position, len(self._ids))
        else:
            self.ready.set_preview_caption(document.name or "카드", "모든 카드가 이 모습으로 나옵니다", 0, 1)

    # ------------------------------------------------------------------ 프린터
    def _on_printer_changed(self, *_args) -> None:
        self._refresh_printer()
        if self._is_ready():
            self._refresh_options_summary()
            self._preflight_timer.start()

    def _refresh_printer(self) -> None:
        printers = self.services.printers
        device, status = printers.current(), printers.status()
        state, pill, _detail = pill_state(device, status, self.services.printing.is_printing,
                                          printers.is_busy and not printers.devices())
        short = {"ready": "준비됨", "busy": "인쇄 중", "error": "확인 필요", "offline": "연결 안 됨"}[state]
        if device is not None and status is None:
            short = "확인 중"
        supplies = supplies_lines(status)
        detail = status_sentence(status) if device is not None else \
            "전원과 USB 선을 확인한 뒤 프린터 화면에서 다시 찾아 주세요."
        self.ready.set_printer(device.name if device else "쓸 프린터가 없습니다", state, short, detail,
                               "\n".join(supplies),
                               len(printers.devices()) > 1)

    def _change_printer(self) -> None:
        printers = self.services.printers
        devices = printers.devices()
        if len(devices) <= 1:
            self._navigate(PAGE_PRINTER)
            return
        menu = QMenu(self)
        mark_themed(menu)
        current = printers.current()
        for device in devices:
            action = menu.addAction(device.name + ("  (지금 쓰는 프린터)" if device == current else ""))
            action.triggered.connect(lambda _checked=False, d=device: printers.select(d))
        menu.addSeparator()
        menu.addAction("프린터 화면 열기").triggered.connect(lambda: self._navigate(PAGE_PRINTER))
        menu.exec(self.ready.change_printer.mapToGlobal(self.ready.change_printer.rect().bottomLeft()))

    # ------------------------------------------------------------------ 인쇄 설정
    def choices(self) -> PrintChoices:
        return self._choices

    def _capabilities(self) -> set[str]:
        return self.services.printers.capabilities()

    def _refresh_options_summary(self) -> None:
        document = self.workspace.editor.document
        capabilities = self._capabilities() if self.services.printers.current() else \
            {CAP_DUAL_SIDE, CAP_WHITE, CAP_UV, CAP_MAGNETIC}
        self.ready.set_options_summary(summary_text(self._choices, document, capabilities, self.is_people_mode()))

    def make_options_sheet(self) -> OptionsSheet:
        people = self.is_people_mode()
        choices = self._choices.clone()
        if not people:
            choices.options.copies = self._copy_count
        return OptionsSheet(choices, self.workspace.editor.document, self._capabilities(),
                            self.workspace.column_names(), people, self)

    def open_options(self) -> None:
        sheet = self.make_options_sheet()
        if sheet.exec() == QDialog.Accepted:
            self.apply_choices(sheet.choices())

    def apply_choices(self, choices: PrintChoices) -> None:
        if not self.is_people_mode():
            self._copy_count = max(1, choices.options.copies)
            choices.options.copies = self._choices.options.copies
        self._choices = choices
        self.prefs.blockSignals(True)
        self.prefs.set_offsets(choices.options.offset_x_mm, choices.options.offset_y_mm)
        self.prefs.blockSignals(False)
        self._defaults.offset_x_mm, self._defaults.offset_y_mm = self.prefs.offsets()
        self._recompute(True)

    # ------------------------------------------------------------------ 인쇄 전 확인
    def _job_options(self):
        options = copy.deepcopy(self._choices.options)
        if not self.is_people_mode():
            options.copies = self._copy_count
        if not options.job_name:
            options.job_name = self.workspace.editor.document.name
        return options

    def _run_preflight(self) -> None:
        self._preflight_timer.stop()
        if not self._is_ready():
            return
        ws, printers = self.workspace, self.services.printers
        document = self._choices.document_for(ws.editor.document)
        people = self.is_people_mode()
        problems: list[Problem] = []
        self._report = None
        device, status = printers.current(), printers.status()
        if device is None:
            problems.append(Problem(
                "no_printer", "error",
                "쓸 프린터가 없습니다. 프린터 전원과 USB 선을 확인한 뒤 프린터 화면에서 다시 찾아 주세요.",
                [Action("프린터 화면 열기", ACT_OPEN_PRINTER)]))
        if _is_blank(ws.editor.document):
            problems.insert(0, Problem(         # 카드 꾸미기가 먼저다
                "blank_design", "error",
                "카드에 아직 아무것도 없습니다. 디자인 화면에서 카드를 먼저 꾸며 주세요.",
                [Action("카드 꾸미러 가기", ACT_OPEN_DESIGN)]))
        if people and not self._ids:
            text = self._resolution.error or "뽑을 사람이 없습니다. 왼쪽에서 뽑을 사람을 고르세요."
            problems.append(Problem("no_targets", "error", text))
        else:
            options = self._job_options()
            options.capabilities = self._capabilities() if device is not None else None
            try:
                self._report = preflight.check(
                    document, ws.store if people else None, self._ids if people else None, device, status,
                    capabilities=options.capabilities, options=options)
                problems += problems_from(self._report)
            except Exception as exc:
                log.exception("인쇄 전 확인 실패")
                problems.append(Problem("preflight_failed", "error",
                                        f"인쇄 전 확인을 끝내지 못했습니다. 디자인과 명단을 다시 확인해 주세요. ({exc})"))
            printed = self._printed_target_ids() if people and self._mode != MODE_UNPRINTED else []
            if printed:
                problems.append(Problem(
                    "already_printed", "warn",
                    f"이 중 {len(printed):,}명은 이미 뽑은 적이 있습니다. 그대로 인쇄하면 카드가 한 장 더 나옵니다.",
                    [Action(f"이미 뽑은 {len(printed):,}명 빼기", ACT_EXCLUDE, printed)], printed))
        if device is not None and status is None and not self.services.printing.is_printing:
            problems.append(Problem("status_unknown", "idle",
                                    "프린터 상태를 확인하고 있습니다. 카드와 리본이 들어 있는지 직접 봐 주세요."))
        problems.sort(key=lambda p: {"error": 0, "warn": 1}.get(p.tone, 2))
        self._problems = problems
        self.ready.set_problems(problems)
        self.ready.update_tags(self._tags())
        self._refresh_stats()
        self._refresh_start()

    def problems(self) -> list[Problem]:
        return list(self._problems)

    def report(self) -> preflight.PreflightReport | None:
        return self._report

    def _on_acknowledge(self, checked: bool) -> None:
        self._acknowledged = warning_signature(self._problems) if checked else None
        self._refresh_start()

    def acknowledge_warnings(self, checked: bool = True) -> None:
        self.ready.acknowledge.setChecked(checked)
        self._on_acknowledge(checked)

    def blocking_reason(self) -> str:
        errors = [p for p in self._problems if p.tone == "error"]
        if errors:
            return f"고쳐야 할 것 {len(errors)}개를 먼저 해결해야 인쇄할 수 있습니다."
        if self._preflight_timer.isActive() or self._targets_timer.isActive():
            return "인쇄해도 되는지 확인하고 있습니다."
        warnings = warning_signature(self._problems)
        if warnings and self._acknowledged != warnings:
            return "알아 둘 것을 읽고 위의 확인란에 체크하면 인쇄할 수 있습니다."
        return ""

    def _refresh_start(self) -> None:
        warnings = warning_signature(self._problems)
        acknowledged = bool(warnings) and self._acknowledged == warnings
        self.ready.set_acknowledge(bool(warnings) and not any(p.tone == "error" for p in self._problems),
                                   f"알아 둘 것 {len(warnings)}개를 읽었습니다. 그대로 인쇄합니다.", acknowledged)
        reason = self.blocking_reason()
        cards = self.card_count()
        if self._composing:
            text = f"{cards:,}장 다음 작업으로 걸기" if cards else "다음 작업으로 걸기"
        else:
            text = f"{cards:,}장 인쇄 시작" if cards else "인쇄 시작"
        tone = "error" if any(p.tone == "error" for p in self._problems) else ("warn" if reason else "")
        self.ready.set_start(text, not reason and cards > 0, reason, tone)

    def _on_finding_action(self, key: str, payload: Any) -> None:
        if key == ACT_EXCLUDE:
            self.exclude(payload or [])
            self.refresh_now()
        elif key == ACT_SHOW_PEOPLE:
            self.show_people_in_list(list(payload or []))
        elif key == ACT_FIX_DESIGN:
            item_ids, side = payload
            editor = self.workspace.editor
            if side and side != editor.current_side:
                editor.set_side(side)
            editor.select(item_ids)
            self._navigate(PAGE_DESIGN)
        elif key == ACT_OPEN_DESIGN:
            self._navigate(PAGE_DESIGN)
        elif key == ACT_OPEN_LIST:
            self._navigate(PAGE_LIST)
        elif key == ACT_OPEN_PRINTER:
            self._navigate(PAGE_PRINTER)
        elif key == ACT_RECHECK:
            if not self.services.printers.poll_now():
                self._toast("프린터가 다른 일을 하고 있습니다. 잠시 뒤에 다시 눌러 주세요.")

    def show_people_in_list(self, record_ids: list[int]) -> None:
        """명단 화면으로 가서 이 사람들만 보여 준다(명단 화면에 show_people이 있으면 그것을 쓴다)."""
        self.showPeopleRequested.emit(list(record_ids))
        if record_ids:
            self.workspace.set_current_record(record_ids[0])
        if self._shell is None:
            return
        try:
            page = self._shell.page(PAGE_LIST)
        except KeyError:
            page = None
        self._shell.navigate(PAGE_LIST)
        hook = getattr(page, "show_people", None)
        if callable(hook):
            hook(list(record_ids))

    def refresh_now(self) -> None:
        """미뤄 둔 계산(대상, 미리 보기, 인쇄 전 확인)을 지금 끝낸다."""
        if not self._is_ready():
            return
        self._recompute()
        self._update_preview()
        self._run_preflight()

    # ------------------------------------------------------------------ 인쇄 시작
    def start_printing(self) -> PrintJobHandle | None:
        if not self._is_ready():
            return None
        self._run_preflight()
        if self.blocking_reason() or not self.card_count():
            return None
        ws, printers = self.workspace, self.services.printers
        people = self.is_people_mode()
        document = self._choices.document_for(ws.editor.document)
        options = self._job_options()
        options.capabilities = self._choices.capabilities_for(self._capabilities())
        device = printers.current()
        status = printers.status()
        context = {
            "supplies": status,
            "sides": 2 if (prints_back(document) and CAP_DUAL_SIDE in options.capabilities) else 1,
            "document": copy.deepcopy(document),
            "store": ws.store if people else None,
        }
        queueing = self._composing
        if not queueing:
            self._apply_job_context(context)
        job_store = context["store"]
        rows = self._rows_for_ids(self._ids, options.copies) if people else \
            [{"name": f"{i + 1}번째 카드", "record_id": None, "position": i} for i in range(options.copies)]
        title = f"이번에 뽑는 {len(self._ids):,}명" if people else f"이번에 뽑는 {options.copies:,}장"
        try:
            handle = self.services.printing.start(
                document, job_store, list(self._ids) if people else None, device, options)
        except PrintServiceError as exc:
            self._toast(plain(str(exc)), "error")
            self._problems.insert(0, Problem("start_failed", "error", plain(str(exc))))
            self.ready.set_problems(self._problems)
            self._refresh_start()
            return None
        if queueing:
            self._waiting.append({"handle": handle, "title": title, "rows": rows, "context": context})
            self._composing = False
            self.stack.setCurrentWidget(self.run)
            self._refresh_queue_bar()
            self._toast(f"{handle.total:,}장을 다음 작업으로 걸어 두었습니다. 지금 인쇄가 끝나면 이어서 뽑습니다.", "ok")
            return handle
        self._begin_run(handle, title, rows)
        return handle

    def _apply_job_context(self, context: dict) -> None:
        self._supplies_start = context["supplies"]
        self._job_sides = context["sides"]
        self._job_document = context["document"]
        self._job_store = context["store"]

    def _rows_for_ids(self, ids: list[int | None], copies: int = 1) -> list[dict]:
        rows = []
        for position, record_id in enumerate(ids):
            name = self._person_name(record_id, position) if record_id is not None else f"{position + 1}번째 카드"
            record = self._records.get(record_id) if record_id is not None else None
            extras = {f"extra{i}": str((record or {}).get(column) or "") for i, column in enumerate(self._extra_columns)}
            for copy_index in range(max(1, copies)):
                suffix = f" ({copy_index + 1}/{copies})" if copies > 1 else ""
                rows.append({"name": name + suffix, "record_id": record_id, "position": position, **extras})
        return rows

    def _ribbon_panels(self) -> tuple[str, ...]:
        document = self._job_document
        panels = ["y", "m", "c", "k"]
        if document is not None and uses_white(document):
            panels.append("w")
        if document is not None and uses_uv(document):
            panels.append("uv")
        return tuple(panels)

    def _begin_run(self, handle: PrintJobHandle, title: str, rows: list[dict]) -> None:
        self._job = handle
        self._last_error = ""
        self._durations = []
        self._job_started_at = self._last_card_at = time.monotonic()
        self._shown_index = 0
        extras = self._extra_columns if any(r.get("record_id") is not None for r in rows) else []
        self.run.begin(title, rows, extras, self._ribbon_panels())
        self._refresh_run_supplies(0)
        self._set_state(STATE_RUNNING)
        self.stack.setCurrentWidget(self.run)
        self._refresh_queue_bar()
        if handle.state == STATE_PRINTING:
            self.run.set_progress(0, handle.total, 1, "프린터에 연결하고 있습니다")
            self._show_run_card(1)
        self._refresh_printer()

    def _set_state(self, state: str) -> None:
        if state != self._state:
            self._state = state
            self.stateChanged.emit(state)

    # ------------------------------------------------------------------ 인쇄 중
    def _mine(self, job_id: int) -> bool:
        return self._job is not None and self._job.id == job_id

    def _on_progress(self, progress) -> None:
        if not self._mine(progress.job_id) or self._state != STATE_RUNNING:
            return
        self.run.set_progress(progress.done, progress.total, progress.index,
                              stage_sentence(progress.stage_key, progress.stage, progress.side),
                              progress.panel, progress.card_percent)
        self._show_run_card(progress.index)

    def _show_run_card(self, index: int) -> None:
        if index != self._shown_index:
            self._shown_index = index
            self._run_preview_timer.start()

    def _update_run_preview(self) -> None:
        if self._job is None or self._state != STATE_RUNNING:
            return
        row = self._shown_index - 1
        if not 0 <= row < self.run.model.rowCount():
            return
        data = self.run.model.row(row)
        record_id = data.get("record_id")
        # 다시 뽑거나 이어서 뽑을 때도 일련번호는 처음 정한 순서 그대로 보여 준다.
        image = render_card_image(self._job_document, self._job_store, record_id, "front",
                                  dpi=_RUN_PREVIEW_DPI, position=int(data.get("position", row)))
        sub = " ".join(v for v in (data.get("extra0", ""), data.get("extra1", "")) if v)
        self.run.set_current_card(image, data.get("name", ""), sub)

    def _on_card_result(self, result) -> None:
        if not self._mine(result.job_id):
            return
        state = {CARD_SUCCESS: ROW_SUCCESS, CARD_FAILED: ROW_FAILED, CARD_CANCELLED: ROW_CANCELLED}[result.status]
        self.run.set_row_result(result.index - 1, state, brief(result.message))
        now = time.monotonic()
        self._durations.append(now - self._last_card_at)
        self._last_card_at = now
        job = self._job
        remaining = job.total - job.done
        if len(self._durations) >= 2 and remaining > 0:
            recent = self._durations[-8:]
            self.run.set_eta(duration_text(sum(recent) / len(recent) * remaining))
        elif remaining <= 0:
            self.run.set_eta("")
        self.run.count_done.setText(f"{job.done:,}")
        self._refresh_run_supplies(job.success_count)
        if result.ok and result.record_id is not None:
            self._notify_timer.start()

    def _refresh_run_supplies(self, printed: int) -> None:
        status = self._supplies_start
        if status is None:
            self.run.set_supplies("", "", "")
            return
        feeder = "" if status.feeder_empty is None else ("카드 없음" if status.feeder_empty else "카드 있음")
        self.run.set_supplies(
            supply_text(status.ribbon_remaining, status.ribbon_unit, printed * self._job_sides),
            supply_text(status.film_remaining, status.film_unit, printed * self._job_sides), feeder)

    def _notify_workspace(self) -> None:
        self._notify_timer.stop()
        self.workspace.notify_records_changed()

    def _on_error(self, handle, message: str) -> None:
        if self._job is None or handle is not self._job:
            return
        self._last_error = brief(message)
        self._toast(self._last_error, "error")

    def stop_after_current(self) -> bool:
        if self._job is None or self._state != STATE_RUNNING:
            return False
        self.run.set_stopping(CANCEL_AFTER_CURRENT)
        return self.services.printing.cancel(self._job, CANCEL_AFTER_CURRENT)

    def stop_now(self) -> bool:
        if self._job is None or self._state != STATE_RUNNING:
            return False
        if not self.confirm(
                self, "지금 멈출까요?",
                "프린터 안에 이미 들어간 카드는 멈출 수 없어 끝까지 찍혀 나옵니다. "
                "아직 들어가지 않은 카드는 뽑지 않습니다.",
                confirm_text="지금 멈추기", cancel_text="계속 인쇄", danger=True):
            return False
        if self._job is None or not self._job.is_active:
            return False
        self.run.set_stopping(CANCEL_NOW)
        return self.services.printing.cancel(self._job, CANCEL_NOW)

    # ------------------------------------------------------------------ 끝난 뒤
    def _on_finished(self, summary) -> None:
        waiting = next((e for e in self._waiting if e["handle"].id == summary.job_id), None)
        if waiting is not None and not self._mine(summary.job_id):
            self._waiting.remove(waiting)          # 시작하기 전에 빠진 작업
            self._refresh_queue_bar()
            return
        if self._mine(summary.job_id) and self._waiting:
            self._notify_workspace()
            tone = "ok" if summary.state == STATE_COMPLETED else "warn"
            self._toast(f"앞 작업: {summary.total:,}장 중 {summary.success:,}장을 뽑았습니다. "
                        "다음 작업을 이어서 뽑습니다.", tone)
            entry = self._waiting.pop(0)
            self._composing = False
            self._apply_job_context(entry["context"])
            self._begin_run(entry["handle"], entry["title"], entry["rows"])
            return
        if not self._mine(summary.job_id):
            if self._is_ready():
                self._preflight_timer.start()
            return
        job = self._job
        failures = [(r.record_name, brief(r.message)) for r in job.results if r.status == CARD_FAILED]
        left = summary.cancelled        # 뽑지 않고 남은 카드(실패한 카드는 따로 센다)
        total, success, failed = summary.total, summary.success, summary.failed
        view = SummaryView("", failures=failures, retry_count=failed, continue_count=left)
        if left:
            view.continue_text = (f"실패한 카드까지 {failed + left:,}장 이어서 인쇄" if failed
                                  else f"남은 {left:,}장 이어서 인쇄")
        if summary.state == STATE_COMPLETED:
            view.headline = f"{total:,}장을 모두 뽑았습니다"
            view.detail = "명단에 뽑은 날짜를 적어 두었습니다." if self._job_store is not None else ""
        elif summary.state == STATE_CANCELLED:
            view.tone = "idle"
            view.headline = "인쇄를 멈췄습니다"
            view.detail = (f"{success:,}장은 뽑았고 {left:,}장은 뽑지 않았습니다."
                           + (f" {failed:,}장은 실패했습니다." if failed else ""))
        elif summary.state == STATE_FAILED:
            view.tone = "error"
            view.headline = "카드를 한 장도 뽑지 못했습니다"
            view.detail = self._last_error or brief(summary.message) or "프린터를 확인한 뒤 다시 해 보세요."
        else:
            view.tone = "warn"
            view.headline = f"{total:,}장 중 {success:,}장을 뽑았습니다"
            view.detail = (f"{failed:,}장이 실패했습니다."
                           + (f" 남은 {left:,}장은 뽑지 않고 멈췄습니다." if left else ""))
        if failed:
            view.note = ("실패한 카드는 빈 카드나 반쯤 찍힌 카드로 나왔을 수 있으니 꺼내서 버려 주세요. "
                         "프린터에 오류 표시가 있으면 먼저 해결한 뒤 다시 인쇄하세요.")
        leftover = ROW_CANCELLED if summary.state == STATE_CANCELLED else ROW_SKIPPED
        self.run.finish(view, success, failed, left, leftover)
        self._composing = False
        self._set_state(STATE_DONE)
        self._refresh_queue_bar()
        self._notify_workspace()
        if self._shell is not None:
            if summary.state == STATE_COMPLETED:
                self._shell.set_step_done(PAGE_PRINT, True)
                self._toast(f"{total:,}장을 모두 뽑았습니다.", "ok")
            elif not self._entered:
                self._toast("인쇄가 끝났습니다. 인쇄 화면에서 결과를 확인하세요.", "warn")
        self._refresh_printer()
        QTimer.singleShot(400, self.services.printers.poll_now)

    def _restart(self, include_unprinted: bool) -> PrintJobHandle | None:
        if self._job is None or self._state != STATE_DONE:
            return None
        previous = self._job
        device = self.services.printers.current() or previous.device
        try:
            handle = self.services.printing.retry_failed(previous, include_unprinted=include_unprinted,
                                                         device=device)
        except PrintServiceError as exc:
            self._toast(plain(str(exc)), "error")
            return None
        indices = set(previous.failed_indices) | (set(previous.unprinted_indices) if include_unprinted else set())
        old_rows = self.run.model.rows()
        rows = [{k: v for k, v in old_rows[i].items() if k in ("name", "record_id", "position", "extra0", "extra1")}
                for i in sorted(indices) if i < len(old_rows)]
        self._supplies_start = self.services.printers.status() or self._supplies_start
        self.services.printing.forget(previous)
        title = f"이어서 뽑는 {len(rows):,}장" if include_unprinted else f"다시 뽑는 {len(rows):,}장"
        self._begin_run(handle, title, rows)
        return handle

    def retry_failed(self) -> PrintJobHandle | None:
        return self._restart(False)

    def continue_remaining(self) -> PrintJobHandle | None:
        return self._restart(True)

    def back_to_ready(self) -> None:
        if self._state == STATE_RUNNING:
            return
        if self._job is not None:
            self.services.printing.forget(self._job)
        self._job = None
        self._job_document = None
        self._job_store = None
        self._acknowledged = None
        if self._mode != MODE_PICKED:
            self._reset_edits()
        self._set_state(STATE_READY)
        self.stack.setCurrentWidget(self.ready)
        self._refresh_printer()
        self.refresh_now()

    def _open_history(self) -> None:
        if self._shell is None:
            return
        try:
            page = self._shell.page(PAGE_HISTORY)
        except KeyError:
            page = None
        self._shell.navigate(PAGE_HISTORY)
        history_id = self._job.history_id if self._job is not None else None
        show_job = getattr(page, "show_job", None)
        if callable(show_job) and history_id is not None and show_job(history_id):
            return
        hook = getattr(page, "show_latest", None)
        if callable(hook):
            hook()
