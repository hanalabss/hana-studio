"""기록 화면: 언제, 어떤 카드를, 몇 장 뽑았는지. 한 번의 인쇄를 고르면 사람별 결과가 보인다."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QButtonGroup, QFrame, QHBoxLayout, QLabel, QSizePolicy, QStackedWidget, QVBoxLayout, QWidget,
)

from data.filters import quick_search
from printer.history import (
    CARD_FAILED, CARD_SUCCESS, JOB_CANCELLED, JOB_COMPLETED, JOB_FAILED, JOB_PARTIAL, JobRecord,
)
from services.render_job import display_name, guess_name_column
from ui.pages.printing.style import DETAIL_WIDTH, apply_page_style
from ui.pages.printing.tables import ColumnSpec, RowsModel, RowsTable
from ui.pages.printing.text import (
    CARD_STATE_LABELS, CARD_STATE_TONES, FAMILY_NAMES, JOB_STATE_LABELS, JOB_STATE_TONES, brief,
    date_text, job_counts_short, job_counts_text, short_clock,
)
from ui.pages.settings.prefs import Prefs
from ui.shell import PAGE_LIST, PAGE_PRINT, BasePage
from ui.theme import SPACE
from ui.widgets import (
    ComboBox, EmptyState, FilterChip, PrimaryButton, QuietButton, SearchField, SecondaryButton, confirm,
    label, set_prop,
)

log = logging.getLogger(__name__)

PERIODS = (("today", "오늘"), ("week", "이번 주"), ("month", "이번 달"), ("all", "전체"))
STATUS_CHOICES = ((None, "모든 결과"), (JOB_COMPLETED, "완료"), (JOB_PARTIAL, "일부 실패"),
                  (JOB_FAILED, "실패"), (JOB_CANCELLED, "취소"))
_PERSON_SEARCH_LIMIT = 300


def period_start(period: str, now: datetime | None = None) -> str | None:
    now = now or datetime.now()
    day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    if period == "today":
        start = day
    elif period == "week":
        start = day - timedelta(days=day.weekday())
    elif period == "month":
        start = day.replace(day=1)
    else:
        return None
    return start.isoformat(timespec="seconds")


def _record_id(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class HistoryPage(BasePage):
    showPeopleRequested = Signal(list)
    reprintRequested = Signal(list)

    def __init__(self, workspace, services, parent: QWidget | None = None, page_size: int = 20,
                 prefs: Prefs | None = None) -> None:
        super().__init__("기록", parent)
        self.workspace = workspace
        self.services = services
        self.history = services.history
        self.prefs = prefs or Prefs.instance()
        self.confirm = confirm
        self.page_size = max(1, int(page_size))
        self._period = "all"
        self._status: str | None = None
        self._search = ""
        self._offset = 0
        self._total = 0
        self._jobs: list[JobRecord] = []
        self._selected_id: int | None = None
        self._detail_people: list[int] = []
        self._detail_failed: list[int] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        self.stack = QStackedWidget()
        root.addWidget(self.stack)
        self.empty = EmptyState(
            "아직 인쇄한 기록이 없습니다",
            "카드를 인쇄하면 언제, 누구 카드를, 몇 장 뽑았는지 여기에 남습니다.",
            "인쇄 화면으로 가기", "history")
        self.empty.actionTriggered.connect(lambda: self._navigate(PAGE_PRINT))
        self.stack.addWidget(self.empty)
        self.stack.addWidget(self._build_main())
        apply_page_style(self)

        self._search_timer = QTimer(self)
        self._search_timer.setSingleShot(True)
        self._search_timer.setInterval(250)
        self._search_timer.timeout.connect(self._apply_search)
        services.printing.finished.connect(lambda _summary: self.reload())
        workspace.store_changed.connect(self._refresh_detail)
        self.reload()

    # ------------------------------------------------------------------ 만들기
    def _build_main(self) -> QWidget:
        page = QWidget()
        outer = QHBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        left = QWidget()
        left.setObjectName("Stage")
        layout = QVBoxLayout(left)
        layout.setContentsMargins(SPACE["xl"], SPACE["xl"], SPACE["xl"], SPACE["lg"])
        layout.setSpacing(SPACE["md"])
        head = QHBoxLayout()
        head.setSpacing(SPACE["lg"])
        head.addWidget(label("인쇄 기록", "heading"))
        head.addStretch(1)
        self.search = SearchField("사람 이름이나 인쇄 이름으로 찾기")
        self.search.setMinimumWidth(220)
        self.search.setMaximumWidth(320)
        self.search.textChanged.connect(lambda _text: self._search_timer.start())
        head.addWidget(self.search, 2)
        layout.addLayout(head)

        filters = QHBoxLayout()
        filters.setSpacing(SPACE["sm"])
        self._period_group = QButtonGroup(self)
        self._period_group.setExclusive(True)
        self.period_chips: dict[str, FilterChip] = {}
        for key, text in PERIODS:
            chip = FilterChip(text, checked=key == self._period)
            self._period_group.addButton(chip)
            chip.toggled.connect(lambda on, k=key: on and self.set_period(k))
            self.period_chips[key] = chip
            filters.addWidget(chip)
        filters.addSpacing(SPACE["sm"])
        self.status_combo = ComboBox([text for _, text in STATUS_CHOICES])
        self.status_combo.setFixedWidth(140)
        self.status_combo.setAccessibleName("결과로 걸러 보기")
        self.status_combo.currentIndexChanged.connect(lambda i: self.set_status(STATUS_CHOICES[i][0]))
        filters.addWidget(self.status_combo)
        filters.addStretch(1)
        layout.addLayout(filters)

        card = QFrame()
        card.setObjectName("TableCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(6, 4, 6, 6)
        self.jobs_model = RowsModel([
            ColumnSpec("when", "언제", 132), ColumnSpec("name", "인쇄 이름", None),
            ColumnSpec("printer", "프린터", 160, optional=1),
            ColumnSpec("result", "결과", 90, tone_key="tone"), ColumnSpec("counts", "장수", 150),
        ])
        self.jobs_table = RowsTable(self.jobs_model)
        self.jobs_table.setAccessibleName("인쇄 기록 목록")
        self.jobs_table.selectionModel().selectionChanged.connect(lambda *_: self._on_job_selected())
        card_layout.addWidget(self.jobs_table)
        self.no_match = QWidget()
        no_match = QVBoxLayout(self.no_match)
        no_match.setContentsMargins(SPACE["xl"], SPACE["xxl"], SPACE["xl"], SPACE["xxl"])
        no_match.setSpacing(SPACE["sm"])
        text = label("조건에 맞는 기록이 없습니다", "title")
        text.setAlignment(Qt.AlignHCenter)
        no_match.addStretch(1)
        no_match.addWidget(text)
        hint = label("기간을 넓히거나 찾는 말을 지워 보세요.", "muted")
        hint.setAlignment(Qt.AlignHCenter)
        no_match.addWidget(hint)
        self.clear_filters_button = QuietButton("조건 모두 지우기", accent=True)
        self.clear_filters_button.clicked.connect(self.clear_filters)
        no_match.addWidget(self.clear_filters_button, 0, Qt.AlignHCenter)
        no_match.addStretch(2)
        card_layout.addWidget(self.no_match)
        layout.addWidget(card, 1)

        paging = QHBoxLayout()
        paging.setSpacing(SPACE["sm"])
        self.page_label = label("", "muted")
        self.prev_button = SecondaryButton("이전", "sm", icon="chevron-left")
        self.next_button = SecondaryButton("다음", "sm", icon="chevron-right")
        self.prev_button.clicked.connect(lambda: self.go_to_offset(self._offset - self.page_size))
        self.next_button.clicked.connect(lambda: self.go_to_offset(self._offset + self.page_size))
        paging.addWidget(self.page_label)
        paging.addStretch(1)
        self.purge_button = QuietButton("30일 지난 기록 지우기", "sm", icon="trash")
        self.purge_button.clicked.connect(self.purge_old)
        paging.addWidget(self.purge_button)
        paging.addWidget(self.prev_button)
        paging.addWidget(self.next_button)
        layout.addLayout(paging)
        outer.addWidget(left, 1)
        outer.addWidget(self._build_detail())
        return page

    def _build_detail(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("SidePanel")
        panel.setProperty("edge", "left")
        panel.setFixedWidth(DETAIL_WIDTH)
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(0, 0, 0, 0)
        self.detail_stack = QStackedWidget()
        outer.addWidget(self.detail_stack)

        blank = QWidget()
        blank_layout = QVBoxLayout(blank)
        blank_layout.setContentsMargins(SPACE["xl"], SPACE["xl"], SPACE["xl"], SPACE["xl"])
        blank_layout.addStretch(1)
        note = label("왼쪽에서 기록을 누르면\n사람별 결과가 여기에 보입니다.", "muted")
        note.setAlignment(Qt.AlignCenter)
        blank_layout.addWidget(note)
        blank_layout.addStretch(2)
        self.detail_stack.addWidget(blank)

        body = QWidget()
        body.setObjectName("PanelBody")
        layout = QVBoxLayout(body)
        layout.setContentsMargins(SPACE["xl"], SPACE["xl"], SPACE["lg"], SPACE["lg"])
        layout.setSpacing(SPACE["sm"])
        top = QHBoxLayout()
        self.detail_title = label("", "title", wrap=True)
        top.addWidget(self.detail_title, 1)
        self.detail_tag = QLabel("")
        self.detail_tag.setObjectName("Tag")
        top.addWidget(self.detail_tag, 0, Qt.AlignTop)
        layout.addLayout(top)
        self.detail_meta = label("", "muted", wrap=True)
        layout.addWidget(self.detail_meta)
        self.detail_counts = label("", "strong", wrap=True)
        layout.addWidget(self.detail_counts)
        self.detail_error = QLabel("")
        self.detail_error.setObjectName("Reason")
        self.detail_error.setProperty("tone", "error")
        self.detail_error.setWordWrap(True)
        layout.addWidget(self.detail_error)

        self.cards_model = RowsModel([
            ColumnSpec("order", "순서", 58), ColumnSpec("person", "사람", 92),
            ColumnSpec("result", "결과", 84, tone_key="tone"), ColumnSpec("reason", "이유", None),
            ColumnSpec("time", "시각", 62),
        ])
        self.cards_table = RowsTable(self.cards_model, selectable=False)
        self.cards_table.setAccessibleName("사람별 결과")
        layout.addWidget(self.cards_table, 1)
        self.names_note = label("", "hint", wrap=True)
        layout.addWidget(self.names_note)

        self.show_failed_button = SecondaryButton("실패한 사람만 명단에서 보기")
        self.reprint_button = PrimaryButton("이 사람들 다시 인쇄", icon="print")
        self.delete_button = QuietButton("이 기록 지우기", icon="trash")
        for button in (self.reprint_button, self.show_failed_button, self.delete_button):
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            layout.addWidget(button)
        self.show_failed_button.clicked.connect(self.show_failed_in_list)
        self.reprint_button.clicked.connect(self.reprint_selected)
        self.delete_button.clicked.connect(self.delete_selected)
        self.detail_stack.addWidget(body)
        return panel

    # ------------------------------------------------------------------ 틀
    def on_enter(self) -> None:
        self.reload()

    def _navigate(self, page_id: str) -> None:
        if self.shell is not None:
            self.shell.navigate(page_id)

    def _toast(self, text: str, kind: str = "info") -> None:
        if self.shell is not None:
            self.shell.show_toast(text, kind)

    # ------------------------------------------------------------------ 걸러 보기
    def set_period(self, period: str) -> None:
        if period == self._period:
            return
        self._period = period
        if not self.period_chips[period].isChecked():
            self.period_chips[period].setChecked(True)
        self._offset = 0
        self.reload()

    def set_status(self, status: str | None) -> None:
        if status == self._status:
            return
        self._status = status
        index = next(i for i, (key, _) in enumerate(STATUS_CHOICES) if key == status)
        if self.status_combo.currentIndex() != index:
            self.status_combo.setCurrentIndex(index)
        self._offset = 0
        self.reload()

    def set_search(self, text: str) -> None:
        if self.search.text() != text:
            self.search.setText(text)
        self._apply_search()

    def _apply_search(self) -> None:
        self._search_timer.stop()
        text = self.search.text().strip()
        if text != self._search:
            self._search = text
            self._offset = 0
            self.reload()

    def clear_filters(self) -> None:
        self.search.blockSignals(True)
        self.search.clear()
        self.search.blockSignals(False)
        self._search = ""
        self._status = None
        self.status_combo.blockSignals(True)
        self.status_combo.setCurrentIndex(0)
        self.status_combo.blockSignals(False)
        self._period = "all"
        self.period_chips["all"].setChecked(True)
        self._offset = 0
        self.reload()

    def go_to_offset(self, offset: int) -> None:
        offset = max(0, min(int(offset), max(0, self._total - 1)))
        offset -= offset % self.page_size
        if offset != self._offset:
            self._offset = offset
            self.reload()

    def page_text(self) -> str:
        return self.page_label.text()

    # ------------------------------------------------------------------ 읽기
    def _person_job_ids(self, text: str) -> set[int]:
        """지금 열린 명단에서 이름 등이 맞는 사람의 카드가 들어 있는 인쇄들."""
        store = self.workspace.store
        if store is None or not text:
            return set()
        try:
            ids = store.ids(quick_search(text))[:_PERSON_SEARCH_LIMIT]
        except Exception:
            return set()
        found: set[int] = set()
        for record_id in ids:
            found |= {card.job_id for card in self.history.cards_for_record(record_id)}
        # 사람 번호는 명단마다 다시 세므로, 다른 명단으로 뽑은 기록은 뺀다.
        return {job_id for job_id in found if self._same_list(self.history.get_job(job_id)) is not False}

    def _same_list(self, job: JobRecord | None) -> bool | None:
        """이 기록을 지금 열린 명단으로 뽑았는지. 예전 기록이라 알 수 없으면 None."""
        if job is None:
            return False
        list_id = getattr(job, "list_id", None)
        if not list_id:
            return None
        store = self.workspace.store
        if store is None:
            return False
        try:
            return store.list_id() == list_id
        except Exception:
            return False

    def _query(self) -> tuple[list[JobRecord], int]:
        since = period_start(self._period)
        person_jobs = self._person_job_ids(self._search)
        if not person_jobs:
            total = self.history.count_jobs(status=self._status, since=since, search=self._search or None)
            jobs = self.history.list_jobs(self.page_size, self._offset, status=self._status, since=since,
                                          search=self._search or None)
            return jobs, total
        by_name = self.history.list_jobs(10000, 0, status=self._status, since=since, search=self._search)
        merged = {job.id: job for job in by_name}
        for job_id in person_jobs - set(merged):
            job = self.history.get_job(job_id)
            if job is None or (self._status and job.status != self._status):
                continue
            if since and job.created_at < since:
                continue
            merged[job.id] = job
        ordered = sorted(merged.values(), key=lambda job: job.id, reverse=True)
        return ordered[self._offset:self._offset + self.page_size], len(ordered)

    def reload(self) -> None:
        try:
            any_jobs = self.history.count_jobs() > 0
            jobs, total = self._query()
            if not jobs and total and self._offset:
                self._offset = max(0, ((total - 1) // self.page_size) * self.page_size)
                jobs, total = self._query()
        except Exception as exc:
            log.exception("인쇄 기록을 읽지 못했습니다.")
            self._toast(f"인쇄 기록을 읽지 못했습니다. ({exc})", "error")
            return
        self._jobs, self._total = jobs, total
        self.stack.setCurrentIndex(1 if any_jobs else 0)
        rows = []
        for job in jobs:
            rows.append({
                "id": job.id, "when": date_text(job.created_at), "name": job.name or "이름 없는 인쇄",
                "printer": job.device_name or "알 수 없음",
                "result": JOB_STATE_LABELS.get(job.status, job.status),
                "tone": JOB_STATE_TONES.get(job.status, ""),
                "counts": job_counts_short(job.total_cards, job.success_cards),
                "_tip": brief(job.error_text) if job.error_text else "",
            })
        keep = self._selected_id
        self.jobs_model.set_rows(rows)
        self.jobs_table.setVisible(bool(rows))
        self.no_match.setVisible(not rows)
        first, last = self._offset + 1, self._offset + len(jobs)
        self.page_label.setText(f"{first:,}–{last:,}번째, 모두 {total:,}건" if jobs else "모두 0건")
        self.prev_button.setEnabled(self._offset > 0)
        self.next_button.setEnabled(self._offset + self.page_size < total)
        row = self.jobs_model.find(lambda r: r["id"] == keep) if keep is not None else -1
        if row >= 0:
            self.jobs_table.selectRow(row)
        else:
            self._selected_id = None
            self._refresh_detail()

    def job_rows(self) -> list[dict]:
        return list(self.jobs_model.rows())

    def show_latest(self) -> None:
        """가장 최근 인쇄를 골라 보여 준다(인쇄 화면의 '기록에서 자세히 보기')."""
        self.clear_filters()
        if self.jobs_model.rowCount():
            self.jobs_table.selectRow(0)

    def show_job(self, job_id: int | None) -> bool:
        """그 인쇄 기록을 골라 보여 준다. 걸러 보던 조건은 풀고, 다른 쪽에 있으면 그쪽으로 넘긴다."""
        if job_id is None or self.history.get_job(job_id) is None:
            return False
        self.clear_filters()
        if self.select_job(job_id):
            return True
        newer = sum(1 for job in self.history.list_jobs(100000, 0) if job.id > job_id)
        self._offset = newer - newer % self.page_size
        self.reload()
        return self.select_job(job_id)

    def select_job(self, job_id: int) -> bool:
        row = self.jobs_model.find(lambda r: r["id"] == job_id)
        if row < 0:
            return False
        self.jobs_table.selectRow(row)
        return True

    def selected_job_id(self) -> int | None:
        return self._selected_id

    # ------------------------------------------------------------------ 자세히
    def _on_job_selected(self) -> None:
        row = self.jobs_table.current_row()
        self._selected_id = self.jobs_model.row(row)["id"] if row >= 0 else None
        self._refresh_detail()

    def _refresh_detail(self) -> None:
        job = self.history.get_job(self._selected_id) if self._selected_id is not None else None
        if job is None:
            self.detail_stack.setCurrentIndex(0)
            self._detail_people, self._detail_failed = [], []
            return
        self.detail_stack.setCurrentIndex(1)
        self.detail_title.setText(job.name or "이름 없는 인쇄")
        self.detail_tag.setText(JOB_STATE_LABELS.get(job.status, job.status))
        set_prop(self.detail_tag, "tone", JOB_STATE_TONES.get(job.status, ""))
        family = FAMILY_NAMES.get(job.family, "")
        printer = " ".join(part for part in (job.device_name, f"({family})" if family else "") if part)
        self.detail_meta.setText(", ".join(part for part in (date_text(job.created_at), printer) if part))
        self.detail_counts.setText(job_counts_text(job.total_cards, job.success_cards, job.failed_cards))
        self.detail_error.setText(brief(job.error_text) if job.error_text and job.status != JOB_COMPLETED else "")
        self.detail_error.setVisible(bool(self.detail_error.text()))

        store = self.workspace.store
        same_list = self._same_list(job)        # True | False | None(예전 기록이라 모름)
        lookup = store if (store is not None and same_list is not False) else None
        name_column = guess_name_column(lookup)
        cards = self.history.list_cards(job.id)
        rows, people, failed = [], [], []
        known = guessed = 0
        for card in cards:
            record_id = _record_id(card.record_id)
            record = lookup.get(record_id) if (lookup is not None and record_id is not None) else None
            saved_name = (getattr(card, "record_name", None) or "").strip()
            if record_id is None:
                person = f"{card.card_index}번째 카드"
            elif saved_name:
                person = saved_name             # 인쇄할 때 적어 둔 이름. 명단이 바뀌어도 그대로다.
            elif record is not None:
                person = display_name(record, name_column, f"{card.card_index}번째 사람")
                guessed += 1
            else:
                person = "명단에 없음"
            if record is not None:
                known += 1
                if record_id not in people:
                    people.append(record_id)
                if card.status != CARD_SUCCESS and record_id not in failed:
                    failed.append(record_id)
            rows.append({
                "order": f"{card.card_index:,}", "person": person,
                "result": CARD_STATE_LABELS.get(card.status, card.status),
                "tone": CARD_STATE_TONES.get(card.status, ""),
                "reason": brief(card.error_text) if card.status == CARD_FAILED else "",
                "time": short_clock(card.finished_at),
                "_tip": brief(card.error_text) if card.error_text else "",
            })
        missing = job.total_cards - len(cards)
        if missing > 0:
            first, last = len(cards) + 1, job.total_cards
            rows.append({"order": f"{first:,}" if first == last else f"{first:,}–{last:,}",
                         "person": f"{missing:,}장", "result": "뽑지 않음", "tone": "idle",
                         "reason": "", "time": ""})
        self.cards_model.set_rows(rows)
        # 다시 뽑아 성공한 사람은 실패 목록에서 뺀다.
        succeeded = {_record_id(c.record_id) for c in cards if c.status == CARD_SUCCESS}
        failed = [i for i in failed if i not in succeeded]
        self._detail_people, self._detail_failed = people, failed
        has_people = any(_record_id(card.record_id) is not None for card in cards)
        list_name = (getattr(job, "list_name", None) or "").strip()
        if not has_people:
            note = ""
        elif same_list is False:
            which = f"'{list_name}' 명단" if list_name else "다른 명단"
            note = f"{which}으로 뽑은 기록입니다. 그 명단을 열면 여기서 다시 인쇄할 수 있습니다."
        elif same_list is True:
            note = "" if known else "이 기록의 사람들은 명단에서 지워졌습니다."
        elif store is None:
            note = "명단을 열면 사람 이름이 보이고, 다시 인쇄할 수 있습니다."
        elif guessed:
            note = ("예전 기록이라 이름을 지금 열려 있는 명단에서 찾아 보여 줍니다. 다른 명단으로 뽑은 "
                    "기록이면 이름이 맞지 않을 수 있습니다.")
        else:
            note = "" if known else "이 기록의 사람들은 지금 열려 있는 명단에 없습니다."
        self.names_note.setText(note)
        self.names_note.setVisible(bool(note))
        self.show_failed_button.setVisible(bool(failed))
        self.show_failed_button.setText(f"실패한 {len(failed):,}명만 명단에서 보기")
        self.reprint_button.setVisible(bool(people))
        self.reprint_button.setText(f"실패한 {len(failed):,}명 다시 인쇄" if failed
                                    else f"이 {len(people):,}명 다시 인쇄")

    def card_rows(self) -> list[dict]:
        return list(self.cards_model.rows())

    # ------------------------------------------------------------------ 할 수 있는 일
    def show_failed_in_list(self) -> None:
        ids = list(self._detail_failed)
        if not ids:
            return
        self.showPeopleRequested.emit(ids)
        self.workspace.set_current_record(ids[0])
        if self.shell is None:
            return
        try:
            page = self.shell.page(PAGE_LIST)
        except KeyError:
            page = None
        self.shell.navigate(PAGE_LIST)
        hook = getattr(page, "show_people", None)
        if callable(hook):
            hook(ids)

    def reprint_selected(self) -> None:
        ids = list(self._detail_failed or self._detail_people)
        if not ids:
            return
        self.reprintRequested.emit(ids)
        if self.shell is None:
            return
        try:
            page = self.shell.page(PAGE_PRINT)
        except KeyError:
            page = None
        hook = getattr(page, "print_people", None)
        if callable(hook):
            hook(ids)
        self.shell.navigate(PAGE_PRINT)

    def delete_selected(self) -> bool:
        job = self.history.get_job(self._selected_id) if self._selected_id is not None else None
        if job is None:
            return False
        if not self.confirm(self, "이 기록을 지울까요?",
                            f"'{job.name or '이름 없는 인쇄'}' ({date_text(job.created_at)}) 기록을 지웁니다. "
                            "명단에 적힌 '뽑은 날짜'는 그대로 남습니다. 지운 기록은 되돌릴 수 없습니다.",
                            confirm_text="기록 지우기", cancel_text="그대로 두기", danger=True):
            return False
        self.history.delete_job(job.id)
        self._selected_id = None
        self.reload()
        self._toast("기록을 지웠습니다.")
        return True

    def purge_old(self, days: int = 30) -> int:
        cutoff = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
        count = self.history.count_jobs(until=cutoff)
        if not count:
            self._toast(f"{days}일이 지난 기록이 없습니다.")
            return 0
        if not self.confirm(self, f"{days}일 지난 기록을 지울까요?",
                            f"{days}일보다 오래된 기록 {count:,}건을 지웁니다. 지운 기록은 되돌릴 수 없습니다.",
                            confirm_text=f"{count:,}건 지우기", cancel_text="그대로 두기", danger=True):
            return 0
        removed = self.history.purge_older_than(days)
        self.reload()
        self._toast(f"오래된 기록 {removed:,}건을 지웠습니다.")
        return removed
