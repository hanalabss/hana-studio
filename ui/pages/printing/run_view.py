"""인쇄 중과 인쇄가 끝난 뒤: 사람별 상태 표, 큰 진행 숫자, 멈추기, 결과 요약."""
from __future__ import annotations

from dataclasses import dataclass, field

from PySide6.QtCore import QModelIndex, QSize, QSortFilterProxyModel, Qt, Signal
from PySide6.QtGui import QImage
from PySide6.QtWidgets import (
    QButtonGroup, QFrame, QGridLayout, QHBoxLayout, QLabel, QScrollArea, QSizePolicy, QStackedWidget,
    QVBoxLayout, QWidget,
)

from ui.theme import SPACE
from ui.widgets import (
    RIBBON_PANELS, DangerButton, Divider, FilterChip, PrimaryButton, ProgressBar, QuietButton,
    RibbonProgress, SecondaryButton, label, set_prop,
)

from .findings import FindingRow, Problem
from .preview import CardPreview
from .style import RUN_WIDTH
from .tables import ColumnSpec, RowsModel, RowsTable

ROW_WAITING = "waiting"
ROW_PRINTING = "printing"
ROW_SUCCESS = "success"
ROW_FAILED = "failed"
ROW_CANCELLED = "cancelled"
ROW_SKIPPED = "skipped"

_ROW_TONE = {ROW_WAITING: "muted", ROW_PRINTING: "busy", ROW_SUCCESS: "ok", ROW_FAILED: "error",
             ROW_CANCELLED: "idle", ROW_SKIPPED: "idle"}
_ROW_TEXT = {ROW_WAITING: "기다리는 중", ROW_PRINTING: "인쇄 중", ROW_SUCCESS: "끝남",
             ROW_CANCELLED: "취소됨", ROW_SKIPPED: "뽑지 않음"}
_FILTERS = (("all", "전체"), (ROW_SUCCESS, "끝남"), (ROW_WAITING, "기다리는 중"), (ROW_FAILED, "실패"))

_PANEL_KEYS = {
    "y": "y", "m": "m", "c": "c", "k": "k", "o": "o", "w": "w", "uv": "uv",
    "화이트": "w", "형광": "uv",
}


def panel_position(panel: str, panels: tuple[str, ...]) -> tuple[int, float] | None:
    """드라이버가 알려 준 패널 이름 → RibbonProgress의 (몇 번째, 얼마나)."""
    key = str(panel or "").strip().lower()
    if not key:
        return None
    if key in ("ymck", "ymcko"):
        return (0, 0.5) if "y" in panels else None
    mapped = _PANEL_KEYS.get(key)
    if mapped in panels:
        return panels.index(mapped), 0.5
    return None


@dataclass
class SummaryView:
    headline: str
    detail: str = ""
    tone: str = "ok"                       # ok | warn | error | idle
    failures: list[tuple[str, str]] = field(default_factory=list)    # (이름, 이유)
    retry_count: int = 0
    continue_count: int = 0
    continue_text: str = ""
    note: str = ""


class _StateFilter(QSortFilterProxyModel):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._state = "all"

    def set_state(self, state: str) -> None:
        self._state = state
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:
        if self._state == "all":
            return True
        state = self.sourceModel().row(source_row).get("state")
        if self._state == ROW_WAITING:
            return state in (ROW_WAITING, ROW_PRINTING)
        return state == self._state


class RunView(QWidget):
    stopAfterRequested = Signal()
    stopNowRequested = Signal()
    retryRequested = Signal()
    continueRequested = Signal()
    historyRequested = Signal()
    backRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_table(), 1)
        root.addWidget(self._build_panel())
        self._total = 0
        self._panels: tuple[str, ...] = ()
        self._finished = False

    # ------------------------------------------------------------------ 왼쪽 표
    def _build_table(self) -> QWidget:
        side = QWidget()
        side.setObjectName("Stage")
        layout = QVBoxLayout(side)
        layout.setContentsMargins(SPACE["xl"], SPACE["xl"], SPACE["xl"], SPACE["xl"])
        layout.setSpacing(SPACE["md"])
        head = QHBoxLayout()
        head.setSpacing(SPACE["sm"])
        self.title = label("이번에 뽑는 카드", "heading")
        head.addWidget(self.title)
        head.addSpacing(SPACE["sm"])
        self._chip_group = QButtonGroup(self)
        self._chip_group.setExclusive(True)
        self.chips: dict[str, FilterChip] = {}
        for key, text in _FILTERS:
            chip = FilterChip(text, 0, tone="warn" if key == ROW_FAILED else "neutral", checked=key == "all")
            self._chip_group.addButton(chip)
            chip.toggled.connect(lambda on, k=key: on and self._proxy.set_state(k))
            self.chips[key] = chip
            head.addWidget(chip)
        head.addStretch(1)
        layout.addLayout(head)

        card = QFrame()
        card.setObjectName("TableCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(6, 4, 6, 6)
        self.model = RowsModel(self._columns([]))
        self._proxy = _StateFilter(self)
        self._proxy.setSourceModel(self.model)
        self.table = RowsTable(self.model, selectable=False, view_model=self._proxy)
        self.table.setAccessibleName("사람별 인쇄 상태")
        card_layout.addWidget(self.table)
        layout.addWidget(card, 1)
        return side

    @staticmethod
    def _columns(extras: list[str]) -> list[ColumnSpec]:
        columns = [ColumnSpec("order", "순서", 60), ColumnSpec("name", "이름", None)]
        columns += [ColumnSpec(f"extra{i}", title, None) for i, title in enumerate(extras)]
        columns.append(ColumnSpec("status", "상태", 280 if not extras else 240, tone_key="tone"))
        return columns

    # ------------------------------------------------------------------ 오른쪽 패널
    def _build_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("SidePanel")
        panel.setProperty("edge", "left")
        panel.setFixedWidth(RUN_WIDTH)
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(0, 0, 0, 0)
        self.stack = QStackedWidget()
        outer.addWidget(self.stack)
        self.stack.addWidget(self._build_progress())
        self.stack.addWidget(self._build_summary())
        return panel

    def _build_progress(self) -> QWidget:
        page = QWidget()
        page.setObjectName("PanelBody")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(SPACE["xl"], SPACE["xl"], SPACE["xl"], SPACE["xl"])
        layout.setSpacing(SPACE["md"])

        count = QHBoxLayout()
        count.setSpacing(SPACE["sm"])
        self.count_done = QLabel("0")
        self.count_done.setObjectName("BigCount")
        self.count_total = QLabel("/ 0장")
        self.count_total.setObjectName("BigCountRest")
        count.addWidget(self.count_done, 0, Qt.AlignBottom)
        rest = QVBoxLayout()
        rest.addStretch(1)
        rest.addWidget(self.count_total)
        rest.addSpacing(12)
        count.addLayout(rest)
        count.addStretch(1)
        layout.addLayout(count)
        self.eta = label("", "secondary")
        layout.addWidget(self.eta)

        self.preview = CardPreview(minimum=QSize(160, 96), margin=3.0)
        self.preview.setMaximumHeight(200)
        self.preview.set_placeholder("카드를 준비하고 있습니다")
        layout.addWidget(self.preview, 4)
        self.current_name = label("", "strong")
        self.current_sub = label("", "muted")
        who = QHBoxLayout()
        who.setSpacing(SPACE["sm"])
        who.addStretch(1)
        who.addWidget(self.current_name)
        who.addWidget(self.current_sub)
        who.addStretch(1)
        layout.addLayout(who)

        self.stage_title = label("이 카드에서 지금 하는 일", "label")
        layout.addWidget(self.stage_title)
        self.stage_text = label("", "strong", wrap=True)
        layout.addWidget(self.stage_text)
        self.bar = ProgressBar()
        layout.addWidget(self.bar)
        self.ribbon: RibbonProgress | None = None
        self._ribbon_slot = QVBoxLayout()
        self._ribbon_slot.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(self._ribbon_slot)

        layout.addSpacing(SPACE["xs"])
        layout.addWidget(Divider())
        grid = QGridLayout()
        grid.setHorizontalSpacing(SPACE["lg"])
        grid.setVerticalSpacing(6)
        self.supply_labels: dict[str, QLabel] = {}
        self._supply_titles: dict[str, QLabel] = {}
        for row, (key, title) in enumerate((("ribbon", "리본"), ("film", "필름"), ("feeder", "카드 넣는 곳"))):
            name = label(title, "muted")
            value = label("", None)
            grid.addWidget(name, row, 0)
            grid.addWidget(value, row, 1)
            self._supply_titles[key] = name
            self.supply_labels[key] = value
        grid.setColumnStretch(1, 1)
        layout.addLayout(grid)
        layout.addStretch(1)

        self.stop_note = QLabel("")
        self.stop_note.setObjectName("Reason")
        self.stop_note.setWordWrap(True)
        layout.addWidget(self.stop_note)
        buttons = QHBoxLayout()
        buttons.setSpacing(SPACE["sm"])
        self.stop_after_button = SecondaryButton("이 카드까지만", "lg")
        self.stop_after_button.setToolTip("지금 찍는 카드가 끝나면 멈춥니다")
        self.stop_now_button = DangerButton("지금 멈추기", "lg")
        for button in (self.stop_after_button, self.stop_now_button):
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            buttons.addWidget(button)
        self.stop_after_button.clicked.connect(self.stopAfterRequested)
        self.stop_now_button.clicked.connect(self.stopNowRequested)
        layout.addLayout(buttons)
        return page

    def _build_summary(self) -> QWidget:
        page = QWidget()
        page.setObjectName("PanelBody")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(SPACE["xl"], SPACE["xl"], SPACE["xl"], SPACE["xl"])
        layout.setSpacing(SPACE["md"])
        self.summary_headline = QLabel("")
        self.summary_headline.setObjectName("Headline")
        self.summary_headline.setWordWrap(True)
        layout.addWidget(self.summary_headline)
        self.summary_detail = label("", "secondary", wrap=True)
        layout.addWidget(self.summary_detail)

        counts = QFrame()
        counts.setObjectName("Block")
        counts_layout = QHBoxLayout(counts)
        counts_layout.setContentsMargins(SPACE["md"], SPACE["md"], SPACE["md"], SPACE["md"])
        self.summary_counts: dict[str, QLabel] = {}
        for key, caption in (("success", "끝난 카드"), ("failed", "실패한 카드"), ("left", "뽑지 않은 카드")):
            column = QVBoxLayout()
            column.setSpacing(2)
            column.addWidget(label(caption, "caption"))
            number = QLabel("0장")
            number.setObjectName("StatNumber")
            column.addWidget(number)
            self.summary_counts[key] = number
            counts_layout.addLayout(column, 1)
        layout.addWidget(counts)

        self.failures_title = label("실패한 카드와 이유", "label")
        layout.addWidget(self.failures_title)
        self._failures_scroll = QScrollArea()
        self._failures_scroll.setObjectName("PanelScroll")
        self._failures_scroll.setWidgetResizable(True)
        self._failures_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        body = QWidget()
        body.setObjectName("PanelBody")
        self._failures_box = QVBoxLayout(body)
        self._failures_box.setContentsMargins(0, 0, 0, 0)
        self._failures_box.setSpacing(SPACE["sm"])
        self._failures_box.addStretch(1)
        self._failures_scroll.setWidget(body)
        layout.addWidget(self._failures_scroll, 1)
        self.failure_rows: list[FindingRow] = []
        self._summary_stretch = QWidget()
        self._summary_stretch.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        layout.addWidget(self._summary_stretch, 1)

        self.summary_note = QLabel("")
        self.summary_note.setObjectName("Reason")
        self.summary_note.setWordWrap(True)
        layout.addWidget(self.summary_note)
        self.retry_button = PrimaryButton("", "lg", icon="print")
        self.continue_button = SecondaryButton("", "lg")
        self.back_button = SecondaryButton("인쇄 준비로 돌아가기", "lg")
        self.history_button = QuietButton("기록에서 자세히 보기", accent=True, icon="history")
        for button in (self.retry_button, self.continue_button, self.back_button, self.history_button):
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            layout.addWidget(button)
        self.retry_button.clicked.connect(self.retryRequested)
        self.continue_button.clicked.connect(self.continueRequested)
        self.back_button.clicked.connect(self.backRequested)
        self.history_button.clicked.connect(self.historyRequested)
        return page

    # ------------------------------------------------------------------ 시작
    def begin(self, title: str, rows: list[dict], extras: list[str], panels: tuple[str, ...] = ()) -> None:
        """rows: {"name", "extra0", "extra1", "record_id"}. 순서와 상태는 여기서 채운다."""
        self._finished = False
        self._total = len(rows)
        self.title.setText(title)
        filled = []
        for index, row in enumerate(rows):
            filled.append({**row, "order": f"{index + 1:,}", "state": ROW_WAITING,
                           "status": _ROW_TEXT[ROW_WAITING], "tone": _ROW_TONE[ROW_WAITING]})
        self.model.set_columns(self._columns(extras))
        self.model.set_rows(filled)
        self.table.apply_widths()
        self.chips["all"].setChecked(True)
        self._sync_chips()
        self._set_ribbon(panels)
        self.count_done.setText("0")
        self.count_total.setText(f"/ {self._total:,}장")
        self.eta.setText("")
        self.stage_text.setText("프린터에 연결하고 있습니다")
        self.bar.set_fraction(0.0)
        self.preview.set_image(None)
        self.current_name.setText("")
        self.current_sub.setText("")
        self.stop_note.setText("")
        self.stop_note.setVisible(False)
        self.stop_after_button.setEnabled(self._total > 1)
        self.stop_now_button.setEnabled(True)
        self.stack.setCurrentIndex(0)

    def _set_ribbon(self, panels: tuple[str, ...]) -> None:
        if self.ribbon is not None:
            self._ribbon_slot.removeWidget(self.ribbon)
            self.ribbon.setParent(None)
            self.ribbon.deleteLater()
            self.ribbon = None
        self._panels = tuple(p for p in panels if p in RIBBON_PANELS)
        if self._panels:
            self.ribbon = RibbonProgress(self._panels)
            self.ribbon.setVisible(False)
            self._ribbon_slot.addWidget(self.ribbon)

    def is_finished(self) -> bool:
        return self._finished

    # ------------------------------------------------------------------ 진행
    def set_progress(self, done: int, total: int, index: int, stage: str, panel: str = "",
                     card_percent: int = 0) -> None:
        """card_percent: 지금 찍는 카드 한 장 안에서의 진행(0~100)."""
        self.count_done.setText(f"{done:,}")
        self.count_total.setText(f"/ {total:,}장")
        self.bar.setValue(max(0, min(100, int(card_percent))))
        stage = stage.rstrip(". ").strip()
        position = panel_position(panel, self._panels) if self.ribbon is not None else None
        if position is not None:
            self.ribbon.setVisible(True)
            self.ribbon.set_progress(*position)
            self.stage_title.setText("지금 찍는 색")
        else:
            if self.ribbon is not None:
                self.ribbon.setVisible(False)
            self.stage_title.setText("이 카드에서 지금 하는 일")
        self.stage_text.setText(stage or "인쇄하고 있습니다")
        self.set_row_printing(index - 1, stage)

    def set_row_printing(self, row: int, stage: str = "") -> None:
        if not 0 <= row < self.model.rowCount():
            return
        if self.model.row(row).get("state") not in (ROW_WAITING, ROW_PRINTING):
            return
        was_current = self.model.row(row).get("_current")
        for other in range(self.model.rowCount()):
            if other != row and self.model.row(other).get("_current"):
                self.model.update_row(other, _current=False)
        self.model.update_row(row, state=ROW_PRINTING, status=stage or _ROW_TEXT[ROW_PRINTING],
                              tone=_ROW_TONE[ROW_PRINTING], _current=True)
        if not was_current:
            self._sync_chips()
            self._scroll_to(row)

    def set_row_result(self, row: int, state: str, message: str = "") -> None:
        if not 0 <= row < self.model.rowCount():
            return
        text = f"실패: {message}" if state == ROW_FAILED and message else \
            ("실패" if state == ROW_FAILED else _ROW_TEXT[state])
        self.model.update_row(row, state=state, status=text, tone=_ROW_TONE[state], _current=False,
                              _tip=message if state == ROW_FAILED else "")
        self._sync_chips()

    def row_state(self, row: int) -> str:
        return self.model.row(row).get("state", "")

    def row_status(self, row: int) -> str:
        return self.model.row(row).get("status", "")

    def _scroll_to(self, row: int) -> None:
        index = self._proxy.mapFromSource(self.model.index(row, 0))
        if index.isValid():
            self.table.scrollTo(index, self.table.ScrollHint.PositionAtCenter)

    def _sync_chips(self) -> None:
        states = [row.get("state") for row in self.model.rows()]
        waiting = sum(1 for s in states if s in (ROW_WAITING, ROW_PRINTING))
        failed = states.count(ROW_FAILED)
        self.chips["all"].set_count(len(states))
        self.chips[ROW_SUCCESS].set_count(states.count(ROW_SUCCESS))
        self.chips[ROW_WAITING].set_count(waiting)
        self.chips[ROW_FAILED].set_count(failed)
        self.chips[ROW_FAILED].setVisible(failed > 0 or self.chips[ROW_FAILED].isChecked())

    def set_eta(self, text: str) -> None:
        self.eta.setText(text)

    def set_current_card(self, image: QImage | None, name: str, sub: str = "") -> None:
        self.preview.set_image(image)
        self.current_name.setText(name)
        self.current_sub.setText(sub)
        self.current_sub.setVisible(bool(sub))

    def set_supplies(self, ribbon: str, film: str, feeder: str) -> None:
        for key, text in (("ribbon", ribbon), ("film", film), ("feeder", feeder)):
            self.supply_labels[key].setText(text)
            self.supply_labels[key].setVisible(bool(text))
            self._supply_titles[key].setVisible(bool(text))

    def set_stopping(self, mode: str) -> None:
        self.stop_after_button.setEnabled(False)
        if mode == "now":
            self.stop_now_button.setEnabled(False)
            self.stop_note.setText("멈추는 중입니다. 프린터 안에 들어간 카드는 끝까지 찍혀 나옵니다.")
        else:
            self.stop_note.setText("지금 찍는 카드가 끝나면 멈춥니다. 당장 멈추려면 '지금 멈추기'를 누르세요.")
        self.stop_note.setVisible(True)

    # ------------------------------------------------------------------ 끝
    def finish(self, summary: SummaryView, success: int, failed: int, left: int,
               leftover_state: str = ROW_SKIPPED) -> None:
        self._finished = True
        for row in range(self.model.rowCount()):
            if self.model.row(row).get("state") in (ROW_WAITING, ROW_PRINTING):
                self.set_row_result(row, leftover_state)
        self.count_done.setText(f"{success:,}")
        self.summary_headline.setText(summary.headline)
        self.summary_detail.setText(summary.detail)
        self.summary_detail.setVisible(bool(summary.detail))
        self.summary_counts["success"].setText(f"{success:,}장")
        self.summary_counts["failed"].setText(f"{failed:,}장")
        self.summary_counts["left"].setText(f"{left:,}장")
        set_prop(self.summary_counts["failed"], "tone", "warn" if failed else "muted")
        set_prop(self.summary_counts["left"], "tone", "" if left else "muted")

        while self._failures_box.count() > 1:
            item = self._failures_box.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self.failure_rows = []
        for name, reason in summary.failures:
            row = FindingRow(Problem("failed_card", "error", f"{name}: {reason}" if reason else name))
            self._failures_box.insertWidget(self._failures_box.count() - 1, row)
            self.failure_rows.append(row)
        has_failures = bool(summary.failures)
        self.failures_title.setVisible(has_failures)
        self._failures_scroll.setVisible(has_failures)
        self._summary_stretch.setVisible(not has_failures)

        self.summary_note.setText(summary.note)
        self.summary_note.setVisible(bool(summary.note))
        self.retry_button.setVisible(summary.retry_count > 0)
        self.retry_button.setText(f"실패한 {summary.retry_count:,}장 다시 인쇄")
        self.continue_button.setVisible(summary.continue_count > 0)
        self.continue_button.setText(summary.continue_text or f"남은 {summary.continue_count:,}장 이어서 인쇄")
        self.continue_button.set_variant("primary" if summary.retry_count == 0 else "secondary")
        primary_back = summary.retry_count == 0 and summary.continue_count == 0
        self.back_button.set_variant("primary" if primary_back else "secondary")
        self.back_button.setText("다른 카드 뽑기" if primary_back else "인쇄 준비로 돌아가기")
        self.stack.setCurrentIndex(1)

    def failure_texts(self) -> list[str]:
        return [row.text() for row in self.failure_rows]
