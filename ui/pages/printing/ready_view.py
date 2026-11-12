"""인쇄 준비: 누구를 뽑을지, 카드가 어떻게 나오는지, 프린터와 인쇄 전 확인."""
from __future__ import annotations

from PySide6.QtCore import QEvent, QModelIndex, QRect, QSize, Qt, Signal
from PySide6.QtGui import QFont, QPainter
from PySide6.QtWidgets import (
    QAbstractItemView, QButtonGroup, QCheckBox, QFrame, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QRadioButton, QScrollArea, QSizePolicy, QStyle, QStyledItemDelegate, QVBoxLayout,
    QWidget,
)

from ui.theme import FONT_SIZES, SPACE, ThemeManager
from ui.widgets import (
    Divider, IconButton, PrimaryButton, QuietButton, SegmentedControl, StatusPill, StepperField, label,
    set_prop, theme_color,
)

from .findings import FindingRow, Problem
from .preview import CardPreview
from .style import LEFT_WIDTH, RIGHT_WIDTH
from .targets import (
    MODE_ALL, MODE_CHECKED, MODE_COPIES, MODE_CURRENT, MODE_PICKED, MODE_RANGES, MODE_UNPRINTED,
    TargetCounts,
)

ROLE_ID = Qt.UserRole
ROLE_SUB = Qt.UserRole + 1
ROLE_TAG = Qt.UserRole + 2
ROLE_TAG_TONE = Qt.UserRole + 3
_ROW = 38


class _PersonDelegate(QStyledItemDelegate):
    """순서 번호, 이름, 덧붙는 말, 오른쪽 꼬리표."""

    def sizeHint(self, option, index: QModelIndex) -> QSize:
        return QSize(120, _ROW)

    def paint(self, painter: QPainter, option, index: QModelIndex) -> None:
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        rect: QRect = option.rect
        selected = bool(option.state & QStyle.State_Selected)
        hovered = bool(option.state & QStyle.State_MouseOver)
        if selected or hovered:
            painter.setPen(Qt.NoPen)
            painter.setBrush(theme_color("accent_soft" if selected else "surface_hover"))
            painter.drawRoundedRect(rect.adjusted(0, 1, 0, -1), 6, 6)
        base = QFont(option.font)
        small = QFont(base)
        small.setPixelSize(FONT_SIZES["caption"])
        painter.setFont(small)
        painter.setPen(theme_color("text_muted"))
        painter.drawText(QRect(rect.left(), rect.top(), 34, rect.height()), Qt.AlignVCenter | Qt.AlignRight,
                         str(index.row() + 1))
        right = rect.right() - 8
        tag = index.data(ROLE_TAG)
        if tag:
            tone = index.data(ROLE_TAG_TONE) or "muted"
            painter.setPen(theme_color({"warn": "warn_text", "error": "danger_text"}.get(tone, "text_muted")))
            width = painter.fontMetrics().horizontalAdvance(tag) + 4
            painter.drawText(QRect(right - width, rect.top(), width, rect.height()),
                             Qt.AlignVCenter | Qt.AlignRight, tag)
            right -= width + 8
        name_font = QFont(base)
        name_font.setPixelSize(FONT_SIZES["body"])
        if selected:
            name_font.setWeight(QFont.Weight(ThemeManager.instance().font_spec.strong_weight))
        painter.setFont(name_font)
        painter.setPen(theme_color("text"))
        left = rect.left() + 46
        name = str(index.data(Qt.DisplayRole) or "")
        metrics = painter.fontMetrics()
        sub = str(index.data(ROLE_SUB) or "")
        available = max(20, right - left)
        name_width = min(metrics.horizontalAdvance(name) + 2, available)
        painter.drawText(QRect(left, rect.top(), name_width, rect.height()), Qt.AlignVCenter | Qt.AlignLeft,
                         metrics.elidedText(name, Qt.ElideRight, name_width))
        if sub and available - name_width > 40:
            painter.setFont(small)
            painter.setPen(theme_color("text_muted"))
            sub_rect = QRect(left + name_width + 8, rect.top(), available - name_width - 8, rect.height())
            painter.drawText(sub_rect, Qt.AlignVCenter | Qt.AlignLeft,
                             painter.fontMetrics().elidedText(sub, Qt.ElideRight, sub_rect.width()))
        painter.restore()


class _Stat(QWidget):
    def __init__(self, caption: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.caption = label(caption, "caption", wrap=True)
        self.number = QLabel("0명")
        self.number.setObjectName("StatNumber")
        layout.addWidget(self.caption)
        layout.addWidget(self.number)
        layout.addStretch(1)

    def set(self, text: str, caption: str | None = None, tone: str = "") -> None:
        self.number.setText(text)
        if caption is not None:
            self.caption.setText(caption)
        set_prop(self.number, "tone", tone)


class ReadyView(QWidget):
    modeChanged = Signal(str)
    rangeEdited = Signal(str)
    copiesChanged = Signal(int)
    peopleEdited = Signal(list)          # 순서를 바꾸거나 뺀 뒤의 _id 목록
    restoreRequested = Signal()
    personSelected = Signal(int)         # 목록에서의 위치
    stepRequested = Signal(int)          # -1 이전, +1 다음
    sideChanged = Signal(str)
    optionsRequested = Signal()
    printerChangeRequested = Signal()
    findingAction = Signal(str, object)
    acknowledgeToggled = Signal(bool)
    startRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_left())
        root.addWidget(self._build_centre(), 1)
        root.addWidget(self._build_right())
        self._problems: list[Problem] = []
        self.finding_rows: list[FindingRow] = []

    # ------------------------------------------------------------------ 왼쪽
    def _build_left(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("SidePanel")
        panel.setProperty("edge", "right")
        panel.setFixedWidth(LEFT_WIDTH)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(SPACE["xl"], SPACE["xl"], SPACE["lg"], SPACE["lg"])
        layout.setSpacing(SPACE["sm"])
        layout.addWidget(label("누구 카드를 뽑을까요", "title"))
        layout.addSpacing(SPACE["xs"])

        self._group = QButtonGroup(self)
        self.mode_buttons: dict[str, QRadioButton] = {}
        for mode in (MODE_PICKED, MODE_CHECKED, MODE_UNPRINTED, MODE_CURRENT, MODE_RANGES, MODE_ALL, MODE_COPIES):
            button = QRadioButton("")
            button.setObjectName("Target")
            button.setCursor(Qt.PointingHandCursor)
            self._group.addButton(button)
            self.mode_buttons[mode] = button
            button.toggled.connect(lambda on, m=mode: on and self._on_mode(m))
            layout.addWidget(button)
            if mode == MODE_RANGES:
                self.range_edit = QLineEdit()
                self.range_edit.setPlaceholderText("예: 1,3,5-9")
                self.range_edit.setAccessibleName("뽑을 사람 번호")
                self.range_edit.setFixedHeight(40)
                self.range_edit.textChanged.connect(self.rangeEdited)
                self.range_hint = label("", "hint", wrap=True)
                self.range_box = QWidget()
                box = QVBoxLayout(self.range_box)
                box.setContentsMargins(28, 0, 0, 4)
                box.setSpacing(4)
                box.addWidget(self.range_edit)
                box.addWidget(self.range_hint)
                layout.addWidget(self.range_box)
            if mode == MODE_COPIES:
                self.copies_field = StepperField(1, 1, 999, 1, 0, "장")
                self.copies_field.setAccessibleName("뽑을 장수")
                self.copies_field.valueChanged.connect(lambda v: self.copiesChanged.emit(int(v)))
                self.copies_box = QWidget()
                box = QHBoxLayout(self.copies_box)
                box.setContentsMargins(28, 0, 0, 4)
                self.copies_field.setFixedWidth(200)
                box.addWidget(self.copies_field)
                box.addStretch(1)
                layout.addWidget(self.copies_box)
        self.no_list_note = label("", "hint", wrap=True)
        layout.addWidget(self.no_list_note)

        layout.addSpacing(SPACE["sm"])
        stats = QFrame()
        stats.setObjectName("Block")
        stats_layout = QHBoxLayout(stats)
        stats_layout.setContentsMargins(SPACE["md"], SPACE["md"], SPACE["md"], SPACE["md"])
        stats_layout.setSpacing(SPACE["md"])
        self.stat_targets = _Stat("뽑을 사람")
        self.stat_printed = _Stat("이미 뽑은 사람")
        self.stat_photo = _Stat("사진 없음")
        for stat in (self.stat_targets, self.stat_printed, self.stat_photo):
            stats_layout.addWidget(stat, 1)
        layout.addWidget(stats)

        layout.addSpacing(SPACE["sm"])
        head = QHBoxLayout()
        head.setSpacing(2)
        self.order_title = label("뽑는 순서", "label")
        head.addWidget(self.order_title)
        head.addStretch(1)
        self.up_button = IconButton("chevron-up", "위로 올리기", size="sm")
        self.down_button = IconButton("chevron-down", "아래로 내리기", size="sm")
        self.remove_button = QuietButton("빼기", "sm", icon="close")
        self.remove_button.setToolTip("고른 사람을 이번 인쇄에서 뺍니다 (Delete)")
        self.up_button.clicked.connect(lambda: self._move_selected(-1))
        self.down_button.clicked.connect(lambda: self._move_selected(1))
        self.remove_button.clicked.connect(self._remove_selected)
        for button in (self.up_button, self.down_button, self.remove_button):
            head.addWidget(button)
        layout.addLayout(head)

        self.people = QListWidget()
        self.people.setObjectName("People")
        self.people.setItemDelegate(_PersonDelegate(self.people))
        self.people.setUniformItemSizes(True)
        self.people.setMouseTracking(True)
        self.people.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.people.setDragDropMode(QAbstractItemView.InternalMove)
        self.people.setDefaultDropAction(Qt.MoveAction)
        self.people.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.people.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.people.setAccessibleName("이번에 뽑을 사람들")
        self.people.model().rowsMoved.connect(self._on_rows_moved)
        self.people.currentRowChanged.connect(self._on_current_row)
        self.people.itemSelectionChanged.connect(self._sync_tools)
        self.people.installEventFilter(self)
        layout.addWidget(self.people, 1)
        self.people_empty = label("", "muted", wrap=True)
        layout.addWidget(self.people_empty)

        foot = QHBoxLayout()
        self.excluded_note = label("", "caption")
        self.restore_button = QuietButton("다시 넣기", "sm", accent=True)
        self.restore_button.clicked.connect(self.restoreRequested)
        foot.addWidget(self.excluded_note)
        foot.addStretch(1)
        foot.addWidget(self.restore_button)
        self.excluded_row = QWidget()
        self.excluded_row.setLayout(foot)
        foot.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.excluded_row)
        self._mode = MODE_UNPRINTED
        self._filling = False
        return panel

    # ------------------------------------------------------------------ 가운데
    def _build_centre(self) -> QWidget:
        stage = QWidget()
        stage.setObjectName("Stage")
        layout = QVBoxLayout(stage)
        layout.setContentsMargins(SPACE["xl"], SPACE["xl"], SPACE["xl"], SPACE["lg"])
        layout.setSpacing(SPACE["md"])
        top = QHBoxLayout()
        self.preview_name = label("", "heading")
        self.preview_sub = label("", "secondary")
        names = QVBoxLayout()
        names.setSpacing(2)
        names.addWidget(self.preview_name)
        names.addWidget(self.preview_sub)
        top.addLayout(names, 1)
        self.side_control = SegmentedControl([("front", "앞면"), ("back", "뒷면")])
        self.side_control.currentChanged.connect(self.sideChanged)
        top.addWidget(self.side_control, 0, Qt.AlignTop)
        layout.addLayout(top)
        self._stage_layout = layout
        layout.addStretch(1)
        self.preview = CardPreview()
        self.preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(self.preview)
        nav = QHBoxLayout()
        nav.setSpacing(SPACE["md"])
        self.prev_button = IconButton("chevron-left", "이전 사람", variant="secondary")
        self.next_button = IconButton("chevron-right", "다음 사람", variant="secondary")
        self.position_label = label("", "strong")
        self.position_label.setMinimumWidth(96)
        self.position_label.setAlignment(Qt.AlignCenter)
        self.prev_button.clicked.connect(lambda: self.stepRequested.emit(-1))
        self.next_button.clicked.connect(lambda: self.stepRequested.emit(1))
        nav.addStretch(1)
        nav.addWidget(self.prev_button)
        nav.addWidget(self.position_label)
        nav.addWidget(self.next_button)
        nav.addStretch(1)
        layout.addLayout(nav)
        layout.addStretch(1)
        self._stage = stage
        return stage

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.fit_preview()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.fit_preview()

    def fit_preview(self) -> None:
        """카드를 가운데 칸 너비에 맞추되, 위아래 글자와 단추 자리는 남긴다."""
        margins = self._stage_layout.contentsMargins()
        width = max(120, self._stage.width() - margins.left() - margins.right())
        room = max(140, self._stage.height() - 190)
        aspect = self.preview.aspect()
        height = int(min(room, (width - 24) / aspect + 24))
        if self.preview.height() != height:
            self.preview.setFixedHeight(height)

    # ------------------------------------------------------------------ 오른쪽
    def _build_right(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("SidePanel")
        panel.setProperty("edge", "left")
        panel.setFixedWidth(RIGHT_WIDTH)
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setObjectName("PanelScroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        body = QWidget()
        body.setObjectName("PanelBody")
        layout = QVBoxLayout(body)
        layout.setContentsMargins(SPACE["lg"], SPACE["xl"], SPACE["lg"], SPACE["md"])
        layout.setSpacing(SPACE["md"])
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)

        printer = QFrame()
        printer.setObjectName("Block")
        box = QVBoxLayout(printer)
        box.setContentsMargins(SPACE["md"], SPACE["md"], SPACE["md"], SPACE["md"])
        box.setSpacing(6)
        row = QHBoxLayout()
        self.printer_name = label("프린터 없음", "strong")
        row.addWidget(self.printer_name, 1)
        self.printer_pill = StatusPill("연결 안 됨", "offline")
        self.printer_pill.setFocusPolicy(Qt.NoFocus)
        self.printer_pill.clicked.connect(self.printerChangeRequested)
        row.addWidget(self.printer_pill)
        box.addLayout(row)
        self.printer_detail = label("", "muted", wrap=True)
        box.addWidget(self.printer_detail)
        self.supplies = label("", "muted", wrap=True)
        box.addWidget(self.supplies)
        change = QHBoxLayout()
        self.change_printer = QuietButton("프린터 바꾸기", "sm", accent=True)
        self.change_printer.clicked.connect(self.printerChangeRequested)
        change.addWidget(self.change_printer)
        change.addStretch(1)
        box.addLayout(change)
        layout.addWidget(printer)

        settings = QVBoxLayout()
        settings.setSpacing(2)
        settings.addWidget(label("인쇄 설정", "label"))
        self.options_summary = label("", None, wrap=True)
        settings.addWidget(self.options_summary)
        opts = QHBoxLayout()
        self.options_button = QuietButton("인쇄 설정 바꾸기", "sm", accent=True)
        self.options_button.clicked.connect(self.optionsRequested)
        opts.addWidget(self.options_button)
        opts.addStretch(1)
        settings.addLayout(opts)
        layout.addLayout(settings)
        layout.addWidget(Divider())

        self.findings_title = label("인쇄 전 확인", "label")
        layout.addWidget(self.findings_title)
        self._findings_box = QVBoxLayout()
        self._findings_box.setSpacing(SPACE["sm"])
        layout.addLayout(self._findings_box)
        self.acknowledge = QCheckBox("")
        self.acknowledge.setObjectName("Acknowledge")
        self.acknowledge.toggled.connect(self.acknowledgeToggled)
        layout.addWidget(self.acknowledge)
        layout.addStretch(1)

        bottom = QFrame()
        bottom_layout = QVBoxLayout(bottom)
        bottom_layout.setContentsMargins(SPACE["lg"], SPACE["md"], SPACE["lg"], SPACE["lg"])
        bottom_layout.setSpacing(SPACE["sm"])
        bottom_layout.addWidget(Divider())
        self.reason = QLabel("")
        self.reason.setObjectName("Reason")
        self.reason.setWordWrap(True)
        bottom_layout.addWidget(self.reason)
        self.start_button = PrimaryButton("인쇄 시작", "lg", icon="print")
        self.start_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.start_button.clicked.connect(self.startRequested)
        bottom_layout.addWidget(self.start_button)
        outer.addWidget(bottom)
        return panel

    # ------------------------------------------------------------------ 고르는 방식
    def mode(self) -> str:
        return self._mode

    def _on_mode(self, mode: str) -> None:
        if mode == self._mode:
            self._sync_mode_boxes()
            return
        self._mode = mode
        self._sync_mode_boxes()
        self.modeChanged.emit(mode)

    def set_mode(self, mode: str) -> None:
        self._mode = mode
        button = self.mode_buttons[mode]
        if not button.isChecked():
            button.blockSignals(True)
            button.setChecked(True)
            button.blockSignals(False)
        self._sync_mode_boxes()

    def _sync_mode_boxes(self) -> None:
        self.range_box.setVisible(self._mode == MODE_RANGES and not self.mode_buttons[MODE_RANGES].isHidden())
        self.copies_box.setVisible(self._mode == MODE_COPIES and not self.mode_buttons[MODE_COPIES].isHidden())

    def set_counts(self, counts: TargetCounts, has_list: bool, copies: int) -> None:
        buttons = self.mode_buttons

        def show(mode: str, visible: bool, text: str, enabled: bool = True) -> None:
            buttons[mode].setVisible(visible)
            buttons[mode].setText(text)
            buttons[mode].setEnabled(enabled)

        show(MODE_PICKED, bool(counts.picked), f"기록에서 고른 {counts.picked or 0:,}명")
        show(MODE_CHECKED, has_list, f"선택한 {counts.checked or 0:,}명" if counts.checked
             else "선택한 사람 (명단에서 체크하면 됩니다)", bool(counts.checked))
        show(MODE_UNPRINTED, has_list, f"아직 안 뽑은 {counts.unprinted or 0:,}명 모두" if counts.unprinted
             else "아직 안 뽑은 사람 (모두 뽑았습니다)", bool(counts.unprinted))
        name = f" ({counts.current_name})" if counts.current_name else ""
        show(MODE_CURRENT, has_list, f"지금 보는 한 사람{name}" if counts.current
             else "지금 보는 한 사람 (명단에서 누르면 됩니다)", bool(counts.current))
        show(MODE_RANGES, has_list, "번호로 고르기", bool(counts.visible))
        show(MODE_ALL, has_list, f"명단에 보이는 {counts.visible or 0:,}명 모두", bool(counts.visible))
        show(MODE_COPIES, counts.copies_available, f"명단 없이 같은 카드 {copies:,}장")
        if not has_list:
            self.no_list_note.setText("명단을 열면 사람마다 이름과 사진이 바뀐 카드를 뽑을 수 있습니다.")
        elif not counts.copies_available:
            self.no_list_note.setText("")
        else:
            self.no_list_note.setText("")
        self.no_list_note.setVisible(bool(self.no_list_note.text()))
        if int(self.copies_field.value()) != copies:
            self.copies_field.set_value(copies, emit=False)
        self._sync_mode_boxes()

    def set_range_text(self, text: str) -> None:
        if self.range_edit.text() != text:
            self.range_edit.blockSignals(True)
            self.range_edit.setText(text)
            self.range_edit.blockSignals(False)

    def set_range_error(self, message: str) -> None:
        set_prop(self.range_edit, "invalid", "true" if message else "false")
        self.range_hint.setText(message or "명단 화면에 보이는 순서의 번호입니다. 쉼표와 줄표로 적습니다.")
        set_prop(self.range_hint, "tone", "error" if message else "")

    def set_stats(self, targets: str, printed: str, printed_caption: str, photo: str,
                  printed_tone: str = "", photo_tone: str = "", caption: str = "뽑을 사람") -> None:
        self.stat_targets.set(targets, caption)
        self.stat_printed.set(printed, printed_caption, printed_tone)
        self.stat_photo.set(photo, None, photo_tone)

    # ------------------------------------------------------------------ 사람 목록
    def set_people(self, rows: list[dict], excluded: int = 0, empty_text: str = "") -> None:
        """rows: {"id", "name", "sub", "tag", "tone"}."""
        self._filling = True
        keep = self.people.currentRow()
        self.people.clear()
        for row in rows:
            item = QListWidgetItem(row.get("name", ""))
            item.setData(ROLE_ID, row["id"])
            item.setData(ROLE_SUB, row.get("sub", ""))
            item.setData(ROLE_TAG, row.get("tag", ""))
            item.setData(ROLE_TAG_TONE, row.get("tone", ""))
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsDragEnabled)
            self.people.addItem(item)
        self._filling = False
        self.people.setVisible(bool(rows))
        self.people_empty.setText(empty_text)
        self.people_empty.setVisible(not rows and bool(empty_text))
        self.order_title.setText(f"뽑는 순서 ({len(rows):,}명)" if rows else "뽑는 순서")
        self.excluded_note.setText(f"직접 뺀 사람 {excluded:,}명")
        self.excluded_row.setVisible(excluded > 0)
        if rows and 0 <= keep < len(rows):
            self.people.blockSignals(True)
            self.people.setCurrentRow(keep)
            self.people.blockSignals(False)
        self._sync_tools()

    def update_tags(self, tags: dict[int, tuple[str, str]]) -> None:
        for row in range(self.people.count()):
            item = self.people.item(row)
            text, tone = tags.get(item.data(ROLE_ID), ("", ""))
            if item.data(ROLE_TAG) != text:
                item.setData(ROLE_TAG, text)
                item.setData(ROLE_TAG_TONE, tone)

    def people_ids(self) -> list[int]:
        return [self.people.item(row).data(ROLE_ID) for row in range(self.people.count())]

    def select_position(self, position: int) -> None:
        if 0 <= position < self.people.count() and self.people.currentRow() != position:
            self.people.blockSignals(True)
            self.people.setCurrentRow(position)
            self.people.blockSignals(False)
            self._sync_tools()

    def _selected_rows(self) -> list[int]:
        return sorted(index.row() for index in self.people.selectedIndexes())

    def _sync_tools(self) -> None:
        rows = self._selected_rows()
        count = self.people.count()
        self.remove_button.setEnabled(bool(rows))
        self.up_button.setEnabled(bool(rows) and rows[0] > 0)
        self.down_button.setEnabled(bool(rows) and rows[-1] < count - 1)

    def _on_current_row(self, row: int) -> None:
        if not self._filling and row >= 0:
            self.personSelected.emit(row)

    def _on_rows_moved(self, *_args) -> None:
        if not self._filling:
            self.people.viewport().update()
            self.peopleEdited.emit(self.people_ids())

    def _move_selected(self, delta: int) -> None:
        rows = self._selected_rows()
        if not rows:
            return
        ids = self.people_ids()
        if delta < 0 and rows[0] == 0 or delta > 0 and rows[-1] == len(ids) - 1:
            return
        for row in (rows if delta < 0 else reversed(rows)):
            ids[row + delta], ids[row] = ids[row], ids[row + delta]
        self.peopleEdited.emit(ids)
        self.people.clearSelection()
        for row in rows:
            self.people.item(row + delta).setSelected(True)
        self.people.blockSignals(True)
        self.people.setCurrentRow(rows[0] + delta, self.people.selectionModel().SelectionFlag.NoUpdate)
        self.people.blockSignals(False)
        self._sync_tools()

    def _remove_selected(self) -> None:
        rows = set(self._selected_rows())
        if not rows:
            return
        ids = [record_id for row, record_id in enumerate(self.people_ids()) if row not in rows]
        self.peopleEdited.emit(ids)

    def eventFilter(self, obj, event) -> bool:
        if obj is self.people and event.type() == QEvent.KeyPress and event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self._remove_selected()
            return True
        return super().eventFilter(obj, event)

    # ------------------------------------------------------------------ 미리 보기
    def set_preview_caption(self, name: str, sub: str, position: int, total: int) -> None:
        self.preview_name.setText(name)
        self.preview_sub.setText(sub)
        self.preview_sub.setVisible(bool(sub))
        self.position_label.setText(f"{position + 1:,} / {total:,}" if total else "")
        self.prev_button.setEnabled(total > 1 and position > 0)
        self.next_button.setEnabled(total > 1 and position < total - 1)
        visible = total > 1
        for widget in (self.prev_button, self.next_button, self.position_label):
            widget.setVisible(visible)

    def set_sides(self, dual: bool, current: str) -> None:
        self.side_control.setVisible(dual)
        if self.side_control.current() != current:
            self.side_control.set_current(current, emit=False)

    # ------------------------------------------------------------------ 프린터, 설정, 확인
    def set_printer(self, name: str, state: str, pill_text: str, detail: str, supplies: str,
                    can_change: bool) -> None:
        self.printer_name.setText(name)
        self.printer_pill.set_status(state, pill_text)
        self.printer_detail.setText(detail)
        self.printer_detail.setVisible(bool(detail))
        self.supplies.setText(supplies)
        self.supplies.setVisible(bool(supplies))
        self.change_printer.setText("프린터 바꾸기" if can_change else "프린터 화면 열기")

    def set_options_summary(self, text: str) -> None:
        self.options_summary.setText(text)

    def set_problems(self, problems: list[Problem], checking: bool = False) -> None:
        self._problems = list(problems)
        while self._findings_box.count():
            item = self._findings_box.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.hide()
                widget.setParent(None)
                widget.deleteLater()
        self.finding_rows = []
        shown = list(problems)
        if not shown:
            shown = [Problem("checking", "idle", "확인하고 있습니다.")] if checking else \
                [Problem("ok", "ok", "문제를 찾지 못했습니다. 바로 인쇄할 수 있습니다.")]
        for problem in shown:
            row = FindingRow(problem)
            row.triggered.connect(self.findingAction)
            self._findings_box.addWidget(row)
            self.finding_rows.append(row)
        errors = sum(1 for p in problems if p.tone == "error")
        warnings = sum(1 for p in problems if p.tone == "warn")
        parts = []
        if errors:
            parts.append(f"고쳐야 할 것 {errors}건")
        if warnings:
            parts.append(f"알아 둘 것 {warnings}건")
        self.findings_title.setText("인쇄 전 확인" + (f" ({', '.join(parts)})" if parts else ""))

    def problems(self) -> list[Problem]:
        return list(self._problems)

    def finding(self, code: str) -> FindingRow | None:
        return next((row for row in self.finding_rows if row.problem.code == code), None)

    def set_acknowledge(self, visible: bool, text: str, checked: bool) -> None:
        self.acknowledge.setVisible(visible)
        self.acknowledge.setText(text)
        if self.acknowledge.isChecked() != checked:
            self.acknowledge.blockSignals(True)
            self.acknowledge.setChecked(checked)
            self.acknowledge.blockSignals(False)

    def set_start(self, text: str, enabled: bool, reason: str, tone: str = "") -> None:
        self.start_button.setText(text)
        self.start_button.setEnabled(enabled)
        self.start_button.setToolTip("" if enabled else reason)
        self.reason.setText(reason)
        self.reason.setVisible(bool(reason))
        set_prop(self.reason, "tone", tone)
