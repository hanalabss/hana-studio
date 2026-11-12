"""명단 화면."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from PySide6.QtCore import QEvent, QModelIndex, QPoint, Qt, QTimer
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QMenu,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)

from data.errors import DataError
from data.export import export_csv, export_xlsx
from data.filters import Condition, Group, all_of, quick_search, unprinted_only, printed_only
from data.photos import set_photo_folder
from data.store import DataStore, PhotoAdjustment
from data.values import COLUMN_TYPES, is_empty
from design.binding import columns_used
from ui.shell import PAGE_DESIGN, PAGE_PRINT, BasePage, FooterAction
from ui.theme import SPACE
from ui.widgets import (
    Banner,
    FilterChip,
    IconButton,
    PrimaryButton,
    QuietButton,
    SearchField,
    SecondaryButton,
    Toast,
    confirm,
    inform,
    label,
)

from . import storage
from .dialogs import (
    AddColumnDialog,
    ConditionDialog,
    ExportDialog,
    MatchFieldDialog,
    NameDialog,
    PersonDialog,
    describe_filter,
    person_name,
    pick_photo_file,
)
from .empty import EmptyView
from .importing import IMPORT_SUFFIXES, FILE_FILTER, ImportWizard, design_fields
from .model import HIDDEN_OPTION, KIND_FIELD, KIND_STATUS, IdsFilter, PeopleTableModel, PhotoIndex, SpecRole
from .preview import Issue, PreviewPanel
from .inputs import TableFrame
from .table import HEADER_HEIGHT, PeopleTable
from .text import TYPE_NAMES, display_value, josa, people

CHIP_ALL, CHIP_UNPRINTED, CHIP_PRINTED, CHIP_MISSING = "all", "unprinted", "printed", "missing"
DEFAULT_COLUMNS = ("이름", "부서")


class PeoplePage(BasePage):
    """workspace의 명단을 보여 주고 고친다. 보는 조건과 체크한 사람은 workspace에 그대로 알린다."""

    def __init__(self, workspace, parent: QWidget | None = None, settings=None):
        super().__init__("명단", parent)
        self.workspace = workspace
        self._settings = settings
        self.photos = PhotoIndex()
        self.model = PeopleTableModel(self.photos, self)
        self._chip = CHIP_ALL
        self._search = ""
        self._condition: Group | None = None
        self._people_ids: list[int] | None = None      # show_people로 좁혀 보는 사람들
        self._pending_delete: list[int] = []
        self._own_change = False
        self._syncing_current = False
        self._select_all_banner = False
        self._toast: Toast | None = None
        self.wizard: ImportWizard | None = None
        self.setAcceptDrops(True)

        self._stack = QStackedLayout(self)
        self._stack.setContentsMargins(0, 0, 0, 0)
        self.empty_view = EmptyView(self)
        self.empty_view.importRequested.connect(lambda: self.start_import())
        self.empty_view.manualRequested.connect(lambda: self.start_manual())
        self.empty_view.openRequested.connect(self.open_list)
        self._stack.addWidget(self.empty_view)
        self._main = QWidget(self)
        self._stack.addWidget(self._main)
        self._build_main()

        self._search_timer = QTimer(self)
        self._search_timer.setSingleShot(True)
        self._search_timer.setInterval(180)
        self._search_timer.timeout.connect(lambda: self.set_search(self.search_field.text()))

        shortcut = QShortcut(QKeySequence.Find, self)
        shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        shortcut.activated.connect(self.focus_search)

        self.model.checkedChanged.connect(self._on_checked_changed)
        self.model.orderChanged.connect(self._on_order_changed)
        self.model.recordEdited.connect(self._on_record_edited)
        self.model.editFailed.connect(lambda message: self.toast(message, "error"))
        workspace.store_changed.connect(self._on_store_changed)
        workspace.columns_changed.connect(self._on_columns_changed)
        workspace.records_changed.connect(self._on_records_changed)
        workspace.current_record_changed.connect(self._on_workspace_current)
        self._on_store_changed()

    # ================================================================ 화면 만들기
    def _build_main(self) -> None:
        outer = QHBoxLayout(self._main)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        left = QVBoxLayout()
        left.setContentsMargins(20, 18, 20, 16)
        left.setSpacing(SPACE["md"])
        outer.addLayout(left, 1)

        head = QHBoxLayout()
        head.setSpacing(SPACE["md"])
        self.title_label = label("명단", "heading")
        head.addWidget(self.title_label)
        self.stats_label = label("", "muted")
        head.addWidget(self.stats_label)
        head.addStretch(1)
        self.import_button = SecondaryButton("엑셀·CSV 가져오기", icon="import")
        self.import_button.clicked.connect(lambda: self.start_import())
        self.add_button = SecondaryButton("한 명 추가", icon="plus")
        self.add_button.clicked.connect(lambda: self.add_person())
        self.more_button = SecondaryButton("더 보기", icon="chevron-down")
        self.more_button.clicked.connect(self._show_more_menu)
        for button in (self.import_button, self.add_button, self.more_button):
            head.addWidget(button)
        left.addLayout(head)

        tools = QHBoxLayout()
        tools.setSpacing(SPACE["sm"])
        self.search_field = SearchField("이름, 부서, 사번으로 찾기")
        self.search_field.setFixedWidth(280)
        self.search_field.setAccessibleName("명단에서 찾기")
        self.search_field.textChanged.connect(lambda *_: self._search_timer.start())
        self.search_field.returnPressed.connect(lambda: self.set_search(self.search_field.text()))
        self.search_field.installEventFilter(self)
        tools.addWidget(self.search_field)
        self.chips: dict[str, FilterChip] = {}
        for key, text, tone in ((CHIP_ALL, "전체", "neutral"), (CHIP_UNPRINTED, "아직 안 뽑음", "neutral"),
                                (CHIP_PRINTED, "인쇄함", "neutral"), (CHIP_MISSING, "사진 없음", "warn")):
            chip = FilterChip(text, 0, tone, checked=key == CHIP_ALL)
            chip.clicked.connect(lambda _checked=False, k=key: self.set_chip(k))
            self.chips[key] = chip
            tools.addWidget(chip)
        tools.addStretch(1)
        self.condition_button = QuietButton("조건으로 찾기", "sm", accent=True)
        self.condition_button.clicked.connect(self.open_condition_dialog)
        tools.addWidget(self.condition_button)
        self.condition_clear = IconButton("close", "조건 지우기", "sm")
        self.condition_clear.clicked.connect(lambda: self.set_condition(None))
        self.condition_clear.setVisible(False)
        tools.addWidget(self.condition_clear)
        left.addLayout(tools)

        self.people_banner = Banner("info", "", "전체 보기")
        self.people_banner.actionTriggered.connect(self.show_everyone)
        self.people_banner.setVisible(False)
        left.addWidget(self.people_banner)

        self.select_banner = Banner("info", "", "선택 해제")
        self.select_banner.actionTriggered.connect(self.clear_checked)
        self.select_banner.setVisible(False)
        left.addWidget(self.select_banner)

        self.table_card = TableFrame(HEADER_HEIGHT)
        self.table = PeopleTable(self.model, self.table_card)
        self.table_card.body.addWidget(self.table)
        left.addWidget(self.table_card, 1)
        self.table.currentRowChanged.connect(self._on_current_row)
        self.table.rowActivated.connect(self.edit_person_at)
        self.table.deleteRequested.connect(lambda: self.delete_people())
        self.table.photoPickRequested.connect(self._pick_photo_for)
        self.table.columnMenuRequested.connect(self._show_column_menu)
        self.table.allToggled.connect(self._on_all_toggled)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._show_row_menu)
        self.table.installEventFilter(self)

        self.blank = QWidget(self.table)
        blank = QVBoxLayout(self.blank)
        blank.setContentsMargins(SPACE["xl"], SPACE["xl"], SPACE["xl"], SPACE["xl"])
        blank.setSpacing(SPACE["sm"])
        blank.addStretch(2)
        self.blank_title = label("", "title")
        self.blank_title.setAlignment(Qt.AlignHCenter)
        blank.addWidget(self.blank_title)
        self.blank_text = label("", "secondary", wrap=True)
        self.blank_text.setAlignment(Qt.AlignHCenter)
        blank.addWidget(self.blank_text)
        blank.addSpacing(SPACE["sm"])
        self.blank_button = PrimaryButton("")
        self.blank_button.clicked.connect(self._blank_action)
        blank.addWidget(self.blank_button, 0, Qt.AlignHCenter)
        blank.addStretch(3)
        self.blank.setVisible(False)

        self.preview = PreviewPanel(self.workspace, self._main)
        self.preview.previousRequested.connect(lambda: self.step_person(-1))
        self.preview.nextRequested.connect(lambda: self.step_person(1))
        self.preview.issueActionRequested.connect(self._on_issue_action)
        self.preview.applyToAllRequested.connect(self.apply_adjustment_to_all)
        outer.addWidget(self.preview)

    # ================================================================ 틀과의 약속
    def on_enter(self) -> None:
        self._refresh_design()
        self._refresh_empty_view()

    def on_leave(self) -> None:
        self.commit_pending_delete()

    @property
    def store(self) -> DataStore | None:
        return self.workspace.store

    def toast(self, text: str, kind: str = "info", action_text: str | None = None, on_action=None,
              timeout_ms: int | None = None) -> Toast:
        shell = self.shell
        if shell is not None and hasattr(shell, "show_toast"):
            self._toast = shell.show_toast(text, kind, action_text, on_action, timeout_ms)
        else:
            self._toast = Toast.show_message(self, text, kind, action_text, on_action, timeout_ms)
        return self._toast

    def last_toast(self) -> Toast | None:
        return self._toast

    def _navigate(self, page_id: str) -> None:
        self.commit_pending_delete()
        if self.shell is not None:
            self.shell.navigate(page_id)

    def _sync_footer(self) -> None:
        store = self.store
        back = FooterAction("이전: 카드 꾸미기", lambda: self._navigate(PAGE_DESIGN), "secondary")
        if store is None:
            hint = "명단을 넣으면 사람마다 이름과 사진이 다른 카드를 뽑을 수 있습니다."
            forward = FooterAction("다음: 인쇄하기", lambda: self._navigate(PAGE_PRINT), enabled=False,
                                   tooltip="먼저 명단을 넣어 주세요.")
        else:
            checked = self.model.checked_ids()
            count = len(checked)
            if count == 0:
                hint = "인쇄할 사람의 왼쪽 칸을 체크하세요. 표 맨 위 칸을 누르면 보이는 사람을 모두 고릅니다."
                forward = FooterAction("다음: 인쇄하기", lambda: self._navigate(PAGE_PRINT), enabled=False,
                                       tooltip="인쇄할 사람을 한 명 이상 체크하면 넘어갈 수 있습니다.")
            else:
                missing = len(checked & self._missing_ids())
                hint = f"{people(count)}을 골랐습니다."
                if missing:
                    hint += f" 그중 {people(missing)}은 사진이 없어 사진 칸이 비어 나옵니다."
                forward = FooterAction(f"다음: {count:,}장 인쇄하기", lambda: self._navigate(PAGE_PRINT))
        self._footer_hint = hint
        self._footer_actions = [back, forward]
        self.footerChanged.emit()

    # ================================================================ 명단 열기·만들기
    def _design_path(self) -> str | None:
        return getattr(self.workspace.editor, "path", None)

    def _create_store(self, name_hint: str | None = None) -> DataStore:
        path = storage.new_list_path(self._design_path(), name_hint)
        return DataStore(path)

    def open_list(self, path: str | os.PathLike) -> bool:
        if not Path(path).exists():
            storage.forget_list(path, self._settings)
            self._refresh_empty_view()
            self.toast("그 명단 파일이 없어졌습니다. 옮겼거나 지운 것 같습니다.", "warn")
            return False
        try:
            store = DataStore(path)
        except DataError as exc:
            self.toast(str(exc), "error")
            return False
        self.commit_pending_delete()
        self.workspace.set_store(store)
        return True

    def close_list(self) -> None:
        self.commit_pending_delete()
        self.workspace.set_store(None)

    def start_manual(self, open_form: bool = True) -> None:
        """파일 없이 시작한다. 카드가 쓰는 칸으로 빈 명단을 만든다."""
        fields = design_fields(self.workspace.editor.document)
        try:
            store = self._create_store()
            if fields:
                for f in fields:
                    store.add_column(f.name, "photo" if f.is_photo else "text")
            else:
                for name in DEFAULT_COLUMNS:
                    store.add_column(name)
        except (DataError, OSError) as exc:
            self.toast(f"명단을 만들 수 없습니다. {exc}", "error")
            return
        self.workspace.set_store(store)
        if open_form:
            self.add_person()

    def start_import(self, path: str | os.PathLike | None = None, run: bool = True) -> ImportWizard | None:
        """가져오기 창을 연다. run=False면 창을 띄우지 않고 돌려준다(테스트, 스크린샷용)."""
        self.commit_pending_delete()
        if path is None:
            start = str(Path(self._design_path()).parent) if self._design_path() else ""
            path, _ = QFileDialog.getOpenFileName(self, "명단 파일 고르기", start, FILE_FILTER)
            if not path:
                return None
        wizard = ImportWizard(self.workspace, self._create_store, self)
        self.wizard = wizard
        wizard.busyChanged.connect(self.model.set_frozen)
        wizard.finished.connect(lambda _code, w=wizard: self._after_import(w))
        wizard.load_file(path)
        if run:
            wizard.exec()
        return wizard

    def _after_import(self, wizard: ImportWizard) -> None:
        self.model.set_frozen(False)
        if self.wizard is wizard:
            self.wizard = None
        if getattr(wizard, "_after_import_done", False):     # 창이 닫히며 finished가 한 번 더 올 수 있다
            return
        wizard._after_import_done = True
        outcome = wizard.outcome
        if outcome is None or wizard.store is None:
            return
        if wizard.store is not self.workspace.store:
            self.workspace.set_store(wizard.store)
            if outcome.result.inserted:
                self.model.replace_checked(wizard.store.ids())
                self._on_all_toggled(True)
        else:
            self.workspace.notify_columns_changed()
            self.workspace.notify_records_changed()
        wizard.deleteLater()

    # ================================================================ workspace 신호
    def _on_store_changed(self) -> None:
        store = self.store
        self._pending_delete = []
        self._chip, self._search, self._condition = CHIP_ALL, "", None
        self._people_ids = None
        self.people_banner.setVisible(False)
        self._select_all_banner = False
        self.search_field.blockSignals(True)
        self.search_field.clear()
        self.search_field.blockSignals(False)
        self.photos.configure(store, columns_used(self.workspace.editor.document))
        self.table.thumbs.forget()
        self.model.set_store(store)
        self._stack.setCurrentWidget(self._main if store is not None else self.empty_view)
        if store is not None and store.path != ":memory:":
            storage.remember_list(store.path, self._settings)
        self._refresh_empty_view()
        self._sync_all()
        self._select_first()

    def _just_opened(self) -> bool:
        """set_store가 이어서 보내는 columns_changed, records_changed인지(방금 다 읽었으니 넘긴다)."""
        return bool(getattr(self.workspace, "store_opening", False))

    def _on_columns_changed(self) -> None:
        if self.store is None or self._just_opened():
            return
        current = self.workspace.current_record_id
        self.photos.configure(self.store, columns_used(self.workspace.editor.document))
        names = set(self.store.column_names())
        if self._condition is not None:
            kept = [c for c in self._condition.items if not isinstance(c, Condition) or c.column in names]
            self._condition = Group(self._condition.mode, kept) if kept else None
        self.model.rebuild_columns()
        self._apply_view(keep=current)
        self._sync_all()

    def _on_records_changed(self) -> None:
        if self.store is None or self._just_opened() or self._own_change:
            return
        current = self.workspace.current_record_id
        self.photos.refresh_folders()
        checked = self.model.checked_ids()
        if checked:
            self.model.replace_checked(checked.intersection(self.store.ids()))
        self._apply_view(keep=current)
        self._sync_all()

    def _on_workspace_current(self, record: dict | None) -> None:
        if self._syncing_current or record is None or self.store is None:
            return
        row = self.model.row_of(record["_id"])
        if row >= 0 and row != self.table.current_row():
            self._syncing_current = True
            try:
                self.table.go_to_row(row)
            finally:
                self._syncing_current = False
            self._show_record(row)

    def _refresh_design(self) -> None:
        """카드 디자인이 바뀌었을 수 있으니 카드가 쓰는 칸을 다시 본다."""
        if self.store is None:
            return
        before = self.photos.primary
        self.photos.configure(self.store, columns_used(self.workspace.editor.document))
        if self.photos.primary != before:
            self.model.rebuild_columns()
            self._apply_view(keep=self.workspace.current_record_id)
        self._sync_all()
        self.preview.refresh()

    def _refresh_empty_view(self) -> None:
        self.empty_view.set_recent(storage.recent_lists(self._design_path(), self._settings))
        folder = storage.new_list_path(self._design_path(), "명단").parent
        if self._design_path():
            where = f"명단은 카드 파일과 같은 폴더({folder})에 알아서 저장됩니다."
        else:
            where = f"명단은 알아서 저장됩니다. 저장 위치: {folder}"
        self.empty_view.set_details(where)

    # ================================================================ 보기 조건
    def _base_filter(self):
        return IdsFilter(frozenset(self._pending_delete), negate=True) if self._pending_delete else None

    def _missing_ids(self) -> set[int]:
        missing = self.photos.missing_ids()
        return missing.difference(self._pending_delete) if self._pending_delete else missing

    def compose_filter(self):
        parts = [self._base_filter()]
        if self._chip == CHIP_UNPRINTED:
            parts.append(unprinted_only())
        elif self._chip == CHIP_PRINTED:
            parts.append(printed_only())
        elif self._chip == CHIP_MISSING:
            parts.append(IdsFilter(frozenset(self._missing_ids())))
        if self._search.strip():
            parts.append(quick_search(self._search.strip()))
        if self._condition is not None:
            parts.append(self._condition)
        if self._people_ids is not None:
            parts.append(IdsFilter(frozenset(self._people_ids)))
        parts = [p for p in parts if p is not None]
        if not parts:
            return None
        return parts[0] if len(parts) == 1 else all_of(*parts)

    def _apply_view(self, keep: int | None = None) -> None:
        scroll = self.table.verticalScrollBar().value()
        view_filter = self.compose_filter()
        self.model.set_view(view_filter, self.model.view_order)
        self.workspace.set_view(self.model.view_filter, self.model.view_order)
        row = self.model.row_of(keep) if keep is not None else -1
        if row >= 0:
            self._syncing_current = True
            try:
                self.table.go_to_row(row)
            finally:
                self._syncing_current = False
            self.table.verticalScrollBar().setValue(scroll)
            self.table.scrollTo(self.model.index(row, 0))
            self._show_record(row)
        else:
            self._select_first()
        self._push_checked()

    def _select_first(self) -> None:
        if self.store is not None and self.model.rowCount() > 0:
            self.table.go_to_row(0, 0)
            self._show_record(0)
        else:
            self._show_record(-1)

    def chip(self) -> str:
        return self._chip

    def set_chip(self, key: str) -> None:
        if key not in self.chips:
            raise KeyError(key)
        self._chip = key
        for name, chip in self.chips.items():
            chip.setChecked(name == key)
        self._view_changed()

    def search_text(self) -> str:
        return self._search

    def set_search(self, text: str) -> None:
        self._search_timer.stop()
        if self.search_field.text() != text:
            self.search_field.blockSignals(True)
            self.search_field.setText(text)
            self.search_field.blockSignals(False)
        if text == self._search:
            return
        self._search = text
        self._view_changed()

    def condition(self) -> Group | None:
        return self._condition

    def set_condition(self, group: Group | None) -> None:
        if group is not None and not group.items:
            group = None
        self._condition = group
        self._view_changed()

    def clear_filters(self) -> None:
        self._chip, self._condition = CHIP_ALL, None
        self._people_ids = None
        self.people_banner.setVisible(False)
        for name, chip in self.chips.items():
            chip.setChecked(name == CHIP_ALL)
        self.set_search("")
        self._search = ""
        self._view_changed()

    def show_people(self, record_ids: Iterable[int], note: str | None = None) -> bool:
        """이 사람들만 보여 준다(인쇄·기록 화면의 "명단에서 보기"). 띠의 '전체 보기'로 되돌린다.

        note: 띠에 적을 말. 없으면 "확인할 N명만 보고 있습니다. …"
        명단에 한 명도 남아 있지 않으면 아무것도 바꾸지 않고 False.
        """
        store = self.store
        if store is None:
            return False
        self.commit_pending_delete()
        wanted = list(dict.fromkeys(int(i) for i in record_ids))
        existing = set(store.ids())
        ids = [i for i in wanted if i in existing]
        if not ids:
            self.toast("그 사람들은 지금 열려 있는 명단에 없습니다.", "warn")
            return False
        self._chip, self._condition, self._search = CHIP_ALL, None, ""
        for name, chip in self.chips.items():
            chip.setChecked(name == CHIP_ALL)
        self._search_timer.stop()
        self.search_field.blockSignals(True)
        self.search_field.clear()
        self.search_field.blockSignals(False)
        self._people_ids = ids
        text = note or f"확인할 {people(len(ids))}만 보고 있습니다. 다 보았으면 '전체 보기'를 누르세요."
        if len(ids) < len(wanted):
            text += f" 나머지 {people(len(wanted) - len(ids))}은 명단에서 지워져 보이지 않습니다."
        self.people_banner.set_text(text)
        self.people_banner.setVisible(True)
        self._select_all_banner = False
        self._apply_view(keep=ids[0])
        self._sync_stats()
        return True

    def shown_people(self) -> list[int] | None:
        """show_people로 좁혀 보는 중이면 그 사람들, 아니면 None."""
        return None if self._people_ids is None else list(self._people_ids)

    def show_everyone(self) -> None:
        """show_people로 좁힌 것만 푼다(찾는 글자와 조건은 그대로)."""
        if self._people_ids is None:
            return
        self._people_ids = None
        self.people_banner.setVisible(False)
        self._view_changed()

    def _view_changed(self) -> None:
        if self.store is None:
            return
        self._select_all_banner = False
        self._apply_view(keep=self.workspace.current_record_id)
        self._sync_stats()

    def open_condition_dialog(self) -> None:
        if self.store is None:
            return
        dialog = ConditionDialog(self.store, self._condition, self, self._base_filter())
        if dialog.exec() == QDialog.Accepted:
            self.set_condition(dialog.result_filter())

    def focus_search(self) -> None:
        if self.store is not None:
            self.search_field.setFocus(Qt.ShortcutFocusReason)
            self.search_field.selectAll()

    def _on_order_changed(self) -> None:
        self.workspace.view_order = self.model.view_order
        row = self.model.row_of(self.workspace.current_record_id)
        if row >= 0:
            self._syncing_current = True
            try:
                self.table.go_to_row(row)
            finally:
                self._syncing_current = False
            self._show_record(row)
        else:
            self._select_first()
        self._push_checked()

    # ================================================================ 숫자, 안내 갱신
    def _sync_all(self) -> None:
        self._sync_stats()
        self._sync_issues()

    def counts(self) -> dict[str, int]:
        store = self.store
        if store is None:
            return {CHIP_ALL: 0, CHIP_UNPRINTED: 0, CHIP_PRINTED: 0, CHIP_MISSING: 0}
        base = self._base_filter()
        total = store.count(base)
        unprinted = store.count(unprinted_only() if base is None else all_of(base, unprinted_only()))
        return {CHIP_ALL: total, CHIP_UNPRINTED: unprinted, CHIP_PRINTED: total - unprinted,
                CHIP_MISSING: len(self._missing_ids())}

    def _sync_stats(self) -> None:
        store = self.store
        if store is None:
            self._sync_footer()
            return
        counts = self.counts()
        for key, chip in self.chips.items():
            chip.set_count(counts[key])
        self.chips[CHIP_MISSING].setVisible(bool(self.photos.columns))
        total, shown, checked = counts[CHIP_ALL], self.model.rowCount(), len(self.model.checked_ids())
        text = people(total) if shown == total else f"{people(total)} 중 {people(shown)} 보는 중"
        if checked:
            text += f" · {people(checked)} 선택"
        self.stats_label.setText(text)

        names = [s.name for s in self.model.specs if s.kind == KIND_FIELD and not s.is_photo][:3]
        self.search_field.setPlaceholderText(f"{', '.join(names)}{'으로' if _ro(names[-1]) else '로'} 찾기" if names else "찾기")

        if self._condition is not None:
            self.condition_button.setText(f"조건 {len(self._condition.items)}개로 찾는 중")
            self.condition_button.setToolTip(describe_filter(self._condition))
        else:
            self.condition_button.setText("조건으로 찾기")
            self.condition_button.setToolTip("부서가 영업팀인 사람처럼, 조건을 정해서 찾습니다.")
        self.condition_clear.setVisible(self._condition is not None)

        filtered = self.model.view_filter is not None and (self._chip != CHIP_ALL or self._search.strip()
                                                           or self._condition is not None
                                                           or self._people_ids is not None)
        show_banner = self._select_all_banner and checked > 0
        if show_banner:
            if filtered:
                self.select_banner.set_text(f"현재 조건의 {people(shown)}을 모두 선택했습니다.")
            else:
                self.select_banner.set_text(f"{people(shown)}을 모두 선택했습니다. 빼고 싶은 사람은 체크를 지우세요.")
        self.select_banner.setVisible(show_banner)

        if shown == 0:
            if total == 0:
                self.blank_title.setText("아직 아무도 없습니다")
                self.blank_text.setText("한 명씩 직접 넣거나, 위의 '엑셀·CSV 가져오기'로 한꺼번에 넣을 수 있습니다.")
                self.blank_button.setText("한 명 추가")
            else:
                self.blank_title.setText("맞는 사람이 없습니다")
                self.blank_text.setText("찾는 글자나 조건을 바꿔 보세요.")
                self.blank_button.setText("전체 보기")
        self.blank.setVisible(shown == 0)
        self._place_blank()
        self.preview.set_position(self.table.current_row(), shown)
        self._sync_footer()

    def _blank_action(self) -> None:
        if self.counts()[CHIP_ALL] == 0:
            self.add_person()
        else:
            self.clear_filters()

    def _place_blank(self) -> None:
        top = self.table.horizontalHeader().height()
        self.blank.setGeometry(0, top, self.table.width(), max(0, self.table.height() - top))
        self.blank.raise_()

    def issues(self) -> list[Issue]:
        store = self.store
        if store is None:
            return []
        result: list[Issue] = []
        base = self._base_filter()
        names = store.column_names()
        fields = design_fields(self.workspace.editor.document)
        for f in fields:
            if f.name not in names:
                result.append(Issue("unmatched_field", f"카드의 '{f.name}' 칸에 넣을 항목이 명단에 없습니다.",
                                    "맞추기", f.name))
        missing = len(self._missing_ids())
        if missing:
            result.append(Issue("missing_photo", f"사진이 없는 사람이 {people(missing)} 있습니다.", "모아 보기"))
        for f in fields:
            if f.name not in names or f.is_photo:
                continue
            empty = Condition(f.name, "empty")
            count = store.count(empty if base is None else all_of(base, empty))
            if count:
                result.append(Issue("empty_field", f"{josa(f.name, '이', '가')} 비어 있는 사람이 {people(count)} 있습니다.",
                                    "모아 보기", f.name))
        return result

    def _sync_issues(self) -> None:
        self.preview.set_issues(self.issues())

    def _on_issue_action(self, issue: Issue) -> None:
        if issue.kind == "missing_photo":
            self.set_chip(CHIP_MISSING)
        elif issue.kind == "empty_field":
            self._chip = CHIP_ALL
            for name, chip in self.chips.items():
                chip.setChecked(name == CHIP_ALL)
            self.set_condition(Group("and", [Condition(issue.payload, "empty")]))
        elif issue.kind == "unmatched_field":
            self.match_field(issue.payload)

    def match_field(self, field_name: str, choice: str | None | bool = False) -> None:
        """카드의 칸에 명단 항목을 맞춘다. choice: 항목 이름, None(새로 만들기), False(물어보기)."""
        store = self.store
        if store is None:
            return
        used = columns_used(self.workspace.editor.document)
        is_photo = any(f.name == field_name and f.is_photo for f in design_fields(self.workspace.editor.document))
        candidates = [c.name for c in store.columns() if c.name not in used and (c.type == "photo") == is_photo]
        if choice is False:
            dialog = MatchFieldDialog(field_name, candidates, self)
            if dialog.exec() != QDialog.Accepted:
                return
            choice = dialog.choice()
        try:
            if choice is None:
                store.add_column(field_name, "photo" if is_photo else "text")
            else:
                store.rename_column(choice, field_name)
        except DataError as exc:
            self.toast(str(exc), "error")
            return
        self.workspace.notify_columns_changed()
        self.toast(f"카드의 '{field_name}' 칸을 명단과 맞췄습니다.", "ok")

    # ================================================================ 체크
    def _on_checked_changed(self) -> None:
        self._select_all_banner = False
        self._push_checked()
        self._sync_stats()

    def _on_all_toggled(self, checked: bool) -> None:
        self._select_all_banner = checked
        self._sync_stats()

    def _push_checked(self) -> None:
        store = self.store
        checked = self.model.checked_ids()
        if store is None or not checked:
            self.workspace.set_checked_ids([])
            return
        ordered = [i for i in store.ids(self._base_filter(), self.model.view_order) if i in checked]
        self.workspace.set_checked_ids(ordered)

    def toggle_all_in_view(self) -> None:
        self._on_all_toggled(self.model.toggle_all_in_view())

    def clear_checked(self) -> None:
        self._select_all_banner = False
        self.model.replace_checked([])
        self._sync_stats()

    # ================================================================ 지금 보는 사람
    def _on_current_row(self, row: int) -> None:
        if not self._syncing_current:
            self._show_record(row)

    def _show_record(self, row: int) -> None:
        record = self.model.record_at(row) if row >= 0 else None
        self._syncing_current = True
        try:
            self.workspace.set_current_record(record["_id"] if record else None)
        finally:
            self._syncing_current = False
        self.preview.set_position(row if record else -1, self.model.rowCount())
        self.preview.set_record(record, max(0, row))

    def step_person(self, step: int) -> None:
        row = self.table.current_row() + step
        if 0 <= row < self.model.rowCount():
            self.table.go_to_row(row)

    def _on_record_edited(self, record_id: int, column: str) -> None:
        self._own_change = True
        try:
            self.workspace.notify_records_changed()
        finally:
            self._own_change = False
        self._sync_all()
        if record_id == self.workspace.current_record_id:
            self.preview.set_record(self.model.record_at(self.table.current_row()), max(0, self.table.current_row()))

    def apply_adjustment_to_all(self, column: str, adjustment: PhotoAdjustment, ask: bool = True) -> None:
        store = self.store
        if store is None:
            return
        total = store.count()
        if ask and not confirm(
                self, "모든 사람의 사진을 똑같이 맞출까요?",
                f"{people(total)} 모두의 사진을 지금 이 사람과 같은 크기와 위치로 맞춥니다. "
                "사람마다 따로 맞춰 둔 것은 사라집니다.", f"{people(total)}에게 적용"):
            return
        if adjustment == PhotoAdjustment():
            store.clear_photo_adjustments(column)
        else:
            store.apply_photo_adjustment_to_all(column, adjustment)
        self.preview.refresh()
        self.toast(f"{people(total)}의 사진을 같은 크기와 위치로 맞췄습니다.", "ok")

    # ================================================================ 사람 넣기·고치기·지우기
    def add_person(self, run: bool = True) -> PersonDialog | None:
        if self.store is None:
            return None
        self.commit_pending_delete()
        dialog = PersonDialog(self.workspace, self.photos, None, self)
        dialog.saved.connect(lambda _id: self.workspace.notify_records_changed())
        dialog.finished.connect(lambda _code, d=dialog: self._after_person(d))
        if run:
            dialog.exec()
        return dialog

    def edit_person_at(self, row: int, run: bool = True) -> PersonDialog | None:
        record = self.model.record_at(row)
        if record is None or self.store is None:
            return None
        dialog = PersonDialog(self.workspace, self.photos, self.store.get(record["_id"]), self)
        dialog.saved.connect(lambda _id: self.workspace.notify_records_changed())
        dialog.finished.connect(lambda _code, d=dialog: self._after_person(d))
        if run:
            dialog.exec()
        return dialog

    def _after_person(self, dialog: PersonDialog) -> None:
        if dialog.saved_ids:
            self.reveal(dialog.saved_ids[-1], "넣었습니다" if len(dialog.saved_ids) == 1 else
                        f"{people(len(dialog.saved_ids))}을 넣었습니다")
        dialog.deleteLater()

    def reveal(self, record_id: int, done_text: str = "") -> None:
        """그 사람 줄로 간다. 지금 조건에 안 맞아 안 보이면 알려 준다."""
        row = self.model.row_of(record_id)
        if row >= 0:
            self.table.go_to_row(row)
            self.table.setFocus()
        elif done_text:
            self.toast(f"{done_text}. 지금 찾는 조건에는 맞지 않아 보이지 않습니다.", "info",
                       "전체 보기", lambda: (self.clear_filters(), self.reveal(record_id)))

    def _target_ids(self, ids: Iterable[int] | None) -> list[int]:
        if ids is not None:
            return [int(i) for i in ids]
        selected = self.table.selected_ids()
        if selected:
            return selected
        current = self.model.id_at(self.table.current_row())
        return [current] if current is not None else []

    def delete_people(self, ids: Iterable[int] | None = None) -> int:
        """지운다. 알림이 떠 있는 동안에는 되돌릴 수 있고, 그 뒤에 파일에서 정말 지운다."""
        store = self.store
        targets = self._target_ids(ids)
        if store is None or not targets:
            return 0
        self.commit_pending_delete()
        row = self.table.current_row()
        names = [person_name(store.get(i), store) for i in targets[:1]]
        self._pending_delete = targets
        self.model.set_checked(targets, False)
        self._apply_view(keep=None)
        if self.model.rowCount():
            self.table.go_to_row(min(max(0, row), self.model.rowCount() - 1))
        self._sync_all()
        if len(targets) == 1 and names[0]:
            text = f"{names[0]} 님을 명단에서 지웠습니다."
        else:
            text = f"{people(len(targets))}을 명단에서 지웠습니다."
        toast = self.toast(text, "info", "되돌리기", self.undo_delete, timeout_ms=8000)
        toast.closed.connect(lambda t=tuple(targets): self._toast_closed(t))
        return len(targets)

    def _toast_closed(self, targets: tuple) -> None:
        if tuple(self._pending_delete) == targets:
            QTimer.singleShot(0, lambda: self._commit_if(targets))

    def _commit_if(self, targets: tuple) -> None:
        if tuple(self._pending_delete) == targets:
            self.commit_pending_delete()

    def undo_delete(self) -> bool:
        if not self._pending_delete:
            return False
        restored, self._pending_delete = self._pending_delete, []
        self._apply_view(keep=restored[0])
        self._sync_all()
        return True

    def commit_pending_delete(self) -> None:
        store = self.store
        targets, self._pending_delete = self._pending_delete, []
        if store is None or not targets:
            return
        try:
            store.delete(targets)
        except DataError as exc:
            self.toast(str(exc), "error")
        self._own_change = True
        try:
            self.workspace.notify_records_changed()
        finally:
            self._own_change = False
        self.photos.invalidate()
        self._apply_view(keep=self.workspace.current_record_id)
        self._sync_all()

    def duplicate_people(self, ids: Iterable[int] | None = None) -> list[int]:
        store = self.store
        targets = self._target_ids(ids)
        if store is None or not targets:
            return []
        created: list[int] = []
        columns = store.columns()
        try:
            with store.transaction():
                for record_id in targets:
                    record = store.get(record_id)
                    if record is None:
                        continue
                    new_id = store.insert({c.name: record.get(c.name) for c in columns})
                    for column in columns:
                        if column.type == "photo":
                            adjustment = store.get_photo_adjustment(record_id, column.name)
                            if adjustment != PhotoAdjustment():
                                store.set_photo_adjustment(new_id, column.name, adjustment)
                    created.append(new_id)
        except DataError as exc:
            self.toast(str(exc), "error")
            return []
        self.workspace.notify_records_changed()
        if created:
            self.reveal(created[-1], f"{people(len(created))}을 복제했습니다")
        return created

    def _pick_photo_for(self, index: QModelIndex) -> None:
        spec = index.data(SpecRole)
        if spec is None or not spec.is_photo:
            return
        folder = self.photos.folder(spec.name)
        path = pick_photo_file(self, str(folder) if folder else "")
        if path:
            self.set_photo(index, path)

    def set_photo(self, index: QModelIndex, path: str) -> None:
        spec = index.data(SpecRole)
        folder = self.photos.folder(spec.name)
        chosen = Path(path)
        value = chosen.name if folder is not None and chosen.parent.resolve() == Path(folder).resolve() else str(chosen)
        self.photos.refresh_folders()
        self.table.thumbs.forget(str(chosen))
        self.model.setData(index, value, Qt.EditRole)

    # ================================================================ 항목(열)
    def add_column(self, name: str | None = None, type: str = "text", options: dict | None = None) -> bool:
        store = self.store
        if store is None:
            return False
        if name is None:
            dialog = AddColumnDialog(self)
            if dialog.exec() != QDialog.Accepted:
                return False
            name, type, options = dialog.name(), dialog.column_type(), dialog.options()
        try:
            store.add_column(name, type, options or {})
        except DataError as exc:
            self.toast(str(exc), "error")
            return False
        self.workspace.notify_columns_changed()
        return True

    def rename_column(self, old: str, new: str | None = None) -> bool:
        store = self.store
        if store is None:
            return False
        if new is None:
            used = old in columns_used(self.workspace.editor.document)
            note = (f"카드의 '{old}' 칸과 이어져 있습니다. 이름을 바꾸면 카드에서 그 칸을 다시 맞춰야 합니다."
                    if used else None)
            dialog = NameDialog(f"'{old}' 이름 바꾸기", note, "새 이름", old, "이름 바꾸기", self)
            if dialog.exec() != QDialog.Accepted:
                return False
            new = dialog.text()
        if new == old:
            return False
        try:
            store.rename_column(old, new)
        except DataError as exc:
            self.toast(str(exc), "error")
            return False
        if self._condition is not None:
            items = [Condition(new, c.op, c.value) if isinstance(c, Condition) and c.column == old else c
                     for c in self._condition.items]
            self._condition = Group(self._condition.mode, items)
        order = self.model.view_order
        if order:
            self.model.set_view(self.model.view_filter, [(new if n == old else n, d) for n, d in order])
        self.workspace.notify_columns_changed()
        return True

    def change_column_type(self, name: str, type: str) -> bool:
        store = self.store
        if store is None or type not in COLUMN_TYPES:
            return False
        try:
            column = store.column(name)
            options = dict(column.options)
            if type == "list" and not options.get("choices"):
                options["choices"] = self._distinct_values(name)
            failed = store.set_column_type(name, type, options)
        except DataError as exc:
            self.toast(str(exc), "error")
            return False
        self.workspace.notify_columns_changed()
        self.workspace.notify_records_changed()
        kind = TYPE_NAMES[type] + ("으로" if _ro(TYPE_NAMES[type]) else "로")
        if type == "photo" and not options.get("folder"):
            self.toast(f"'{name}' 항목을 {kind} 바꿨습니다. 사진이 들어 있는 폴더를 알려 주세요.", "info",
                       "사진 폴더 고르기", lambda: self.change_photo_folder(name), timeout_ms=10000)
        elif failed:
            self.toast(f"'{name}' 항목을 {kind} 바꿨습니다. {failed}칸은 {kind} 읽을 수 없어 적힌 그대로 두었습니다.", "warn")
        else:
            self.toast(f"'{name}' 항목을 {kind} 바꿨습니다.", "ok")
        return True

    def _distinct_values(self, name: str, limit: int = 40) -> list[str]:
        values: dict[str, None] = {}
        offset, chunk = 0, 2000
        while len(values) <= limit:
            rows = self.store.rows(None, None, offset, chunk)
            for record in rows:
                if not is_empty(record.get(name)):
                    values[display_value(record[name])] = None
            if len(rows) < chunk:
                break
            offset += chunk
        return list(values)[:limit]

    def remove_column(self, name: str, ask: bool = True) -> bool:
        store = self.store
        if store is None:
            return False
        total = store.count()
        used = name in columns_used(self.workspace.editor.document)
        message = f"'{name}' 항목과 {people(total)}의 {name} 값이 모두 지워집니다. 되돌릴 수 없습니다."
        if used:
            message += f" 카드의 '{name}' 칸은 비어 나오게 됩니다."
        if ask and not confirm(self, f"'{name}' 항목을 지울까요?", message, f"'{name}' 항목 지우기", danger=True):
            return False
        try:
            store.remove_column(name)
        except DataError as exc:
            self.toast(str(exc), "error")
            return False
        self.workspace.notify_columns_changed()
        return True

    def hide_column(self, name: str, hidden: bool = True) -> None:
        store = self.store
        if store is None:
            return
        options = dict(store.column(name).options)
        if hidden:
            options[HIDDEN_OPTION] = True
        else:
            options.pop(HIDDEN_OPTION, None)
        store.set_column_options(name, options)
        self.workspace.notify_columns_changed()
        if hidden:
            self.toast(f"'{name}' 항목을 숨겼습니다. 값은 그대로 있습니다.", "info", "다시 보이기",
                       lambda: self.hide_column(name, False))

    def change_photo_folder(self, column: str, folder: str | None = None) -> bool:
        store = self.store
        if store is None:
            return False
        if folder is None:
            current = self.photos.folder(column)
            folder = QFileDialog.getExistingDirectory(self, "사진이 들어 있는 폴더 고르기", str(current) if current else "")
            if not folder:
                return False
        set_photo_folder(store, column, folder)
        self.table.thumbs.forget()
        self.workspace.notify_columns_changed()
        self.workspace.notify_records_changed()
        missing = len(self._missing_ids())
        total = store.count()
        if missing:
            self.toast(f"{people(total)} 중 {people(total - missing)}의 사진을 찾았습니다. {people(missing)}은 못 찾았습니다.", "warn")
        else:
            self.toast(f"{people(total)} 모두 사진을 찾았습니다.", "ok")
        return True

    def _show_column_menu(self, section: int, pos: QPoint) -> None:
        spec = self.model.headerData(section, Qt.Horizontal, SpecRole)
        if spec is None or self.store is None:
            return
        menu = QMenu(self)
        if spec.kind == KIND_STATUS:
            menu.addAction("인쇄 안 한 사람부터 보기", lambda: self.model.sort(section, Qt.AscendingOrder))
            menu.addAction("인쇄한 사람부터 보기", lambda: self.model.sort(section, Qt.DescendingOrder))
            menu.addSeparator()
            menu.addAction("인쇄 기록 지우기…", lambda: self.reset_print_status())
            menu.exec(pos)
            return
        name = spec.name
        if not spec.is_photo:
            menu.addAction("가나다순으로 보기", lambda: self.model.sort(section, Qt.AscendingOrder))
            menu.addAction("거꾸로 보기", lambda: self.model.sort(section, Qt.DescendingOrder))
            menu.addSeparator()
        else:
            menu.addAction("사진 폴더 바꾸기…", lambda: self.change_photo_folder(name))
            menu.addSeparator()
        menu.addAction("이름 바꾸기…", lambda: self.rename_column(name))
        types = menu.addMenu("종류 바꾸기")
        for key in COLUMN_TYPES:
            action = types.addAction(TYPE_NAMES[key], lambda k=key: self.change_column_type(name, k))
            action.setCheckable(True)
            action.setChecked(key == spec.type)
        menu.addAction("숨기기", lambda: self.hide_column(name))
        menu.addAction("지우기…", lambda: self.remove_column(name))
        menu.addSeparator()
        menu.addAction("항목 추가…", lambda: self.add_column())
        menu.exec(pos)

    # ================================================================ 메뉴
    def _show_row_menu(self, pos: QPoint) -> None:
        index = self.table.indexAt(pos)
        if not index.isValid() or self.store is None:
            return
        if index.row() not in self.table.selected_rows():
            self.table.go_to_row(index.row(), index.column())
        ids = self.table.selected_ids()
        count = len(ids)
        menu = QMenu(self)
        if count == 1:
            menu.addAction("이 사람 고치기…", lambda: self.edit_person_at(index.row()))
        all_checked = all(self.model.is_checked(i) for i in ids)
        who = "이 사람" if count == 1 else f"고른 {people(count)}"
        menu.addAction(f"{who} 인쇄에서 빼기" if all_checked else f"{who} 인쇄할 사람으로 체크",
                       lambda: self.model.set_checked(ids, not all_checked))
        menu.addSeparator()
        menu.addAction(f"{who} 복제", lambda: self.duplicate_people(ids))
        menu.addAction(f"{who} 지우기", lambda: self.delete_people(ids))
        menu.exec(self.table.viewport().mapToGlobal(pos))

    def _show_more_menu(self) -> None:
        store = self.store
        if store is None:
            return
        menu = QMenu(self)
        menu.addAction("명단 내보내기…", lambda: self.export_dialog())
        menu.addSeparator()
        menu.addAction("항목 추가…", lambda: self.add_column())
        hidden = self.model.hidden_columns()
        if hidden:
            shown = menu.addMenu("숨긴 항목 다시 보이기")
            for name in hidden:
                shown.addAction(name, lambda n=name: self.hide_column(n, False))
        for column in store.columns():
            if column.type == "photo":
                menu.addAction(f"'{column.name}' 폴더 바꾸기…", lambda n=column.name: self.change_photo_folder(n))
        menu.addSeparator()
        checked = len(self.model.checked_ids())
        menu.addAction(f"선택한 {people(checked)}의 인쇄 기록 지우기…" if checked else "인쇄 기록 모두 지우기…",
                       lambda: self.reset_print_status())
        menu.addSeparator()
        menu.addAction("명단 파일이 있는 곳 보기", self._show_location)
        menu.addAction("이 명단 닫기", self.close_list)
        menu.exec(self.more_button.mapToGlobal(QPoint(0, self.more_button.height() + 4)))

    def _show_location(self) -> None:
        store = self.store
        if store is not None:
            inform(self, "명단 파일이 있는 곳",
                   f"{store.path}\n\n고친 내용은 바로바로 이 파일에 저장됩니다. 따로 저장할 필요가 없습니다.")

    # ================================================================ 내보내기, 인쇄 기록
    def export_to(self, path: str | os.PathLike, scope: str = "all", include_status: bool = False) -> int:
        """scope: "all" | "view" | "checked". 내보낸 사람 수를 돌려준다."""
        store = self.store
        if store is None:
            return 0
        self.commit_pending_delete()
        if scope == "checked":
            export_filter = IdsFilter(frozenset(self.model.checked_ids()))
        elif scope == "view":
            export_filter = self.model.view_filter
        else:
            export_filter = None
        writer = export_csv if str(path).lower().endswith(".csv") else export_xlsx
        return writer(store, path, export_filter, self.model.view_order, include_print_status=include_status)

    def export_dialog(self) -> None:
        store = self.store
        if store is None:
            return
        total = store.count(self._base_filter())
        shown = self.model.rowCount()
        dialog = ExportDialog(total, shown, len(self.model.checked_ids()), shown != total, self)
        if dialog.exec() != QDialog.Accepted:
            return
        suffix = dialog.file_format()
        name = f"{Path(store.path).stem}.{suffix}" if store.path != ":memory:" else f"명단.{suffix}"
        kinds = "엑셀 파일 (*.xlsx)" if suffix == "xlsx" else "CSV 파일 (*.csv)"
        path, _ = QFileDialog.getSaveFileName(self, "명단 내보내기", name, kinds)
        if not path:
            return
        if not path.lower().endswith("." + suffix):
            path += "." + suffix
        try:
            count = self.export_to(path, dialog.scope(), dialog.include_status())
        except DataError as exc:
            self.toast(str(exc), "error")
            return
        self.toast(f"{people(count)}을 {Path(path).name} 파일로 내보냈습니다.", "ok")

    def reset_print_status(self, ids: Iterable[int] | None = None, ask: bool = True) -> int:
        """체크한 사람이 있으면 그 사람들만, 없으면 모두의 인쇄 기록을 지운다."""
        store = self.store
        if store is None:
            return 0
        self.commit_pending_delete()
        targets = list(ids) if ids is not None else sorted(self.model.checked_ids())
        everyone = not targets
        count = store.count() if everyone else len(targets)
        who = f"모든 사람({people(count)})" if everyone else f"선택한 {people(count)}"
        if ask and not confirm(
                self, "인쇄 기록을 지울까요?",
                f"{who}의 인쇄 기록을 지웁니다. 모두 '아직'으로 돌아가고, 언제 몇 장 뽑았는지는 되돌릴 수 없습니다.",
                f"{people(count)}의 기록 지우기", danger=True):
            return 0
        changed = store.reset_print_status(None if everyone else targets)
        self.workspace.notify_records_changed()
        self.toast(f"{people(count)}의 인쇄 기록을 지웠습니다.", "ok")
        return changed

    # ================================================================ 끌어다 놓기, 키
    @staticmethod
    def _dropped_file(event) -> str | None:
        mime = event.mimeData()
        if not mime.hasUrls():
            return None
        for url in mime.urls():
            path = url.toLocalFile()
            if path and Path(path).suffix.lower() in IMPORT_SUFFIXES:
                return path
        return None

    def dragEnterEvent(self, event) -> None:
        if self._dropped_file(event):
            self.empty_view.set_drop_active(True)
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event) -> None:
        self.empty_view.set_drop_active(False)

    def dropEvent(self, event) -> None:
        self.empty_view.set_drop_active(False)
        path = self._dropped_file(event)
        if path:
            event.acceptProposedAction()
            QTimer.singleShot(0, lambda: self.start_import(path))

    def hideEvent(self, event) -> None:
        # 창을 닫을 때도 "지웠습니다"라고 한 것은 정말 지워 둔다.
        if self.store is not None:
            self.commit_pending_delete()
        super().hideEvent(event)

    def eventFilter(self, obj, event) -> bool:
        if obj is getattr(self, "table", None) and event.type() == QEvent.Resize:
            self._place_blank()
        elif obj is self.search_field and event.type() == QEvent.KeyPress:
            if event.key() == Qt.Key_Escape and self.search_field.text():
                self.set_search("")
                return True
            if event.key() == Qt.Key_Down:
                self.table.setFocus()
                return True
        return super().eventFilter(obj, event)


def _ro(word: str) -> bool:
    """'(으)로' 가운데 '으로'를 쓰는 낱말인지."""
    for ch in reversed(word.strip()):
        if "가" <= ch <= "힣":
            return (ord(ch) - 0xAC00) % 28 not in (0, 8)
        break
    return False
