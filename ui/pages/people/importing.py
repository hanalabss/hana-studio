"""엑셀·CSV 가져오기. 계획(ImportPlan)을 세우는 부분은 화면 없이도 돈다."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QThread, QTimer, Qt, Signal
from PySide6.QtGui import QPainter, QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QPlainTextEdit,
    QScrollArea,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from data.errors import DataError
from data.importers import ImportPreview, ImportResult, import_into, preview_file, suggest_column_map
from data.photos import PhotoLocator, PhotoResolver, set_photo_folder
from data.store import Column, DataStore
from data.values import is_empty
from design.binding import columns_used
from ui.theme import SPACE
from ui.widgets import (
    Banner,
    Card,
    ComboBox,
    Dialog,
    IconButton,
    IconWidget,
    ProgressBar,
    QuietButton,
    SecondaryButton,
    SegmentedControl,
    label,
    theme_color,
)

from .inputs import ElidedLabel, Rethemed
from .preview import CardView, render_card_image
from .text import DELIMITER_NAMES, ENCODING_NAMES, display_value, has_final_consonant, people

FILE_FILTER = "명단 파일 (*.xlsx *.xlsm *.csv *.txt *.tsv);;엑셀 (*.xlsx *.xlsm);;CSV (*.csv *.txt *.tsv)"
IMPORT_SUFFIXES = (".xlsx", ".xlsm", ".csv", ".txt", ".tsv", ".tab", ".xls")
DEFAULT_PHOTO_FIELD = "사진"
_KEY_WORDS = ("사번", "사원번호", "직원번호", "학번", "회원번호", "번호", "id", "no", "코드")
_NAME_WORDS = ("이름", "성명", "name")
PREVIEW_ROWS = 10


class ImportCancelled(Exception):
    pass


# ---------------------------------------------------------------- 카드가 쓰는 칸

@dataclass(frozen=True)
class DesignField:
    name: str
    is_photo: bool = False
    in_design: bool = True


def design_fields(document) -> list[DesignField]:
    """카드에서 사람마다 바뀌는 칸. 놓인 순서대로, 같은 이름은 한 번만."""
    found: dict[str, bool] = {}
    if document is None:
        return []
    for side in document.sides.values():
        for item in side.sorted_items():
            source = getattr(item, "source", None)
            if source is None:
                continue
            for name in _ordered_columns(source):
                found[name] = found.get(name, False) or item.kind == "image"
    used = columns_used(document)
    return [DesignField(name, is_photo) for name, is_photo in found.items() if name in used]


def _ordered_columns(source) -> list[str]:
    parts = getattr(source, "parts", None)
    if parts is not None:
        return [p.column for p in parts if p.column]
    column = getattr(source, "column", "")
    return [column] if column else sorted(source.columns())


def target_fields(document, store: DataStore | None) -> list[DesignField]:
    """맞출 칸: 카드가 쓰는 칸 먼저, 그다음 이미 명단에 있는 항목."""
    fields = design_fields(document)
    names = {f.name for f in fields}
    if store is not None:
        for column in store.columns():
            if column.name not in names:
                fields.append(DesignField(column.name, column.type == "photo", in_design=False))
                names.add(column.name)
            elif column.type == "photo":
                fields = [replace(f, is_photo=True) if f.name == column.name else f for f in fields]
    return fields


def suggest_field_map(fields: list[DesignField], preview: ImportPreview) -> dict[str, str | None]:
    """칸 이름 -> 파일의 열. data.importers.suggest_column_map의 비슷한 이름 찾기를 그대로 쓴다."""
    result: dict[str, str | None] = {f.name: None for f in fields}
    if not fields:
        return result
    scratch = DataStore(":memory:")
    try:
        for f in fields:
            scratch.add_column(f.name)
        for source, target in suggest_column_map(scratch, preview).items():
            if target in result and result[target] is None and scratch.has_column(target):
                if _is_real_match(source, target, preview, result):
                    result[target] = source
    finally:
        scratch.close()
    # '부서'와 '소속부서'처럼 한쪽 이름이 다른 쪽에 들어 있으면, 헷갈릴 열이 없을 때만 맞춘다.
    used = {source for source in result.values() if source}
    for f in fields:
        if result[f.name] is not None:
            continue
        key = _name_key(f.name)
        candidates = [
            c.name for c in preview.columns
            if c.name not in used and len(key) >= 2 and len(_name_key(c.name)) >= 2
            and (key in _name_key(c.name) or _name_key(c.name) in key)
        ]
        if len(candidates) == 1:
            result[f.name] = candidates[0]
            used.add(candidates[0])
    return result


def _name_key(name: str) -> str:
    return re.sub(r"[\s_\-./()\[\]:·]+", "", name).lower()


def _is_real_match(source: str, target: str, preview: ImportPreview, result: dict) -> bool:
    return source in {c.name for c in preview.columns} and source not in result.values()


def suggest_photo_source(preview: ImportPreview, folder: str | os.PathLike | None,
                         preferred: str | None = None) -> str | None:
    """사진 파일 이름이 들어 있을 만한 열. 폴더가 있으면 실제로 가장 많이 맞는 열을 고른다."""
    names = [c.name for c in preview.columns]
    if not names:
        return None
    if folder:
        locator = PhotoLocator(folder)
        sample = preview.rows[:300]
        best, best_hits = None, 0
        for index, name in enumerate(names):
            hits = sum(1 for row in sample if locator.find(row[index]) is not None)
            if hits > best_hits or (hits == best_hits and hits and name == preferred):
                best, best_hits = name, hits
        if best is not None:
            return best
    if preferred in names:
        return preferred
    for column in preview.columns:
        if column.type == "photo":
            return column.name
    for words in (_KEY_WORDS, _NAME_WORDS):
        for name in names:
            key = name.replace(" ", "").lower()
            if any(word in key for word in words):
                return name
    return None


def suggest_key_field(field_names: list[str]) -> str | None:
    for words in (_KEY_WORDS, _NAME_WORDS):
        for name in field_names:
            key = name.replace(" ", "").lower()
            if any(word in key for word in words):
                return name
    return field_names[0] if field_names else None


def count_photos(preview: ImportPreview, source: str | None, folder: str | os.PathLike | None) -> tuple[int, list[int]]:
    """(찾은 사람 수, 못 찾은 줄 번호들)."""
    names = [c.name for c in preview.columns]
    if not source or source not in names or not folder:
        return 0, list(range(len(preview.rows)))
    index = names.index(source)
    locator = PhotoLocator(folder)
    missing = [i for i, row in enumerate(preview.rows) if locator.find(row[index]) is None]
    return len(preview.rows) - len(missing), missing


def header_is_uncertain(preview: ImportPreview) -> bool:
    if not preview.has_header:
        return True
    names = [c.name for c in preview.columns]
    return any(name.replace(".", "").replace("-", "").isdigit() for name in names)


def text_is_uncertain(preview: ImportPreview) -> bool:
    if preview.kind != "csv":
        return False
    if preview.warnings:
        return True
    if len(preview.columns) == 1 and preview.total_rows > 0:
        return True
    return any("�" in cell for row in preview.sample_rows for cell in row)


# ---------------------------------------------------------------- 계획과 실행

@dataclass
class ImportPlan:
    preview: ImportPreview
    field_map: dict[str, str | None] = field(default_factory=dict)   # 칸 -> 파일의 열
    include_rest: bool = True
    mode: str = "append"
    skip_existing: bool = False
    key_field: str | None = None
    photo_field: str | None = None
    photo_source: str | None = None
    photo_folder: str | None = None


@dataclass
class ImportOutcome:
    result: ImportResult
    skipped_existing: int = 0
    photo_field: str | None = None
    photo_found: int = 0
    photo_missing: int = 0
    total: int = 0
    file_name: str = ""


def _unique_name(name: str, taken: set[str]) -> str:
    candidate, number = name, 1
    while candidate in taken:
        number += 1
        candidate = f"{name} ({number})"
    return candidate


def resolve_plan(plan: ImportPlan) -> tuple[ImportPreview, dict[str, str | None]]:
    """import_into에 넘길 (미리보기, {파일의 열: 명단 항목})."""
    preview = plan.preview
    names = [c.name for c in preview.columns]
    column_map: dict[str, str | None] = {}
    for target, source in plan.field_map.items():
        if not source or target == plan.photo_field:
            continue
        if source not in names:
            raise DataError(f"파일에 '{source}' 열이 없습니다. 다시 골라 주세요.")
        if source in column_map:
            raise DataError(
                f"'{source}' 열을 '{column_map[source]}'{'과' if _final(column_map[source]) else '와'} '{target}' 두 칸에 "
                "골랐습니다. 한 열은 한 칸에만 쓸 수 있습니다.")
        column_map[source] = target

    photo_direct = False
    if plan.photo_field and plan.photo_source in names and plan.photo_source not in column_map:
        source_column = preview.columns[names.index(plan.photo_source)]
        if source_column.type == "photo":
            column_map[plan.photo_source] = plan.photo_field
            photo_direct = True

    taken = {t for t in column_map.values() if t} | {t for t in plan.field_map if t}
    if plan.photo_field:
        taken.add(plan.photo_field)
    for name in names:
        if name in column_map:
            continue
        if plan.include_rest:
            target = name if name not in taken else _unique_name(name, taken)
            taken.add(target)
            column_map[name] = target
        else:
            column_map[name] = None

    if plan.photo_field and plan.photo_source in names and not photo_direct:
        index = names.index(plan.photo_source)
        extra = _unique_name(f"{plan.photo_field} 파일 이름", set(names))
        preview = replace(
            preview,
            columns=list(preview.columns) + [Column(extra, "photo")],
            rows=[list(row) + [row[index]] for row in preview.rows],
            sample_rows=[list(row) + [row[index]] for row in preview.sample_rows],
        )
        column_map[extra] = plan.photo_field
    if not any(column_map.values()):
        raise DataError("가져올 열이 없습니다. 칸을 하나 이상 맞추거나 '나머지 열도 함께 가져오기'를 켜세요.")
    return preview, column_map


def _final(word: str) -> bool:
    return has_final_consonant(word)


class _ReportingRows(list):
    """import_into가 줄을 읽어 가는 만큼 진행을 알리고, 그만두라고 하면 멈춘다(전부 되돌려진다)."""

    def __init__(self, rows, report, cancelled):
        super().__init__(rows)
        self._report = report
        self._cancelled = cancelled

    def __iter__(self):
        total = len(self)
        for index, row in enumerate(list.__iter__(self)):
            if index % 100 == 0:
                if self._cancelled():
                    raise ImportCancelled()
                self._report(index, total)
            yield row


def _existing_keys(store: DataStore, column: str) -> set[str]:
    keys: set[str] = set()
    offset, chunk = 0, 2000
    while True:
        rows = store.rows(None, None, offset, chunk)
        keys.update(display_value(r.get(column)).strip() for r in rows if not is_empty(r.get(column)))
        if len(rows) < chunk:
            return keys
        offset += chunk


def run_import(store: DataStore, plan: ImportPlan,
               report: Callable[[int, int], None] = lambda done, total: None,
               cancelled: Callable[[], bool] = lambda: False) -> ImportOutcome:
    preview, column_map = resolve_plan(plan)
    outcome = ImportOutcome(ImportResult(), photo_field=plan.photo_field, file_name=Path(preview.path).name)
    names = [c.name for c in preview.columns]

    key_source = plan.field_map.get(plan.key_field) if plan.key_field else None
    if (plan.skip_existing and plan.mode == "append" and key_source in names
            and plan.key_field and store.has_column(plan.key_field)):
        existing = _existing_keys(store, plan.key_field)
        index = names.index(key_source)
        kept = [(n, row) for n, row in zip(preview.row_numbers, preview.rows)
                if not (row[index].strip() and row[index].strip() in existing)]
        outcome.skipped_existing = len(preview.rows) - len(kept)
        preview = replace(preview, rows=[row for _, row in kept], row_numbers=[n for n, _ in kept])

    if plan.photo_field and plan.photo_folder:
        source = next((s for s, t in column_map.items() if t == plan.photo_field), None)
        found, missing = count_photos(preview, source, plan.photo_folder)
        outcome.photo_found, outcome.photo_missing = found, len(missing)

    if not preview.rows:
        outcome.total = store.count()
        return outcome

    total = len(preview.rows)
    preview = replace(preview, rows=_ReportingRows(preview.rows, report, cancelled))
    with store.transaction():
        outcome.result = import_into(store, preview, column_map, plan.mode)
        if plan.photo_field and store.has_column(plan.photo_field):
            if store.column(plan.photo_field).type != "photo":
                store.set_column_type(plan.photo_field, "photo")
            if plan.photo_folder:
                set_photo_folder(store, plan.photo_field, plan.photo_folder)
    report(total, total)
    outcome.total = store.count()
    return outcome


class Worker(QThread):
    progress = Signal(int, int)
    succeeded = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, job: Callable[[Callable[[int, int], None], Callable[[], bool]], Any], parent=None):
        super().__init__(parent)
        self._job = job
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        try:
            result = self._job(self.progress.emit, lambda: self._cancel)
        except ImportCancelled:
            self.cancelled.emit()
        except DataError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # 사용자에게 이유를 보여 주고 창은 살려 둔다.
            self.failed.emit(f"문제가 생겨 멈췄습니다. 명단은 바뀌지 않았습니다. ({exc})")
        else:
            if self._cancel:
                self.cancelled.emit()
            else:
                self.succeeded.emit(result)


# ---------------------------------------------------------------- 화면 조각

class _FieldTag(Rethemed, QFrame):
    """카드 칸 이름표. 카드가 쓰는 칸은 노란색, 명단에만 있는 항목은 회색."""

    def __init__(self, text: str, in_design: bool = True, parent: QWidget | None = None):
        super().__init__(parent)
        self._text = text
        self._in_design = in_design
        self.setFixedHeight(28)
        self.setSizePolicy(self.sizePolicy().horizontalPolicy(), self.sizePolicy().verticalPolicy())
        self.follow_theme()

    def sizeHint(self):
        from PySide6.QtCore import QSize

        return QSize(self.fontMetrics().horizontalAdvance(self._text) + 26, 28)

    def minimumSizeHint(self):
        return self.sizeHint()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(Qt.NoPen)
        painter.setBrush(theme_color("field_bg" if self._in_design else "idle_bg"))
        width = min(self.width(), self.sizeHint().width())
        painter.drawRoundedRect(0, 0, width, 28, 14, 14)
        painter.setPen(theme_color("field_text" if self._in_design else "idle_text"))
        font = painter.font()
        font.setWeight(font.Weight.Medium)
        painter.setFont(font)
        painter.drawText(0, 0, width, 28, Qt.AlignCenter, self._text)
        painter.end()


class _PhotoCard(Rethemed, QFrame):
    """사진 칸 카드. 못 찾은 사진이 있으면 노란 테두리로 알린다."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._warn = False
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(14, 14, 14, 14)
        self.body.setSpacing(SPACE["sm"])
        self.follow_theme()

    def set_warn(self, warn: bool) -> None:
        if warn != self._warn:
            self._warn = warn
            self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setBrush(theme_color("warn_bg" if self._warn else "surface"))
        painter.setPen(QPen(theme_color("warn_mark" if self._warn else "line"), 1.5 if self._warn else 1))
        painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 12, 12)
        painter.end()


def _table_item(text: str) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    item.setFlags(Qt.ItemIsEnabled)
    return item


# ---------------------------------------------------------------- 가져오기 창

PAGE_FILE, PAGE_MATCH, PAGE_WORK, PAGE_DONE = range(4)


class ImportWizard(Dialog):
    """파일 확인 -> 칸 맞추기(사진 포함) -> 결과. 끝나면 wizard.outcome과 wizard.store를 본다."""

    busyChanged = Signal(bool)

    def __init__(self, workspace, store_factory: Callable[[str], DataStore] | None = None,
                 parent: QWidget | None = None, threaded: bool = True):
        super().__init__("명단 가져오기", None, parent, width=980)
        self.setModal(True)
        self.resize(980, 700)
        self._workspace = workspace
        self._store_factory = store_factory
        self._threaded = threaded
        self.store: DataStore | None = workspace.store
        self.created_store = False
        self.preview: ImportPreview | None = None
        self.outcome: ImportOutcome | None = None
        self.error: str | None = None
        self._path: str | None = None
        self._worker: Worker | None = None
        self._busy = False
        self._fields: list[DesignField] = []
        self._photo_field: str | None = None
        self._photo_folder: str | None = None
        self._photo_source_touched = False
        self._sample_index = 0
        self._loading_options: dict[str, Any] = {}
        self._heading_description = ""
        self.field_combos: dict[str, ComboBox] = {}

        self.stack = QStackedWidget(self)
        self.body.addWidget(self.stack, 1)
        self.stack.addWidget(self._build_file_page())
        self.stack.addWidget(self._build_match_page())
        self.stack.addWidget(self._build_work_page())
        self.stack.addWidget(self._build_done_page())

        self.cancel_button = self.add_button("취소", "quiet", None, left=True)
        self.cancel_button.clicked.connect(self._on_cancel)
        self.back_button = self.add_button("이전", "secondary", None)
        self.back_button.clicked.connect(self.go_back)
        self.next_button = self.add_button("다음", "primary", None, default=True)
        self.next_button.clicked.connect(self.go_next)
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(120)
        self._preview_timer.timeout.connect(self._render_sample)
        self._show_page(PAGE_FILE)

    # ================================================================ 1. 파일
    def _build_file_page(self) -> QWidget:
        page = QWidget()
        box = QVBoxLayout(page)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(SPACE["md"])

        self.file_banner = Banner("warn", "")
        self.file_banner.setVisible(False)
        box.addWidget(self.file_banner)

        self.sheet_row = QWidget()
        sheet = QHBoxLayout(self.sheet_row)
        sheet.setContentsMargins(0, 0, 0, 0)
        sheet.setSpacing(SPACE["md"])
        sheet.addWidget(label("어느 시트를 가져올까요?", "strong"))
        self.sheet_combo = ComboBox()
        self.sheet_combo.setMinimumWidth(240)
        self.sheet_combo.setAccessibleName("가져올 시트")
        self.sheet_combo.activated.connect(lambda *_: self._reload_preview())
        sheet.addWidget(self.sheet_combo)
        sheet.addStretch(1)
        box.addWidget(self.sheet_row)

        self.preview_table = QTableWidget(0, 0)
        self.preview_table.verticalHeader().setVisible(False)
        self.preview_table.verticalHeader().setDefaultSectionSize(36)
        self.preview_table.setShowGrid(False)
        self.preview_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.preview_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.preview_table.setFocusPolicy(Qt.NoFocus)
        self.preview_table.horizontalHeader().setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.preview_table.horizontalHeader().setHighlightSections(False)
        self.preview_table.horizontalHeader().setSectionsClickable(False)
        self.preview_table.setAccessibleName("파일 미리 보기")
        box.addWidget(self.preview_table, 1)

        self.loading_box = QWidget()
        loading = QVBoxLayout(self.loading_box)
        loading.setContentsMargins(0, SPACE["xxl"], 0, SPACE["xxl"])
        loading.setSpacing(SPACE["md"])
        loading.addStretch(1)
        self.loading_label = label("파일을 읽는 중입니다", "title")
        self.loading_label.setAlignment(Qt.AlignHCenter)
        loading.addWidget(self.loading_label)
        self.loading_bar = ProgressBar()
        self.loading_bar.setRange(0, 0)
        self.loading_bar.setFixedWidth(320)
        loading.addWidget(self.loading_bar, 0, Qt.AlignHCenter)
        loading.addStretch(1)
        self.loading_box.setVisible(False)
        box.addWidget(self.loading_box, 1)

        foot = QHBoxLayout()
        foot.setSpacing(SPACE["sm"])
        self.preview_caption = label("", "hint")
        foot.addWidget(self.preview_caption)
        foot.addStretch(1)
        self.detail_toggle = QuietButton("자세히", "sm", icon="chevron-down")
        self.detail_toggle.setCheckable(True)
        self.detail_toggle.toggled.connect(self._toggle_details)
        foot.addWidget(self.detail_toggle)
        self.other_file_button = SecondaryButton("다른 파일로 바꾸기", "sm")
        self.other_file_button.clicked.connect(self.choose_file)
        foot.addWidget(self.other_file_button)
        box.addLayout(foot)

        self.detail_box = Card(padding=SPACE["md"], sunken=True, spacing=SPACE["sm"])
        self.header_check = QCheckBox("첫 줄이 제목입니다")
        self.header_check.setToolTip("첫 줄에 '이름', '부서' 같은 제목이 적혀 있으면 켜 두세요.")
        self.header_check.clicked.connect(lambda *_: self._reload_preview())
        self.detail_box.body.addWidget(self.header_check)
        self.text_row = QWidget()
        text_row = QHBoxLayout(self.text_row)
        text_row.setContentsMargins(0, 0, 0, 0)
        text_row.setSpacing(SPACE["md"])
        text_row.addWidget(label("글자가 깨져 보이면", "label"))
        self.encoding_combo = ComboBox()
        for value, name in ENCODING_NAMES:
            self.encoding_combo.addItem(name, value)
        self.encoding_combo.setAccessibleName("글자 읽는 방식")
        self.encoding_combo.setMinimumWidth(260)
        self.encoding_combo.activated.connect(lambda *_: self._reload_preview())
        text_row.addWidget(self.encoding_combo)
        text_row.addSpacing(SPACE["md"])
        text_row.addWidget(label("칸이 안 나뉘어 보이면", "label"))
        self.delimiter_combo = ComboBox()
        for value, name in DELIMITER_NAMES:
            self.delimiter_combo.addItem(name, value)
        self.delimiter_combo.setAccessibleName("칸을 나누는 글자")
        self.delimiter_combo.setMinimumWidth(170)
        self.delimiter_combo.activated.connect(lambda *_: self._reload_preview())
        text_row.addWidget(self.delimiter_combo)
        text_row.addStretch(1)
        self.detail_box.body.addWidget(self.text_row)
        self.detail_box.setVisible(False)
        box.addWidget(self.detail_box)
        return page

    def _toggle_details(self, on: bool) -> None:
        self.detail_box.setVisible(on)
        self.detail_toggle.set_icon_name("chevron-up" if on else "chevron-down")

    def choose_file(self) -> bool:
        start = str(Path(self._path).parent) if self._path else ""
        path, _ = QFileDialog.getOpenFileName(self, "명단 파일 고르기", start, FILE_FILTER)
        if not path:
            return False
        self.load_file(path)
        return True

    def load_file(self, path: str | os.PathLike, **options: Any) -> None:
        """파일을 읽어 첫 화면을 채운다. options: sheet, has_header, encoding, delimiter."""
        self._path = str(path)
        self._photo_source_touched = False
        self._start_preview(options, first=True)

    def _reload_preview(self) -> None:
        if not self._path or self.preview is None:
            return
        options: dict[str, Any] = {"has_header": self.header_check.isChecked()}
        if self.preview.kind == "excel":
            if self.sheet_combo.currentText():
                options["sheet"] = self.sheet_combo.currentText()
        else:
            if self.encoding_combo.currentData():
                options["encoding"] = self.encoding_combo.currentData()
            if self.delimiter_combo.currentData():
                options["delimiter"] = self.delimiter_combo.currentData()
        self._start_preview(options, first=False)

    def _start_preview(self, options: dict[str, Any], first: bool) -> None:
        path = self._path
        self._loading_options = dict(options)
        self._loading_first = first
        self.error = None
        self._show_page(PAGE_FILE)
        self.loading_label.setText(f"{Path(path).name} 파일을 읽는 중입니다")
        self._run(lambda report, cancelled: preview_file(path, **options),
                  self._preview_ready, self._preview_failed, loading=True)

    def _preview_failed(self, message: str) -> None:
        self.error = message
        if self._loading_first:
            self.preview = None
            self.preview_table.setRowCount(0)
            self.preview_table.setColumnCount(0)
            self.sheet_row.setVisible(False)
            self.preview_caption.setText("")
            self._set_heading("이 파일은 가져올 수 없습니다", None)
        self.file_banner.set_kind("error")
        self.file_banner.set_text(message)
        self.file_banner.setVisible(True)
        self._sync_buttons()

    def _preview_ready(self, preview: ImportPreview) -> None:
        self.preview = preview
        name = Path(preview.path).name
        self._set_heading(f"{name}에서 {people(preview.total_rows)}을 찾았습니다",
                          "아래처럼 읽었습니다. 맞으면 다음으로 넘어가세요. 원래 파일은 고치지 않습니다.")
        is_excel = preview.kind == "excel"
        self.sheet_row.setVisible(is_excel and len(preview.sheets) > 1)
        if is_excel:
            self.sheet_combo.blockSignals(True)
            self.sheet_combo.clear()
            self.sheet_combo.addItems(preview.sheets)
            self.sheet_combo.setCurrentText(preview.sheet or "")
            self.sheet_combo.blockSignals(False)
        self.text_row.setVisible(not is_excel)
        self.header_check.setChecked(preview.has_header)
        if "encoding" not in self._loading_options:
            self.encoding_combo.setCurrentIndex(0)
        if "delimiter" not in self._loading_options:
            self.delimiter_combo.setCurrentIndex(0)

        table = self.preview_table
        rows = preview.sample_rows[:PREVIEW_ROWS]
        table.clear()
        table.setColumnCount(len(preview.columns))
        table.setRowCount(len(rows))
        table.setHorizontalHeaderLabels([c.name for c in preview.columns])
        for r, row in enumerate(rows):
            for c, value in enumerate(row):
                table.setItem(r, c, _table_item(value))
        header = table.horizontalHeader()
        if len(preview.columns) <= 7:
            header.setSectionResizeMode(QHeaderView.Stretch)
        else:
            header.setSectionResizeMode(QHeaderView.Interactive)
            header.setDefaultSectionSize(140)
        if preview.total_rows > len(rows):
            self.preview_caption.setText(f"처음 {len(rows)}줄만 보여 줍니다. 모두 {people(preview.total_rows)}입니다.")
        else:
            self.preview_caption.setText(f"모두 {people(preview.total_rows)}입니다.")

        problems = []
        if header_is_uncertain(preview):
            problems.append("첫 줄이 제목인지 확실하지 않습니다. 아래 '첫 줄이 제목입니다'가 맞는지 봐 주세요.")
        if text_is_uncertain(preview):
            problems.append("글자가 깨졌거나 칸이 제대로 나뉘지 않았을 수 있습니다. 아래에서 바꿔 볼 수 있습니다.")
        problems += [w for w in preview.warnings if w not in problems][:1] if not problems else []
        if preview.total_rows == 0:
            problems = ["제목 줄만 있고 사람이 한 명도 없습니다. 다른 파일이나 다른 시트를 골라 주세요."]
        self.file_banner.set_kind("warn")
        self.file_banner.set_text(" ".join(problems))
        self.file_banner.setVisible(bool(problems))
        if self._loading_first:
            self.detail_toggle.setChecked(bool(problems) and preview.total_rows > 0)
        self._prepare_match()
        self._sync_buttons()

    # ================================================================ 2. 칸 맞추기
    def _build_match_page(self) -> QWidget:
        page = QWidget()
        outer = QHBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(SPACE["xl"])

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        inner = QWidget()
        inner.setObjectName("ScrollBody")
        scroll.setWidget(inner)
        self.match_box = QVBoxLayout(inner)
        self.match_box.setContentsMargins(0, 0, SPACE["lg"], 0)
        self.match_box.setSpacing(SPACE["md"])

        self.match_banner = Banner("error", "")
        self.match_banner.setVisible(False)
        self.match_box.addWidget(self.match_banner)

        # 이미 명단이 있을 때
        self.mode_card = Card(padding=14, spacing=SPACE["sm"])
        self.mode_card.body.addWidget(label("지금 있는 명단은 어떻게 할까요?", "strong"))
        self.mode_control = SegmentedControl([("append", "뒤에 추가"), ("replace", "전부 바꾸기")], current="append")
        self.mode_control.currentChanged.connect(lambda *_: self._sync_match_summary())
        self.mode_card.body.addWidget(self.mode_control)
        self.mode_hint = label("", "secondary", wrap=True)
        self.mode_card.body.addWidget(self.mode_hint)
        skip_row = QHBoxLayout()
        skip_row.setSpacing(SPACE["sm"])
        self.skip_check = QCheckBox("이미 있는 사람은 건너뛰기")
        self.skip_check.toggled.connect(lambda *_: self._sync_match_summary())
        skip_row.addWidget(self.skip_check)
        self.key_combo = ComboBox()
        self.key_combo.setMinimumWidth(150)
        self.key_combo.setAccessibleName("같은 사람인지 볼 항목")
        self.key_combo.activated.connect(lambda *_: self._sync_match_summary())
        skip_row.addWidget(self.key_combo)
        self.key_suffix = label("같으면 같은 사람으로 봅니다", "muted")
        skip_row.addWidget(self.key_suffix)
        skip_row.addStretch(1)
        self.mode_card.body.addLayout(skip_row)
        self.match_box.addWidget(self.mode_card)

        self.field_grid_host = QWidget()
        self.field_grid = QGridLayout(self.field_grid_host)
        self.field_grid.setContentsMargins(0, 0, 0, 0)
        self.field_grid.setSpacing(SPACE["md"])
        self.match_box.addWidget(self.field_grid_host)

        # 사진
        self.photo_card = _PhotoCard()
        head = QHBoxLayout()
        head.setSpacing(SPACE["sm"])
        self.photo_tag_holder = QHBoxLayout()
        head.addLayout(self.photo_tag_holder)
        head.addWidget(label("사진은 어디에 있나요?", "strong"))
        head.addStretch(1)
        self.photo_card.body.addLayout(head)
        folder_row = QHBoxLayout()
        folder_row.setSpacing(SPACE["md"])
        self.folder_button = SecondaryButton("사진 폴더 고르기", icon="folder")
        self.folder_button.clicked.connect(self.choose_photo_folder)
        folder_row.addWidget(self.folder_button)
        self.folder_label = ElidedLabel("아직 고르지 않았습니다. 사진 없이 가져와도 됩니다.", "muted")
        folder_row.addWidget(self.folder_label, 1)
        self.photo_card.body.addLayout(folder_row)
        source_row = QHBoxLayout()
        source_row.setSpacing(SPACE["md"])
        source_row.addWidget(label("사진 파일 이름과 같은 열", "label"))
        self.photo_source_combo = ComboBox()
        self.photo_source_combo.setMinimumWidth(180)
        self.photo_source_combo.setAccessibleName("사진 파일 이름과 같은 열")
        self.photo_source_combo.activated.connect(self._on_photo_source_picked)
        source_row.addWidget(self.photo_source_combo)
        source_row.addStretch(1)
        self.photo_card.body.addLayout(source_row)
        self.photo_status = label("", "secondary", wrap=True)
        self.photo_card.body.addWidget(self.photo_status)
        self.match_box.addWidget(self.photo_card)
        self.photo_link = QuietButton("사진 파일도 함께 쓰기", "sm", icon="image", accent=True)
        self.photo_link.clicked.connect(lambda: self._set_photo_visible(True))
        link_row = QHBoxLayout()
        link_row.addWidget(self.photo_link)
        link_row.addStretch(1)
        self.match_box.addLayout(link_row)

        self.rest_check = QCheckBox("나머지 열도 함께 가져오기")
        self.rest_check.setChecked(True)
        self.rest_check.toggled.connect(lambda *_: self._sync_match_summary())
        self.match_box.addWidget(self.rest_check)
        self.rest_hint = label("", "hint", wrap=True)
        self.rest_hint.setContentsMargins(28, 0, 0, 0)
        self.match_box.addWidget(self.rest_hint)

        self.match_box.addStretch(1)
        outer.addWidget(scroll, 1)

        # 오른쪽 미리 보기
        side = QVBoxLayout()
        side.setSpacing(SPACE["md"])
        side.addWidget(label("이렇게 나옵니다", "title"))
        self.sample_card = CardView()
        self.sample_card.setFixedWidth(320)
        side.addWidget(self.sample_card)
        nav = QHBoxLayout()
        nav.setSpacing(SPACE["md"])
        nav.addStretch(1)
        self.sample_prev = IconButton("chevron-left", "이전 사람", variant="secondary")
        self.sample_next = IconButton("chevron-right", "다음 사람", variant="secondary")
        self.sample_prev.clicked.connect(lambda: self._step_sample(-1))
        self.sample_next.clicked.connect(lambda: self._step_sample(1))
        self.sample_position = label("", "secondary")
        self.sample_position.setAlignment(Qt.AlignCenter)
        self.sample_position.setMinimumWidth(80)
        nav.addWidget(self.sample_prev)
        nav.addWidget(self.sample_position)
        nav.addWidget(self.sample_next)
        nav.addStretch(1)
        side.addLayout(nav)
        self.sample_hint = label("", "hint", wrap=True)
        self.sample_hint.setFixedWidth(320)
        side.addWidget(self.sample_hint)
        side.addStretch(1)
        outer.addLayout(side)
        return page

    def _prepare_match(self) -> None:
        preview = self.preview
        if preview is None:
            return
        document = self._workspace.editor.document
        self._fields = target_fields(document, self.store)
        suggestion = suggest_field_map(self._fields, preview)
        photo_fields = [f for f in self._fields if f.is_photo]
        self._photo_field = photo_fields[0].name if photo_fields else None
        file_has_photo = any(c.type == "photo" for c in preview.columns)
        if self._photo_field is None and file_has_photo:
            self._photo_field = DEFAULT_PHOTO_FIELD

        while self.field_grid.count():
            child = self.field_grid.takeAt(0)
            if child.widget() is not None:
                child.widget().hide()
                child.widget().deleteLater()
        self.field_combos = {}
        normal = [f for f in self._fields if f.name != self._photo_field]
        columns = 2 if len(normal) <= 4 else 3
        names = [c.name for c in preview.columns]
        samples = {name: next((row[i] for row in preview.sample_rows if row[i]), "") for i, name in enumerate(names)}
        self._samples = samples
        for position, f in enumerate(normal):
            card = Card(padding=12, spacing=SPACE["sm"])
            card.body.addWidget(_FieldTag(f.name, f.in_design))
            combo = ComboBox()
            combo.addItem("가져오지 않음", None)
            for name in names:
                combo.addItem(name, name)
            combo.setAccessibleName(f"카드의 {f.name} 칸에 넣을 파일의 열")
            combo.setToolTip(f"파일의 어느 열을 '{f.name}' 칸에 넣을지 고릅니다.")
            chosen = suggestion.get(f.name)
            combo.setCurrentIndex(names.index(chosen) + 1 if chosen in names else 0)
            combo.activated.connect(lambda *_: self._on_mapping_changed())
            card.body.addWidget(combo)
            hint = label("", "hint", wrap=True)
            hint.setMinimumHeight(18)
            card.body.addWidget(hint)
            combo.hint_label = hint
            combo.in_design = f.in_design
            self.field_combos[f.name] = combo
            self.field_grid.addWidget(card, position // columns, position % columns)
        for column in range(columns):
            self.field_grid.setColumnStretch(column, 1)
        self.field_grid_host.setVisible(bool(normal))

        while self.photo_tag_holder.count():
            child = self.photo_tag_holder.takeAt(0)
            if child.widget() is not None:
                child.widget().deleteLater()
        self.photo_tag_holder.addWidget(_FieldTag(self._photo_field or DEFAULT_PHOTO_FIELD))
        self.photo_source_combo.blockSignals(True)
        self.photo_source_combo.clear()
        for name in names:
            self.photo_source_combo.addItem(name, name)
        self.photo_source_combo.blockSignals(False)
        if self._photo_folder is None and self.store is not None and self._photo_field \
                and self.store.has_column(self._photo_field):
            folder = self.store.column(self._photo_field).options.get("folder")
            self._photo_folder = str(folder) if folder else None
        preferred = suggestion.get(self._photo_field) if self._photo_field else None
        self._preferred_photo_source = preferred
        self._pick_photo_source(suggest_photo_source(preview, self._photo_folder, preferred))
        self._set_photo_visible(self._photo_field is not None)

        has_people = self.store is not None and self.store.count() > 0
        self.mode_card.setVisible(has_people)
        self.key_combo.blockSignals(True)
        self.key_combo.clear()
        key_names = [f.name for f in self._fields if not f.is_photo]
        for name in key_names:
            self.key_combo.addItem(name, name)
        key = suggest_key_field(key_names)
        if key:
            self.key_combo.setCurrentIndex(key_names.index(key))
        self.key_combo.blockSignals(False)
        self._sample_index = 0
        self._on_mapping_changed()

    def _set_photo_visible(self, visible: bool) -> None:
        if visible and self._photo_field is None:
            self._photo_field = DEFAULT_PHOTO_FIELD
        self.photo_card.setVisible(visible)
        self.photo_link.setVisible(not visible)
        self._sync_photo()

    def _pick_photo_source(self, name: str | None) -> None:
        index = self.photo_source_combo.findData(name) if name else -1
        self.photo_source_combo.setCurrentIndex(max(0, index))

    def _on_photo_source_picked(self, *_args) -> None:
        self._photo_source_touched = True
        self._sync_photo()
        self._schedule_sample()

    def choose_photo_folder(self) -> None:
        start = self._photo_folder or (str(Path(self._path).parent) if self._path else "")
        folder = QFileDialog.getExistingDirectory(self, "사진이 들어 있는 폴더 고르기", start)
        if folder:
            self.set_photo_folder(folder)

    def set_photo_folder(self, folder: str | os.PathLike | None) -> None:
        self._photo_folder = str(folder) if folder else None
        if self._photo_field is None:
            self._photo_field = DEFAULT_PHOTO_FIELD
        self.photo_card.setVisible(True)
        self.photo_link.setVisible(False)
        if self.preview is not None and not self._photo_source_touched:
            self._pick_photo_source(suggest_photo_source(
                self.preview, self._photo_folder, getattr(self, "_preferred_photo_source", None)))
        self._sync_photo()
        self._schedule_sample()

    def photo_source(self) -> str | None:
        return self.photo_source_combo.currentData() if self._photo_field else None

    def photo_counts(self) -> tuple[int, int]:
        """(사진을 찾은 사람 수, 못 찾은 사람 수)."""
        if self.preview is None:
            return 0, 0
        found, missing = count_photos(self.preview, self.photo_source(), self._photo_folder)
        return found, len(missing)

    def _sync_photo(self) -> None:
        preview = self.preview
        if preview is None:
            return
        if not self._photo_folder:
            self.folder_label.set_full_text("아직 고르지 않았습니다. 사진 없이 가져와도 됩니다.")
            self.photo_status.setText(
                "사진 파일 이름이 사번이나 이름과 같으면, 폴더만 고르면 알아서 짝을 맞춥니다.")
            self.photo_card.set_warn(False)
            return
        self.folder_label.set_full_text(self._photo_folder)
        found, missing_rows = count_photos(preview, self.photo_source(), self._photo_folder)
        total = len(preview.rows)
        source = self.photo_source() or ""
        if not missing_rows:
            self.photo_status.setText(f"'{source}' 열과 같은 이름의 파일을 찾았습니다. {people(total)} 모두 사진이 있습니다.")
            self.photo_card.set_warn(False)
            return
        who = self._who(missing_rows)
        if found == 0:
            text = (f"이 폴더에는 '{source}' 열과 같은 이름의 사진이 하나도 없습니다. "
                    "폴더나 열을 다시 골라 주세요.")
        else:
            text = (f"'{source}' 열과 같은 이름의 파일을 찾습니다. {people(total)} 중 {people(found)}은 찾았고, "
                    f"{people(len(missing_rows))}은 못 찾았습니다{who}. 가져온 뒤 명단에서 한 명씩 넣을 수 있습니다.")
        self.photo_status.setText(text)
        self.photo_card.set_warn(True)

    def _who(self, rows: list[int]) -> str:
        preview = self.preview
        names = [c.name for c in preview.columns]
        source = None
        for field_name, combo in self.field_combos.items():
            if any(word in field_name.lower() for word in _NAME_WORDS) and combo.currentData():
                source = combo.currentData()
                break
        if source not in names:
            return ""
        index = names.index(source)
        shown = [preview.rows[r][index] for r in rows[:3] if preview.rows[r][index]]
        if not shown:
            return ""
        rest = len(rows) - len(shown)
        return f" ({', '.join(shown)}{f' 외 {rest}명' if rest > 0 else ''})"

    def mapping(self) -> dict[str, str | None]:
        return {name: combo.currentData() for name, combo in self.field_combos.items()}

    def set_mapping(self, field_name: str, column: str | None) -> None:
        combo = self.field_combos[field_name]
        combo.setCurrentIndex(max(0, combo.findData(column)) if column else 0)
        self._on_mapping_changed()

    def _on_mapping_changed(self) -> None:
        for name, combo in self.field_combos.items():
            source = combo.currentData()
            hint = combo.hint_label
            if source:
                sample = self._samples.get(source, "")
                hint.setText(f"예: {sample}" if sample else "")
                hint.setProperty("tone", "")
            elif combo.in_design:
                hint.setText("맞는 열을 직접 골라 주세요.")
                hint.setProperty("tone", "warn")
            else:
                hint.setText("")
                hint.setProperty("tone", "")
            hint.style().unpolish(hint)
            hint.style().polish(hint)
        self.match_banner.setVisible(False)
        self._sync_match_summary()
        self._schedule_sample()

    def plan(self) -> ImportPlan:
        use_photo = self._photo_field is not None and (
            self._photo_folder is not None
            or (self.preview is not None and any(
                c.type == "photo" and c.name == self.photo_source_combo.currentData() for c in self.preview.columns))
            or any(f.is_photo and f.name == self._photo_field for f in self._fields))
        return ImportPlan(
            preview=self.preview,
            field_map=self.mapping(),
            include_rest=self.rest_check.isChecked(),
            mode=self.mode_control.current() if self._has_people() else "append",
            skip_existing=self.skip_check.isChecked() and self._has_people(),
            key_field=self.key_combo.currentData(),
            photo_field=self._photo_field if use_photo else None,
            photo_source=self.photo_source_combo.currentData() if use_photo else None,
            photo_folder=self._photo_folder if use_photo else None,
        )

    def _has_people(self) -> bool:
        return self.store is not None and self.store.count() > 0

    def _sync_match_summary(self) -> None:
        preview = self.preview
        if preview is None:
            return
        name = Path(preview.path).name
        matched = sum(1 for combo in self.field_combos.values() if combo.currentData() and combo.in_design)
        design_count = sum(1 for combo in self.field_combos.values() if combo.in_design)
        if design_count:
            if matched == design_count:
                description = f"카드의 칸 {design_count}개를 파일의 열과 맞췄습니다. 틀린 곳이 있으면 눌러서 고치세요."
            else:
                description = (f"카드의 칸 {design_count}개 중 {matched}개를 맞췄습니다. "
                               "노란 글씨가 있는 칸은 직접 골라 주세요.")
        elif self.field_combos:
            description = "명단의 항목을 파일의 열과 맞췄습니다. 틀린 곳이 있으면 눌러서 고치세요."
        else:
            description = "카드에 아직 사람마다 바뀌는 칸이 없습니다. 파일의 열을 그대로 모두 가져옵니다."
        if self.stack.currentIndex() == PAGE_MATCH:
            self._set_heading(f"{name}에서 {people(preview.total_rows)}을 가져옵니다", description)
        self._match_description = description

        used = {combo.currentData() for combo in self.field_combos.values() if combo.currentData()}
        rest = [c.name for c in preview.columns if c.name not in used]
        self.rest_check.setVisible(bool(self.field_combos) and bool(rest))
        self.rest_hint.setVisible(self.rest_check.isVisible())
        if rest:
            shown = ", ".join(rest[:6]) + (f" 외 {len(rest) - 6}개" if len(rest) > 6 else "")
            if self.rest_check.isChecked():
                self.rest_hint.setText(f"{shown}도 명단에 넣습니다. 지금 카드에 쓰지 않아도 나중에 쓸 수 있습니다.")
            else:
                self.rest_hint.setText(f"{shown}{'은' if _final(rest[min(len(rest), 6) - 1]) else '는'} 가져오지 않습니다.")

        if self._has_people():
            current = self.store.count()
            replace_mode = self.mode_control.current() == "replace"
            if replace_mode:
                self.mode_hint.setText(
                    f"지금 있는 {people(current)}을 모두 지우고 새로 넣습니다. 인쇄 기록도 함께 지워지고 되돌릴 수 없습니다.")
            else:
                self.mode_hint.setText(
                    f"지금 있는 {people(current)}은 그대로 두고, 그 뒤에 {people(preview.total_rows)}을 더합니다.")
            for widget in (self.skip_check, self.key_combo, self.key_suffix):
                widget.setEnabled(not replace_mode)
            self.key_combo.setEnabled(not replace_mode and self.skip_check.isChecked())
        self._sync_buttons()

    # ---- 미리 보기 카드 ----
    def _schedule_sample(self) -> None:
        self._preview_timer.start()

    def _step_sample(self, step: int) -> None:
        count = len(self.preview.sample_rows) if self.preview else 0
        if count:
            self._sample_index = max(0, min(count - 1, self._sample_index + step))
            self._render_sample()

    def sample_record(self, index: int | None = None) -> dict | None:
        """미리 보기 카드에 넣을 값. 사진은 폴더에서 찾은 실제 파일."""
        preview = self.preview
        if preview is None or not preview.sample_rows:
            return None
        index = self._sample_index if index is None else index
        row = preview.sample_rows[index]
        names = [c.name for c in preview.columns]
        record: dict[str, Any] = {name: row[i] for i, name in enumerate(names)}
        for target, source in self.mapping().items():
            record[target] = row[names.index(source)] if source in names else ""
        plan_field = self._photo_field
        if plan_field:
            source = self.photo_source_combo.currentData()
            value = row[names.index(source)] if source in names else ""
            found = PhotoLocator(self._photo_folder).find(value) if self._photo_folder else None
            record[plan_field] = str(found) if found else ""
        return record

    def _render_sample(self) -> None:
        preview = self.preview
        if preview is None:
            return
        count = len(preview.sample_rows)
        self.sample_position.setText(f"{self._sample_index + 1} / {count}" if count else "")
        self.sample_prev.setEnabled(self._sample_index > 0)
        self.sample_next.setEnabled(self._sample_index < count - 1)
        document = self._workspace.editor.document
        side = document.active_side_keys()[0] if document.active_side_keys() else "front"
        try:
            image = render_card_image(document, side, self.sample_record(), index=self._sample_index,
                                      width_px=round(640 * max(1.0, self.devicePixelRatioF())))
        except Exception:
            image = None
        self.sample_card.set_image(image)
        if not design_fields(document):
            self.sample_hint.setText("카드에 이름이나 사진 칸을 놓으면 여기에 사람마다 다르게 나옵니다.")
        else:
            self.sample_hint.setText("파일의 처음 몇 명으로 미리 본 모습입니다.")

    # ================================================================ 3. 가져오는 중 / 4. 결과
    def _build_work_page(self) -> QWidget:
        page = QWidget()
        box = QVBoxLayout(page)
        box.setSpacing(SPACE["md"])
        box.addStretch(1)
        self.work_label = label("가져오는 중입니다", "title")
        self.work_label.setAlignment(Qt.AlignHCenter)
        box.addWidget(self.work_label)
        self.work_bar = ProgressBar()
        self.work_bar.setFixedWidth(360)
        box.addWidget(self.work_bar, 0, Qt.AlignHCenter)
        self.work_detail = label("", "muted")
        self.work_detail.setAlignment(Qt.AlignHCenter)
        box.addWidget(self.work_detail)
        box.addStretch(1)
        return page

    def _build_done_page(self) -> QWidget:
        page = QWidget()
        box = QVBoxLayout(page)
        box.setContentsMargins(0, SPACE["sm"], 0, 0)
        box.setSpacing(SPACE["md"])
        self.done_lines = QVBoxLayout()
        self.done_lines.setSpacing(SPACE["sm"])
        box.addLayout(self.done_lines)
        self.warning_toggle = QuietButton("", "sm", icon="chevron-down", accent=True)
        self.warning_toggle.setCheckable(True)
        self.warning_toggle.toggled.connect(self._toggle_warnings)
        toggle_row = QHBoxLayout()
        toggle_row.addWidget(self.warning_toggle)
        toggle_row.addStretch(1)
        box.addLayout(toggle_row)
        self.warning_list = QPlainTextEdit()
        self.warning_list.setReadOnly(True)
        self.warning_list.setVisible(False)
        self.warning_list.setAccessibleName("확인이 필요한 값")
        box.addWidget(self.warning_list, 1)
        self.fix_hint = label("", "muted", wrap=True)
        box.addWidget(self.fix_hint)
        self.done_cards_box = Card(padding=SPACE["lg"], sunken=True, spacing=SPACE["md"])
        self.done_cards_title = label("가져온 사람의 카드는 이렇게 나옵니다", "label")
        self.done_cards_box.body.addWidget(self.done_cards_title)
        cards = QHBoxLayout()
        cards.setSpacing(SPACE["md"])
        self.done_cards: list[CardView] = []
        for _ in range(3):
            view = CardView(well=False)
            view.set_view_width(288)
            self.done_cards.append(view)
            cards.addWidget(view)
        cards.addStretch(1)
        self.done_cards_box.body.addLayout(cards)
        box.addWidget(self.done_cards_box)
        self._done_spacer = QWidget()
        box.addWidget(self._done_spacer, 1)
        return page

    def _fill_done_cards(self, inserted: int) -> None:
        store = self.store
        shown = 0
        if store is not None and inserted:
            fresh = store.count() == inserted
            rows = store.rows(None, None if fresh else [("_id", "desc")], 0, len(self.done_cards))
            if not fresh:
                rows.reverse()
            resolver = PhotoResolver(store)
            document = self._workspace.editor.document
            sides = document.active_side_keys()
            for view, record in zip(self.done_cards, rows):
                try:
                    image = render_card_image(
                        document, sides[0] if sides else "front", resolver.binding_record(record),
                        store=store, record_id=record["_id"], index=shown,
                        width_px=round(576 * max(1.0, self.devicePixelRatioF())))
                except Exception:
                    continue
                view.set_image(image)
                view.setVisible(True)
                shown += 1
        for view in self.done_cards[shown:]:
            view.setVisible(False)
        self.done_cards_box.setVisible(shown > 0)

    def _toggle_warnings(self, on: bool) -> None:
        self.warning_list.setVisible(on)
        self._done_spacer.setVisible(not on)
        if hasattr(self, "done_cards_box"):
            self.done_cards_box.setVisible(not on and any(v.image() is not None for v in self.done_cards))
        self.warning_toggle.set_icon_name("chevron-up" if on else "chevron-down")

    def start_import(self) -> None:
        if self.preview is None or self._busy:
            return
        plan = self.plan()
        try:
            resolve_plan(plan)
        except DataError as exc:
            self._show_match_error(str(exc))
            return
        if self.store is None:
            if self._store_factory is None:
                self._show_match_error("명단을 저장할 곳을 정하지 못했습니다.")
                return
            try:
                self.store = self._store_factory(Path(self.preview.path).stem)
            except (DataError, OSError) as exc:
                self._show_match_error(f"명단 파일을 만들 수 없습니다. {exc}")
                return
            self.created_store = True
        store = self.store
        total = self.preview.total_rows
        self.work_label.setText(f"{people(total)}을 가져오는 중입니다")
        self.work_bar.setRange(0, max(1, total))
        self.work_bar.setValue(0)
        self.work_detail.setText("")
        self._show_page(PAGE_WORK)
        self._run(lambda report, cancelled: run_import(store, plan, report, cancelled),
                  self._import_done, self._import_failed, on_progress=self._on_progress,
                  on_cancelled=self._import_cancelled)

    def _on_progress(self, done: int, total: int) -> None:
        self.work_bar.setRange(0, max(1, total))
        self.work_bar.setValue(done)
        self.work_detail.setText(f"{people(total)} 중 {people(done)}")

    def _show_match_error(self, message: str) -> None:
        self.error = message
        self._show_page(PAGE_MATCH)
        self.match_banner.set_text(message)
        self.match_banner.setVisible(True)

    def _discard_created_store(self) -> None:
        if self.created_store and self.store is not None:
            path = self.store.path
            self.store.close()
            for suffix in ("", "-wal", "-shm"):
                try:
                    os.remove(path + suffix)
                except OSError:
                    pass
            self.store = None
            self.created_store = False

    def _import_failed(self, message: str) -> None:
        self._discard_created_store()
        self._show_match_error(message)

    def _import_cancelled(self) -> None:
        self._discard_created_store()
        self._show_page(PAGE_MATCH)
        self.match_banner.set_kind("info")
        self.match_banner.set_text("가져오기를 멈췄습니다. 명단은 바뀌지 않았습니다.")
        self.match_banner.setVisible(True)

    def _import_done(self, outcome: ImportOutcome) -> None:
        self.outcome = outcome
        result = outcome.result
        if result.inserted:
            heading = f"{outcome.file_name}에서 {people(result.inserted)}을 가져왔습니다"
        else:
            heading = "새로 가져온 사람이 없습니다"
        self._set_heading(heading, f"이제 명단에는 모두 {people(outcome.total)}이 있습니다.")

        while self.done_lines.count():
            child = self.done_lines.takeAt(0)
            if child.layout() is not None:
                while child.layout().count():
                    inner = child.layout().takeAt(0)
                    if inner.widget() is not None:
                        inner.widget().deleteLater()
        lines: list[tuple[str, str, str]] = []
        if result.created_columns:
            names = ", ".join(result.created_columns[:8])
            more = f" 외 {len(result.created_columns) - 8}개" if len(result.created_columns) > 8 else ""
            lines.append(("check", "ok_text", f"명단에 들어간 항목: {names}{more}"))
        if outcome.skipped_existing:
            lines.append(("info", "info_text", f"이미 명단에 있는 {people(outcome.skipped_existing)}은 건너뛰었습니다."))
        if outcome.photo_field and (outcome.photo_found or outcome.photo_missing):
            if outcome.photo_missing:
                lines.append(("warning", "warn_text",
                              f"사진을 못 찾은 사람이 {people(outcome.photo_missing)} 있습니다. "
                              "명단 위쪽의 '사진 없음'을 누르면 모아 볼 수 있고, 사진 칸을 두 번 눌러 넣을 수 있습니다."))
            else:
                lines.append(("check", "ok_text", f"{people(outcome.photo_found)} 모두 사진을 찾았습니다."))
        blank = result.skipped
        if blank:
            lines.append(("info", "info_text", f"빈 줄 {blank}개는 건너뛰었습니다."))
        for icon, role, text in lines:
            row = QHBoxLayout()
            row.setSpacing(SPACE["sm"])
            mark = IconWidget(icon, 18, role)
            row.addWidget(mark, 0, Qt.AlignTop)
            row.addWidget(label(text, "secondary", wrap=True), 1)
            self.done_lines.addLayout(row)

        value_warnings = [w for w in result.warnings]
        self.warning_toggle.setVisible(bool(value_warnings))
        self.warning_toggle.setText(f"확인이 필요한 값 {len(value_warnings)}건 보기")
        self.warning_toggle.setChecked(False)
        self.warning_list.setPlainText("\n".join(value_warnings))
        self._toggle_warnings(False)
        self._fill_done_cards(result.inserted)
        if value_warnings:
            self.fix_hint.setText("값은 파일에 적힌 그대로 넣었습니다. 명단에서 그 칸을 골라 Enter를 누르면 바로 고칠 수 있습니다. "
                                  "원래 파일은 바뀌지 않았습니다.")
        else:
            self.fix_hint.setText("원래 파일은 바뀌지 않았습니다. 명단에서 고친 내용은 이 프로그램 안에만 저장됩니다.")
        self._show_page(PAGE_DONE)

    # ================================================================ 공통
    def _set_heading(self, title: str, description: str | None) -> None:
        self.set_title(title, window_title=False)
        self._heading_description = description or ""
        self.set_description(description)

    def description(self) -> str:
        return self._heading_description

    def _show_page(self, page: int) -> None:
        self.stack.setCurrentIndex(page)
        if page == PAGE_FILE and self.preview is not None:
            name = Path(self.preview.path).name
            self._set_heading(f"{name}에서 {people(self.preview.total_rows)}을 찾았습니다",
                              "아래처럼 읽었습니다. 맞으면 다음으로 넘어가세요. 원래 파일은 고치지 않습니다.")
        elif page == PAGE_MATCH and self.preview is not None:
            self._sync_match_summary()
            self._render_sample()
        self._sync_buttons()

    def page_index(self) -> int:
        return self.stack.currentIndex()

    def _sync_buttons(self) -> None:
        page = self.stack.currentIndex()
        loading = self._busy and page == PAGE_FILE
        self.preview_table.setVisible(not loading)
        self.loading_box.setVisible(loading)
        ready = self.preview is not None and self.preview.total_rows > 0 and not self._busy
        self.back_button.setVisible(page == PAGE_MATCH)
        self.cancel_button.setVisible(page != PAGE_DONE)
        self.cancel_button.setText("가져오기 멈추기" if page == PAGE_WORK else "취소")
        self.next_button.setVisible(page != PAGE_WORK)
        if page == PAGE_FILE:
            self.next_button.setText("다음")
            self.next_button.setEnabled(ready)
        elif page == PAGE_MATCH:
            total = self.preview.total_rows if self.preview else 0
            replace_mode = self._has_people() and self.mode_control.current() == "replace"
            self.next_button.setText(f"{people(total)}으로 바꾸기" if replace_mode else f"{people(total)} 가져오기")
            self.next_button.setEnabled(ready)
        elif page == PAGE_DONE:
            self.next_button.setText("명단 보기")
            self.next_button.setEnabled(True)

    def go_next(self) -> None:
        page = self.stack.currentIndex()
        if page == PAGE_FILE and self.preview is not None and self.preview.total_rows > 0 and not self._busy:
            self._show_page(PAGE_MATCH)
        elif page == PAGE_MATCH:
            self.start_import()
        elif page == PAGE_DONE:
            self.accept()

    def go_back(self) -> None:
        if self.stack.currentIndex() == PAGE_MATCH:
            self._show_page(PAGE_FILE)

    def _on_cancel(self) -> None:
        if self._busy and self._worker is not None and self.stack.currentIndex() == PAGE_WORK:
            self._worker.cancel()
            return
        self.reject()

    def reject(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self._worker.wait(30000)
            QApplication.sendPostedEvents(self)
            self._drop_worker()
            self._busy = False
        if self.outcome is None:
            self._discard_created_store()
            super().reject()
        else:
            super().accept()

    # ---- 일을 다른 스레드에서 ----
    def is_busy(self) -> bool:
        return self._busy

    def _set_busy(self, busy: bool) -> None:
        if busy != self._busy:
            self._busy = busy
            self.busyChanged.emit(busy)
        self._sync_buttons()

    def _run(self, job, on_done, on_failed, on_progress=None, on_cancelled=None, loading: bool = False) -> None:
        self._drop_worker()
        self._handlers = (on_done, on_failed, on_cancelled or (lambda: None), on_progress)
        if not self._threaded:
            self._set_busy(True)
            try:
                result = job(on_progress or (lambda done, total: None), lambda: False)
            except DataError as exc:
                self._set_busy(False)
                on_failed(str(exc))
                return
            self._set_busy(False)
            on_done(result)
            return
        worker = Worker(job, self)
        self._worker = worker
        worker.succeeded.connect(self._worker_succeeded, Qt.QueuedConnection)
        worker.failed.connect(self._worker_failed, Qt.QueuedConnection)
        worker.cancelled.connect(self._worker_cancelled, Qt.QueuedConnection)
        worker.progress.connect(self._worker_progress, Qt.QueuedConnection)
        self._set_busy(True)
        worker.start()

    def _drop_worker(self) -> None:
        worker, self._worker = self._worker, None
        if worker is None:
            return
        worker.cancel()
        for signal in (worker.succeeded, worker.failed, worker.cancelled, worker.progress):
            signal.disconnect()
        worker.wait(30000)
        worker.deleteLater()

    def _settle(self) -> tuple:
        worker, self._worker = self._worker, None
        if worker is not None:
            worker.wait(30000)
            worker.deleteLater()
        self._set_busy(False)
        return self._handlers

    def _worker_succeeded(self, result) -> None:
        if self.sender() is self._worker:
            self._settle()[0](result)

    def _worker_failed(self, message: str) -> None:
        if self.sender() is self._worker:
            self._settle()[1](message)

    def _worker_cancelled(self) -> None:
        if self.sender() is self._worker:
            self._settle()[2]()

    def _worker_progress(self, done: int, total: int) -> None:
        if self.sender() is self._worker and self._handlers[3] is not None:
            self._handlers[3](done, total)

    def wait_idle(self, timeout_ms: int = 30000) -> bool:
        """하던 일이 끝날 때까지 화면을 돌리며 기다린다(테스트, 스크린샷용)."""
        from PySide6.QtCore import QElapsedTimer

        timer = QElapsedTimer()
        timer.start()
        while self._busy and timer.elapsed() < timeout_ms:
            QApplication.processEvents()
            QThread.msleep(2)
        QApplication.processEvents()
        return not self._busy

    def closeEvent(self, event) -> None:
        self.reject()
        event.accept()
