"""여러 명단 항목과 글자를 이어 붙이는 창."""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLineEdit, QMenu, QVBoxLayout, QWidget

from design.binding import ConcatPart, ConcatSource
from ui.theme import SPACE
from ui.widgets import ComboBox, Dialog, IconButton, SecondaryButton, SegmentedControl, label

from ..common import as_menu_button, group

SEPARATORS = ((" ", "띄어 쓰기"), ("", "붙여 쓰기"), ("custom", "사이에 글자 넣기"))


class _PartRow(QFrame):
    changed = Signal()
    removeRequested = Signal(object)
    moveRequested = Signal(object, int)

    def __init__(self, part: ConcatPart, columns: list[str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._part = part
        self.is_column = bool(part.column)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.prefix_edit = self.suffix_edit = self.text_edit = self.column_combo = None
        if self.is_column:
            self.prefix_edit = QLineEdit(part.prefix)
            self.prefix_edit.setPlaceholderText("앞에 붙일 글자")
            self.prefix_edit.setAccessibleName("앞에 붙일 글자")
            self.prefix_edit.setFixedWidth(112)
            self.column_combo = ComboBox()
            self.column_combo.setAccessibleName("명단 항목")
            for name in columns:
                self.column_combo.addItem(name, name)
            if part.column not in columns:
                self.column_combo.addItem(part.column, part.column)
            self.column_combo.setCurrentIndex(self.column_combo.findData(part.column))
            self.suffix_edit = QLineEdit(part.suffix)
            self.suffix_edit.setPlaceholderText("뒤에 붙일 글자")
            self.suffix_edit.setAccessibleName("뒤에 붙일 글자")
            self.suffix_edit.setFixedWidth(112)
            layout.addWidget(self.prefix_edit)
            layout.addWidget(self.column_combo, 1)
            layout.addWidget(self.suffix_edit)
            self.prefix_edit.textChanged.connect(self.changed)
            self.suffix_edit.textChanged.connect(self.changed)
            self.column_combo.currentIndexChanged.connect(self.changed)
        else:
            self.text_edit = QLineEdit(part.text)
            self.text_edit.setPlaceholderText("그대로 찍을 글자")
            self.text_edit.setAccessibleName("그대로 찍을 글자")
            layout.addWidget(self.text_edit, 1)
            self.text_edit.textChanged.connect(self.changed)
        self.up_button = IconButton("chevron-up", "앞으로 옮기기", size="sm")
        self.down_button = IconButton("chevron-down", "뒤로 옮기기", size="sm")
        self.remove_button = IconButton("close", "빼기", size="sm")
        self.up_button.clicked.connect(lambda: self.moveRequested.emit(self, -1))
        self.down_button.clicked.connect(lambda: self.moveRequested.emit(self, 1))
        self.remove_button.clicked.connect(lambda: self.removeRequested.emit(self))
        for button in (self.up_button, self.down_button, self.remove_button):
            layout.addWidget(button)

    def part(self) -> ConcatPart:
        if self.is_column:
            return ConcatPart(
                column=self.column_combo.currentData() or self._part.column, text="",
                prefix=self.prefix_edit.text(), suffix=self.suffix_edit.text(), extra=dict(self._part.extra),
            )
        return ConcatPart(column=None, text=self.text_edit.text(), extra=dict(self._part.extra))


class ConcatBuilder(QWidget):
    """parts를 위에서 아래로 이어 붙인다. source()가 결과다."""

    changed = Signal()

    def __init__(self, columns: list[str], source: ConcatSource | None = None,
                 preview: Callable[[ConcatSource], str] | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._columns = list(columns)
        self._preview = preview
        self._extra = dict(source.extra) if source is not None else {}
        self.rows: list[_PartRow] = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACE["md"])

        self._rows_host = QWidget()
        self._rows_box = QVBoxLayout(self._rows_host)
        self._rows_box.setContentsMargins(0, 0, 0, 0)
        self._rows_box.setSpacing(SPACE["sm"])
        layout.addWidget(self._rows_host)
        self.empty_label = label("아래 버튼으로 이어 붙일 것을 넣으세요.", "muted")
        layout.addWidget(self.empty_label)

        buttons = QHBoxLayout()
        buttons.setSpacing(SPACE["sm"])
        self.add_column_button = SecondaryButton("명단 항목 넣기", size="sm", icon="plus")
        as_menu_button(self.add_column_button)
        self._column_menu = QMenu(self)
        for name in self._columns:
            action = QAction(name, self)
            action.triggered.connect(lambda _=False, column=name: self.add_column_part(column))
            self._column_menu.addAction(action)
        self.add_column_button.setMenu(self._column_menu)
        self.add_column_button.setEnabled(bool(self._columns))
        self.add_text_button = SecondaryButton("글자 넣기", size="sm", icon="plus")
        self.add_text_button.clicked.connect(lambda: self.add_text_part(""))
        buttons.addWidget(self.add_column_button)
        buttons.addWidget(self.add_text_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self.separator_control = SegmentedControl(SEPARATORS)
        self.separator_edit = QLineEdit()
        self.separator_edit.setPlaceholderText("예: · 또는 -")
        self.separator_edit.setAccessibleName("사이에 넣을 글자")
        self.separator_edit.setFixedWidth(140)
        separator_row = QHBoxLayout()
        separator_row.setSpacing(SPACE["sm"])
        separator_row.addWidget(self.separator_control)
        separator_row.addWidget(self.separator_edit)
        separator_row.addStretch(1)
        holder = QWidget()
        holder.setLayout(separator_row)
        separator_row.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(group("항목 사이", holder))

        self.preview_label = label("", "strong", wrap=True)
        layout.addWidget(group("이렇게 찍힙니다", self.preview_label))

        separator = source.separator if source is not None else " "
        if separator in (" ", ""):
            self.separator_control.set_current(separator, emit=False)
        else:
            self.separator_control.set_current("custom", emit=False)
            self.separator_edit.setText(separator)
        for part in (source.parts if source is not None else []):
            self._append(part)
        self.separator_control.currentChanged.connect(self._on_changed)
        self.separator_edit.textChanged.connect(self._on_changed)
        self._on_changed()

    def _append(self, part: ConcatPart) -> _PartRow:
        row = _PartRow(part, self._columns)
        row.changed.connect(self._on_changed)
        row.removeRequested.connect(self._remove)
        row.moveRequested.connect(self._move)
        self.rows.append(row)
        self._rows_box.addWidget(row)
        return row

    def add_column_part(self, column: str, prefix: str = "", suffix: str = "") -> None:
        self._append(ConcatPart(column=column, prefix=prefix, suffix=suffix))
        self._on_changed()

    def add_text_part(self, text: str) -> None:
        row = self._append(ConcatPart(text=text))
        self._on_changed()
        row.text_edit.setFocus()

    def _remove(self, row: _PartRow) -> None:
        self.rows.remove(row)
        self._rows_box.removeWidget(row)
        row.hide()
        row.deleteLater()
        self._on_changed()

    def _move(self, row: _PartRow, delta: int) -> None:
        index = self.rows.index(row)
        target = index + delta
        if not 0 <= target < len(self.rows):
            return
        self.rows.insert(target, self.rows.pop(index))
        for existing in self.rows:
            self._rows_box.removeWidget(existing)
        for existing in self.rows:
            self._rows_box.addWidget(existing)
        self._on_changed()

    def separator(self) -> str:
        key = self.separator_control.current()
        return self.separator_edit.text() if key == "custom" else key

    def source(self) -> ConcatSource:
        return ConcatSource(parts=[row.part() for row in self.rows], separator=self.separator(),
                            extra=dict(self._extra))

    def _on_changed(self, *_args) -> None:
        self.separator_edit.setVisible(self.separator_control.current() == "custom")
        self.empty_label.setVisible(not self.rows)
        for index, row in enumerate(self.rows):
            row.up_button.setEnabled(index > 0)
            row.down_button.setEnabled(index < len(self.rows) - 1)
        text = self._preview(self.source()) if self._preview and self.rows else ""
        self.preview_label.setText(text or "(아직 내용이 없습니다)")
        self.changed.emit()


class ConcatDialog(Dialog):
    def __init__(self, columns: list[str], source: ConcatSource | None,
                 preview: Callable[[ConcatSource], str] | None, parent: QWidget | None = None) -> None:
        super().__init__(
            "여러 항목 이어 붙이기",
            "명단 항목과 글자를 순서대로 이어 한 칸에 찍습니다. 예: 부서 + 직급 → 경영지원팀 사원. "
            "항목이 비어 있는 사람은 앞뒤에 붙인 글자도 함께 빠집니다.",
            parent, width=600,
        )
        self.builder = ConcatBuilder(columns, source, preview)
        self.set_body(self.builder)
        self.add_button("취소", "secondary", "reject")
        self.apply_button = self.add_button("이대로 넣기", "primary", "accept", default=True)
        self.builder.changed.connect(self._sync)
        self._sync()

    def _sync(self) -> None:
        self.apply_button.setEnabled(any(row.is_column for row in self.builder.rows))


def describe(source: ConcatSource) -> str:
    """'부서 + 직급'처럼 짧게 요약한다."""
    names = [part.column if part.column else f"“{part.text}”" for part in source.parts if part.column or part.text]
    return " + ".join(names) if names else "(비어 있음)"
