"""글자를 골랐을 때: 내용(직접 입력 / 명단 항목 / 날짜·번호), 글꼴, 크기, 모양, 색, 맞춤."""

from __future__ import annotations

import datetime as _dt
from dataclasses import replace

from PySide6.QtWidgets import (
    QCheckBox, QDialog, QLineEdit, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget,
)

from design.binding import (
    ColumnSource, ConcatPart, ConcatSource, CounterSource, DateSource, FixedSource, format_date, resolve,
)
from design.model import TextItem
from ui.widgets import ComboBox, QuietButton, SecondaryButton, SegmentedControl, StepperField, SwatchPicker, label

from ..common import LinkButton, NoteRow, field_row, group, hbox, link_row
from .base import SWATCHES, FontCombo, InspectorContext, Section, set_combo_data, set_text_if_changed, show_color
from .concat import ConcatDialog, describe
from .detail import ColumnCombo

DATE_FORMATS = ("%Y.%m.%d", "%Y-%m-%d", "%Y년 %-m월 %-d일", "%y.%m.%d", "%Y.%m")
DEFAULT_COLUMN = "이름"


def source_mode(source) -> str:
    if isinstance(source, (ColumnSource, ConcatSource)):
        return "column"
    if isinstance(source, (DateSource, CounterSource)):
        return "auto"
    return "fixed"


class _GlyphToggle(QPushButton):
    def __init__(self, tooltip: str, bold=False, italic=False, underline=False, parent=None) -> None:
        super().__init__("가", parent)
        self.setProperty("glyph", "true")
        self.setCheckable(True)
        self.setAutoDefault(False)
        self.setFixedSize(40, 40)
        self.setToolTip(tooltip)
        self.setAccessibleName(tooltip)
        font = self.font()
        font.setBold(bold)
        font.setItalic(italic)
        font.setUnderline(underline)
        self.setFont(font)


class TextSection(Section):
    def __init__(self, context: InspectorContext, parent: QWidget | None = None) -> None:
        super().__init__(context, parent)

        # ---- 내용 ----
        self.mode_control = SegmentedControl(
            [("fixed", "직접 입력"), ("column", "명단 항목"), ("auto", "날짜·번호")], stretch=True)
        self.mode_control.currentChanged.connect(self.set_mode)
        self.mode_pages: list[QWidget] = []

        self.text_edit = QPlainTextEdit()
        self.text_edit.setObjectName("ContentEdit")
        self.text_edit.setPlaceholderText("카드에 찍을 글자")
        self.text_edit.setAccessibleName("카드에 찍을 글자")
        self.text_edit.setTabChangesFocus(True)
        self.text_edit.textChanged.connect(self._on_text)
        self.mode_pages.append(self._page(self.text_edit))

        self.column_combo = ColumnCombo()
        self.column_combo.setAccessibleName("보여 줄 명단 항목")
        self.column_combo.activated.connect(self._on_column)
        self.column_hint = label("", "hint", wrap=True)
        self.concat_button = LinkButton("여러 항목 이어 붙이기")
        self.concat_button.setToolTip("부서와 직급처럼 두 항목 이상을 한 칸에 찍습니다")
        self.concat_button.clicked.connect(self.open_concat_dialog)
        self.concat_note = NoteRow("note", "")
        self.concat_edit_button = SecondaryButton("고치기", size="sm")
        self.concat_edit_button.clicked.connect(self.open_concat_dialog)
        self.concat_single_button = QuietButton("한 항목만 쓰기", size="sm")
        self.concat_single_button.clicked.connect(self.use_single_column)
        self.concat_tools = hbox(self.concat_edit_button, self.concat_single_button, None)
        self.mode_pages.append(self._page(
            self.column_combo, self.concat_note, self.concat_tools, self.column_hint,
            link_row(self.concat_button)))

        self.auto_control = SegmentedControl([("date", "날짜"), ("counter", "일련번호")], stretch=True)
        self.auto_control.currentChanged.connect(self.set_auto_kind)
        self.date_combo = ComboBox()
        self.date_combo.setAccessibleName("날짜 모양")
        self.date_combo.activated.connect(self._on_date)
        self.date_offset = StepperField(0, -3650, 3650, 1, 0, "일 뒤")
        self.date_offset.valueChanged.connect(self._on_date)
        self.date_box = self._page(
            self.date_combo, field_row("날짜 옮기기", self.date_offset),
            label("0이면 인쇄하는 날의 날짜입니다. 유효기간처럼 1년 뒤를 찍으려면 365로 맞추세요.", "hint", wrap=True))
        self.counter_start = StepperField(1, 0, 99999999, 1, 0)
        self.counter_start.valueChanged.connect(self._on_counter)
        self.counter_pad = StepperField(0, 0, 12, 1, 0, "자리")
        self.counter_pad.valueChanged.connect(self._on_counter)
        self.counter_prefix = QLineEdit()
        self.counter_prefix.setPlaceholderText("예: No. ")
        self.counter_prefix.textEdited.connect(self._on_counter)
        self.counter_box = self._page(
            field_row("시작 번호", self.counter_start), field_row("자릿수", self.counter_pad),
            field_row("앞 글자", self.counter_prefix),
            label("한 장 찍을 때마다 1씩 커집니다. 자릿수를 4로 하면 0001처럼 앞을 0으로 채웁니다.", "hint", wrap=True))
        self.mode_pages.append(self._page(self.auto_control, self.date_box, self.counter_box))
        self.column.addWidget(group("내용", self.mode_control, *self.mode_pages))

        # ---- 글꼴 ----
        self.font_combo = FontCombo()
        self.font_combo.activated.connect(lambda _i: self.edit({"font_family": self.font_combo.family()}))
        self.column.addWidget(group("글꼴", self.font_combo))

        self.size_field = StepperField(14, 4, 300, 1, 1, "pt")
        self.size_field.valueChanged.connect(lambda value: self.edit({"font_size_pt": float(value)}, merge="size"))
        self.column.addWidget(field_row("크기", self.size_field))

        self.bold_toggle = _GlyphToggle("굵게", bold=True)
        self.italic_toggle = _GlyphToggle("기울이기", italic=True)
        self.underline_toggle = _GlyphToggle("밑줄", underline=True)
        self.bold_toggle.toggled.connect(lambda on: self.edit({"bold": on}))
        self.italic_toggle.toggled.connect(lambda on: self.edit({"italic": on}))
        self.underline_toggle.toggled.connect(lambda on: self.edit({"underline": on}))
        self.column.addWidget(field_row(
            "모양", hbox(self.bold_toggle, self.italic_toggle, self.underline_toggle, None, spacing=6)))

        self.color_picker = SwatchPicker(SWATCHES)
        self.color_picker.colorChanged.connect(lambda color: self.edit({"color": color}, merge="color"))
        self.column.addWidget(group("글자색", self.color_picker))

        self.align_control = SegmentedControl(
            [("left", "왼쪽"), ("center", "가운데"), ("right", "오른쪽")], stretch=True)
        self.align_control.currentChanged.connect(lambda key: self.edit({"h_align": key}))
        self.column.addWidget(group("맞춤", self.align_control))

        self.shrink_check = QCheckBox("길면 글자를 줄여 맞춤")
        self.shrink_check.setToolTip("이름이 긴 사람도 칸 밖으로 넘치지 않게 글자를 알아서 줄입니다")
        self.shrink_check.toggled.connect(lambda on: self.edit({"auto_shrink": on}))
        self.column.addWidget(self.shrink_check)

        # ---- 자세한 설정에 들어갈 것 ----
        self.v_align_control = SegmentedControl(
            [("top", "위"), ("middle", "가운데"), ("bottom", "아래")], stretch=True)
        self.v_align_control.currentChanged.connect(lambda key: self.edit({"v_align": key}))
        self.wrap_check = QCheckBox("칸 너비에 맞춰 여러 줄로 나누기")
        self.wrap_check.toggled.connect(lambda on: self.edit({"wrap": on}))
        self._detail = [group("칸 안에서 세로 맞춤", self.v_align_control), self.wrap_check]

    @staticmethod
    def _page(*widgets: QWidget) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        for widget in widgets:
            layout.addWidget(widget)
        return page

    def detail_widgets(self) -> list[QWidget]:
        return self._detail

    # ---- 값 채우기 ----
    def load(self) -> None:
        item = self.item
        if item is None:
            return
        source = item.source
        mode = source_mode(source)
        self.mode_control.set_current(mode, emit=False)
        self._show_mode(mode)
        if mode == "fixed":
            set_text_if_changed(self.text_edit, source.value if isinstance(source, FixedSource) else item.text)
        elif mode == "column":
            self._load_column(source)
        else:
            self._load_auto(source)
        self.font_combo.set_family(item.font_family)
        self.size_field.set_value(item.font_size_pt, emit=False)
        for toggle, value in ((self.bold_toggle, item.bold), (self.italic_toggle, item.italic),
                              (self.underline_toggle, item.underline)):
            toggle.setChecked(bool(value))
        show_color(self.color_picker, item.color)
        self.align_control.set_current(item.h_align, emit=False)
        self.shrink_check.setChecked(bool(item.auto_shrink))
        self.v_align_control.set_current(item.v_align, emit=False)
        self.wrap_check.setChecked(bool(item.wrap))

    def _show_mode(self, mode: str) -> None:
        index = {"fixed": 0, "column": 1, "auto": 2}[mode]
        for i, page in enumerate(self.mode_pages):
            page.setVisible(i == index)

    def _text_columns(self) -> list[str]:
        return [name for name, kind in self.context.columns() if kind != "photo"]

    def _load_column(self, source) -> None:
        is_concat = isinstance(source, ConcatSource)
        self.column_combo.setVisible(not is_concat)
        self.concat_button.setVisible(not is_concat)
        self.concat_note.setVisible(is_concat)
        self.concat_tools.setVisible(is_concat)
        has_list = self.context.has_list()
        if is_concat:
            self.concat_note.set_text(f"{describe(source)} 순서로 이어 붙였습니다.")
            missing = sorted(source.columns() - {name for name, _ in self.context.columns()}) if has_list else []
        else:
            self.column_combo.set_columns(self._text_columns(), source.column, has_list)
            missing = [source.column] if has_list and source.column not in self._text_columns() else []
        if not has_list:
            self.column_hint.setText("명단을 넣으면 사람마다 실제 내용으로 바뀝니다. 지금은 «칸 이름»으로 보입니다.")
        elif missing:
            self.column_hint.setText(f"열린 명단에 '{', '.join(missing)}' 항목이 없습니다. 다른 항목을 고르세요.")
        else:
            self.column_hint.setText("")
        self.column_hint.setVisible(bool(self.column_hint.text()))
        self.column_hint.setProperty("tone", "warn" if missing else "")
        self.column_hint.style().unpolish(self.column_hint)
        self.column_hint.style().polish(self.column_hint)

    def _load_auto(self, source) -> None:
        is_date = isinstance(source, DateSource)
        self.auto_control.set_current("date" if is_date else "counter", emit=False)
        self.date_box.setVisible(is_date)
        self.counter_box.setVisible(not is_date)
        if is_date:
            now = _dt.datetime.now()
            formats = list(DATE_FORMATS)
            if source.format not in formats:
                formats.append(source.format)
            self.date_combo.blockSignals(True)
            self.date_combo.clear()
            for fmt in formats:
                self.date_combo.addItem(format_date(now, fmt), fmt)
            set_combo_data(self.date_combo, source.format)
            self.date_combo.blockSignals(False)
            self.date_offset.set_value(source.offset_days, emit=False)
        else:
            self.counter_start.set_value(source.start, emit=False)
            self.counter_pad.set_value(source.pad, emit=False)
            if not self.counter_prefix.hasFocus():
                set_text_if_changed(self.counter_prefix, source.prefix)

    # ---- 내용 바꾸기 ----
    def set_mode(self, mode: str) -> None:
        item = self.item
        if self._loading or item is None or mode == source_mode(item.source):
            return
        if mode == "fixed":
            if isinstance(item.source, FixedSource):
                self.edit({"source": None, "text": item.source.value}, text="내용 연결 변경")
            else:
                self.set_source(None)
        elif mode == "column":
            self.set_source(ColumnSource(column=self._default_column(item)))
        else:
            self.set_source(DateSource(format=DATE_FORMATS[0]))

    def _default_column(self, item) -> str:
        columns = self._text_columns()
        if item.name in columns:
            return item.name
        return columns[0] if columns else DEFAULT_COLUMN

    def set_column(self, column: str) -> None:
        if column:
            self.set_source(ColumnSource(column=column))

    def _on_column(self, _index: int) -> None:
        self.set_column(self.column_combo.column())

    def use_single_column(self) -> None:
        item = self.item
        if item is None:
            return
        columns = sorted(item.source.columns()) if item.source is not None else []
        first = next((part.column for part in getattr(item.source, "parts", []) if part.column), None)
        self.set_source(ColumnSource(column=first or (columns[0] if columns else self._default_column(item))))

    def preview_text(self, source) -> str:
        probe = TextItem(source=source)
        return str(resolve(probe, self.controller.preview_record) or "")

    def concat_seed(self) -> ConcatSource:
        source = self.item.source if self.item is not None else None
        if isinstance(source, ConcatSource):
            return source
        if isinstance(source, ColumnSource) and source.column:
            return ConcatSource(parts=[ConcatPart(column=source.column)], separator=" ")
        return ConcatSource(separator=" ")

    def open_concat_dialog(self) -> None:
        if self.item is None:
            return
        seed = self.concat_seed()
        columns = self._text_columns() or sorted(seed.columns()) or [DEFAULT_COLUMN]
        dialog = ConcatDialog(columns, seed, self.preview_text, self.window())
        if dialog.exec() == QDialog.Accepted:
            self.apply_concat(dialog.builder.source())

    def apply_concat(self, source: ConcatSource) -> None:
        self.set_source(source)

    def set_auto_kind(self, kind: str) -> None:
        if self._loading or self.item is None:
            return
        if kind == "date" and not isinstance(self.item.source, DateSource):
            self.set_source(DateSource(format=DATE_FORMATS[0]))
        elif kind == "counter" and not isinstance(self.item.source, CounterSource):
            self.set_source(CounterSource(start=1, pad=4))

    def _on_text(self) -> None:
        item = self.item
        if self._loading or item is None or source_mode(item.source) != "fixed":
            return
        text = self.text_edit.toPlainText()
        if isinstance(item.source, FixedSource):
            self.edit({"source": FixedSource(value=text)}, merge="text", text="글자 수정")
        else:
            self.edit({"text": text}, merge="text", text="글자 수정")

    def _on_date(self, *_args) -> None:
        item = self.item
        if self._loading or item is None or not isinstance(item.source, DateSource):
            return
        updated = replace(item.source, format=self.date_combo.currentData() or item.source.format,
                          offset_days=int(self.date_offset.value()))
        self.edit({"source": updated}, merge="date", text="날짜 설정")

    def _on_counter(self, *_args) -> None:
        item = self.item
        if self._loading or item is None or not isinstance(item.source, CounterSource):
            return
        updated = replace(item.source, start=int(self.counter_start.value()), pad=int(self.counter_pad.value()),
                          prefix=self.counter_prefix.text())
        self.edit({"source": updated}, merge="counter", text="일련번호 설정")

    def focus_content(self) -> None:
        mode = source_mode(self.item.source) if self.item is not None else "fixed"
        target = {"fixed": self.text_edit, "column": self.column_combo, "auto": self.auto_control}[mode]
        target.setFocus()
