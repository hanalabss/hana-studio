"""여러 개를 골랐을 때: 줄 맞춤, 간격 맞춤, 함께 바꿀 수 있는 속성."""

from __future__ import annotations

from PySide6.QtCore import QSize
from PySide6.QtWidgets import QHBoxLayout, QWidget

from ui.editor import EditorError
from ui.theme import ThemeManager
from ui.widgets import IconButton, SecondaryButton, StepperField, SwatchPicker

from .. import icons
from ..canvas_bar import ALIGN_ACTIONS, DISTRIBUTE_ACTIONS
from ..common import NoteRow, field_row, group, hbox
from .base import SWATCHES, FontCombo, InspectorContext, Section, show_color


class MultiSection(Section):
    def __init__(self, context: InspectorContext, parent: QWidget | None = None) -> None:
        super().__init__(context, parent)
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(6)
        self.align_buttons: dict[str, IconButton] = {}
        self._icon_names: dict[str, str] = {}
        for mode, text, icon_name in ALIGN_ACTIONS:
            button = IconButton("plus", text, variant="secondary")
            button.set_icon_name(None)
            button.clicked.connect(lambda _=False, m=mode: self.align(m))
            self.align_buttons[mode] = button
            self._icon_names[mode] = icon_name
            row_layout.addWidget(button)
        row_layout.addStretch(1)
        self.column.addWidget(group("줄 맞추기", row))

        self.distribute_buttons: dict[str, SecondaryButton] = {}
        for axis, text, _icon in DISTRIBUTE_ACTIONS:
            button = SecondaryButton(text, size="sm")
            button.clicked.connect(lambda _=False, a=axis: self.distribute(a))
            self.distribute_buttons[axis] = button
        self.distribute_hint = NoteRow("note", "간격은 3개 이상 골랐을 때 맞출 수 있습니다.")
        self.column.addWidget(group(
            "사이 간격", hbox(self.distribute_buttons["horizontal"], self.distribute_buttons["vertical"], None),
            self.distribute_hint))

        self.font_combo = FontCombo()
        self.font_combo.activated.connect(lambda _i: self.edit({"font_family": self.font_combo.family()}))
        self.size_field = StepperField(14, 4, 300, 1, 1, "pt")
        self.size_field.valueChanged.connect(lambda value: self.edit({"font_size_pt": float(value)}, merge="size"))
        self.color_picker = SwatchPicker(SWATCHES)
        self.color_picker.colorChanged.connect(lambda color: self.edit({"color": color}, merge="color"))
        self.font_box = group("글꼴", self.font_combo)
        self.size_row = field_row("크기", self.size_field)
        self.color_box = group("색", self.color_picker)
        for widget in (self.font_box, self.size_row, self.color_box):
            self.column.addWidget(widget)
        self.mixed_note = NoteRow("note", "종류가 서로 달라 줄 맞춤과 자세한 설정만 함께 바꿀 수 있습니다.")
        self.column.addWidget(self.mixed_note)

        ThemeManager.instance().themeChanged.connect(self._refresh_icons)
        self._refresh_icons()

    def _refresh_icons(self, *_args) -> None:
        for mode, button in self.align_buttons.items():
            button.setIcon(icons.icon(self._icon_names[mode], "text", 18))
            button.setIconSize(QSize(18, 18))

    def load(self) -> None:
        items = self.items
        if not items:
            return
        all_text = all(item.kind == "text" for item in items)
        has_color = all(hasattr(item, "color") and item.kind in ("text", "line") for item in items)
        self.font_box.setVisible(all_text)
        self.size_row.setVisible(all_text)
        self.color_box.setVisible(has_color)
        self.mixed_note.setVisible(not has_color)
        if all_text:
            self.font_combo.set_family(items[0].font_family)
            self.size_field.set_value(items[0].font_size_pt, emit=False)
        if has_color:
            show_color(self.color_picker, items[0].color)
        enough = len(items) >= 3
        for button in self.distribute_buttons.values():
            button.setEnabled(enough)
        self.distribute_hint.setVisible(not enough)

    def align(self, mode: str) -> None:
        try:
            self.controller.align(mode)
        except EditorError as exc:
            self.controller.message.emit(str(exc))

    def distribute(self, axis: str) -> None:
        try:
            self.controller.distribute(axis)
        except EditorError as exc:
            self.controller.message.emit(str(exc))
