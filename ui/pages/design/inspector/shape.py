"""도형(사각형, 원)과 선을 골랐을 때: 채우기, 테두리, 굵기, 모서리."""

from __future__ import annotations

from PySide6.QtWidgets import QCheckBox, QWidget

from ui.widgets import MmField, SegmentedControl, SwatchPicker

from ..common import field_row, group
from .base import BLACK, SWATCHES, InspectorContext, Section, show_color

LINE_STYLES = (("solid", "실선"), ("dash", "긴 점선"), ("dot", "점선"))


class ShapeSection(Section):
    """사각형과 원."""

    def __init__(self, context: InspectorContext, parent: QWidget | None = None) -> None:
        super().__init__(context, parent)
        self._last_fill: str | None = None
        self._last_stroke = BLACK

        self.fill_check = QCheckBox("색 채우기")
        self.fill_check.toggled.connect(self._on_fill_toggled)
        self.fill_picker = SwatchPicker(SWATCHES)
        self.fill_picker.colorChanged.connect(lambda color: self.edit({"fill": color}, merge="fill"))
        self.column.addWidget(group(None, self.fill_check, self.fill_picker))

        self.stroke_check = QCheckBox("테두리")
        self.stroke_check.toggled.connect(self._on_stroke_toggled)
        self.stroke_picker = SwatchPicker(SWATCHES)
        self.stroke_picker.colorChanged.connect(lambda color: self.edit({"stroke_color": color}, merge="stroke"))
        self.stroke_width = MmField(0.3, 0.1, 10.0, 0.1)
        self.stroke_width.valueChanged.connect(
            lambda value: self.edit({"stroke_width_mm": float(value)}, merge="stroke_width"))
        self.stroke_width_row = field_row("굵기", self.stroke_width)
        self.column.addWidget(group(None, self.stroke_check, self.stroke_picker, self.stroke_width_row))

        self.radius_field = MmField(0.0, 0.0, 50.0, 0.5)
        self.radius_field.valueChanged.connect(
            lambda value: self.edit({"corner_radius_mm": float(value)}, merge="radius"))
        self.radius_row = field_row("모서리", self.radius_field)
        self.radius_row.setToolTip("모서리를 둥글게 깎는 정도입니다. 0이면 각진 모서리입니다.")
        self.column.addWidget(self.radius_row)

    def load(self) -> None:
        item = self.item
        if item is None:
            return
        has_fill = bool(item.fill)
        self.fill_check.setChecked(has_fill)
        self.fill_picker.setVisible(has_fill)
        if has_fill:
            self._last_fill = item.fill
            show_color(self.fill_picker, item.fill)
        has_stroke = bool(item.stroke_color) and item.stroke_width_mm > 0
        self.stroke_check.setChecked(has_stroke)
        self.stroke_picker.setVisible(has_stroke)
        self.stroke_width_row.setVisible(has_stroke)
        if has_stroke:
            self._last_stroke = item.stroke_color
            show_color(self.stroke_picker, item.stroke_color)
            self.stroke_width.set_value(item.stroke_width_mm, emit=False)
        is_rect = all(it.kind == "rect" for it in self.items)
        self.radius_row.setVisible(is_rect)
        if is_rect:
            self.radius_field.set_value(item.corner_radius_mm, emit=False)

    def _on_fill_toggled(self, on: bool) -> None:
        if self._loading:
            return
        self.edit({"fill": (self._last_fill or self._fallback_fill()) if on else None})

    def _fallback_fill(self) -> str:
        return self.fill_picker.current() or self.fill_picker.colors()[2]

    def _on_stroke_toggled(self, on: bool) -> None:
        if self._loading or self.item is None:
            return
        if on:
            width = self.item.stroke_width_mm if self.item.stroke_width_mm > 0 else 0.3
            self.edit({"stroke_color": self._last_stroke, "stroke_width_mm": width})
        else:
            self.edit({"stroke_color": None})


class LineSection(Section):
    def __init__(self, context: InspectorContext, parent: QWidget | None = None) -> None:
        super().__init__(context, parent)
        self.color_picker = SwatchPicker(SWATCHES)
        self.color_picker.colorChanged.connect(lambda color: self.edit({"color": color}, merge="color"))
        self.column.addWidget(group("선 색", self.color_picker))
        self.width_field = MmField(0.3, 0.1, 10.0, 0.1)
        self.width_field.valueChanged.connect(lambda value: self.edit({"width_mm": float(value)}, merge="width"))
        self.column.addWidget(field_row("굵기", self.width_field))
        self.style_control = SegmentedControl(LINE_STYLES, stretch=True)
        self.style_control.currentChanged.connect(lambda key: self.edit({"style": key}))
        self.column.addWidget(group("선 모양", self.style_control))

    def load(self) -> None:
        item = self.item
        if item is None:
            return
        show_color(self.color_picker, item.color)
        self.width_field.set_value(item.width_mm, emit=False)
        self.style_control.set_current(item.style if item.style in dict(LINE_STYLES) else "solid", emit=False)
