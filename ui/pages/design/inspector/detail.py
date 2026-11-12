"""자세한 설정: 위치, 크기, 회전, 인쇄 층, 잠금. 평소에는 접혀 있다."""

from __future__ import annotations

from PySide6.QtWidgets import QCheckBox, QVBoxLayout, QWidget

from ui.widgets import ComboBox, MmField, SegmentedControl, StepperField, label

from ..common import Collapsible, field_row, group
from .base import InspectorContext, Section

LAYER_OPTIONS = (("color", "컬러"), ("white", "화이트"), ("uv", "UV"))
LAYER_HINTS = {
    "color": "보통의 색 인쇄입니다.",
    "white": "흰색 잉크로 찍습니다. 화이트가 있는 리본에서만 나옵니다. 화면에는 반투명하게 보입니다.",
    "uv": "빛을 비추면 보이는 UV 잉크로 찍습니다. UV가 있는 리본에서만 나옵니다.",
}


class ColumnCombo(ComboBox):
    """명단 항목 고르기. 지금 묶인 항목이 명단에 없어도 그대로 보여 준다."""

    def set_columns(self, columns: list[str], current: str | None, has_list: bool) -> None:
        self.blockSignals(True)
        self.clear()
        for name in columns:
            self.addItem(name, name)
        if current and current not in columns:
            self.addItem(f"{current} (명단에 없음)" if has_list else current, current)
        self.setCurrentIndex(self.findData(current) if current else -1)
        self.blockSignals(False)

    def column(self) -> str:
        return self.currentData() or ""


class DetailSection(Section):
    def __init__(self, context: InspectorContext, parent: QWidget | None = None) -> None:
        super().__init__(context, parent)
        self.box = Collapsible("자세한 설정 (위치, 회전, 인쇄 층)")
        self.column.addWidget(self.box)

        self.extras_host = QWidget()
        self.extras = QVBoxLayout(self.extras_host)
        self.extras.setContentsMargins(0, 0, 0, 0)
        self.extras.setSpacing(12)
        self.box.body.addWidget(self.extras_host)

        self.geometry_host = QWidget()
        geometry = QVBoxLayout(self.geometry_host)
        geometry.setContentsMargins(0, 0, 0, 0)
        geometry.setSpacing(8)
        geometry.addWidget(label("위치와 크기", "label"))
        self.x_field = MmField(0, -300, 400)
        self.y_field = MmField(0, -300, 400)
        self.w_field = MmField(10, 0.5, 400)
        self.h_field = MmField(10, 0.0, 400)
        self.rotation_field = StepperField(0, -360, 360, 1, 0, "°")
        self.fields = {"x": self.x_field, "y": self.y_field, "w": self.w_field, "h": self.h_field}
        for name, text in (("x", "왼쪽에서"), ("y", "위에서"), ("w", "너비"), ("h", "높이")):
            geometry.addWidget(field_row(text, self.fields[name]))
            self.fields[name].valueChanged.connect(lambda value, n=name: self.edit({n: float(value)}, merge=n))
        geometry.addWidget(field_row("회전", self.rotation_field))
        self.rotation_field.valueChanged.connect(
            lambda value: self.edit({"rotation": float(value) % 360.0}, merge="rotation"))
        self.box.body.addWidget(self.geometry_host)

        self.layer_control = SegmentedControl(LAYER_OPTIONS, stretch=True)
        self.layer_control.currentChanged.connect(self._on_layer)
        self.layer_hint = label("", "hint", wrap=True)
        self.box.body.addWidget(group("인쇄 층", self.layer_control, self.layer_hint))

        self.lock_check = QCheckBox("움직이지 않게 잠그기")
        self.lock_check.toggled.connect(lambda on: self._loading or self.controller.set_locked(self.ids, on))
        self.printable_check = QCheckBox("인쇄하지 않고 화면에서만 보기")
        self.printable_check.toggled.connect(
            lambda on: self._loading or self.controller.set_printable(self.ids, not on))
        checks = QVBoxLayout()
        checks.setContentsMargins(0, 0, 0, 0)
        checks.setSpacing(4)
        checks.addWidget(self.lock_check)
        checks.addWidget(self.printable_check)
        self.box.body.addLayout(checks)

    def set_extras(self, widgets: list[QWidget]) -> None:
        while self.extras.count():
            widget = self.extras.takeAt(0).widget()
            if widget is not None:
                widget.setParent(None)
        for widget in widgets:
            self.extras.addWidget(widget)
            widget.show()
        self.extras_host.setVisible(bool(widgets))

    def _on_layer(self, layer: str) -> None:
        if not self._loading:
            self.controller.set_layer(self.ids, layer)
        self.layer_hint.setText(LAYER_HINTS[layer])

    def load(self) -> None:
        item = self.item
        if item is None:
            return
        single = len(self.items) == 1
        self.geometry_host.setVisible(single)
        if single:
            for name, control in self.fields.items():
                control.set_value(getattr(item, name), emit=False)
            rotation = item.rotation if item.rotation <= 180 else item.rotation - 360
            self.rotation_field.set_value(rotation, emit=False)
            self.h_field.setEnabled(not (item.kind == "line" and item.h == 0))
        self.layer_control.set_current(item.layer, emit=False)
        self.layer_hint.setText(LAYER_HINTS.get(item.layer, ""))
        self.lock_check.setChecked(all(it.locked for it in self.items))
        self.printable_check.setChecked(all(not it.printable for it in self.items))
