"""바코드와 QR을 골랐을 때: 담을 내용, 종류, 글자 표시, 복원력. 잘못된 값은 그 자리에서 알려 준다."""

from __future__ import annotations

from PySide6.QtWidgets import QCheckBox, QLineEdit, QVBoxLayout, QWidget

from design import barcodes, renderer
from design.binding import ColumnSource
from ui.widgets import ComboBox, SegmentedControl, label, set_prop

from ..common import group
from .base import InspectorContext, Section, set_combo_data, set_text_if_changed
from .detail import ColumnCombo

SYMBOLOGY_OPTIONS = (
    ("code128", "Code 128 (가장 많이 씀)", "영문, 숫자, 기호를 모두 담습니다. 사번이나 회원번호에 알맞습니다."),
    ("code39", "Code 39", "숫자와 영문 대문자만 담습니다. 오래된 리더기와 잘 맞습니다."),
    ("ean13", "EAN-13 (상품 바코드)", "숫자 12~13자리만 담습니다. 가게에서 쓰는 상품 바코드입니다."),
)
QR_LEVEL_OPTIONS = (
    ("L", "가볍게", "가장 큼직하게 찍힙니다. 카드가 긁히면 읽기 어려울 수 있습니다."),
    ("M", "보통", "대부분의 카드에 알맞습니다."),
    ("Q", "튼튼하게", "조금 긁히거나 더러워져도 읽힙니다. 무늬가 더 촘촘해집니다."),
    ("H", "가장 튼튼하게", "많이 가려져도 읽힙니다. 무늬가 가장 촘촘하니 크게 넣으세요."),
)


class CodeSection(Section):
    def __init__(self, context: InspectorContext, parent: QWidget | None = None) -> None:
        super().__init__(context, parent)
        self.source_control = SegmentedControl([("fixed", "직접 입력"), ("column", "명단 항목")], stretch=True)
        self.source_control.currentChanged.connect(self.set_mode)
        self.column_combo = ColumnCombo()
        self.column_combo.setAccessibleName("담을 명단 항목")
        self.column_combo.activated.connect(lambda _i: self.set_column(self.column_combo.column()))
        self.data_label = label("", "label")
        self.data_edit = QLineEdit()
        self.data_edit.textEdited.connect(self._on_data)
        self.data_label.setBuddy(self.data_edit)
        self.validation = label("", "hint", wrap=True)
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        for widget in (self.column_combo, self.data_label, self.data_edit, self.validation):
            layout.addWidget(widget)
        self.column.addWidget(group("담을 내용", self.source_control, box))

        self.symbology_combo = ComboBox()
        for key, text, _hint in SYMBOLOGY_OPTIONS:
            self.symbology_combo.addItem(text, key)
        self.symbology_combo.activated.connect(self._on_symbology)
        self.symbology_hint = label("", "hint", wrap=True)
        self.symbology_box = group("바코드 종류", self.symbology_combo, self.symbology_hint)
        self.column.addWidget(self.symbology_box)
        self.show_text_check = QCheckBox("바코드 아래에 글자도 찍기")
        self.show_text_check.toggled.connect(lambda on: self.edit({"show_text": on}))
        self.column.addWidget(self.show_text_check)

        self.level_combo = ComboBox()
        for key, text, _hint in QR_LEVEL_OPTIONS:
            self.level_combo.addItem(text, key)
        self.level_combo.activated.connect(self._on_level)
        self.level_hint = label("", "hint", wrap=True)
        self.level_box = group("긁힘에 견디는 정도", self.level_combo, self.level_hint)
        self.column.addWidget(self.level_box)

        self.controller.preview_changed.connect(self._refresh_validation)

    def load(self) -> None:
        item = self.item
        if item is None:
            return
        is_barcode = item.kind == "barcode"
        bound = item.source is not None and bool(item.source.columns())
        self.source_control.set_current("column" if bound else "fixed", emit=False)
        self.column_combo.setVisible(bound)
        if bound:
            column = next(iter(sorted(item.source.columns())), "")
            names = [name for name, kind in self.context.columns() if kind != "photo"]
            self.column_combo.set_columns(names, column, self.context.has_list())
            self.data_label.setText("예시 값 (명단을 보지 않을 때 보여 줄 값)")
        else:
            self.data_label.setText("바코드에 담을 글자" if is_barcode else "QR에 담을 글자나 주소")
        self.data_edit.setAccessibleName(self.data_label.text())
        if not self.data_edit.hasFocus():
            set_text_if_changed(self.data_edit, item.data)
        self.symbology_box.setVisible(is_barcode)
        self.show_text_check.setVisible(is_barcode)
        self.level_box.setVisible(not is_barcode)
        if is_barcode:
            set_combo_data(self.symbology_combo, item.symbology)
            self.symbology_hint.setText(self._hint(SYMBOLOGY_OPTIONS, item.symbology))
            self.show_text_check.setChecked(bool(item.show_text))
        else:
            set_combo_data(self.level_combo, str(item.error_correction).upper())
            self.level_hint.setText(self._hint(QR_LEVEL_OPTIONS, str(item.error_correction).upper()))
        self._refresh_validation()

    @staticmethod
    def _hint(options, key: str) -> str:
        return next((hint for value, _text, hint in options if value == key), "")

    def validation_message(self) -> str:
        """지금 값으로 코드를 만들 수 없으면 그 이유. 문제없으면 빈 글자."""
        item = self.item
        if item is None:
            return ""
        value = renderer.value_for(self.controller.document, item, self.controller.preview_record, None)
        data = "" if value is None else str(value)
        if data.startswith("«") and self.controller.preview_record is None:
            return ""
        try:
            if item.kind == "barcode":
                barcodes.normalize(item.symbology, data)
            else:
                barcodes.qr_matrix(data, item.error_correction)
        except barcodes.BarcodeError as exc:
            return str(exc)
        return ""

    def _refresh_validation(self) -> None:
        if not self.items or self.item.kind not in ("barcode", "qr"):
            return
        message = self.validation_message()
        self.validation.setText(message)
        self.validation.setVisible(bool(message))
        set_prop(self.validation, "tone", "error" if message else "")
        set_prop(self.data_edit, "invalid", "true" if message else "false")

    # ---- 고치기 ----
    def set_mode(self, mode: str) -> None:
        item = self.item
        if self._loading or item is None:
            return
        bound = item.source is not None and bool(item.source.columns())
        if mode == "column" and not bound:
            names = [name for name, kind in self.context.columns() if kind != "photo"]
            self.set_source(ColumnSource(column=names[0] if names else "번호"))
        elif mode == "fixed" and bound:
            self.set_source(None)

    def set_column(self, column: str) -> None:
        if column:
            self.set_source(ColumnSource(column=column))

    def _on_data(self, text: str) -> None:
        self.edit({"data": text}, merge="data", text="코드 내용 수정")

    def _on_symbology(self, index: int) -> None:
        key = self.symbology_combo.itemData(index)
        self.symbology_hint.setText(self._hint(SYMBOLOGY_OPTIONS, key))
        self.edit({"symbology": key})

    def _on_level(self, index: int) -> None:
        key = self.level_combo.itemData(index)
        self.level_hint.setText(self._hint(QR_LEVEL_OPTIONS, key))
        self.edit({"error_correction": key})
