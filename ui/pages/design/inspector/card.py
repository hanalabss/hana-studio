"""아무것도 고르지 않았을 때: 카드 설정과 '확인할 것' 목록."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLineEdit, QVBoxLayout, QWidget

from design import presets
from ui.editor import EditorError, Problem
from ui.theme import SPACE
from ui.widgets import ComboBox, Divider, SecondaryButton, SegmentedControl, SwatchPicker, label

from ..common import NoteRow, clear_layout, group, hbox
from .base import SWATCHES, WHITE, InspectorContext, Section, set_combo_data, show_color

SIZE_LABELS = {
    "cr80": "표준 카드 (CR80) 85.6 × 54 mm",
    "cut_54x86": "54 × 86 mm",
    "cut_54x86_bleed": "54 × 86 mm + 재단 여백 2 mm",
}
BACKGROUND_COLORS = tuple(sorted(SWATCHES, key=lambda entry: entry[1] != WHITE))


def matching_preset(document) -> str | None:
    for preset in presets.PRESETS:
        if (abs(document.card_width_mm - preset.width_mm) < 0.01
                and abs(document.card_height_mm - preset.height_mm) < 0.01
                and abs(document.bleed_mm - preset.bleed_mm) < 0.01):
            return preset.key
    return None


class CardSection(Section):
    problemActivated = Signal(object)

    def __init__(self, context: InspectorContext, parent: QWidget | None = None) -> None:
        super().__init__(context, parent)
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("예: 2026 신입 사원증")
        self.name_edit.editingFinished.connect(self._on_name)
        self.column.addWidget(group("디자인 이름", self.name_edit))

        self.size_combo = ComboBox()
        self.size_combo.activated.connect(self._on_size)
        self.column.addWidget(group("카드 크기", self.size_combo))

        self.orientation_control = SegmentedControl([("landscape", "가로"), ("portrait", "세로")], stretch=True)
        self.orientation_control.currentChanged.connect(self._on_orientation)
        self.column.addWidget(group("방향", self.orientation_control))

        self.sides_control = SegmentedControl([("single", "한 면"), ("dual", "양면")], stretch=True)
        self.sides_control.currentChanged.connect(self._on_sides)
        self.column.addWidget(group("인쇄 면", self.sides_control))

        self.background_picker = SwatchPicker(BACKGROUND_COLORS)
        self.background_picker.colorChanged.connect(self._on_background_color)
        self.background_image_button = SecondaryButton("배경 그림 넣기", size="sm", icon="image")
        self.background_image_button.clicked.connect(self.choose_background_image)
        self.background_clear_button = SecondaryButton("바탕 비우기", size="sm")
        self.background_clear_button.setToolTip(
            "카드의 바탕 그림이나 바탕색을 빼고 흰 카드로 돌립니다. "
            "사진에서 사람만 남기려면 '사진'으로 넣은 뒤 '배경 지우기'를 쓰세요.")
        self.background_clear_button.clicked.connect(lambda: self.controller.set_side_background(None))
        self.background_label = label("바탕", "label")
        self.column.addWidget(group(
            None, self.background_label, self.background_picker,
            hbox(self.background_image_button, self.background_clear_button, None),
        ))
        self.background_to_photo_button = SecondaryButton("바탕 그림에서 사람만 남기기", size="sm")
        self.background_to_photo_button.setToolTip(
            "바탕 그림을 카드 전체 크기의 사진으로 바꿉니다. 그 사진을 고르면 '배경 지우기'를 쓸 수 있습니다.")
        self.background_to_photo_button.clicked.connect(self.background_to_photo)
        self.column.addWidget(self.background_to_photo_button)

        self.column.addWidget(Divider())
        self.problems_title = label("확인할 것", "strong")
        self.column.addWidget(self.problems_title)
        self.problems_host = QWidget()
        self.problems_box = QVBoxLayout(self.problems_host)
        self.problems_box.setContentsMargins(0, 0, 0, 0)
        self.problems_box.setSpacing(SPACE["sm"])
        self.column.addWidget(self.problems_host)
        self.problem_rows: list[NoteRow] = []
        self._problems: list[Problem] = []

        self.controller.problems_changed.connect(self.refresh_problems)
        self.controller.items_changed.connect(self._load_background)
        self.controller.card_changed.connect(self.reload)
        self.controller.document_changed.connect(self.reload)
        self.controller.side_changed.connect(lambda _key: self.reload())

    # ---- 값 채우기 ----
    def load(self) -> None:
        document = self.controller.document
        if not self.name_edit.hasFocus():
            self.name_edit.setText(document.name)
        current = matching_preset(document)
        self.size_combo.blockSignals(True)
        self.size_combo.clear()
        for preset in presets.PRESETS:
            self.size_combo.addItem(SIZE_LABELS.get(preset.key, preset.label), preset.key)
        if current is None:
            cut_w = document.card_width_mm - 2 * document.bleed_mm
            cut_h = document.card_height_mm - 2 * document.bleed_mm
            self.size_combo.addItem(f"직접 정한 크기 ({cut_h:g} × {cut_w:g} mm)", "custom")
        set_combo_data(self.size_combo, current or "custom")
        self.size_combo.blockSignals(False)
        self.orientation_control.set_current(self.controller.side().orientation, emit=False)
        self.sides_control.set_current("dual" if document.dual_side else "single", emit=False)
        self._load_background()
        self.refresh_problems()

    def _load_background(self) -> None:
        with self.loading():
            side_name = "뒷면" if self.controller.current_side == "back" else "앞면"
            self.background_label.setText(f"{side_name} 바탕" if self.controller.document.dual_side else "바탕")
            background = self.controller.side().background
            is_image = bool(background) and not background.startswith("#")
            if not is_image:
                show_color(self.background_picker, background or WHITE)
            self.background_clear_button.setVisible(bool(background))
            self.background_to_photo_button.setVisible(is_image)
            self.background_clear_button.setText("바탕 그림 빼기" if is_image else "바탕색 빼기")
            self.background_image_button.setText("배경 그림 바꾸기" if is_image else "배경 그림 넣기")

    def refresh_problems(self) -> None:
        self._problems = self.controller.problems()
        clear_layout(self.problems_box)
        self.problem_rows = []
        if not self._problems:
            self.problems_box.addWidget(NoteRow("ok", "걸리는 것이 없습니다. 이대로 인쇄할 수 있습니다."))
            return
        dual = self.controller.document.dual_side
        for problem in self._problems:
            prefix = ("뒷면 · " if problem.side == "back" else "앞면 · ") if dual else ""
            row = NoteRow("problem", prefix + problem.text)
            row.setToolTip("눌러서 카드에서 찾아보기" if problem.item_id else "")
            row.clicked.connect(lambda p=problem: self.problemActivated.emit(p))
            self.problems_box.addWidget(row)
            self.problem_rows.append(row)

    def problems(self) -> list[Problem]:
        return list(self._problems)

    # ---- 고치기 ----
    def _setup(self, **changes) -> None:
        if self._loading:
            return
        try:
            self.controller.set_card_setup(**changes)
        except EditorError as exc:
            self.controller.message.emit(str(exc))

    def _on_name(self) -> None:
        name = self.name_edit.text().strip()
        if name and name != self.controller.document.name:
            self._setup(name=name)

    def _on_size(self, index: int) -> None:
        key = self.size_combo.itemData(index)
        if key and key != "custom" and key != matching_preset(self.controller.document):
            self._setup(preset=key)

    def _on_orientation(self, orientation: str) -> None:
        self._setup(orientation=orientation)

    def _on_sides(self, key: str) -> None:
        self._setup(dual_side=key == "dual")
        if key == "single" and self.controller.current_side == "back":
            self.controller.set_side("front")

    def _on_background_color(self, color: str) -> None:
        if not self._loading:
            self.controller.set_side_background(None if color.upper() == WHITE else color)

    def choose_background_image(self) -> None:
        path = self.context.pick_image("배경으로 쓸 그림 고르기")
        if path:
            self.set_background_image(path)

    def background_to_photo(self):
        """바탕 그림을 카드 전체 크기의 사진 항목으로 바꾸고 고른다. 실행취소 한 번으로 되돌아간다."""
        import os

        controller = self.controller
        background = controller.side().background
        if not background or background.startswith("#"):
            return None
        base = controller.document.base_dir or ""
        path = background if os.path.isabs(background) else os.path.join(base, background)
        width, height = controller.side_size_mm()
        stack = controller.undo_stack
        stack.beginMacro("바탕 그림을 사진으로 바꾸기")
        try:
            item = controller.add_image_from_file(path)
            controller.set_geometry(item.id, {"x": 0.0, "y": 0.0, "w": width, "h": height, "rotation": 0.0})
            controller.set_properties([item.id], {"fit": "fill", "name": "바탕 사진"})
            controller.send_to_back([item.id])
            controller.set_side_background(None)
        except EditorError as exc:
            stack.endMacro()
            controller.message.emit(str(exc))
            return None
        stack.endMacro()
        controller.select([item.id])
        controller.message.emit("바탕 그림을 사진으로 바꿨습니다. 오른쪽의 '배경 지우기'를 누르면 사람만 남습니다.")
        return item

    def set_background_image(self, path: str) -> None:
        try:
            self.controller.set_side_background_image(path)
        except EditorError as exc:
            self.controller.message.emit(str(exc))
