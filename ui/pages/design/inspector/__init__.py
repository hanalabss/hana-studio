"""오른쪽 속성 패널. 고른 것에 따라 구역이 바뀐다."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QScrollArea, QVBoxLayout, QWidget

from ui.editor import KIND_LABELS, display_name
from ui.theme import SPACE
from ui.widgets import IconButton, label

from ..common import NoteRow, clear_layout
from ..style import INSPECTOR_WIDTH
from .base import InspectorContext, Section
from .card import CardSection
from .code import CodeSection
from .detail import DetailSection
from .image import ImageSection
from .multi import MultiSection
from .shape import LineSection, ShapeSection
from .text import TextSection

__all__ = ["Inspector", "InspectorContext"]

_SECTION_FOR_KIND = {
    "text": "text", "image": "image", "rect": "shape", "ellipse": "shape", "line": "line",
    "barcode": "code", "qr": "code",
}


class Inspector(QFrame):
    problemActivated = Signal(object)

    def __init__(self, context: InspectorContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("DesignInspector")
        self.setFixedWidth(INSPECTOR_WIDTH)
        self.context = context
        self.controller = context.controller
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(SPACE["lg"], SPACE["md"], SPACE["sm"], SPACE["sm"])
        header_layout.setSpacing(SPACE["sm"])
        titles = QVBoxLayout()
        titles.setContentsMargins(0, 0, 0, 0)
        titles.setSpacing(2)
        self.title_label = label("카드 설정", "title")
        self.subtitle_label = label("", "muted")
        titles.addWidget(self.title_label)
        titles.addWidget(self.subtitle_label)
        header_layout.addLayout(titles, 1)
        self.more_button = IconButton("plus", "하나 더 만들기, 앞뒤 순서 바꾸기", size="sm")
        self.more_button.set_icon_name(None)
        self.delete_button = IconButton("trash", "지우기 (Delete)", size="sm")
        self.delete_button.clicked.connect(lambda: self.controller.delete())
        header_layout.addWidget(self.more_button, 0, Qt.AlignTop)
        header_layout.addWidget(self.delete_button, 0, Qt.AlignTop)
        outer.addWidget(header)

        self.scroll = QScrollArea(self)
        self.scroll.setObjectName("InspectorScroll")
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        outer.addWidget(self.scroll, 1)
        body = QWidget()
        body.setObjectName("InspectorBody")
        self.scroll.setWidget(body)
        column = QVBoxLayout(body)
        body.setFixedWidth(INSPECTOR_WIDTH - 12)
        column.setContentsMargins(SPACE["lg"], SPACE["sm"], SPACE["sm"], SPACE["lg"])
        column.setSpacing(SPACE["md"] + 2)

        self.warnings_host = QWidget()
        self.warnings_box = QVBoxLayout(self.warnings_host)
        self.warnings_box.setContentsMargins(0, 0, 0, 0)
        self.warnings_box.setSpacing(SPACE["sm"])
        column.addWidget(self.warnings_host)

        self.card = CardSection(context)
        self.text = TextSection(context)
        self.image = ImageSection(context)
        self.shape = ShapeSection(context)
        self.line = LineSection(context)
        self.code = CodeSection(context)
        self.multi = MultiSection(context)
        self.detail = DetailSection(context)
        self.sections: dict[str, Section] = {
            "card": self.card, "text": self.text, "image": self.image, "shape": self.shape,
            "line": self.line, "code": self.code, "multi": self.multi,
        }
        for section in self.sections.values():
            column.addWidget(section)
            section.setVisible(False)
        column.addWidget(self.detail)
        column.addStretch(1)
        self.card.problemActivated.connect(self.problemActivated)

        self._current = "card"
        self.controller.selection_changed.connect(lambda _ids: self.refresh())
        self.controller.properties_changed.connect(self._on_properties_changed)
        self.controller.items_changed.connect(self._refresh_header)
        self.controller.problems_changed.connect(self._refresh_warnings)
        self.controller.preview_changed.connect(self.reload_values)
        self.controller.document_changed.connect(self.refresh)
        context.workspace.columns_changed.connect(self.reload_values)
        self.refresh()

    def current_section(self) -> str:
        return self._current

    def active(self) -> Section:
        return self.sections[self._current]

    def refresh(self) -> None:
        """선택이 바뀌었을 때: 보여 줄 구역을 고르고 값을 채운다."""
        self.context.session.close()
        items = self.controller.selected_items()
        if not items:
            name = "card"
        elif len(items) > 1:
            name = "multi"
        else:
            name = _SECTION_FOR_KIND.get(items[0].kind, "card")
        self._current = name
        for key, section in self.sections.items():
            section.setVisible(key == name)
        section = self.sections[name]
        section.set_items(items)
        has_selection = bool(items)
        self.detail.setVisible(has_selection)
        self.delete_button.setVisible(has_selection)
        self.more_button.setVisible(has_selection)
        if has_selection:
            self.detail.set_extras(section.detail_widgets() if len(items) == 1 else [])
            self.detail.set_items(items)
        self._refresh_header()
        self._refresh_warnings()

    def reload_values(self, *_args) -> None:
        self.active().reload()
        if self.controller.selection:
            self.detail.reload()

    def _on_properties_changed(self, ids: list[str]) -> None:
        if set(ids) & set(self.controller.selection):
            items = self.controller.selected_items()
            self.active().items = items
            self.detail.items = items
            self.reload_values()
            self._refresh_header()

    def _refresh_header(self) -> None:
        items = self.controller.selected_items()
        if not items:
            self.title_label.setText("카드 설정")
            width, height = self.controller.side_size_mm()
            self.subtitle_label.setText(f"{width:g} × {height:g} mm")
        elif len(items) > 1:
            self.title_label.setText(f"{len(items)}개를 골랐습니다")
            self.subtitle_label.setText("함께 바꿀 수 있는 것만 보입니다")
        else:
            item = items[0]
            kind = KIND_LABELS.get(item.kind, "항목")
            name = display_name(item)
            bound = item.source is not None and bool(item.source.columns())
            self.title_label.setText(kind)
            parts = [name] if name != kind else []
            if bound:
                parts.append("사람마다 바뀌는 칸")
            self.subtitle_label.setText(", ".join(parts))
        self.subtitle_label.setVisible(bool(self.subtitle_label.text()))

    def _refresh_warnings(self) -> None:
        clear_layout(self.warnings_box)
        messages: list[str] = []
        ids = self.controller.selection
        if len(ids) == 1:
            messages = self.controller.item_warnings(ids[0])
        for message in messages:
            self.warnings_box.addWidget(NoteRow("note", message))
        self.warnings_host.setVisible(bool(messages))
