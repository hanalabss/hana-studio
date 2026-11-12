"""문서의 한 면을 보여주는 QGraphicsScene. 좌표 단위는 mm."""

from __future__ import annotations

from typing import Any, Mapping

from PySide6.QtCore import QRectF
from PySide6.QtWidgets import QGraphicsScene

from design import renderer
from design.binding import BindingContext
from design.model import Document, Item
from ui.editor import geometry
from ui.editor.items import CanvasItem, CardBackgroundItem

_MARGIN_RATIO = 0.75
CARD_RADIUS_MM = 3.0


class EditorScene(QGraphicsScene):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.document: Document | None = None
        self.side_key = "front"
        self.record: Mapping[str, Any] | None = None
        self.context = BindingContext()
        self.editing_id: str | None = None
        self.rounded_card = False   # 캔버스 위젯이 켠다. 꺼져 있으면 인쇄 그림과 똑같은 사각형.
        self._items: dict[str, CanvasItem] = {}
        self._card = CardBackgroundItem()
        self.addItem(self._card)
        self.setItemIndexMethod(QGraphicsScene.ItemIndexMethod.NoIndex)

    # ------------------------------------------------------------ 문서 연결

    def set_document(self, document: Document, side_key: str = "front") -> None:
        self.document = document
        self.side_key = side_key
        self.rebuild()

    def set_side(self, side_key: str) -> None:
        self.side_key = side_key
        self.rebuild()

    def side(self):
        return self.document.side(self.side_key)

    def card_size(self) -> tuple[float, float]:
        if self.document is None:
            return (85.6, 53.98)
        return self.document.side_size_mm(self.side_key)

    def card_rect(self) -> QRectF:
        width, height = self.card_size()
        return QRectF(0.0, 0.0, width, height)

    def cut_rect(self) -> QRectF:
        left, top, right, bottom = self.document.side_bounds(self.side_key, inside_bleed=True)
        return QRectF(left, top, right - left, bottom - top)

    def card_corner_radius(self) -> float:
        """도련이 있으면 바깥 테두리는 살짝만 둥글리고, 칼선을 카드 모서리 반경으로 그린다."""
        bleed = self.document.bleed_mm if self.document is not None else 0.0
        return min(1.0, bleed) if bleed > 0 else CARD_RADIUS_MM

    def set_rounded_card(self, rounded: bool) -> None:
        self.rounded_card = bool(rounded)
        if self.document is not None:
            self._sync_card()

    def rebuild(self) -> None:
        for canvas_item in self._items.values():
            self.removeItem(canvas_item)
        self._items.clear()
        self.editing_id = None
        if self.document is None:
            return
        self._sync_card()
        self.sync_structure()

    def _sync_card(self) -> None:
        width, height = self.card_size()
        side = self.side()
        path = None
        if side.background and not side.background.startswith("#"):
            path = renderer.asset_path(self.document, side.background)
        self._card.configure(
            width, height, side.background, path, self.card_corner_radius() if self.rounded_card else 0.0
        )
        margin = max(width, height) * _MARGIN_RATIO
        self.setSceneRect(QRectF(-margin, -margin, width + 2 * margin, height + 2 * margin))

    # ------------------------------------------------------------ 모델 → 화면

    def sync_structure(self) -> None:
        """항목 추가·삭제·순서·배경 변화를 반영한다."""
        if self.document is None:
            return
        self._sync_card()
        models = {item.id: item for item in self.side().items}
        for item_id in [i for i in self._items if i not in models or self._items[i].model is not models[i]]:
            self.removeItem(self._items.pop(item_id))
        for item_id, model in models.items():
            canvas_item = self._items.get(item_id)
            if canvas_item is None:
                canvas_item = CanvasItem(model)
                self._items[item_id] = canvas_item
                self.addItem(canvas_item)
            else:
                canvas_item.sync()

    def sync_items(self, ids=None, attrs: set[str] | None = None) -> None:
        targets = self._items.values() if ids is None else (self._items.get(i) for i in ids)
        for canvas_item in list(targets):
            if canvas_item is not None:
                canvas_item.sync(attrs)

    def canvas_item(self, item_id: str) -> CanvasItem | None:
        return self._items.get(item_id)

    def set_editing(self, item_id: str | None) -> None:
        previous, self.editing_id = self.editing_id, item_id
        for changed in (previous, item_id):
            canvas_item = self._items.get(changed) if changed else None
            if canvas_item is not None:
                canvas_item.update()

    # ------------------------------------------------------------ 값

    def set_preview(self, record: Mapping[str, Any] | None, context: BindingContext | None) -> None:
        self.record = record
        self.context = context or BindingContext()
        self.sync_items()

    def value_of(self, item: Item):
        value = renderer.value_for(self.document, item, self.record, self.context)
        crop = item.crop_for(self.context.record_key) if item.kind == "image" else None
        return value, crop

    # ------------------------------------------------------------ 찾기

    def item_at(self, x: float, y: float, tolerance: float = 0.0, *, include_locked: bool = False) -> Item | None:
        """가장 위에 있는 항목. 잠긴 항목과 숨긴 항목은 건너뛴다."""
        if self.document is None:
            return None
        for item in reversed(self.side().sorted_items()):
            if not item.visible or (item.locked and not include_locked):
                continue
            if geometry.hit_item(item, x, y, tolerance):
                return item
        return None

    def items_in_bounds(self, left: float, top: float, right: float, bottom: float) -> list[Item]:
        found = []
        for item in self.side().sorted_items():
            if not item.visible or item.locked:
                continue
            b_left, b_top, b_right, b_bottom = item.bounds()
            if b_right >= left and b_left <= right and b_bottom >= top and b_top <= bottom:
                found.append(item)
        return found
