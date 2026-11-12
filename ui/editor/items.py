"""모델 항목을 감싸는 QGraphicsItem. 그리기는 design.renderer에 맡겨 화면과 인쇄가 같다."""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath
from PySide6.QtWidgets import QGraphicsItem

from design import renderer

# 인쇄(300dpi)와 같은 배율로 그리고 painter를 줄여서 픽셀 단위 처리까지 일치시킨다.
RENDER_SCALE = renderer.mm_to_px_for(300.0)
_GEOMETRY_ATTRS = frozenset({"x", "y", "rotation", "z"})
_CACHED_KINDS = frozenset({"image", "barcode", "qr"})
_MASK_LAYER_OPACITY = 0.5


class CanvasItem(QGraphicsItem):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model
        self._rect = QRectF()
        if model.kind in _CACHED_KINDS:
            self.setCacheMode(QGraphicsItem.CacheMode.DeviceCoordinateCache)
        self.sync()

    def sync(self, attrs: set[str] | None = None) -> None:
        model = self.model
        pad = 0.0
        if model.kind == "line":
            pad = max(0.2, getattr(model, "width_mm", 0.0))
        rect = QRectF(-pad, -pad, model.w + 2 * pad, model.h + 2 * pad)
        if rect != self._rect:
            self.prepareGeometryChange()
            self._rect = rect
            self.setTransformOriginPoint(model.w / 2.0, model.h / 2.0)
        self.setPos(model.x, model.y)
        self.setRotation(model.rotation)
        self.setZValue(model.z)
        self.setVisible(bool(model.visible))
        self.setOpacity(1.0 if model.layer == "color" else _MASK_LAYER_OPACITY)
        if attrs is None or not attrs <= _GEOMETRY_ATTRS:
            self.update()

    def boundingRect(self) -> QRectF:
        return self._rect

    def paint(self, painter: QPainter, option, widget=None) -> None:
        scene = self.scene()
        if scene is None or scene.editing_id == self.model.id:
            return
        value, crop = scene.value_of(self.model)
        painter.scale(1.0 / RENDER_SCALE, 1.0 / RENDER_SCALE)
        renderer.paint_item_local(painter, self.model, value, RENDER_SCALE, crop=crop, design_mode=True)


class CardBackgroundItem(QGraphicsItem):
    """카드 바탕(흰색)과 면 배경."""

    def __init__(self) -> None:
        super().__init__()
        self._rect = QRectF()
        self.fill = QColor("#FFFFFF")
        self.background: str | None = None
        self.background_path: str | None = None
        self.radius = 0.0
        self.setZValue(-1.0e9)
        self.setCacheMode(QGraphicsItem.CacheMode.DeviceCoordinateCache)

    def configure(
        self, width: float, height: float, background: str | None, background_path: str | None,
        radius: float = 0.0,
    ) -> None:
        rect = QRectF(0.0, 0.0, width, height)
        if rect != self._rect:
            self.prepareGeometryChange()
            self._rect = rect
        self.background = background
        self.background_path = background_path
        self.radius = radius
        self.update()

    def boundingRect(self) -> QRectF:
        return self._rect

    def paint(self, painter: QPainter, option, widget=None) -> None:
        if self.radius > 0:
            path = QPainterPath()
            path.addRoundedRect(self._rect, self.radius, self.radius)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setClipPath(path)
        painter.fillRect(self._rect, self.fill)
        if not self.background:
            return
        if self.background.startswith("#"):
            color = QColor(self.background)
            if color.isValid():
                painter.fillRect(self._rect, color)
            return
        image = renderer.load_image(self.background_path) if self.background_path else None
        if image is not None:
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            painter.drawImage(self._rect, image)
