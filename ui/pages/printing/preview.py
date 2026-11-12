"""카드 한 장 미리 보기: 인쇄할 때와 같은 길(RenderJob)로 그린 그림을 보여 준다."""
from __future__ import annotations

import logging
from collections import OrderedDict

from PySide6.QtCore import QRectF, QSize, Qt
from PySide6.QtGui import QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QSizePolicy, QWidget

from data.store import DataStore
from design.model import Document
from services.render_job import PrintOptions, RenderJob
from ui.theme import FONT_SIZES
from ui.widgets import theme_color

log = logging.getLogger(__name__)

PREVIEW_DPI = 150.0
_CORNER_RATIO = 3.18 / 85.6     # CR80 모서리 둥글기


def render_card_image(document: Document, store: DataStore | None, record_id: int | None,
                      side_key: str = "front", dpi: float = PREVIEW_DPI, position: int = 0) -> QImage | None:
    """record_id 사람의 카드 한 면. 그릴 수 없으면 None."""
    job = None
    try:
        options = PrintOptions(dpi=dpi, counter_offset=position)
        if record_id is not None and store is not None:
            job = RenderJob.for_records(document, store, [record_id], options)
        else:
            job = RenderJob.for_copies(document, 1, options)
        if side_key == "back":
            job.document.dual_side = True
        card = job.render(0)
        side = card.back if side_key == "back" else card.front
        if side is None:
            return None
        image = QImage(side.color_path)
        return None if image.isNull() else image.copy()
    except Exception as exc:
        log.info("미리 보기를 그리지 못했습니다: %s", exc)
        return None
    finally:
        if job is not None:
            job.close()


class PreviewCache:
    def __init__(self, capacity: int = 24) -> None:
        self._items: OrderedDict = OrderedDict()
        self._capacity = capacity

    def get(self, key):
        if key in self._items:
            self._items.move_to_end(key)
            return self._items[key]
        return None

    def put(self, key, value) -> None:
        self._items[key] = value
        self._items.move_to_end(key)
        while len(self._items) > self._capacity:
            self._items.popitem(last=False)

    def clear(self) -> None:
        self._items.clear()


class CardPreview(QWidget):
    """바탕 위에 카드 한 장. 그림이 없으면 점선 카드와 안내 글자를 보여 준다."""

    def __init__(self, parent: QWidget | None = None, minimum: QSize = QSize(240, 160),
                 margin: float = 12.0) -> None:
        super().__init__(parent)
        self._margin = float(margin)
        self._pixmap: QPixmap | None = None
        self._placeholder = "미리 볼 카드가 없습니다"
        self._aspect = 85.6 / 53.98
        self.setMinimumSize(minimum)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAccessibleName("카드 미리 보기")

    def set_image(self, image: QImage | None) -> None:
        if image is None or image.isNull():
            self._pixmap = None
        else:
            self._pixmap = QPixmap.fromImage(image)
            self._aspect = image.width() / max(1, image.height())
        self.update()

    def set_aspect(self, width_mm: float, height_mm: float) -> None:
        if width_mm > 0 and height_mm > 0:
            self._aspect = width_mm / height_mm
            self.update()

    def aspect(self) -> float:
        return self._aspect

    def set_placeholder(self, text: str) -> None:
        self._placeholder = text
        self.update()

    def has_image(self) -> bool:
        return self._pixmap is not None

    def card_rect(self) -> QRectF:
        margin = self._margin
        box_w, box_h = max(1.0, self.width() - 2 * margin), max(1.0, self.height() - 2 * margin)
        width = min(box_w, box_h * self._aspect)
        height = width / self._aspect
        return QRectF((self.width() - width) / 2, (self.height() - height) / 2, width, height)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        rect = self.card_rect()
        radius = max(4.0, max(rect.width(), rect.height()) * _CORNER_RATIO)
        path = QPainterPath()
        path.addRoundedRect(rect, radius, radius)
        if self._pixmap is not None:
            painter.save()
            painter.setClipPath(path)
            painter.drawPixmap(rect, self._pixmap, QRectF(self._pixmap.rect()))
            painter.restore()
            painter.setPen(QPen(theme_color("control_line"), 1))
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(path)
        else:
            painter.setBrush(theme_color("surface"))
            pen = QPen(theme_color("indicator_line"), 1.4, Qt.CustomDashLine)
            pen.setDashPattern([4, 3.5])
            painter.setPen(pen)
            painter.drawPath(path)
            font = painter.font()
            font.setPixelSize(FONT_SIZES["small"])
            painter.setFont(font)
            painter.setPen(theme_color("text_muted"))
            painter.drawText(rect.adjusted(12, 12, -12, -12), Qt.AlignCenter | Qt.TextWordWrap, self._placeholder)
        painter.end()
