"""디자인 화면이 쓰는 아이콘 이름과 얇은 도우미.

그림은 모두 공용 아이콘 모음(ui.widgets.icons)에 있다. 이 모듈은 예전 호출부가 그대로 돌도록
같은 이름(paint, pixmap, icon, KIND_ICONS)만 남겨 둔다.
"""

from __future__ import annotations

from PySide6.QtCore import QRectF
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap

from ui.widgets import icon_pixmap, paint_icon, themed_icon

KIND_ICONS = {
    "text": "text", "image": "image", "rect": "rect", "ellipse": "ellipse", "line": "line",
    "barcode": "barcode", "qr": "qr",
}


def paint(painter: QPainter, name: str, rect: QRectF, color: QColor | str | None = None) -> None:
    """아이콘을 rect 안에 그린다. color는 QColor 또는 테마 역할 이름."""
    paint_icon(painter, name, rect, color)


def pixmap(name: str, size: int = 18, color: QColor | str | None = None) -> QPixmap:
    return icon_pixmap(name, size, color)


def icon(name: str, role: str = "text", size: int = 18, checked_role: str | None = None) -> QIcon:
    return themed_icon(name, role, size, checked_role=checked_role)
