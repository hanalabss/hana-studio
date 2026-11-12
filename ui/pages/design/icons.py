"""디자인 화면에서만 쓰는 선 아이콘. ui.widgets.icons와 같은 24 격자, 같은 굵기로 그린다."""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QGuiApplication, QIcon, QPainter, QPainterPath, QPen, QPixmap, QPolygonF

from ui.widgets import ICON_NAMES, paint_icon, theme_color

_STROKE = 1.75


def _line(p: QPainter, *pts: float) -> None:
    p.drawPolyline(QPolygonF([QPointF(pts[i], pts[i + 1]) for i in range(0, len(pts), 2)]))


def _fill_rect(p: QPainter, x: float, y: float, w: float, h: float) -> None:
    p.save()
    p.setBrush(p.pen().color())
    p.setPen(Qt.NoPen)
    p.drawRect(QRectF(x, y, w, h))
    p.restore()


def _shape(p):
    p.drawRoundedRect(QRectF(3.5, 3.5, 11, 11), 1.5, 1.5)
    p.drawEllipse(QPointF(14.5, 14.5), 6, 6)


def _rect(p):
    p.drawRoundedRect(QRectF(4, 6, 16, 12), 1.5, 1.5)


def _ellipse(p):
    p.drawEllipse(QPointF(12, 12), 8, 8)


def _diagonal(p):
    _line(p, 5, 19, 19, 5)


def _barcode(p):
    for x, w in ((4, 1.6), (7.4, 0.8), (9.8, 2.2), (13.6, 0.8), (15.8, 1.6), (18.8, 1.2)):
        _fill_rect(p, x, 5, w, 14)


def _qr(p):
    for x, y in ((4, 4), (14, 4), (4, 14)):
        p.drawRect(QRectF(x, y, 6, 6))
    for x, y in ((14, 14), (18, 18), (18, 14), (14, 18)):
        _fill_rect(p, x, y, 2.2, 2.2)


def _eye(p):
    path = QPainterPath(QPointF(2.5, 12))
    path.quadTo(12, 2.5, 21.5, 12)
    path.quadTo(12, 21.5, 2.5, 12)
    p.drawPath(path)
    p.drawEllipse(QPointF(12, 12), 2.8, 2.8)


def _eye_off(p):
    path = QPainterPath(QPointF(2.5, 12))
    path.quadTo(12, 21.5, 21.5, 12)
    p.drawPath(path)
    _line(p, 6, 16.5, 4.5, 19)
    _line(p, 12, 17.5, 12, 20.5)
    _line(p, 18, 16.5, 19.5, 19)


def _lock(p):
    p.drawRoundedRect(QRectF(5, 10.5, 14, 9.5), 2, 2)
    path = QPainterPath(QPointF(8, 10.5))
    path.lineTo(8, 8)
    path.arcTo(QRectF(8, 4, 8, 8), 180, -180)
    path.lineTo(16, 10.5)
    p.drawPath(path)


def _unlock(p):
    p.drawRoundedRect(QRectF(5, 10.5, 14, 9.5), 2, 2)
    path = QPainterPath(QPointF(8, 10.5))
    path.lineTo(8, 7.5)
    path.arcTo(QRectF(8, 3.5, 8, 8), 180, -150)
    p.drawPath(path)


def _grid(p):
    p.drawRoundedRect(QRectF(4, 4, 16, 16), 1.5, 1.5)
    _line(p, 9.3, 4, 9.3, 20)
    _line(p, 14.7, 4, 14.7, 20)
    _line(p, 4, 9.3, 20, 9.3)
    _line(p, 4, 14.7, 20, 14.7)


def _magnet(p):
    path = QPainterPath(QPointF(5, 4.5))
    path.lineTo(5, 12)
    path.arcTo(QRectF(5, 5, 14, 14), 180, 180)
    path.lineTo(19, 4.5)
    path.lineTo(15, 4.5)
    path.lineTo(15, 12)
    path.arcTo(QRectF(9, 9, 6, 6), 0, -180)
    path.lineTo(9, 4.5)
    path.closeSubpath()
    p.drawPath(path)
    _line(p, 5, 8, 9, 8)
    _line(p, 15, 8, 19, 8)


def _cut(p):
    pen = QPen(p.pen())
    pen.setStyle(Qt.CustomDashLine)
    pen.setDashPattern([2.2, 1.8])
    pen.setCapStyle(Qt.FlatCap)
    p.save()
    p.setPen(pen)
    p.drawRoundedRect(QRectF(4, 6, 16, 12), 2.5, 2.5)
    p.restore()


def _fit(p):
    _line(p, 4, 9, 4, 4, 9, 4)
    _line(p, 15, 4, 20, 4, 20, 9)
    _line(p, 20, 15, 20, 20, 15, 20)
    _line(p, 9, 20, 4, 20, 4, 15)


def _align(edge: str) -> Callable[[QPainter], None]:
    def draw(p):
        if edge in ("left", "center", "right"):
            x = {"left": 4, "center": 12, "right": 20}[edge]
            _line(p, x, 3.5, x, 20.5)
            for y, w in ((7, 12), (14, 8)):
                left = {"left": 6.5, "center": 12 - w / 2, "right": 17.5 - w}[edge]
                p.drawRoundedRect(QRectF(left, y, w, 3.5), 1, 1)
        else:
            y = {"top": 4, "middle": 12, "bottom": 20}[edge]
            _line(p, 3.5, y, 20.5, y)
            for x, h in ((7, 12), (14, 8)):
                top = {"top": 6.5, "middle": 12 - h / 2, "bottom": 17.5 - h}[edge]
                p.drawRoundedRect(QRectF(x, top, 3.5, h), 1, 1)
    return draw


def _distribute(horizontal: bool) -> Callable[[QPainter], None]:
    def draw(p):
        if horizontal:
            _line(p, 4, 4, 4, 20)
            _line(p, 20, 4, 20, 20)
            p.drawRoundedRect(QRectF(10, 7, 4, 10), 1, 1)
        else:
            _line(p, 4, 4, 20, 4)
            _line(p, 4, 20, 20, 20)
            p.drawRoundedRect(QRectF(7, 10, 10, 4), 1, 1)
    return draw


def _layers(p):
    _line(p, 12, 4, 20, 8.5, 12, 13, 4, 8.5, 12, 4)
    _line(p, 4, 12.5, 12, 17, 20, 12.5)
    _line(p, 4, 16.5, 12, 21, 20, 16.5)


def _card_blank(p):
    p.drawRoundedRect(QRectF(3, 5.5, 18, 13), 2.5, 2.5)


def _sparkle(p):
    path = QPainterPath(QPointF(10, 3.5))
    path.quadTo(10.6, 10.4, 17.5, 11)
    path.quadTo(10.6, 11.6, 10, 18.5)
    path.quadTo(9.4, 11.6, 2.5, 11)
    path.quadTo(9.4, 10.4, 10, 3.5)
    p.drawPath(path)
    _line(p, 18.5, 15, 18.5, 20)
    _line(p, 16, 17.5, 21, 17.5)


def _save(p):
    path = QPainterPath(QPointF(5.5, 4))
    path.lineTo(16, 4)
    path.lineTo(20, 8)
    path.lineTo(20, 18.5)
    path.quadTo(20, 20, 18.5, 20)
    path.lineTo(5.5, 20)
    path.quadTo(4, 20, 4, 18.5)
    path.lineTo(4, 5.5)
    path.quadTo(4, 4, 5.5, 4)
    p.drawPath(path)
    p.drawRect(QRectF(8, 13, 8, 7))
    _line(p, 8, 4, 8, 8, 14, 8, 14, 4)


def _more(p):
    for x in (6, 12, 18):
        p.save()
        p.setBrush(p.pen().color())
        p.setPen(Qt.NoPen)
        p.drawEllipse(QPointF(x, 12), 1.5, 1.5)
        p.restore()


_PAINTERS: dict[str, Callable[[QPainter], None]] = {
    "shape": _shape, "rect": _rect, "ellipse": _ellipse, "line": _diagonal, "barcode": _barcode, "qr": _qr,
    "eye": _eye, "eye-off": _eye_off, "lock": _lock, "unlock": _unlock,
    "grid": _grid, "magnet": _magnet, "cut": _cut, "fit": _fit, "layers": _layers,
    "card-blank": _card_blank, "sparkle": _sparkle, "save": _save, "more": _more,
    "align-left": _align("left"), "align-center": _align("center"), "align-right": _align("right"),
    "align-top": _align("top"), "align-middle": _align("middle"), "align-bottom": _align("bottom"),
    "distribute-horizontal": _distribute(True), "distribute-vertical": _distribute(False),
}

KIND_ICONS = {
    "text": "text", "image": "image", "rect": "rect", "ellipse": "ellipse", "line": "line",
    "barcode": "barcode", "qr": "qr",
}


def _resolve(color: QColor | str | None) -> QColor:
    return color if isinstance(color, QColor) else theme_color(color or "text")


def paint(painter: QPainter, name: str, rect: QRectF, color: QColor | str | None = None) -> None:
    """디자인 화면 아이콘 또는 공용 아이콘을 rect 안에 그린다."""
    if name in ICON_NAMES:
        paint_icon(painter, name, rect, _resolve(color))
        return
    draw = _PAINTERS[name]
    side = min(rect.width(), rect.height())
    painter.save()
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.translate(rect.x() + (rect.width() - side) / 2, rect.y() + (rect.height() - side) / 2)
    painter.scale(side / 24.0, side / 24.0)
    painter.setPen(QPen(_resolve(color), _STROKE, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    painter.setBrush(Qt.NoBrush)
    draw(painter)
    painter.restore()


def pixmap(name: str, size: int = 18, color: QColor | str | None = None) -> QPixmap:
    app = QGuiApplication.instance()
    dpr = app.devicePixelRatio() if app else 1.0
    pm = QPixmap(QSize(round(size * dpr), round(size * dpr)))
    pm.setDevicePixelRatio(dpr)
    pm.fill(Qt.transparent)
    painter = QPainter(pm)
    try:
        paint(painter, name, QRectF(0, 0, size, size), color)
    finally:
        painter.end()
    return pm


def icon(name: str, role: str = "text", size: int = 18, checked_role: str | None = None) -> QIcon:
    result = QIcon()
    result.addPixmap(pixmap(name, size, role), QIcon.Normal, QIcon.Off)
    result.addPixmap(pixmap(name, size, "text_disabled"), QIcon.Disabled, QIcon.Off)
    if checked_role:
        result.addPixmap(pixmap(name, size, checked_role), QIcon.Normal, QIcon.On)
    return result
