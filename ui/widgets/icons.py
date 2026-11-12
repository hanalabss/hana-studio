"""QPainter로 그리는 선 아이콘. 24 단위 격자에 그리고, 색은 테마 역할 이름으로 받는다."""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QGuiApplication, QIcon, QPainter, QPainterPath, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import QSizePolicy, QWidget

from ui.theme import ThemeManager

_STROKE = 1.75


def _line(p: QPainter, *pts: float) -> None:
    poly = QPolygonF([QPointF(pts[i], pts[i + 1]) for i in range(0, len(pts), 2)])
    p.drawPolyline(poly)


def _dot(p: QPainter, x: float, y: float, r: float) -> None:
    p.save()
    p.setBrush(p.pen().color())
    p.setPen(Qt.NoPen)
    p.drawEllipse(QPointF(x, y), r, r)
    p.restore()


def _design(p):
    p.drawRoundedRect(QRectF(3, 5, 18, 14), 2.5, 2.5)
    p.drawRoundedRect(QRectF(6.25, 8.5, 4.5, 6), 1, 1)
    _line(p, 13.5, 9.75, 18, 9.75)
    _line(p, 13.5, 13.25, 16.5, 13.25)


def _people(p):
    for y in (7, 12, 17):
        _dot(p, 5, y, 1.25)
        _line(p, 9, y, 20, y)


def _print(p):
    p.drawRoundedRect(QRectF(3, 9, 18, 8), 2, 2)
    _line(p, 7, 9, 7, 4, 17, 4, 17, 9)
    p.drawRect(QRectF(7, 14, 10, 6))


def _printer(p):
    p.drawRoundedRect(QRectF(3, 7, 18, 10.5), 2.5, 2.5)
    _line(p, 7, 11, 17, 11)
    _dot(p, 7.5, 14.4, 1.1)


def _history(p):
    p.drawEllipse(QPointF(12, 12), 8, 8)
    _line(p, 12, 7.5, 12, 12, 15, 14)


def _bolt(p):
    _line(p, 13, 3, 6, 13.5, 11.5, 13.5, 10.5, 21, 18, 10, 12.5, 10, 13, 3)


def _settings(p):
    for y, x in ((7, 9), (12, 15), (17, 8)):
        _line(p, 4, y, x - 2.2, y)
        _line(p, x + 2.2, y, 20, y)
        p.drawEllipse(QPointF(x, y), 2.2, 2.2)


def _undo(p):
    _line(p, 8, 5, 4, 9, 8, 13)
    path = QPainterPath(QPointF(4, 9))
    path.lineTo(14.5, 9)
    path.arcTo(QRectF(9.5, 9, 10, 10), 90, -180)
    path.lineTo(10, 19)
    p.drawPath(path)


def _redo(p):
    _line(p, 16, 5, 20, 9, 16, 13)
    path = QPainterPath(QPointF(20, 9))
    path.lineTo(9.5, 9)
    path.arcTo(QRectF(4.5, 9, 10, 10), 90, 180)
    path.lineTo(14, 19)
    p.drawPath(path)


def _sun(p):
    p.drawEllipse(QPointF(12, 12), 4, 4)
    for dx, dy in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        _line(p, 12 + dx * 7, 12 + dy * 7, 12 + dx * 9, 12 + dy * 9)
    for dx, dy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        _line(p, 12 + dx * 5, 12 + dy * 5, 12 + dx * 6.4, 12 + dy * 6.4)


def _moon(p):
    outer = QPainterPath()
    outer.addEllipse(QPointF(11.5, 12.5), 8, 8)
    cut = QPainterPath()
    cut.addEllipse(QPointF(16, 8), 6.5, 6.5)
    p.drawPath(outer.subtracted(cut))


def _check(p):
    _line(p, 5, 12.5, 10, 17.5, 19, 7)


def _close(p):
    _line(p, 6, 6, 18, 18)
    _line(p, 18, 6, 6, 18)


def _plus(p):
    _line(p, 12, 5, 12, 19)
    _line(p, 5, 12, 19, 12)


def _minus(p):
    _line(p, 5, 12, 19, 12)


def _chevron_left(p):
    _line(p, 14.5, 6, 8.5, 12, 14.5, 18)


def _chevron_right(p):
    _line(p, 9.5, 6, 15.5, 12, 9.5, 18)


def _chevron_up(p):
    _line(p, 6, 14.5, 12, 8.5, 18, 14.5)


def _chevron_down(p):
    _line(p, 6, 9.5, 12, 15.5, 18, 9.5)


def _info(p):
    p.drawEllipse(QPointF(12, 12), 8.5, 8.5)
    _line(p, 12, 11, 12, 16.5)
    _dot(p, 12, 7.8, 1.1)


def _warning(p):
    path = QPainterPath(QPointF(12, 4))
    path.lineTo(21, 19.5)
    path.lineTo(3, 19.5)
    path.closeSubpath()
    p.drawPath(path)
    _line(p, 12, 9.5, 12, 14)
    _dot(p, 12, 16.8, 1.1)


def _error(p):
    p.drawEllipse(QPointF(12, 12), 8.5, 8.5)
    _line(p, 12, 7.5, 12, 13)
    _dot(p, 12, 16.2, 1.1)


def _search(p):
    p.drawEllipse(QPointF(10.5, 10.5), 6, 6)
    _line(p, 15, 15, 20, 20)


def _folder(p):
    path = QPainterPath(QPointF(3, 7))
    path.lineTo(3, 18)
    path.quadTo(3, 19.5, 4.5, 19.5)
    path.lineTo(19.5, 19.5)
    path.quadTo(21, 19.5, 21, 18)
    path.lineTo(21, 9.5)
    path.quadTo(21, 8, 19.5, 8)
    path.lineTo(12, 8)
    path.lineTo(10, 5.5)
    path.lineTo(4.5, 5.5)
    path.quadTo(3, 5.5, 3, 7)
    p.drawPath(path)


def _image(p):
    p.drawRoundedRect(QRectF(3.5, 4.5, 17, 15), 2, 2)
    p.drawEllipse(QPointF(8.5, 9.5), 1.6, 1.6)
    _line(p, 4.5, 17.5, 10, 12.5, 13.5, 16, 16, 13.5, 19.5, 17)


def _text(p):
    _line(p, 5, 7.5, 5, 5, 19, 5, 19, 7.5)
    _line(p, 12, 5, 12, 19)
    _line(p, 9, 19, 15, 19)


def _trash(p):
    _line(p, 4, 7, 20, 7)
    _line(p, 9, 7, 9, 4, 15, 4, 15, 7)
    _line(p, 6, 7, 7, 20, 17, 20, 18, 7)


def _import(p):
    _line(p, 12, 3.5, 12, 14.5)
    _line(p, 7.5, 10, 12, 14.5, 16.5, 10)
    _line(p, 4, 15.5, 4, 20, 20, 20, 20, 15.5)


def _grip(p):
    for x in (9, 15):
        for y in (6.5, 12, 17.5):
            _dot(p, x, y, 1.3)


def _dot_icon(p):
    _dot(p, 12, 12, 9.6)


def _radio_dot(p):
    _dot(p, 12, 12, 6)


_PAINTERS: dict[str, Callable[[QPainter], None]] = {
    "design": _design,
    "people": _people,
    "print": _print,
    "printer": _printer,
    "history": _history,
    "bolt": _bolt,
    "settings": _settings,
    "undo": _undo,
    "redo": _redo,
    "sun": _sun,
    "moon": _moon,
    "check": _check,
    "close": _close,
    "plus": _plus,
    "minus": _minus,
    "chevron-left": _chevron_left,
    "chevron-right": _chevron_right,
    "chevron-up": _chevron_up,
    "chevron-down": _chevron_down,
    "info": _info,
    "warning": _warning,
    "error": _error,
    "search": _search,
    "folder": _folder,
    "image": _image,
    "text": _text,
    "trash": _trash,
    "import": _import,
    "grip": _grip,
    "dot": _dot_icon,
    "radio-dot": _radio_dot,
}

ICON_NAMES: tuple[str, ...] = tuple(_PAINTERS)


def _resolve(color: QColor | str | None) -> QColor:
    if isinstance(color, QColor):
        return color
    return ThemeManager.instance().theme.color(color or "text")


def paint_icon(painter: QPainter, name: str, rect: QRectF, color: QColor | str | None = None,
               stroke: float = _STROKE) -> None:
    """rect 안에 아이콘을 그린다. color는 QColor 또는 테마 역할 이름."""
    draw = _PAINTERS[name]
    side = min(rect.width(), rect.height())
    painter.save()
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.translate(rect.x() + (rect.width() - side) / 2, rect.y() + (rect.height() - side) / 2)
    painter.scale(side / 24.0, side / 24.0)
    pen = QPen(_resolve(color), stroke, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)
    draw(painter)
    painter.restore()


def icon_pixmap(name: str, size: int = 20, color: QColor | str | None = None,
                dpr: float | None = None, stroke: float = _STROKE) -> QPixmap:
    if name not in _PAINTERS:
        raise KeyError(f"없는 아이콘 이름: {name}")
    if dpr is None:
        app = QGuiApplication.instance()
        dpr = app.devicePixelRatio() if app else 1.0
    pm = QPixmap(QSize(round(size * dpr), round(size * dpr)))
    pm.setDevicePixelRatio(dpr)
    pm.fill(Qt.transparent)
    painter = QPainter(pm)
    try:
        paint_icon(painter, name, QRectF(0, 0, size, size), color, stroke)
    finally:
        painter.end()
    return pm


def themed_icon(name: str, role: str = "text", size: int = 20, disabled_role: str = "text_disabled") -> QIcon:
    """현재 테마 색으로 만든 QIcon. 테마가 바뀌면 다시 만들어야 한다."""
    icon = QIcon()
    icon.addPixmap(icon_pixmap(name, size, role), QIcon.Normal)
    icon.addPixmap(icon_pixmap(name, size, disabled_role), QIcon.Disabled)
    return icon


class IconWidget(QWidget):
    """테마가 바뀌면 알아서 다시 그려지는 아이콘."""

    def __init__(self, name: str, size: int = 20, role: str = "text", parent: QWidget | None = None):
        super().__init__(parent)
        self._name = name
        self._role = role
        self._size = size
        self.setFixedSize(size, size)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)

    def set_icon(self, name: str, role: str | None = None) -> None:
        self._name = name
        if role is not None:
            self._role = role
        self.update()

    def icon_name(self) -> str:
        return self._name

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        role = self._role if self.isEnabled() else "text_disabled"
        paint_icon(painter, self._name, QRectF(0, 0, self._size, self._size), role)
        painter.end()
