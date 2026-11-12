"""카드 한 면을 보여주고 마우스·키보드로 직접 고치게 하는 캔버스 위젯."""

from __future__ import annotations

import math

from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush, QColor, QFont, QFontMetricsF, QKeySequence, QPainter, QPainterPath, QPen, QPolygonF,
    QTextOption, QTransform,
)
from PySide6.QtWidgets import QFrame, QGraphicsView, QPlainTextEdit, QWidget

from design import renderer
from design.binding import FixedSource
from ui.editor import geometry, snapping, tag_placement
from ui.editor.controller import (
    EditorController, EditorError, bound_columns, image_paths_from_mime, parse_field_mime,
)
from ui.editor.palette import EditorPalette
from ui.editor.scene import CARD_RADIUS_MM

RULER_PX = 24
HANDLE_PX = 8.0
HANDLE_HIT_PX = 7.0
ROTATE_OFFSET_PX = 26.0
SNAP_PX = 6.0
TAG_HEIGHT_PX = 20.0
HIT_PX = 4.0
DRAG_THRESHOLD_PX = 3
MIN_ZOOM, MAX_ZOOM = 0.1, 16.0
FIT_MARGIN_PX = 44
_KEEP_ASPECT_KINDS = ("image", "qr")
_UI_FONT_FAMILIES = ["Malgun Gothic", "맑은 고딕", "Noto Sans KR", "Segoe UI"]


def _ui_font(pixel_size: int, bold: bool = False) -> QFont:
    font = QFont()
    font.setFamilies(_UI_FONT_FAMILIES)
    font.setPixelSize(pixel_size)
    font.setBold(bold)
    return font


class Ruler(QWidget):
    """mm 눈금자. 보기의 확대·이동을 따라간다."""

    def __init__(self, view: "EditorView", orientation: Qt.Orientation) -> None:
        super().__init__(view)
        self._view = view
        self._orientation = orientation
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

    @staticmethod
    def steps(px_per_mm: float) -> tuple[float, float]:
        """(숫자를 적는 간격, 작은 눈금 간격) mm."""
        label = next((s for s in (1, 2, 5, 10, 20, 50, 100, 200) if s * px_per_mm >= 46), 500)
        for divisor in (10, 5, 2):
            minor = label / divisor
            if minor * px_per_mm >= 4.5 and (minor >= 1 or px_per_mm >= 30):
                return float(label), minor
        return float(label), float(label)

    def paintEvent(self, event) -> None:
        view = self._view
        palette = view.palette_colors
        horizontal = self._orientation == Qt.Orientation.Horizontal
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(palette.ruler_bg))
        length = self.width() if horizontal else self.height()
        thickness = self.height() if horizontal else self.width()
        ppm = view.px_per_mm
        origin_point = view.viewportTransform().map(QPointF(0.0, 0.0))
        origin = origin_point.x() if horizontal else origin_point.y()

        span = view.selection_span(horizontal)
        if span is not None:
            start, end = origin + span[0] * ppm, origin + span[1] * ppm
            band = QRectF(start, 0, end - start, thickness) if horizontal else QRectF(0, start, thickness, end - start)
            painter.fillRect(band, QColor(palette.ruler_span))

        label_step, minor_step = self.steps(ppm)
        tick_pen = QPen(QColor(palette.ruler_tick), 1)
        painter.setFont(_ui_font(10))
        first = math.floor((0 - origin) / ppm / minor_step) - 1
        last = math.ceil((length - origin) / ppm / minor_step) + 1
        per_label = max(1, round(label_step / minor_step))
        for index in range(first, last + 1):
            pos = math.floor(origin + index * minor_step * ppm) + 0.5
            if pos < -1 or pos > length + 1:
                continue
            is_label = index % per_label == 0
            is_half = per_label % 2 == 0 and index % (per_label // 2) == 0
            tick = 9.0 if is_label else (6.0 if is_half else 3.5)
            painter.setPen(tick_pen)
            if horizontal:
                painter.drawLine(QPointF(pos, thickness - tick), QPointF(pos, thickness))
            else:
                painter.drawLine(QPointF(thickness - tick, pos), QPointF(thickness, pos))
            if is_label:
                text = f"{index * minor_step:g}"
                painter.setPen(QColor(palette.ruler_text))
                if horizontal:
                    painter.drawText(QRectF(pos + 3, 1, 60, 13), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, text)
                else:
                    painter.save()
                    painter.translate(1, pos - 3)
                    painter.rotate(-90)
                    painter.drawText(QRectF(0, 0, 60, 13), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, text)
                    painter.restore()

        marker = view.cursor_mm
        if marker is not None:
            pos = math.floor(origin + (marker[0] if horizontal else marker[1]) * ppm) + 0.5
            painter.setPen(QPen(QColor(palette.ruler_marker), 1))
            if horizontal:
                painter.drawLine(QPointF(pos, 0), QPointF(pos, thickness))
            else:
                painter.drawLine(QPointF(0, pos), QPointF(thickness, pos))

        painter.setPen(QPen(QColor(palette.ruler_border), 1))
        if horizontal:
            painter.drawLine(QPointF(0, thickness - 0.5), QPointF(length, thickness - 0.5))
        else:
            painter.drawLine(QPointF(thickness - 0.5, 0), QPointF(thickness - 0.5, length))


class _RulerCorner(QWidget):
    def __init__(self, view: "EditorView") -> None:
        super().__init__(view)
        self._view = view

    def paintEvent(self, event) -> None:
        palette = self._view.palette_colors
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(palette.ruler_bg))
        painter.setFont(_ui_font(9))
        painter.setPen(QColor(palette.ruler_text))
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "mm")
        painter.setPen(QPen(QColor(palette.ruler_border), 1))
        painter.drawLine(QPointF(0, self.height() - 0.5), QPointF(self.width(), self.height() - 0.5))
        painter.drawLine(QPointF(self.width() - 0.5, 0), QPointF(self.width() - 0.5, self.height()))


class _InlineTextEdit(QPlainTextEdit):
    """캔버스 위에서 글자를 바로 고치는 입력 칸. Enter로 확정, Shift+Enter로 줄바꿈, Esc로 취소."""

    committed = Signal(str)
    cancelled = Signal()

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setAccessibleName("글자 입력")
        self._done = False

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self._finish(False)
            return
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if not event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                self._finish(True)
                return
        super().keyPressEvent(event)

    def focusOutEvent(self, event) -> None:
        super().focusOutEvent(event)
        self._finish(True)

    def _finish(self, commit: bool) -> None:
        if self._done:
            return
        self._done = True
        if commit:
            self.committed.emit(self.toPlainText())
        else:
            self.cancelled.emit()


class EditorView(QGraphicsView):
    """편집기 캔버스. `EditorView(controller)`를 레이아웃에 넣으면 된다."""

    zoom_changed = Signal(float)
    cursor_moved = Signal(float, float)

    def __init__(
        self, controller: EditorController, palette: EditorPalette | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(controller.scene, parent)
        self.controller = controller
        self.palette_colors = palette or EditorPalette.light()
        self._zoom = 1.0
        self._auto_fit = True
        self._base_px_per_mm = max(1.0, self.logicalDpiX() / 25.4)
        self.cursor_mm: tuple[float, float] | None = None

        self._mode: str | None = None
        self._press_view = QPoint()
        self._press_scene = (0.0, 0.0)
        self._dragging = False
        self._token: object | None = None
        self._start_positions: dict[str, tuple[float, float]] = {}
        self._start_bounds = (0.0, 0.0, 0.0, 0.0)
        self._start_box: geometry.Box | None = None
        self._active_id: str | None = None
        self._handle: str | None = None
        self._grab_offset = (0.0, 0.0)
        self._fixed_point = (0.0, 0.0)
        self._targets = snapping.SnapTargets()
        self._guides: list[snapping.Guide] = []
        self._rubber: tuple[float, float, float, float] | None = None
        self._rubber_base: list[str] = []
        self._click_candidate: str | None = None
        self._toggle_candidate: str | None = None
        self._hover_id: str | None = None
        self._readout: str | None = None
        self._space_down = False
        self._pan_last = QPoint()
        self._nudge_token: object | None = None
        self._editor: _InlineTextEdit | None = None
        self._editing_id: str | None = None

        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.setAcceptDrops(True)
        self.setAccessibleName("카드 디자인 캔버스")

        self._h_ruler = Ruler(self, Qt.Orientation.Horizontal)
        self._v_ruler = Ruler(self, Qt.Orientation.Vertical)
        self._corner = _RulerCorner(self)
        self._rulers_visible = True
        self.setViewportMargins(RULER_PX, RULER_PX, 0, 0)

        controller.scene.set_rounded_card(True)
        self._apply_palette()
        self._apply_zoom(1.0)
        controller.attach_view(self)
        controller.card_changed.connect(self._on_card_changed)
        controller.side_changed.connect(self._on_card_changed)
        controller.selection_changed.connect(lambda _ids: self.refresh_overlay())

    # ================================================================ 모양

    def set_palette(self, palette: EditorPalette) -> None:
        self.palette_colors = palette
        self._apply_palette()
        self.refresh_overlay()

    def _apply_palette(self) -> None:
        self.setBackgroundBrush(QBrush(QColor(self.palette_colors.canvas_bg)))
        card = self.controller.scene._card
        card.fill = QColor(self.palette_colors.card_fill)
        card.update()

    def set_rulers_visible(self, visible: bool) -> None:
        self._rulers_visible = bool(visible)
        margin = RULER_PX if visible else 0
        self.setViewportMargins(margin, margin, 0, 0)
        for widget in (self._h_ruler, self._v_ruler, self._corner):
            widget.setVisible(visible)
        self._layout_rulers()

    def _layout_rulers(self) -> None:
        if not self._rulers_visible:
            return
        self._corner.setGeometry(0, 0, RULER_PX, RULER_PX)
        self._h_ruler.setGeometry(RULER_PX, 0, max(0, self.width() - RULER_PX), RULER_PX)
        self._v_ruler.setGeometry(0, RULER_PX, RULER_PX, max(0, self.height() - RULER_PX))

    def refresh_overlay(self) -> None:
        self.viewport().update()
        self._h_ruler.update()
        self._v_ruler.update()

    @property
    def active_guides(self) -> list[snapping.Guide]:
        """끄는 동안 보이는 달라붙기 안내선."""
        return list(self._guides)

    def selection_span(self, horizontal: bool) -> tuple[float, float] | None:
        items = self.controller.selected_items()
        if not items:
            return None
        left, top, right, bottom = geometry.union_bounds(item.bounds() for item in items)
        return (left, right) if horizontal else (top, bottom)

    # ================================================================ 확대 / 이동

    @property
    def zoom(self) -> float:
        """1.0이 실제 크기(100%)."""
        return self._zoom

    @property
    def px_per_mm(self) -> float:
        return self._base_px_per_mm * self._zoom

    def _apply_zoom(self, zoom: float) -> None:
        self._zoom = min(MAX_ZOOM, max(MIN_ZOOM, zoom))
        scale = self.px_per_mm
        self.setTransform(QTransform.fromScale(scale, scale))
        self._update_scrollbars()

    def _update_scrollbars(self) -> None:
        """카드가 화면에 다 들어오면 스크롤 막대를 숨긴다."""
        card = self.controller.scene.card_rect()
        ppm = self.px_per_mm
        fits = (card.width() * ppm + 16 <= self.viewport().width()
                and card.height() * ppm + 16 <= self.viewport().height())
        policy = Qt.ScrollBarPolicy.ScrollBarAlwaysOff if fits else Qt.ScrollBarPolicy.ScrollBarAsNeeded
        if self.horizontalScrollBarPolicy() != policy:
            self.setHorizontalScrollBarPolicy(policy)
            self.setVerticalScrollBarPolicy(policy)

    def set_zoom(self, zoom: float, anchor: QPointF | None = None, *, keep_auto_fit: bool = False) -> None:
        """anchor(뷰포트 좌표) 아래의 지점이 제자리에 있도록 확대한다."""
        self.commit_text_edit()
        if not keep_auto_fit:
            self._auto_fit = False
        if anchor is None:
            anchor = QPointF(self.viewport().rect().center())
        before = self._to_scene(anchor)
        previous = self._zoom
        self._apply_zoom(zoom)
        after = self.viewportTransform().map(QPointF(*before))
        delta = after - anchor
        self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() + round(delta.x()))
        self.verticalScrollBar().setValue(self.verticalScrollBar().value() + round(delta.y()))
        self.refresh_overlay()
        if abs(previous - self._zoom) > 1e-9:
            self.zoom_changed.emit(self._zoom)

    def zoom_in(self) -> None:
        self.set_zoom(self._zoom * 1.25)

    def zoom_out(self) -> None:
        self.set_zoom(self._zoom / 1.25)

    def zoom_actual(self) -> None:
        self.set_zoom(1.0)
        self.centerOn(self.controller.scene.card_rect().center())
        self.refresh_overlay()

    def zoom_fit(self) -> None:
        """카드 전체가 보이게 맞춘다. 창 크기가 바뀌면 계속 따라간다."""
        self._fit_rect(self.controller.scene.card_rect(), FIT_MARGIN_PX)
        self._auto_fit = True

    def zoom_to_selection(self) -> None:
        items = self.controller.selected_items()
        if not items:
            self.zoom_fit()
            return
        left, top, right, bottom = geometry.union_bounds(item.bounds() for item in items)
        self._fit_rect(QRectF(left, top, max(1.0, right - left), max(1.0, bottom - top)), 72)
        self._auto_fit = False

    def _fit_rect(self, rect: QRectF, margin: int) -> None:
        viewport = self.viewport().rect()
        available_w = max(40, viewport.width() - 2 * margin)
        available_h = max(40, viewport.height() - 2 * margin)
        zoom = min(available_w / (rect.width() * self._base_px_per_mm),
                   available_h / (rect.height() * self._base_px_per_mm))
        previous = self._zoom
        self.commit_text_edit()
        self._apply_zoom(zoom)
        self.centerOn(rect.center())
        self.refresh_overlay()
        if abs(previous - self._zoom) > 1e-9:
            self.zoom_changed.emit(self._zoom)

    def visible_center_mm(self) -> tuple[float, float]:
        return self._to_scene(QPointF(self.viewport().rect().center()))

    def _to_scene(self, view_pos: QPointF) -> tuple[float, float]:
        inverted, _ = self.viewportTransform().inverted()
        point = inverted.map(QPointF(view_pos))
        return (point.x(), point.y())

    def _to_view(self, x: float, y: float) -> QPointF:
        return self.viewportTransform().map(QPointF(x, y))

    def mm_to_viewport(self, x: float, y: float) -> QPointF:
        """면 좌표(mm)를 뷰포트 픽셀 좌표로."""
        return self._to_view(x, y)

    def viewport_to_mm(self, pos: QPointF | QPoint) -> tuple[float, float]:
        return self._to_scene(QPointF(pos))

    def _on_card_changed(self, *_args) -> None:
        self._cancel_text_edit()
        self._mode = None
        self._guides = []
        self._hover_id = None
        self.zoom_fit()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._layout_rulers()
        if self._auto_fit:
            self._fit_rect(self.controller.scene.card_rect(), FIT_MARGIN_PX)
        else:
            self._update_scrollbars()

    def scrollContentsBy(self, dx: int, dy: int) -> None:
        super().scrollContentsBy(dx, dy)
        self._h_ruler.update()
        self._v_ruler.update()
        self._reposition_editor()

    def wheelEvent(self, event) -> None:
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            steps = event.angleDelta().y() / 120.0
            if steps:
                self.set_zoom(self._zoom * (1.15 ** steps), event.position())
            event.accept()
            return
        self._auto_fit = False
        super().wheelEvent(event)

    # ================================================================ 손잡이

    def _handle_points(self, item) -> dict[str, QPointF]:
        """선택한 항목의 손잡이 위치(뷰포트 좌표)."""
        if item.kind == "line":
            a, b = geometry.line_endpoints(item)
            return {"p1": self._to_view(*a), "p2": self._to_view(*b)}
        box = geometry.box_of(item)
        ppm = self.px_per_mm
        width_px, height_px = item.w * ppm, item.h * ppm
        names = list(geometry.CORNER_HANDLES)
        if width_px >= 30:
            names += ["n", "s"]
        if height_px >= 30:
            names += ["e", "w"]
        push = 6.0 if min(width_px, height_px) < 16 else 0.0
        points: dict[str, QPointF] = {}
        for name in names:
            fx, fy = geometry.HANDLES[name]
            point = self._to_view(*geometry.to_scene(box, fx, fy))
            if push:
                ox, oy = geometry.rotate_vec((fx - 0.5) * 2 * push, (fy - 0.5) * 2 * push, item.rotation)
                point = QPointF(point.x() + ox, point.y() + oy)
            points[name] = point
        top = self._to_view(*geometry.to_scene(box, 0.5, 0.0))
        ux, uy = geometry.rotate_vec(0.0, -(ROTATE_OFFSET_PX + push), item.rotation)
        points["rotate"] = QPointF(top.x() + ux, top.y() + uy)
        return points

    def _single_editable(self):
        items = self.controller.selected_items()
        if len(items) == 1 and not items[0].locked and items[0].visible:
            return items[0]
        return None

    def _handle_at(self, view_pos: QPointF) -> tuple[str, object] | None:
        item = self._single_editable()
        if item is None:
            return None
        points = self._handle_points(item)
        order = [n for n in ("p1", "p2", "rotate", *geometry.CORNER_HANDLES, "n", "e", "s", "w") if n in points]
        for name in order:
            point = points[name]
            reach = HANDLE_HIT_PX + (2 if name == "rotate" else 0)
            if abs(point.x() - view_pos.x()) <= reach and abs(point.y() - view_pos.y()) <= reach:
                return name, item
        return None

    def _handle_cursor(self, name: str, item) -> Qt.CursorShape:
        if name == "rotate":
            return Qt.CursorShape.CrossCursor
        if name in ("p1", "p2"):
            return Qt.CursorShape.CrossCursor
        fx, fy = geometry.HANDLES[name]
        dx, dy = geometry.rotate_vec(fx - 0.5, fy - 0.5, item.rotation)
        angle = math.degrees(math.atan2(dy, dx)) % 180.0
        if angle < 22.5 or angle >= 157.5:
            return Qt.CursorShape.SizeHorCursor
        if angle < 67.5:
            return Qt.CursorShape.SizeFDiagCursor
        if angle < 112.5:
            return Qt.CursorShape.SizeVerCursor
        return Qt.CursorShape.SizeBDiagCursor

    # ================================================================ 마우스

    def _tolerance_mm(self, pixels: float = HIT_PX) -> float:
        return pixels / self.px_per_mm

    def _snap_active(self, modifiers) -> bool:
        return self.controller.snap_enabled and not modifiers & Qt.KeyboardModifier.AltModifier

    def _grid_step(self) -> float | None:
        return self.controller.grid_step_mm if self.controller.grid_visible else None

    def mousePressEvent(self, event) -> None:
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        self.commit_text_edit()
        position = event.position()
        self._press_view = position.toPoint()
        self._press_scene = self._to_scene(position)
        self._dragging = False
        self._click_candidate = None
        self._toggle_candidate = None
        if event.button() == Qt.MouseButton.MiddleButton or (
            event.button() == Qt.MouseButton.LeftButton and self._space_down
        ):
            self._mode = "pan"
            self._pan_last = position.toPoint()
            self._auto_fit = False
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return

        controller = self.controller
        modifiers = event.modifiers()
        toggle = bool(modifiers & (Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.ControlModifier))

        handle = self._handle_at(position)
        if handle is not None:
            name, item = handle
            self._begin_handle_drag(name, item)
            return

        x, y = self._press_scene
        tolerance = self._tolerance_mm()
        hit = controller.scene.item_at(x, y, tolerance)
        selected = set(controller.selection)
        if toggle:
            if hit is None:
                self._begin_rubber(additive=True)
                return
            # 이미 고른 항목은 끌지 않고 놓았을 때만 선택에서 뺀다(Shift로 방향을 고정해 끌 수 있게).
            if hit.id in selected:
                self._toggle_candidate = hit.id
            else:
                controller.select([hit.id], additive=True)
            self._begin_move()
            return
        if hit is None:
            self._begin_rubber(additive=False)
            return
        if hit.id not in selected:
            under_selected = any(
                not it.locked and geometry.hit_item(it, x, y, tolerance) for it in controller.selected_items()
            )
            if under_selected:
                self._click_candidate = hit.id
            else:
                controller.select([hit.id])
        self._begin_move()

    def _begin_rubber(self, additive: bool) -> None:
        self._mode = "rubber"
        self._rubber_base = self.controller.selection if additive else []
        self._rubber = None

    def _begin_move(self) -> None:
        controller = self.controller
        items = [it for it in controller.selected_items() if not it.locked]
        if not items:
            self._mode = None
            return
        self._mode = "move"
        self._token = controller.new_merge_token()
        self._start_positions = {it.id: (it.x, it.y) for it in items}
        self._start_bounds = geometry.union_bounds(it.bounds() for it in items)
        self._targets = snapping.collect_targets(
            controller.document, controller.current_side, self._start_positions.keys()
        )

    def _begin_handle_drag(self, name: str, item) -> None:
        controller = self.controller
        self._token = controller.new_merge_token()
        self._active_id = item.id
        self._handle = name
        self._start_box = geometry.box_of(item)
        self._targets = snapping.collect_targets(controller.document, controller.current_side, [item.id])
        px, py = self._press_scene
        if name == "rotate":
            self._mode = "rotate"
            return
        if name in ("p1", "p2"):
            a, b = geometry.line_endpoints(item)
            moving, fixed = (a, b) if name == "p1" else (b, a)
            self._mode = "line"
            self._fixed_point = fixed
            self._grab_offset = (moving[0] - px, moving[1] - py)
            return
        hx, hy = geometry.handle_point(self._start_box, name)
        self._mode = "resize"
        self._grab_offset = (hx - px, hy - py)

    def mouseMoveEvent(self, event) -> None:
        position = event.position()
        scene_pos = self._to_scene(position)
        self.cursor_mm = scene_pos
        self.cursor_moved.emit(*scene_pos)
        mode = self._mode
        if mode == "pan":
            delta = position.toPoint() - self._pan_last
            self._pan_last = position.toPoint()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            self.refresh_overlay()
            return
        if mode is None or not event.buttons() & Qt.MouseButton.LeftButton:
            if mode is not None:
                self._end_drag()
            self._update_hover(position)
            self._h_ruler.update()
            self._v_ruler.update()
            return
        if not self._dragging:
            if (position.toPoint() - self._press_view).manhattanLength() < DRAG_THRESHOLD_PX:
                return
            self._dragging = True
        modifiers = event.modifiers()
        if mode == "move":
            self._drag_move(scene_pos, modifiers)
        elif mode == "resize":
            self._drag_resize(scene_pos, modifiers)
        elif mode == "rotate":
            self._drag_rotate(scene_pos, modifiers)
        elif mode == "line":
            self._drag_line(scene_pos, modifiers)
        elif mode == "rubber":
            self._drag_rubber(scene_pos)
        self.refresh_overlay()

    def mouseReleaseEvent(self, event) -> None:
        mode = self._mode
        if mode == "pan":
            self._mode = None
            self.viewport().setCursor(
                Qt.CursorShape.OpenHandCursor if self._space_down else Qt.CursorShape.ArrowCursor
            )
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if not self._dragging:
            if mode == "rubber" and not self._rubber_base:
                self.controller.clear_selection()
            elif mode == "move" and self._toggle_candidate:
                self.controller.toggle_selected(self._toggle_candidate)
            elif mode == "move" and self._click_candidate:
                self.controller.select([self._click_candidate])
        self._end_drag()
        self._update_hover(event.position())

    def _end_drag(self) -> None:
        self._mode = None
        self._dragging = False
        self._guides = []
        self._rubber = None
        self._readout = None
        self._token = None
        self._click_candidate = None
        self._toggle_candidate = None
        self.refresh_overlay()

    def _drag_move(self, scene_pos, modifiers) -> None:
        dx = scene_pos[0] - self._press_scene[0]
        dy = scene_pos[1] - self._press_scene[1]
        lock_x = lock_y = False
        if modifiers & Qt.KeyboardModifier.ShiftModifier:
            if abs(dx) >= abs(dy):
                dy, lock_y = 0.0, True
            else:
                dx, lock_x = 0.0, True
        dx, dy = round(dx, 2), round(dy, 2)
        self._guides = []
        if self._snap_active(modifiers):
            left, top, right, bottom = self._start_bounds
            moved = (left + dx, top + dy, right + dx, bottom + dy)
            snap_dx, snap_dy, guides = snapping.snap_bounds(
                moved, self._targets, self._tolerance_mm(SNAP_PX), self._grid_step()
            )
            if lock_x:
                snap_dx, guides = 0.0, [g for g in guides if g.orientation != "v"]
            if lock_y:
                snap_dy, guides = 0.0, [g for g in guides if g.orientation != "h"]
            dx, dy = dx + snap_dx, dy + snap_dy
            self._guides = guides
        positions = {i: (x + dx, y + dy) for i, (x, y) in self._start_positions.items()}
        self.controller.move_items_to(positions, self._token)
        left, top = self._start_bounds[0] + dx, self._start_bounds[1] + dy
        self._readout = f"X {left:.1f}  Y {top:.1f} mm"

    def _drag_resize(self, scene_pos, modifiers) -> None:
        item = self.controller.item(self._active_id)
        if item is None or self._start_box is None:
            return
        px = scene_pos[0] + self._grab_offset[0]
        py = scene_pos[1] + self._grab_offset[1]
        fx, fy = geometry.HANDLES[self._handle]
        shift = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
        if item.kind == "qr":
            keep_aspect = True
        elif item.kind in _KEEP_ASPECT_KINDS:
            keep_aspect = (fx != 0.5 and fy != 0.5) and not shift
        else:
            keep_aspect = shift
        self._guides = []
        rotation = self._start_box[4] % 360.0
        if self._snap_active(modifiers) and not keep_aspect and min(rotation, 360.0 - rotation) < 0.01:
            threshold = self._tolerance_mm(SNAP_PX)
            x0, y0, w0, h0, _ = self._start_box
            if fx != 0.5:
                px, guides = snapping.snap_value(
                    px, self._targets.xs, threshold, "v", (y0, y0 + h0), self._grid_step()
                )
                self._guides += guides
            if fy != 0.5:
                py, guides = snapping.snap_value(
                    py, self._targets.ys, threshold, "h", (x0, x0 + w0), self._grid_step()
                )
                self._guides += guides
        x, y, w, h = geometry.resize_box(self._start_box, self._handle, px, py, keep_aspect=keep_aspect)
        self.controller.set_geometry(item.id, {"x": x, "y": y, "w": w, "h": h}, self._token)
        self._readout = f"{w:.1f} × {h:.1f} mm"

    def _drag_rotate(self, scene_pos, modifiers) -> None:
        if self._start_box is None:
            return
        cx, cy = geometry.box_center(self._start_box)
        if math.hypot(scene_pos[0] - cx, scene_pos[1] - cy) < 1e-6:
            return
        angle = geometry.angle_to(cx, cy, scene_pos[0], scene_pos[1])
        if modifiers & Qt.KeyboardModifier.ShiftModifier:
            angle = geometry.snap_angle(angle, step=15.0)
        elif self._snap_active(modifiers):
            angle = geometry.snap_angle(angle, soft=3.0)
        angle = round(angle, 1) % 360.0
        self.controller.set_rotation([self._active_id], angle, self._token)
        self._readout = f"{angle:g}°"

    def _drag_line(self, scene_pos, modifiers) -> None:
        px = scene_pos[0] + self._grab_offset[0]
        py = scene_pos[1] + self._grab_offset[1]
        fixed_x, fixed_y = self._fixed_point
        self._guides = []
        if modifiers & Qt.KeyboardModifier.ShiftModifier:
            length = math.hypot(px - fixed_x, py - fixed_y)
            angle = math.radians(round(math.degrees(math.atan2(py - fixed_y, px - fixed_x)) / 15.0) * 15.0)
            px, py = fixed_x + length * math.cos(angle), fixed_y + length * math.sin(angle)
            if abs(math.sin(angle)) < 1e-9:
                py = fixed_y
            if abs(math.cos(angle)) < 1e-9:
                px = fixed_x
        elif self._snap_active(modifiers):
            threshold = self._tolerance_mm(SNAP_PX)
            px, guides_x = snapping.snap_value(px, self._targets.xs, threshold, "v", (py, py), self._grid_step())
            py, guides_y = snapping.snap_value(py, self._targets.ys, threshold, "h", (px, px), self._grid_step())
            if abs(py - fixed_y) <= threshold:
                py = fixed_y
            if abs(px - fixed_x) <= threshold:
                px = fixed_x
            self._guides = guides_x + guides_y
        values = geometry.line_from_points((fixed_x, fixed_y), (px, py))
        self.controller.set_geometry(self._active_id, values, self._token)
        self._readout = f"{math.hypot(px - fixed_x, py - fixed_y):.1f} mm"

    def _drag_rubber(self, scene_pos) -> None:
        x0, y0 = self._press_scene
        x1, y1 = scene_pos
        self._rubber = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        found = [it.id for it in self.controller.scene.items_in_bounds(*self._rubber)]
        self.controller.select(self._rubber_base + [i for i in found if i not in self._rubber_base])

    def _update_hover(self, position: QPointF) -> None:
        if self._space_down:
            self.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
            return
        handle = self._handle_at(position)
        hover_id = None
        if handle is not None:
            cursor = self._handle_cursor(*handle)
        else:
            x, y = self._to_scene(position)
            hit = self.controller.scene.item_at(x, y, self._tolerance_mm())
            hover_id = hit.id if hit is not None else None
            cursor = Qt.CursorShape.SizeAllCursor if hit is not None else Qt.CursorShape.ArrowCursor
        self.viewport().setCursor(cursor)
        if hover_id != self._hover_id:
            self._hover_id = hover_id
            self.viewport().update()

    def leaveEvent(self, event) -> None:
        super().leaveEvent(event)
        self.cursor_mm = None
        if self._hover_id is not None:
            self._hover_id = None
        self.refresh_overlay()

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        x, y = self._to_scene(event.position())
        hit = self.controller.scene.item_at(x, y, self._tolerance_mm())
        self._end_drag()
        if hit is None:
            return
        self.controller.select([hit.id])
        if hit.kind == "text" and (hit.source is None or isinstance(hit.source, FixedSource)):
            self.begin_text_edit(hit.id)
        elif hit.kind == "text":
            self.controller.message.emit("명단에서 가져오는 글자입니다. 내용은 명단에서 바꿉니다.")
            self.controller.bound_edit_requested.emit(hit.id)
        else:
            self.controller.item_activated.emit(hit.id)

    # ================================================================ 글자 바로 고치기

    def begin_text_edit(self, item_id: str) -> bool:
        item = self.controller.item(item_id)
        if item is None or item.kind != "text" or item.locked:
            return False
        if item.source is not None and not isinstance(item.source, FixedSource):
            return False
        self.commit_text_edit()
        editor = _InlineTextEdit(self.viewport())
        self._editor, self._editing_id = editor, item_id
        text = item.source.value if isinstance(item.source, FixedSource) else item.text
        editor.setPlainText(text)
        font = QFont(renderer.resolve_font_family(item.font_family))
        font.setPixelSize(max(9, round(item.font_size_pt * 25.4 / 72.0 * self.px_per_mm)))
        font.setBold(item.bold)
        font.setItalic(item.italic)
        editor.setFont(font)
        option = QTextOption()
        option.setAlignment({
            "center": Qt.AlignmentFlag.AlignHCenter, "right": Qt.AlignmentFlag.AlignRight,
        }.get(item.h_align, Qt.AlignmentFlag.AlignLeft))
        option.setWrapMode(
            QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere if item.wrap else QTextOption.WrapMode.NoWrap
        )
        editor.document().setDefaultTextOption(option)
        editor.document().setDocumentMargin(2)
        color = QColor(item.color)
        if not color.isValid():
            color = QColor("#000000")
        light_text = color.lightnessF() > 0.6
        background = "#1B2430" if light_text else "#FFFFFF"
        editor.setStyleSheet(
            f"QPlainTextEdit {{ background: {background}; color: {color.name()};"
            f" border: 2px solid {QColor(self.palette_colors.selection).name()}; }}"
        )
        self.controller.scene.set_editing(item_id)
        self._reposition_editor()
        editor.committed.connect(self._finish_text_edit)
        editor.cancelled.connect(self._cancel_text_edit)
        editor.show()
        editor.setFocus(Qt.FocusReason.OtherFocusReason)
        editor.selectAll()
        return True

    def is_editing_text(self) -> bool:
        return self._editor is not None

    def text_editor(self) -> QPlainTextEdit | None:
        return self._editor

    def _reposition_editor(self) -> None:
        if self._editor is None:
            return
        item = self.controller.item(self._editing_id)
        if item is None:
            return
        left, top, right, bottom = item.bounds()
        top_left, bottom_right = self._to_view(left, top), self._to_view(right, bottom)
        metrics = QFontMetricsF(self._editor.font())
        width = max(bottom_right.x() - top_left.x(), 120.0) + 8
        height = max(bottom_right.y() - top_left.y(), metrics.height() + 10) + 4
        self._editor.setGeometry(round(top_left.x()) - 4, round(top_left.y()) - 2, round(width), round(height))

    def _close_editor(self) -> str | None:
        editor, item_id = self._editor, self._editing_id
        self._editor, self._editing_id = None, None
        if editor is None:
            return None
        editor._done = True
        editor.hide()
        editor.deleteLater()
        self.controller.scene.set_editing(None)
        self.setFocus(Qt.FocusReason.OtherFocusReason)
        return item_id

    def _finish_text_edit(self, text: str) -> None:
        item_id = self._close_editor()
        if item_id is not None:
            self.controller.set_text(item_id, text)
        self.refresh_overlay()

    def commit_text_edit(self) -> None:
        """캔버스에서 고치던 글자를 확정한다(저장하기 직전 등). 고치는 중이 아니면 아무 일도 없다."""
        if self._editor is not None:
            self._finish_text_edit(self._editor.toPlainText())

    _commit_text_edit = commit_text_edit

    def _cancel_text_edit(self) -> None:
        self._close_editor()
        self.refresh_overlay()

    # ================================================================ 키보드

    def keyPressEvent(self, event) -> None:
        controller = self.controller
        key = event.key()
        modifiers = event.modifiers()
        ctrl = bool(modifiers & Qt.KeyboardModifier.ControlModifier)
        shift = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)

        if key == Qt.Key.Key_Space:
            if not event.isAutoRepeat() and self._mode is None:
                self._space_down = True
                self.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
            event.accept()
            return
        arrows = {
            Qt.Key.Key_Left: (-1, 0), Qt.Key.Key_Right: (1, 0), Qt.Key.Key_Up: (0, -1), Qt.Key.Key_Down: (0, 1),
        }
        if key in arrows and not ctrl:
            step = 1.0 if shift else 0.1
            if not event.isAutoRepeat() or self._nudge_token is None:
                self._nudge_token = controller.new_merge_token()
            dx, dy = arrows[key]
            controller.nudge(dx * step, dy * step, merge_token=self._nudge_token)
            event.accept()
            return
        if key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            controller.delete()
        elif key == Qt.Key.Key_Escape:
            if self._mode is not None:
                controller.cancel_merged(self._token)
                self._end_drag()
            else:
                controller.clear_selection()
        elif event.matches(QKeySequence.StandardKey.Undo):
            controller.undo_stack.undo()
        elif event.matches(QKeySequence.StandardKey.Redo) or (ctrl and shift and key == Qt.Key.Key_Z):
            controller.undo_stack.redo()
        elif event.matches(QKeySequence.StandardKey.Copy):
            controller.copy()
        elif event.matches(QKeySequence.StandardKey.Cut):
            controller.cut()
        elif event.matches(QKeySequence.StandardKey.Paste):
            controller.paste()
        elif event.matches(QKeySequence.StandardKey.SelectAll):
            controller.select_all()
        elif ctrl and key == Qt.Key.Key_D:
            controller.duplicate()
        elif ctrl and key == Qt.Key.Key_BracketRight:
            controller.bring_to_front() if shift else controller.bring_forward()
        elif ctrl and key == Qt.Key.Key_BracketLeft:
            controller.send_to_back() if shift else controller.send_backward()
        elif ctrl and key == Qt.Key.Key_0:
            self.zoom_fit()
        elif ctrl and key == Qt.Key.Key_1:
            self.zoom_actual()
        elif ctrl and key == Qt.Key.Key_2:
            self.zoom_to_selection()
        elif ctrl and key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self.zoom_in()
        elif ctrl and key == Qt.Key.Key_Minus:
            self.zoom_out()
        elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_F2):
            item = self._single_editable()
            if item is None or not self.begin_text_edit(item.id):
                super().keyPressEvent(event)
                return
        else:
            super().keyPressEvent(event)
            return
        event.accept()

    def focusNextPrevChild(self, forward: bool) -> bool:
        """Tab으로 항목을 위에서부터 차례로 고른다. 끝에 닿으면 다음 위젯으로 넘어간다."""
        if self._editor is None and self.hasFocus():
            controller = self.controller
            ids = [it.id for it in reversed(controller.side().sorted_items()) if it.visible and not it.locked]
            selection = controller.selection
            if ids:
                if len(selection) == 1 and selection[0] in ids:
                    index = ids.index(selection[0]) + (1 if forward else -1)
                elif not selection:
                    index = 0 if forward else len(ids) - 1
                else:
                    index = -1
                if 0 <= index < len(ids):
                    controller.select([ids[index]])
                    return True
                controller.clear_selection()
        return super().focusNextPrevChild(forward)

    def keyReleaseEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_down = False
            if self._mode != "pan":
                self.viewport().setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        if not event.isAutoRepeat():
            self._nudge_token = None
        super().keyReleaseEvent(event)

    def focusOutEvent(self, event) -> None:
        super().focusOutEvent(event)
        self._space_down = False
        self._nudge_token = None

    # ================================================================ 끌어다 놓기

    def _accepts(self, mime) -> bool:
        return parse_field_mime(mime) is not None or bool(image_paths_from_mime(mime))

    def dragEnterEvent(self, event) -> None:
        if self._accepts(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:
        if self._accepts(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:
        mime = event.mimeData()
        pos = self._to_scene(event.position())
        field = parse_field_mime(mime)
        try:
            if field is not None:
                self.controller.add_bound_field(field[0], field[1], pos)
            else:
                paths = image_paths_from_mime(mime)
                if not paths:
                    event.ignore()
                    return
                for index, path in enumerate(paths):
                    self.controller.add_image_from_file(path, (pos[0] + 4.0 * index, pos[1] + 4.0 * index))
        except EditorError as exc:
            self.controller.message.emit(str(exc))
        event.acceptProposedAction()
        self.setFocus(Qt.FocusReason.OtherFocusReason)

    # ================================================================ 그리기

    def _card_paths(self) -> tuple[QRectF, QPainterPath, QRectF | None, QPainterPath | None]:
        scene = self.controller.scene
        transform = self.viewportTransform()
        ppm = self.px_per_mm
        card = transform.mapRect(scene.card_rect())
        bleed = self.controller.document.bleed_mm
        card_path = QPainterPath()
        outer_radius = scene.card_corner_radius() * ppm
        card_path.addRoundedRect(card, outer_radius, outer_radius)
        if bleed > 0:
            cut = transform.mapRect(scene.cut_rect())
            cut_path = QPainterPath()
            cut_path.addRoundedRect(cut, CARD_RADIUS_MM * ppm, CARD_RADIUS_MM * ppm)
            return card, card_path, cut, cut_path
        return card, card_path, None, None

    def drawForeground(self, painter: QPainter, rect: QRectF) -> None:
        if self.controller.document is None:
            return
        palette = self.palette_colors
        painter.save()
        painter.setWorldTransform(QTransform())
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        viewport = QRectF(self.viewport().rect())
        card, card_path, cut, cut_path = self._card_paths()

        outside = QPainterPath()
        outside.addRect(viewport)
        outside = outside.subtracted(card_path)
        dim = QColor(palette.canvas_bg)
        dim.setAlphaF(palette.outside_opacity)
        painter.fillPath(outside, dim)
        self._paint_shadow(painter, card, card_path, outside)

        painter.save()
        painter.setClipPath(card_path)
        if self.controller.grid_visible:
            self._paint_grid(painter, card)
        if cut_path is not None and self.controller.cut_line_visible:
            painter.fillPath(card_path.subtracted(cut_path), QColor(palette.bleed_tint))
        painter.restore()
        if cut_path is not None and self.controller.cut_line_visible:
            pen = QPen(QColor(palette.cut_line), 1.4)
            pen.setDashPattern([5.0, 3.5])
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(cut_path)

        self._paint_field_tints(painter, card_path)
        self._paint_warnings(painter)
        self._paint_hover(painter)
        self._paint_selection(painter)
        self._paint_guides(painter, card)
        self._paint_rubber(painter)
        self._paint_readout(painter)
        painter.restore()

    def _paint_shadow(self, painter, card: QRectF, card_path: QPainterPath, outside: QPainterPath) -> None:
        palette = self.palette_colors
        if palette.shadow_opacity <= 0:
            return
        painter.save()
        painter.setClipPath(outside)
        painter.setPen(Qt.PenStyle.NoPen)
        radius = CARD_RADIUS_MM * self.px_per_mm
        layers = 18
        color = QColor(palette.card_shadow)
        color.setAlphaF(min(1.0, palette.shadow_opacity * 0.045))
        for index in range(layers, 0, -1):
            grow = index * 1.8
            path = QPainterPath()
            path.addRoundedRect(
                card.adjusted(-grow * 0.7, -grow * 0.3 + 2, grow * 0.7, grow + 4), radius + grow, radius + grow
            )
            painter.fillPath(path, color)
        edge = QColor(palette.card_shadow)
        edge.setAlphaF(min(1.0, palette.shadow_opacity * 0.75))
        painter.setClipping(False)
        painter.setPen(QPen(edge, 1.0))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(card_path)
        painter.restore()

    def _paint_grid(self, painter, card: QRectF) -> None:
        palette = self.palette_colors
        ppm = self.px_per_mm
        step = self.controller.grid_step_mm
        levels = [m for m in (1, 5, 10, 50, 100) if m * step * ppm >= 7]
        if not levels:
            return
        minor = levels[0]
        major = levels[1] if len(levels) > 1 else None
        origin = self._to_view(0.0, 0.0)
        width_mm, height_mm = self.controller.scene.card_size()
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        for multiplier, color in ((minor, palette.grid), (major, palette.grid_major)):
            if multiplier is None:
                continue
            painter.setPen(QPen(QColor(color), 1))
            spacing = multiplier * step
            for index in range(1, int(width_mm / spacing) + 1):
                x = round(origin.x() + index * spacing * ppm)
                painter.drawLine(QPointF(x, card.top()), QPointF(x, card.bottom()))
            for index in range(1, int(height_mm / spacing) + 1):
                y = round(origin.y() + index * spacing * ppm)
                painter.drawLine(QPointF(card.left(), y), QPointF(card.right(), y))
        painter.restore()

    def _outline(self, item) -> QPolygonF:
        if item.kind == "line":
            a, b = geometry.line_endpoints(item)
            return QPolygonF([self._to_view(*a), self._to_view(*b)])
        return QPolygonF([self._to_view(x, y) for x, y in geometry.corners(geometry.box_of(item))])

    def _paint_field_tints(self, painter, card_path: QPainterPath) -> None:
        """명단에서 가져오는 항목을 옅은 노랑으로 알려 준다. 화면에만 그리고 인쇄에는 나오지 않는다."""
        controller = self.controller
        if not controller.field_tint_visible:
            return
        selected = set(controller.selection)
        painter.save()
        painter.setClipPath(card_path)
        outlines = [
            self._outline(item) for item in controller.side().items
            if item.visible and item.kind != "line" and item.id not in selected
            and item.id != self._editing_id and bound_columns(item)
        ]
        # 곱하기로 얹어 흰 바탕만 물들이고 글자·바코드의 검정은 그대로 둔다.
        # 실제 사람을 넣어 미리 볼 때는 사진과 글자가 나올 모습 그대로 보이도록 물들이지 않고 테두리만 남긴다.
        if controller.preview_record is None:
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Multiply)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(self.palette_colors.field_tint))
            for outline in outlines:
                painter.drawPolygon(outline)
        # 어두운 바탕 위에서는 물든 것이 안 보이므로 가는 테두리로도 알린다.
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        painter.setPen(QPen(QColor(self.palette_colors.field_tint_line), 1.0))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for outline in outlines:
            painter.drawPolygon(outline)
        painter.restore()

    def _paint_warnings(self, painter) -> None:
        warnings = self.controller.warnings
        if not warnings:
            return
        side = self.controller.side()
        color = QColor(self.palette_colors.warning)
        for item_id in warnings:
            item = side.get(item_id)
            if item is None or not item.visible:
                continue
            outline = self._outline(item)
            pen = QPen(color, 1.2)
            pen.setDashPattern([3.0, 3.0])
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            if item.kind == "line":
                painter.drawPolyline(outline)
            else:
                painter.drawPolygon(outline)
            anchor = outline.boundingRect().topRight()
            badge = QRectF(anchor.x() - 7, anchor.y() - 7, 14, 14)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(color)
            painter.drawEllipse(badge)
            painter.setPen(QColor("#1B2430"))
            painter.setFont(_ui_font(10, bold=True))
            painter.drawText(badge, Qt.AlignmentFlag.AlignCenter, "!")

    def _paint_hover(self, painter) -> None:
        if self._hover_id is None or self._mode is not None or self._hover_id in self.controller.selection:
            return
        item = self.controller.item(self._hover_id)
        if item is None:
            return
        painter.setPen(QPen(QColor(self.palette_colors.hover), 1.5))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        outline = self._outline(item)
        if item.kind == "line":
            painter.drawPolyline(outline)
        else:
            painter.drawPolygon(outline)

    def _paint_selection(self, painter) -> None:
        palette = self.palette_colors
        items = self.controller.selected_items()
        if not items or self._editor is not None:
            return
        selection = QColor(palette.selection)
        for item in items:
            if not item.visible:
                pen = QPen(QColor(palette.locked), 1.0, Qt.PenStyle.DotLine)
            elif item.locked:
                pen = QPen(QColor(palette.locked), 1.5)
                pen.setDashPattern([4.0, 3.0])
            else:
                pen = QPen(selection, 1.5)
            pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            outline = self._outline(item)
            if item.kind == "line":
                painter.drawPolyline(outline)
            else:
                painter.drawPolygon(outline)

        if len(items) > 1:
            left, top, right, bottom = geometry.union_bounds(item.bounds() for item in items)
            group = QRectF(self._to_view(left, top), self._to_view(right, bottom))
            pen = QPen(selection, 1.0)
            pen.setDashPattern([4.0, 3.0])
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(group.adjusted(-3, -3, 3, 3))
            return

        item = items[0]
        if item.locked or not item.visible:
            self._paint_tag(painter, item)
            return
        points = self._handle_points(item)
        dragging_other = self._dragging and self._mode in ("move",)
        if not dragging_other:
            if "rotate" in points:
                box = geometry.box_of(item)
                top = self._to_view(*geometry.to_scene(box, 0.5, 0.0))
                painter.setPen(QPen(selection, 1.2))
                painter.drawLine(top, points["rotate"])
            painter.setPen(QPen(selection, 1.5))
            painter.setBrush(QColor(palette.handle_fill))
            half = HANDLE_PX / 2.0
            for name, point in points.items():
                center = QPointF(round(point.x()), round(point.y()))
                if name in ("rotate", "p1", "p2"):
                    painter.drawEllipse(center, half + 0.5, half + 0.5)
                else:
                    painter.drawRect(QRectF(center.x() - half, center.y() - half, HANDLE_PX, HANDLE_PX))
        self._paint_tag(painter, item)

    def _tag_text(self, item) -> str:
        if item.locked:
            return "잠김"
        if not item.visible:
            return "숨김"
        columns = bound_columns(item)
        return "명단: " + ", ".join(columns) if columns else ""

    def _tag_layout(self, item, text: str) -> tuple[str, QRectF]:
        """꼬리표를 놓을 자리. 다른 항목이나 회전 손잡이를 가리지 않는 곳을 고른다."""
        def as_tuple(rect: QRectF) -> tuple[float, float, float, float]:
            return (rect.left(), rect.top(), rect.right(), rect.bottom())

        width = QFontMetricsF(_ui_font(11, bold=True)).horizontalAdvance(text) + 14
        box = self._outline(item).boundingRect()
        obstacles = [
            as_tuple(self._outline(other).boundingRect())
            for other in self.controller.side().items if other.id != item.id and other.visible
        ]
        keep_clear = []
        if not item.locked and item.visible:
            rotate = self._handle_points(item).get("rotate")
            if rotate is not None:
                stem = self._to_view(*geometry.to_scene(geometry.box_of(item), 0.5, 0.0))
                reach = HANDLE_PX / 2.0 + 3.0
                keep_clear.append((
                    min(stem.x(), rotate.x()) - reach, min(stem.y(), rotate.y()) - reach,
                    max(stem.x(), rotate.x()) + reach, max(stem.y(), rotate.y()) + reach,
                ))
        where, (left, top, right, bottom) = tag_placement.place_tag(
            as_tuple(box), (width, TAG_HEIGHT_PX), obstacles,
            viewport=as_tuple(QRectF(self.viewport().rect())), keep_clear=keep_clear,
            allow_inside=item.kind != "line",
        )
        return where, QRectF(left, top, right - left, bottom - top)

    def selection_tag(self) -> tuple[str, str, QRectF] | None:
        """하나만 골랐을 때 붙는 꼬리표의 (글자, 자리 "above" | "below" | "inside", 뷰포트 사각형)."""
        items = self.controller.selected_items()
        if len(items) != 1 or self._editor is not None:
            return None
        text = self._tag_text(items[0])
        if not text:
            return None
        where, rect = self._tag_layout(items[0], text)
        return text, where, rect

    def _paint_tag(self, painter, item) -> None:
        text = self._tag_text(item)
        if not text:
            return
        palette = self.palette_colors
        if item.locked or not item.visible:
            background, foreground = QColor(palette.locked), QColor("#FFFFFF")
        else:
            background, foreground = QColor(palette.tag_bg), QColor(palette.tag_text)
        where, rect = self._tag_layout(item, text)
        if where == tag_placement.INSIDE:
            # 사진 위에 얹힐 때도 읽히도록 글자색으로 가는 테두리를 두른다.
            painter.setPen(QPen(foreground, 1.0))
        else:
            painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(background)
        painter.drawRoundedRect(rect, 4, 4)
        painter.setFont(_ui_font(11, bold=True))
        painter.setPen(foreground)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

    def _paint_guides(self, painter, card: QRectF) -> None:
        if not self._guides:
            return
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.setPen(QPen(QColor(self.palette_colors.guide), 1))
        extend = 8.0
        for guide in self._guides:
            if guide.orientation == "v":
                a, b = self._to_view(guide.pos, guide.start), self._to_view(guide.pos, guide.end)
                x = round(a.x())
                painter.drawLine(QPointF(x, a.y() - extend), QPointF(x, b.y() + extend))
            else:
                a, b = self._to_view(guide.start, guide.pos), self._to_view(guide.end, guide.pos)
                y = round(a.y())
                painter.drawLine(QPointF(a.x() - extend, y), QPointF(b.x() + extend, y))
        painter.restore()

    def _paint_rubber(self, painter) -> None:
        if self._rubber is None:
            return
        left, top, right, bottom = self._rubber
        rect = QRectF(self._to_view(left, top), self._to_view(right, bottom))
        painter.setPen(QPen(QColor(self.palette_colors.selection), 1))
        painter.setBrush(QColor(self.palette_colors.rubber_fill))
        painter.drawRect(rect)

    def _paint_readout(self, painter) -> None:
        if not self._readout or not self._dragging:
            return
        items = self.controller.selected_items()
        if not items:
            return
        left, top, right, bottom = geometry.union_bounds(item.bounds() for item in items)
        anchor = self._to_view((left + right) / 2.0, bottom)
        font = _ui_font(11, bold=True)
        metrics = QFontMetricsF(font)
        width = metrics.horizontalAdvance(self._readout) + 16
        rect = QRectF(anchor.x() - width / 2.0, anchor.y() + 12, width, 22)
        viewport = QRectF(self.viewport().rect())
        if rect.bottom() > viewport.bottom() - 4:
            rect.moveBottom(self._to_view(left, top).y() - 12)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(self.palette_colors.readout_bg))
        painter.drawRoundedRect(rect, 5, 5)
        painter.setFont(font)
        painter.setPen(QColor(self.palette_colors.readout_text))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, self._readout)
