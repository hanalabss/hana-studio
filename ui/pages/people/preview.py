"""오른쪽 "이렇게 나옵니다": 지금 고른 사람의 카드, 사진 맞추기, 확인할 것."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QFrame, QHBoxLayout, QScrollArea, QSizePolicy, QSlider, QVBoxLayout, QWidget

from data.store import DataStore, PhotoAdjustment
from design import renderer
from design.binding import BindingContext, ColumnSource
from design.model import Crop
from services.render_job import crop_from_adjustment
from ui.theme import SPACE
from ui.widgets import (
    Divider,
    IconButton,
    IconWidget,
    QuietButton,
    SegmentedControl,
    label,
    theme_color,
)

from .inputs import Rethemed

SIDE_NAMES = {"front": "앞면", "back": "뒷면"}
ZOOM_MIN, ZOOM_MAX = 1.0, 3.0
_EPS = 1e-6


# ---------------------------------------------------------------- 사진 조정값 <-> 렌더러 Crop

def _drawn_size(image_size, frame_size, fit: str, zoom: float) -> tuple[float, float]:
    _, _, w, h = renderer.image_target_rect(
        float(image_size[0]), float(image_size[1]), float(frame_size[0]), float(frame_size[1]), fit, Crop(zoom=zoom))
    return w, h


# 명단 미리 보기와 인쇄가 같은 계산을 쓴다(services.render_job 하나뿐).
crop_for_adjustment = crop_from_adjustment


def clamp_adjustment(adjustment: PhotoAdjustment, image_size, frame_size, fit: str = "fill") -> PhotoAdjustment:
    """빈 곳이 생기지 않는 범위로 줄인다."""
    zoom = max(ZOOM_MIN, min(ZOOM_MAX, float(adjustment.zoom or 1.0)))
    draw_w, draw_h = _drawn_size(image_size, frame_size, fit, zoom)

    def limit(offset: float, drawn: float, frame: float) -> float:
        if frame <= 0:
            return 0.0
        room = abs(drawn - frame) / 2.0 / frame
        return max(-room, min(room, float(offset)))

    return PhotoAdjustment(
        zoom, limit(adjustment.offset_x, draw_w, float(frame_size[0])),
        limit(adjustment.offset_y, draw_h, float(frame_size[1])))


def photo_items(document, side_key: str, store: DataStore | None) -> list:
    """이 면에서 명단의 사진 항목과 이어진 사진 칸들."""
    if store is None:
        return []
    photo_columns = {c.name for c in store.columns() if c.type == "photo"}
    items = []
    for item in document.side(side_key).sorted_items():
        if item.kind == "image" and isinstance(item.source, ColumnSource) and item.source.column in photo_columns:
            items.append(item)
    return items


def render_card_image(
    document, side_key: str, record: Mapping[str, Any] | None, *,
    store: DataStore | None = None, record_id: int | None = None, index: int = 0,
    width_px: int = 600, adjustments: Mapping[str, PhotoAdjustment] | None = None,
) -> QImage:
    """record는 사진 경로까지 풀린 값(workspace.binding_record). None이면 빈 카드 모양."""
    width_mm, _ = document.side_size_mm(side_key)
    visible_mm = max(1.0, width_mm - 2 * float(getattr(document, "bleed_mm", 0.0) or 0.0))
    dpi = max(40.0, width_px * 25.4 / visible_mm)
    key = str(record_id) if record_id is not None else "__preview__"
    context = BindingContext(counter_index=max(0, index), record_key=key)
    applied = []
    try:
        if record is not None:
            for item in photo_items(document, side_key, store):
                column = item.source.column
                adjustment = (adjustments or {}).get(column)
                if adjustment is None and record_id is not None and store is not None:
                    adjustment = store.get_photo_adjustment(record_id, column)
                if adjustment is None or adjustment == PhotoAdjustment():
                    continue
                image = renderer.load_image(record.get(column) or "")
                if image is None:
                    continue
                applied.append((item, item.record_crops.get(key)))
                item.set_record_crop(key, crop_from_adjustment(
                    adjustment, (image.width(), image.height()), (item.w, item.h), item.fit))
        rendered = renderer.render_side(document, side_key, record, dpi, context, crop_bleed=True)
    finally:
        for item, previous in applied:
            item.set_record_crop(key, previous)
    pil = rendered.color.convert("RGB")
    data = pil.tobytes("raw", "RGB")
    return QImage(data, pil.width, pil.height, pil.width * 3, QImage.Format.Format_RGB888).copy()


# ---------------------------------------------------------------- 카드 그림

class CardView(Rethemed, QWidget):
    """둥근 모서리 카드 한 장. 사진 칸을 끌면 dragged(dx, dy)를 틀 크기 대비 비율로 보낸다."""

    dragStarted = Signal()
    dragged = Signal(float, float)
    dragFinished = Signal()

    def __init__(self, parent: QWidget | None = None, well: bool = True):
        super().__init__(parent)
        self._image: QImage | None = None
        self._frame: tuple[float, float, float, float] | None = None   # 카드 대비 비율 (x, y, w, h)
        self._press: QPointF | None = None
        self._well = well
        self._ratio = 54.0 / 86.0
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setMouseTracking(True)
        self.set_view_width(288)
        self.follow_theme()

    def set_view_width(self, width: int) -> None:
        self._view_width = int(width)
        self.setFixedSize(self._view_width, self._card_size(self._view_width).height() + 2 * self._pad())

    def setFixedWidth(self, width: int) -> None:
        self.set_view_width(width)

    def set_image(self, image: QImage | None) -> None:
        self._image = image
        if image is not None and image.width() > 0:
            ratio = image.height() / image.width()
            if abs(ratio - self._ratio) > 1e-3:
                self._ratio = ratio
                self.set_view_width(self._view_width)
        self.update()

    def image(self) -> QImage | None:
        return self._image

    def set_photo_frame(self, frame: tuple[float, float, float, float] | None) -> None:
        self._frame = frame
        if frame is None:
            self.unsetCursor()
        self.update()

    def _pad(self) -> int:
        return 18 if self._well else 6

    def _card_size(self, width: int) -> QSize:
        room = max(60, width - 2 * (16 if self._well else 6))
        w = float(room)
        h = w * self._ratio
        limit = 300.0
        if h > limit:
            w, h = limit / self._ratio, limit
        return QSize(round(w), round(h))

    def card_rect(self) -> QRectF:
        size = self._card_size(self.width())
        return QRectF((self.width() - size.width()) / 2, (self.height() - size.height()) / 2,
                      size.width(), size.height())

    def frame_rect(self) -> QRectF | None:
        if self._frame is None:
            return None
        card = self.card_rect()
        x, y, w, h = self._frame
        return QRectF(card.x() + x * card.width(), card.y() + y * card.height(), w * card.width(), h * card.height())

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        if self._well:
            painter.setPen(Qt.NoPen)
            painter.setBrush(theme_color("ground"))
            painter.drawRoundedRect(QRectF(self.rect()), 10, 10)
        card = self.card_rect()
        radius = card.width() * 3.18 / 86.0
        shadow = QColor(theme_color("text"))
        for grow, alpha in ((7, 5), (4, 9), (2, 14)):
            shadow.setAlpha(alpha)
            painter.setPen(Qt.NoPen)
            painter.setBrush(shadow)
            painter.drawRoundedRect(card.adjusted(-grow * 0.6, -grow * 0.2 + 2, grow * 0.6, grow + 2), radius + grow, radius + grow)
        clip = QPainterPath()
        clip.addRoundedRect(card, radius, radius)
        painter.save()
        painter.setClipPath(clip)
        if self._image is not None and not self._image.isNull():
            painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
            dpr = self.devicePixelRatioF()
            target = QSize(round(card.width() * dpr), round(card.height() * dpr))
            image = self._image
            if image.size() != target:
                image = image.scaled(target, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
            painter.drawImage(card, image)
        else:
            painter.fillRect(card, theme_color("surface"))
        painter.restore()
        painter.setPen(QPen(theme_color("line"), 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(card.adjusted(0.5, 0.5, -0.5, -0.5), radius, radius)
        frame = self.frame_rect()
        if frame is not None:
            pen = QPen(theme_color("focus"), 1.5, Qt.CustomDashLine)
            pen.setDashPattern([4, 3])
            painter.setPen(pen)
            painter.drawRoundedRect(frame.adjusted(-2, -2, 2, 2), 3, 3)
        painter.end()

    def mousePressEvent(self, event) -> None:
        frame = self.frame_rect()
        if event.button() == Qt.LeftButton and frame is not None and frame.contains(event.position()):
            self._press = event.position()
            self.setCursor(Qt.ClosedHandCursor)
            self.dragStarted.emit()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        frame = self.frame_rect()
        if self._press is not None and frame is not None:
            delta = event.position() - self._press
            self.dragged.emit(delta.x() / max(1.0, frame.width()), delta.y() / max(1.0, frame.height()))
            event.accept()
            return
        if frame is not None:
            self.setCursor(Qt.OpenHandCursor if frame.contains(event.position()) else Qt.ArrowCursor)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._press is not None:
            self._press = None
            self.setCursor(Qt.OpenHandCursor)
            self.dragFinished.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)


# ---------------------------------------------------------------- 확인할 것

@dataclass(frozen=True)
class Issue:
    kind: str               # "missing_photo" | "empty_field" | "unmatched_field"
    text: str
    action_text: str | None = None
    payload: Any = None
    tone: str = "warn"


class _IssueRow(QWidget):
    def __init__(self, issue: Issue, on_action, parent: QWidget | None = None):
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(SPACE["sm"])
        role = {"warn": "warn_text", "ok": "ok_text", "info": "text_muted"}.get(issue.tone, "warn_text")
        icon = {"warn": "warning", "ok": "check", "info": "info"}.get(issue.tone, "warning")
        mark = IconWidget(icon, 16, role)
        holder = QVBoxLayout()
        holder.setContentsMargins(0, 8, 0, 0)
        holder.addWidget(mark)
        holder.addStretch(1)
        row.addLayout(holder)
        text = label(issue.text, "muted", wrap=True)
        text.setMinimumHeight(32)
        row.addWidget(text, 1)
        self.button = None
        if issue.action_text:
            self.button = QuietButton(issue.action_text, "sm", accent=True)
            self.button.clicked.connect(lambda: on_action(issue))
            row.addWidget(self.button, 0, Qt.AlignTop)


class PreviewPanel(Rethemed, QFrame):
    previousRequested = Signal()
    nextRequested = Signal()
    issueActionRequested = Signal(object)
    adjustmentChanged = Signal(int)         # 사람 id
    applyToAllRequested = Signal(str, object)   # 사진 항목 이름, PhotoAdjustment

    WIDTH = 340

    def __init__(self, workspace, parent: QWidget | None = None):
        super().__init__(parent)
        self._workspace = workspace
        self._record: dict | None = None
        self._index = 0
        self._side = "front"
        self._drag_base: PhotoAdjustment | None = None
        self._live: PhotoAdjustment | None = None
        self._issues: list[Issue] = []
        self.setFixedWidth(self.WIDTH)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        body = QWidget()
        body.setObjectName("ScrollBody")
        scroll.setWidget(body)
        outer.addWidget(scroll)
        column = QVBoxLayout(body)
        column.setContentsMargins(20, 18, 20, SPACE["lg"])
        column.setSpacing(SPACE["md"])

        head = QHBoxLayout()
        head.setSpacing(SPACE["sm"])
        head.addWidget(label("이렇게 나옵니다", "title"))
        head.addStretch(1)
        self.side_control = SegmentedControl([("front", "앞면"), ("back", "뒷면")], current="front")
        self.side_control.currentChanged.connect(self.set_side)
        head.addWidget(self.side_control)
        column.addLayout(head)

        self.card = CardView(self)
        self.card.set_view_width(self.WIDTH - 2 * 20 - 12)
        self.card.dragStarted.connect(self._drag_start)
        self.card.dragged.connect(self._drag_move)
        self.card.dragFinished.connect(self._drag_end)
        column.addWidget(self.card, 0, Qt.AlignHCenter)

        nav = QHBoxLayout()
        nav.setSpacing(SPACE["md"])
        nav.addStretch(1)
        self.prev_button = IconButton("chevron-left", "이전 사람", variant="secondary")
        self.next_button = IconButton("chevron-right", "다음 사람", variant="secondary")
        self.position_label = label("", "secondary")
        self.position_label.setAlignment(Qt.AlignCenter)
        self.position_label.setMinimumWidth(96)
        self.prev_button.clicked.connect(self.previousRequested)
        self.next_button.clicked.connect(self.nextRequested)
        nav.addWidget(self.prev_button)
        nav.addWidget(self.position_label)
        nav.addWidget(self.next_button)
        nav.addStretch(1)
        column.addLayout(nav)

        self.adjust_toggle = QuietButton("사진 크기와 위치 맞추기", "sm", icon="image", accent=True)
        self.adjust_toggle.setCheckable(True)
        self.adjust_toggle.toggled.connect(self._toggle_adjust)
        toggle_row = QHBoxLayout()
        toggle_row.addStretch(1)
        toggle_row.addWidget(self.adjust_toggle)
        toggle_row.addStretch(1)
        column.addLayout(toggle_row)

        self.adjust_box = QWidget(self)
        adjust = QVBoxLayout(self.adjust_box)
        adjust.setContentsMargins(0, 0, 0, 0)
        adjust.setSpacing(SPACE["sm"])
        zoom_row = QHBoxLayout()
        zoom_row.setSpacing(SPACE["md"])
        zoom_row.addWidget(label("크기", "label"))
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(int(ZOOM_MIN * 100), int(ZOOM_MAX * 100))
        self.zoom_slider.setPageStep(10)
        self.zoom_slider.setAccessibleName("사진 크기")
        self.zoom_slider.setFixedHeight(28)
        self.zoom_slider.valueChanged.connect(self._zoom_changed)
        self.zoom_slider.sliderReleased.connect(self._commit_live)
        zoom_row.addWidget(self.zoom_slider, 1)
        self.zoom_value = label("100%", "muted")
        self.zoom_value.setFixedWidth(44)
        self.zoom_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        zoom_row.addWidget(self.zoom_value)
        adjust.addLayout(zoom_row)
        adjust.addWidget(label("카드에서 점선 안의 사진을 끌면 위치가 바뀝니다.", "hint", wrap=True))
        buttons = QHBoxLayout()
        buttons.setSpacing(SPACE["sm"])
        self.reset_button = QuietButton("처음대로", "sm")
        self.reset_button.clicked.connect(self.reset_adjustment)
        self.apply_all_button = QuietButton("모든 사람에게 같은 조정 적용", "sm", accent=True)
        self.apply_all_button.clicked.connect(self._apply_all)
        buttons.addWidget(self.reset_button)
        buttons.addStretch(1)
        buttons.addWidget(self.apply_all_button)
        adjust.addLayout(buttons)
        self.adjust_box.setVisible(False)
        column.addWidget(self.adjust_box)

        column.addWidget(Divider())
        column.addWidget(label("확인할 것", "strong"))
        self.issue_box = QVBoxLayout()
        self.issue_box.setSpacing(2)
        column.addLayout(self.issue_box)
        column.addStretch(1)

        self.follow_theme()
        self.set_position(-1, 0)
        self.set_issues([])
        self._sync_sides()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), theme_color("surface"))
        painter.setPen(QPen(theme_color("line"), 1))
        painter.drawLine(0, 0, 0, self.height())
        painter.end()

    # ---- 상태 ----
    @property
    def document(self):
        return self._workspace.editor.document

    @property
    def store(self) -> DataStore | None:
        return self._workspace.store

    def side(self) -> str:
        return self._side

    def set_side(self, side: str) -> None:
        if side != self._side:
            self._side = side
            self.side_control.set_current(side, emit=False)
            self.refresh()

    def _sync_sides(self) -> None:
        keys = list(self.document.active_side_keys())
        self.side_control.setVisible(len(keys) > 1)
        if self._side not in keys:
            self._side = keys[0] if keys else "front"
            if self._side in self.side_control.keys():
                self.side_control.set_current(self._side, emit=False)

    def set_position(self, row: int, total: int) -> None:
        self.position_label.setText(f"{row + 1:,} / {total:,}" if row >= 0 and total else "고른 사람 없음")
        self.prev_button.setEnabled(row > 0)
        self.next_button.setEnabled(total > 0 and row < total - 1)

    def set_record(self, record: dict | None, index: int = 0) -> None:
        self._record = record
        self._index = index
        self._live = None
        self.refresh()

    def record(self) -> dict | None:
        return self._record

    # ---- 그리기 ----
    def _adjust_item(self):
        """지금 사람에게서 맞출 수 있는 사진 칸(사진 파일이 있을 때만)."""
        if self._record is None or self.store is None:
            return None, None
        binding = self._workspace.binding_record(self._record) or {}
        for item in photo_items(self.document, self._side, self.store):
            path = binding.get(item.source.column)
            image = renderer.load_image(path) if path else None
            if image is not None:
                return item, image
        return None, None

    def refresh(self) -> None:
        self._sync_sides()
        document = self.document
        record = self._record
        binding = self._workspace.binding_record(record) if record is not None else None
        item, image = self._adjust_item()
        overrides = {}
        if item is not None and self._live is not None:
            overrides[item.source.column] = self._live
        width = round(300 * max(1.0, self.devicePixelRatioF()) * 2)
        try:
            rendered = render_card_image(
                document, self._side, binding, store=self.store,
                record_id=record["_id"] if record is not None else None,
                index=self._index, width_px=width, adjustments=overrides)
        except Exception:
            rendered = None
        self.card.set_image(rendered)

        can_adjust = item is not None
        self.adjust_toggle.setVisible(can_adjust or self.adjust_toggle.isChecked())
        self.adjust_toggle.setEnabled(can_adjust)
        self.adjust_box.setEnabled(can_adjust)
        show = can_adjust and self.adjust_toggle.isChecked()
        if show:
            bleed = float(getattr(document, "bleed_mm", 0.0) or 0.0)
            width_mm, height_mm = document.side_size_mm(self._side)
            w, h = max(1.0, width_mm - 2 * bleed), max(1.0, height_mm - 2 * bleed)
            self.card.set_photo_frame(((item.x - bleed) / w, (item.y - bleed) / h, item.w / w, item.h / h))
        else:
            self.card.set_photo_frame(None)
        current = self.current_adjustment()
        self.zoom_slider.blockSignals(True)
        self.zoom_slider.setValue(round(current.zoom * 100))
        self.zoom_slider.blockSignals(False)
        self.zoom_value.setText(f"{round(current.zoom * 100)}%")
        self.reset_button.setEnabled(can_adjust and current != PhotoAdjustment())

    def _toggle_adjust(self, on: bool) -> None:
        self.adjust_box.setVisible(on)
        self.adjust_toggle.setText("사진 맞추기 닫기" if on else "사진 크기와 위치 맞추기")
        self.refresh()

    # ---- 사진 맞추기 ----
    def current_adjustment(self) -> PhotoAdjustment:
        if self._live is not None:
            return self._live
        item, _ = self._adjust_item()
        if item is None or self._record is None or self.store is None:
            return PhotoAdjustment()
        return self.store.get_photo_adjustment(self._record["_id"], item.source.column)

    def set_adjustment(self, adjustment: PhotoAdjustment, commit: bool = True) -> None:
        item, image = self._adjust_item()
        if item is None:
            return
        self._live = clamp_adjustment(adjustment, (image.width(), image.height()), (item.w, item.h), item.fit)
        if commit:
            self._commit_live()
        else:
            self.refresh()

    def _commit_live(self) -> None:
        item, _ = self._adjust_item()
        if item is None or self._live is None or self._record is None or self.store is None:
            return
        adjustment, self._live = self._live, None
        record_id = self._record["_id"]
        if adjustment == PhotoAdjustment():
            self.store.clear_photo_adjustments(item.source.column, [record_id])
        else:
            self.store.set_photo_adjustment(record_id, item.source.column, adjustment)
        self.refresh()
        self.adjustmentChanged.emit(record_id)

    def reset_adjustment(self) -> None:
        self.set_adjustment(PhotoAdjustment())

    def _zoom_changed(self, value: int) -> None:
        current = self.current_adjustment()
        self.set_adjustment(PhotoAdjustment(value / 100.0, current.offset_x, current.offset_y),
                            commit=not self.zoom_slider.isSliderDown())

    def _drag_start(self) -> None:
        self._drag_base = self.current_adjustment()

    def _drag_move(self, dx: float, dy: float) -> None:
        base = self._drag_base
        if base is None:
            return
        self.set_adjustment(PhotoAdjustment(base.zoom, base.offset_x + dx, base.offset_y + dy), commit=False)

    def _drag_end(self) -> None:
        self._drag_base = None
        self._commit_live()

    def _apply_all(self) -> None:
        item, _ = self._adjust_item()
        if item is not None:
            self.applyToAllRequested.emit(item.source.column, self.current_adjustment())

    # ---- 확인할 것 ----
    def issues(self) -> list[Issue]:
        return list(self._issues)

    def set_issues(self, issues: list[Issue]) -> None:
        self._issues = list(issues)
        while self.issue_box.count():
            child = self.issue_box.takeAt(0)
            if child.widget() is not None:
                child.widget().hide()
                child.widget().deleteLater()
        shown = self._issues or [Issue("ok", "확인할 것이 없습니다. 이대로 인쇄해도 됩니다.", tone="ok")]
        self.issue_rows = []
        for issue in shown:
            row = _IssueRow(issue, self.issueActionRequested.emit, self)
            self.issue_rows.append(row)
            self.issue_box.addWidget(row)
