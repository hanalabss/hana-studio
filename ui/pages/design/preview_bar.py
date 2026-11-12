"""캔버스 아래 줄: 명단의 사람을 넘겨 가며 카드를 미리 본다."""

from __future__ import annotations

from collections import OrderedDict

from PySide6.QtCore import QAbstractListModel, QModelIndex, QRectF, QSize, Qt, QTimer
from PySide6.QtGui import QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QFrame, QHBoxLayout, QLabel, QListView, QStyledItemDelegate, QVBoxLayout, QWidget,
)

from data.photos import find_missing
from design.binding import BindingContext, columns_used
from ui.theme import SPACE, ThemeManager
from ui.widgets import IconButton, QuietButton, label, theme_color

from .start_panel import card_pixmap
from .style import BAR_HEIGHT

NAME_COLUMNS = ("이름", "성명", "성함", "name", "Name", "NAME")
_PHOTO_SCAN_LIMIT = 20000
_THUMB_W, _THUMB_H = 150, 96
_CACHE_SIZE = 240
_NAME_LIMIT = 8
_TITLE_MIN_WIDTH = 600
_PLACEHOLDER_MIN_WIDTH = 520


def person_name(store, record: dict | None) -> str:
    if store is None or record is None:
        return ""
    columns = store.columns()
    chosen = next((c.name for c in columns if c.name in NAME_COLUMNS), None)
    if chosen is None:
        chosen = next((c.name for c in columns if c.type == "text"), None)
    value = record.get(chosen) if chosen else None
    return "" if value is None else str(value).strip()


class _ThumbModel(QAbstractListModel):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.ids: list[int] = []

    def set_ids(self, ids: list[int]) -> None:
        self.beginResetModel()
        self.ids = list(ids)
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.ids)

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid() and role == Qt.UserRole:
            return self.ids[index.row()]
        return None


class _ThumbDelegate(QStyledItemDelegate):
    def __init__(self, strip: "ThumbStrip") -> None:
        super().__init__(strip)
        self._strip = strip

    def sizeHint(self, option, index) -> QSize:
        return QSize(_THUMB_W + 10, _THUMB_H + 10)

    def paint(self, painter: QPainter, option, index) -> None:
        record_id = index.data(Qt.UserRole)
        rect = QRectF(option.rect).adjusted(5, 5, -5, -5)
        pixmap = self._strip.thumbnail(record_id)
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        current = record_id == self._strip.current_id()
        if pixmap is not None:
            painter.setOpacity(1.0 if current else 0.8)
            painter.drawPixmap(rect.topLeft(), pixmap)
            painter.setOpacity(1.0)
        else:
            painter.setPen(Qt.NoPen)
            painter.setBrush(theme_color("surface_hover"))
            painter.drawRoundedRect(rect.adjusted(8, 8, -8, -8), 6, 6)
        if current:
            painter.setPen(QPen(theme_color("accent"), 2))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(rect.adjusted(1, 1, -1, -1), 8, 8)
        painter.restore()


class ThumbStrip(QFrame):
    """사람별 카드 그림 줄. 화면에 보이는 것만 차례로 그린다."""

    def __init__(self, workspace, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("ThumbStrip")
        self.setFixedHeight(_THUMB_H + 24)
        self.workspace = workspace
        self._cache: OrderedDict[int, QPixmap] = OrderedDict()
        self._queue: list[int] = []
        layout = QHBoxLayout(self)
        layout.setContentsMargins(SPACE["sm"], 0, SPACE["sm"], 0)
        self.view = QListView(self)
        self.view.setObjectName("Thumbs")
        self.view.setFlow(QListView.LeftToRight)
        self.view.setWrapping(False)
        self.view.setUniformItemSizes(True)
        self.view.setSelectionMode(QAbstractItemView.NoSelection)
        self.view.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.view.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setAccessibleName("사람별 카드 미리 보기")
        self.model = _ThumbModel(self)
        self.view.setModel(self.model)
        self.view.setItemDelegate(_ThumbDelegate(self))
        self.view.clicked.connect(lambda index: workspace.set_current_record(index.data(Qt.UserRole)))
        layout.addWidget(self.view)

        self._render_timer = QTimer(self)
        self._render_timer.setInterval(0)
        self._render_timer.timeout.connect(self._render_next)
        self._stale_timer = QTimer(self)
        self._stale_timer.setSingleShot(True)
        self._stale_timer.setInterval(250)
        self._stale_timer.timeout.connect(self._drop_cache)

    def current_id(self):
        return self.workspace.current_record_id

    def set_ids(self, ids: list[int]) -> None:
        if ids != self.model.ids:
            self.model.set_ids(ids)
        self.invalidate()

    def invalidate(self) -> None:
        self._stale_timer.start()

    def _drop_cache(self) -> None:
        self._cache.clear()
        self._queue.clear()
        self.view.viewport().update()

    def rendered_count(self) -> int:
        return len(self._cache)

    def thumbnail(self, record_id: int) -> QPixmap | None:
        pixmap = self._cache.get(record_id)
        if pixmap is None:
            if record_id not in self._queue:
                self._queue.append(record_id)
            if not self._render_timer.isActive():
                self._render_timer.start()
        return pixmap

    def scroll_to_current(self) -> None:
        current = self.current_id()
        if current in self.model.ids:
            self.view.scrollTo(self.model.index(self.model.ids.index(current)), QAbstractItemView.EnsureVisible)
        self.view.viewport().update()

    def _render_next(self) -> None:
        store = self.workspace.store
        if not self._queue or store is None or not self.isVisible():
            self._queue.clear()
            self._render_timer.stop()
            return
        record_id = self._queue.pop()
        record = store.get(record_id)
        if record is None:
            return
        editor = self.workspace.editor
        try:
            pixmap = card_pixmap(
                editor.document, editor.current_side, self.workspace.binding_record(record),
                _THUMB_W, _THUMB_H, context=BindingContext(record_key=str(record_id)),
                dpr=self.devicePixelRatioF(), margin=3,
            )
        except Exception:
            pixmap = QPixmap(_THUMB_W, _THUMB_H)
            pixmap.fill(Qt.transparent)
        self._cache[record_id] = pixmap
        while len(self._cache) > _CACHE_SIZE:
            self._cache.popitem(last=False)
        if record_id in self.model.ids:
            self.view.update(self.model.index(self.model.ids.index(record_id)))


class PreviewBar(QWidget):
    def __init__(self, workspace, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.workspace = workspace
        self._ids: list[int] = []
        self._strip_allowed = False
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.strip = ThumbStrip(workspace, self)
        self.strip.setVisible(False)
        outer.addWidget(self.strip)

        bar = QFrame(self)
        bar.setObjectName("PreviewBar")
        bar.setFixedHeight(BAR_HEIGHT + 4)
        outer.addWidget(bar)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(SPACE["lg"], 0, SPACE["md"], 0)
        layout.setSpacing(SPACE["sm"])
        self.title_label = label("명단으로 미리 보기", "strong")
        layout.addWidget(self.title_label)
        layout.addSpacing(SPACE["xs"])

        self.prev_button = IconButton("chevron-left", "이전 사람", size="sm", variant="secondary")
        self.prev_button.clicked.connect(lambda: self.step(-1))
        layout.addWidget(self.prev_button)
        self.position_label = QLabel("")
        self.position_label.setObjectName("PreviewName")
        self.position_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.position_label)
        self.next_button = IconButton("chevron-right", "다음 사람", size="sm", variant="secondary")
        self.next_button.clicked.connect(lambda: self.step(1))
        layout.addWidget(self.next_button)
        self.placeholder_button = QuietButton("칸 이름 보기", size="sm")
        self.placeholder_button.setToolTip("사람 대신 «이름» 같은 칸 이름을 보여 줍니다")
        self.placeholder_button.clicked.connect(lambda: workspace.set_current_record(None))
        layout.addWidget(self.placeholder_button)

        self.empty_label = label("", "muted")
        layout.addWidget(self.empty_label)
        layout.addStretch(1)

        self.column_chip = QLabel("")
        self.column_chip.setObjectName("WarnChip")
        self.column_chip.setFixedHeight(28)
        layout.addWidget(self.column_chip)
        self.photo_chip = QLabel("")
        self.photo_chip.setObjectName("WarnChip")
        self.photo_chip.setFixedHeight(28)
        layout.addWidget(self.photo_chip)

        self._check_timer = QTimer(self)
        self._check_timer.setSingleShot(True)
        self._check_timer.setInterval(300)
        self._check_timer.timeout.connect(self.refresh_warnings)

        editor = workspace.editor
        workspace.store_changed.connect(self.reload)
        workspace.records_changed.connect(self._on_records_changed)
        view_changed = getattr(workspace, "view_changed", None)
        if view_changed is not None:
            view_changed.connect(self._on_view_changed)
        workspace.columns_changed.connect(self._schedule_check)
        workspace.current_record_changed.connect(self._on_current_changed)
        for signal in (editor.items_changed, editor.document_changed, editor.card_changed, editor.side_changed):
            signal.connect(self._on_design_changed)
        editor.properties_changed.connect(self._on_design_changed)
        ThemeManager.instance().themeChanged.connect(lambda *_: self.strip.invalidate())
        self.reload()

    # ---- 사람 넘기기 ----
    def record_ids(self) -> list[int]:
        return list(self._ids)

    def _query_ids(self) -> list[int]:
        """명단 화면에서 보는 조건과 순서 그대로."""
        store = self.workspace.store
        if store is None:
            return []
        try:
            return store.ids(self.workspace.view_filter, self.workspace.view_order)
        except Exception:
            return store.ids()

    def reload(self) -> None:
        self._ids = self._query_ids()
        self.strip.set_ids(self._ids)
        self._sync()
        self.refresh_warnings()

    def _on_records_changed(self) -> None:
        if not getattr(self.workspace, "store_opening", False):      # 명단을 열 때는 store_changed에서 읽었다
            self.reload()

    def _on_view_changed(self) -> None:
        """명단 화면에서 찾거나 줄 세운 대로 따라간다. 카드 그림은 사람마다 그대로라 다시 그리지 않는다."""
        ids = self._query_ids()
        if ids == self._ids:
            return
        self._ids = ids
        self.strip.model.set_ids(ids)
        self._sync()
        self.strip.scroll_to_current()

    def step(self, delta: int) -> None:
        if not self._ids:
            return
        current = self.workspace.current_record_id
        if current in self._ids:
            index = min(len(self._ids) - 1, max(0, self._ids.index(current) + delta))
        else:
            index = 0 if delta > 0 else len(self._ids) - 1
        self.workspace.set_current_record(self._ids[index])

    def _on_current_changed(self, _record) -> None:
        self._sync()
        self.strip.scroll_to_current()

    def _on_design_changed(self, *_args) -> None:
        self.strip.invalidate()
        self._schedule_check()

    def _schedule_check(self) -> None:
        self._check_timer.start()

    def set_strip_allowed(self, allowed: bool) -> None:
        self._strip_allowed = allowed
        self._sync_strip()

    def _sync_strip(self) -> None:
        self.strip.setVisible(self._strip_allowed and bool(self._ids))

    def _sync(self) -> None:
        store = self.workspace.store
        has_people = bool(self._ids)
        for widget in (self.prev_button, self.next_button, self.position_label):
            widget.setVisible(has_people)
        self.empty_label.setVisible(not has_people)
        current = self.workspace.current_record_id
        showing = has_people and current in self._ids
        self.placeholder_button.setVisible(showing and self.width() >= _PLACEHOLDER_MIN_WIDTH)
        self.title_label.setVisible(not has_people or self.width() >= _TITLE_MIN_WIDTH)
        if not has_people:
            self.empty_label.setText(
                "명단을 넣으면 사람마다 바뀐 카드를 여기서 넘겨 볼 수 있습니다." if store is None
                else "명단에 아직 사람이 없습니다.")
        elif showing:
            index = self._ids.index(current)
            name = person_name(store, self.workspace.current_record())
            self.position_label.setToolTip(name)
            if len(name) > _NAME_LIMIT:
                name = name[:_NAME_LIMIT - 1] + "…"
            self.position_label.setText(f"{index + 1} / {len(self._ids):,}" + (f"  {name}" if name else ""))
            self.prev_button.setEnabled(index > 0)
            self.next_button.setEnabled(index < len(self._ids) - 1)
        else:
            self.position_label.setText(f"{len(self._ids):,}명")
            self.position_label.setToolTip("")
            self.prev_button.setEnabled(False)
            self.next_button.setEnabled(True)
            self.next_button.setToolTip("첫 사람의 카드 보기")
        if showing:
            self.next_button.setToolTip("다음 사람")
        self._sync_strip()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._sync()

    # ---- 알림 칩 ----
    def refresh_warnings(self) -> None:
        store = self.workspace.store
        self.column_chip.setVisible(False)
        self.photo_chip.setVisible(False)
        if store is None:
            return
        used = columns_used(self.workspace.editor.document)
        columns = {c.name: c for c in store.columns()}
        missing_columns = sorted(used - set(columns))
        if missing_columns:
            shown = ", ".join(missing_columns[:3]) + (" 등" if len(missing_columns) > 3 else "")
            self.column_chip.setText(f"명단에 없는 항목: {shown}")
            self.column_chip.setToolTip("카드에서 쓰는 항목이 열린 명단에 없습니다. 명단의 항목 이름을 확인하세요.")
            self.column_chip.setVisible(True)
        photo_columns = [name for name in sorted(used) if name in columns and columns[name].type == "photo"]
        if not photo_columns or len(self._ids) > _PHOTO_SCAN_LIMIT:
            return
        people: set[int] = set()
        for name in photo_columns:
            people.update(record["_id"] for record in find_missing(store, name))
        if people:
            self.photo_chip.setText(f"사진 없는 사람 {len(people):,}명")
            self.photo_chip.setToolTip("사진 파일을 찾지 못했습니다. 명단 화면에서 사진 폴더와 파일 이름을 확인하세요.")
            self.photo_chip.setVisible(True)
