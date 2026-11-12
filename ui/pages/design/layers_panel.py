"""카드에 놓인 것 목록: 끌어서 순서 바꾸기, 숨기기, 잠그기, 이름 바꾸기."""

from __future__ import annotations

from PySide6.QtCore import QEvent, QRect, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import (
    QAbstractItemView, QListWidget, QListWidgetItem, QStyle, QStyledItemDelegate, QVBoxLayout, QWidget,
)

from ui.editor import KIND_LABELS, EditorController, LayerInfo
from ui.theme import SPACE, ThemeManager
from ui.widgets import label, theme_color

from . import icons

ROW_HEIGHT = 34
_INFO_ROLE = Qt.UserRole + 1
_BUTTON = 24


def _button_rects(rect: QRect) -> tuple[QRect, QRect]:
    """(잠금, 보이기) 버튼 자리."""
    top = rect.y() + (rect.height() - _BUTTON) // 2
    eye = QRect(rect.right() - _BUTTON - 3, top, _BUTTON, _BUTTON)
    lock = QRect(eye.x() - _BUTTON, top, _BUTTON, _BUTTON)
    return lock, eye


class _LayerDelegate(QStyledItemDelegate):
    def __init__(self, panel: "LayersPanel") -> None:
        super().__init__(panel)
        self._panel = panel

    def sizeHint(self, option, index) -> QSize:
        return QSize(80, ROW_HEIGHT)

    def paint(self, painter: QPainter, option, index) -> None:
        info: LayerInfo | None = index.data(_INFO_ROLE)
        if info is None:
            return
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        rect = QRectF(option.rect).adjusted(0, 1, 0, -1)
        selected = bool(option.state & QStyle.State_Selected)
        hovered = bool(option.state & QStyle.State_MouseOver)
        if selected or hovered:
            painter.setPen(Qt.NoPen)
            painter.setBrush(theme_color("accent_soft" if selected else "surface_hover"))
            painter.drawRoundedRect(rect, 6, 6)

        badge = QRectF(rect.x() + 6, rect.center().y() - 11, 22, 22)
        if info.bound:
            painter.setPen(Qt.NoPen)
            painter.setBrush(theme_color("field_bg"))
            painter.drawRoundedRect(badge, 6, 6)
            icon_role = "field_text"
        else:
            icon_role = "accent_text" if selected else "text_secondary"
        icons.paint(painter, icons.KIND_ICONS.get(info.kind, "rect"), badge.adjusted(3, 3, -3, -3),
                    icon_role if info.visible else "text_disabled")

        lock, eye = _button_rects(option.rect)
        right = lock.x() - 2
        if info.warning:
            painter.setPen(Qt.NoPen)
            painter.setBrush(theme_color("warn_mark"))
            painter.drawEllipse(QRectF(right - 10, rect.center().y() - 4, 8, 8))
            right -= 16
        if info.locked or hovered:
            icons.paint(painter, "lock" if info.locked else "unlock", QRectF(lock).adjusted(5, 5, -5, -5),
                        "text" if info.locked else "text_muted")
        if not info.visible or hovered:
            icons.paint(painter, "eye" if info.visible else "eye-off", QRectF(eye).adjusted(4, 4, -4, -4),
                        "text_muted")

        if self._panel.list.isPersistentEditorOpen(index) or self._panel.editing_row() == index.row():
            painter.restore()
            return
        text_rect = QRectF(badge.right() + 8, rect.y(), right - badge.right() - 12, rect.height())
        role = "text_disabled" if not info.visible else ("accent_text" if selected else "text")
        painter.setPen(theme_color(role))
        metrics = painter.fontMetrics()
        painter.drawText(text_rect, Qt.AlignVCenter | Qt.AlignLeft,
                         metrics.elidedText(info.name, Qt.ElideRight, int(text_rect.width())))
        painter.restore()

    def editorEvent(self, event, model, option, index) -> bool:
        if event.type() in (QEvent.MouseButtonPress, QEvent.MouseButtonRelease, QEvent.MouseButtonDblClick):
            info: LayerInfo | None = index.data(_INFO_ROLE)
            lock, eye = _button_rects(option.rect)
            pos = event.position().toPoint()
            target = "lock" if lock.contains(pos) else "eye" if eye.contains(pos) else None
            if info is not None and target and event.button() == Qt.LeftButton:
                if event.type() == QEvent.MouseButtonRelease:
                    if target == "lock":
                        self._panel.toggle_locked(info.id)
                    else:
                        self._panel.toggle_visible(info.id)
                return True
        return super().editorEvent(event, model, option, index)

    def updateEditorGeometry(self, editor, option, index) -> None:
        lock, _eye = _button_rects(option.rect)
        left = option.rect.x() + 32
        editor.setGeometry(QRect(left, option.rect.y() + 4, lock.x() - left - 2, option.rect.height() - 8))

    def setEditorData(self, editor, index) -> None:
        info: LayerInfo | None = index.data(_INFO_ROLE)
        editor.setText(info.name if info else "")
        editor.selectAll()

    def setModelData(self, editor, model, index) -> None:
        info: LayerInfo | None = index.data(_INFO_ROLE)
        if info is not None:
            self._panel.rename(info.id, editor.text())


class _LayerList(QListWidget):
    dropped = Signal()

    def dropEvent(self, event) -> None:
        super().dropEvent(event)
        self.dropped.emit()


class LayersPanel(QWidget):
    def __init__(self, controller: EditorController, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.controller = controller
        self._syncing = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(label("카드에 놓인 것", "strong"))
        layout.addSpacing(2)
        self.hint = label("", "hint", wrap=True)
        layout.addWidget(self.hint)
        layout.addSpacing(SPACE["xs"] + 2)

        self.list = _LayerList(self)
        self.list.setObjectName("Layers")
        self.list.setItemDelegate(_LayerDelegate(self))
        self.list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list.setDragDropMode(QAbstractItemView.InternalMove)
        self.list.setDefaultDropAction(Qt.MoveAction)
        self.list.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed)
        self.list.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.list.setMouseTracking(True)
        self.list.setUniformItemSizes(True)
        self.list.setMinimumHeight(ROW_HEIGHT * 3)
        self.list.setAccessibleName("카드에 놓인 것 목록")
        layout.addWidget(self.list, 1)

        self.list.itemSelectionChanged.connect(self._on_list_selection)
        self.list.dropped.connect(self._commit_order)
        controller.items_changed.connect(self.rebuild)
        controller.problems_changed.connect(self.rebuild)
        controller.selection_changed.connect(self._sync_selection)
        ThemeManager.instance().themeChanged.connect(lambda *_: self.list.viewport().update())
        self.rebuild()

    def set_compact(self, compact: bool) -> None:
        self.hint.setVisible(not compact)
        self.list.setToolTip("위에 있을수록 앞에 보입니다. 끌어서 순서를 바꿉니다." if compact else "")

    # ---- 목록 ----
    def editing_row(self) -> int:
        return self.list.currentRow() if self.list.state() == QAbstractItemView.EditingState else -1

    def ids(self) -> list[str]:
        return [self.list.item(row).data(_INFO_ROLE).id for row in range(self.list.count())]

    def info(self, item_id: str) -> LayerInfo | None:
        row = self._row_of(item_id)
        return self.list.item(row).data(_INFO_ROLE) if row >= 0 else None

    def _row_of(self, item_id: str) -> int:
        for row in range(self.list.count()):
            if self.list.item(row).data(_INFO_ROLE).id == item_id:
                return row
        return -1

    def rebuild(self) -> None:
        if self.list.state() == QAbstractItemView.EditingState:
            QTimer.singleShot(0, self.rebuild)
            return
        layers = self.controller.layers()
        self._syncing = True
        try:
            scroll = self.list.verticalScrollBar().value()
            self.list.clear()
            for info in layers:
                entry = QListWidgetItem(info.name)
                entry.setData(_INFO_ROLE, info)
                entry.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable | Qt.ItemIsDragEnabled)
                entry.setSizeHint(QSize(80, ROW_HEIGHT))
                tip = KIND_LABELS.get(info.kind, "")
                if info.bound:
                    tip += " · 명단에 따라 바뀜"
                if info.warning:
                    tip += " · 확인이 필요합니다"
                entry.setToolTip(f"{tip} (두 번 눌러 이름 바꾸기)")
                self.list.addItem(entry)
                entry.setSelected(info.selected)
            self.list.verticalScrollBar().setValue(scroll)
        finally:
            self._syncing = False
        if layers:
            self.hint.setText("위에 있을수록 앞에 보입니다. 끌어서 순서를 바꿉니다.")
        else:
            self.hint.setText("아직 아무것도 없습니다. 위의 버튼으로 넣어 보세요.")

    def _sync_selection(self, ids: list[str]) -> None:
        if self._syncing:
            return
        wanted = set(ids)
        self._syncing = True
        try:
            for row in range(self.list.count()):
                entry = self.list.item(row)
                entry.setSelected(entry.data(_INFO_ROLE).id in wanted)
        finally:
            self._syncing = False
        self.list.viewport().update()

    def _on_list_selection(self) -> None:
        if self._syncing:
            return
        self._syncing = True
        try:
            self.controller.select([entry.data(_INFO_ROLE).id for entry in self.list.selectedItems()])
        finally:
            self._syncing = False

    # ---- 동작 ----
    def _commit_order(self) -> None:
        order = self.ids()
        if order != [info.id for info in self.controller.layers()]:
            self.controller.set_layer_order(order)
        else:
            self.rebuild()

    def move_layer(self, item_id: str, to_row: int) -> None:
        """목록에서 한 줄을 to_row 자리로 옮긴다(끌어다 놓은 것과 같다)."""
        row = self._row_of(item_id)
        if row < 0:
            return
        self._syncing = True
        try:
            entry = self.list.takeItem(row)
            self.list.insertItem(max(0, min(to_row, self.list.count())), entry)
        finally:
            self._syncing = False
        self._commit_order()

    def toggle_visible(self, item_id: str) -> None:
        item = self.controller.item(item_id)
        if item is not None:
            self.controller.set_visible(item_id, not item.visible)

    def toggle_locked(self, item_id: str) -> None:
        item = self.controller.item(item_id)
        if item is not None:
            self.controller.set_locked(item_id, not item.locked)

    def rename(self, item_id: str, name: str) -> None:
        self.controller.rename(item_id, name.strip())
