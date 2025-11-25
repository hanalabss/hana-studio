"""
LayerPanel - 레이어 관리 패널
레이어 목록, 순서 변경, 가시성/잠금 토글
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QListWidget, QListWidgetItem,
    QHBoxLayout, QPushButton, QLabel, QGroupBox
)
from PySide6.QtCore import Signal, Qt, Slot
from PySide6.QtGui import QFont, QIcon


class LayerListItem(QWidget):
    """레이어 목록 아이템"""

    visibility_changed = Signal(object, bool)  # layer, visible
    lock_changed = Signal(object, bool)  # layer, locked

    def __init__(self, layer, parent=None):
        super().__init__(parent)
        self.layer = layer

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # 레이어 이름
        layer_type = layer.__class__.__name__.replace("Layer", "")
        if hasattr(layer, 'text_content'):
            name = f"{layer_type}: {layer.text_content[:20]}"
        elif hasattr(layer, 'image_path'):
            import os
            name = f"{layer_type}: {os.path.basename(layer.image_path) if layer.image_path else 'Image'}"
        else:
            name = layer_type

        self.name_label = QLabel(name)
        layout.addWidget(self.name_label, 1)

        # 가시성 토글
        self.visible_btn = QPushButton("👁")
        self.visible_btn.setMaximumWidth(30)
        self.visible_btn.setCheckable(True)
        self.visible_btn.setChecked(True)
        self.visible_btn.clicked.connect(self._on_visibility_clicked)
        layout.addWidget(self.visible_btn)

        # 잠금 토글
        self.lock_btn = QPushButton("🔓")
        self.lock_btn.setMaximumWidth(30)
        self.lock_btn.setCheckable(True)
        self.lock_btn.setChecked(False)
        self.lock_btn.clicked.connect(self._on_lock_clicked)
        layout.addWidget(self.lock_btn)

    @Slot()
    def _on_visibility_clicked(self):
        """가시성 토글"""
        visible = self.visible_btn.isChecked()
        self.visible_btn.setText("👁" if visible else "👁‍🗨")
        self.visibility_changed.emit(self.layer, visible)

    @Slot()
    def _on_lock_clicked(self):
        """잠금 토글"""
        locked = self.lock_btn.isChecked()
        self.lock_btn.setText("🔒" if locked else "🔓")
        self.lock_changed.emit(self.layer, locked)


class LayerPanel(QWidget):
    """레이어 관리 패널"""

    layer_selected = Signal(object)  # 선택된 레이어
    layer_order_changed = Signal()  # 레이어 순서 변경
    layer_visibility_changed = Signal(object, bool)  # layer, visible
    layer_lock_changed = Signal(object, bool)  # layer, locked

    def __init__(self):
        super().__init__()
        self.setObjectName("layer_panel")
        self.setFixedWidth(250)
        self._setup_ui()

    def _setup_ui(self):
        """UI 설정"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # 타이틀
        title = QLabel("레이어")
        title.setObjectName("title_label")
        title.setFont(QFont("맑은 고딕", 14, QFont.Weight.Bold))
        layout.addWidget(title)

        # 레이어 목록 그룹
        layers_group = QGroupBox("📚 레이어 목록")
        layers_layout = QVBoxLayout(layers_group)
        layers_layout.setSpacing(8)

        # 레이어 리스트
        self.layer_list = QListWidget()
        self.layer_list.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self.layer_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.layer_list.currentItemChanged.connect(self._on_layer_selected)
        self.layer_list.model().rowsMoved.connect(self._on_rows_moved)
        layers_layout.addWidget(self.layer_list)

        layout.addWidget(layers_group)

        # 레이어 관리 버튼
        buttons_layout = QHBoxLayout()

        self.move_up_btn = QPushButton("▲")
        self.move_up_btn.setMaximumWidth(40)
        self.move_up_btn.clicked.connect(self.move_layer_up)
        buttons_layout.addWidget(self.move_up_btn)

        self.move_down_btn = QPushButton("▼")
        self.move_down_btn.setMaximumWidth(40)
        self.move_down_btn.clicked.connect(self.move_layer_down)
        buttons_layout.addWidget(self.move_down_btn)

        buttons_layout.addStretch()

        layout.addLayout(buttons_layout)

        layout.addStretch()

    def update_layers(self, scene):
        """씬의 레이어 목록 업데이트"""
        self.layer_list.clear()

        from canvas.layers import ImageLayer, TextLayer

        # 씬의 모든 레이어 가져오기 (역순으로 - 위에 있는 것부터)
        layers = [item for item in scene.items() if isinstance(item, (ImageLayer, TextLayer))]

        for layer in reversed(layers):
            # 리스트 아이템 생성
            item = QListWidgetItem(self.layer_list)
            widget = LayerListItem(layer)

            # 시그널 연결
            widget.visibility_changed.connect(self._on_visibility_changed)
            widget.lock_changed.connect(self._on_lock_changed)

            # 아이템 크기 설정
            item.setSizeHint(widget.sizeHint())

            # 위젯 설정
            self.layer_list.setItemWidget(item, widget)

    @Slot()
    def _on_layer_selected(self, current, previous):
        """레이어 선택"""
        if current:
            widget = self.layer_list.itemWidget(current)
            if widget:
                self.layer_selected.emit(widget.layer)

    @Slot()
    def _on_rows_moved(self, parent, start, end, destination, row):
        """레이어 순서 변경 (드래그 앤 드롭)"""
        self.layer_order_changed.emit()

    @Slot(object, bool)
    def _on_visibility_changed(self, layer, visible):
        """가시성 변경"""
        layer.setVisible(visible)
        self.layer_visibility_changed.emit(layer, visible)

    @Slot(object, bool)
    def _on_lock_changed(self, layer, locked):
        """잠금 변경"""
        layer.setFlag(layer.GraphicsItemFlag.ItemIsMovable, not locked)
        layer.setFlag(layer.GraphicsItemFlag.ItemIsSelectable, not locked)
        self.layer_lock_changed.emit(layer, locked)

    @Slot()
    def move_layer_up(self):
        """레이어 위로 이동"""
        current_row = self.layer_list.currentRow()
        if current_row > 0:
            item = self.layer_list.takeItem(current_row)
            self.layer_list.insertItem(current_row - 1, item)
            self.layer_list.setCurrentRow(current_row - 1)
            self.layer_order_changed.emit()

    @Slot()
    def move_layer_down(self):
        """레이어 아래로 이동"""
        current_row = self.layer_list.currentRow()
        if current_row < self.layer_list.count() - 1 and current_row >= 0:
            item = self.layer_list.takeItem(current_row)
            self.layer_list.insertItem(current_row + 1, item)
            self.layer_list.setCurrentRow(current_row + 1)
            self.layer_order_changed.emit()

    def set_selected_layer(self, layer):
        """외부에서 레이어 선택"""
        for i in range(self.layer_list.count()):
            item = self.layer_list.item(i)
            widget = self.layer_list.itemWidget(item)
            if widget and widget.layer == layer:
                self.layer_list.setCurrentRow(i)
                break
