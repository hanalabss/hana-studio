"""
PropertyPanel - 오른쪽 속성 편집 패널
선택된 레이어의 속성 편집
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QGroupBox, QSpinBox, QDoubleSpinBox,
    QPushButton, QHBoxLayout, QComboBox, QColorDialog, QFontComboBox,
    QSpacerItem, QSizePolicy, QSlider, QCheckBox
)
from PySide6.QtCore import Signal, Qt, Slot
from PySide6.QtGui import QFont, QColor


class PropertyPanel(QWidget):
    """속성 편집 패널"""

    # 시그널
    position_changed = Signal(float, float)  # x, y
    rotation_changed = Signal(float)
    size_changed = Signal(float, float)  # width, height

    # 텍스트 속성
    text_changed = Signal(str)
    font_changed = Signal(QFont)
    font_size_changed = Signal(int)
    color_changed = Signal(QColor)

    # 인쇄
    print_requested = Signal()

    # 캔버스 맞추기
    fit_to_canvas_requested = Signal()

    # 방향 변경
    orientation_changed = Signal(str)

    # 인쇄 모드 변경
    print_mode_changed = Signal(str)  # "normal" 또는 "layered"

    # 뒷면 미리보기 업데이트 요청
    back_preview_update_requested = Signal()

    def __init__(self):
        super().__init__()
        self.setObjectName("right_panel")
        self.setFixedWidth(300)
        self.current_layer = None
        self._setup_ui()

    def _setup_ui(self):
        """UI 설정"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # 타이틀
        title = QLabel("속성")
        title.setObjectName("title_label")
        title.setFont(QFont("맑은 고딕", 14, QFont.Weight.Bold))
        layout.addWidget(title)

        # 선택 정보
        self.selection_label = QLabel("선택된 요소: 없음")
        self.selection_label.setWordWrap(True)
        layout.addWidget(self.selection_label)

        # 위치 그룹
        self.position_group = QGroupBox("📍 위치")
        position_layout = QVBoxLayout(self.position_group)
        position_layout.setSpacing(8)

        # X 위치
        x_layout = QHBoxLayout()
        x_layout.addWidget(QLabel("X:"))
        self.x_spinbox = QDoubleSpinBox()
        self.x_spinbox.setRange(-1000, 1000)
        self.x_spinbox.setSingleStep(1)
        self.x_spinbox.valueChanged.connect(self._on_position_changed)
        x_layout.addWidget(self.x_spinbox)
        position_layout.addLayout(x_layout)

        # Y 위치
        y_layout = QHBoxLayout()
        y_layout.addWidget(QLabel("Y:"))
        self.y_spinbox = QDoubleSpinBox()
        self.y_spinbox.setRange(-1000, 1000)
        self.y_spinbox.setSingleStep(1)
        self.y_spinbox.valueChanged.connect(self._on_position_changed)
        y_layout.addWidget(self.y_spinbox)
        position_layout.addLayout(y_layout)

        layout.addWidget(self.position_group)
        self.position_group.setVisible(False)

        # 이미지 속성 그룹
        self.image_group = QGroupBox("🖼️ 이미지 속성")
        image_layout = QVBoxLayout(self.image_group)
        image_layout.setSpacing(8)

        # 반전 버튼
        flip_layout = QHBoxLayout()
        self.flip_h_btn = QPushButton("↔️ 좌우반전")
        self.flip_h_btn.clicked.connect(self._on_flip_horizontal)
        flip_layout.addWidget(self.flip_h_btn)

        self.flip_v_btn = QPushButton("↕️ 상하반전")
        self.flip_v_btn.clicked.connect(self._on_flip_vertical)
        flip_layout.addWidget(self.flip_v_btn)
        image_layout.addLayout(flip_layout)

        # 투명도 슬라이더
        opacity_layout = QHBoxLayout()
        opacity_layout.addWidget(QLabel("투명도:"))
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(100)
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        opacity_layout.addWidget(self.opacity_slider)
        self.opacity_label = QLabel("100%")
        self.opacity_label.setMinimumWidth(45)
        opacity_layout.addWidget(self.opacity_label)
        image_layout.addLayout(opacity_layout)

        # 비율 고정 체크박스
        self.aspect_ratio_check = QCheckBox("비율 고정")
        self.aspect_ratio_check.setChecked(True)
        self.aspect_ratio_check.stateChanged.connect(self._on_aspect_ratio_changed)
        image_layout.addWidget(self.aspect_ratio_check)

        # 캔버스에 맞추기 버튼
        self.fit_to_canvas_btn = QPushButton("📐 캔버스에 맞추기")
        self.fit_to_canvas_btn.clicked.connect(self._on_fit_to_canvas)
        image_layout.addWidget(self.fit_to_canvas_btn)

        layout.addWidget(self.image_group)
        self.image_group.setVisible(False)

        # 텍스트 속성 그룹
        self.text_group = QGroupBox("✏️ 텍스트 속성")
        text_layout = QVBoxLayout(self.text_group)
        text_layout.setSpacing(8)

        # 폰트
        font_layout = QHBoxLayout()
        font_layout.addWidget(QLabel("폰트:"))
        self.font_combo = QFontComboBox()
        self.font_combo.currentFontChanged.connect(self._on_font_changed)
        font_layout.addWidget(self.font_combo)
        text_layout.addLayout(font_layout)

        # 폰트 크기
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("크기:"))
        self.font_size_spinbox = QSpinBox()
        self.font_size_spinbox.setRange(8, 200)
        self.font_size_spinbox.setValue(24)
        self.font_size_spinbox.valueChanged.connect(self._on_font_size_changed)
        size_layout.addWidget(self.font_size_spinbox)
        text_layout.addLayout(size_layout)

        # 색상
        color_layout = QHBoxLayout()
        color_layout.addWidget(QLabel("색상:"))
        self.color_btn = QPushButton("색상 선택")
        self.color_btn.clicked.connect(self._on_color_clicked)
        color_layout.addWidget(self.color_btn)
        text_layout.addLayout(color_layout)

        # 스타일 버튼
        style_layout = QHBoxLayout()
        self.bold_check = QCheckBox("B")
        self.bold_check.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        self.bold_check.setMaximumWidth(40)
        self.bold_check.stateChanged.connect(self._on_bold_changed)
        style_layout.addWidget(self.bold_check)

        self.italic_check = QCheckBox("I")
        italic_font = QFont("Arial", 10)
        italic_font.setItalic(True)
        self.italic_check.setFont(italic_font)
        self.italic_check.setMaximumWidth(40)
        self.italic_check.stateChanged.connect(self._on_italic_changed)
        style_layout.addWidget(self.italic_check)

        self.underline_check = QCheckBox("U")
        underline_font = QFont("Arial", 10)
        underline_font.setUnderline(True)
        self.underline_check.setFont(underline_font)
        self.underline_check.setMaximumWidth(40)
        self.underline_check.stateChanged.connect(self._on_underline_changed)
        style_layout.addWidget(self.underline_check)

        style_layout.addStretch()
        text_layout.addLayout(style_layout)

        # 정렬 버튼
        align_layout = QHBoxLayout()
        align_layout.addWidget(QLabel("정렬:"))

        self.align_left_btn = QPushButton("◀")
        self.align_left_btn.setMaximumWidth(40)
        self.align_left_btn.clicked.connect(lambda: self._on_alignment_changed(Qt.AlignmentFlag.AlignLeft))
        align_layout.addWidget(self.align_left_btn)

        self.align_center_btn = QPushButton("▣")
        self.align_center_btn.setMaximumWidth(40)
        self.align_center_btn.clicked.connect(lambda: self._on_alignment_changed(Qt.AlignmentFlag.AlignCenter))
        align_layout.addWidget(self.align_center_btn)

        self.align_right_btn = QPushButton("▶")
        self.align_right_btn.setMaximumWidth(40)
        self.align_right_btn.clicked.connect(lambda: self._on_alignment_changed(Qt.AlignmentFlag.AlignRight))
        align_layout.addWidget(self.align_right_btn)

        align_layout.addStretch()
        text_layout.addLayout(align_layout)

        layout.addWidget(self.text_group)
        self.text_group.setVisible(False)

        # 스페이서
        layout.addItem(QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))

        # 인쇄 그룹
        print_group = QGroupBox("🖨️ 인쇄")
        print_layout = QVBoxLayout(print_group)
        print_layout.setSpacing(8)

        # 프린터 상태 표시
        self.printer_status_label = QLabel("프린터: 초기화 중...")
        self.printer_status_label.setStyleSheet("padding: 4px; background-color: #f0f0f0; border-radius: 4px;")
        print_layout.addWidget(self.printer_status_label)

        # 인쇄 매수 설정
        quantity_layout = QHBoxLayout()
        quantity_layout.addWidget(QLabel("매수:"))
        self.print_quantity_spinbox = QSpinBox()
        self.print_quantity_spinbox.setRange(1, 100)
        self.print_quantity_spinbox.setValue(1)
        quantity_layout.addWidget(self.print_quantity_spinbox)
        print_layout.addLayout(quantity_layout)

        # 인쇄 방향 선택
        orientation_layout = QHBoxLayout()
        orientation_layout.addWidget(QLabel("방향:"))

        self.orientation_combo = QComboBox()
        self.orientation_combo.addItem("가로 (Landscape)", "landscape")
        self.orientation_combo.addItem("세로 (Portrait)", "portrait")
        self.orientation_combo.setCurrentIndex(1)  # 기본값: 세로 (캔버스 기본값과 일치)
        self.orientation_combo.currentIndexChanged.connect(self._on_orientation_changed)
        orientation_layout.addWidget(self.orientation_combo)
        print_layout.addLayout(orientation_layout)

        # 인쇄 모드 선택
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("모드:"))

        self.print_mode_combo = QComboBox()
        self.print_mode_combo.addItem("일반 인쇄 (Normal)", "normal")
        self.print_mode_combo.addItem("레이아웃 인쇄 (Layered)", "layered")
        self.print_mode_combo.currentIndexChanged.connect(self._on_print_mode_changed)
        mode_layout.addWidget(self.print_mode_combo)
        print_layout.addLayout(mode_layout)

        # 양면인쇄 체크박스
        self.dual_side_checkbox = QCheckBox("양면 인쇄")
        self.dual_side_checkbox.setChecked(False)
        print_layout.addWidget(self.dual_side_checkbox)

        self.print_btn = QPushButton("인쇄하기")
        self.print_btn.setObjectName("primary_btn")
        self.print_btn.setMinimumHeight(45)
        self.print_btn.clicked.connect(self._on_print_clicked)
        print_layout.addWidget(self.print_btn)

        layout.addWidget(print_group)

    def set_layer(self, layer):
        """선택된 레이어 설정"""
        # 이전 레이어 시그널 연결 해제
        if self.current_layer is not None:
            try:
                self.current_layer.signals.transform_changed.disconnect(
                    self._on_layer_transform_changed
                )
            except (RuntimeError, TypeError):
                pass  # 연결 안 되어 있거나 이미 해제됨

        self.current_layer = layer

        if layer is None:
            self.selection_label.setText("선택된 요소: 없음")
            self.position_group.setVisible(False)
            self.image_group.setVisible(False)
            self.text_group.setVisible(False)
            return

        # 레이어 타입 표시
        layer_type = layer.__class__.__name__.replace("Layer", "")
        self.selection_label.setText(f"선택된 요소: {layer_type}")

        # 위치 그룹 표시
        self.position_group.setVisible(True)
        self.x_spinbox.blockSignals(True)
        self.y_spinbox.blockSignals(True)
        self.x_spinbox.setValue(layer.pos().x())
        self.y_spinbox.setValue(layer.pos().y())
        self.x_spinbox.blockSignals(False)
        self.y_spinbox.blockSignals(False)

        # 이미지 레이어인 경우
        from canvas.layers import ImageLayer, TextLayer
        if isinstance(layer, ImageLayer):
            self.image_group.setVisible(True)
            self.text_group.setVisible(False)

            # 투명도
            self.opacity_slider.blockSignals(True)
            opacity_percent = int(layer.get_opacity() * 100)
            self.opacity_slider.setValue(opacity_percent)
            self.opacity_label.setText(f"{opacity_percent}%")
            self.opacity_slider.blockSignals(False)

            # 비율 고정
            self.aspect_ratio_check.blockSignals(True)
            self.aspect_ratio_check.setChecked(layer.keep_aspect_ratio)
            self.aspect_ratio_check.blockSignals(False)

            # 레이어 변형 시 속성 패널 갱신
            layer.signals.transform_changed.connect(self._on_layer_transform_changed)

        # 텍스트 레이어인 경우
        elif isinstance(layer, TextLayer):
            self.text_group.setVisible(True)
            self.image_group.setVisible(False)

            # 폰트 설정
            self.font_combo.blockSignals(True)
            self.font_combo.setCurrentFont(layer.get_font())
            self.font_combo.blockSignals(False)

            # 폰트 크기
            self.font_size_spinbox.blockSignals(True)
            self.font_size_spinbox.setValue(layer.get_font_size())
            self.font_size_spinbox.blockSignals(False)

            # 스타일
            self.bold_check.blockSignals(True)
            self.bold_check.setChecked(layer.get_bold())
            self.bold_check.blockSignals(False)

            self.italic_check.blockSignals(True)
            self.italic_check.setChecked(layer.get_italic())
            self.italic_check.blockSignals(False)

            self.underline_check.blockSignals(True)
            self.underline_check.setChecked(layer.get_underline())
            self.underline_check.blockSignals(False)

            # 레이어 변형 시 속성 패널 갱신
            layer.signals.transform_changed.connect(self._on_layer_transform_changed)
        else:
            self.text_group.setVisible(False)
            self.image_group.setVisible(False)

    def _on_layer_transform_changed(self):
        """레이어 변형 시 속성 패널 갱신"""
        if not self.current_layer:
            return

        # 위치 갱신
        self.x_spinbox.blockSignals(True)
        self.y_spinbox.blockSignals(True)
        self.x_spinbox.setValue(self.current_layer.pos().x())
        self.y_spinbox.setValue(self.current_layer.pos().y())
        self.x_spinbox.blockSignals(False)
        self.y_spinbox.blockSignals(False)

        # 텍스트 레이어인 경우 폰트 크기 등 갱신
        from canvas.layers import TextLayer
        if isinstance(self.current_layer, TextLayer):
            self.font_size_spinbox.blockSignals(True)
            self.font_size_spinbox.setValue(self.current_layer.get_font_size())
            self.font_size_spinbox.blockSignals(False)

            self.bold_check.blockSignals(True)
            self.bold_check.setChecked(self.current_layer.get_bold())
            self.bold_check.blockSignals(False)

            self.italic_check.blockSignals(True)
            self.italic_check.setChecked(self.current_layer.get_italic())
            self.italic_check.blockSignals(False)

    def _on_position_changed(self):
        """위치 변경"""
        if self.current_layer:
            x = self.x_spinbox.value()
            y = self.y_spinbox.value()
            self.current_layer.setPos(x, y)
            self.position_changed.emit(x, y)

    def _on_font_changed(self, font):
        """폰트 변경"""
        if self.current_layer:
            from canvas.layers import TextLayer
            if isinstance(self.current_layer, TextLayer):
                new_font = self.current_layer.get_font()
                new_font.setFamily(font.family())
                self.current_layer.set_font(new_font)
                self.font_changed.emit(new_font)

    def _on_font_size_changed(self, size):
        """폰트 크기 변경"""
        if self.current_layer:
            from canvas.layers import TextLayer
            if isinstance(self.current_layer, TextLayer):
                self.current_layer.set_font_size(size)
                self.font_size_changed.emit(size)

    def _on_color_clicked(self):
        """색상 선택"""
        if self.current_layer:
            from canvas.layers import TextLayer
            if isinstance(self.current_layer, TextLayer):
                current_color = self.current_layer.get_color()
                color = QColorDialog.getColor(current_color, self, "텍스트 색상 선택")

                if color.isValid():
                    self.current_layer.set_color(color)
                    self.color_changed.emit(color)

    def _on_print_clicked(self):
        """인쇄 버튼 클릭"""
        self.print_requested.emit()

    @Slot()
    def _on_flip_horizontal(self):
        """좌우 반전"""
        if self.current_layer:
            from canvas.layers import ImageLayer
            if isinstance(self.current_layer, ImageLayer):
                self.current_layer.flip_horizontal()

    @Slot()
    def _on_flip_vertical(self):
        """상하 반전"""
        if self.current_layer:
            from canvas.layers import ImageLayer
            if isinstance(self.current_layer, ImageLayer):
                self.current_layer.flip_vertical()

    @Slot(int)
    def _on_opacity_changed(self, value):
        """투명도 변경"""
        if self.current_layer:
            from canvas.layers import ImageLayer
            if isinstance(self.current_layer, ImageLayer):
                opacity = value / 100.0
                self.current_layer.set_opacity(opacity)
                self.opacity_label.setText(f"{value}%")

    @Slot(int)
    def _on_aspect_ratio_changed(self, state):
        """비율 고정 변경"""
        if self.current_layer:
            from canvas.layers import ImageLayer
            if isinstance(self.current_layer, ImageLayer):
                self.current_layer.keep_aspect_ratio = (state == Qt.CheckState.Checked.value)

    @Slot(int)
    def _on_bold_changed(self, state):
        """굵기 변경"""
        if self.current_layer:
            from canvas.layers import TextLayer
            if isinstance(self.current_layer, TextLayer):
                self.current_layer.set_bold(state == Qt.CheckState.Checked.value)

    @Slot(int)
    def _on_italic_changed(self, state):
        """이탤릭 변경"""
        if self.current_layer:
            from canvas.layers import TextLayer
            if isinstance(self.current_layer, TextLayer):
                self.current_layer.set_italic(state == Qt.CheckState.Checked.value)

    @Slot(int)
    def _on_underline_changed(self, state):
        """밑줄 변경"""
        if self.current_layer:
            from canvas.layers import TextLayer
            if isinstance(self.current_layer, TextLayer):
                self.current_layer.set_underline(state == Qt.CheckState.Checked.value)

    @Slot()
    def _on_alignment_changed(self, alignment):
        """정렬 변경"""
        if self.current_layer:
            from canvas.layers import TextLayer
            if isinstance(self.current_layer, TextLayer):
                # 세로 정렬은 유지하고 가로 정렬만 변경
                vertical_align = Qt.AlignmentFlag.AlignTop
                self.current_layer.set_alignment(alignment | vertical_align)

    @Slot()
    def _on_fit_to_canvas(self):
        """캔버스에 맞추기"""
        self.fit_to_canvas_requested.emit()

    @Slot()
    def _on_orientation_changed(self):
        """방향 변경"""
        orientation = self.orientation_combo.currentData()
        self.orientation_changed.emit(orientation)

    @Slot()
    def _on_print_mode_changed(self):
        """인쇄 모드 변경"""
        mode = self.print_mode_combo.currentData()
        self.print_mode_changed.emit(mode)

    def get_print_orientation(self):
        """인쇄 방향 가져오기"""
        return self.orientation_combo.currentData()

    def get_print_mode(self):
        """인쇄 모드 가져오기"""
        return self.print_mode_combo.currentData()

    def set_print_mode(self, mode: str):
        """인쇄 모드 설정

        Args:
            mode: "normal" (일반 인쇄) 또는 "layered" (레이아웃 인쇄)
        """
        if mode not in ["normal", "layered"]:
            print(f"[WARNING] 잘못된 인쇄 모드: {mode}, 기본값 'normal' 사용")
            mode = "normal"

        # 콤보박스에서 해당 모드의 인덱스 찾기
        for i in range(self.print_mode_combo.count()):
            if self.print_mode_combo.itemData(i) == mode:
                self.print_mode_combo.setCurrentIndex(i)
                mode_text = "일반 인쇄" if mode == "normal" else "레이아웃 인쇄"
                print(f"[UI] 인쇄 모드 변경: {mode} ({mode_text})")
                break

    def get_is_dual_side(self):
        """양면인쇄 여부 가져오기"""
        return self.dual_side_checkbox.isChecked()

    def get_print_quantity(self):
        """인쇄 매수 가져오기"""
        return self.print_quantity_spinbox.value()

    def update_printer_status(self, status: str):
        """프린터 상태 업데이트

        Args:
            status: 프린터 상태 텍스트
        """
        self.printer_status_label.setText(status)

        # 상태에 따라 색상 변경
        if "준비됨" in status or "선택됨" in status:
            self.printer_status_label.setStyleSheet("padding: 4px; background-color: #d4edda; border-radius: 4px; color: #155724;")
        elif "초기화 중" in status:
            self.printer_status_label.setStyleSheet("padding: 4px; background-color: #fff3cd; border-radius: 4px; color: #856404;")
        elif "없음" in status or "실패" in status or "검색 안됨" in status:
            self.printer_status_label.setStyleSheet("padding: 4px; background-color: #f8d7da; border-radius: 4px; color: #721c24;")
        else:
            self.printer_status_label.setStyleSheet("padding: 4px; background-color: #f0f0f0; border-radius: 4px;")
