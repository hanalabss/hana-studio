"""
ui/components/control_panels.py 수정
전역 카드 방향 선택 제거, 기존 요소들 재배치
"""

import re
from PySide6.QtWidgets import (
    QSizePolicy, QVBoxLayout, QHBoxLayout, QGroupBox, 
    QLabel, QRadioButton, QButtonGroup, QProgressBar, QTextEdit, QCheckBox,
    QDoubleSpinBox,QSpinBox,QFrame
)
from PySide6.QtCore import Signal,Qt
from .modern_button import ModernButton


def truncate_text(text: str, max_length: int = 30) -> str:
    """텍스트가 너무 길면 줄임 처리"""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."


class PositionAdjustPanel(QGroupBox):
    """카드 위치 미세조정 패널 - 크기 및 여백 최적화"""
    position_changed = Signal(float, float)  # x, y 조정값 (float)
    
    def __init__(self):
        super().__init__("📐 인쇄 위치 조정")
        # 기본값 설정: x 0, y 0
        self.adjusted_x = 0
        self.adjusted_y = 0
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(25)  # 8 → 12로 증가 (세로 간격)
        layout.setContentsMargins(8, 16, 8, 12)  # 상단 여백 12 → 16으로 증가
        
        # X축 조정
        x_layout = QHBoxLayout()
        x_layout.setSpacing(5)
        x_layout.setContentsMargins(0, 0, 0, 0)
        
        x_label = QLabel("[X] X:")
        x_label.setFixedWidth(32)
        x_label.setStyleSheet("""
            QLabel {
                background: transparent;
                border: none;
                font-size: 12px;
                font-weight: 600;
                color: #495057;
            }
        """)
        
        # X축 감소 버튼
        self.x_minus_btn = ModernButton("-")
        self.x_minus_btn.setFixedSize(24, 24)
        self.x_minus_btn.setStyleSheet("""
            QPushButton {
                background-color: #DC3545;
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #C82333;
            }
            QPushButton:pressed {
                background-color: #A71E2A;
            }
        """)
        self.x_minus_btn.clicked.connect(lambda: self._adjust_x(-0.01))
        
        # X축 DoubleSpinBox
        self.x_spinbox = QDoubleSpinBox()
        self.x_spinbox.setRange(-50.0, 50.0)
        self.x_spinbox.setValue(0)
        self.x_spinbox.setSingleStep(0.01)
        self.x_spinbox.setDecimals(2)
        self.x_spinbox.setFixedSize(62, 24)
        self.x_spinbox.setStyleSheet("""
            QDoubleSpinBox {
                background-color: #FFFFFF;
                border: 1px solid #DEE2E6;
                border-radius: 4px;
                padding: 2px 4px;
                font-size: 11px;
                color: #495057;
                font-weight: 600;
            }
            QDoubleSpinBox:focus {
                border-color: #4A90E2;
                border-width: 2px;
            }
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
                width: 14px;
                border: none;
                background-color: #F8F9FA;
            }
            QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {
                background-color: #E9ECEF;
            }
        """)
        
        # X축 증가 버튼
        self.x_plus_btn = ModernButton("+")
        self.x_plus_btn.setFixedSize(24, 24)
        self.x_plus_btn.setStyleSheet("""
            QPushButton {
                background-color: #28A745;
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #218838;
            }
            QPushButton:pressed {
                background-color: #1E7E34;
            }
        """)
        self.x_plus_btn.clicked.connect(lambda: self._adjust_x(0.01))
        
        # 단위 라벨
        x_unit_label = QLabel("mm")
        x_unit_label.setFixedWidth(20)
        x_unit_label.setStyleSheet("""
            QLabel {
                background: transparent;
                border: none;
                font-size: 10px;
                color: #6C757D;
                font-weight: 500;
            }
        """)
        
        x_layout.addWidget(x_label)
        x_layout.addWidget(self.x_minus_btn)
        x_layout.addWidget(self.x_spinbox)
        x_layout.addWidget(self.x_plus_btn)
        x_layout.addWidget(x_unit_label)
        x_layout.addStretch()
        
        # Y축 조정
        y_layout = QHBoxLayout()
        y_layout.setSpacing(5)
        y_layout.setContentsMargins(0, 0, 0, 0)
        
        y_label = QLabel("[Y] Y:")
        y_label.setFixedWidth(32)
        y_label.setStyleSheet("""
            QLabel {
                background: transparent;
                border: none;
                font-size: 12px;
                font-weight: 600;
                color: #495057;
            }
        """)
        
        # Y축 감소 버튼
        self.y_minus_btn = ModernButton("-")
        self.y_minus_btn.setFixedSize(24, 24)
        self.y_minus_btn.setStyleSheet("""
            QPushButton {
                background-color: #DC3545;
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #C82333;
            }
            QPushButton:pressed {
                background-color: #A71E2A;
            }
        """)
        self.y_minus_btn.clicked.connect(lambda: self._adjust_y(-0.01))
        
        # Y축 DoubleSpinBox
        self.y_spinbox = QDoubleSpinBox()
        self.y_spinbox.setRange(-50.0, 50.0)
        self.y_spinbox.setValue(0)
        self.y_spinbox.setSingleStep(0.01)
        self.y_spinbox.setDecimals(2)
        self.y_spinbox.setFixedSize(62, 24)
        self.y_spinbox.setStyleSheet("""
            QDoubleSpinBox {
                background-color: #FFFFFF;
                border: 1px solid #DEE2E6;
                border-radius: 4px;
                padding: 2px 4px;
                font-size: 11px;
                color: #495057;
                font-weight: 600;
            }
            QDoubleSpinBox:focus {
                border-color: #28A745;
                border-width: 2px;
            }
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
                width: 14px;
                border: none;
                background-color: #F8F9FA;
            }
            QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {
                background-color: #E9ECEF;
            }
        """)
        
        # Y축 증가 버튼
        self.y_plus_btn = ModernButton("+")
        self.y_plus_btn.setFixedSize(24, 24)
        self.y_plus_btn.setStyleSheet("""
            QPushButton {
                background-color: #28A745;
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #218838;
            }
            QPushButton:pressed {
                background-color: #1E7E34;
            }
        """)
        self.y_plus_btn.clicked.connect(lambda: self._adjust_y(0.01))
        
        # 단위 라벨
        y_unit_label = QLabel("mm")
        y_unit_label.setFixedWidth(20)
        y_unit_label.setStyleSheet("""
            QLabel {
                background: transparent;
                border: none;
                font-size: 10px;
                color: #6C757D;
                font-weight: 500;
            }
        """)
        
        y_layout.addWidget(y_label)
        y_layout.addWidget(self.y_minus_btn)
        y_layout.addWidget(self.y_spinbox)
        y_layout.addWidget(self.y_plus_btn)
        y_layout.addWidget(y_unit_label)
        y_layout.addStretch()
        
        # 리셋 버튼 - 여백 추가
        reset_container = QHBoxLayout()
        reset_container.setContentsMargins(0, 8, 0, 0)  # 상단 여백 4 → 8로 증가
        
        self.reset_btn = ModernButton("초기화")
        self.reset_btn.setFixedHeight(42)
        self.reset_btn.setFixedWidth(150)
        self.reset_btn.setStyleSheet("""
        QPushButton {
            background-color: #4A90E2;
            color: white;
            border: none;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
            padding: 2px 8px;
        }
        QPushButton:hover {
            background-color: #357ABD;
        }
        QPushButton:pressed {
            background-color: #2A5F98;
        }
        """)
        self.reset_btn.clicked.connect(self._reset_position)
        
        reset_container.addStretch()
        reset_container.addWidget(self.reset_btn)
        reset_container.addStretch()
        
        # 레이아웃에 추가 - 여백 조정
        layout.addLayout(x_layout)
        layout.addSpacing(8)  # X, Y축 사이 추가 여백 4 → 8로 증가
        layout.addLayout(y_layout)
        layout.addSpacing(10)  # Y축과 리셋 버튼 사이 여백 6 → 10으로 증가
        layout.addLayout(reset_container)
        layout.addStretch()
        
        # 시그널 연결
        self.x_spinbox.valueChanged.connect(self._on_x_changed)
        self.y_spinbox.valueChanged.connect(self._on_y_changed)
        
        # 초기값 설정 및 시그널 발송
        self._update_initial_values()
    
    def _update_initial_values(self):
        """초기값 설정 및 시그널 발송"""
        self.adjusted_x = self.x_spinbox.value()
        self.adjusted_y = self.y_spinbox.value()
        self.position_changed.emit(self.adjusted_x, self.adjusted_y)
    
    def _adjust_x(self, delta):
        """X축 값 조정"""
        current_value = self.x_spinbox.value()
        new_value = current_value + delta
        new_value = max(-50.0, min(50.0, new_value))  # 범위 제한
        self.x_spinbox.setValue(new_value)
    
    def _adjust_y(self, delta):
        """Y축 값 조정"""
        current_value = self.y_spinbox.value()
        new_value = current_value + delta
        new_value = max(-50.0, min(50.0, new_value))  # 범위 제한
        self.y_spinbox.setValue(new_value)
    
    def _on_x_changed(self, value):
        """X축 값 변경"""
        self.adjusted_x = value
        self.position_changed.emit(self.adjusted_x, self.adjusted_y)
    
    def _on_y_changed(self, value):
        """Y축 값 변경"""
        self.adjusted_y = value
        self.position_changed.emit(self.adjusted_x, self.adjusted_y)
    
    def _reset_position(self):
        """위치 초기화 - 기본값으로 복원"""
        self.x_spinbox.setValue(0)  # 0mm
        self.y_spinbox.setValue(0)  # 0mm
    
    def get_position(self):
        """현재 위치 조정값 반환 (float)"""
        return self.adjusted_x, self.adjusted_y
    
    def set_position(self, x, y):
        """위치 조정값 설정 (float)"""
        self.x_spinbox.setValue(x)
        self.y_spinbox.setValue(y)
          
class FileSelectionPanel(QGroupBox):
    """파일 선택 패널"""
    file_selected = Signal(str)  # 파일 경로 시그널
    
    def __init__(self):
        super().__init__("📁 파일 선택")
        # 내용에 맞춰 크기 조정
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(5)
        layout.setContentsMargins(6, 6, 6, 6)
        
        # 앞면 이미지 선택
        self.front_btn = ModernButton("앞면 이미지 선택", primary=True)
        self.front_btn.setFixedHeight(35)
        
        self.front_label = QLabel("앞면: 선택된 파일이 없습니다")
        self.front_label.setStyleSheet("""
                font-size: 11px; 
                font-weight: bold;
                color: #6C757D;
                background-color: #F8F9FA;
                border-radius: 2px;
                padding: 4px;
        """)
        self.front_label.setWordWrap(True)
        self.front_label.setMaximumHeight(30)
        
        # 뒷면 이미지 선택
        self.back_btn = ModernButton("뒷면 이미지 선택", primary=False)
        self.back_btn.setFixedHeight(35)
        
        self.back_label = QLabel("뒷면: 선택된 파일이 없습니다")
        self.back_label.setStyleSheet("""
                font-size: 11px; 
                font-weight: bold;
                color: #6C757D;
                background-color: #F8F9FA;
                border-radius: 2px;
                padding: 4px;
            """)
        self.back_label.setWordWrap(True)
        self.back_label.setMaximumHeight(30)
        
        layout.addWidget(self.front_btn)
        layout.addWidget(self.front_label)
        layout.addWidget(self.back_btn)
        layout.addWidget(self.back_label)
        
        # 초기에는 뒷면 비활성화 (인쇄모드 패널에서 제어됨)
        self.back_btn.setEnabled(False)
    
    def set_dual_side_enabled(self, enabled: bool):
        """양면 모드 활성화/비활성화 (외부에서 호출)"""
        self.back_btn.setEnabled(enabled)
        if enabled:
            self.back_btn.set_primary(True)
        else:
            self.back_btn.set_primary(False)
            self.back_label.setText("뒷면: 선택된 파일이 없습니다")
    
    def update_front_file_info(self, file_path: str):
        """앞면 파일 정보 업데이트"""
        import os
        filename = os.path.basename(file_path)
        filename = truncate_text(filename, 25)
        self.front_label.setText(f"앞면: 📁 {filename}")
    
    def update_back_file_info(self, file_path: str):
        """뒷면 파일 정보 업데이트"""
        import os
        filename = os.path.basename(file_path)
        filename = truncate_text(filename, 25)
        self.back_label.setText(f"뒷면: 📁 {filename}")


class PrintModePanel(QGroupBox):
    """인쇄 모드 패널 - 전역 카드 방향 제거"""
    mode_changed = Signal(str)
    dual_side_changed = Signal(bool)
    
    def __init__(self):
        super().__init__("📋 인쇄 설정")
        self.print_mode = "normal"
        self.is_dual_side = False
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(8, 8, 8, 8)
        
        # 양면 인쇄 체크박스
        self.dual_side_check = QCheckBox("양면 인쇄 사용")
        self.dual_side_check.setStyleSheet("""
            QCheckBox {
                font-size: 13px;
                font-weight: 600;
                color: #495057;
                padding: 8px;
                spacing: 8px;
                background-color: #F8F9FA;
                border: 1px solid #E9ECEF;
                border-radius: 6px;
            }
            QCheckBox:hover {
                background-color: #E9ECEF;
                border-color: #4A90E2;
                color: #4A90E2;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
            QCheckBox::indicator:unchecked {
                border: 2px solid #6C757D;
                border-radius: 3px;
                background-color: white;
            }
            QCheckBox::indicator:checked {
                border: 2px solid #4A90E2;
                border-radius: 3px;
                background-color: #4A90E2;
                image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTIiIGhlaWdodD0iMTIiIHZpZXdCb3g9IjAgMCAxMiAxMiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEwIDNMNC41IDguNUwyIDYiIHN0cm9rZT0id2hpdGUiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIi8+Cjwvc3ZnPgo=);
            }
        """)
        self.dual_side_check.toggled.connect(self._on_dual_side_toggled)
        
        # 인쇄 모드 섹션
        mode_frame = QFrame()
        mode_frame.setStyleSheet("""
            QFrame {
                background-color: #F8F9FA;
                border: 1px solid #E9ECEF;
                border-radius: 6px;
                padding: 6px;
            }
        """)
        mode_layout = QVBoxLayout(mode_frame)
        mode_layout.setSpacing(8)  # 간격 더 증가
        mode_layout.setContentsMargins(8, 8, 8, 8)  # 여백 더 증가
        
        # 모드 라벨
        mode_label = QLabel("인쇄 모드:")
        mode_label.setStyleSheet("""
            QLabel {
                background: transparent;
                border: none;
                padding: 0px;
                font-size: 12px;
                font-weight: 600;
                color: #495057;
                margin-bottom: 4px;
            }
        """)
        
        # 라디오 버튼들
        self.normal_radio = QRadioButton("[PRINTER] 일반")
        self.layered_radio = QRadioButton("🎭 레이어(YMCW)")
        self.normal_radio.setChecked(True)
        
        # 라디오 버튼 스타일 - 크기 증가
        radio_style = """
            QRadioButton {
                font-size: 12px;
                font-weight: 500;
                color: #495057;
                padding: 5px 8px;
                spacing: 8px;
                background: transparent;
                border-radius: 4px;
            }
            QRadioButton:hover {
                background-color: rgba(74, 144, 226, 0.1);
                color: #4A90E2;
            }
            QRadioButton::indicator {
                width: 14px;
                height: 14px;
            }
            QRadioButton::indicator:unchecked {
                border: 2px solid #6C757D;
                border-radius: 7px;
                background-color: white;
            }
            QRadioButton::indicator:checked {
                border: 2px solid #4A90E2;
                border-radius: 7px;
                background-color: #4A90E2;
            }
        """
        
        self.normal_radio.setStyleSheet(radio_style)
        self.layered_radio.setStyleSheet(radio_style)
        
        # 인쇄 모드 버튼 그룹
        self.mode_group = QButtonGroup()
        self.mode_group.addButton(self.normal_radio, 0)
        self.mode_group.addButton(self.layered_radio, 1)
        
        # 모드 프레임에 위젯 추가
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.normal_radio)
        mode_layout.addWidget(self.layered_radio)
        
        # 메인 레이아웃에 위젯 추가
        layout.addWidget(self.dual_side_check)
        layout.addWidget(mode_frame, 2)  # 더 많은 공간 할당
        layout.addStretch()  # 남은 공간을 아래쪽으로 밀어냄
        
        # 신호 연결
        self.normal_radio.toggled.connect(self._on_mode_changed)
    
    def _on_dual_side_toggled(self, checked):
        """양면 인쇄 토글"""
        self.is_dual_side = checked
        # self._update_description()  # 주석 처리
        self.dual_side_changed.emit(checked)
    
    def _on_mode_changed(self):
        """인쇄 모드 변경"""
        if self.normal_radio.isChecked():
            self.print_mode = "normal"
        else:
            self.print_mode = "layered"
        
        # self._update_description()  # 주석 처리
        self.mode_changed.emit(self.print_mode)
    
    # def _update_description(self):
    #     """설명 업데이트 - 주석 처리됨"""
    #     side_text = "양면" if self.is_dual_side else "단면"
    #     mode_text = "레이어" if self.print_mode == "layered" else "일반"
    #     
    #     if self.print_mode == "normal":
    #         description = f"📖 원본 이미지를 {side_text} {mode_text} 인쇄합니다"
    #     else:
    #         description = f"📖 마스크 워터마크를 포함하여 {side_text} {mode_text} 인쇄합니다"
    #     
    #     self.description_label.setText(description)


class PrintQuantityPanel(QGroupBox):
    """인쇄 매수 선택 패널 - 통일된 컨트롤 버튼 스타일"""
    quantity_changed = Signal(int)

    def __init__(self):
        super().__init__("[DATA] 인쇄 매수")
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

        # 예상 시간 계산용 상태
        self._is_dual_side = False
        self._print_mode = "normal"

        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(8, 16, 8, 12)
        
        # 매수 선택 영역 - 위치 조정 패널과 동일한 스타일
        quantity_layout = QHBoxLayout()
        quantity_layout.setSpacing(5)
        quantity_layout.setContentsMargins(0, 0, 0, 0)
        
        quantity_label = QLabel("📄 매수:")
        quantity_label.setFixedWidth(52)
        quantity_label.setStyleSheet("""
            QLabel {
                background: transparent;
                border: none;
                font-size: 12px;
                font-weight: 600;
                color: #495057;
            }
        """)
        
        # 감소 버튼 - 위치 조정과 동일한 스타일
        self.minus_btn = ModernButton("-")
        self.minus_btn.setFixedSize(24, 24)
        self.minus_btn.setStyleSheet("""
            QPushButton {
                background-color: #DC3545;
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #C82333;
            }
            QPushButton:pressed {
                background-color: #A71E2A;
            }
        """)
        
        # 매수 SpinBox - 위치 조정과 동일한 스타일
        self.quantity_spinbox = QSpinBox()
        self.quantity_spinbox.setMinimum(1)
        self.quantity_spinbox.setMaximum(99999)
        self.quantity_spinbox.setValue(1)
        self.quantity_spinbox.setFixedSize(62, 24)
        self.quantity_spinbox.setStyleSheet("""
            QSpinBox {
                background-color: #FFFFFF;
                border: 1px solid #DEE2E6;
                border-radius: 4px;
                padding: 2px 4px;
                font-size: 11px;
                color: #495057;
                font-weight: 600;
            }
            QSpinBox:focus {
                border-color: #4A90E2;
                border-width: 2px;
            }
            QSpinBox::up-button, QSpinBox::down-button {
                width: 14px;
                border: none;
                background-color: #F8F9FA;
            }
            QSpinBox::up-button:hover, QSpinBox::down-button:hover {
                background-color: #E9ECEF;
            }
        """)
        
        # 증가 버튼 - 위치 조정과 동일한 스타일
        self.plus_btn = ModernButton("+")
        self.plus_btn.setFixedSize(24, 24)
        self.plus_btn.setStyleSheet("""
            QPushButton {
                background-color: #28A745;
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #218838;
            }
            QPushButton:pressed {
                background-color: #1E7E34;
            }
        """)
        
        # 단위 라벨
        unit_label = QLabel("장")
        unit_label.setFixedWidth(20)
        unit_label.setStyleSheet("""
            QLabel {
                background: transparent;
                border: none;
                font-size: 10px;
                color: #6C757D;
                font-weight: 500;
            }
        """)
        
        quantity_layout.addWidget(quantity_label)
        quantity_layout.addWidget(self.minus_btn)
        quantity_layout.addWidget(self.quantity_spinbox)
        quantity_layout.addWidget(self.plus_btn)
        quantity_layout.addWidget(unit_label)
        quantity_layout.addStretch()
        
        # 예상 시간 표시 (동적 계산)
        from printer.print_time_tracker import print_time_tracker
        initial_time = print_time_tracker.get_estimated_time_formatted(
            quantity=1,
            is_duplex=self._is_dual_side,
            print_mode=self._print_mode
        )
        self.time_estimate_label = QLabel(f"[TIME] 예상 시간: {initial_time}")
        self.time_estimate_label.setStyleSheet("""
            color: #6C757D; 
            font-size: 13px;
            padding: 6px;
            background-color: #F8F9FA;
            border-left: 3px solid #28A745;
            border-radius: 4px;
        """)
        self.time_estimate_label.setWordWrap(True)
        
        layout.addLayout(quantity_layout)
        layout.addSpacing(8)  # 간격 추가
        layout.addWidget(self.time_estimate_label)
        layout.addStretch()
        
        # 시그널 연결
        self.quantity_spinbox.valueChanged.connect(self._on_quantity_changed)
        self.minus_btn.clicked.connect(self._decrease_quantity)
        self.plus_btn.clicked.connect(self._increase_quantity)
    
    def _decrease_quantity(self):
        """매수 감소"""
        current_value = self.quantity_spinbox.value()
        if current_value > 1:
            self.quantity_spinbox.setValue(current_value - 1)
    
    def _increase_quantity(self):
        """매수 증가"""
        current_value = self.quantity_spinbox.value()
        if current_value < 99999:
            self.quantity_spinbox.setValue(current_value + 1)
    
    def _on_quantity_changed(self, value):
        """매수 변경 시 예상 시간 업데이트"""
        self._update_time_estimate()
        self.quantity_changed.emit(value)

    def _update_time_estimate(self):
        """예상 시간 라벨 업데이트"""
        from printer.print_time_tracker import print_time_tracker
        quantity = self.quantity_spinbox.value()
        time_str = print_time_tracker.get_estimated_time_formatted(
            quantity=quantity,
            is_duplex=self._is_dual_side,
            print_mode=self._print_mode
        )
        self.time_estimate_label.setText(f"[TIME] 예상 시간: {time_str}")

    def set_print_settings(self, is_dual_side: bool = None, print_mode: str = None):
        """인쇄 설정 변경 시 예상 시간 업데이트"""
        if is_dual_side is not None:
            self._is_dual_side = is_dual_side
        if print_mode is not None:
            self._print_mode = print_mode
        self._update_time_estimate()
    
    def get_quantity(self) -> int:
        """선택된 매수 반환"""
        return self.quantity_spinbox.value()
    
    def set_quantity(self, quantity: int):
        """매수 설정"""
        self.quantity_spinbox.setValue(quantity)

class PrinterPanel(QGroupBox):
    """프린터 연동 패널 - 개별 면 방향 지원"""
    test_requested = Signal()
    print_requested = Signal()
    
    def __init__(self):
        super().__init__("[PRINTER] 프린터 연동")
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(5)
        layout.setContentsMargins(8, 8, 8, 8)
        
        # 상태 라벨
        self.printer_status_label = QLabel("프린터 상태 확인 중...")
        self.printer_status_label.setStyleSheet("""
            font-size: 14px; 
            color: #6C757D;
            padding: 4px;
            background-color: #F8F9FA;
            border-radius: 3px;
        """)
        self.printer_status_label.setWordWrap(True)
        
        # 버튼들
        self.test_printer_btn = ModernButton("프린터 연결 테스트")
        self.test_printer_btn.setFixedHeight(45)
        
        # 기본 버튼 텍스트 단순화
        self.print_card_btn = ModernButton("카드 인쇄", primary=True)
        self.print_card_btn.setEnabled(False)
        self.print_card_btn.setFixedHeight(50)
        
        layout.addWidget(self.printer_status_label)
        layout.addWidget(self.test_printer_btn)
        layout.addWidget(self.print_card_btn)
        
        # 신호 연결
        self.test_printer_btn.clicked.connect(self.test_requested.emit)
        self.print_card_btn.clicked.connect(self.print_requested.emit)
    
    def update_status(self, status: str):
        """프린터 상태 업데이트"""
        truncated_status = truncate_text(status, 40)
        self.printer_status_label.setText(truncated_status)
    
    def set_test_enabled(self, enabled: bool):
        """테스트 버튼 활성화/비활성화"""
        self.test_printer_btn.setEnabled(enabled)
    
    def set_print_enabled(self, enabled: bool):
        """인쇄 버튼 활성화/비활성화"""
        self.print_card_btn.setEnabled(enabled)
    
    def update_print_button_text(self, mode: str, is_dual: bool = False, quantity: int = 1, orientation: str = "portrait"):
        """인쇄 버튼 텍스트 업데이트 - 개별 면 방향이므로 단순화"""
        # 인쇄 모드 텍스트
        if mode == "normal":
            base_text = "양면 일반 인쇄" if is_dual else "단면 일반 인쇄"
        else:
            base_text = "양면 레이어 인쇄" if is_dual else "단면 레이어 인쇄"
        
        # 매수 추가
        if quantity > 1:
            text = f"{base_text} ({quantity}장)"
        else:
            text = base_text
        
        # 텍스트 길이 제한
        text = truncate_text(text, 20)
        self.print_card_btn.setText(text)


class ProgressPanel(QGroupBox):
    """진행 상황 패널"""
    
    def __init__(self):
        super().__init__("[DATA] 진행 상황")
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(8, 8, 8, 8)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(25)
        
        self.status_label = QLabel("[PAUSE] 대기 중...")
        self.status_label.setStyleSheet("""
            font-size: 14px; 
            color: #495057;
            padding: 4px;
            background-color: #F8F9FA;
            border-radius: 3px;
        """)
        self.status_label.setWordWrap(True)
        
        # 인쇄 진행상황 표시용 라벨
        self.print_progress_label = QLabel("")
        self.print_progress_label.setStyleSheet("""
            color: #4A90E2; 
            font-size: 14px; 
            font-weight: 600;
            padding: 4px;
            background-color: rgba(74, 144, 226, 0.1);
            border-radius: 3px;
        """)
        self.print_progress_label.setVisible(False)
        
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)
        layout.addWidget(self.print_progress_label)

    def _get_user_friendly_status(self, technical_status: str) -> str:
        """사용자 친화적 상태 메시지로 변환"""
        status = technical_status.lower()
        
        # 🎯 기술적 정보 제거 및 단순화 매핑
        simple_mappings = {
            # 이미지 처리 관련
            "배경 제거": "이미지 처리",
            "마스크 생성": "이미지 처리",
            "ai 모델": "이미지 처리",
            "임계값": "",
            
            # 인쇄 준비 관련  
            "앞면:": "",
            "뒷면:": "",
            "세로형": "",
            "가로형": "",
            "단면": "",
            "양면": "",
            "레이어": "",
            "일반": "",
            
            # 위치 조정 정보 제거
            "(위치조정:": "(",
            "x+": "",
            "x-": "",
            "y+": "",
            "y-": "",
            "mm": "",
        }
        
        # 기술적 용어 단순화
        result = technical_status
        for tech_term, simple_term in simple_mappings.items():
            result = result.replace(tech_term, simple_term)
        
        # 🎯 상태별 간단 메시지로 변환 (이모지 추가)
        if "처리 중" in status or "로딩" in status or "다운로드" in status:
            return "🔄 이미지 처리 중..."
        elif "인쇄 준비" in status:
            # 매수 정보만 추출
            quantity_match = re.search(r'(\d+)장', result)
            if quantity_match:
                return f"📋 카드 {quantity_match.group(1)}장 인쇄 준비"
            else:
                return "📋 카드 인쇄 준비"
        elif "인쇄 진행" in status or "인쇄 중" in status:
            return "[PRINTER] 카드 인쇄 중..."
        elif "인쇄 완료" in status or "완료" in status:
            return "[OK] 인쇄 완료!"
        elif "실패" in status or "오류" in status:
            return "❌ 작업 실패"
        elif "테스트" in status:
            if "성공" in status:
                return "[OK] 프린터 연결 확인됨"
            elif "실패" in status:
                return "❌ 프린터 연결 실패"
            else:
                return "[SEARCH] 프린터 확인 중..."
        elif "대기" in status:
            return "[PAUSE] 대기 중..."
        
        # 🎯 괄호와 상세 정보 제거
        result = re.sub(r'\(.*?\)', '', result)  # 모든 괄호 내용 제거
        result = re.sub(r'[,:].*$', '', result)  # 콤마나 콜론 이후 내용 제거
        result = result.strip()
        
        # 빈 문자열이면 기본값 반환
        if not result or len(result.strip()) < 3:
            return "[CONFIG] 작업 중..."
            
        return result[:30]  # 최대 30자 제한

    def show_progress(self, indeterminate=True):
        """진행바 표시"""
        if indeterminate:
            self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(True)
    
    def hide_progress(self):
        """진행바 숨기기"""
        self.progress_bar.setVisible(False)
        self.print_progress_label.setVisible(False)
    
    def update_status(self, status: str):
        """상태 메시지 업데이트 - 사용자 친화적으로 변환"""
        user_friendly_status = self._get_user_friendly_status(status)
        truncated_status = truncate_text(user_friendly_status, 30)
        self.status_label.setText(truncated_status)
    
    def show_print_progress(self, current: int, total: int):
        """인쇄 진행상황 표시 - 단순화"""
        progress_text = f"📄 {current}/{total} 장"
        self.print_progress_label.setText(progress_text)
        self.print_progress_label.setVisible(True)
        
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(current)
    
    def update_print_status(self, current: int, total: int, status: str):
        """인쇄 상태와 진행률 동시 업데이트 - 단순화"""
        self.update_status("[PRINTER] 카드 인쇄 중...")
        self.show_print_progress(current, total)


class LogPanel(QGroupBox):
    """로그 패널 - 숨김 처리"""
    
    def __init__(self):
        super().__init__("📝 처리 로그")
        # 로그 패널을 완전히 숨김
        self.setVisible(False)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.setMinimumHeight(0)
        self.setMaximumHeight(0)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setVisible(False)  # 텍스트 에디터도 숨김
        layout.addWidget(self.log_text)
    
    def add_log(self, message: str):
        """로그 메시지 추가 - 실제로는 콘솔에만 출력"""
        print(f"[LOG] {message}")
        # UI에는 표시하지 않음
    
    def clear_log(self):
        """로그 클리어 - 아무것도 하지 않음"""
        pass