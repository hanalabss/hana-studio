"""
ui/components/control_panels.py 수정
처리 옵션 패널에서 배경제거 버튼 제거 (이제 개별 이미지 뷰어에 있음)
"""

from PySide6.QtWidgets import (
    QSizePolicy, QVBoxLayout, QHBoxLayout, QGroupBox, 
    QLabel, QRadioButton, QButtonGroup, QProgressBar, QTextEdit, QCheckBox,
    QSpinBox
)
from PySide6.QtCore import Signal, Qt
from .modern_button import ModernButton


def truncate_text(text: str, max_length: int = 30) -> str:
    """텍스트가 너무 길면 줄임 처리"""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."


class FileSelectionPanel(QGroupBox):
    """파일 선택 패널 - 양면인쇄 체크박스 제거"""
    file_selected = Signal(str)  # 파일 경로 시그널
    
    def __init__(self):
        super().__init__("📁 파일 선택")
        # 내용에 맞춰 크기 조정
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(5)  # 간격 늘림 (3 → 5)
        layout.setContentsMargins(6, 6, 6, 6)
        
        # 앞면 이미지 선택
        self.front_btn = ModernButton("앞면 이미지 선택", primary=True)
        self.front_btn.setFixedHeight(35)  # 높이 증가 (30 → 35)
        
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
        self.front_label.setMaximumHeight(30)  # 높이 증가 (26 → 30)
        
        # 뒷면 이미지 선택
        self.back_btn = ModernButton("뒷면 이미지 선택", primary=False)
        self.back_btn.setFixedHeight(35)  # 높이 증가 (30 → 35)
        
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
        self.back_label.setMaximumHeight(30)  # 높이 증가 (26 → 30)
        
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


# ProcessingOptionsPanel 클래스 완전 삭제 - 더 이상 필요없음


class PrintModePanel(QGroupBox):
    """인쇄 모드 패널 - 양면인쇄 체크박스 추가"""
    mode_changed = Signal(str)  # "normal" 또는 "layered"
    dual_side_changed = Signal(bool)  # 양면인쇄 변경 시그널 추가
    
    def __init__(self):
        super().__init__("📋 인쇄 모드")
        self.print_mode = "normal"
        self.is_dual_side = False
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(6)  # 간격 늘림 (4 → 6)
        layout.setContentsMargins(8, 8, 8, 8)
        
        # 양면 인쇄 활성화 체크박스 (파일선택 패널에서 이동)
        self.dual_side_check = QCheckBox("양면 인쇄 사용")
        self.dual_side_check.setStyleSheet("""
            QCheckBox {
                font-size: 14px;
                font-weight: 600;
                color: #495057;
                spacing: 5px;
                padding: 4px;
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
            }
        """)
        self.dual_side_check.toggled.connect(self._on_dual_side_toggled)
        
        # 라디오 버튼들
        self.normal_radio = QRadioButton("일반 인쇄")
        self.layered_radio = QRadioButton("레이어 인쇄 (YMCW)")
        self.normal_radio.setChecked(True)
        
        # 라디오 버튼 스타일 조정
        radio_style = """
            QRadioButton {
                font-size: 15px;
                font-weight: 500;
                color: #495057;
                spacing: 5px;
                padding: 3px;
            }
        """
        self.normal_radio.setStyleSheet(radio_style)
        self.layered_radio.setStyleSheet(radio_style)
        
        # 버튼 그룹
        self.button_group = QButtonGroup()
        self.button_group.addButton(self.normal_radio, 0)
        self.button_group.addButton(self.layered_radio, 1)
        
        # 설명 라벨 - 초기는 단면으로 시작
        self.mode_description_label = QLabel("📖 원본 이미지를 단면 인쇄합니다")
        self.mode_description_label.setStyleSheet("""
            color: #6C757D; 
            font-size: 13px;
            padding: 8px;
            background-color: #F8F9FA;
            border-left: 3px solid #4A90E2;
            border-radius: 4px;
        """)
        self.mode_description_label.setWordWrap(True)
        
        layout.addWidget(self.dual_side_check)
        layout.addWidget(self.normal_radio)
        layout.addWidget(self.layered_radio)
        layout.addWidget(self.mode_description_label)
        
        # 신호 연결
        self.normal_radio.toggled.connect(self._on_mode_changed)
    
    def _on_dual_side_toggled(self, checked):
        """양면 인쇄 토글"""
        self.is_dual_side = checked
        self._update_description()
        self.dual_side_changed.emit(checked)  # 시그널 발송
    
    def _on_mode_changed(self):
        """모드 변경 시 처리"""
        if self.normal_radio.isChecked():
            self.print_mode = "normal"
        else:
            self.print_mode = "layered"
        
        self._update_description()
        self.mode_changed.emit(self.print_mode)
    
    def _update_description(self):
        """설명 업데이트 - 양면/단면 구분"""
        side_text = "양면" if self.is_dual_side else "단면"
        
        if self.print_mode == "normal":
            description = f"📖 원본 이미지를 {side_text} 인쇄합니다"
        else:
            description = f"📖 원본 이미지 위에 마스크 워터마크를 포함하여 {side_text} 인쇄합니다"
        
        self.mode_description_label.setText(description)
    
    def get_mode(self) -> str:
        """현재 모드 반환"""
        return self.print_mode
    
    def is_dual_side_enabled(self) -> bool:
        """양면 인쇄 활성화 여부"""
        return self.dual_side_check.isChecked()


class PrintQuantityPanel(QGroupBox):
    """인쇄 매수 선택 패널"""
    quantity_changed = Signal(int)
    
    def __init__(self):
        super().__init__("📊 인쇄 매수")
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(5)
        layout.setContentsMargins(8, 8, 8, 8)
        
        # 매수 선택 영역
        quantity_layout = QHBoxLayout()
        quantity_layout.setSpacing(6)
        
        quantity_label = QLabel("매수:")
        quantity_label.setStyleSheet("""
            font-weight: 600; 
            color: #495057; 
            font-size: 16px;
        """)
        
        self.quantity_spinbox = QSpinBox()
        self.quantity_spinbox.setMinimum(1)
        self.quantity_spinbox.setMaximum(100)
        self.quantity_spinbox.setValue(1)
        self.quantity_spinbox.setSuffix(" 장")
        self.quantity_spinbox.setFixedSize(80, 35)
        self.quantity_spinbox.setStyleSheet("""
            QSpinBox {
                background-color: #FFFFFF;
                border: 2px solid #E9ECEF;
                border-radius: 4px;
                padding: 3px 6px;
                font-size: 15px;
                color: #495057;
                font-weight: 600;
            }
            QSpinBox:focus {
                border-color: #4A90E2;
            }
            QSpinBox::up-button, QSpinBox::down-button {
                width: 16px;
                border: none;
                background-color: #F8F9FA;
            }
            QSpinBox::up-button:hover, QSpinBox::down-button:hover {
                background-color: #E9ECEF;
            }
        """)
        
        quantity_layout.addWidget(quantity_label)
        quantity_layout.addWidget(self.quantity_spinbox)
        quantity_layout.addStretch()
        
        # 예상 시간 표시
        self.time_estimate_label = QLabel("⏱️ 예상 시간: 약 30초")
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
        layout.addWidget(self.time_estimate_label)
        
        # 신호 연결
        self.quantity_spinbox.valueChanged.connect(self._on_quantity_changed)
    
    def _on_quantity_changed(self, value):
        """매수 변경 시 예상 시간 업데이트"""
        estimated_seconds = value * 30
        
        if estimated_seconds < 60:
            time_text = f"⏱️ 예상 시간: 약 {estimated_seconds}초"
        else:
            minutes = estimated_seconds // 60
            seconds = estimated_seconds % 60
            if seconds == 0:
                time_text = f"⏱️ 예상 시간: 약 {minutes}분"
            else:
                time_text = f"⏱️ 예상 시간: 약 {minutes}분 {seconds}초"
        
        self.time_estimate_label.setText(time_text)
        self.quantity_changed.emit(value)
    
    def get_quantity(self) -> int:
        """선택된 매수 반환"""
        return self.quantity_spinbox.value()
    
    def set_quantity(self, quantity: int):
        """매수 설정"""
        self.quantity_spinbox.setValue(quantity)


class PrinterPanel(QGroupBox):
    """프린터 연동 패널"""
    test_requested = Signal()
    print_requested = Signal()
    
    def __init__(self):
        super().__init__("🖨️ 프린터 연동")
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
        
        # 초기 버튼 텍스트를 "단면 일반 인쇄"로 설정
        self.print_card_btn = ModernButton("단면 일반 인쇄", primary=True)
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
    
    def update_print_button_text(self, mode: str, is_dual: bool = False, quantity: int = 1):
        """인쇄 버튼 텍스트 업데이트"""
        if mode == "normal":
            base_text = "양면 일반 인쇄" if is_dual else "단면 일반 인쇄"
        else:
            base_text = "양면 레이어 인쇄" if is_dual else "단면 레이어 인쇄"
        
        if quantity > 1:
            text = f"{base_text} ({quantity}장)"
        else:
            text = base_text
        
        text = truncate_text(text, 20)
        self.print_card_btn.setText(text)


class ProgressPanel(QGroupBox):
    """진행 상황 패널"""
    
    def __init__(self):
        super().__init__("📊 진행 상황")
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(8, 8, 8, 8)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(25)
        
        self.status_label = QLabel("대기 중...")
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
        """상태 메시지 업데이트"""
        truncated_status = truncate_text(status, 30)
        self.status_label.setText(truncated_status)
    
    def show_print_progress(self, current: int, total: int):
        """인쇄 진행상황 표시"""
        progress_text = f"📄 {current}/{total} 장 인쇄 중..."
        self.print_progress_label.setText(progress_text)
        self.print_progress_label.setVisible(True)
        
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(current)
    
    def update_print_status(self, current: int, total: int, status: str):
        """인쇄 상태와 진행률 동시 업데이트"""
        self.update_status(status)
        self.show_print_progress(current, total)


class LogPanel(QGroupBox):
    """로그 패널 - 확장 가능"""
    
    def __init__(self):
        super().__init__("📝 처리 로그")
        # 로그 패널은 확장 가능
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.setMinimumHeight(120)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(8, 8, 8, 8)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #FFFFFF;
                border: 2px solid #E9ECEF;
                border-radius: 4px;
                padding: 6px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 12px;
                line-height: 1.3;
                color: #495057;
            }
            QTextEdit:focus {
                border-color: #4A90E2;
            }
        """)
        
        layout.addWidget(self.log_text)
    
    def add_log(self, message: str):
        """로그 메시지 추가"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        self.log_text.append(formatted_message)
        
        # 스크롤을 맨 아래로
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )
    
    def clear_log(self):
        """로그 클리어"""
        self.log_text.clear()