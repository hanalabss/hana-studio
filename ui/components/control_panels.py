"""
좌측 컨트롤 패널 컴포넌트들 - 양면 인쇄 및 여러장 인쇄 지원
마진값 축소로 스크롤 방지
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
    """파일 선택 패널 - 양면 지원"""
    file_selected = Signal(str)  # 파일 경로 시그널
    
    def __init__(self):
        super().__init__("📁 파일 선택")
        # 내용에 맞춰 크기 조정
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self._setup_ui()
        self._calculate_height()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(6)  # 간격 줄임 (10 → 6)
        layout.setContentsMargins(12, 12, 12, 12)  # 마진 줄임 (15,20,15,15 → 12,15,12,10)
        
        # 앞면 이미지 선택
        self.front_btn = ModernButton("앞면 이미지 선택", primary=True)
        self.front_btn.setFixedHeight(35)  # 높이 줄임 (40 → 35)
        
        self.front_label = QLabel("앞면: 선택된 파일이 없습니다")
        self.front_label.setStyleSheet("""
            font-size: 10px; 
            color: #6C757D;
            padding: 3px;
            background-color: #F8F9FA;
            border-radius: 3px;
        """)
        self.front_label.setWordWrap(True)
        self.front_label.setMinimumHeight(20)  # 높이 줄임 (25 → 20)
        
        # 양면 인쇄 활성화 체크박스
        self.dual_side_check = QCheckBox("양면 인쇄 사용")
        self.dual_side_check.setStyleSheet("""
            QCheckBox {
                font-size: 11px;
                font-weight: 500;
                color: #495057;
                padding: 5px;
                spacing: 6px;
            }
        """)
        self.dual_side_check.setMinimumHeight(25)  # 높이 줄임 (30 → 25)
        self.dual_side_check.toggled.connect(self._on_dual_side_toggled)
        
        # 뒷면 이미지 선택
        self.back_btn = ModernButton("뒷면 이미지 선택")
        self.back_btn.setFixedHeight(35)  # 높이 줄임 (40 → 35)
        
        self.back_label = QLabel("뒷면: 선택된 파일이 없습니다")
        self.back_label.setStyleSheet("""
            font-size: 10px; 
            color: #6C757D;
            padding: 3px;
            background-color: #F8F9FA;
            border-radius: 3px;
        """)
        self.back_label.setWordWrap(True)
        self.back_label.setMinimumHeight(20)  # 높이 줄임 (25 → 20)
        
        layout.addWidget(self.front_btn)
        layout.addWidget(self.front_label)
        layout.addWidget(self.dual_side_check)
        layout.addWidget(self.back_btn)
        layout.addWidget(self.back_label)
        
        # 초기에는 뒷면 비활성화
        self.back_btn.setEnabled(False)
        
    def _calculate_height(self):
        """내용에 맞는 높이 계산"""
        # 버튼 2개 + 라벨 2개 + 체크박스 + 여백 계산
        content_height = 35 + 20 + 25 + 35 + 20  # 요소들 높이
        spacing_height = 6 * 4  # 간격
        margin_height = 15 + 10  # 상하 여백
        total_height = content_height + spacing_height + margin_height
        self.setFixedHeight(total_height)
        
    def _on_dual_side_toggled(self, checked):
        """양면 인쇄 토글"""
        self.back_btn.setEnabled(checked)
        if not checked:
            self.back_label.setText("뒷면: 선택된 파일이 없습니다")
    
    def update_front_file_info(self, file_path: str):
        """앞면 파일 정보 업데이트"""
        import os
        filename = os.path.basename(file_path)
        filename = truncate_text(filename, 35)
        self.front_label.setText(f"앞면: 📁 {filename}")
    
    def update_back_file_info(self, file_path: str):
        """뒷면 파일 정보 업데이트"""
        import os
        filename = os.path.basename(file_path)
        filename = truncate_text(filename, 35)
        self.back_label.setText(f"뒷면: 📁 {filename}")
    
    def is_dual_side_enabled(self) -> bool:
        """양면 인쇄 활성화 여부"""
        return self.dual_side_check.isChecked()


class ProcessingOptionsPanel(QGroupBox):
    """처리 옵션 패널"""
    process_requested = Signal()
    export_requested = Signal()
    
    def __init__(self):
        super().__init__("⚙️ 처리 옵션")
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self._setup_ui()
        self._calculate_height()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(6)  # 간격 줄임 (10 → 6)
        layout.setContentsMargins(12, 12, 12, 12)  # 마진 줄임
        
        self.process_btn = ModernButton("배경 제거 시작", primary=True)
        self.process_btn.setEnabled(False)
        self.process_btn.setFixedHeight(35)  # 높이 줄임 (40 → 35)
        
        self.export_btn = ModernButton("결과 저장")
        self.export_btn.setEnabled(False)
        self.export_btn.setFixedHeight(35)  # 높이 줄임 (40 → 35)
        
        layout.addWidget(self.process_btn)
        layout.addWidget(self.export_btn)
        
        # 신호 연결
        self.process_btn.clicked.connect(self.process_requested.emit)
        self.export_btn.clicked.connect(self.export_requested.emit)
    
    def _calculate_height(self):
        """내용에 맞는 높이 계산"""
        content_height = 35 + 35  # 버튼 2개
        spacing_height = 6 * 1  # 간격
        margin_height = 15 + 10  # 상하 여백
        total_height = content_height + spacing_height + margin_height
        self.setFixedHeight(total_height)
    
    def set_process_enabled(self, enabled: bool):
        """처리 버튼 활성화/비활성화"""
        self.process_btn.setEnabled(enabled)
    
    def set_export_enabled(self, enabled: bool):
        """저장 버튼 활성화/비활성화"""
        self.export_btn.setEnabled(enabled)


class PrintModePanel(QGroupBox):
    """인쇄 모드 패널 - 양면 지원"""
    mode_changed = Signal(str)  # "normal" 또는 "layered"
    
    def __init__(self):
        super().__init__("📋 인쇄 모드")
        self.print_mode = "normal"
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self._setup_ui()
        self._calculate_height()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(5)  # 간격 줄임 (8 → 5)
        layout.setContentsMargins(12, 12, 12, 12)  # 마진 줄임
        
        # 라디오 버튼들
        self.normal_radio = QRadioButton("일반 인쇄")
        self.layered_radio = QRadioButton("레이어 인쇄 (YMCW)")
        self.normal_radio.setChecked(True)
        
        # 라디오 버튼 스타일 조정
        radio_style = """
            QRadioButton {
                font-size: 11px;
                font-weight: 500;
                color: #495057;
                padding: 3px;
                spacing: 6px;
            }
        """
        self.normal_radio.setStyleSheet(radio_style)
        self.layered_radio.setStyleSheet(radio_style)
        self.normal_radio.setMinimumHeight(22)  # 높이 줄임 (25 → 22)
        self.layered_radio.setMinimumHeight(22)  # 높이 줄임 (25 → 22)
        
        # 버튼 그룹
        self.button_group = QButtonGroup()
        self.button_group.addButton(self.normal_radio, 0)
        self.button_group.addButton(self.layered_radio, 1)
        
        # 설명 라벨
        self.mode_description_label = QLabel("📖 원본 이미지를 양면 인쇄합니다")
        self.mode_description_label.setStyleSheet("""
            color: #6C757D; 
            font-size: 9px;
            padding: 5px;
            background-color: #F8F9FA;
            border-left: 3px solid #4A90E2;
            border-radius: 3px;
        """)
        self.mode_description_label.setWordWrap(True)
        self.mode_description_label.setMinimumHeight(28)  # 높이 줄임 (35 → 28)
        
        layout.addWidget(self.normal_radio)
        layout.addWidget(self.layered_radio)
        layout.addWidget(self.mode_description_label)
        
        # 신호 연결
        self.normal_radio.toggled.connect(self._on_mode_changed)
    
    def _calculate_height(self):
        """내용에 맞는 높이 계산"""
        content_height = 22 + 22 + 28  # 라디오 버튼 2개 + 설명 라벨
        spacing_height = 5 * 2  # 간격
        margin_height = 15 + 10  # 상하 여백
        total_height = content_height + spacing_height + margin_height
        self.setFixedHeight(total_height)
    
    def _on_mode_changed(self):
        """모드 변경 시 처리"""
        if self.normal_radio.isChecked():
            self.print_mode = "normal"
            description = "📖 원본 이미지를 양면 인쇄합니다"
        else:
            self.print_mode = "layered"
            description = "📖 원본 이미지 위에 마스크 워터마크를 포함하여 양면 인쇄합니다"
        
        self.mode_description_label.setText(description)
        self.mode_changed.emit(self.print_mode)
    
    def get_mode(self) -> str:
        """현재 모드 반환"""
        return self.print_mode


class PrintQuantityPanel(QGroupBox):
    """인쇄 매수 선택 패널"""
    quantity_changed = Signal(int)
    
    def __init__(self):
        super().__init__("📊 인쇄 매수")
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self._setup_ui()
        self._calculate_height()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)  # 간격 줄임 (10 → 6)
        layout.setContentsMargins(12, 12, 12, 12)  # 마진 줄임
        
        # 매수 선택 영역
        quantity_layout = QHBoxLayout()
        quantity_layout.setSpacing(8)
        
        quantity_label = QLabel("매수:")
        quantity_label.setStyleSheet("""
            font-weight: 500; 
            color: #495057; 
            font-size: 11px;
            padding: 3px;
        """)
        quantity_label.setFixedWidth(35)
        
        self.quantity_spinbox = QSpinBox()
        self.quantity_spinbox.setMinimum(1)
        self.quantity_spinbox.setMaximum(100)
        self.quantity_spinbox.setValue(1)
        self.quantity_spinbox.setSuffix(" 장")
        self.quantity_spinbox.setFixedSize(85, 30)  # 크기 줄임 (90,35 → 85,30)
        self.quantity_spinbox.setStyleSheet("""
            QSpinBox {
                background-color: #FFFFFF;
                border: 2px solid #E9ECEF;
                border-radius: 4px;
                padding: 3px 6px;
                font-size: 11px;
                color: #495057;
                font-weight: 500;
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
            font-size: 9px;
            padding: 5px;
            background-color: #F8F9FA;
            border-left: 3px solid #28A745;
            border-radius: 3px;
        """)
        self.time_estimate_label.setWordWrap(True)
        self.time_estimate_label.setMinimumHeight(25)  # 높이 줄임 (30 → 25)
        
        layout.addLayout(quantity_layout)
        layout.addWidget(self.time_estimate_label)
        
        # 신호 연결
        self.quantity_spinbox.valueChanged.connect(self._on_quantity_changed)
    
    def _calculate_height(self):
        """내용에 맞는 높이 계산"""
        content_height = 30 + 25  # 스핀박스 + 시간 라벨
        spacing_height = 6 * 1  # 간격
        margin_height = 15 + 10  # 상하 여백
        total_height = content_height + spacing_height + margin_height
        self.setFixedHeight(total_height)
    
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
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self._setup_ui()
        self._calculate_height()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(6)  # 간격 줄임 (10 → 6)
        layout.setContentsMargins(12, 12, 12, 12)  # 마진 줄임
        
        # 상태 라벨
        self.printer_status_label = QLabel("프린터 상태 확인 중...")
        self.printer_status_label.setStyleSheet("""
            font-size: 10px; 
            color: #6C757D;
            padding: 5px;
            background-color: #F8F9FA;
            border-radius: 3px;
        """)
        self.printer_status_label.setWordWrap(True)
        self.printer_status_label.setMinimumHeight(25)  # 높이 줄임 (30 → 25)
        
        # 버튼들
        self.test_printer_btn = ModernButton("프린터 연결 테스트")
        self.test_printer_btn.setFixedHeight(30)  # 높이 줄임 (35 → 30)
        
        self.print_card_btn = ModernButton("양면 카드 인쇄", primary=True)
        self.print_card_btn.setEnabled(False)
        self.print_card_btn.setFixedHeight(35)  # 높이 줄임 (40 → 35)
        
        layout.addWidget(self.printer_status_label)
        layout.addWidget(self.test_printer_btn)
        layout.addWidget(self.print_card_btn)
        
        # 신호 연결
        self.test_printer_btn.clicked.connect(self.test_requested.emit)
        self.print_card_btn.clicked.connect(self.print_requested.emit)
    
    def _calculate_height(self):
        """내용에 맞는 높이 계산"""
        content_height = 25 + 30 + 35  # 상태 라벨 + 테스트 버튼 + 인쇄 버튼
        spacing_height = 6 * 2  # 간격
        margin_height = 15 + 10  # 상하 여백
        total_height = content_height + spacing_height + margin_height
        self.setFixedHeight(total_height)
    
    def update_status(self, status: str):
        """프린터 상태 업데이트"""
        truncated_status = truncate_text(status, 50)
        self.printer_status_label.setText(truncated_status)
    
    def set_test_enabled(self, enabled: bool):
        """테스트 버튼 활성화/비활성화"""
        self.test_printer_btn.setEnabled(enabled)
    
    def set_print_enabled(self, enabled: bool):
        """인쇄 버튼 활성화/비활성화"""
        self.print_card_btn.setEnabled(enabled)
    
    def update_print_button_text(self, mode: str, is_dual: bool = True, quantity: int = 1):
        """인쇄 버튼 텍스트 업데이트"""
        if mode == "normal":
            base_text = "양면 일반 인쇄" if is_dual else "단면 일반 인쇄"
        else:
            base_text = "양면 레이어 인쇄" if is_dual else "단면 레이어 인쇄"
        
        if quantity > 1:
            text = f"{base_text} ({quantity}장)"
        else:
            text = base_text
        
        text = truncate_text(text, 25)
        self.print_card_btn.setText(text)


class ProgressPanel(QGroupBox):
    """진행 상황 패널"""
    
    def __init__(self):
        super().__init__("📊 진행 상황")
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self._setup_ui()
        self._calculate_height()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(5)  # 간격 줄임 (8 → 5)
        layout.setContentsMargins(12, 12, 12, 12)  # 마진 줄임
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(22)  # 높이 줄임 (25 → 22)
        
        self.status_label = QLabel("대기 중...")
        self.status_label.setStyleSheet("""
            font-size: 10px; 
            color: #495057;
            padding: 3px;
            background-color: #F8F9FA;
            border-radius: 3px;
        """)
        self.status_label.setWordWrap(True)
        self.status_label.setMinimumHeight(22)  # 높이 줄임 (25 → 22)
        
        # 인쇄 진행상황 표시용 라벨
        self.print_progress_label = QLabel("")
        self.print_progress_label.setStyleSheet("""
            color: #4A90E2; 
            font-size: 10px; 
            font-weight: 600;
            padding: 3px 6px;
            background-color: rgba(74, 144, 226, 0.1);
            border-radius: 3px;
        """)
        self.print_progress_label.setVisible(False)
        self.print_progress_label.setMinimumHeight(22)  # 높이 줄임 (25 → 22)
        
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)
        layout.addWidget(self.print_progress_label)
    
    def _calculate_height(self):
        """내용에 맞는 높이 계산"""
        content_height = 22 + 22 + 22  # 진행바 + 상태 라벨 + 인쇄 진행 라벨
        spacing_height = 5 * 2  # 간격
        margin_height = 15 + 10  # 상하 여백
        total_height = content_height + spacing_height + margin_height
        self.setFixedHeight(total_height)
    
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
        truncated_status = truncate_text(status, 40)
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
        # 로그 패널만 확장 가능
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.setMinimumHeight(150)  # 최소 높이 줄임 (200 → 150)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(12, 12, 12, 12)  # 마진 줄임
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #FFFFFF;
                border: 2px solid #E9ECEF;
                border-radius: 4px;
                padding: 8px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 9px;
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