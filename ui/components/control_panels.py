"""
ui/components/control_panels.py 수정
전역 카드 방향 선택 제거, 기존 요소들 재배치
"""

from PySide6.QtWidgets import (
    QSizePolicy, QVBoxLayout, QHBoxLayout, QGroupBox, 
    QLabel, QRadioButton, QButtonGroup, QProgressBar, QTextEdit, QCheckBox,
    QSpinBox,QFrame
)
from PySide6.QtCore import Signal
from .modern_button import ModernButton


def truncate_text(text: str, max_length: int = 30) -> str:
    """텍스트가 너무 길면 줄임 처리"""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."


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
        self.normal_radio = QRadioButton("🖨️ 일반")
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
            QLabel {
                background: transparent;
                border: none;
                padding: 0px;
                font-size: 14px;
                color: #495057;
            }
        """)

        self.quantity_spinbox = QSpinBox()
        self.quantity_spinbox.setMinimum(1)
        self.quantity_spinbox.setMaximum(99999)
        self.quantity_spinbox.setValue(1)
        self.quantity_spinbox.setSuffix(" 장")
        self.quantity_spinbox.setFixedSize(100, 35)
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
    """프린터 연동 패널 - 개별 면 방향 지원"""
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