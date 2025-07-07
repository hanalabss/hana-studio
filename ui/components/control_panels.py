"""
좌측 컨트롤 패널 컴포넌트들
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
    QLabel, QRadioButton, QButtonGroup, QProgressBar, QTextEdit
)
from PySide6.QtCore import Signal
from .modern_button import ModernButton


class FileSelectionPanel(QGroupBox):
    """파일 선택 패널"""
    file_selected = Signal(str)  # 파일 경로 시그널
    
    def __init__(self):
        super().__init__("📁 파일 선택")
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        self.select_btn = ModernButton("이미지 선택", primary=True)
        self.file_label = QLabel("선택된 파일이 없습니다")
        
        layout.addWidget(self.select_btn)
        layout.addWidget(self.file_label)
        
        # 신호 연결은 외부에서 처리
    
    def update_file_info(self, file_path: str):
        """파일 정보 업데이트"""
        import os
        self.file_label.setText(f"📁 {os.path.basename(file_path)}")


class ProcessingOptionsPanel(QGroupBox):
    """처리 옵션 패널"""
    process_requested = Signal()
    export_requested = Signal()
    
    def __init__(self):
        super().__init__("⚙️ 처리 옵션")
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        self.process_btn = ModernButton("배경 제거 시작", primary=True)
        self.process_btn.setEnabled(False)
        
        self.export_btn = ModernButton("결과 저장")
        self.export_btn.setEnabled(False)
        
        layout.addWidget(self.process_btn)
        layout.addWidget(self.export_btn)
        
        # 신호 연결
        self.process_btn.clicked.connect(self.process_requested.emit)
        self.export_btn.clicked.connect(self.export_requested.emit)
    
    def set_process_enabled(self, enabled: bool):
        """처리 버튼 활성화/비활성화"""
        self.process_btn.setEnabled(enabled)
    
    def set_export_enabled(self, enabled: bool):
        """저장 버튼 활성화/비활성화"""
        self.export_btn.setEnabled(enabled)


class PrintModePanel(QGroupBox):
    """인쇄 모드 패널"""
    mode_changed = Signal(str)  # "normal" 또는 "layered"
    
    def __init__(self):
        super().__init__("📋 인쇄 모드")
        self.print_mode = "normal"
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        # 라디오 버튼들
        self.normal_radio = QRadioButton("일반 인쇄")
        self.layered_radio = QRadioButton("레이어 인쇄")
        self.normal_radio.setChecked(True)
        
        # 버튼 그룹
        self.button_group = QButtonGroup()
        self.button_group.addButton(self.normal_radio, 0)
        self.button_group.addButton(self.layered_radio, 1)
        
        # 설명 라벨
        self.mode_description_label = QLabel("📖 원본 이미지만 인쇄합니다")
        self.mode_description_label.setStyleSheet("""
            color: #6C757D; font-size: 10px;
            padding: 4px 8px;
            background-color: #F8F9FA;
            border-left: 3px solid #4A90E2;
        """)
        self.mode_description_label.setWordWrap(True)
        
        layout.addWidget(self.normal_radio)
        layout.addWidget(self.layered_radio)
        layout.addWidget(self.mode_description_label)
        
        # 신호 연결
        self.normal_radio.toggled.connect(self._on_mode_changed)
    
    def _on_mode_changed(self):
        """모드 변경 시 처리"""
        if self.normal_radio.isChecked():
            self.print_mode = "normal"
            self.mode_description_label.setText("📖 원본 이미지만 인쇄합니다")
        else:
            self.print_mode = "layered"
            self.mode_description_label.setText("📖 원본 이미지 위에 마스크 워터마크를 포함하여 인쇄합니다")
        
        self.mode_changed.emit(self.print_mode)
    
    def get_mode(self) -> str:
        """현재 모드 반환"""
        return self.print_mode


class PrinterPanel(QGroupBox):
    """프린터 연동 패널"""
    test_requested = Signal()
    print_requested = Signal()
    
    def __init__(self):
        super().__init__("🖨️ 프린터 연동")
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        # 상태 라벨
        self.printer_status_label = QLabel("프린터 상태 확인 중...")
        self.printer_status_label.setStyleSheet("font-size: 10px; color: #6C757D;")
        self.printer_status_label.setWordWrap(True)
        
        # 버튼들
        self.test_printer_btn = ModernButton("프린터 연결 테스트")
        self.print_card_btn = ModernButton("카드 인쇄", primary=True)
        self.print_card_btn.setEnabled(False)
        
        layout.addWidget(self.printer_status_label)
        layout.addWidget(self.test_printer_btn)
        layout.addWidget(self.print_card_btn)
        
        # 신호 연결
        self.test_printer_btn.clicked.connect(self.test_requested.emit)
        self.print_card_btn.clicked.connect(self.print_requested.emit)
    
    def update_status(self, status: str):
        """프린터 상태 업데이트"""
        self.printer_status_label.setText(status)
    
    def set_test_enabled(self, enabled: bool):
        """테스트 버튼 활성화/비활성화"""
        self.test_printer_btn.setEnabled(enabled)
    
    def set_print_enabled(self, enabled: bool):
        """인쇄 버튼 활성화/비활성화"""
        self.print_card_btn.setEnabled(enabled)
    
    def update_print_button_text(self, mode: str):
        """인쇄 버튼 텍스트 업데이트"""
        if mode == "normal":
            self.print_card_btn.setText("일반 카드 인쇄")
        else:
            self.print_card_btn.setText("레이어 카드 인쇄")


class ProgressPanel(QGroupBox):
    """진행 상황 패널"""
    
    def __init__(self):
        super().__init__("📊 진행 상황")
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        self.status_label = QLabel("대기 중...")
        
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)
    
    def show_progress(self, indeterminate=True):
        """진행바 표시"""
        if indeterminate:
            self.progress_bar.setRange(0, 0)  # 무한 진행바
        self.progress_bar.setVisible(True)
    
    def hide_progress(self):
        """진행바 숨기기"""
        self.progress_bar.setVisible(False)
    
    def update_status(self, status: str):
        """상태 메시지 업데이트"""
        self.status_label.setText(status)


class LogPanel(QGroupBox):
    """로그 패널"""
    
    def __init__(self):
        super().__init__("📝 처리 로그")
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        
        layout.addWidget(self.log_text)
    
    def add_log(self, message: str):
        """로그 메시지 추가"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        
        # 스크롤을 맨 아래로
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )
    
    def clear_log(self):
        """로그 클리어"""
        self.log_text.clear()