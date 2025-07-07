"""
메인 윈도우 UI 구성 - 양면 인쇄 지원
"""

from PySide6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QGridLayout, QWidget, 
    QFrame, QLabel, QSplitter
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from .components.image_viewer import ImageViewer
from .components.control_panels import (
    FileSelectionPanel, ProcessingOptionsPanel, PrintModePanel,
    PrinterPanel, ProgressPanel, LogPanel
)
from .styles import get_header_style, get_status_bar_style


class HanaStudioMainWindow:
    """메인 윈도우 UI 구성을 담당하는 클래스 - 양면 인쇄 지원"""
    
    def __init__(self, parent_widget):
        self.parent = parent_widget
        self.setup_ui()
    
    def setup_ui(self):
        """UI 구성"""
        # 중앙 위젯
        central_widget = QWidget()
        central_widget.setStyleSheet("background-color: #F8F9FA;")
        self.parent.setCentralWidget(central_widget)
        
        # 메인 레이아웃
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # 헤더
        self.create_header(main_layout)
        
        # 메인 컨텐츠 (좌우 분할)
        content_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(content_splitter)
        
        # 좌측 패널 (컨트롤)
        self.create_left_panel(content_splitter)
        
        # 우측 패널 (이미지 뷰어)
        self.create_right_panel(content_splitter)
        
        # 하단 상태바
        self.create_status_bar(main_layout)
        
        # 분할 비율 설정
        content_splitter.setSizes([500, 1300])
    
    def create_header(self, parent_layout):
        """헤더 생성"""
        header_frame = QFrame()
        header_frame.setFixedHeight(80)
        header_frame.setStyleSheet(get_header_style())
        
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(30, 15, 30, 15)
        
        # 로고/제목
        title_label = QLabel("🎨 Hana Studio")
        title_label.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        title_label.setStyleSheet("color: #2C3E50;")
        
        subtitle_label = QLabel("AI 기반 이미지 배경 제거 및 양면 카드 인쇄 도구")
        subtitle_label.setFont(QFont("Segoe UI", 11))
        subtitle_label.setStyleSheet("color: #7F8C8D; margin-top: 5px;")
        
        title_layout = QVBoxLayout()
        title_layout.addWidget(title_label)
        title_layout.addWidget(subtitle_label)
        title_layout.setSpacing(0)
        
        header_layout.addLayout(title_layout)
        header_layout.addStretch()
        
        parent_layout.addWidget(header_frame)
    
    def create_left_panel(self, splitter):
        """좌측 컨트롤 패널 생성"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 각 패널들 생성
        self.file_panel = FileSelectionPanel()
        self.processing_panel = ProcessingOptionsPanel()
        self.print_mode_panel = PrintModePanel()
        self.printer_panel = PrinterPanel()
        self.progress_panel = ProgressPanel()
        self.log_panel = LogPanel()
        
        # 패널들을 레이아웃에 추가
        layout.addWidget(self.file_panel)
        layout.addWidget(self.processing_panel)
        layout.addWidget(self.print_mode_panel)
        layout.addWidget(self.printer_panel)
        layout.addWidget(self.progress_panel)
        layout.addWidget(self.log_panel)
        layout.addStretch()
        
        splitter.addWidget(panel)
    
    def create_right_panel(self, parent_splitter):
        """우측 이미지 뷰어 패널 생성 - 양면 미리보기"""
        right_panel = QWidget()
        right_panel.setStyleSheet("background-color: #F8F9FA;")
        right_layout = QVBoxLayout(right_panel)
        
        # 원본 이미지 뷰어 그룹
        from PySide6.QtWidgets import QGroupBox
        original_group = QGroupBox("📷 원본 이미지")
        original_layout = QHBoxLayout(original_group)
        original_layout.setSpacing(15)
        
        # 원본 이미지 뷰어들
        self.front_original_viewer = ImageViewer("📄 앞면 원본")
        self.back_original_viewer = ImageViewer("📄 뒷면 원본")
        
        original_layout.addWidget(self.front_original_viewer)
        original_layout.addWidget(self.back_original_viewer)
        
        # 처리 결과 뷰어 그룹
        result_group = QGroupBox("✨ 처리 결과")
        result_layout = QHBoxLayout(result_group)
        result_layout.setSpacing(15)
        
        # 처리 결과 뷰어들
        self.front_result_viewer = ImageViewer("🎭 앞면 처리 결과")
        self.back_result_viewer = ImageViewer("🎭 뒷면 처리 결과")
        
        result_layout.addWidget(self.front_result_viewer)
        result_layout.addWidget(self.back_result_viewer)
        
        # 최종 미리보기 그룹
        preview_group = QGroupBox("🖼️ 최종 미리보기")
        preview_layout = QVBoxLayout(preview_group)
        
        self.final_preview_viewer = ImageViewer("💎 최종 카드 미리보기")
        preview_layout.addWidget(self.final_preview_viewer)
        
        # 레이아웃에 추가
        right_layout.addWidget(original_group, 1)
        right_layout.addWidget(result_group, 1)
        right_layout.addWidget(preview_group, 1)
        
        parent_splitter.addWidget(right_panel)
    
    def create_status_bar(self, parent_layout):
        """하단 상태바 생성"""
        status_frame = QFrame()
        status_frame.setFixedHeight(40)
        status_frame.setStyleSheet(get_status_bar_style())
        
        status_layout = QHBoxLayout(status_frame)
        status_layout.setContentsMargins(20, 10, 20, 10)
        
        self.status_text = QLabel("준비 완료 | AI 모델 초기화 중...")
        self.status_text.setStyleSheet("color: #6C757D; font-size: 11px;")
        
        status_layout.addWidget(self.status_text)
        status_layout.addStretch()
        
        version_label = QLabel("Hana Studio v1.0 - 양면 인쇄 지원")
        version_label.setStyleSheet("color: #ADB5BD; font-size: 10px;")
        status_layout.addWidget(version_label)
        
        parent_layout.addWidget(status_frame)
    
    # 컴포넌트 접근을 위한 속성들
    @property
    def components(self):
        """모든 UI 컴포넌트에 대한 접근"""
        return {
            'file_panel': self.file_panel,
            'processing_panel': self.processing_panel,
            'print_mode_panel': self.print_mode_panel,
            'printer_panel': self.printer_panel,
            'progress_panel': self.progress_panel,
            'log_panel': self.log_panel,
            'front_original_viewer': self.front_original_viewer,
            'back_original_viewer': self.back_original_viewer,
            'front_result_viewer': self.front_result_viewer,
            'back_result_viewer': self.back_result_viewer,
            'final_preview_viewer': self.final_preview_viewer,
            'status_text': self.status_text
        }