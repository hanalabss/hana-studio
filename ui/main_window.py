"""
메인 윈도우 UI 구성 - 헤더 밑줄 제거 및 이미지 레이아웃 개선
깔끔하고 전문적인 디자인으로 업데이트
"""

from PySide6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QProgressBar, QWidget, 
    QFrame, QLabel, QSplitter, QScrollArea
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from .components.image_viewer import ImageViewer
from .components.control_panels import (
    FileSelectionPanel, ProcessingOptionsPanel, PrintModePanel,
    PrinterPanel, ProgressPanel, LogPanel, PrintQuantityPanel
)
from .styles import get_header_style, get_status_bar_style


class HanaStudioMainWindow:
    """메인 윈도우 UI 구성을 담당하는 클래스 - 개선된 헤더 및 이미지 레이아웃"""
    
    def __init__(self, parent_widget):
        self.parent = parent_widget
        self.setup_ui()
    
    def setup_ui(self):
        """UI 구성"""
        # 중앙 위젯
        central_widget = QWidget()
        central_widget.setStyleSheet("background-color: #F8F9FA;")
        self.parent.setCentralWidget(central_widget)
        
        # 메인 레이아웃 (세로)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(12)  # 15 → 12로 축소
        main_layout.setContentsMargins(15, 15, 15, 15)  # 20 → 15로 축소
        
        # 1. 헤더 (밑줄 제거)
        self.create_header(main_layout)
        
        # 2. 컨트롤 패널 영역 (가로 배치)
        self.create_control_area(main_layout)
        
        # 3. 메인 컨텐츠 영역 (이미지 뷰어만, 개선된 레이아웃)
        self.create_image_area(main_layout)
        
        # 4. 하단 상태바 (밑줄 제거)
        self.create_status_bar(main_layout)
    
    def create_header(self, parent_layout):
        """헤더 생성 - 밑줄 제거, 깔끔한 스타일 (높이 축소)"""
        header_frame = QFrame()
        header_frame.setFixedHeight(65)  # 75 → 65로 축소
        header_frame.setStyleSheet("""
            QFrame {
                background-color: #FFFFFF;
                border: none;
                border-radius: 10px;
            }
        """)
        
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(20, 12, 20, 12)  # 여백 축소
        
        # 로고/제목 영역
        title_layout = QVBoxLayout()
        title_layout.setSpacing(2)
        
        title_label = QLabel("🎨 Hana Studio")
        title_label.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))  # 22 → 20으로 축소
        title_label.setStyleSheet("color: #2C3E50; background: transparent;")
        
        # subtitle_label = QLabel("AI 기반 이미지 배경 제거 및 양면 카드 인쇄 도구")
        # subtitle_label.setFont(QFont("Segoe UI", 10))  # 11 → 10으로 축소
        # subtitle_label.setStyleSheet("color: #6C757D; background: transparent;")
        
        title_layout.addWidget(title_label)
        # title_layout.addWidget(subtitle_label)
        
        header_layout.addLayout(title_layout)
        header_layout.addStretch()
        
        parent_layout.addWidget(header_frame)
    
    def create_control_area(self, parent_layout):
        """컨트롤 패널 영역 생성 - 높이 축소"""
        # 컨트롤 컨테이너
        control_container = QFrame()
        control_container.setFixedHeight(260)  # 280 → 260으로 축소
        control_container.setStyleSheet("""
            QFrame {
                border: 2px solid #DEE2E6;
                border-radius: 10px;
                background-color: #FFFFFF;
            }
        """)
        
        # 가로 레이아웃
        control_layout = QHBoxLayout(control_container)
        control_layout.setSpacing(20)  # 25 → 20으로 축소
        control_layout.setContentsMargins(15, 10, 15, 10)  # 여백 축소
        
        # 각 패널들 생성
        self.file_panel = FileSelectionPanel()
        self.processing_panel = ProcessingOptionsPanel()
        self.print_mode_panel = PrintModePanel()
        self.print_quantity_panel = PrintQuantityPanel()
        self.printer_panel = PrinterPanel()
        self.progress_panel = ProgressPanel()
        self.log_panel = LogPanel()
        
        # 패널들 리스트
        panels = [
            self.file_panel, 
            self.processing_panel, 
            self.print_mode_panel,
            self.print_quantity_panel, 
            self.printer_panel, 
            self.progress_panel, 
            self.log_panel
        ]
        
        # 각 패널별로 개별 너비 설정 (축소)
        default_panel_width = 200  # 210 → 200으로 축소
        
        panel_widths = {
            self.file_panel: 240,  # 250 → 240으로 축소
            self.processing_panel: default_panel_width,
            self.print_mode_panel: default_panel_width,
            self.print_quantity_panel: default_panel_width,
            self.printer_panel: default_panel_width,
            self.progress_panel: default_panel_width,
            self.log_panel: 240  # 250 → 240으로 축소
        }
        
        for panel in panels:
            panel_width = panel_widths[panel]
            panel.setFixedWidth(panel_width)
            control_layout.addWidget(panel)
        
        parent_layout.addWidget(control_container)
    
    def create_image_area(self, parent_layout):
        """이미지 뷰어 영역 생성 - 높이 최적화로 잘림 방지"""
        image_widget = QWidget()
        image_widget.setStyleSheet("background-color: #F8F9FA;")
        image_layout = QVBoxLayout(image_widget)
        image_layout.setSpacing(10)  # 12 → 10으로 축소
        image_layout.setContentsMargins(15, 8, 15, 10)  # 상단 여백 더 축소
        
        # 앞면 이미지 영역 (1줄) - 높이 축소
        front_group = QFrame()
        front_group.setStyleSheet("""
            QFrame {
                background-color: #FFFFFF;
                border: 2px solid #DEE2E6;
                border-radius: 10px;
                padding: 8px;
            }
        """)
        front_group.setFixedHeight(200)  # 220 → 200으로 축소
        
        front_layout = QHBoxLayout(front_group)
        front_layout.setSpacing(10)  # 12 → 10으로 축소
        front_layout.setContentsMargins(8, 8, 8, 8)  # 여백 축소
        
        # 앞면 라벨
        front_title = QLabel("📄 앞면")
        front_title.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))  # 11 → 10으로 축소
        front_title.setStyleSheet("color: #495057; border: none; padding: 2px;")
        front_title.setFixedWidth(50)  # 55 → 50으로 축소
        front_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # 앞면 이미지 뷰어들 - 크기 축소
        self.front_original_viewer = ImageViewer("원본")
        self.front_original_viewer.setFixedSize(240, 150)  # 260×170 → 240×150으로 축소
        
        # 화살표
        arrow1 = QLabel("→")
        arrow1.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))  # 18 → 16으로 축소
        arrow1.setStyleSheet("color: #4A90E2; border: none;")
        arrow1.setAlignment(Qt.AlignmentFlag.AlignCenter)
        arrow1.setFixedWidth(30)  # 35 → 30으로 축소
        
        self.front_result_viewer = ImageViewer("배경제거")
        self.front_result_viewer.setFixedSize(240, 150)  # 크기 축소
        
        front_layout.addWidget(front_title)
        front_layout.addWidget(self.front_original_viewer)
        front_layout.addWidget(arrow1)
        front_layout.addWidget(self.front_result_viewer)
        front_layout.addStretch()
        
        # 뒷면 이미지 영역 (2줄) - 높이 축소
        back_group = QFrame()
        back_group.setStyleSheet("""
            QFrame {
                background-color: #FFFFFF;
                border: 2px solid #DEE2E6;
                border-radius: 10px;
                padding: 8px;
            }
        """)
        back_group.setFixedHeight(200)  # 220 → 200으로 축소
        
        back_layout = QHBoxLayout(back_group)
        back_layout.setSpacing(10)
        back_layout.setContentsMargins(8, 8, 8, 8)
        
        # 뒷면 라벨
        back_title = QLabel("📄 뒷면")
        back_title.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        back_title.setStyleSheet("color: #495057; border: none; padding: 2px;")
        back_title.setFixedWidth(50)
        back_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # 뒷면 이미지 뷰어들
        self.back_original_viewer = ImageViewer("원본")
        self.back_original_viewer.setFixedSize(240, 150)  # 크기 축소
        
        # 화살표
        arrow2 = QLabel("→")
        arrow2.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        arrow2.setStyleSheet("color: #4A90E2; border: none;")
        arrow2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        arrow2.setFixedWidth(30)
        
        self.back_result_viewer = ImageViewer("배경제거")
        self.back_result_viewer.setFixedSize(240, 150)  # 크기 축소
        
        back_layout.addWidget(back_title)
        back_layout.addWidget(self.back_original_viewer)
        back_layout.addWidget(arrow2)
        back_layout.addWidget(self.back_result_viewer)
        back_layout.addStretch()
        
        # 레이아웃에 추가
        image_layout.addWidget(front_group)
        image_layout.addWidget(back_group)
        image_layout.addStretch()  # 남은 공간
        
        parent_layout.addWidget(image_widget, 1)
    
    def create_status_bar(self, parent_layout):
        """하단 상태바 생성 - 밑줄 제거"""
        status_frame = QFrame()
        status_frame.setFixedHeight(32)  # 높이 축소
        status_frame.setStyleSheet("""
            QFrame {
                background-color: #FFFFFF;
                border: none;
                border-radius: 8px;
            }
        """)
        
        status_layout = QHBoxLayout(status_frame)
        status_layout.setContentsMargins(20, 6, 20, 6)
        
        self.status_text = QLabel("준비 완료 | AI 모델 초기화 중...")
        self.status_text.setStyleSheet("color: #6C757D; font-size: 10px; background: transparent;")
        
        status_layout.addWidget(self.status_text)
        status_layout.addStretch()
        
        version_label = QLabel("Hana Studio v1.0 - 양면 및 여러장 인쇄 지원")
        version_label.setStyleSheet("color: #ADB5BD; font-size: 9px; background: transparent;")
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
            'print_quantity_panel': self.print_quantity_panel,
            'printer_panel': self.printer_panel,
            'progress_panel': self.progress_panel,
            'log_panel': self.log_panel,
            'front_original_viewer': self.front_original_viewer,
            'back_original_viewer': self.back_original_viewer,
            'front_result_viewer': self.front_result_viewer,
            'back_result_viewer': self.back_result_viewer,
            'final_preview_viewer': None,
            'status_text': self.status_text,
            'progress_bar': self.progress_panel.progress_bar,
            'status_label': self.progress_panel.status_label,
            'log_text': self.log_panel.log_text
        }