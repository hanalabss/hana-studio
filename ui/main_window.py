"""
메인 윈도우 UI 구성 - 가로 배치 레이아웃
유동적 크기 조절로 내용에 맞게 자동 크기 조정
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
    """메인 윈도우 UI 구성을 담당하는 클래스 - 가로 배치 레이아웃"""
    
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
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # 1. 헤더
        self.create_header(main_layout)
        
        # 2. 컨트롤 패널 영역 (가로 배치) - 유동적 크기
        self.create_control_area(main_layout)
        
        # 3. 메인 컨텐츠 영역 (이미지 뷰어만)
        self.create_image_area(main_layout)
        
        # 4. 하단 상태바
        self.create_status_bar(main_layout)
    
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
        
        subtitle_label = QLabel("AI 기반 이미지 배경 제거 및 양면 카드 인쇄 도구 - 여러장 인쇄 지원")
        subtitle_label.setFont(QFont("Segoe UI", 11))
        subtitle_label.setStyleSheet("color: #7F8C8D; margin-top: 5px;")
        
        title_layout = QVBoxLayout()
        title_layout.addWidget(title_label)
        title_layout.addWidget(subtitle_label)
        title_layout.setSpacing(0)
        
        header_layout.addLayout(title_layout)
        header_layout.addStretch()
        
        parent_layout.addWidget(header_frame)
    
    def create_control_area(self, parent_layout):
        """컨트롤 패널 영역 생성 - 가로 배치, 유동적 크기"""
        # 스크롤 가능한 컨트롤 영역 - 높이를 훨씬 작게 설정
        scroll_area = QScrollArea()
        scroll_area.setMinimumHeight(120)  # 최소 높이 대폭 줄임 (150 → 120)
        scroll_area.setMaximumHeight(280)  # 최대 높이 대폭 줄임 (180 → 140)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: 2px solid #DEE2E6;
                border-radius: 10px;
                background-color: #FFFFFF;
            }
            QScrollBar:horizontal {
                background-color: #F8F9FA;
                height: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:horizontal {
                background-color: #CED4DA;
                border-radius: 6px;
                min-width: 20px;
            }
            QScrollBar::handle:horizontal:hover {
                background-color: #ADB5BD;
            }
        """)
        
        # 컨트롤 컨테이너
        control_container = QWidget()
        scroll_area.setWidget(control_container)
        scroll_area.setWidgetResizable(True)
        
        # 가로 레이아웃
        control_layout = QHBoxLayout(control_container)
        control_layout.setSpacing(10)  # 패널 간 간격 줄임 (15 → 10)
        control_layout.setContentsMargins(10, 8, 10, 8)  # 여백 줄임 (15,15,15,15 → 10,8,10,8)
        
        # 각 패널들 생성 및 추가 - 고정 너비 제거
        self.file_panel = FileSelectionPanel()
        self.processing_panel = ProcessingOptionsPanel()
        self.print_mode_panel = PrintModePanel()
        self.print_quantity_panel = PrintQuantityPanel()
        self.printer_panel = PrinterPanel()
        self.progress_panel = ProgressPanel()  # 진행상황 패널 추가
        self.log_panel = LogPanel()
        
        # 패널들을 레이아웃에 추가 - 각각 최소 너비만 설정
        panels = [
            self.file_panel, self.processing_panel, self.print_mode_panel,
            self.print_quantity_panel, self.printer_panel, self.progress_panel, self.log_panel
        ]
        
        for panel in panels:
            panel.setMinimumWidth(160)  # 최소 너비 줄임 (200 → 160)
            panel.setMaximumWidth(220)  # 최대 너비 줄임 (280 → 220)
            control_layout.addWidget(panel)
        
        control_layout.addStretch()  # 남은 공간 채우기
        
        parent_layout.addWidget(scroll_area)
    
    def create_image_area(self, parent_layout):
        """이미지 뷰어 영역 생성 - 2줄 배치, 크기 축소"""
        image_widget = QWidget()
        image_widget.setStyleSheet("background-color: #F8F9FA;")
        image_layout = QVBoxLayout(image_widget)
        image_layout.setSpacing(15)
        image_layout.setContentsMargins(12, 12, 12, 12)
        
        # 앞면 이미지 영역 (1줄)
        front_group = QFrame()
        front_group.setStyleSheet("""
            QFrame {
                background-color: #FFFFFF;
                border: 2px solid #DEE2E6;
                border-radius: 10px;
                padding: 12px;
            }
        """)
        front_group.setFixedHeight(240)  # 높이 축소 (300 → 240)
        
        front_layout = QHBoxLayout(front_group)
        front_layout.setSpacing(15)
        
        # 앞면 라벨
        front_title = QLabel("📄 앞면")
        front_title.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        front_title.setStyleSheet("color: #495057; border: none; padding: 3px;")
        front_title.setFixedWidth(60)
        front_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # 앞면 이미지 뷰어들 - 크기 축소
        self.front_original_viewer = ImageViewer("원본")
        self.front_original_viewer.setFixedSize(280, 190)  # 크기 축소 (350x250 → 280x190)
        
        # 화살표
        arrow1 = QLabel("→")
        arrow1.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        arrow1.setStyleSheet("color: #4A90E2; border: none;")
        arrow1.setAlignment(Qt.AlignmentFlag.AlignCenter)
        arrow1.setFixedWidth(40)
        
        self.front_result_viewer = ImageViewer("배경제거")
        self.front_result_viewer.setFixedSize(280, 190)  # 크기 축소
        
        front_layout.addWidget(front_title)
        front_layout.addWidget(self.front_original_viewer)
        front_layout.addWidget(arrow1)
        front_layout.addWidget(self.front_result_viewer)
        front_layout.addStretch()
        
        # 뒷면 이미지 영역 (2줄)
        back_group = QFrame()
        back_group.setStyleSheet("""
            QFrame {
                background-color: #FFFFFF;
                border: 2px solid #DEE2E6;
                border-radius: 10px;
                padding: 12px;
            }
        """)
        back_group.setFixedHeight(240)  # 높이 축소 (300 → 240)
        
        back_layout = QHBoxLayout(back_group)
        back_layout.setSpacing(15)
        
        # 뒷면 라벨
        back_title = QLabel("📄 뒷면")
        back_title.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        back_title.setStyleSheet("color: #495057; border: none; padding: 3px;")
        back_title.setFixedWidth(60)
        back_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # 뒷면 이미지 뷰어들 - 크기 축소
        self.back_original_viewer = ImageViewer("원본")
        self.back_original_viewer.setFixedSize(280, 190)  # 크기 축소
        
        # 화살표
        arrow2 = QLabel("→")
        arrow2.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        arrow2.setStyleSheet("color: #4A90E2; border: none;")
        arrow2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        arrow2.setFixedWidth(40)
        
        self.back_result_viewer = ImageViewer("배경제거")
        self.back_result_viewer.setFixedSize(280, 190)  # 크기 축소
        
        back_layout.addWidget(back_title)
        back_layout.addWidget(self.back_original_viewer)
        back_layout.addWidget(arrow2)
        back_layout.addWidget(self.back_result_viewer)
        back_layout.addStretch()
        
        # 레이아웃에 추가
        image_layout.addWidget(front_group)
        image_layout.addWidget(back_group)
        image_layout.addStretch()  # 남은 공간
        
        parent_layout.addWidget(image_widget, 1)  # 확장 가능하게 추가
    
    def create_status_bar(self, parent_layout):
        """하단 상태바 생성"""
        status_frame = QFrame()
        status_frame.setFixedHeight(35)  # 높이 축소 (40 → 35)
        status_frame.setStyleSheet(get_status_bar_style())
        
        status_layout = QHBoxLayout(status_frame)
        status_layout.setContentsMargins(20, 8, 20, 8)
        
        self.status_text = QLabel("준비 완료 | AI 모델 초기화 중...")
        self.status_text.setStyleSheet("color: #6C757D; font-size: 10px;")
        
        status_layout.addWidget(self.status_text)
        status_layout.addStretch()
        
        version_label = QLabel("Hana Studio v1.0 - 양면 및 여러장 인쇄 지원")
        version_label.setStyleSheet("color: #ADB5BD; font-size: 9px;")
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
            'progress_panel': self.progress_panel,  # 진행상황 패널 추가
            'log_panel': self.log_panel,
            'front_original_viewer': self.front_original_viewer,
            'back_original_viewer': self.back_original_viewer,
            'front_result_viewer': self.front_result_viewer,
            'back_result_viewer': self.back_result_viewer,
            'final_preview_viewer': None,  # 최종 미리보기 제거
            'status_text': self.status_text,
            # 통합된 진행상황 컴포넌트들 (상단 패널에서 참조)
            'progress_bar': self.progress_panel.progress_bar,
            'status_label': self.progress_panel.status_label,
            'log_text': self.log_panel.log_text  # 상단 로그 패널의 log_text 참조
        }