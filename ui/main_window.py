"""
메인 윈도우 UI 구성 - 이미지 영역 대폭 확대 및 중앙 정렬
남는 공간을 최대한 활용하여 더 큰 이미지 미리보기 제공
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
    """메인 윈도우 UI 구성을 담당하는 클래스 - 대형 이미지 뷰어 및 중앙 정렬"""
    
    def __init__(self, parent_widget):
        self.parent = parent_widget
        self.setup_ui()
    
    def setup_ui(self):
        """UI 구성"""
        # 중앙 위젯
        central_widget = QWidget()
        central_widget.setStyleSheet("background-color: #F8F9FA;")
        self.parent.setCentralWidget(central_widget)
        
        # 메인 레이아웃 (세로) - 여백 대폭 축소
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(8)  # 12 → 8로 축소
        main_layout.setContentsMargins(10, 8, 10, 8)  # 15,15,15,15 → 10,8,10,8로 축소
        
        # 1. 헤더
        self.create_header(main_layout)
        
        # 2. 컨트롤 패널 영역 (가로 배치)
        self.create_control_area(main_layout)
        
        # 3. 메인 컨텐츠 영역 (이미지 뷰어 대폭 확대)
        self.create_large_image_area(main_layout)
        
        # 4. 하단 상태바
        self.create_status_bar(main_layout)
    
    def create_header(self, parent_layout):
        """헤더 생성 - 높이 축소"""
        header_frame = QFrame()
        header_frame.setFixedHeight(55)  # 65 → 55로 축소
        header_frame.setStyleSheet("""
            QFrame {
                background-color: #FFFFFF;
                border: none;
                border-radius: 10px;
            }
        """)
        
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(20, 8, 20, 8)  # 12 → 8로 축소
        
        # 로고/제목 영역
        title_layout = QVBoxLayout()
        title_layout.setSpacing(2)
        
        title_label = QLabel("🎨 Hana Studio")
        title_label.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))  # 20 → 18로 축소
        title_label.setStyleSheet("color: #2C3E50; background: transparent;")

        
        title_layout.addWidget(title_label)

        header_layout.addLayout(title_layout)
        header_layout.addStretch()
        
        parent_layout.addWidget(header_frame)
    
    def create_control_area(self, parent_layout):
        """컨트롤 패널 영역 생성 - 높이 축소"""
        control_container = QFrame()
        control_container.setFixedHeight(240)  # 260 → 240으로 축소
        control_container.setStyleSheet("""
            QFrame {
                border: 2px solid #DEE2E6;
                border-radius: 10px;
                background-color: #FFFFFF;
            }
        """)
        
        # 가로 레이아웃
        control_layout = QHBoxLayout(control_container)
        control_layout.setSpacing(18)  # 20 → 18로 축소
        control_layout.setContentsMargins(12, 8, 12, 8)  # 15,10,15,10 → 12,8,12,8로 축소
        
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
        
        # 각 패널별로 개별 너비 설정
        default_panel_width = 200
        
        panel_widths = {
            self.file_panel: 240,
            self.processing_panel: default_panel_width,
            self.print_mode_panel: default_panel_width,
            self.print_quantity_panel: default_panel_width,
            self.printer_panel: default_panel_width,
            self.progress_panel: default_panel_width,
            self.log_panel: 240
        }
        
        for panel in panels:
            panel_width = panel_widths[panel]
            panel.setFixedWidth(panel_width)
            control_layout.addWidget(panel)
        
        parent_layout.addWidget(control_container)
    
    def create_large_image_area(self, parent_layout):
        """이미지 뷰어 영역 대폭 확대 - 상단 여백 최소화"""
        image_widget = QWidget()
        image_widget.setStyleSheet("background-color: #F8F9FA;")
        image_layout = QVBoxLayout(image_widget)
        image_layout.setSpacing(12)  # 15 → 12로 축소
        image_layout.setContentsMargins(15, 5, 15, 10)  # 20,15,20,15 → 15,5,15,10으로 축소 (특히 상단)
        
        # === 앞면 이미지 영역 (대폭 확대) ===
        front_group = self.create_image_row("📄 앞면", is_front=True)
        
        # === 뒷면 이미지 영역 (대폭 확대) ===
        back_group = self.create_image_row("📄 뒷면", is_front=False)
        
        # 레이아웃에 추가 - 남는 공간을 모두 활용
        image_layout.addWidget(front_group, 1)  # 비율 1
        image_layout.addWidget(back_group, 1)   # 비율 1
        
        parent_layout.addWidget(image_widget, 1)  # 메인에서도 최대 확장
    
    def create_image_row(self, title: str, is_front: bool) -> QFrame:
        """이미지 행 생성 - 중앙 정렬 및 대형 뷰어"""
        group = QFrame()
        group.setStyleSheet("""
            QFrame {
                background-color: #FFFFFF;
                border: 2px solid #DEE2E6;
                border-radius: 12px;
                padding: 15px;
            }
        """)
        
        # 행 레이아웃 (가로)
        row_layout = QHBoxLayout(group)
        row_layout.setSpacing(18)  # 20 → 18로 축소
        row_layout.setContentsMargins(15, 15, 15, 15)  # 20,20,20,20 → 15,15,15,15로 축소
        
        # === 왼쪽 여백 (중앙 정렬을 위한) ===
        row_layout.addStretch(1)
        
        # === 제목 라벨 ===
        title_label = QLabel(title)
        title_label.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))  # 폰트 크기 증가
        title_label.setStyleSheet("""
            color: #495057; 
            border: none; 
            padding: 8px 12px;
            background-color: transparent;
            min-width: 80px;
        """)
        title_label.setFixedWidth(100)  # 너비 증가
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # === 원본 이미지 뷰어 (대폭 확대) ===
        if is_front:
            self.front_original_viewer = ImageViewer("원본")
            original_viewer = self.front_original_viewer
        else:
            self.back_original_viewer = ImageViewer("원본")
            original_viewer = self.back_original_viewer
        
        # 이미지 뷰어 크기 대폭 증가 (기존 240x170 → 350x260) - 약간 축소로 화면에 잘 맞추기
        original_viewer.setFixedSize(350, 260)
        
        # === 화살표 (크기 증가) ===
        arrow = QLabel("→")
        arrow.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))  # 폰트 크기 증가
        arrow.setStyleSheet("""
            color: #4A90E2; 
            border: none;
            padding: 10px;
            background-color: transparent;
        """)
        arrow.setAlignment(Qt.AlignmentFlag.AlignCenter)
        arrow.setFixedSize(60, 60)  # 화살표 영역도 증가
        
        # === 결과 이미지 뷰어 (대폭 확대) ===
        if is_front:
            self.front_result_viewer = ImageViewer("배경제거")
            result_viewer = self.front_result_viewer
        else:
            self.back_result_viewer = ImageViewer("배경제거")
            result_viewer = self.back_result_viewer
        
        result_viewer.setFixedSize(350, 260)  # 동일한 크기
        
        # === 컴포넌트 배치 (중앙 정렬) ===
        row_layout.addWidget(title_label)
        row_layout.addWidget(original_viewer)
        row_layout.addWidget(arrow)
        row_layout.addWidget(result_viewer)
        
        # === 오른쪽 여백 (중앙 정렬을 위한) ===
        row_layout.addStretch(1)
        
        return group
    
    def create_status_bar(self, parent_layout):
        """하단 상태바 생성"""
        status_frame = QFrame()
        status_frame.setFixedHeight(32)
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