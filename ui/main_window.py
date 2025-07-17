"""
ui/main_window.py 수정
탭 기반 이미지 뷰어로 변경 - 통일된 크기 사용
"""

from PySide6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QProgressBar, QWidget, 
    QFrame, QLabel, QSplitter, QScrollArea, QTabWidget
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont

from .components.image_viewer import ImageViewer, UnifiedMaskViewer, UNIFIED_VIEWER_WIDTH, UNIFIED_VIEWER_HEIGHT
from .components.control_panels import (
    FileSelectionPanel, PrintModePanel,
    PrinterPanel, ProgressPanel, LogPanel, PrintQuantityPanel
)
from .styles import get_header_style, get_status_bar_style


class ImageTabWidget(QTabWidget):
    """프로페셔널한 이미지 탭 위젯"""
    tab_changed = Signal(int)  # 탭 변경 시그널
    
    def __init__(self):
        super().__init__()
        self.setTabPosition(QTabWidget.TabPosition.North)
        self.setMovable(False)
        self.setTabsClosable(False)
        
        # 프로페셔널한 탭 스타일
        self.setStyleSheet("""
            QTabWidget::pane {
                border: 2px solid #DEE2E6;
                border-radius: 12px;
                background-color: #FFFFFF;
                margin-top: -1px;
            }
            QTabWidget::tab-bar {
                alignment: center;
            }
            QTabBar::tab {
                background-color: #F8F9FA;
                border: 2px solid #DEE2E6;
                border-bottom: none;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                padding: 12px 24px;
                margin-right: 2px;
                font-size: 13px;
                font-weight: 600;
                color: #6C757D;
                min-width: 100px;
            }
            QTabBar::tab:hover {
                background-color: #E9ECEF;
                color: #495057;
            }
            QTabBar::tab:selected {
                background-color: #FFFFFF;
                color: #4A90E2;
                border-color: #DEE2E6;
                font-weight: bold;
            }
            QTabBar::tab:!selected {
                margin-top: 2px;
            }
        """)
        
        # 탭 변경 시그널 연결
        self.currentChanged.connect(self.tab_changed.emit)
    
    def set_dual_side_enabled(self, enabled: bool):
        """양면 모드에 따른 탭 표시/숨김"""
        if enabled:
            self.setTabVisible(1, True)  # 뒷면 탭 표시
        else:
            self.setTabVisible(1, False)  # 뒷면 탭 숨김
            self.setCurrentIndex(0)  # 앞면 탭으로 강제 이동


class HanaStudioMainWindow:
    """메인 윈도우 UI 구성 - 탭 기반 이미지 뷰어"""
    
    def __init__(self, parent_widget):
        self.parent = parent_widget
        self.current_tab_index = 0  # 현재 선택된 탭
        self.setup_ui()
    
    def setup_ui(self):
        """UI 구성"""
        # 중앙 위젯
        central_widget = QWidget()
        central_widget.setStyleSheet("background-color: #F8F9FA;")
        self.parent.setCentralWidget(central_widget)
        
        # 메인 레이아웃 (세로)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(10, 8, 10, 8)
        
        # 1. 헤더
        self.create_header(main_layout)
        
        # 2. 컨트롤 패널 영역 (기존과 동일)
        self.create_simplified_control_area(main_layout)
        
        # 3. 탭 기반 이미지 영역 (확대된 공간)
        self.create_tabbed_image_area(main_layout)
    
    def create_header(self, parent_layout):
        """헤더 생성 - 기존과 동일"""
        header_frame = QFrame()
        header_frame.setFixedHeight(55)
        header_frame.setStyleSheet("""
            QFrame {
                background-color: #FFFFFF;
                border: none;
                border-radius: 10px;
            }
        """)
        
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(20, 8, 20, 8)
        
        # 로고/제목 영역
        title_layout = QVBoxLayout()
        title_layout.setSpacing(2)
        
        title_label = QLabel("🎨 Hana Studio")
        title_label.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        title_label.setStyleSheet("color: #2C3E50; background: transparent;")

        title_layout.addWidget(title_label)

        header_layout.addLayout(title_layout)
        header_layout.addStretch()
        
        parent_layout.addWidget(header_frame)
    
    def create_simplified_control_area(self, parent_layout):
        """간소화된 컨트롤 패널 영역 - 기존과 동일"""
        control_container = QFrame()
        control_container.setFixedHeight(240)
        control_container.setStyleSheet("""
            QFrame {
                border: 2px solid #DEE2E6;
                border-radius: 10px;
                background-color: #FFFFFF;
            }
        """)
        
        # 가로 레이아웃
        control_layout = QHBoxLayout(control_container)
        control_layout.setSpacing(20)
        control_layout.setContentsMargins(15, 8, 15, 8)
        
        # 각 패널들 생성
        self.file_panel = FileSelectionPanel()
        self.print_mode_panel = PrintModePanel()
        self.print_quantity_panel = PrintQuantityPanel()
        self.printer_panel = PrinterPanel()
        self.progress_panel = ProgressPanel()
        self.log_panel = LogPanel()
        
        # 패널들 리스트
        panels = [
            self.file_panel, 
            self.print_mode_panel,
            self.print_quantity_panel, 
            self.printer_panel, 
            self.progress_panel, 
            self.log_panel
        ]
        
        # 각 패널별로 개별 너비 설정
        panel_widths = {
            self.file_panel: 260,
            self.print_mode_panel: 220,
            self.print_quantity_panel: 220,
            self.printer_panel: 220,
            self.progress_panel: 220,
            self.log_panel: 260
        }
        
        for panel in panels:
            panel_width = panel_widths[panel]
            panel.setFixedWidth(panel_width)
            control_layout.addWidget(panel)
        
        parent_layout.addWidget(control_container)
    
    def create_tabbed_image_area(self, parent_layout):
        """탭 기반 이미지 뷰어 영역 - 확대된 공간"""
        # 탭 위젯 생성
        self.image_tab_widget = ImageTabWidget()
        
        # 앞면 탭 생성
        front_tab = self.create_image_tab_content(is_front=True)
        self.image_tab_widget.addTab(front_tab, "📄 앞면")
        
        # 뒷면 탭 생성
        back_tab = self.create_image_tab_content(is_front=False)
        self.image_tab_widget.addTab(back_tab, "📄 뒷면")
        
        # 초기에는 뒷면 탭 숨김 (양면 모드가 활성화되면 표시)
        self.image_tab_widget.set_dual_side_enabled(False)
        
        # 탭 변경 시그널 연결
        self.image_tab_widget.tab_changed.connect(self._on_tab_changed)
        
        parent_layout.addWidget(self.image_tab_widget, 1)
    
    def create_image_tab_content(self, is_front: bool) -> QWidget:
        """개별 탭 컨텐츠 생성 - 확대된 이미지 뷰어"""
        tab_widget = QWidget()
        tab_layout = QVBoxLayout(tab_widget)
        tab_layout.setContentsMargins(20, 15, 20, 15)
        tab_layout.setSpacing(0)
        
        # 이미지 뷰어 행 생성 (기존 함수 재사용, 크기만 확대)
        image_row = self.create_unified_image_row_large(is_front)
        tab_layout.addWidget(image_row)
        
        return tab_widget
    
    def create_unified_image_row_large(self, is_front: bool) -> QFrame:
        """통일된 크기의 이미지 뷰어 행 생성"""
        row_frame = QFrame()
        row_frame.setStyleSheet("""
            QFrame {
                background-color: transparent;
                border: none;
            }
        """)
        
        # 행 레이아웃 (가로)
        row_layout = QHBoxLayout(row_frame)
        row_layout.setSpacing(20)
        row_layout.setContentsMargins(0, 0, 0, 0)
        
        # === 중앙 정렬을 위한 여백 ===
        row_layout.addStretch(1)
        
        # === 원본 이미지 뷰어 (통일된 크기) ===
        if is_front:
            self.front_original_viewer = ImageViewer("원본 이미지", enable_process_button=True)
            original_viewer = self.front_original_viewer
        else:
            self.back_original_viewer = ImageViewer("원본 이미지", enable_process_button=True)
            original_viewer = self.back_original_viewer
        
        # ✨ 통일된 크기 사용: ImageViewer가 자체적으로 크기 설정함
        original_viewer.setStyleSheet("""
            QWidget {
                background-color: #FFFFFF;
                border: 2px solid #DEE2E6;
                border-radius: 12px;
            }
        """)
        
        # === 프로페셔널한 화살표 1 ===
        arrow1 = self.create_professional_arrow("→", "#4A90E2")
        
        # === 통합 마스킹 미리보기 뷰어 (통일된 크기) ===
        if is_front:
            self.front_unified_mask_viewer = UnifiedMaskViewer("마스킹 미리보기")
            unified_mask_viewer = self.front_unified_mask_viewer
        else:
            self.back_unified_mask_viewer = UnifiedMaskViewer("마스킹 미리보기")
            unified_mask_viewer = self.back_unified_mask_viewer
        
        # ✨ 통일된 크기 사용: UnifiedMaskViewer가 자체적으로 크기 설정함
        unified_mask_viewer.setStyleSheet("""
            QWidget {
                background-color: #FFFFFF;
                border: 2px solid #28A745;
                border-radius: 12px;
            }
        """)
        
        # === 프로페셔널한 화살표 2 ===
        arrow2 = self.create_professional_arrow("←", "#28A745")
        
        # === 수동 마스킹 업로드 뷰어 (통일된 크기) ===
        if is_front:
            self.front_manual_mask_viewer = ImageViewer(
                "수동 마스킹\n클릭하여 업로드", 
                enable_click_upload=True, 
                show_orientation_buttons=False
            )
            manual_mask_viewer = self.front_manual_mask_viewer
        else:
            self.back_manual_mask_viewer = ImageViewer(
                "수동 마스킹\n클릭하여 업로드", 
                enable_click_upload=True, 
                show_orientation_buttons=False
            )
            manual_mask_viewer = self.back_manual_mask_viewer
        
        # ✨ 통일된 크기 사용: ImageViewer가 자체적으로 크기 설정함
        manual_mask_viewer.setStyleSheet("""
            QWidget {
                background-color: #FFFFFF;
                border: 2px solid #6C757D;
                border-radius: 12px;
            }
        """)
        
        # === 컴포넌트 배치 ===
        row_layout.addWidget(original_viewer)
        row_layout.addWidget(arrow1)
        row_layout.addWidget(unified_mask_viewer)
        row_layout.addWidget(arrow2)
        row_layout.addWidget(manual_mask_viewer)
        
        # === 중앙 정렬을 위한 여백 ===
        row_layout.addStretch(1)
        
        return row_frame
    
    def create_professional_arrow(self, arrow_text: str, color: str) -> QLabel:
        """프로페셔널한 화살표 생성"""
        arrow = QLabel(arrow_text)
        arrow.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        arrow.setStyleSheet(f"""
            QLabel {{
                color: {color};
                background-color: #F8F9FA;
                border: 2px solid {color};
                border-radius: 25px;
                padding: 8px;
            }}
        """)
        arrow.setAlignment(Qt.AlignmentFlag.AlignCenter)
        arrow.setFixedSize(50, 50)
        return arrow
    
    def _on_tab_changed(self, index: int):
        """탭 변경 시 처리"""
        self.current_tab_index = index
        # 필요시 추가 로직 구현 (예: 상태 업데이트)
    
    def set_dual_side_enabled(self, enabled: bool):
        """양면 모드 활성화/비활성화"""
        self.image_tab_widget.set_dual_side_enabled(enabled)
    
    def get_current_tab_index(self) -> int:
        """현재 선택된 탭 인덱스 반환"""
        return self.current_tab_index
    
    def set_current_tab(self, index: int):
        """탭 선택"""
        if index < self.image_tab_widget.count():
            self.image_tab_widget.setCurrentIndex(index)
    
    # 컴포넌트 접근을 위한 속성들 - 기존과 동일
    @property
    def components(self):
        """모든 UI 컴포넌트에 대한 접근 - 기존 함수명 유지"""
        return {
            'file_panel': self.file_panel,
            'print_mode_panel': self.print_mode_panel,
            'print_quantity_panel': self.print_quantity_panel,
            'printer_panel': self.printer_panel,
            'progress_panel': self.progress_panel,
            'log_panel': self.log_panel,
            
            # 원본 이미지 뷰어들 (방향 선택 + 배경제거 버튼 포함)
            'front_original_viewer': self.front_original_viewer,
            'back_original_viewer': self.back_original_viewer,
            
            # 통합 마스킹 미리보기 뷰어들
            'front_unified_mask_viewer': self.front_unified_mask_viewer,
            'back_unified_mask_viewer': self.back_unified_mask_viewer,
            
            # 수동 마스킹 업로드 뷰어들
            'front_manual_mask_viewer': self.front_manual_mask_viewer,
            'back_manual_mask_viewer': self.back_manual_mask_viewer,
            
            # 하위 호환성을 위한 별칭들 - 기존 함수명 유지
            'front_result_viewer': self.front_unified_mask_viewer,
            'back_result_viewer': self.back_unified_mask_viewer,
            'front_auto_result_viewer': self.front_unified_mask_viewer,
            'back_auto_result_viewer': self.back_unified_mask_viewer,
            
            # 탭 관련 추가
            'image_tab_widget': self.image_tab_widget,
            
            # 기타
            'progress_bar': self.progress_panel.progress_bar,
            'status_label': self.progress_panel.status_label,
            'log_text': self.log_panel.log_text
        }