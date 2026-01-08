"""
ui/main_window.py 수정
로그 패널을 위치 조정 패널로 교체
"""

from PySide6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QProgressBar, QWidget,
    QFrame, QLabel, QSplitter, QScrollArea, QTabWidget, QApplication
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QFont

from .components.image_viewer import ImageViewer, UnifiedMaskViewer, UNIFIED_VIEWER_WIDTH, UNIFIED_VIEWER_HEIGHT
from .components.control_panels import (
    FileSelectionPanel, PrintModePanel,
    PrinterPanel, ProgressPanel, LogPanel, PrintQuantityPanel,
    PositionAdjustPanel  # 새로 추가
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
    """메인 윈도우 UI 구성 - 위치 조정 패널 적용"""
    
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
        QApplication.processEvents()

        # 메인 레이아웃 (세로)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(10, 8, 10, 8)

        # 1. 헤더 제거됨 - 제목표시줄에 프로그램명이 이미 표시됨

        # 2. 컨트롤 패널 영역 (위치 조정 패널로 교체)
        self.create_control_area_with_position(main_layout)
        QApplication.processEvents()

        # 3. 탭 기반 이미지 영역
        self.create_tabbed_image_area(main_layout)
        QApplication.processEvents()
    
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
        
        title_label = QLabel("[DESIGN] Hana Studio")
        title_label.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        title_label.setStyleSheet("color: #2C3E50; background: transparent;")

        title_layout.addWidget(title_label)

        header_layout.addLayout(title_layout)
        header_layout.addStretch()
        
        parent_layout.addWidget(header_frame)
    
    def create_control_area_with_position(self, parent_layout):
        """위치 조정 패널이 포함된 컨트롤 영역 - QTimer 지연 초기화"""
        self.control_container = QFrame()
        self.control_container.setFixedHeight(240)
        self.control_container.setStyleSheet("""
            QFrame {
                border: 2px solid #DEE2E6;
                border-radius: 10px;
                background-color: #FFFFFF;
            }
        """)

        # 가로 레이아웃
        self.control_layout = QHBoxLayout(self.control_container)
        self.control_layout.setSpacing(20)
        self.control_layout.setContentsMargins(15, 8, 15, 8)

        # 패널 너비 설정 저장
        self.panel_widths = [260, 220, 180, 220, 220, 220]

        # 지연 초기화를 위한 상태 변수
        self._panels_initialized = False
        self._panel_init_step = 0

        parent_layout.addWidget(self.control_container)

    def initialize_panels_deferred(self):
        """패널들을 QTimer로 단계별 지연 생성 - 응답없음 방지"""
        if self._panels_initialized:
            return

        # 첫 번째 패널 생성 시작
        self._panel_init_step = 0
        self._create_next_panel()

    def _create_next_panel(self):
        """다음 패널 생성 (QTimer 체인)"""
        step = self._panel_init_step

        if step == 0:
            self.file_panel = FileSelectionPanel()
            self.file_panel.setFixedWidth(self.panel_widths[0])
            self.control_layout.addWidget(self.file_panel)
            print("[PANEL] 1/7 파일선택 패널 생성")
        elif step == 1:
            self.print_mode_panel = PrintModePanel()
            self.print_mode_panel.setFixedWidth(self.panel_widths[1])
            self.control_layout.addWidget(self.print_mode_panel)
            print("[PANEL] 2/7 인쇄설정 패널 생성")
        elif step == 2:
            self.position_panel = PositionAdjustPanel()
            self.position_panel.setFixedWidth(self.panel_widths[2])
            self.control_layout.addWidget(self.position_panel)
            print("[PANEL] 3/7 위치조정 패널 생성")
        elif step == 3:
            self.print_quantity_panel = PrintQuantityPanel()
            self.print_quantity_panel.setFixedWidth(self.panel_widths[3])
            self.control_layout.addWidget(self.print_quantity_panel)
            print("[PANEL] 4/7 인쇄매수 패널 생성")
        elif step == 4:
            self.printer_panel = PrinterPanel()
            self.printer_panel.setFixedWidth(self.panel_widths[4])
            self.control_layout.addWidget(self.printer_panel)
            print("[PANEL] 5/7 프린터연동 패널 생성")
        elif step == 5:
            self.progress_panel = ProgressPanel()
            self.progress_panel.setFixedWidth(self.panel_widths[5])
            self.control_layout.addWidget(self.progress_panel)
            print("[PANEL] 6/7 진행상황 패널 생성")
        elif step == 6:
            # 로그 패널은 숨김 처리 (백그라운드에서만 존재)
            self.log_panel = LogPanel()
            print("[PANEL] 7/7 로그 패널 생성")
            self._panels_initialized = True
            print("[OK] 모든 패널 초기화 완료")
            return  # 완료

        # 다음 패널 생성 예약 (0ms = 이벤트 루프에 양보)
        self._panel_init_step += 1
        QTimer.singleShot(0, self._create_next_panel)
    
    def create_tabbed_image_area(self, parent_layout):
        """탭 기반 이미지 뷰어 영역 - 지연 초기화 준비"""
        # 탭 위젯 생성 (가벼움)
        self.image_tab_widget = ImageTabWidget()

        # 탭 변경 시그널 연결
        self.image_tab_widget.tab_changed.connect(self._on_tab_changed)

        parent_layout.addWidget(self.image_tab_widget, 1)

        # 이미지 뷰어 초기화 상태
        self._viewers_initialized = False
        self._viewer_init_step = 0

    def initialize_viewers_deferred(self):
        """이미지 뷰어들을 QTimer로 단계별 지연 생성"""
        if self._viewers_initialized:
            return

        self._viewer_init_step = 0
        self._create_next_viewer()

    def _create_next_viewer(self):
        """다음 이미지 뷰어 생성 (QTimer 체인)"""
        step = self._viewer_init_step

        if step == 0:
            # 앞면 탭 컨테이너 생성
            self._front_tab = QWidget()
            self._front_tab_layout = QVBoxLayout(self._front_tab)
            self._front_tab_layout.setContentsMargins(20, 15, 20, 15)
            self._front_tab_layout.setSpacing(0)
            self.image_tab_widget.addTab(self._front_tab, "📄 앞면")
            print("[VIEWER] 1/8 앞면 탭 생성")
        elif step == 1:
            # 앞면 원본 뷰어
            self.front_original_viewer = ImageViewer("원본 이미지", enable_process_button=True)
            print("[VIEWER] 2/8 앞면 원본 뷰어 생성")
        elif step == 2:
            # 앞면 통합 마스킹 뷰어
            self.front_unified_mask_viewer = UnifiedMaskViewer("마스킹 미리보기")
            print("[VIEWER] 3/8 앞면 마스킹 뷰어 생성")
        elif step == 3:
            # 앞면 수동 마스킹 뷰어
            self.front_manual_mask_viewer = ImageViewer(
                "수동 마스킹\n클릭하여 업로드",
                enable_click_upload=True,
                show_orientation_buttons=False
            )
            print("[VIEWER] 4/8 앞면 수동 마스킹 뷰어 생성")
            # 앞면 탭 조립
            self._assemble_front_tab()
        elif step == 4:
            # 뒷면 탭 컨테이너 생성
            self._back_tab = QWidget()
            self._back_tab_layout = QVBoxLayout(self._back_tab)
            self._back_tab_layout.setContentsMargins(20, 15, 20, 15)
            self._back_tab_layout.setSpacing(0)
            self.image_tab_widget.addTab(self._back_tab, "📄 뒷면")
            print("[VIEWER] 5/8 뒷면 탭 생성")
        elif step == 5:
            # 뒷면 원본 뷰어
            self.back_original_viewer = ImageViewer("원본 이미지", enable_process_button=True)
            print("[VIEWER] 6/8 뒷면 원본 뷰어 생성")
        elif step == 6:
            # 뒷면 통합 마스킹 뷰어
            self.back_unified_mask_viewer = UnifiedMaskViewer("마스킹 미리보기")
            print("[VIEWER] 7/8 뒷면 마스킹 뷰어 생성")
        elif step == 7:
            # 뒷면 수동 마스킹 뷰어
            self.back_manual_mask_viewer = ImageViewer(
                "수동 마스킹\n클릭하여 업로드",
                enable_click_upload=True,
                show_orientation_buttons=False
            )
            print("[VIEWER] 8/8 뒷면 수동 마스킹 뷰어 생성")
            # 뒷면 탭 조립 및 완료
            self._assemble_back_tab()
            self.image_tab_widget.set_dual_side_enabled(False)
            self._viewers_initialized = True
            print("[OK] 모든 이미지 뷰어 초기화 완료")
            return

        self._viewer_init_step += 1
        QTimer.singleShot(0, self._create_next_viewer)

    def _assemble_front_tab(self):
        """앞면 탭 레이아웃 조립"""
        row_frame = QFrame()
        row_frame.setStyleSheet("QFrame { background-color: transparent; border: none; }")
        row_layout = QHBoxLayout(row_frame)
        row_layout.setSpacing(20)
        row_layout.setContentsMargins(0, 0, 0, 0)

        row_layout.addStretch(1)

        # 스타일 적용
        self.front_original_viewer.setStyleSheet("""
            QWidget { background-color: #FFFFFF; border: 2px solid #DEE2E6; border-radius: 12px; }
        """)
        self.front_unified_mask_viewer.setStyleSheet("""
            QWidget { background-color: #FFFFFF; border: 2px solid #28A745; border-radius: 12px; }
        """)
        self.front_manual_mask_viewer.setStyleSheet("""
            QWidget { background-color: #FFFFFF; border: 2px solid #6C757D; border-radius: 12px; }
        """)

        row_layout.addWidget(self.front_original_viewer)
        row_layout.addWidget(self.create_professional_arrow("→", "#4A90E2"))
        row_layout.addWidget(self.front_unified_mask_viewer)
        row_layout.addWidget(self.create_professional_arrow("←", "#28A745"))
        row_layout.addWidget(self.front_manual_mask_viewer)
        row_layout.addStretch(1)

        self._front_tab_layout.addWidget(row_frame)

    def _assemble_back_tab(self):
        """뒷면 탭 레이아웃 조립"""
        row_frame = QFrame()
        row_frame.setStyleSheet("QFrame { background-color: transparent; border: none; }")
        row_layout = QHBoxLayout(row_frame)
        row_layout.setSpacing(20)
        row_layout.setContentsMargins(0, 0, 0, 0)

        row_layout.addStretch(1)

        # 스타일 적용
        self.back_original_viewer.setStyleSheet("""
            QWidget { background-color: #FFFFFF; border: 2px solid #DEE2E6; border-radius: 12px; }
        """)
        self.back_unified_mask_viewer.setStyleSheet("""
            QWidget { background-color: #FFFFFF; border: 2px solid #28A745; border-radius: 12px; }
        """)
        self.back_manual_mask_viewer.setStyleSheet("""
            QWidget { background-color: #FFFFFF; border: 2px solid #6C757D; border-radius: 12px; }
        """)

        row_layout.addWidget(self.back_original_viewer)
        row_layout.addWidget(self.create_professional_arrow("→", "#4A90E2"))
        row_layout.addWidget(self.back_unified_mask_viewer)
        row_layout.addWidget(self.create_professional_arrow("←", "#28A745"))
        row_layout.addWidget(self.back_manual_mask_viewer)
        row_layout.addStretch(1)

        self._back_tab_layout.addWidget(row_frame)
    
    def create_image_tab_content(self, is_front: bool) -> QWidget:
        """개별 탭 컨텐츠 생성"""
        tab_widget = QWidget()
        tab_layout = QVBoxLayout(tab_widget)
        tab_layout.setContentsMargins(20, 15, 20, 15)
        tab_layout.setSpacing(0)
        
        # 이미지 뷰어 행 생성
        image_row = self.create_unified_image_row_large(is_front)
        tab_layout.addWidget(image_row)
        
        return tab_widget
    
    def create_unified_image_row_large(self, is_front: bool) -> QFrame:
        """통일된 크기의 이미지 뷰어 행 생성 - 기존과 동일"""
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
        
        # === 원본 이미지 뷰어 ===
        if is_front:
            self.front_original_viewer = ImageViewer("원본 이미지", enable_process_button=True)
            original_viewer = self.front_original_viewer
        else:
            self.back_original_viewer = ImageViewer("원본 이미지", enable_process_button=True)
            original_viewer = self.back_original_viewer
        
        original_viewer.setStyleSheet("""
            QWidget {
                background-color: #FFFFFF;
                border: 2px solid #DEE2E6;
                border-radius: 12px;
            }
        """)
        
        # === 화살표 1 ===
        arrow1 = self.create_professional_arrow("→", "#4A90E2")
        
        # === 통합 마스킹 미리보기 뷰어 ===
        if is_front:
            self.front_unified_mask_viewer = UnifiedMaskViewer("마스킹 미리보기")
            unified_mask_viewer = self.front_unified_mask_viewer
        else:
            self.back_unified_mask_viewer = UnifiedMaskViewer("마스킹 미리보기")
            unified_mask_viewer = self.back_unified_mask_viewer
        
        unified_mask_viewer.setStyleSheet("""
            QWidget {
                background-color: #FFFFFF;
                border: 2px solid #28A745;
                border-radius: 12px;
            }
        """)
        
        # === 화살표 2 ===
        arrow2 = self.create_professional_arrow("←", "#28A745")
        
        # === 수동 마스킹 업로드 뷰어 ===
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
    
    # 컴포넌트 접근을 위한 속성들
    @property
    def components(self):
        """모든 UI 컴포넌트에 대한 접근"""
        return {
            'file_panel': self.file_panel,
            'print_mode_panel': self.print_mode_panel,
            'print_quantity_panel': self.print_quantity_panel,
            'printer_panel': self.printer_panel,
            'progress_panel': self.progress_panel,
            'log_panel': self.log_panel,  # 숨김 처리되었지만 유지
            'position_panel': self.position_panel,  # 새로 추가
            
            # 원본 이미지 뷰어들
            'front_original_viewer': self.front_original_viewer,
            'back_original_viewer': self.back_original_viewer,
            
            # 통합 마스킹 미리보기 뷰어들
            'front_unified_mask_viewer': self.front_unified_mask_viewer,
            'back_unified_mask_viewer': self.back_unified_mask_viewer,
            
            # 수동 마스킹 업로드 뷰어들
            'front_manual_mask_viewer': self.front_manual_mask_viewer,
            'back_manual_mask_viewer': self.back_manual_mask_viewer,
            
            # 하위 호환성을 위한 별칭들
            'front_result_viewer': self.front_unified_mask_viewer,
            'back_result_viewer': self.back_unified_mask_viewer,
            'front_auto_result_viewer': self.front_unified_mask_viewer,
            'back_auto_result_viewer': self.back_unified_mask_viewer,
            
            # 탭 관련
            'image_tab_widget': self.image_tab_widget,
            
            # 기타
            'status_label': self.progress_panel.status_label,
            'log_text': self.log_panel.log_text
        }