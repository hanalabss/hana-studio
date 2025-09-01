"""
프린터 선택 대화상자 - 프린터 연결 및 선택
사용 가능한 프린터 목록 표시 UI
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QListWidget, 
    QListWidgetItem, QPushButton, QProgressBar, QFrame, QMessageBox, 
    QWidget, QSizePolicy, QSpacerItem
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QSize
from PySide6.QtGui import QFont, QPixmap, QPainter, QColor, QLinearGradient
from typing import List, Optional

from printer.printer_discovery import PrinterInfo, discover_available_printers, get_printer_display_name
from .modern_button import ModernButton


class PrinterDiscoveryThread(QThread):
    """프린터 검색 스레드"""
    printers_found = Signal(list, str)
    progress_update = Signal(str)
    finished_discovery = Signal()
    
    def __init__(self, dll_path: str):
        super().__init__()
        self.dll_path = dll_path
    
    def run(self):
        """프린터 검색 실행"""
        try:
            self.progress_update.emit("프린터 검색 중...")
            printers, summary = discover_available_printers(self.dll_path)
            self.printers_found.emit(printers, summary)
        except Exception as e:
            self.printers_found.emit([], f"검색 오류: {e}")
        finally:
            self.finished_discovery.emit()


class PrinterItem(QFrame):
    """프린터 아이템 위젯"""
    
    def __init__(self, printer: PrinterInfo, index: int, parent_dialog):
        super().__init__()
        self.printer = printer
        self.index = index
        self.parent_dialog = parent_dialog
        self.is_selected = False
        
        self.setFixedHeight(52)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._setup_ui()
        self._apply_style(False)
    
    def _setup_ui(self):
        """UI 구성"""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(12)
        
        # 연결 타입 아이콘
        self.type_label = QLabel("🔌" if self.printer.connection_type == "USB" else "🌐")
        self.type_label.setFont(QFont("Arial", 8))
        self.type_label.setFixedSize(12, 12)
        if self.printer.connection_type == "TCP":
            self.type_label.setStyleSheet("color: #10B981; background: transparent;")  # 초록색
        else:
            self.type_label.setStyleSheet("color: #3B82F6; background: transparent;")  # 파란색
        
        # 프린터 이름
        self.name_label = QLabel(self.printer.name)
        self.name_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Medium))
        self.name_label.setStyleSheet("color: #1F2937; background: transparent;")
        
        # 연결 타입 텍스트
        self.connection_label = QLabel(self.printer.connection_type)
        self.connection_label.setFont(QFont("Segoe UI", 9))
        self.connection_label.setStyleSheet("color: #6B7280; background: transparent;")
        
        # 선택 체크 마크
        self.check_label = QLabel()
        self.check_label.setFixedSize(16, 16)
        self.check_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        layout.addWidget(self.type_label)
        layout.addWidget(self.name_label)
        layout.addStretch()
        layout.addWidget(self.connection_label)
        layout.addWidget(self.check_label)
    
    def _apply_style(self, selected: bool):
        """스타일 적용 - 간단하고 깔끔한 디자인"""
        if selected:
            self.setStyleSheet("""
                QFrame {
                    background-color: #EBF4FF;
                    border-radius: 4px;
                }
            """)
            self.check_label.setText("✓")
            self.check_label.setStyleSheet("color: #3B82F6; font-weight: bold; background: transparent;")
            self.is_selected = True
        else:
            self.setStyleSheet("""
                QFrame {
                    background-color: #FFFFFF;
                    border-radius: 4px;
                }
                QFrame:hover {
                    background-color: #F8F9FA;
                }
            """)
            self.check_label.setText("")
            self.is_selected = False
    
    def mousePressEvent(self, event):
        """마우스 클릭 처리"""
        self.parent_dialog.select_printer(self.index)
    
    def set_selected(self, selected: bool):
        """선택 상태 설정"""
        self._apply_style(selected)


class PrinterSelectionDialog(QDialog):
    """프린터 선택 대화상자"""
    
    def __init__(self, dll_path: str, parent=None):
        super().__init__(parent)
        self.dll_path = dll_path
        self.selected_printer = None
        self.printers = []
        self.printer_items = []
        self.discovery_thread = None
        self.selected_index = -1
        
        self._setup_dialog()
        self._setup_ui()
        self._start_discovery()
    
    def _setup_dialog(self):
        """다이얼로그 설정"""
        self.setWindowTitle("프린터 선택 - Hana Studio")
        self.setModal(True)
        self.setFixedSize(460, 380)
        self.setStyleSheet("""
            QDialog {
                background-color: #F9FAFB;
                font-family: 'Segoe UI', system-ui, sans-serif;
            }
        """)
    
    def _setup_ui(self):
        """UI 구성"""
        layout = QVBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 헤더
        self._create_header(layout)
        
        # 컨텐츠 영역
        content_widget = QWidget()
        content_widget.setStyleSheet("background-color: #FFFFFF;")
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(24, 20, 24, 20)
        content_layout.setSpacing(16)
        
        # 상태 섹션
        self._create_status_section(content_layout)
        
        # 프린터 리스트
        self._create_printer_list(content_layout)
        
        # 프로그레스바
        self._create_progress_bar(content_layout)
        
        layout.addWidget(content_widget, 1)
        
        # 푸터 버튼
        self._create_footer(layout)
    
    def _create_header(self, parent_layout):
        """헤더 생성 - 심플하고 모던한 디자인"""
        header = QFrame()
        header.setFixedHeight(70)
        header.setStyleSheet("""
            QFrame {
                background-color: #FFFFFF;
                border: none;
            }
        """)
        
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(24, 16, 24, 12)
        header_layout.setSpacing(4)
        
        title = QLabel("프린터 선택")
        title.setFont(QFont("Segoe UI", 16, QFont.Weight.DemiBold))
        title.setStyleSheet("color: #111827; background: transparent;")
        
        subtitle = QLabel("사용 가능한 RTAI 프린터 목록")
        subtitle.setFont(QFont("Segoe UI", 10))
        subtitle.setStyleSheet("color: #6B7280; background: transparent;")
        
        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        
        parent_layout.addWidget(header)
    
    def _create_status_section(self, parent_layout):
        """상태 섹션 - 현재 검색 상태 표시"""
        self.status_frame = QFrame()
        self.status_frame.setFixedHeight(36)
        self.status_frame.setStyleSheet("""
            QFrame {
                background-color: #F8F9FA;
                border-radius: 4px;
            }
        """)
        
        status_layout = QHBoxLayout(self.status_frame)
        status_layout.setContentsMargins(12, 8, 12, 8)
        
        self.status_icon = QLabel("⚡")
        self.status_icon.setFont(QFont("Arial", 8))
        self.status_icon.setStyleSheet("color: #F59E0B; background: transparent;")
        
        self.status_text = QLabel("프린터 검색 중...")
        self.status_text.setFont(QFont("Segoe UI", 9))
        self.status_text.setStyleSheet("color: #374151; background: transparent;")
        
        status_layout.addWidget(self.status_icon)
        status_layout.addWidget(self.status_text)
        status_layout.addStretch()
        
        parent_layout.addWidget(self.status_frame)
    
    def _create_printer_list(self, parent_layout):
        """프린터 리스트 - 스크롤 가능한 리스트"""
        # 리스트 라벨
        list_label = QLabel("검색된 프린터")
        list_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Medium))
        list_label.setStyleSheet("color: #374151; background: transparent; margin-bottom: 8px;")
        
        # 리스트 컨테이너 - 부드러운 테두리
        self.list_container = QFrame()
        self.list_container.setMinimumHeight(140)
        self.list_container.setMaximumHeight(180)
        self.list_container.setStyleSheet("""
            QFrame {
                background-color: #FAFAFA;
                border-radius: 6px;
            }
        """)
        
        self.list_layout = QVBoxLayout(self.list_container)
        self.list_layout.setContentsMargins(8, 8, 8, 8)
        self.list_layout.setSpacing(4)
        
        # 로딩 메시지
        self.loading_label = QLabel("프린터 검색 중...")
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.loading_label.setFont(QFont("Segoe UI", 9))
        self.loading_label.setStyleSheet("color: #9CA3AF; background: transparent; padding: 40px;")
        self.list_layout.addWidget(self.loading_label)
        
        parent_layout.addWidget(list_label)
        parent_layout.addWidget(self.list_container)
    
    def _create_progress_bar(self, parent_layout):
        """프로그레스바"""
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setFixedHeight(3)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: none;
                border-radius: 1px;
                background-color: #E5E7EB;
            }
            QProgressBar::chunk {
                background-color: #3B82F6;
                border-radius: 1px;
            }
        """)
        
        parent_layout.addWidget(self.progress_bar)
    
    def _create_footer(self, parent_layout):
        """푸터 버튼 영역 - 모던한 버튼 디자인"""
        footer = QFrame()
        footer.setFixedHeight(64)
        footer.setStyleSheet("""
            QFrame {
                background-color: #F9FAFB;
            }
        """)
        
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(24, 16, 24, 16)
        footer_layout.setSpacing(12)
        
        # 새로고침 버튼
        self.refresh_btn = ModernButton("새로고침")
        self.refresh_btn.setFixedSize(110, 32)
        self.refresh_btn.setEnabled(False)
        self.refresh_btn.clicked.connect(self._refresh_printers)
        
        footer_layout.addWidget(self.refresh_btn)
        footer_layout.addStretch()
        
        # 취소/선택 버튼
        self.cancel_btn = ModernButton("취소")
        self.cancel_btn.setFixedSize(70, 32)
        self.cancel_btn.clicked.connect(self.reject)
        
        self.select_btn = ModernButton("선택", primary=True)
        self.select_btn.setFixedSize(70, 32)
        self.select_btn.setEnabled(False)
        self.select_btn.clicked.connect(self.accept)
        
        footer_layout.addWidget(self.cancel_btn)
        footer_layout.addWidget(self.select_btn)
        
        parent_layout.addWidget(footer)
    
    def _start_discovery(self):
        """프린터 검색 시작"""
        self.discovery_thread = PrinterDiscoveryThread(self.dll_path)
        self.discovery_thread.printers_found.connect(self._on_printers_found)
        self.discovery_thread.progress_update.connect(self._update_status)
        self.discovery_thread.finished_discovery.connect(self._on_discovery_finished)
        self.discovery_thread.start()
    
    def _update_status(self, message: str):
        """상태 업데이트"""
        self.status_text.setText(message)
    
    def _on_printers_found(self, printers: List[PrinterInfo], summary: str):
        """프린터 발견 시 처리"""
        self.printers = printers
        
        # 기존 리스트 클리어
        self._clear_list()
        
        if printers:
            # 프린터 발견
            self.status_icon.setStyleSheet("color: #10B981; background: transparent;")
            self.status_text.setText(f"{len(printers)}개의 프린터를 찾았습니다")
            
            # 프린터 아이템 추가
            for i, printer in enumerate(printers):
                item = PrinterItem(printer, i, self)
                self.printer_items.append(item)
                self.list_layout.addWidget(item)
            
            # 스페이서 추가
            spacer = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
            self.list_layout.addItem(spacer)
            
            # 첫 번째 프린터 자동 선택
            if printers:
                self.select_printer(0)
        else:
            # 프린터 없음
            self.status_icon.setStyleSheet("color: #EF4444; background: transparent;")
            self.status_text.setText("프린터를 찾을 수 없습니다")
            
            no_printer_label = QLabel("사용 가능한 프린터가 없습니다\n프린터 연결 상태를 확인해주세요")
            no_printer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            no_printer_label.setFont(QFont("Segoe UI", 9))
            no_printer_label.setStyleSheet("color: #9CA3AF; background: transparent; padding: 30px;")
            self.list_layout.addWidget(no_printer_label)
    
    def _clear_list(self):
        """리스트 클리어"""
        while self.list_layout.count():
            child = self.list_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self.printer_items.clear()
    
    def _on_discovery_finished(self):
        """검색 완료"""
        self.progress_bar.setVisible(False)
        self.refresh_btn.setEnabled(True)
        
        if not self.printers:
            QMessageBox.information(
                self, 
                "알림", 
                "프린터를 찾을 수 없습니다.\n\n프린터 전원과 연결 상태를 확인해주세요."
            )
    
    def select_printer(self, index: int):
        """프린터 선택"""
        # 이전 선택 해제
        if self.selected_index >= 0 and self.selected_index < len(self.printer_items):
            self.printer_items[self.selected_index].set_selected(False)
        
        # 새로 선택
        self.selected_index = index
        self.selected_printer = self.printers[index]
        self.printer_items[index].set_selected(True)
        
        # 선택 버튼 활성화
        self.select_btn.setEnabled(True)
        
        # 상태 업데이트
        self.status_text.setText(f"선택됨: {self.selected_printer.name}")
    
    def _refresh_printers(self):
        """새로고침"""
        self._clear_list()
        self.selected_index = -1
        self.selected_printer = None
        self.select_btn.setEnabled(False)
        self.refresh_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        
        # 로딩 메시지
        self.loading_label = QLabel("프린터 검색 중...")
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.loading_label.setFont(QFont("Segoe UI", 9))
        self.loading_label.setStyleSheet("color: #9CA3AF; background: transparent; padding: 40px;")
        self.list_layout.addWidget(self.loading_label)
        
        self.status_icon.setStyleSheet("color: #F59E0B; background: transparent;")
        self.status_text.setText("프린터 검색 중...")
        
        # 기존 스레드 정지
        if self.discovery_thread and self.discovery_thread.isRunning():
            self.discovery_thread.quit()
            self.discovery_thread.wait()
        
        self._start_discovery()
    
    def get_selected_printer(self) -> Optional[PrinterInfo]:
        """선택된 프린터 반환"""
        return self.selected_printer
    
    def closeEvent(self, event):
        """대화상자 닫기 시 스레드 정리"""
        if self.discovery_thread and self.discovery_thread.isRunning():
            self.discovery_thread.quit()
            self.discovery_thread.wait()
        event.accept()


def show_printer_selection_dialog(dll_path: str, parent=None) -> Optional[PrinterInfo]:
    """프린터 선택 대화상자 표시 헬퍼 함수"""
    dialog = PrinterSelectionDialog(dll_path, parent)
    
    if dialog.exec() == QDialog.DialogCode.Accepted:
        return dialog.get_selected_printer()
    else:
        return None