"""
프린터 선택 다이얼로그 - 시작 시 프린터 선택
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QListWidget, 
    QListWidgetItem, QPushButton, QProgressBar, QTextEdit,
    QFrame, QMessageBox, QDialogButtonBox, QGroupBox
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QFont, QIcon, QPixmap
from typing import List, Optional

from printer.printer_discovery import PrinterInfo, discover_available_printers, get_printer_display_name
from .modern_button import ModernButton


class PrinterDiscoveryThread(QThread):
    """프린터 탐지 백그라운드 스레드"""
    printers_found = Signal(list, str)  # List[PrinterInfo], summary
    progress_update = Signal(str)
    finished_discovery = Signal()
    
    def __init__(self, dll_path: str):
        super().__init__()
        self.dll_path = dll_path
    
    def run(self):
        """스레드 실행"""
        try:
            self.progress_update.emit("🔍 프린터 탐지 중...")
            
            # 프린터 탐지 실행
            printers, summary = discover_available_printers(self.dll_path)
            
            self.printers_found.emit(printers, summary)
            
        except Exception as e:
            error_msg = f"❌ 프린터 탐지 실패: {e}"
            self.printers_found.emit([], error_msg)
        
        finally:
            self.finished_discovery.emit()


class PrinterSelectionDialog(QDialog):
    """프린터 선택 다이얼로그"""
    
    def __init__(self, dll_path: str, parent=None):
        super().__init__(parent)
        self.dll_path = dll_path
        self.selected_printer = None
        self.printers = []
        self.discovery_thread = None
        
        self.setWindowTitle("🖨️ Hana Studio - 프린터 선택")
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.WindowCloseButtonHint)
        self.setModal(True)
        self.resize(600, 500)
        
        self._setup_ui()
        self._start_discovery()
    
    def _setup_ui(self):
        """UI 구성"""
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(25, 25, 25, 25)
        
        # 헤더
        header_frame = QFrame()
        header_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                                           stop: 0 #4A90E2, stop: 1 #357ABD);
                border-radius: 10px;
                padding: 15px;
            }
        """)
        header_frame.setFixedHeight(100)
        
        header_layout = QVBoxLayout(header_frame)
        
        title_label = QLabel("🖨️ 프린터 선택")
        title_label.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        title_label.setStyleSheet("color: white; background: transparent;")
        
        subtitle_label = QLabel("Hana Studio를 사용하기 위해 연결된 Rtai 프린터를 선택해주세요.")
        subtitle_label.setFont(QFont("Segoe UI", 11))
        subtitle_label.setStyleSheet("color: rgba(255, 255, 255, 0.9); background: transparent;")
        
        header_layout.addWidget(title_label)
        header_layout.addWidget(subtitle_label)
        
        # 프린터 목록
        list_group = QGroupBox("🔍 사용 가능한 프린터")
        list_layout = QVBoxLayout(list_group)
        
        self.status_label = QLabel("프린터를 탐지하고 있습니다...")
        self.status_label.setStyleSheet("""
            color: #6C757D; 
            font-size: 13px;
            padding: 8px;
            background-color: #F8F9FA;
            border-radius: 5px;
        """)
        
        self.printer_list = QListWidget()
        self.printer_list.setMinimumHeight(200)
        self.printer_list.setStyleSheet("""
            QListWidget {
                background-color: #FFFFFF;
                border: 2px solid #E9ECEF;
                border-radius: 8px;
                padding: 5px;
                font-size: 14px;
            }
            QListWidget::item {
                padding: 12px;
                border-bottom: 1px solid #F8F9FA;
                border-radius: 4px;
                margin: 2px;
            }
            QListWidget::item:hover {
                background-color: #E3F2FD;
            }
            QListWidget::item:selected {
                background-color: #4A90E2;
                color: white;
            }
        """)
        self.printer_list.itemSelectionChanged.connect(self._on_printer_selected)
        
        # 진행바
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setFixedHeight(25)
        
        # 버튼들
        button_layout = QHBoxLayout()
        
        self.refresh_button = ModernButton("🔄 새로고침")
        self.refresh_button.clicked.connect(self._refresh_printers)
        
        self.cancel_button = ModernButton("취소")
        self.cancel_button.clicked.connect(self.reject)
        
        self.ok_button = ModernButton("선택 완료", primary=True)
        self.ok_button.clicked.connect(self.accept)
        self.ok_button.setEnabled(False)
        
        button_layout.addWidget(self.refresh_button)
        button_layout.addStretch()
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.ok_button)
        
        # 레이아웃 구성
        list_layout.addWidget(self.status_label)
        list_layout.addWidget(self.printer_list)
        
        layout.addWidget(header_frame)
        layout.addWidget(list_group)
        layout.addWidget(self.progress_bar)
        layout.addLayout(button_layout)
        
        # 초기 상태
        self.printer_list.setEnabled(False)
        self.refresh_button.setEnabled(False)
    
    def _start_discovery(self):
        """프린터 탐지 시작"""
        self.discovery_thread = PrinterDiscoveryThread(self.dll_path)
        self.discovery_thread.printers_found.connect(self._on_printers_found)
        self.discovery_thread.progress_update.connect(self._on_progress_update)
        self.discovery_thread.finished_discovery.connect(self._on_discovery_finished)
        self.discovery_thread.start()
    
    def _on_progress_update(self, message: str):
        """진행상황 업데이트"""
        self.status_label.setText(message)
    
    def _on_printers_found(self, printers: List[PrinterInfo], summary: str):
        """프린터 탐지 완료"""
        self.printers = printers
        self.status_label.setText(summary)
        
        self.printer_list.clear()
        
        if printers:
            for printer in printers:
                item = QListWidgetItem(get_printer_display_name(printer))
                item.setData(Qt.ItemDataRole.UserRole, printer)
                self.printer_list.addItem(item)
            
            self.printer_list.setCurrentRow(0)
            self.printer_list.setEnabled(True)
        else:
            no_printer_item = QListWidgetItem("❌ 연결된 프린터가 없습니다.")
            no_printer_item.setFlags(Qt.ItemFlag.NoItemFlags)
            self.printer_list.addItem(no_printer_item)
    
    def _on_discovery_finished(self):
        """탐지 완료"""
        self.progress_bar.setVisible(False)
        self.refresh_button.setEnabled(True)
        
        if self.printers:
            self.ok_button.setEnabled(True)
        else:
            QMessageBox.warning(
                self,
                "프린터 없음",
                "연결된 프린터가 없습니다.\n\n"
                "프린터 연결 상태를 확인하고 '새로고침'을 눌러주세요."
            )
    
    def _on_printer_selected(self):
        """프린터 선택 시"""
        current_item = self.printer_list.currentItem()
        if current_item:
            printer = current_item.data(Qt.ItemDataRole.UserRole)
            if printer:
                self.selected_printer = printer
                self.ok_button.setEnabled(True)
                self.status_label.setText(f"✅ 선택됨: {get_printer_display_name(printer)}")
    
    def _refresh_printers(self):
        """프린터 목록 새로고침"""
        self.printer_list.clear()
        self.printer_list.setEnabled(False)
        self.ok_button.setEnabled(False)
        self.refresh_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        
        self.status_label.setText("🔄 프린터 목록을 새로고침하고 있습니다...")
        
        if self.discovery_thread and self.discovery_thread.isRunning():
            self.discovery_thread.quit()
            self.discovery_thread.wait()
        
        self._start_discovery()
    
    def get_selected_printer(self) -> Optional[PrinterInfo]:
        """선택된 프린터 반환"""
        return self.selected_printer


def show_printer_selection_dialog(dll_path: str, parent=None) -> Optional[PrinterInfo]:
    """
    프린터 선택 다이얼로그 표시 (편의 함수)
    
    Returns:
        선택된 프린터 정보 또는 None (취소/오류 시)
    """
    dialog = PrinterSelectionDialog(dll_path, parent)
    
    if dialog.exec() == QDialog.DialogCode.Accepted:
        return dialog.get_selected_printer()
    else:
        return None