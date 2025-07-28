"""
ui/simple_loading.py
즉시 표시되는 간단한 로딩 윈도우 - 멀티스레드 지원 + 아이콘 표시
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar
from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtGui import QFont, QPixmap
import os


class InitializationThread(QThread):
    """초기화 작업을 백그라운드에서 수행하는 스레드"""
    
    progress_update = Signal(str)  # 진행 상황 메시지
    finished = Signal()           # 완료 시그널
    error = Signal(str)          # 오류 시그널
    
    def run(self):
        """백그라운드 초기화 작업"""
        try:
            import time
            
            # 1단계: 기본 설정 로딩
            self.progress_update.emit("기본 설정 로딩 중...")
            time.sleep(0.3)  # UI 응답성을 위한 짧은 대기
            
            from config import config, AppConstants, get_resource_path
            
            # 2단계: UI 스타일 로딩
            self.progress_update.emit("UI 테마 로딩 중...")
            time.sleep(0.2)
            
            from ui.styles import get_light_palette
            
            # 3단계: 다이얼로그 모듈 로딩
            self.progress_update.emit("AI 엔진 모듈 준비 중...")
            time.sleep(0.2)
            
            from ui.installation_dialog import InstallationDialog
            
            # 4단계: 완료
            self.progress_update.emit("준비 완료!")
            time.sleep(0.3)
            
            self.finished.emit()
            
        except Exception as e:
            self.error.emit(str(e))


class SimpleLoadingWindow(QWidget):
    """즉시 표시되는 간단한 로딩 윈도우 - 아이콘 + 멀티스레드"""
    
    def __init__(self):
        super().__init__()
        self.init_thread = None
        self._setup_ui()
        self._setup_window()
        self._start_initialization()
    
    def _setup_window(self):
        """윈도우 기본 설정"""
        self.setWindowTitle("Hana Studio")
        self.setFixedSize(380, 140)  # 아이콘 공간을 위해 폭 증가
        
        # 화면 중앙에 배치
        from PySide6.QtWidgets import QApplication
        screen = QApplication.primaryScreen().geometry()
        x = (screen.width() - self.width()) // 2
        y = (screen.height() - self.height()) // 2
        self.move(x, y)
        
        # 항상 위에 표시
        self.setWindowFlags(
            Qt.WindowType.WindowStaysOnTopHint | 
            Qt.WindowType.FramelessWindowHint
        )
        
        # 스타일 설정
        self.setStyleSheet("""
            QWidget {
                background-color: white;
                border: 2px solid #4A90E2;
                border-radius: 12px;
            }
            QLabel {
                color: #2C3E50;
                background: transparent;
                border: none;
            }
            QProgressBar {
                border: none;
                background-color: #E9ECEF;
                border-radius: 6px;
                height: 8px;
            }
            QProgressBar::chunk {
                background-color: #4A90E2;
                border-radius: 6px;
            }
        """)
    
    def _setup_ui(self):
        """UI 구성 - 아이콘 포함"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 15, 20, 15)
        layout.setSpacing(12)
        
        # 헤더 영역 (아이콘 + 제목)
        header_layout = QHBoxLayout()
        header_layout.setSpacing(12)
        
        # 🎨 아이콘 표시
        self.icon_label = QLabel()
        self._load_icon()
        header_layout.addWidget(self.icon_label)
        
        # 제목
        title = QLabel("Hana Studio")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        # 상태 메시지
        self.status_label = QLabel("시작 중...")
        self.status_label.setFont(QFont("Segoe UI", 10))
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # 진행바 (애니메이션용)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # 무한 애니메이션
        
        layout.addLayout(header_layout)
        layout.addWidget(self.status_label)
        layout.addWidget(self.progress_bar)
    
    def _load_icon(self):
        """hana.ico 아이콘 로딩"""
        try:
            from config import get_resource_path
            icon_path = get_resource_path("hana.ico")
            
            if os.path.exists(icon_path):
                pixmap = QPixmap(icon_path)
                # 적절한 크기로 스케일링
                scaled_pixmap = pixmap.scaled(
                    48, 48, 
                    Qt.AspectRatioMode.KeepAspectRatio, 
                    Qt.TransformationMode.SmoothTransformation
                )
                self.icon_label.setPixmap(scaled_pixmap)
                print(f"✅ 아이콘 로드 성공: {icon_path}")
            else:
                # 아이콘이 없으면 이모지 사용
                self.icon_label.setText("🎨")
                self.icon_label.setFont(QFont("Segoe UI", 32))
                self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                print(f"⚠️ 아이콘 파일 없음, 이모지 사용: {icon_path}")
                
        except Exception as e:
            # 오류 시 기본 이모지
            self.icon_label.setText("🎨")
            self.icon_label.setFont(QFont("Segoe UI", 32))
            self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            print(f"❌ 아이콘 로드 실패: {e}")
        
        # 아이콘 레이블 크기 고정
        self.icon_label.setFixedSize(48, 48)
    
    def _start_initialization(self):
        """멀티스레드로 초기화 시작"""
        self.init_thread = InitializationThread()
        
        # 시그널 연결
        self.init_thread.progress_update.connect(self.update_status)
        self.init_thread.finished.connect(self._on_initialization_finished)
        self.init_thread.error.connect(self._on_initialization_error)
        
        # 스레드 시작
        self.init_thread.start()
    
    def update_status(self, message):
        """상태 메시지 업데이트 (스레드 안전)"""
        self.status_label.setText(message)
        # processEvents 호출하지 않음 (스레드에서 호출하면 안전하지 않음)
    
    def _on_initialization_finished(self):
        """초기화 완료 처리"""
        self.update_status("준비 완료!")
        
        # 잠시 대기 후 다음 단계로
        QTimer.singleShot(500, self._proceed_to_next_step)
    
    def _on_initialization_error(self, error_msg):
        """초기화 오류 처리"""
        self.update_status(f"오류: {error_msg}")
        self.progress_bar.setStyleSheet("""
            QProgressBar::chunk {
                background-color: #E74C3C;
            }
        """)
        
        # 3초 후 종료
        QTimer.singleShot(3000, self.close)
    
    def _proceed_to_next_step(self):
        """다음 단계로 진행 (설치 다이얼로그 표시)"""
        try:
            # 설치 다이얼로그 생성 및 표시
            from ui.installation_dialog import InstallationDialog
            
            dialog = InstallationDialog()
            dialog.show()
            dialog.raise_()
            dialog.activateWindow()
            
            # 로딩 윈도우 숨기기
            self.hide()
            
            # 다이얼로그 실행 및 결과 처리
            result = dialog.exec()
            
            if result == dialog.DialogCode.Accepted:
                # AI 모델 로딩 완료, 메인 윈도우 표시
                self._show_main_window()
            else:
                # 취소됨, 프로그램 종료
                import sys
                sys.exit(1)
                
        except Exception as e:
            print(f"다음 단계 진행 오류: {e}")
            import traceback
            traceback.print_exc()
            self.close()
    
    def _show_main_window(self):
        """메인 윈도우 표시"""
        try:
            print("메인 윈도우 생성 중...")
            from hana_studio import HanaStudio
            
            window = HanaStudio()
            window.show()
            
            # 로딩 윈도우 완전히 닫기
            self.close()
            
            print("🎉 Hana Studio 시작 완료!")
            
        except Exception as e:
            print(f"메인 윈도우 표시 오류: {e}")
            import traceback
            traceback.print_exc()
            
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(
                None,
                "시작 오류",
                f"메인 윈도우 생성 중 오류가 발생했습니다:\n\n{str(e)}"
            )
            import sys
            sys.exit(1)
    
    def closeEvent(self, event):
        """윈도우 닫기 시 스레드 정리"""
        if self.init_thread and self.init_thread.isRunning():
            self.init_thread.quit()
            self.init_thread.wait(1000)
        
        event.accept()