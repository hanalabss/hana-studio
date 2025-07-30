"""
ui/simple_loading.py - 완전히 재작성
즉시 표시되는 통합 로딩 윈도우 - 중복 실행 방지
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar, QPushButton, QApplication
from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtGui import QFont, QPixmap
import os
import time
import sys
from pathlib import Path


class InitializationThread(QThread):
    """초기화 작업 스레드"""
    
    progress_update = Signal(str)    # 진행 상황 메시지
    progress_percent = Signal(int)   # 진행률 퍼센트
    step_changed = Signal(str, str)  # (제목, 설명)
    finished = Signal()             # 완료 시그널
    error = Signal(str)            # 오류 시그널
    
    def __init__(self):
        super().__init__()
        self.cancelled = False
        self.model_name = None
        
    def cancel(self):
        """초기화 취소"""
        self.cancelled = True
    
    def run(self):
        """초기화 작업 실행"""
        try:
            # 1단계: 기본 설정 로딩
            self.step_changed.emit("기본 설정", "프로그램 구성요소를 준비하고 있습니다...")
            self.progress_percent.emit(10)
            time.sleep(0.3)
            
            if self.cancelled:
                return
                
            # config 모듈 로드
            from config import config, AppConstants, get_resource_path
            self.model_name = config.get('ai_model', 'isnet-general-use')
            
            # 2단계: UI 테마 로딩
            self.step_changed.emit("UI 테마", "사용자 인터페이스를 준비하고 있습니다...")
            self.progress_percent.emit(20)
            time.sleep(0.2)
            
            if self.cancelled:
                return
            
            # UI 스타일 로드
            from ui.styles import get_light_palette
            
            # 3단계: AI 엔진 확인
            self.step_changed.emit("AI 엔진 확인", "AI 모델 상태를 확인하고 있습니다...")
            self.progress_percent.emit(30)
            time.sleep(0.3)
            
            if self.cancelled:
                return
            
            # 모델 정보 가져오기
            model_info = self._get_model_info()
            cache_exists = self._check_cache_exists()
            
            # 4단계: AI 모델 로딩/다운로드
            if cache_exists:
                self.step_changed.emit("AI 엔진 로딩", "기존 AI 엔진을 메모리에 로딩하고 있습니다...")
                self.progress_percent.emit(60)
            else:
                self.step_changed.emit(
                    "AI 엔진 다운로드", 
                    f"고품질 AI 엔진을 다운로드하고 있습니다...\n파일 크기: 약 {model_info.get('size', '176MB')}"
                )
                self.progress_percent.emit(40)
            
            time.sleep(0.5)
            
            if self.cancelled:
                return
            
            # 실제 AI 모델 로딩
            self.step_changed.emit("AI 엔진 초기화", "AI 모델을 메모리에 로딩하고 있습니다...")
            self.progress_percent.emit(70)
            
            # rembg 모듈 로드
            from rembg import new_session
            session = new_session(model_name=self.model_name)
            
            if self.cancelled:
                return
            
            self.progress_percent.emit(85)
            
            # 5단계: 완료
            self.step_changed.emit("초기화 완료", "Hana Studio 준비가 완료되었습니다!")
            self.progress_percent.emit(100)
            time.sleep(0.5)
            
            self.finished.emit()
            
        except Exception as e:
            if not self.cancelled:
                self.error.emit(f"초기화 중 오류가 발생했습니다: {str(e)}")
    
    def _get_model_info(self):
        """AI 모델 정보 반환"""
        model_info = {
            'isnet-general-use': {'name': '고품질 AI 엔진', 'size': '176MB'},
            'u2net': {'name': '표준 AI 엔진', 'size': '176MB'},
            'u2netp': {'name': '경량 AI 엔진', 'size': '4.7MB'},
            'silueta': {'name': '정밀 AI 엔진', 'size': '43MB'}
        }
        return model_info.get(self.model_name, {'name': 'AI 엔진', 'size': '176MB'})
    
    def _check_cache_exists(self):
        """AI 모델 캐시 존재 여부 확인"""
        try:
            cache_base = Path.home() / ".cache" / "huggingface" / "hub"
            
            if not cache_base.exists():
                return False
            
            # 간단한 캐시 확인
            for pattern in ["*model*", "*.onnx", "*.pth"]:
                if list(cache_base.glob(f"**/{pattern}")):
                    return True
            
            return False
        except Exception:
            return False


class SimpleLoadingWindow(QWidget):
    """통합 로딩 윈도우 - 중복 실행 방지"""
    
    def __init__(self):
        super().__init__()
        self.init_thread = None
        self.current_phase = "initialization"
        self.main_window_created = False
        self.main_window = None
        
        # 중복 실행 방지
        if self._check_already_running():
            print("⚠️ Hana Studio가 이미 실행 중입니다.")
            sys.exit(0)
        
        # UI 설정
        self._setup_window()
        self._setup_ui()
        
        # 윈도우 표시
        self.show()
        self.raise_()
        self.activateWindow()
        
        # 초기화 시작
        QTimer.singleShot(200, self._start_initialization)
    
    def _check_already_running(self):
        """이미 실행 중인 인스턴스 확인"""
        try:
            import psutil
            current_pid = os.getpid()
            current_name = "HanaStudio.exe" if getattr(sys, 'frozen', False) else "python.exe"
            
            # 같은 이름의 프로세스 찾기
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if proc.info['name'] and proc.info['name'].lower() == current_name.lower():
                        if proc.info['pid'] != current_pid:
                            # 명령줄 인수 확인 (개발 환경)
                            if not getattr(sys, 'frozen', False):
                                cmdline = proc.info.get('cmdline', [])
                                if any('main.py' in arg or 'hana_studio' in arg.lower() for arg in cmdline):
                                    return True
                            else:
                                return True
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            return False
        except ImportError:
            # psutil이 없는 경우 락 파일 방식
            return self._simple_running_check()
        except Exception:
            return False
    
    def _simple_running_check(self):
        """간단한 실행 중 확인"""
        try:
            import tempfile
            
            lock_file_path = os.path.join(tempfile.gettempdir(), "hana_studio.lock")
            
            try:
                self.lock_file = open(lock_file_path, 'w')
                self.lock_file.write(str(os.getpid()))
                self.lock_file.flush()
                return False  # 락 파일 생성 성공
            except IOError:
                return True  # 이미 실행 중
        except Exception:
            return False
    
    def _setup_window(self):
        """윈도우 기본 설정"""
        self.setWindowTitle("Hana Studio")
        self.setFixedSize(600, 280)
        
        # 화면 중앙에 배치
        screen = QApplication.primaryScreen().geometry()
        x = (screen.width() - self.width()) // 2
        y = (screen.height() - self.height()) // 2
        self.move(x, y)
        
        # 윈도우 플래그 설정
        self.setWindowFlags(
            Qt.WindowType.WindowStaysOnTopHint | 
            Qt.WindowType.FramelessWindowHint
        )
        
        # 스타일 설정
        self.setStyleSheet("""
            QWidget {
                background-color: white;
                border: 2px solid #4A90E2;
                border-radius: 15px;
            }
            QLabel {
                color: #2C3E50;
                background: transparent;
                border: none;
            }
            QProgressBar {
                border: none;
                background-color: #E9ECEF;
                border-radius: 9px;
                height: 18px;
                text-align: center;
                font-size: 12px;
                font-weight: 600;
                color: #495057;
            }
            QProgressBar::chunk {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #4A90E2, stop: 0.5 #357ABD, stop: 1 #4A90E2
                );
                border-radius: 9px;
                margin: 1px;
            }
            QPushButton {
                background-color: #6c757d;
                color: white;
                border: none;
                border-radius: 6px;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 11px;
                font-weight: 600;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #5a6268;
            }
            QPushButton:pressed {
                background-color: #545b62;
            }
        """)
    
    def _setup_ui(self):
        """UI 구성"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 30, 40, 30)
        layout.setSpacing(25)
        
        # 헤더 영역
        header_layout = QHBoxLayout()
        header_layout.setSpacing(25)
        
        # 로고/아이콘
        self.icon_label = QLabel()
        self._load_icon()
        header_layout.addWidget(self.icon_label)
        
        # 상태 정보 영역
        status_layout = QVBoxLayout()
        status_layout.setSpacing(8)
        
        # 단계 제목
        self.step_title = QLabel("시작 중...")
        self.step_title.setFont(QFont("Segoe UI", 16, QFont.Weight.DemiBold))
        self.step_title.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.step_title.setMinimumHeight(30)
        self.step_title.setStyleSheet("color: #2C3E50; background: transparent;")
        
        # 단계 설명
        self.step_description = QLabel("Hana Studio를 준비하고 있습니다.")
        self.step_description.setFont(QFont("Segoe UI", 12))
        self.step_description.setStyleSheet("color: #6C757D; line-height: 1.5; background: transparent;")
        self.step_description.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.step_description.setWordWrap(True)
        self.step_description.setMinimumHeight(60)
        
        status_layout.addWidget(self.step_title)
        status_layout.addWidget(self.step_description)
        status_layout.addStretch()
        
        header_layout.addLayout(status_layout)
        header_layout.addStretch()
        
        # 진행바 영역
        progress_layout = QVBoxLayout()
        progress_layout.setSpacing(15)
        
        # 진행바
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setFixedHeight(20)
        
        progress_layout.addWidget(self.progress_bar)
        
        # 버튼 영역
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 20, 0, 0)
        
        self.cancel_button = QPushButton("취소")
        self.cancel_button.setFixedSize(85, 38)
        self.cancel_button.clicked.connect(self._cancel_operation)
        
        button_layout.addStretch()
        button_layout.addWidget(self.cancel_button)
        
        # 레이아웃 조립
        layout.addLayout(header_layout)
        layout.addLayout(progress_layout)
        layout.addLayout(button_layout)
    
    def _load_icon(self):
        """아이콘 로딩"""
        try:
            from config import get_resource_path
            icon_path = get_resource_path("hana.ico")
            
            if os.path.exists(icon_path):
                pixmap = QPixmap(icon_path)
                scaled_pixmap = pixmap.scaled(
                    64, 64,
                    Qt.AspectRatioMode.KeepAspectRatio, 
                    Qt.TransformationMode.SmoothTransformation
                )
                self.icon_label.setPixmap(scaled_pixmap)
                print(f"✅ 아이콘 로드 성공: {icon_path}")
            else:
                # 아이콘이 없으면 이모지 사용
                self.icon_label.setText("🎨")
                self.icon_label.setFont(QFont("Segoe UI", 36))
                self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                print(f"⚠️ 아이콘 파일 없음, 이모지 사용")
                
        except Exception as e:
            # 오류 시 기본 이모지
            self.icon_label.setText("🎨")
            self.icon_label.setFont(QFont("Segoe UI", 36))
            self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            print(f"❌ 아이콘 로드 실패: {e}")
        
        # 아이콘 크기 설정
        self.icon_label.setFixedSize(64, 64)
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    
    def _start_initialization(self):
        """초기화 시작"""
        self.current_phase = "initialization"
        self.init_thread = InitializationThread()
        
        # 시그널 연결
        self.init_thread.progress_update.connect(self.update_status)
        self.init_thread.progress_percent.connect(self.update_progress)
        self.init_thread.step_changed.connect(self.update_step)
        self.init_thread.finished.connect(self._on_initialization_finished)
        self.init_thread.error.connect(self._on_initialization_error)
        
        # 스레드 시작
        self.init_thread.start()
    
    def update_status(self, message):
        """상태 메시지 업데이트"""
        self.step_description.setText(message)
    
    def update_progress(self, percent):
        """진행률 업데이트"""
        self.progress_bar.setValue(percent)
        
        # 완료 시 버튼 변경
        if percent >= 100:
            self.cancel_button.setText("완료")
            self.cancel_button.setStyleSheet("""
                QPushButton {
                    background-color: #28a745;
                    color: white;
                    border: none;
                    border-radius: 6px;
                    font-family: 'Segoe UI', Arial, sans-serif;
                    font-size: 11px;
                    font-weight: 600;
                    padding: 8px 16px;
                }
                QPushButton:hover {
                    background-color: #218838;
                }
            """)
    
    def update_step(self, title, description):
        """단계 정보 업데이트"""
        self.step_title.setText(title)
        self.step_description.setText(description)
    
    def _on_initialization_finished(self):
        """초기화 완료 처리"""
        if self.main_window_created:
            print("⚠️ 메인 윈도우가 이미 생성됨")
            return
            
        self.current_phase = "complete"
        self.update_step("초기화 완료", "Hana Studio 준비가 완료되었습니다!")
        
        # 메인 윈도우로 전환
        QTimer.singleShot(1000, self._show_main_window)
    
    def _on_initialization_error(self, error_msg):
        """초기화 오류 처리"""
        self.current_phase = "error"
        self.update_step("오류 발생", f"초기화 중 오류가 발생했습니다:\n{error_msg}")
        
        # 진행바 빨간색으로 변경
        self.progress_bar.setStyleSheet("""
            QProgressBar::chunk {
                background-color: #dc3545;
                border-radius: 8px;
            }
        """)
        
        # 버튼을 닫기로 변경
        self.cancel_button.setText("닫기")
        self.cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                color: white;
                border: none;
                border-radius: 6px;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 11px;
                font-weight: 600;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
        """)
        
        # 5초 후 자동 종료
        QTimer.singleShot(5000, self.close)
    
    def _cancel_operation(self):
        """작업 취소/완료"""
        if self.current_phase == "complete":
            # 완료 버튼 클릭 시 메인 윈도우로
            self._show_main_window()
            return
        elif self.current_phase == "error":
            # 오류 시 프로그램 종료
            sys.exit(1)
        else:
            # 초기화 중 취소
            from PySide6.QtWidgets import QMessageBox
            reply = QMessageBox.question(
                self,
                "초기화 취소",
                "Hana Studio 초기화를 취소하시겠습니까?\n\n"
                "취소하면 프로그램을 사용할 수 없습니다.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                if self.init_thread and self.init_thread.isRunning():
                    self.init_thread.cancel()
                    self.init_thread.quit()
                    self.init_thread.wait(3000)
                sys.exit(1)
    
    def _show_main_window(self):
        """메인 윈도우 표시"""
        if self.main_window_created:
            print("⚠️ 메인 윈도우가 이미 생성됨")
            return
            
        try:
            print("메인 윈도우 생성 중...")
            self.main_window_created = True
            
            # 메인 윈도우 생성
            from hana_studio import HanaStudio
            self.main_window = HanaStudio()
            self.main_window.show()
            
            # 로딩 윈도우 닫기
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
            sys.exit(1)
    
    def closeEvent(self, event):
        """윈도우 닫기 시 정리"""
        # 락 파일 정리
        try:
            if hasattr(self, 'lock_file') and self.lock_file:
                self.lock_file.close()
        except Exception:
            pass
        
        # 스레드 정리
        if self.init_thread and self.init_thread.isRunning():
            self.init_thread.cancel()
            self.init_thread.quit()
            self.init_thread.wait(1000)
        
        event.accept()


# 하위 호환성을 위한 함수들
def show_installation_dialog(parent=None):
    """하위 호환성을 위한 별칭"""
    window = SimpleLoadingWindow()
    return True


def show_unified_loading_dialog(parent=None):
    """하위 호환성을 위한 별칭"""
    return show_installation_dialog(parent)


# 하위 호환성을 위한 클래스 별칭들
class InstallationDialog(SimpleLoadingWindow):
    """하위 호환성을 위한 별칭"""
    pass


class LoadingDialog(SimpleLoadingWindow):
    """하위 호환성을 위한 별칭"""
    pass


class UnifiedLoadingDialog(SimpleLoadingWindow):
    """하위 호환성을 위한 별칭"""
    pass