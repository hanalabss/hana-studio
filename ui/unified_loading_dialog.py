"""
ui/unified_loading_dialog.py
모든 로딩 과정을 처리하는 통합 다이얼로그
- 초기 로딩 (simple_loading)
- AI 모델 설치 (installation_dialog)  
- 세부 진행 상황 (loading_dialog)
"""

import os
import sys
import time
from pathlib import Path
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar, 
    QPushButton, QFrame, QGraphicsDropShadowEffect
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QPropertyAnimation, QRect
from PySide6.QtGui import QFont, QPixmap, QIcon, QPainter, QPen, QBrush, QColor
from rembg import new_session
from config import config, get_resource_path


class InitializationThread(QThread):
    """초기화 작업 스레드"""
    progress_update = Signal(str)
    finished = Signal()
    error = Signal(str)
    
    def run(self):
        try:
            # 1단계: 기본 설정
            self.progress_update.emit("기본 설정 로딩 중...")
            time.sleep(0.3)
            
            from config import config, AppConstants, get_resource_path
            
            # 2단계: UI 스타일
            self.progress_update.emit("UI 테마 로딩 중...")
            time.sleep(0.2)
            
            from ui.styles import get_light_palette
            
            # 3단계: 완료
            self.progress_update.emit("초기화 완료!")
            time.sleep(0.2)
            
            self.finished.emit()
            
        except Exception as e:
            self.error.emit(str(e))


class ModelInstallationThread(QThread):
    """AI 모델 설치 스레드"""
    
    step_changed = Signal(str, str)  # (step_title, step_description)
    progress_changed = Signal(int)   # 0-100
    installation_finished = Signal()
    installation_error = Signal(str)
    
    def __init__(self):
        super().__init__()
        self.model_name = config.get('ai_model', 'isnet-general-use')
        self.cancelled = False
    
    def cancel(self):
        self.cancelled = True
    
    def run(self):
        try:
            # 1단계: 시스템 확인
            self.step_changed.emit("시스템 확인", "Hana Studio 실행 환경을 확인하고 있습니다...")
            self.progress_changed.emit(10)
            time.sleep(0.8)
            
            if self.cancelled:
                return
            
            # 2단계: 구성 요소 준비
            self.step_changed.emit("구성 요소 준비", "필요한 구성 요소를 준비하고 있습니다...")
            self.progress_changed.emit(25)
            time.sleep(1.0)
            
            if self.cancelled:
                return
            
            # 3단계: AI 엔진 확인
            model_info = self._get_model_info()
            cache_exists = self._check_cache_exists()
            
            if cache_exists:
                self.step_changed.emit("AI 엔진 로딩", "기존 AI 엔진을 로딩하고 있습니다...")
                self.progress_changed.emit(70)
            else:
                self.step_changed.emit("AI 엔진 다운로드", f"고품질 AI 엔진을 다운로드하고 있습니다... ({model_info.get('size', '176MB')})")
                self.progress_changed.emit(50)
            
            if self.cancelled:
                return
            
            # 4단계: 실제 모델 로딩
            session = new_session(model_name=self.model_name)
            self.progress_changed.emit(85)
            
            if self.cancelled:
                return
            
            # 5단계: 완료
            self.step_changed.emit("설치 완료", "Hana Studio 설치가 완료되었습니다!")
            self.progress_changed.emit(100)
            time.sleep(0.5)
            
            self.installation_finished.emit()
            
        except Exception as e:
            if not self.cancelled:
                self.installation_error.emit(f"설치 중 오류가 발생했습니다: {str(e)}")
    
    def _get_model_info(self):
        """모델 정보 반환"""
        model_info = {
            'isnet-general-use': {'name': '고품질 AI 엔진', 'size': '176MB'},
            'u2net': {'name': '표준 AI 엔진', 'size': '176MB'},
            'u2netp': {'name': '경량 AI 엔진', 'size': '4.7MB'},
            'silueta': {'name': '정밀 AI 엔진', 'size': '43MB'}
        }
        return model_info.get(self.model_name, {'name': 'AI 엔진', 'size': '176MB'})
    
    def _check_cache_exists(self):
        """캐시 존재 여부 확인"""
        try:
            cache_base = Path.home() / ".cache" / "huggingface" / "hub"
            if not cache_base.exists():
                return False
            
            for pattern in ["*model*", "*.onnx", "*.pth"]:
                if list(cache_base.glob(f"**/{pattern}")):
                    return True
            return False
        except Exception:
            return False


class UnifiedLoadingDialog(QDialog):
    """통합 로딩 다이얼로그 - 모든 로딩 과정 처리"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_dialog()
        self._setup_ui()
        self._apply_modern_style()
        
        # 스레드 관리
        self.init_thread = None
        self.install_thread = None
        self.current_phase = "initialization"
        
        # 🚀 즉시 표시 (simple_loading처럼)
        self.show()
        self.raise_()
        self.activateWindow()
        
        # 🎯 UI가 완전히 표시될 때까지 대기
        from PySide6.QtWidgets import QApplication
        QApplication.processEvents()
        QApplication.processEvents()  # 한번 더 확실히
        
        print("✅ 통합 다이얼로그 즉시 표시 완료")
        
        # 🎯 매우 짧은 대기 후 초기화 시작 (UI 표시 우선)
        QTimer.singleShot(100, self._start_initialization)
    
    def _setup_dialog(self):
        """다이얼로그 기본 설정"""
        self.setWindowTitle("Hana Studio")
        self.setFixedSize(550, 320)
        
        self.setWindowFlags(
            Qt.WindowType.Dialog | 
            Qt.WindowType.WindowTitleHint | 
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.WindowSystemMenuHint
        )
        self.setModal(True)
    
    def _setup_ui(self):
        """UI 구성"""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # 메인 컨테이너
        container = QFrame()
        container.setObjectName("mainContainer")
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(40, 30, 40, 30)
        container_layout.setSpacing(25)
        
        # 헤더 영역 (아이콘 + 제목)
        header_layout = QHBoxLayout()
        header_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        
        # 앱 아이콘
        self.icon_label = QLabel()
        self._load_app_icon()
        header_layout.addWidget(self.icon_label)
        
        # 제목 및 설명
        title_layout = QVBoxLayout()
        title_layout.setSpacing(5)
        
        self.app_title = QLabel("Hana Studio")
        self.app_title.setObjectName("appTitle")
        
        self.app_subtitle = QLabel("AI 기반 이미지 배경 제거 도구")
        self.app_subtitle.setObjectName("appSubtitle")
        
        title_layout.addWidget(self.app_title)
        title_layout.addWidget(self.app_subtitle)
        title_layout.addStretch()
        
        header_layout.addLayout(title_layout)
        header_layout.addStretch()
        
        # 진행 영역
        progress_container = QFrame()
        progress_container.setObjectName("progressContainer")
        progress_layout = QVBoxLayout(progress_container)
        progress_layout.setContentsMargins(25, 20, 25, 20)
        progress_layout.setSpacing(15)
        
        # 현재 단계 제목
        self.step_title = QLabel("시작 중...")
        self.step_title.setObjectName("stepTitle")
        
        # 현재 단계 설명
        self.step_description = QLabel("Hana Studio를 준비하고 있습니다.")
        self.step_description.setObjectName("stepDescription")
        self.step_description.setWordWrap(True)
        
        # 진행률 바
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setObjectName("modernProgressBar")
        
        progress_layout.addWidget(self.step_title)
        progress_layout.addWidget(self.step_description)
        progress_layout.addWidget(self.progress_bar)
        
        # 버튼 영역
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 20, 0, 0)
        
        self.cancel_button = QPushButton("취소")
        self.cancel_button.setObjectName("cancelButton")
        self.cancel_button.setFixedSize(80, 35)
        self.cancel_button.clicked.connect(self._cancel_operation)
        
        button_layout.addStretch()
        button_layout.addWidget(self.cancel_button)
        
        # 레이아웃 조립
        container_layout.addLayout(header_layout)
        container_layout.addWidget(progress_container)
        container_layout.addLayout(button_layout)
        
        main_layout.addWidget(container)
        
        # 그림자 효과
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(30)
        shadow.setXOffset(0)
        shadow.setYOffset(10)
        shadow.setColor(QColor(0, 0, 0, 50))
        container.setGraphicsEffect(shadow)
    
    def _load_app_icon(self):
        """앱 아이콘 로드"""
        try:
            icon_path = get_resource_path("hana.ico")
            if os.path.exists(icon_path):
                pixmap = QPixmap(icon_path).scaled(
                    64, 64, 
                    Qt.AspectRatioMode.KeepAspectRatio, 
                    Qt.TransformationMode.SmoothTransformation
                )
                self.icon_label.setPixmap(pixmap)
            else:
                self.icon_label.setText("🎨")
                self.icon_label.setFont(QFont("Segoe UI", 32))
                self.icon_label.setFixedSize(64, 64)
                self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        except Exception:
            self.icon_label.setText("🎨")
            self.icon_label.setFont(QFont("Segoe UI", 32))
            self.icon_label.setFixedSize(64, 64)
            self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    
    def _apply_modern_style(self):
        """현대적인 스타일 적용"""
        self.setStyleSheet("""
            QDialog {
                background-color: #f0f2f5;
            }
            
            #mainContainer {
                background-color: white;
                border-radius: 15px;
                border: 1px solid #e0e6ed;
            }
            
            #appTitle {
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 24px;
                font-weight: bold;
                color: #2c3e50;
                margin: 0;
            }
            
            #appSubtitle {
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 13px;
                color: #7f8c8d;
                margin: 0;
            }
            
            #progressContainer {
                background-color: #f8f9fa;
                border: 1px solid #e9ecef;
                border-radius: 10px;
            }
            
            #stepTitle {
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 16px;
                font-weight: 600;
                color: #2c3e50;
                margin: 0;
            }
            
            #stepDescription {
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 12px;
                color: #6c757d;
                line-height: 1.4;
                margin: 0;
            }
            
            #modernProgressBar {
                border: none;
                background-color: #e9ecef;
                border-radius: 8px;
                height: 16px;
                text-align: center;
                font-size: 11px;
                font-weight: 600;
                color: #495057;
            }
            
            #modernProgressBar::chunk {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #4CAF50, stop: 0.5 #45a049, stop: 1 #4CAF50
                );
                border-radius: 8px;
                margin: 1px;
            }
            
            #cancelButton {
                background-color: #6c757d;
                color: white;
                border: none;
                border-radius: 6px;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 11px;
                font-weight: 600;
                padding: 8px 16px;
            }
            
            #cancelButton:hover {
                background-color: #5a6268;
            }
            
            #cancelButton:pressed {
                background-color: #545b62;
            }
            
            #completeButton {
                background-color: #28a745;
                color: white;
                border: none;
                border-radius: 6px;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 11px;
                font-weight: 600;
                padding: 8px 16px;
            }
            
            #completeButton:hover {
                background-color: #218838;
            }
            
            #errorButton {
                background-color: #dc3545;
                color: white;
                border: none;
                border-radius: 6px;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 11px;
                font-weight: 600;
                padding: 8px 16px;
            }
            
            #errorButton:hover {
                background-color: #c82333;
            }
        """)
    
    def _start_initialization(self):
        """초기화 시작"""
        self.current_phase = "initialization"
        self.step_title.setText("초기화 중...")
        self.step_description.setText("프로그램 구성요소를 준비하고 있습니다...")
        
        self.init_thread = InitializationThread()
        self.init_thread.progress_update.connect(self._on_init_progress)
        self.init_thread.finished.connect(self._on_init_finished)
        self.init_thread.error.connect(self._on_error)
        
        self.init_thread.start()
    
    def _on_init_progress(self, message):
        """초기화 진행 상황"""
        self.step_description.setText(message)
        # 초기화는 빠르게 진행
        current_value = self.progress_bar.value()
        self.progress_bar.setValue(min(current_value + 10, 30))
    
    def _on_init_finished(self):
        """초기화 완료 - AI 모델 설치 시작"""
        self.progress_bar.setValue(30)
        QTimer.singleShot(300, self._start_installation)
    
    def _start_installation(self):
        """AI 모델 설치 시작"""
        self.current_phase = "installation"
        
        self.install_thread = ModelInstallationThread()
        self.install_thread.step_changed.connect(self._on_install_step)
        self.install_thread.progress_changed.connect(self._on_install_progress)
        self.install_thread.installation_finished.connect(self._on_install_finished)
        self.install_thread.installation_error.connect(self._on_error)
        
        self.install_thread.start()
    
    def _on_install_step(self, title, description):
        """설치 단계 업데이트"""
        self.step_title.setText(title)
        self.step_description.setText(description)
    
    def _on_install_progress(self, value):
        """설치 진행률 업데이트"""
        self.progress_bar.setValue(value)
    
    def _on_install_finished(self):
        """설치 완료"""
        self.current_phase = "complete"
        self.cancel_button.setText("완료")
        self.cancel_button.setObjectName("completeButton")
        self.cancel_button.setStyleSheet("""
            #completeButton {
                background-color: #28a745;
                color: white;
                border: none;
                border-radius: 6px;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 11px;
                font-weight: 600;
                padding: 8px 16px;
            }
            #completeButton:hover {
                background-color: #218838;
            }
        """)
        
        # 잠시 대기 후 자동 완료
        QTimer.singleShot(1500, self.accept)
    
    def _on_error(self, error_msg):
        """오류 처리"""
        self.step_title.setText("오류 발생")
        self.step_description.setText(f"오류: {error_msg}")
        
        # 진행바 빨간색으로 변경
        self.progress_bar.setStyleSheet("""
            QProgressBar::chunk {
                background-color: #dc3545;
            }
        """)
        
        self.cancel_button.setText("닫기")
        self.cancel_button.setObjectName("errorButton")
        self.cancel_button.setStyleSheet("""
            #errorButton {
                background-color: #dc3545;
                color: white;
                border: none;
                border-radius: 6px;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 11px;
                font-weight: 600;
                padding: 8px 16px;
            }
            #errorButton:hover {
                background-color: #c82333;
            }
        """)
        
        # 5초 후 자동 닫기
        QTimer.singleShot(5000, self.reject)
    
    def _cancel_operation(self):
        """작업 취소"""
        if self.current_phase == "complete":
            self.accept()
            return
        
        if self.current_phase == "installation":
            from PySide6.QtWidgets import QMessageBox
            reply = QMessageBox.question(
                self,
                "설치 취소",
                "Hana Studio 설치를 취소하시겠습니까?\n\n"
                "취소하면 프로그램을 사용할 수 없습니다.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                if self.install_thread and self.install_thread.isRunning():
                    self.install_thread.cancel()
                    self.install_thread.quit()
                    self.install_thread.wait(3000)
                self.reject()
        else:
            self.reject()
    
    def closeEvent(self, event):
        """다이얼로그 닫기 시 스레드 정리"""
        if self.init_thread and self.init_thread.isRunning():
            self.init_thread.quit()
            self.init_thread.wait(1000)
        
        if self.install_thread and self.install_thread.isRunning():
            self.install_thread.cancel()
            self.install_thread.quit()
            self.install_thread.wait(1000)
        
        event.accept()


# 편의 함수
def show_unified_loading_dialog(parent=None):
    """통합 로딩 다이얼로그 표시 - 즉시 UI 표시"""
    # 🚀 다이얼로그 생성 시 자동으로 즉시 표시됨
    dialog = UnifiedLoadingDialog(parent)
    
    # 🎯 UI가 완전히 표시될 때까지 대기
    from PySide6.QtWidgets import QApplication
    QApplication.processEvents()
    
    # 다이얼로그 실행
    return dialog.exec() == QDialog.DialogCode.Accepted


# 하위 호환성을 위한 별칭들
def show_installation_dialog(parent=None):
    """하위 호환성을 위한 별칭"""
    return show_unified_loading_dialog(parent)


class SimpleLoadingWindow(UnifiedLoadingDialog):
    """하위 호환성을 위한 별칭"""
    pass


class InstallationDialog(UnifiedLoadingDialog):
    """하위 호환성을 위한 별칭"""
    pass


class LoadingDialog(UnifiedLoadingDialog):
    """하위 호환성을 위한 별칭"""
    pass