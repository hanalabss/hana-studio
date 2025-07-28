# ui/loading_dialog.py
import os
import sys
import time
import requests
from pathlib import Path
from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QProgressBar, QPushButton, QHBoxLayout
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QFont, QPixmap, QIcon
from rembg import new_session
from config import config, get_resource_path


class ModelDownloadThread(QThread):
    """AI 모델 다운로드 및 로딩 스레드 - 진행률 추적"""
    
    progress_text = Signal(str)
    progress_percent = Signal(int)
    download_progress = Signal(int, int)  # (downloaded_bytes, total_bytes)
    finished = Signal()
    error = Signal(str)
    
    def __init__(self):
        super().__init__()
        self.model_name = config.get('ai_model', 'isnet-general-use')
        self.cancelled = False
    
    def cancel(self):
        """다운로드 취소"""
        self.cancelled = True
    
    def run(self):
        try:
            if self.cancelled:
                return
                
            self.progress_text.emit("🔍 AI 모델 확인 중...")
            self.progress_percent.emit(5)
            
            # 모델 정보 가져오기
            model_info = self._get_model_info()
            if not model_info:
                self.error.emit("지원하지 않는 AI 모델입니다.")
                return
            
            self.progress_text.emit(f"📦 {model_info['name']} 준비 중...")
            self.progress_percent.emit(10)
            
            if self.cancelled:
                return
            
            # 캐시 디렉토리 확인
            cache_dir = self._get_cache_directory()
            model_exists = self._check_model_exists(cache_dir)
            
            if model_exists:
                self.progress_text.emit("✅ 기존 모델 발견, 로딩 중...")
                self.progress_percent.emit(50)
            else:
                self.progress_text.emit(f"⬇️ {model_info['name']} 다운로드 중...")
                self.progress_text.emit(f"파일 크기: 약 {model_info.get('size', '176MB')}")
                self.progress_percent.emit(20)
            
            if self.cancelled:
                return
            
            # 실제 모델 로딩 (rembg가 자동으로 다운로드 처리)
            self.progress_text.emit("🧠 AI 모델 메모리 로딩 중...")
            self.progress_percent.emit(70)
            
            # 모델 세션 생성
            session = new_session(model_name=self.model_name)
            
            if self.cancelled:
                return
            
            self.progress_text.emit("✅ AI 모델 로딩 완료!")
            self.progress_percent.emit(100)
            
            # 짧은 대기 후 완료
            time.sleep(0.5)
            self.finished.emit()
            
        except Exception as e:
            if not self.cancelled:
                self.error.emit(f"AI 모델 로드 실패: {str(e)}")
    
    def _get_model_info(self):
        """모델 정보 반환"""
        model_info = {
            'isnet-general-use': {
                'name': '범용 고품질 모델 (권장)',
                'size': '176MB',
                'description': '대부분의 이미지에 적합'
            },
            'u2net': {
                'name': 'U²-Net 기본 모델',
                'size': '176MB', 
                'description': '빠른 처리 속도'
            },
            'u2netp': {
                'name': 'U²-Net 경량 모델',
                'size': '4.7MB',
                'description': '초고속 처리'
            },
            'silueta': {
                'name': 'Silueta 정밀 모델',
                'size': '43MB',
                'description': '정밀한 실루엣 처리'
            }
        }
        return model_info.get(self.model_name)
    
    def _get_cache_directory(self):
        """모델 캐시 디렉토리 경로"""
        if sys.platform == "win32":
            cache_base = Path.home() / ".cache" / "huggingface" / "hub"
        else:
            cache_base = Path.home() / ".cache" / "huggingface" / "hub"
        return cache_base
    
    def _check_model_exists(self, cache_dir):
        """모델이 이미 다운로드되어 있는지 확인"""
        try:
            if not cache_dir.exists():
                return False
            
            # 모델 파일이 있는지 대략적으로 확인
            model_patterns = [
                f"*{self.model_name}*",
                "*model*", 
                "*.onnx",
                "*.pth"
            ]
            
            for pattern in model_patterns:
                if list(cache_dir.glob(f"**/{pattern}")):
                    return True
            
            return False
        except Exception:
            return False


class LoadingDialog(QDialog):
    """개선된 AI 모델 로딩 다이얼로그 - 진행률 표시"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Hana Studio - AI 모델 준비")
        self.setFixedSize(480, 220)
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.WindowTitleHint)
        self.setModal(True)
        
        # 스레드 참조
        self.loading_thread = None
        
        self._setup_ui()
        self._start_loading()
    
    def _setup_ui(self):
        """UI 구성"""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(30, 25, 30, 25)
        
        # 헤더 영역
        header_layout = QHBoxLayout()
        
        # 아이콘 (있으면 표시)
        try:
            icon_path = get_resource_path("hana.ico")
            if os.path.exists(icon_path):
                icon_label = QLabel()
                pixmap = QPixmap(icon_path).scaled(
                    48, 48, 
                    Qt.AspectRatioMode.KeepAspectRatio, 
                    Qt.TransformationMode.SmoothTransformation
                )
                icon_label.setPixmap(pixmap)
                header_layout.addWidget(icon_label)
        except Exception:
            pass
        
        # 제목
        title = QLabel("🎨 Hana Studio 초기화")
        title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        title.setStyleSheet("color: #2C3E50; margin-left: 10px;")
        header_layout.addWidget(title)
        header_layout.addStretch()
        
        # 상태 텍스트
        self.status_label = QLabel("시작 중...")
        self.status_label.setFont(QFont("Segoe UI", 11))
        self.status_label.setStyleSheet("color: #34495E; margin: 5px 0;")
        self.status_label.setWordWrap(True)
        
        # 퍼센트 진행률
        self.percent_progress = QProgressBar()
        self.percent_progress.setRange(0, 100)
        self.percent_progress.setValue(0)
        self.percent_progress.setTextVisible(True)
        self.percent_progress.setStyleSheet("""
            QProgressBar {
                border: 2px solid #BDC3C7;
                border-radius: 8px;
                background-color: #ECF0F1;
                height: 25px;
                text-align: center;
                font-weight: bold;
                color: #2C3E50;
            }
            QProgressBar::chunk {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #3498DB, stop: 1 #2980B9
                );
                border-radius: 6px;
                margin: 1px;
            }
        """)
        
        # 다운로드 진행률 (처음엔 숨김)
        self.download_label = QLabel("다운로드 진행률:")
        self.download_label.setFont(QFont("Segoe UI", 9))
        self.download_label.setStyleSheet("color: #7F8C8D; margin-top: 10px;")
        self.download_label.hide()
        
        self.download_progress = QProgressBar()
        self.download_progress.setRange(0, 100)
        self.download_progress.setValue(0)
        self.download_progress.setTextVisible(True)
        self.download_progress.setFormat("%p% (%v MB / %m MB)")
        self.download_progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #BDC3C7;
                border-radius: 6px;
                background-color: #F8F9FA;
                height: 20px;
                text-align: center;
                font-size: 9px;
                color: #2C3E50;
            }
            QProgressBar::chunk {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #27AE60, stop: 1 #229954
                );
                border-radius: 4px;
            }
        """)
        self.download_progress.hide()
        
        # 취소 버튼
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.cancel_button = QPushButton("취소")
        self.cancel_button.setFixedSize(80, 35)
        self.cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #E74C3C;
                color: white;
                border: none;
                border-radius: 6px;
                font-weight: bold;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #C0392B;
            }
            QPushButton:pressed {
                background-color: #A93226;
            }
        """)
        self.cancel_button.clicked.connect(self._cancel_loading)
        button_layout.addWidget(self.cancel_button)
        
        # 레이아웃에 추가
        layout.addLayout(header_layout)
        layout.addWidget(self.status_label)
        layout.addWidget(self.percent_progress)
        layout.addWidget(self.download_label)
        layout.addWidget(self.download_progress)
        layout.addStretch()
        layout.addLayout(button_layout)
    
    def _start_loading(self):
        """모델 로딩 시작"""
        self.loading_thread = ModelDownloadThread()
        
        # 시그널 연결
        self.loading_thread.progress_text.connect(self._update_status)
        self.loading_thread.progress_percent.connect(self._update_percent)
        self.loading_thread.download_progress.connect(self._update_download)
        self.loading_thread.finished.connect(self._on_finished)
        self.loading_thread.error.connect(self._on_error)
        
        self.loading_thread.start()
    
    def _update_status(self, message):
        """상태 메시지 업데이트"""
        self.status_label.setText(message)
        
        # 다운로드 중이면 다운로드 진행률 표시
        if "다운로드 중" in message:
            self.download_label.show()
            self.download_progress.show()
    
    def _update_percent(self, percent):
        """전체 진행률 업데이트"""
        self.percent_progress.setValue(percent)
        
        if percent >= 100:
            self.cancel_button.setText("완료")
            self.cancel_button.setStyleSheet("""
                QPushButton {
                    background-color: #27AE60;
                    color: white;
                    border: none;
                    border-radius: 6px;
                    font-weight: bold;
                    font-size: 10px;
                }
            """)
    
    def _update_download(self, downloaded_mb, total_mb):
        """다운로드 진행률 업데이트"""
        if total_mb > 0:
            percent = int((downloaded_mb / total_mb) * 100)
            self.download_progress.setRange(0, total_mb)
            self.download_progress.setValue(downloaded_mb)
    
    def _cancel_loading(self):
        """로딩 취소"""
        if self.loading_thread and self.loading_thread.isRunning():
            self.loading_thread.cancel()
            self.loading_thread.quit()
            self.loading_thread.wait(3000)  # 3초 대기
        
        self.reject()
    
    def _on_finished(self):
        """로딩 완료"""
        self.accept()
    
    def _on_error(self, error_msg):
        """로딩 오류"""
        self.status_label.setText(f"❌ 오류: {error_msg}")
        self.status_label.setStyleSheet("color: #E74C3C; font-weight: bold;")
        self.percent_progress.setStyleSheet("""
            QProgressBar::chunk {
                background-color: #E74C3C;
            }
        """)
        
        self.cancel_button.setText("닫기")
        self.cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #95A5A6;
                color: white;
                border: none;
                border-radius: 6px;
                font-weight: bold;
            }
        """)
        
        # 3초 후 자동 닫기
        QTimer.singleShot(3000, self.reject)
    
    def closeEvent(self, event):
        """다이얼로그 닫기 시 스레드 정리"""
        if self.loading_thread and self.loading_thread.isRunning():
            self.loading_thread.cancel()
            self.loading_thread.quit()
            self.loading_thread.wait(1000)
        
        event.accept()