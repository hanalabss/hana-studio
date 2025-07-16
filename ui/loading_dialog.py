# ui/loading_dialog.py
from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QProgressBar
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QFont
from rembg import new_session
from config import config


class ModelLoadingThread(QThread):
    progress = Signal(str)
    finished = Signal()
    error = Signal(str)
    
    def run(self):
        try:
            self.progress.emit("AI 모델 확인 중...")
            model_name = config.get('ai_model', 'isnet-general-use')
            
            self.progress.emit(f"AI 모델 다운로드 중: {model_name}")
            self.progress.emit("최초 실행 시 시간이 소요됩니다...")
            
            session = new_session(model_name=model_name)
            
            self.progress.emit("AI 모델 로드 완료!")
            self.finished.emit()
            
        except Exception as e:
            self.error.emit(f"AI 모델 로드 실패: {e}")


class LoadingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Hana Studio - 초기화")
        self.setFixedSize(400, 150)
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.WindowTitleHint)
        self.setModal(True)
        
        self._setup_ui()
        self._start_loading()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(30, 20, 30, 20)
        
        # 제목
        title = QLabel("🎨 Hana Studio 초기화 중...")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: #2C3E50;")
        
        # 상태 텍스트
        self.status_label = QLabel("시작 중...")
        self.status_label.setFont(QFont("Segoe UI", 10))
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color: #7F8C8D;")
        
        # 진행바
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # 무한 진행바
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #BDC3C7;
                border-radius: 5px;
                background-color: #ECF0F1;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #3498DB;
                border-radius: 3px;
            }
        """)
        
        layout.addWidget(title)
        layout.addWidget(self.status_label)
        layout.addWidget(self.progress_bar)
    
    def _start_loading(self):
        self.loading_thread = ModelLoadingThread()
        self.loading_thread.progress.connect(self.status_label.setText)
        self.loading_thread.finished.connect(self.accept)
        self.loading_thread.error.connect(self._on_error)
        self.loading_thread.start()
    
    def _on_error(self, error_msg):
        self.status_label.setText(f"오류: {error_msg}")
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        QTimer.singleShot(3000, self.reject)