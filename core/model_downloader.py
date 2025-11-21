"""
AI 모델 다운로드 다이얼로그
프로그레스바와 함께 모델 다운로드 상태를 표시
"""

import os
import sys
import requests
from pathlib import Path
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QProgressBar,
    QPushButton, QApplication
)
from PySide6.QtCore import Qt, QThread, Signal
from utils.safe_temp_path import get_safe_temp_dir


# rembg 모델 정보 (isnet-general-use)
MODEL_INFO = {
    "isnet-general-use": {
        "url": "https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-general-use.onnx",
        "filename": "isnet-general-use.onnx",
        "size_mb": 175  # 약 175MB
    },
    "u2net": {
        "url": "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx",
        "filename": "u2net.onnx",
        "size_mb": 176
    },
    "u2net_human_seg": {
        "url": "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net_human_seg.onnx",
        "filename": "u2net_human_seg.onnx",
        "size_mb": 176
    },
    "silueta": {
        "url": "https://github.com/danielgatis/rembg/releases/download/v0.0.0/silueta.onnx",
        "filename": "silueta.onnx",
        "size_mb": 44
    }
}


def get_model_cache_path(model_name: str) -> Path:
    """모델 캐시 경로 반환 (한글 경로 안전)"""
    safe_temp = get_safe_temp_dir()
    u2net_home = os.path.join(safe_temp, "u2net_models")
    os.makedirs(u2net_home, exist_ok=True)

    # 환경변수도 설정 (rembg가 이 경로를 사용하도록)
    os.environ['U2NET_HOME'] = u2net_home

    model_info = MODEL_INFO.get(model_name, MODEL_INFO["isnet-general-use"])
    return Path(u2net_home) / model_info["filename"]


def validate_model_file(model_path: Path) -> bool:
    """
    ONNX 모델 파일이 실제 ONNX Runtime으로 로딩 가능한지 검증

    Args:
        model_path: 검증할 모델 파일 경로

    Returns:
        bool: 로딩 가능하면 True, 실패하면 False
    """
    try:
        # 지연 import (시작 속도 최적화)
        import onnxruntime as ort

        safe_print(f"[AI] 모델 파일 검증 중: {model_path.name}")

        # InferenceSession 생성 시도
        # 파일이 손상되었거나 형식이 잘못되면 예외 발생
        session = ort.InferenceSession(
            str(model_path),
            providers=['CPUExecutionProvider']  # CPU만 사용
        )

        # 세션 생성 성공 → 로딩 가능한 파일
        safe_print(f"[AI] 모델 검증 성공: {model_path.name}")
        return True

    except Exception as e:
        # 로딩 실패 → 손상되었거나 잘못된 파일
        safe_print(f"[AI] 모델 검증 실패: {model_path.name} - {type(e).__name__}: {str(e)}")

        # 손상된 파일 자동 삭제
        try:
            if model_path.exists():
                model_path.unlink()
                safe_print(f"[AI] 손상된 모델 파일 삭제됨: {model_path.name}")
        except Exception as del_err:
            safe_print(f"[AI] 파일 삭제 실패: {del_err}")

        return False


def is_model_downloaded(model_name: str) -> bool:
    """
    모델이 이미 다운로드되었고 사용 가능한지 확인

    검증 단계:
    1. 파일 존재 여부
    2. 파일 크기 검증 (불완전 다운로드 체크)
    3. ONNX Runtime 로딩 가능 여부 검증

    Returns:
        bool: 모델 사용 가능 여부
    """
    model_path = get_model_cache_path(model_name)

    # 1단계: 파일 존재 확인
    if not model_path.exists():
        safe_print(f"[AI] 모델 파일 없음: {model_name}")
        return False

    # 2단계: 파일 크기 확인 (불완전한 다운로드 체크)
    model_info = MODEL_INFO.get(model_name, MODEL_INFO["isnet-general-use"])
    expected_size = model_info["size_mb"] * 1024 * 1024 * 0.9  # 90% 이상
    actual_size = model_path.stat().st_size

    if actual_size < expected_size:
        safe_print(f"[AI] 모델 파일 크기 부족: {actual_size / (1024*1024):.1f} MB < {expected_size / (1024*1024):.1f} MB")
        return False

    # 3단계: ONNX Runtime 로딩 검증
    if not validate_model_file(model_path):
        safe_print(f"[AI] 모델 로딩 검증 실패 → 재다운로드 필요")
        return False

    safe_print(f"[AI] 모델 파일 검증 완료: {model_path.name}")
    return True


class DownloadThread(QThread):
    """다운로드를 수행하는 스레드"""
    progress = Signal(int, str)  # 퍼센트, 상태 메시지
    finished = Signal(bool, str)  # 성공 여부, 메시지

    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name
        self.cancelled = False

    def run(self):
        try:
            model_info = MODEL_INFO.get(self.model_name, MODEL_INFO["isnet-general-use"])
            url = model_info["url"]
            model_path = get_model_cache_path(self.model_name)

            self.progress.emit(0, "연결 중...")

            # 다운로드 시작
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0

            # 임시 파일로 먼저 다운로드
            temp_path = model_path.with_suffix('.tmp')

            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if self.cancelled:
                        self.finished.emit(False, "다운로드 취소됨")
                        return

                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

                        if total_size > 0:
                            percent = int(downloaded * 100 / total_size)
                            mb_downloaded = downloaded / (1024 * 1024)
                            mb_total = total_size / (1024 * 1024)
                            self.progress.emit(
                                percent,
                                f"{mb_downloaded:.1f} MB / {mb_total:.1f} MB"
                            )

            # 다운로드 완료 후 파일 이동
            if temp_path.exists():
                if model_path.exists():
                    model_path.unlink()
                temp_path.rename(model_path)

            self.finished.emit(True, "다운로드 완료!")

        except requests.exceptions.RequestException as e:
            self.finished.emit(False, f"네트워크 오류: {str(e)}")
        except Exception as e:
            self.finished.emit(False, f"다운로드 실패: {str(e)}")

    def cancel(self):
        self.cancelled = True


class ModelDownloadDialog(QDialog):
    """모델 다운로드 진행률을 보여주는 다이얼로그"""

    def __init__(self, model_name: str = "isnet-general-use", parent=None):
        super().__init__(parent)
        self.model_name = model_name
        self.download_thread = None
        self.download_success = False

        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle("AI 모델 다운로드")
        self.setFixedSize(400, 180)
        self.setWindowFlags(
            Qt.Dialog |
            Qt.WindowTitleHint |
            Qt.CustomizeWindowHint
        )

        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # 타이틀
        title_label = QLabel("배경제거 AI 모델 다운로드 중...")
        title_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(title_label)

        # 모델 정보
        model_info = MODEL_INFO.get(self.model_name, MODEL_INFO["isnet-general-use"])
        info_label = QLabel(f"모델: {self.model_name} (약 {model_info['size_mb']} MB)")
        info_label.setStyleSheet("color: #666;")
        layout.addWidget(info_label)

        # 프로그레스바
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #ccc;
                border-radius: 5px;
                text-align: center;
                height: 25px;
            }
            QProgressBar::chunk {
                background-color: #4A90E2;
                border-radius: 4px;
            }
        """)
        layout.addWidget(self.progress_bar)

        # 상태 메시지
        self.status_label = QLabel("준비 중...")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        # 취소 버튼
        self.cancel_btn = QPushButton("취소")
        self.cancel_btn.clicked.connect(self.cancel_download)
        layout.addWidget(self.cancel_btn)

    def start_download(self):
        """다운로드 시작"""
        self.download_thread = DownloadThread(self.model_name)
        self.download_thread.progress.connect(self.update_progress)
        self.download_thread.finished.connect(self.download_finished)
        self.download_thread.start()

    def update_progress(self, percent: int, message: str):
        """진행률 업데이트"""
        self.progress_bar.setValue(percent)
        self.status_label.setText(message)
        QApplication.processEvents()  # UI 업데이트

    def download_finished(self, success: bool, message: str):
        """다운로드 완료 처리"""
        self.download_success = success

        if success:
            self.progress_bar.setValue(100)
            self.status_label.setText(message)
            self.cancel_btn.setText("확인")
            self.cancel_btn.clicked.disconnect()
            self.cancel_btn.clicked.connect(self.accept)
        else:
            self.status_label.setText(f"실패: {message}")
            self.cancel_btn.setText("닫기")
            self.cancel_btn.clicked.disconnect()
            self.cancel_btn.clicked.connect(self.reject)

    def cancel_download(self):
        """다운로드 취소"""
        if self.download_thread and self.download_thread.isRunning():
            self.download_thread.cancel()
            self.download_thread.wait()
        self.reject()

    def closeEvent(self, event):
        """다이얼로그 닫을 때"""
        if self.download_thread and self.download_thread.isRunning():
            self.download_thread.cancel()
            self.download_thread.wait()
        super().closeEvent(event)


def ensure_model_downloaded(model_name: str = "isnet-general-use", parent=None) -> bool:
    """
    모델이 다운로드되어 있는지 확인하고, 없으면 다운로드 다이얼로그 표시

    Returns:
        bool: 모델이 사용 가능한지 여부
    """
    # 이미 다운로드되어 있으면 True
    if is_model_downloaded(model_name):
        return True

    # 다운로드 다이얼로그 표시
    dialog = ModelDownloadDialog(model_name, parent)
    dialog.start_download()
    result = dialog.exec()

    return dialog.download_success


def force_download_model(model_name: str = "isnet-general-use", parent=None) -> bool:
    """
    모델을 강제로 다운로드 (기존 파일 덮어쓰기)
    항상 다운로드 다이얼로그를 표시

    Returns:
        bool: 다운로드 성공 여부
    """
    # 기존 모델 파일 삭제
    model_path = get_model_cache_path(model_name)
    if model_path.exists():
        try:
            model_path.unlink()
            safe_print(f"[AI] 기존 모델 파일 삭제: {model_path}")
        except Exception as e:
            safe_print(f"[AI] 기존 모델 삭제 실패: {e}")

    # 다운로드 다이얼로그 표시
    dialog = ModelDownloadDialog(model_name, parent)
    dialog.start_download()
    result = dialog.exec()

    return dialog.download_success


def safe_print(msg):
    """GUI 앱에서 안전하게 출력"""
    try:
        if sys.stdout is not None:
            print(msg)
    except Exception:
        pass
