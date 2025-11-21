"""
AI 모델 백그라운드 로딩 관리자
UI가 완전히 뜬 후에만 모델 로딩 시작
모델이 없으면 다운로드 다이얼로그 표시
"""

import threading
import time
import os
import sys
from pathlib import Path
from typing import Optional
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication
from config import config
from utils.safe_temp_path import get_safe_temp_dir
from core.model_downloader import (
    ensure_model_downloaded,
    is_model_downloaded,
    get_model_cache_path
)


def safe_print(msg):
    """GUI 앱에서 stdout이 None일 때도 안전하게 출력"""
    try:
        if sys.stdout is not None:
            print(msg)
    except Exception:
        pass


class ModelLoadingManager(QObject):
    """AI 모델 백그라운드 로딩 관리자"""

    # 시그널 정의
    loading_started = Signal(str)  # 모델명
    loading_progress = Signal(str)  # 진행 메시지
    loading_completed = Signal(object)  # 로드된 세션
    loading_failed = Signal(str)  # 오류 메시지
    download_required = Signal()  # 다운로드 필요 시그널

    def __init__(self):
        super().__init__()
        self.session = None
        self.is_loading = False
        self.is_loaded = False
        self.loading_thread = None
        self.model_name = config.get('ai_model', 'isnet-general-use')
        self._parent_widget = None  # 다이얼로그 부모

    def set_parent_widget(self, widget):
        """다운로드 다이얼로그의 부모 위젯 설정"""
        self._parent_widget = widget

    def start_background_loading(self):
        """백그라운드에서 모델 로딩 시작"""
        if self.is_loading or self.is_loaded:
            return

        safe_print(f"[AI] 모델 준비 중...")
        self.is_loading = True
        self.loading_started.emit(self.model_name)

        # 모델이 없으면 먼저 다운로드 (메인 스레드에서 다이얼로그 표시)
        if not is_model_downloaded(self.model_name):
            safe_print(f"[AI] 모델 다운로드 필요: {self.model_name}")

            # 다운로드 다이얼로그 표시
            if not ensure_model_downloaded(self.model_name, self._parent_widget):
                # 다운로드 취소됨
                self.is_loading = False
                self.loading_failed.emit("모델 다운로드가 취소되었습니다")
                return

            safe_print(f"[AI] 모델 다운로드 완료!")

        # 별도 스레드에서 로딩
        self.loading_thread = threading.Thread(
            target=self._load_model_worker,
            daemon=True
        )
        self.loading_thread.start()

    def _load_model_worker(self):
        """실제 모델 로딩 작업 (백그라운드 스레드)"""
        try:
            # 진행 상황 업데이트
            self.loading_progress.emit("배경제거 AI 준비 중...")

            # 한글 경로 문제 해결: 모든 관련 환경변수를 ASCII 안전 경로로 설정
            try:
                safe_temp = get_safe_temp_dir()

                # rembg 모델 캐시
                u2net_home = os.path.join(safe_temp, "u2net_models")
                os.makedirs(u2net_home, exist_ok=True)
                os.environ['U2NET_HOME'] = u2net_home

                # onnxruntime 캐시
                onnx_cache = os.path.join(safe_temp, "onnx_cache")
                os.makedirs(onnx_cache, exist_ok=True)
                os.environ['ORT_GLOBAL_THREAD_POOL_OPTIONS'] = ''

                # huggingface 캐시
                hf_cache = os.path.join(safe_temp, "hf_cache")
                os.makedirs(hf_cache, exist_ok=True)
                os.environ['HF_HOME'] = hf_cache
                os.environ['HUGGINGFACE_HUB_CACHE'] = hf_cache
                os.environ['XDG_CACHE_HOME'] = safe_temp

                # 임시 디렉토리 (onnxruntime이 사용)
                os.environ['TEMP'] = safe_temp
                os.environ['TMP'] = safe_temp

                safe_print(f"[AI] ASCII 안전 경로 설정 완료: {safe_temp}")
            except Exception as e:
                safe_print(f"[AI] 안전 경로 설정 실패 (기본 경로 사용): {e}")

            # 모델 파일 경로 확인
            models_dir = Path("models")
            if models_dir.exists():
                model_files = list(models_dir.glob("*.onnx"))
                if model_files:
                    self.loading_progress.emit("AI 모델 파일 확인 완료")
                    safe_print(f"[AI] 로컬 AI 모델: {len(model_files)}개 파일")
                else:
                    self.loading_progress.emit("AI 모델 다운로드 중...")
            else:
                self.loading_progress.emit("AI 모델 다운로드 중...")

            # 실제 모델 세션 생성
            start_time = time.time()
            safe_print(f"[AI] 세션 생성 시작... (모델: {self.model_name})")

            # U2NET_HOME 경로 확인
            u2net_home = os.environ.get('U2NET_HOME', 'not set')
            safe_print(f"[AI] U2NET_HOME: {u2net_home}")

            # ✨ PyInstaller 첫 실행 문제 해결: scipy C-확장 모듈 사전 로딩
            # rembg가 scipy를 사용하는데, PyInstaller onefile 모드에서는
            # C-확장(.pyd) 파일들이 런타임에 임시 디렉토리로 unpack됨.
            # 백그라운드 스레드에서 rembg import 시도 시, scipy C-확장이
            # 아직 준비되지 않아 ImportError 발생 가능.
            # 해결: rembg import 전에 scipy를 명시적으로 import하여
            # C-확장 로딩을 강제로 완료시킴.
            try:
                safe_print(f"[AI] scipy 의존성 로딩 중...")
                import scipy
                import scipy.ndimage
                import scipy.signal
                safe_print(f"[AI] scipy 의존성 로딩 완료")
            except Exception as scipy_err:
                safe_print(f"[AI] scipy 로딩 경고: {scipy_err}")
                # scipy 로딩 실패해도 계속 진행 (rembg가 자체적으로 처리할 수도 있음)

            try:
                safe_print(f"[AI] rembg 모듈 import 시작...")
                from rembg import new_session
                safe_print(f"[AI] rembg 모듈 import 완료!")
            except Exception as import_err:
                safe_print(f"[AI] rembg import 실패: {import_err}")
                raise

            safe_print(f"[AI] rembg.new_session 호출 중...")
            self.session = new_session(model_name=self.model_name)
            safe_print(f"[AI] 세션 생성 완료!")
            load_time = time.time() - start_time

            # 로딩 완료
            self.is_loading = False
            self.is_loaded = True

            safe_print(f"[AI] 배경제거 AI 준비 완료! ({load_time:.1f}초)")
            self.loading_completed.emit(self.session)

        except Exception as e:
            self.is_loading = False
            error_msg = f"배경제거 AI 준비 실패: {str(e)}"
            safe_print(f"[AI] {error_msg}")
            self.loading_failed.emit(error_msg)

    def get_session(self) -> Optional[object]:
        """로드된 세션 반환 (하위 호환성 유지)"""
        return self.session if self.is_loaded else None

    def get_loaded_session(self) -> Optional[object]:
        """
        이미 로딩된 세션만 반환 (로딩 시작 안 함)

        Returns:
            로딩 완료된 세션 또는 None
        """
        return self.session if self.is_loaded else None

    def is_ready(self) -> bool:
        """모델이 사용 가능한지 확인"""
        return self.is_loaded and self.session is not None

    def wait_for_loading(self, timeout: float = 180.0) -> bool:
        """모델 로딩 완료까지 대기 (블로킹) - 첫 실행 고려하여 180초로 증가"""
        if self.is_loaded:
            return True

        if not self.is_loading:
            self.start_background_loading()

        # 로딩 완료까지 대기
        start_time = time.time()
        while self.is_loading and (time.time() - start_time) < timeout:
            time.sleep(0.1)

        return self.is_loaded


# 전역 모델 로더 인스턴스
_model_loader = None


def get_model_loader() -> ModelLoadingManager:
    """전역 모델 로더 인스턴스 반환"""
    global _model_loader
    if _model_loader is None:
        _model_loader = ModelLoadingManager()
    return _model_loader


def is_ai_model_ready() -> bool:
    """AI 모델이 사용 가능한지 확인"""
    loader = get_model_loader()
    return loader.is_ready()


def get_ai_session():
    """
    [DEPRECATED] 이미 로딩된 세션만 반환 (로딩 시작 안 함)

    직접 get_model_loader().get_loaded_session() 사용 권장

    Returns:
        이미 로딩된 세션 또는 None
    """
    loader = get_model_loader()
    return loader.get_loaded_session()
