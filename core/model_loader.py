"""
AI 모델 백그라운드 로딩 관리자
"""

import threading
import time
import os
from pathlib import Path
from typing import Optional, Callable
from PySide6.QtCore import QObject, Signal
from rembg import new_session
from config import config


class ModelLoadingManager(QObject):
    """AI 모델 백그라운드 로딩 관리자"""
    
    # 시그널 정의
    loading_started = Signal(str)  # 모델명
    loading_progress = Signal(str)  # 진행 메시지
    loading_completed = Signal(object)  # 로드된 세션
    loading_failed = Signal(str)  # 오류 메시지
    
    def __init__(self):
        super().__init__()
        self.session = None
        self.is_loading = False
        self.is_loaded = False
        self.loading_thread = None
        self.model_name = config.get('ai_model', 'isnet-general-use')
        
    def start_background_loading(self):
        """백그라운드에서 모델 로딩 시작"""
        if self.is_loading or self.is_loaded:
            return
            
        print(f"🤖 AI 모델 준비 중...")
        self.is_loading = True
        self.loading_started.emit(self.model_name)
        
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
            self.loading_progress.emit("🤖 배경제거 AI 준비 중...")
            
            # 모델 파일 경로 확인
            models_dir = Path("models")
            if models_dir.exists():
                model_files = list(models_dir.glob("*.onnx"))
                if model_files:
                    self.loading_progress.emit("✅ AI 모델 파일 확인 완료")
                    print(f"✅ 로컬 AI 모델: {len(model_files)}개 파일")
                else:
                    self.loading_progress.emit("📥 AI 모델 다운로드 중...")
            else:
                self.loading_progress.emit("📥 AI 모델 다운로드 중...")
            
            # 실제 모델 세션 생성
            start_time = time.time()
            self.session = new_session(model_name=self.model_name)
            load_time = time.time() - start_time
            
            # 로딩 완료
            self.is_loading = False
            self.is_loaded = True
            
            print(f"🎉 배경제거 AI 준비 완료! ({load_time:.1f}초)")
            self.loading_completed.emit(self.session)
            
        except Exception as e:
            self.is_loading = False
            error_msg = f"배경제거 AI 준비 실패: {str(e)}"
            print(f"❌ {error_msg}")
            self.loading_failed.emit(error_msg)
    
    def get_session(self) -> Optional[object]:
        """로드된 세션 반환"""
        return self.session if self.is_loaded else None
    
    def is_ready(self) -> bool:
        """모델이 사용 가능한지 확인"""
        return self.is_loaded and self.session is not None
    
    def wait_for_loading(self, timeout: float = 30.0) -> bool:
        """모델 로딩 완료까지 대기 (블로킹)"""
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


def preload_ai_model():
    """AI 모델 사전 로딩 시작 (프로그램 시작 시 호출)"""
    loader = get_model_loader()
    if not loader.is_loading and not loader.is_loaded:
        loader.start_background_loading()
        print("🚀 프로그램과 함께 배경제거 AI 준비 시작")
    else:
        print("✅ 배경제거 AI 이미 준비됨")


def is_ai_model_ready() -> bool:
    """AI 모델이 사용 가능한지 확인"""
    loader = get_model_loader()
    return loader.is_ready()


def get_ai_session():
    """AI 모델 세션 반환 (필요시 대기)"""
    loader = get_model_loader()
    
    if loader.is_ready():
        return loader.get_session()
    
    # 아직 로딩 중이거나 시작되지 않은 경우
    if not loader.is_loading:
        print("⏳ 배경제거 AI 준비 시작...")
        loader.start_background_loading()
    
    print("⏳ 배경제거 AI 준비 완료 대기 중...")
    if loader.wait_for_loading():
        return loader.get_session()
    else:
        raise RuntimeError("배경제거 AI 준비 타임아웃")