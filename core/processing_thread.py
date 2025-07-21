"""
core/processing_thread.py 수정
임계값을 지원하는 이미지 처리 스레드
"""

import numpy as np
from PySide6.QtCore import QThread, Signal
from .image_processor import ImageProcessor


class ProcessingThread(QThread):
    """이미지 처리를 백그라운드에서 실행하는 스레드 - 임계값 지원"""
    finished = Signal(np.ndarray)
    error = Signal(str)
    progress = Signal(str)
    
    def __init__(self, image_path: str, processor: ImageProcessor, alpha_threshold: int = None):
        super().__init__()
        self.image_path = image_path
        self.processor = processor
        self.alpha_threshold = alpha_threshold
        
    def run(self):
        """스레드 실행"""
        try:
            self.progress.emit("🔄 이미지 처리 중...")
            
            # 모델 준비 상태 확인
            if not self.processor.is_model_ready():
                self.error.emit("AI 모델이 준비되지 않았습니다.")
                return
            
            # 단순화된 진행상황 표시
            self.progress.emit("🔄 이미지 처리 중...")
            
            # 배경 제거 실행 - 임계값 전달
            mask_result = self.processor.remove_background(
                self.image_path, 
                alpha_threshold=self.alpha_threshold
            )
            
            self.progress.emit("✅ 이미지 처리 완료!")
            self.finished.emit(mask_result)
            
        except Exception as e:
            self.error.emit(str(e))