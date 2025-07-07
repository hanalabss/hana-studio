"""
이미지 처리 스레드
"""

import numpy as np
from PySide6.QtCore import QThread, Signal
from .image_processor import ImageProcessor


class ProcessingThread(QThread):
    """이미지 처리를 백그라운드에서 실행하는 스레드"""
    finished = Signal(np.ndarray)
    error = Signal(str)
    progress = Signal(str)
    
    def __init__(self, image_path: str, processor: ImageProcessor):
        super().__init__()
        self.image_path = image_path
        self.processor = processor
        
    def run(self):
        """스레드 실행"""
        try:
            self.progress.emit("AI 모델 로딩 중...")
            
            # 모델 준비 상태 확인
            if not self.processor.is_model_ready():
                self.error.emit("AI 모델이 준비되지 않았습니다.")
                return
            
            self.progress.emit("배경 제거 처리 중...")
            
            # 배경 제거 실행
            mask_result = self.processor.remove_background(self.image_path)
            
            self.progress.emit("마스크 생성 완료!")
            self.finished.emit(mask_result)
            
        except Exception as e:
            self.error.emit(str(e))