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

    def __init__(self, image_path: str, processor: ImageProcessor, session, alpha_threshold: int = None):
        """
        Args:
            image_path: 처리할 이미지 경로
            processor: ImageProcessor 인스턴스
            session: AI 모델 세션 (필수)
            alpha_threshold: 알파 임계값 (선택)
        """
        super().__init__()
        self.image_path = image_path
        self.processor = processor
        self.session = session
        self.alpha_threshold = alpha_threshold
        
    def run(self):
        """스레드 실행"""
        try:
            # 세션 검증
            if not self.session:
                self.error.emit("AI 모델이 준비되지 않았습니다. 잠시 후 다시 시도해주세요.")
                return

            self.progress.emit("🔄 이미지 처리 중...")

            # 배경 제거 실행 - 세션과 임계값 전달
            mask_result = self.processor.remove_background(
                self.image_path,
                session=self.session,  # 명시적으로 세션 전달
                alpha_threshold=self.alpha_threshold
            )

            self.progress.emit("✅ 이미지 처리 완료!")
            self.finished.emit(mask_result)

        except Exception as e:
            self.error.emit(str(e))