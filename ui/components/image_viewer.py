"""
이미지 뷰어 컴포넌트
"""

import numpy as np
from PySide6.QtWidgets import QLabel
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage


class ImageViewer(QLabel):
    """이미지를 표시하는 뷰어 위젯"""
    
    def __init__(self, title=""):
        super().__init__()
        self.title = title
        self.original_pixmap = None
        self.setMinimumSize(300, 200)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._apply_style()
        self._set_placeholder_text()
        
    def _apply_style(self):
        """스타일 적용"""
        self.setStyleSheet("""
            QLabel {
                background: #FFFFFF;
                border: 2px dashed #DDD;
                border-radius: 12px;
                color: #666;
                font-size: 14px;
            }
        """)
    
    def _set_placeholder_text(self):
        """플레이스홀더 텍스트 설정"""
        placeholder = f"{self.title}\n\n이미지를 불러오세요" if self.title else "이미지를 불러오세요"
        self.setText(placeholder)
    
    def set_image(self, image_path_or_array):
        """이미지 설정 (파일 경로 또는 numpy 배열)"""
        try:
            if isinstance(image_path_or_array, str):
                # 파일 경로인 경우
                pixmap = QPixmap(image_path_or_array)
            elif isinstance(image_path_or_array, np.ndarray):
                # numpy array인 경우 QPixmap으로 변환
                pixmap = self._numpy_to_pixmap(image_path_or_array)
            else:
                return
                
            self.original_pixmap = pixmap
            self.update_display()
            
        except Exception as e:
            print(f"이미지 로드 오류: {e}")
            self._set_placeholder_text()
    
    def _numpy_to_pixmap(self, array):
        """numpy 배열을 QPixmap으로 변환"""
        if len(array.shape) == 3:
            # 컬러 이미지
            height, width, channel = array.shape
            bytes_per_line = 3 * width
            q_image = QImage(
                array.data, width, height, 
                bytes_per_line, QImage.Format.Format_RGB888
            ).rgbSwapped()
        else:
            # 그레이스케일 이미지
            height, width = array.shape
            bytes_per_line = width
            q_image = QImage(
                array.data, width, height, 
                bytes_per_line, QImage.Format.Format_Grayscale8
            )
        
        return QPixmap.fromImage(q_image)
    
    def update_display(self):
        """디스플레이 업데이트 (크기에 맞게 조정)"""
        if self.original_pixmap:
            # 비율을 유지하면서 크기 조정
            scaled_pixmap = self.original_pixmap.scaled(
                self.size(), 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            )
            self.setPixmap(scaled_pixmap)
    
    def clear_image(self):
        """이미지 클리어"""
        self.original_pixmap = None
        self.clear()
        self._set_placeholder_text()
    
    def resizeEvent(self, event):
        """리사이즈 이벤트 처리"""
        super().resizeEvent(event)
        if self.original_pixmap:
            self.update_display()