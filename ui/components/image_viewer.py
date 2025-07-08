"""
이미지 뷰어 컴포넌트 - 회전 기능 추가
사용자가 직접 이미지를 회전시킬 수 있는 버튼 포함
"""

import numpy as np
from PySide6.QtWidgets import QLabel, QVBoxLayout, QHBoxLayout, QWidget, QPushButton
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap, QImage, QTransform, QFont
import cv2


class RotateButton(QPushButton):
    """회전 버튼 컴포넌트"""
    
    def __init__(self, text, direction="left"):
        super().__init__(text)
        self.direction = direction
        self.setFixedSize(30, 30)
        self.setFont(QFont("Arial", 12))
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        
        # 스타일 적용
        self.setStyleSheet("""
            QPushButton {
                background-color: #F8F9FA;
                border: 1px solid #DEE2E6;
                border-radius: 4px;
                color: #495057;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #E9ECEF;
                border-color: #ADB5BD;
            }
            QPushButton:pressed {
                background-color: #DEE2E6;
            }
            QPushButton:disabled {
                background-color: #F8F9FA;
                color: #CED4DA;
                border-color: #E9ECEF;
            }
        """)


class ImageViewer(QWidget):
    """회전 기능이 있는 이미지 뷰어 위젯"""
    
    # 이미지 회전 시그널
    image_rotated = Signal()
    
    def __init__(self, title=""):
        super().__init__()
        self.title = title
        self.original_pixmap = None
        self.current_rotation = 0  # 현재 회전 각도 (0, 90, 180, 270)
        self.current_image_array = None  # numpy 배열로 저장된 현재 이미지
        self.image_path = None  # 현재 이미지 경로
        
        self.setMinimumSize(300, 200)
        self._setup_ui()
        self._set_placeholder_text()
        
    def _setup_ui(self):
        """UI 구성"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        
        # 이미지 표시 라벨
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("""
            QLabel {
                background: #FFFFFF;
                border: 2px dashed #DDD;
                border-radius: 12px;
                color: #666;
                font-size: 14px;
            }
        """)
        
        # 회전 버튼들
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(5)
        
        self.rotate_left_btn = RotateButton("↶", "left")
        self.rotate_right_btn = RotateButton("↷", "right")
        
        self.rotate_left_btn.clicked.connect(self.rotate_left)
        self.rotate_right_btn.clicked.connect(self.rotate_right)
        
        # 초기에는 버튼 비활성화
        self.rotate_left_btn.setEnabled(False)
        self.rotate_right_btn.setEnabled(False)
        
        button_layout.addStretch()
        button_layout.addWidget(self.rotate_left_btn)
        button_layout.addWidget(self.rotate_right_btn)
        button_layout.addStretch()
        
        layout.addWidget(self.image_label)
        layout.addLayout(button_layout)
    
    def _set_placeholder_text(self):
        """플레이스홀더 텍스트 설정"""
        placeholder = f"{self.title}\n\n이미지를 불러오세요" if self.title else "이미지를 불러오세요"
        self.image_label.setText(placeholder)
    
    def set_image(self, image_path_or_array):
        """이미지 설정 (파일 경로 또는 numpy 배열) - 디버깅 개선"""
        try:
            self.current_rotation = 0  # 회전 각도 초기화
            print(f"[DEBUG] set_image 호출됨: {type(image_path_or_array)}")
            
            if isinstance(image_path_or_array, str):
                # 파일 경로인 경우
                print(f"[DEBUG] 파일 경로로 이미지 로드: {image_path_or_array}")
                self.image_path = image_path_or_array
                
                # OpenCV로 이미지 읽기
                self.current_image_array = cv2.imread(image_path_or_array)
                if self.current_image_array is None:
                    print(f"[DEBUG] OpenCV로 이미지 읽기 실패: {image_path_or_array}")
                    raise ValueError("이미지를 읽을 수 없습니다")
                
                print(f"[DEBUG] 이미지 크기: {self.current_image_array.shape}")
                
                # QPixmap으로도 로드
                pixmap = QPixmap(image_path_or_array)
                if pixmap.isNull():
                    print(f"[DEBUG] QPixmap 로드 실패, numpy 배열로 변환 시도")
                    pixmap = self._numpy_to_pixmap(self.current_image_array)
                
            elif isinstance(image_path_or_array, np.ndarray):
                # numpy array인 경우
                print(f"[DEBUG] numpy 배열로 이미지 설정: {image_path_or_array.shape}")
                self.image_path = None
                self.current_image_array = image_path_or_array.copy()
                pixmap = self._numpy_to_pixmap(image_path_or_array)
            else:
                print(f"[DEBUG] 지원하지 않는 타입: {type(image_path_or_array)}")
                return
            
            if pixmap.isNull():
                print("[DEBUG] QPixmap이 null임")
                raise ValueError("QPixmap 생성 실패")
                
            self.original_pixmap = pixmap
            print(f"[DEBUG] QPixmap 크기: {pixmap.width()}x{pixmap.height()}")
            
            self.update_display()
            
            # 이미지가 로드되면 회전 버튼 활성화
            self.rotate_left_btn.setEnabled(True)
            self.rotate_right_btn.setEnabled(True)
            
            print("[DEBUG] 이미지 설정 완료!")
            
        except Exception as e:
            print(f"[DEBUG] 이미지 로드 오류: {e}")
            import traceback
            traceback.print_exc()
            self._set_placeholder_text()
            self.rotate_left_btn.setEnabled(False)
            self.rotate_right_btn.setEnabled(False)
    
    def _numpy_to_pixmap(self, array):
        """numpy 배열을 QPixmap으로 변환 - 메모리 관리 개선"""
        try:
            if array is None:
                return QPixmap()
            
            # 배열이 연속적인지 확인
            if not array.flags.c_contiguous:
                array = np.ascontiguousarray(array)
            
            if len(array.shape) == 3:
                # 컬러 이미지 (BGR을 RGB로 변환)
                height, width, channel = array.shape
                if channel == 3:
                    rgb_array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
                    bytes_per_line = 3 * width
                    q_image = QImage(
                        rgb_array.data.tobytes(), width, height, 
                        bytes_per_line, QImage.Format.Format_RGB888
                    )
                elif channel == 4:
                    # RGBA 이미지
                    rgba_array = cv2.cvtColor(array, cv2.COLOR_BGRA2RGBA)
                    bytes_per_line = 4 * width
                    q_image = QImage(
                        rgba_array.data.tobytes(), width, height, 
                        bytes_per_line, QImage.Format.Format_RGBA8888
                    )
                else:
                    return QPixmap()
            else:
                # 그레이스케일 이미지
                height, width = array.shape
                bytes_per_line = width
                q_image = QImage(
                    array.data.tobytes(), width, height, 
                    bytes_per_line, QImage.Format.Format_Grayscale8
                )
            
            if q_image.isNull():
                return QPixmap()
                
            return QPixmap.fromImage(q_image)
            
        except Exception as e:
            print(f"numpy to pixmap 변환 오류: {e}")
            return QPixmap()
    
    def rotate_left(self):
        """왼쪽으로 90도 회전"""
        if self.current_image_array is not None:
            self.current_rotation = (self.current_rotation - 90) % 360
            self.current_image_array = cv2.rotate(self.current_image_array, cv2.ROTATE_90_COUNTERCLOCKWISE)
            self.original_pixmap = self._numpy_to_pixmap(self.current_image_array)
            self.update_display()
            self.image_rotated.emit()
    
    def rotate_right(self):
        """오른쪽으로 90도 회전"""
        if self.current_image_array is not None:
            self.current_rotation = (self.current_rotation + 90) % 360
            self.current_image_array = cv2.rotate(self.current_image_array, cv2.ROTATE_90_CLOCKWISE)
            self.original_pixmap = self._numpy_to_pixmap(self.current_image_array)
            self.update_display()
            self.image_rotated.emit()
    
    def get_current_image_array(self):
        """현재 표시되고 있는 이미지를 numpy 배열로 반환"""
        return self.current_image_array.copy() if self.current_image_array is not None else None
    
    def get_rotation_angle(self):
        """현재 회전 각도 반환"""
        return self.current_rotation
    
    def update_display(self):
        """디스플레이 업데이트 (크기에 맞게 조정) - 디버깅 개선"""
        if self.original_pixmap and not self.original_pixmap.isNull():
            try:
                print(f"[DEBUG] update_display 호출됨")
                print(f"[DEBUG] image_label 크기: {self.image_label.size()}")
                print(f"[DEBUG] original_pixmap 크기: {self.original_pixmap.size()}")
                
                # 비율을 유지하면서 크기 조정
                label_size = self.image_label.size()
                if label_size.width() > 0 and label_size.height() > 0:
                    scaled_pixmap = self.original_pixmap.scaled(
                        label_size, 
                        Qt.AspectRatioMode.KeepAspectRatio, 
                        Qt.TransformationMode.SmoothTransformation
                    )
                    print(f"[DEBUG] scaled_pixmap 크기: {scaled_pixmap.size()}")
                    self.image_label.setPixmap(scaled_pixmap)
                    print("[DEBUG] setPixmap 완료")
                else:
                    print("[DEBUG] image_label 크기가 0임, 나중에 다시 시도")
                    # QTimer로 조금 후에 다시 시도
                    from PySide6.QtCore import QTimer
                    QTimer.singleShot(100, self.update_display)
                    
            except Exception as e:
                print(f"[DEBUG] update_display 오류: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("[DEBUG] original_pixmap이 없거나 null임")
    
    def clear_image(self):
        """이미지 클리어"""
        self.original_pixmap = None
        self.current_image_array = None
        self.current_rotation = 0
        self.image_path = None
        self.image_label.clear()
        self._set_placeholder_text()
        
        # 회전 버튼 비활성화
        self.rotate_left_btn.setEnabled(False)
        self.rotate_right_btn.setEnabled(False)
    
    def resizeEvent(self, event):
        """리사이즈 이벤트 처리"""
        super().resizeEvent(event)
        if self.original_pixmap:
            self.update_display()