"""
이미지 뷰어 컴포넌트 - 회전 기능 추가 및 미리보기 문제 해결
사용자가 직접 이미지를 회전시킬 수 있는 버튼 포함
"""

import numpy as np
from PySide6.QtWidgets import QLabel, QVBoxLayout, QHBoxLayout, QWidget, QPushButton
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QPixmap, QImage, QTransform, QFont
import cv2
import os


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
    """회전 기능이 있는 이미지 뷰어 위젯 - 미리보기 문제 해결"""
    
    # 이미지 회전 시그널
    image_rotated = Signal()
    
    def __init__(self, title=""):
        super().__init__()
        self.title = title
        self.original_pixmap = None
        self.current_rotation = 0  # 현재 회전 각도 (0, 90, 180, 270)
        self.current_image_array = None  # numpy 배열로 저장된 현재 이미지
        self.original_image_array = None  # 회전 전 원본 배열
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
    
    def _safe_imread_unicode(self, image_path: str) -> np.ndarray:
        """한글 경로를 지원하는 안전한 이미지 읽기"""
        try:
            # 파일 존재 여부 확인
            if not os.path.exists(image_path):
                print(f"[DEBUG] 파일이 존재하지 않음: {image_path}")
                return None
            
            # OpenCV가 한글 경로를 읽지 못하는 문제 해결
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            # numpy array로 변환 후 OpenCV로 디코딩
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                print(f"[DEBUG] 이미지 디코딩 실패: {image_path}")
                return None
                
            print(f"[DEBUG] 이미지 로드 성공: {image.shape}")
            return image
            
        except Exception as e:
            print(f"[DEBUG] 이미지 읽기 실패: {image_path}, 오류: {e}")
            return None
    
    def set_image(self, image_path_or_array):
        """이미지 설정 (파일 경로 또는 numpy 배열) - 문제 해결된 버전"""
        try:
            self.current_rotation = 0  # 회전 각도 초기화
            print(f"[DEBUG] set_image 호출됨: {type(image_path_or_array)}")
            
            if isinstance(image_path_or_array, str):
                # 파일 경로인 경우
                print(f"[DEBUG] 파일 경로로 이미지 로드: {image_path_or_array}")
                self.image_path = image_path_or_array
                
                # 안전한 이미지 읽기 (한글 경로 지원)
                image_array = self._safe_imread_unicode(image_path_or_array)
                if image_array is None:
                    print(f"[DEBUG] 이미지 읽기 실패: {image_path_or_array}")
                    self._set_placeholder_text()
                    return
                
                self.original_image_array = image_array.copy()
                self.current_image_array = image_array.copy()
                
                print(f"[DEBUG] 이미지 크기: {self.current_image_array.shape}")
                
            elif isinstance(image_path_or_array, np.ndarray):
                # numpy array인 경우
                print(f"[DEBUG] numpy 배열로 이미지 설정: {image_path_or_array.shape}")
                self.image_path = None
                self.original_image_array = image_path_or_array.copy()
                self.current_image_array = image_path_or_array.copy()
            else:
                print(f"[DEBUG] 지원하지 않는 타입: {type(image_path_or_array)}")
                self._set_placeholder_text()
                return
            
            # QPixmap 생성
            pixmap = self._numpy_to_pixmap(self.current_image_array)
            if pixmap.isNull():
                print("[DEBUG] QPixmap 생성 실패")
                self._set_placeholder_text()
                return
                
            self.original_pixmap = pixmap
            print(f"[DEBUG] QPixmap 크기: {pixmap.width()}x{pixmap.height()}")
            
            # 이미지가 로드되면 회전 버튼 활성화
            self.rotate_left_btn.setEnabled(True)
            self.rotate_right_btn.setEnabled(True)
            
            # 디스플레이 업데이트 (즉시 실행)
            self.update_display()
            
            # 조금 후에 한번 더 업데이트 (위젯 크기가 확정된 후)
            QTimer.singleShot(100, self.update_display)
            
            print("[DEBUG] 이미지 설정 완료!")
            
        except Exception as e:
            print(f"[DEBUG] 이미지 로드 오류: {e}")
            import traceback
            traceback.print_exc()
            self._set_placeholder_text()
            self.rotate_left_btn.setEnabled(False)
            self.rotate_right_btn.setEnabled(False)
    
    def _numpy_to_pixmap(self, array):
        """numpy 배열을 QPixmap으로 변환 - 개선된 버전"""
        try:
            if array is None:
                print("[DEBUG] array가 None임")
                return QPixmap()
            
            # 배열이 연속적인지 확인
            if not array.flags.c_contiguous:
                array = np.ascontiguousarray(array)
            
            print(f"[DEBUG] numpy to pixmap: {array.shape}, dtype: {array.dtype}")
            
            if len(array.shape) == 3:
                # 컬러 이미지 (BGR을 RGB로 변환)
                height, width, channel = array.shape
                if channel == 3:
                    # BGR을 RGB로 변환
                    rgb_array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
                    bytes_per_line = 3 * width
                    q_image = QImage(
                        rgb_array.data, width, height, 
                        bytes_per_line, QImage.Format.Format_RGB888
                    )
                elif channel == 4:
                    # RGBA 이미지
                    rgba_array = cv2.cvtColor(array, cv2.COLOR_BGRA2RGBA)
                    bytes_per_line = 4 * width
                    q_image = QImage(
                        rgba_array.data, width, height, 
                        bytes_per_line, QImage.Format.Format_RGBA8888
                    )
                else:
                    print(f"[DEBUG] 지원하지 않는 채널 수: {channel}")
                    return QPixmap()
            elif len(array.shape) == 2:
                # 그레이스케일 이미지
                height, width = array.shape
                bytes_per_line = width
                q_image = QImage(
                    array.data, width, height, 
                    bytes_per_line, QImage.Format.Format_Grayscale8
                )
            else:
                print(f"[DEBUG] 지원하지 않는 배열 차원: {array.shape}")
                return QPixmap()
            
            if q_image.isNull():
                print("[DEBUG] QImage 생성 실패")
                return QPixmap()
            
            pixmap = QPixmap.fromImage(q_image)
            print(f"[DEBUG] QPixmap 생성 성공: {pixmap.width()}x{pixmap.height()}")
            return pixmap
            
        except Exception as e:
            print(f"[DEBUG] numpy to pixmap 변환 오류: {e}")
            import traceback
            traceback.print_exc()
            return QPixmap()
    
    def rotate_left(self):
        """왼쪽으로 90도 회전"""
        if self.current_image_array is not None:
            print("[DEBUG] 왼쪽 회전 시작")
            self.current_rotation = (self.current_rotation - 90) % 360
            self.current_image_array = cv2.rotate(self.current_image_array, cv2.ROTATE_90_COUNTERCLOCKWISE)
            self.original_pixmap = self._numpy_to_pixmap(self.current_image_array)
            self.update_display()
            self.image_rotated.emit()
            print(f"[DEBUG] 왼쪽 회전 완료: {self.current_rotation}도")
    
    def rotate_right(self):
        """오른쪽으로 90도 회전"""
        if self.current_image_array is not None:
            print("[DEBUG] 오른쪽 회전 시작")
            self.current_rotation = (self.current_rotation + 90) % 360
            self.current_image_array = cv2.rotate(self.current_image_array, cv2.ROTATE_90_CLOCKWISE)
            self.original_pixmap = self._numpy_to_pixmap(self.current_image_array)
            self.update_display()
            self.image_rotated.emit()
            print(f"[DEBUG] 오른쪽 회전 완료: {self.current_rotation}도")
    
    def get_current_image_array(self):
        """현재 표시되고 있는 이미지를 numpy 배열로 반환"""
        return self.current_image_array.copy() if self.current_image_array is not None else None
    
    def get_rotation_angle(self):
        """현재 회전 각도 반환"""
        return self.current_rotation
    
    def update_display(self):
        """디스플레이 업데이트 (크기에 맞게 조정) - 문제 해결된 버전"""
        if self.original_pixmap is None or self.original_pixmap.isNull():
            print("[DEBUG] original_pixmap이 없거나 null임")
            return
            
        try:
            print(f"[DEBUG] update_display 시작")
            
            # 위젯 크기 확인
            widget_size = self.size()
            label_size = self.image_label.size()
            
            print(f"[DEBUG] widget 크기: {widget_size}")
            print(f"[DEBUG] image_label 크기: {label_size}")
            print(f"[DEBUG] original_pixmap 크기: {self.original_pixmap.size()}")
            
            # 라벨 크기가 유효한지 확인
            if label_size.width() <= 0 or label_size.height() <= 0:
                print("[DEBUG] 라벨 크기가 유효하지 않음, 대기 후 재시도")
                QTimer.singleShot(50, self.update_display)
                return
            
            # 버튼 공간을 고려한 실제 이미지 표시 영역 계산
            available_height = label_size.height() - 10  # 여백 고려
            available_width = label_size.width() - 10    # 여백 고려
            
            if available_height <= 0 or available_width <= 0:
                print("[DEBUG] 사용 가능한 영역이 없음")
                return
            
            # 비율을 유지하면서 크기 조정
            scaled_pixmap = self.original_pixmap.scaled(
                available_width, 
                available_height,
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            )
            
            print(f"[DEBUG] scaled_pixmap 크기: {scaled_pixmap.size()}")
            
            if not scaled_pixmap.isNull():
                self.image_label.setPixmap(scaled_pixmap)
                print("[DEBUG] setPixmap 완료")
            else:
                print("[DEBUG] scaled_pixmap이 null임")
                
        except Exception as e:
            print(f"[DEBUG] update_display 오류: {e}")
            import traceback
            traceback.print_exc()
    
    def clear_image(self):
        """이미지 클리어"""
        print("[DEBUG] 이미지 클리어")
        self.original_pixmap = None
        self.current_image_array = None
        self.original_image_array = None
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
        print(f"[DEBUG] resizeEvent: {event.size()}")
        # 리사이즈 후 잠시 후에 디스플레이 업데이트
        if self.original_pixmap:
            QTimer.singleShot(10, self.update_display)
    
    def showEvent(self, event):
        """위젯이 표시될 때 이벤트"""
        super().showEvent(event)
        print("[DEBUG] showEvent 발생")
        # 위젯이 표시된 후 이미지 업데이트
        if self.original_pixmap:
            QTimer.singleShot(50, self.update_display)