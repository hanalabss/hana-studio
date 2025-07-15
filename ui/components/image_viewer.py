"""
ui/components/image_viewer.py 수정
회전 기능 제거, 각 면별 출력 방향 선택 기능 추가
"""

import numpy as np
from PySide6.QtWidgets import (
    QLabel, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QFileDialog,
    QSlider, QSizePolicy, QButtonGroup, QRadioButton,QSpacerItem
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QPixmap, QImage, QFont
import cv2
import os


class OrientationButton(QRadioButton):
    """방향 선택 라디오 버튼"""
    
    def __init__(self, text, orientation="portrait"):
        super().__init__(text)
        self.orientation = orientation
        self.setFixedSize(65, 28)  # 높이 줄임 (32 → 28)
        self.setFont(QFont("Segoe UI", 9))
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        
        # 스타일 적용
        self.setStyleSheet("""
            QRadioButton {
                background-color: #F8F9FA;
                border: 1px solid #DEE2E6;
                border-radius: 4px;
                color: #495057;
                font-weight: 500;
                padding: 6px 8px;
                spacing: 4px;
            }
            QRadioButton:hover {
                background-color: #E9ECEF;
                border-color: #ADB5BD;
            }
            QRadioButton:checked {
                background-color: #4A90E2;
                border-color: #357ABD;
                color: white;
                font-weight: 600;
            }
            QRadioButton::indicator {
                width: 0px;
                height: 0px;
            }
        """)


class ProcessButton(QPushButton):
    """배경제거 버튼 컴포넌트"""
    
    def __init__(self, text):
        super().__init__(text)
        self.setFixedSize(80, 28)  # 높이 줄임 (32 → 28)
        self.setFont(QFont("Segoe UI", 9, QFont.Weight.Medium))
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        
        self.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                           stop: 0 #4A90E2, stop: 1 #357ABD);
                color: white;
                border: none;
                border-radius: 4px;
                padding: 0 6px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                           stop: 0 #5BA0F2, stop: 1 #4A90E2);
            }
            QPushButton:pressed {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                           stop: 0 #357ABD, stop: 1 #2E6B9E);
            }
            QPushButton:disabled {
                background: #CCCCCC;
                color: #888888;
            }
        """)


class CompactThresholdSlider(QWidget):
    """컴팩트한 임계값 슬라이더"""
    threshold_changed = Signal(int)

    def __init__(self, initial_value=45):
        super().__init__()
        self.setFixedSize(120, 28)
        self._setup_ui(initial_value)

    def _setup_ui(self, initial_value):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        label = QLabel("T:")
        label.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        label.setStyleSheet("color: #6C757D; background: transparent;")
        label.setFixedWidth(12)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 255)
        self.slider.setValue(initial_value)
        self.slider.setFixedWidth(70)
        self.slider.setFixedHeight(24)
        self.slider.setStyleSheet("""<슬라이더 스타일 생략>""")

        self.value_label = QLabel(str(initial_value))
        self.value_label.setFont(QFont("Segoe UI", 9))
        self.value_label.setStyleSheet("""<라벨 스타일 생략>""")
        self.value_label.setFixedSize(28, 24)
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(label)
        layout.addWidget(self.slider)
        layout.addWidget(self.value_label)

        self.slider.valueChanged.connect(self._on_value_changed)

    def _on_value_changed(self, value):
        self.value_label.setText(str(value))
        self.threshold_changed.emit(value)

    def get_value(self):
        return self.slider.value()

    def set_value(self, value):
        self.slider.setValue(value)



class ImageViewer(QWidget):
    """각 면별 출력 방향 선택이 가능한 이미지 뷰어 위젯"""
    
    # 시그널들
    image_clicked = Signal()
    file_uploaded = Signal(str)
    process_requested = Signal(int)  # 임계값 포함
    threshold_changed = Signal(int)
    orientation_changed = Signal(str)  # "portrait" 또는 "landscape"
    
    def __init__(self, title="", enable_click_upload=False, enable_process_button=False, show_orientation_buttons=True):
        super().__init__()
        self.title = title
        self.enable_click_upload = enable_click_upload
        self.enable_process_button = enable_process_button
        self.show_orientation_buttons = show_orientation_buttons  # 추가
        self.original_pixmap = None
        self.current_image_array = None
        self.original_image_array = None
        self.image_path = None
        self.current_orientation = "portrait"  # 현재 출력 방향
        
        self.setMinimumSize(160, 0)  # 높이 감소 (280 → 260, 이미지 미리보기 공간 축소)
        self._setup_ui()
        self._set_placeholder_text()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        # layout.setContentsMargins(0, 2, 6, 2)  # 위/아래 여백 최소화
        layout.setSpacing(0)

        # 배경제거 컨트롤
        if self.enable_process_button:
            control_layout = QHBoxLayout()
            control_layout.setSpacing(2)

            self.process_btn = ProcessButton("배경제거")
            self.process_btn.setFixedHeight(26)  # 버튼 높이 축소
            self.process_btn.clicked.connect(self._on_process_clicked)
            self.process_btn.setEnabled(False)

            self.threshold_slider = CompactThresholdSlider(45)
            self.threshold_slider.setFixedHeight(26)  # 슬라이더 높이 축소
            self.threshold_slider.threshold_changed.connect(self.threshold_changed.emit)

            control_layout.addStretch()
            control_layout.addWidget(self.process_btn)
            control_layout.addWidget(self.threshold_slider)
            control_layout.addStretch()

            layout.addLayout(control_layout)

        # 이미지 라벨
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # ✅ 고정 높이로 설정해 너무 커지지 않도록 하면서 충분히 크게 유지
        self.image_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.image_label.setMaximumHeight(200)  # 이미지 높이 조정 포인트 (220~260 권장)

        if self.enable_click_upload:
            self.image_label.setStyleSheet("""
                QLabel {
                    background: #FFFFFF;
                    border: 2px dashed #4A90E2;
                    border-radius: 12px;
                    color: #4A90E2;
                    font-size: 13px;
                    font-weight: 600;
                }
                QLabel:hover {
                    background: #F0F8FF;
                    border-color: #357ABD;
                    color: #357ABD;
                }
            """)
        else:
            self.image_label.setStyleSheet("""
                QLabel {
                    background: #FFFFFF;
                    border: 2px dashed #DDD;
                    border-radius: 12px;
                    color: #666;
                    font-size: 14px;
                }
            """)

        layout.addWidget(self.image_label)  # ✅ stretch=1 제거 → 아래 버튼이 밀리지 않음

        # 방향 버튼 (맨 아래 고정)
        if self.show_orientation_buttons:
            orientation_layout = QHBoxLayout()
            orientation_layout.setSpacing(2)

            self.portrait_btn = OrientationButton("📱 세로", "portrait")
            self.landscape_btn = OrientationButton("📺 가로", "landscape")
            self.portrait_btn.setChecked(True)

            self.orientation_group = QButtonGroup()
            self.orientation_group.addButton(self.portrait_btn, 0)
            self.orientation_group.addButton(self.landscape_btn, 1)

            orientation_layout.addStretch()
            orientation_layout.addWidget(self.portrait_btn)
            orientation_layout.addWidget(self.landscape_btn)
            orientation_layout.addStretch()

            self.portrait_btn.setEnabled(False)
            self.landscape_btn.setEnabled(False)

            self.portrait_btn.toggled.connect(self._on_orientation_changed)
            self.landscape_btn.toggled.connect(self._on_orientation_changed)

            layout.addLayout(orientation_layout)



    def _on_orientation_changed(self):
        """방향 변경 처리"""
        if self.portrait_btn.isChecked():
            new_orientation = "portrait"
        else:
            new_orientation = "landscape"
        
        if new_orientation != self.current_orientation:
            old_orientation = self.current_orientation
            self.current_orientation = new_orientation
            
            # 방향 변경 시그널 발송
            self.orientation_changed.emit(new_orientation)
            print(f"[DEBUG] 출력 방향 변경: {old_orientation} → {new_orientation}")
    
    def _on_process_clicked(self):
        """배경제거 버튼 클릭"""
        if hasattr(self, 'threshold_slider'):
            threshold = self.threshold_slider.get_value()
            self.process_requested.emit(threshold)
        else:
            self.process_requested.emit(200)
    
    def get_threshold_value(self):
        """현재 임계값 반환"""
        if hasattr(self, 'threshold_slider'):
            return self.threshold_slider.get_value()
        return 200
    
    def set_threshold_value(self, value):
        """임계값 설정"""
        if hasattr(self, 'threshold_slider'):
            self.threshold_slider.set_value(value)
    
    def get_current_orientation(self):
        """현재 출력 방향 반환"""
        return self.current_orientation
    
    def set_orientation(self, orientation):
        """출력 방향 설정"""
        if orientation == "portrait":
            self.portrait_btn.setChecked(True)
        else:
            self.landscape_btn.setChecked(True)
    
    def _set_placeholder_text(self):
        """플레이스홀더 텍스트 설정"""
        if self.enable_click_upload:
            placeholder = f"{self.title}\n\n클릭하여 이미지 선택" if self.title else "클릭하여 이미지 선택"
        else:
            placeholder = f"{self.title}\n\n이미지를 불러오세요" if self.title else "이미지를 불러오세요"
        self.image_label.setText(placeholder)
    
    def enable_click_upload_mode(self, enabled: bool = True):
        """클릭 업로드 모드 활성화/비활성화"""
        self.enable_click_upload = enabled
        self._setup_ui()
        self._set_placeholder_text()
    
    def mousePressEvent(self, event):
        """마우스 클릭 이벤트"""
        if event.button() == Qt.MouseButton.LeftButton:
            if self.enable_click_upload:
                self._open_file_dialog()
            self.image_clicked.emit()
        super().mousePressEvent(event)
    
    def _open_file_dialog(self):
        """파일 선택 다이얼로그 열기"""
        from config import config
        
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            f"{self.title} 이미지 선택",
            "",
            config.get_image_filter()
        )
        
        if file_path:
            self.set_image(file_path)
            self.file_uploaded.emit(file_path)
    
    def _safe_imread_unicode(self, image_path: str) -> np.ndarray:
        """한글 경로를 지원하는 안전한 이미지 읽기"""
        try:
            if not os.path.exists(image_path):
                print(f"[DEBUG] 파일이 존재하지 않음: {image_path}")
                return None
            
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
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
    
    def clear_image(self):
        """이미지 클리어"""
        print("[DEBUG] 이미지 클리어")
        self.original_pixmap = None
        self.current_image_array = None
        self.original_image_array = None
        self.image_path = None
        self.image_label.clear()
        self._set_placeholder_text()
        
        # 버튼들 비활성화 - 조건부 처리
        if self.show_orientation_buttons:
            self.portrait_btn.setEnabled(False)
            self.landscape_btn.setEnabled(False)
        if self.enable_process_button:
            self.process_btn.setEnabled(False)

    def set_image(self, image_path_or_array):
        """이미지 설정 (파일 경로 또는 numpy 배열)"""
        try:
            print(f"[DEBUG] set_image 호출됨: {type(image_path_or_array)}")
            
            if isinstance(image_path_or_array, str):
                print(f"[DEBUG] 파일 경로로 이미지 로드: {image_path_or_array}")
                self.image_path = image_path_or_array
                
                image_array = self._safe_imread_unicode(image_path_or_array)
                if image_array is None:
                    print(f"[DEBUG] 이미지 읽기 실패: {image_path_or_array}")
                    self._set_placeholder_text()
                    return
                
                self.original_image_array = image_array.copy()
                self.current_image_array = image_array.copy()
                
            elif isinstance(image_path_or_array, np.ndarray):
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
            
            # 이미지가 로드되면 버튼들 활성화 - 조건부 처리
            if self.show_orientation_buttons:
                self.portrait_btn.setEnabled(True)
                self.landscape_btn.setEnabled(True)
            
            if self.enable_process_button:
                self.process_btn.setEnabled(True)
            
            # 디스플레이 업데이트
            self.update_display()
            QTimer.singleShot(100, self.update_display)
            
            print("[DEBUG] 이미지 설정 완료!")
            
        except Exception as e:
            print(f"[DEBUG] 이미지 로드 오류: {e}")
            import traceback
            traceback.print_exc()
            self._set_placeholder_text()
            
            # 오류 발생 시 버튼들 비활성화 - 조건부 처리
            if self.show_orientation_buttons:
                self.portrait_btn.setEnabled(False)
                self.landscape_btn.setEnabled(False)
            if self.enable_process_button:
                self.process_btn.setEnabled(False)
                
    def _numpy_to_pixmap(self, array):
        """numpy 배열을 QPixmap으로 변환"""
        try:
            if array is None:
                print("[DEBUG] array가 None임")
                return QPixmap()
            
            if not array.flags.c_contiguous:
                array = np.ascontiguousarray(array)
            
            print(f"[DEBUG] numpy to pixmap: {array.shape}, dtype: {array.dtype}")
            
            if len(array.shape) == 3:
                height, width, channel = array.shape
                if channel == 3:
                    rgb_array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
                    bytes_per_line = 3 * width
                    q_image = QImage(
                        rgb_array.data, width, height, 
                        bytes_per_line, QImage.Format.Format_RGB888
                    )
                elif channel == 4:
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
            return pixmap
            
        except Exception as e:
            print(f"[DEBUG] numpy to pixmap 변환 오류: {e}")
            return QPixmap()
    
    def get_current_image_array(self):
        """현재 표시되고 있는 이미지를 numpy 배열로 반환 (원본 그대로)"""
        return self.current_image_array.copy() if self.current_image_array is not None else None
    
    def set_process_enabled(self, enabled: bool):
        """배경제거 버튼 활성화/비활성화"""
        if self.enable_process_button and hasattr(self, 'process_btn'):
            self.process_btn.setEnabled(enabled)
    
    def update_display(self):
        """디스플레이 업데이트 (크기에 맞게 조정)"""
        if self.original_pixmap is None or self.original_pixmap.isNull():
            return
            
        try:
            widget_size = self.size()
            label_size = self.image_label.size()
            
            if label_size.width() <= 0 or label_size.height() <= 0:
                QTimer.singleShot(50, self.update_display)
                return
            
            available_height = label_size.height() - 10
            available_width = label_size.width() - 10
            
            if available_height <= 0 or available_width <= 0:
                return
            
            scaled_pixmap = self.original_pixmap.scaled(
                available_width, 
                available_height,
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            )
            
            if not scaled_pixmap.isNull():
                self.image_label.setPixmap(scaled_pixmap)
                
        except Exception as e:
            print(f"[DEBUG] update_display 오류: {e}")
    
    def resizeEvent(self, event):
        """리사이즈 이벤트 처리"""
        super().resizeEvent(event)
        if self.original_pixmap:
            QTimer.singleShot(10, self.update_display)
    
    def showEvent(self, event):
        """위젯이 표시될 때 이벤트"""
        super().showEvent(event)
        if self.original_pixmap:
            QTimer.singleShot(50, self.update_display)


class UnifiedMaskViewer(QWidget):
    """통합 마스킹 미리보기 뷰어 - 자동/수동 마스킹 중 최신것만 표시, 카드 방향 지원"""
    
    def __init__(self, title=""):
        super().__init__()
        self.title = title
        self.current_mask_type = None  # "auto" 또는 "manual"
        self.auto_mask_array = None
        self.manual_mask_array = None
        self.original_pixmap = None
        self.card_orientation = "portrait"  # 기본값
        
        self.setMinimumSize(260, 320)
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
                border: 2px dashed #28A745;
                border-radius: 12px;
                color: #666;
                font-size: 14px;
            }
        """)
        
        # 마스킹 타입 표시 라벨
        self.type_label = QLabel()
        self.type_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.type_label.setFixedHeight(25)
        self.type_label.setStyleSheet("""
            QLabel {
                background: transparent;
                color: #6C757D;
                font-size: 11px;
                font-weight: 600;
                padding: 2px;
            }
        """)
        
        layout.addWidget(self.image_label)
        layout.addWidget(self.type_label)
    
    def _set_placeholder_text(self):
        """플레이스홀더 텍스트 설정"""
        placeholder = f"{self.title}\n\n배경제거 또는 수동 업로드 필요" if self.title else "배경제거 또는 수동 업로드 필요"
        self.image_label.setText(placeholder)
        self.type_label.setText("")
    
    def set_card_orientation(self, orientation: str):
        """카드 방향 설정 - 미리보기 업데이트"""
        self.card_orientation = orientation
        self._update_display()
        self._update_border_style()
    
    def _update_border_style(self):
        """카드 방향에 따라 테두리 색상 변경"""
        if self.card_orientation == "portrait":
            border_color = "#28A745"  # 초록색 (세로)
        else:
            border_color = "#FF6B35"  # 주황색 (가로)
        
        self.image_label.setStyleSheet(f"""
            QLabel {{
                background: #FFFFFF;
                border: 2px dashed {border_color};
                border-radius: 12px;
                color: #666;
                font-size: 14px;
            }}
        """)
    
    def _get_display_size(self):
        """카드 방향에 따른 표시 크기 계산"""
        widget_width = 280
        widget_height = 200
        
        if self.card_orientation == "portrait":
            # 세로형: 53.98 × 85.6 비율
            aspect_ratio = 53.98 / 85.6  # 약 0.63
            display_height = min(widget_height - 20, 160)
            display_width = int(display_height * aspect_ratio)
        else:
            # 가로형: 85.6 × 53.98 비율
            aspect_ratio = 85.6 / 53.98  # 약 1.59
            display_width = min(widget_width - 20, 200)
            display_height = int(display_width / aspect_ratio)
        
        return display_width, display_height
    
    def set_auto_mask(self, mask_array):
        """자동 배경제거 마스크 설정"""
        self.auto_mask_array = mask_array.copy() if mask_array is not None else None
        self.current_mask_type = "auto"
        self._update_display()
        self.type_label.setText("🤖 자동 마스킹")
        self.type_label.setStyleSheet("""
            QLabel {
                background: rgba(74, 144, 226, 0.1);
                color: #4A90E2;
                font-size: 11px;
                font-weight: 600;
                padding: 4px 8px;
                border-radius: 4px;
            }
        """)
    
    def set_manual_mask(self, mask_array):
        """수동 마스킹 설정"""
        self.manual_mask_array = mask_array.copy() if mask_array is not None else None
        self.current_mask_type = "manual"
        self._update_display()
        self.type_label.setText("✋ 수동 마스킹")
        self.type_label.setStyleSheet("""
            QLabel {
                background: rgba(40, 167, 69, 0.1);
                color: #28A745;
                font-size: 11px;
                font-weight: 600;
                padding: 4px 8px;
                border-radius: 4px;
            }
        """)
    
    def get_current_mask(self):
        """현재 활성화된 마스크 반환"""
        if self.current_mask_type == "manual" and self.manual_mask_array is not None:
            return self.manual_mask_array.copy()
        elif self.current_mask_type == "auto" and self.auto_mask_array is not None:
            return self.auto_mask_array.copy()
        return None
    
    def get_mask_type(self):
        """현재 마스크 타입 반환"""
        return self.current_mask_type
    
    def _update_display(self):
        """디스플레이 업데이트 - 내부용 메서드"""
        current_mask = self.get_current_mask()
        if current_mask is not None:
            pixmap = self._numpy_to_pixmap(current_mask)
            if not pixmap.isNull():
                self.original_pixmap = pixmap
                self.update_display()
        else:
            self._set_placeholder_text()
    
    def _numpy_to_pixmap(self, array):
        """numpy 배열을 QPixmap으로 변환"""
        try:
            if array is None:
                return QPixmap()
            
            if not array.flags.c_contiguous:
                array = np.ascontiguousarray(array)
            
            if len(array.shape) == 3:
                height, width, channel = array.shape
                if channel == 3:
                    rgb_array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
                    bytes_per_line = 3 * width
                    q_image = QImage(
                        rgb_array.data, width, height, 
                        bytes_per_line, QImage.Format.Format_RGB888
                    )
                else:
                    return QPixmap()
            elif len(array.shape) == 2:
                height, width = array.shape
                bytes_per_line = width
                q_image = QImage(
                    array.data, width, height, 
                    bytes_per_line, QImage.Format.Format_Grayscale8
                )
            else:
                return QPixmap()
            
            if q_image.isNull():
                return QPixmap()
            
            pixmap = QPixmap.fromImage(q_image)
            return pixmap
            
        except Exception as e:
            print(f"[DEBUG] numpy to pixmap 변환 오류: {e}")
            return QPixmap()
    
    def update_display(self):
        """카드 방향을 고려한 디스플레이 업데이트 (public 메서드)"""
        if self.original_pixmap is None or self.original_pixmap.isNull():
            return
            
        try:
            display_width, display_height = self._get_display_size()
            
            scaled_pixmap = self.original_pixmap.scaled(
                display_width, 
                display_height,
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            )
            
            if not scaled_pixmap.isNull():
                self.image_label.setPixmap(scaled_pixmap)
                
        except Exception as e:
            print(f"[DEBUG] update_display 오류: {e}")
    
    def clear_mask(self):
        """마스크 클리어"""
        self.auto_mask_array = None
        self.manual_mask_array = None
        self.current_mask_type = None
        self.original_pixmap = None
        self.image_label.clear()
        self._set_placeholder_text()
    
    def resizeEvent(self, event):
        """리사이즈 이벤트 처리"""
        super().resizeEvent(event)
        if self.original_pixmap:
            QTimer.singleShot(10, self.update_display)
    
    def showEvent(self, event):
        """위젯이 표시될 때 이벤트"""
        super().showEvent(event)
        if self.original_pixmap:
            QTimer.singleShot(50, self.update_display)