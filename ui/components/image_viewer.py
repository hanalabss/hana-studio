import numpy as np
from PySide6.QtWidgets import (
    QLabel, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QFileDialog,
    QSlider, QSizePolicy, QButtonGroup, QRadioButton, QSpacerItem
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QPixmap, QImage, QFont
import cv2
import os

# ✨ 통일된 크기 상수 정의
UNIFIED_VIEWER_WIDTH = 380
UNIFIED_VIEWER_HEIGHT = 480
UNIFIED_IMAGE_HEIGHT = 380
UNIFIED_IMAGE_DISPLAY_SIZE = 360
# 추가: 통일된 상단/하단 여백 상수
UNIFIED_TOP_MARGIN = 50
UNIFIED_BOTTOM_MARGIN = 30

"""
ui/components/image_viewer.py에서 수정해야 할 클래스들 - 큰 컨트롤 버전
"""

class OrientationButton(QRadioButton):
    """방향 선택 라디오 버튼 - 크기 증가"""
    def __init__(self, text, orientation="portrait"):
        super().__init__(text)
        self.orientation = orientation
        self.setFixedSize(85, 36)  # 70x30 → 85x36으로 증가
        self.setFont(QFont("Segoe UI", 10))  # 9 → 10으로 증가
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        # 스타일 적용
        self.setStyleSheet("""
            QRadioButton {
                background-color: #F8F9FA;
                border: 1px solid #DEE2E6;
                border-radius: 5px;
                color: #495057;
                font-weight: 500;
                padding: 8px 10px;
                spacing: 5px;
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
    """배경제거 버튼 컴포넌트 - 크기 증가"""
    def __init__(self, text):
        super().__init__(text)
        self.setFixedSize(110, 42)  # 90x36 → 110x42로 증가
        self.setFont(QFont("Segoe UI", 11, QFont.Weight.Medium))  # 10 → 11로 증가
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                           stop: 0 #4A90E2, stop: 1 #357ABD);
                color: white;
                border: none;
                border-radius: 5px;
                padding: 0 10px;
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
    """컴팩트한 임계값 슬라이더 - 크기 증가"""
    threshold_changed = Signal(int)

    def __init__(self, initial_value=45):
        super().__init__()
        self.setFixedSize(210, 42)  # 180x36 → 210x42로 증가
        self._setup_ui(initial_value)

    def _setup_ui(self, initial_value):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)  # 여백 증가
        layout.setSpacing(8)  # 간격 증가
        layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 255)
        self.slider.setValue(initial_value)
        self.slider.setFixedWidth(130)  # 120 → 140으로 증가
        self.slider.setFixedHeight(35)  # 30 → 35로 증가
        self.slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #DEE2E6;
                height: 8px;
                background: #F8F9FA;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #4A90E2;
                border: 1px solid #357ABD;
                width: 18px;
                margin: -7px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: #5BA0F2;
            }
        """)

        self.value_label = QLabel(str(initial_value))
        self.value_label.setFont(QFont("Segoe UI", 11))  # 10 → 11로 증가
        self.value_label.setStyleSheet("""
            QLabel {
                background-color: #F8F9FA;
                border: 1px solid #DEE2E6;
                border-radius: 5px;
                color: #495057;
                font-weight: 600;
                padding: 5px;
            }
        """)
        self.value_label.setFixedSize(60, 32)  # 50x28 → 60x32로 증가
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

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


# 상수도 조정 - 상단 여백 증가
UNIFIED_TOP_MARGIN = 50  # 50 → 55로 증가
UNIFIED_BOTTOM_MARGIN = 55

class ImageViewer(QWidget):
    """통일된 크기의 이미지 뷰어 위젯"""
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
        self.show_orientation_buttons = show_orientation_buttons
        self.original_pixmap = None
        self.current_image_array = None
        self.original_image_array = None
        self.image_path = None
        self.current_orientation = "portrait"  # 현재 출력 방향
        # ✨ 통일된 크기 사용
        self.setFixedSize(UNIFIED_VIEWER_WIDTH, UNIFIED_VIEWER_HEIGHT)
        self.setMinimumSize(UNIFIED_VIEWER_WIDTH, UNIFIED_VIEWER_HEIGHT)
        self._setup_ui()
        self._set_placeholder_text()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(0, 0, 0, 0)

        # 상단 배경제거 컨트롤 영역 (고정 높이 유지)
        if self.enable_process_button:
            control_layout = QHBoxLayout()
            control_layout.setSpacing(8)
            # 배경제거 버튼과 슬라이더
            self.process_btn = ProcessButton("배경제거")
            self.process_btn.clicked.connect(self._on_process_clicked)
            self.process_btn.setEnabled(False)
            self.threshold_slider = CompactThresholdSlider(45)
            self.threshold_slider.threshold_changed.connect(self.threshold_changed.emit)
            # 수평 중앙 정렬 배치
            control_layout.addStretch()
            control_layout.addWidget(self.process_btn)
            control_layout.addWidget(self.threshold_slider)
            control_layout.addStretch()
            # 컨테이너에 넣어 상단 고정 높이 확보 - 투명 배경
            control_container = QWidget()
            control_container.setFixedHeight(UNIFIED_TOP_MARGIN)
            control_container.setStyleSheet("background: transparent; border: none;")
            control_container.setLayout(control_layout)
            layout.addWidget(control_container)
        else:
            # 상단 빈 공간 확보 - 투명 배경
            spacer_top = QWidget()
            spacer_top.setFixedHeight(UNIFIED_TOP_MARGIN)
            spacer_top.setStyleSheet("background: transparent; border: none;")
            layout.addWidget(spacer_top)

        # 이미지 표시 라벨 (고정 높이)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.image_label.setFixedHeight(UNIFIED_IMAGE_HEIGHT)
        self.image_label.setMinimumHeight(UNIFIED_IMAGE_HEIGHT)
        if self.enable_click_upload:
            self.image_label.setStyleSheet("""
                QLabel {
                    background: #FFFFFF;
                    border: 2px dashed #4A90E2;
                    border-radius: 12px;
                    color: #4A90E2;
                    font-size: 14px;
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
                    font-size: 15px;
                }
            """)
        layout.addWidget(self.image_label)

        # 하단 방향 선택 영역 (고정 높이 유지)
        if self.show_orientation_buttons:
            self.portrait_btn = OrientationButton("📱 세로", "portrait")
            self.landscape_btn = OrientationButton("📺 가로", "landscape")
            self.portrait_btn.setChecked(True)
            self.orientation_group = QButtonGroup()
            self.orientation_group.addButton(self.portrait_btn, 0)
            self.orientation_group.addButton(self.landscape_btn, 1)
            # 가로/세로 버튼 배치 (수평 레이아웃)
            orientation_layout = QHBoxLayout()
            orientation_layout.setSpacing(8)
            orientation_layout.addStretch()
            orientation_layout.addWidget(self.portrait_btn)
            orientation_layout.addWidget(self.landscape_btn)
            orientation_layout.addStretch()
            # 컨테이너를 만들어 수직 중앙 정렬 - 투명 배경
            orientation_container = QWidget()
            orientation_container.setFixedHeight(UNIFIED_BOTTOM_MARGIN)
            orientation_container.setStyleSheet("background: transparent; border: none;")
            orientation_vlayout = QVBoxLayout(orientation_container)
            orientation_vlayout.setContentsMargins(0, 18, 0, 0)
            orientation_vlayout.addStretch()
            orientation_vlayout.addLayout(orientation_layout)
            orientation_vlayout.addStretch()
            layout.addWidget(orientation_container)
            # 시그널 연결 및 초기 활성화 설정
            self.portrait_btn.toggled.connect(self._on_orientation_changed)
            self.landscape_btn.toggled.connect(self._on_orientation_changed)
            self.portrait_btn.setEnabled(False)
            self.landscape_btn.setEnabled(False)
        else:
            spacer_bottom = QWidget()
            spacer_bottom.setFixedHeight(UNIFIED_BOTTOM_MARGIN)
            spacer_bottom.setStyleSheet("background: transparent; border: none;")
            layout.addWidget(spacer_bottom)

        # 남은 공간을 위로 밀어 올림
        layout.addStretch()
        
    def _on_orientation_changed(self):
        """방향 변경 처리"""
        new_orientation = "portrait" if self.portrait_btn.isChecked() else "landscape"
        if new_orientation != self.current_orientation:
            old_orientation = self.current_orientation
            self.current_orientation = new_orientation
            # 방향 변경 시그널 발송
            self.orientation_changed.emit(new_orientation)
            print(f"[DEBUG] 출력 방향 변경: {old_orientation} → {new_orientation}")
    
    def _on_process_clicked(self):
        """배경제거 버튼 클릭"""
        # 현재 설정된 임계값과 함께 신호 방출
        threshold = self.threshold_slider.get_value() if hasattr(self, 'threshold_slider') else 200
        self.process_requested.emit(threshold)
    
    def get_threshold_value(self):
        """현재 임계값 반환"""
        return self.threshold_slider.get_value() if hasattr(self, 'threshold_slider') else 200
    
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
        """마우스 클릭 이벤트 처리"""
        if event.button() == Qt.MouseButton.LeftButton:
            if self.enable_click_upload:
                self._open_file_dialog()
            self.image_clicked.emit()
        super().mousePressEvent(event)
    
    def _open_file_dialog(self):
        """파일 선택 다이얼로그 (PyInstaller 호환)"""
        from config import config
        import sys
        # PyInstaller 호환 초기 디렉토리 설정
        initial_dir = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.getcwd()
        print(f"[DEBUG] 초기 디렉토리: {initial_dir}")
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            f"{self.title} 이미지 선택",
            initial_dir,
            config.get_image_filter()
        )
        if file_path:
            file_path = os.path.abspath(file_path)
            print(f"[DEBUG] 선택된 파일: {file_path}, 존재 여부: {os.path.exists(file_path)}")
            if os.path.exists(file_path):
                self.set_image(file_path)
                self.file_uploaded.emit(file_path)
            else:
                print(f"[ERROR] 파일을 찾을 수 없음: {file_path}")
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.warning(self, "오류", f"선택한 파일을 찾을 수 없습니다:\n{file_path}")

    def _safe_imread_unicode(self, image_path: str) -> np.ndarray:
        """한글 경로 지원 안전 이미지 읽기"""
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
        # 상태 초기화
        self.original_pixmap = None
        self.current_image_array = None
        self.original_image_array = None
        self.image_path = None
        self.image_label.clear()
        self._set_placeholder_text()
        # 관련 버튼 비활성화
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
            # QPixmap 생성 및 설정
            pixmap = self._numpy_to_pixmap(self.current_image_array)
            if pixmap.isNull():
                print("[DEBUG] QPixmap 생성 실패")
                self._set_placeholder_text()
                return
            self.original_pixmap = pixmap
            # 이미지 로드 성공 -> 관련 버튼 활성화
            if self.show_orientation_buttons:
                self.portrait_btn.setEnabled(True)
                self.landscape_btn.setEnabled(True)
            if self.enable_process_button:
                self.process_btn.setEnabled(True)
            # 화면 갱신
            self.update_display()
            QTimer.singleShot(100, self.update_display)
            print("[DEBUG] 이미지 설정 완료!")
        except Exception as e:
            print(f"[DEBUG] 이미지 로드 오류: {e}")
            import traceback
            traceback.print_exc()
            self._set_placeholder_text()
            # 오류 발생 -> 버튼 비활성화
            if self.show_orientation_buttons:
                self.portrait_btn.setEnabled(False)
                self.landscape_btn.setEnabled(False)
            if self.enable_process_button:
                self.process_btn.setEnabled(False)
                
    def _numpy_to_pixmap(self, array):
        """numpy 배열을 QPixmap으로 변환"""
        try:
            if array is None:
                print("[DEBUG] 변환 대상 배열이 None임")
                return QPixmap()
            if not array.flags.c_contiguous:
                array = np.ascontiguousarray(array)
            print(f"[DEBUG] numpy to pixmap: shape={array.shape}, dtype={array.dtype}")
            if len(array.shape) == 3:
                height, width, channel = array.shape
                if channel == 3:
                    rgb_array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
                    bytes_per_line = 3 * width
                    q_image = QImage(rgb_array.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
                elif channel == 4:
                    rgba_array = cv2.cvtColor(array, cv2.COLOR_BGRA2RGBA)
                    bytes_per_line = 4 * width
                    q_image = QImage(rgba_array.data, width, height, bytes_per_line, QImage.Format.Format_RGBA8888)
                else:
                    print(f"[DEBUG] 지원하지 않는 채널 수: {channel}")
                    return QPixmap()
            elif len(array.shape) == 2:
                height, width = array.shape
                bytes_per_line = width
                q_image = QImage(array.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8)
            else:
                print(f"[DEBUG] 지원하지 않는 배열 차원: {array.shape}")
                return QPixmap()
            if q_image.isNull():
                print("[DEBUG] QImage 생성 실패")
                return QPixmap()
            return QPixmap.fromImage(q_image)
        except Exception as e:
            print(f"[DEBUG] numpy to pixmap 변환 오류: {e}")
            return QPixmap()
    
    def get_current_image_array(self):
        """현재 표시되고 있는 이미지 배열 반환"""
        return self.current_image_array.copy() if self.current_image_array is not None else None
    
    def set_process_enabled(self, enabled: bool):
        """배경제거 버튼 활성화/비활성화"""
        if self.enable_process_button and hasattr(self, 'process_btn'):
            self.process_btn.setEnabled(enabled)
    
    def update_display(self):
        """디스플레이 업데이트 (통일된 크기로 이미지 표시)"""
        if self.original_pixmap is None or self.original_pixmap.isNull():
            return
        try:
            # 통일된 크기로 Pixmap 스케일링
            scaled_pixmap = self.original_pixmap.scaled(
                UNIFIED_IMAGE_DISPLAY_SIZE, 
                UNIFIED_IMAGE_DISPLAY_SIZE,
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
        """위젯 표시 이벤트 처리"""
        super().showEvent(event)
        if self.original_pixmap:
            QTimer.singleShot(50, self.update_display)

class UnifiedMaskViewer(QWidget):
    """통일된 크기의 마스킹 미리보기 뷰어"""
    def __init__(self, title=""):
        super().__init__()
        self.title = title
        self.current_mask_type = None  # "auto" 또는 "manual"
        self.auto_mask_array = None
        self.manual_mask_array = None
        self.original_pixmap = None
        self.card_orientation = "portrait"  # 기본 출력 방향
        # ✨ 통일된 크기 사용
        self.setFixedSize(UNIFIED_VIEWER_WIDTH, UNIFIED_VIEWER_HEIGHT)
        self.setMinimumSize(UNIFIED_VIEWER_WIDTH, UNIFIED_VIEWER_HEIGHT)
        self._setup_ui()
        self._set_placeholder_text()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        
        # 상단 빈 공간 확보 (원본 뷰어와 정렬) - 투명 배경
        spacer_top = QWidget()
        spacer_top.setFixedHeight(UNIFIED_TOP_MARGIN)
        spacer_top.setStyleSheet("background: transparent; border: none;")
        layout.addWidget(spacer_top)
        
        # 이미지 표시 라벨 (고정 높이)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setFixedHeight(UNIFIED_IMAGE_HEIGHT)
        self.image_label.setMinimumHeight(UNIFIED_IMAGE_HEIGHT)
        self.image_label.setStyleSheet("""
            QLabel {
                background: #FFFFFF;
                border: 2px dashed #28A745;
                border-radius: 12px;
                color: #666;
                font-size: 15px;
            }
        """)
        layout.addWidget(self.image_label)
        
        # 하단 마스킹 타입 라벨 (고정 높이를 작게 설정) - 투명 배경
        self.type_label = QLabel()
        self.type_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.type_label.setFixedHeight(32)  # UNIFIED_BOTTOM_MARGIN 대신 작은 고정값 사용
        self.type_label.setStyleSheet("""
            QLabel {
                background: transparent;
                border: none;
                color: #6C757D;
                font-size: 12px;
                font-weight: 600;
                padding: 4px;
            }
        """)
        layout.addWidget(self.type_label)
        
        # 남은 공간을 아래로 밀어내기
        layout.addStretch()

    def _set_placeholder_text(self):
        """플레이스홀더 텍스트 설정"""
        placeholder = f"{self.title}\n\n배경제거 또는 수동 업로드 필요" if self.title else "배경제거 또는 수동 업로드 필요"
        self.image_label.setText(placeholder)
        self.type_label.setText("")
    
    def set_card_orientation(self, orientation: str):
        """카드 방향 설정 후 미리보기 업데이트"""
        self.card_orientation = orientation
        self._update_border_style()
        self._update_display()
    
    def _update_border_style(self):
        """카드 방향에 따른 테두리 색상 업데이트"""
        if self.card_orientation == "portrait":
            border_color = "#28A745"  # 세로 모드 색상 (초록)
        else:
            border_color = "#FF6B35"  # 가로 모드 색상 (주황)
        self.image_label.setStyleSheet(f"""
            QLabel {{
                background: #FFFFFF;
                border: 2px dashed {border_color};
                border-radius: 12px;
                color: #666;
                font-size: 15px;
            }}
        """)
    
    def set_auto_mask(self, mask_array):
        """자동 마스킹 결과 설정"""
        self.auto_mask_array = mask_array.copy() if mask_array is not None else None
        self.current_mask_type = "auto"
        self._update_display()
        self.type_label.setText("🤖 자동 마스킹")
        self.type_label.setStyleSheet("""
            QLabel {
                background: rgba(74, 144, 226, 0.1);
                color: #4A90E2;
                font-size: 12px;
                font-weight: 600;
                padding: 4px 8px;
                border-radius: 4px;
                margin: 2px;
            }
        """)
    
    def set_manual_mask(self, mask_array):
        """수동 마스킹 결과 설정"""
        self.manual_mask_array = mask_array.copy() if mask_array is not None else None
        self.current_mask_type = "manual"
        self._update_display()
        self.type_label.setText("✋ 수동 마스킹")
        self.type_label.setStyleSheet("""
            QLabel {
                background: rgba(40, 167, 69, 0.1);
                color: #28A745;
                font-size: 12px;
                font-weight: 600;
                padding: 4px 8px;
                border-radius: 4px;
                margin: 2px;
            }
        """)
    
    def get_current_mask(self):
        """현재 활성화된 마스크 배열 반환"""
        if self.current_mask_type == "manual" and self.manual_mask_array is not None:
            return self.manual_mask_array.copy()
        elif self.current_mask_type == "auto" and self.auto_mask_array is not None:
            return self.auto_mask_array.copy()
        return None
    
    def get_mask_type(self):
        """현재 마스크 타입 반환"""
        return self.current_mask_type
    
    def _update_display(self):
        """내부용 디스플레이 업데이트 (마스크 적용)"""
        current_mask = self.get_current_mask()
        if current_mask is not None:
            pixmap = self._numpy_to_pixmap(current_mask)
            if not pixmap.isNull():
                self.original_pixmap = pixmap
                self.update_display()
        else:
            self._set_placeholder_text()
    
    def _numpy_to_pixmap(self, array):
        """numpy 배열 -> QPixmap 변환 (마스크 용)"""
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
                    q_image = QImage(rgb_array.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
                else:
                    return QPixmap()
            elif len(array.shape) == 2:
                height, width = array.shape
                bytes_per_line = width
                q_image = QImage(array.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8)
            else:
                return QPixmap()
            if q_image.isNull():
                return QPixmap()
            return QPixmap.fromImage(q_image)
        except Exception as e:
            print(f"[DEBUG] numpy to pixmap 변환 오류: {e}")
            return QPixmap()
    
    def update_display(self):
        """마스킹 미리보기 디스플레이 업데이트 (통일된 크기로 표시)"""
        if self.original_pixmap is None or self.original_pixmap.isNull():
            return
        try:
            # 통일된 크기로 Pixmap 스케일링
            scaled_pixmap = self.original_pixmap.scaled(
                UNIFIED_IMAGE_DISPLAY_SIZE, 
                UNIFIED_IMAGE_DISPLAY_SIZE,
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            )
            if not scaled_pixmap.isNull():
                self.image_label.setPixmap(scaled_pixmap)
        except Exception as e:
            print(f"[DEBUG] update_display 오류: {e}")
    
    def clear_mask(self):
        """마스크 미리보기 초기화"""
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
        """위젯 표시 이벤트 처리"""
        super().showEvent(event)
        if self.original_pixmap:
            QTimer.singleShot(50, self.update_display)