import sys
import os
import threading
import io
from pathlib import Path
from datetime import datetime

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                              QHBoxLayout, QGridLayout, QPushButton, QLabel, 
                              QFileDialog, QFrame, QSplitter,
                              QProgressBar, QTextEdit, QGroupBox,
                              QMessageBox,QRadioButton,QButtonGroup, QSlider)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap, QFont, QColor, QPalette, QImage

import cv2
import numpy as np
from PIL import Image
from rembg import remove, new_session

# 프로젝트 모듈들
try:
    from config import config, AppConstants
except ImportError:
    try:
        # config 인스턴스가 없는 경우 Config 클래스에서 직접 생성
        from config import Config, AppConstants
        config = Config()
        print("✅ Config 클래스에서 인스턴스 생성 완료")
    except ImportError:
        # config.py에서 인스턴스를 찾을 수 없는 경우 기본 설정 사용
        print("⚠️ config.py에서 설정을 로드할 수 없습니다. 기본 설정을 사용합니다.")
        
        class DefaultConfig:
            def get(self, key, default=None):
                defaults = {
                    'alpha_threshold': 45,
                    'output_quality': 95,
                    'ai_model': 'isnet-general-use',
                    'directories.temp': 'temp',
                    'directories.output': 'output',
                    'printer.card_width': 53.98,
                    'printer.card_height': 85.6,
                    'printer.dll_path': './libDSRetransfer600App.dll',
                    'window_geometry': {'x': 100, 'y': 100, 'width': 1800, 'height': 1000},
                    'improve_mask_quality': True,
                    'mask_kernel_size': 3,
                    'mask_iterations': 1
                }
                keys = key.split('.')
                value = defaults
                try:
                    for k in keys:
                        value = value[k]
                    return value
                except (KeyError, TypeError):
                    return default
            
            def set(self, key, value):
                print(f"설정 변경: {key} = {value}")
            
            def save_settings(self):
                pass
            
            def get_image_filter(self):
                return "이미지 파일 (*.jpg *.jpeg *.png *.bmp *.tiff);;JPEG (*.jpg *.jpeg);;PNG (*.png);;모든 파일 (*.*)"
            
            def is_supported_image(self, file_path):
                from pathlib import Path
                ext = Path(file_path).suffix.lower()
                return ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
            
            def get_ai_model_info(self, model_name):
                models = {
                    'isnet-general-use': {'name': '범용 모델 (권장)'},
                    'u2net': {'name': 'U²-Net'},
                    'u2netp': {'name': 'U²-Net (Light)'},
                    'silueta': {'name': 'Silueta'}
                }
                return models.get(model_name)
        
        class DefaultAppConstants:
            APP_NAME = "Hana Studio"
            APP_VERSION = "1.0.0"
            APP_AUTHOR = "Hana Tech"
            APP_DESCRIPTION = "AI 기반 이미지 배경 제거 도구"
        
        config = DefaultConfig()
        AppConstants = DefaultAppConstants()

# 프린터 모듈 (선택적 import) - 워터마크 지원 버전으로 업데이트
try:
    from printer_integration import PrinterThread, find_printer_dll, test_printer_connection
    PRINTER_AVAILABLE = True
    print("✅ 워터마크 프린터 모듈 로드 성공")
except ImportError as e:
    PRINTER_AVAILABLE = False
    print(f"⚠️ 프린터 모듈 로드 실패: {e}")
    
    # 프린터 모듈이 없는 경우 더미 클래스 정의
    class DummyPrinterThread:
        def __init__(self, *args, **kwargs):
            pass
    
    def find_printer_dll():
        return None
    
    def test_printer_connection():
        return False
    
    PrinterThread = DummyPrinterThread


# ===== 새로 추가된 마스크 품질 개선 함수 =====
def improve_mask_quality(mask, kernel_size=3, iterations=1):
    """
    마스크 품질 개선을 위한 모폴로지 연산
    
    Args:
        mask: 원본 마스크 (numpy array)
        kernel_size: 커널 크기 (홀수)
        iterations: 반복 횟수
    
    Returns:
        개선된 마스크
    """
    # 노이즈 제거를 위한 Opening 연산
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # 작은 구멍 메우기
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    
    # 작은 노이즈 제거
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel, iterations=iterations)
    
    return mask_cleaned


class ModernButton(QPushButton):
    """현대적인 스타일의 버튼"""
    def __init__(self, text, icon_path=None, primary=False):
        super().__init__(text)
        self.primary = primary
        self.setFixedHeight(45)
        self.setFont(QFont("Segoe UI", 10, QFont.Weight.Medium))
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        
        if primary:
            self.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                               stop: 0 #4A90E2, stop: 1 #357ABD);
                    color: white;
                    border: none;
                    border-radius: 8px;
                    padding: 0 20px;
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
        else:
            self.setStyleSheet("""
                QPushButton {
                    background: #F8F9FA;
                    color: #495057;
                    border: 2px solid #E9ECEF;
                    border-radius: 8px;
                    padding: 0 20px;
                    font-weight: 500;
                }
                QPushButton:hover {
                    background: #E9ECEF;
                    border-color: #DEE2E6;
                }
                QPushButton:pressed {
                    background: #DEE2E6;
                    border-color: #CED4DA;
                }
                QPushButton:disabled {
                    background: #F8F9FA;
                    color: #ADB5BD;
                    border-color: #E9ECEF;
                }
            """)


class ImageViewer(QLabel):
    """이미지 뷰어 위젯"""
    def __init__(self, title=""):
        super().__init__()
        self.title = title
        self.original_pixmap = None
        self.setMinimumSize(300, 200)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                background: #FFFFFF;
                border: 2px dashed #DDD;
                border-radius: 12px;
                color: #666;
                font-size: 14px;
            }
        """)
        self.setText(f"{title}\n\n이미지를 불러오세요" if title else "이미지를 불러오세요")
        
    def set_image(self, image_path_or_array):
        try:
            if isinstance(image_path_or_array, str):
                pixmap = QPixmap(image_path_or_array)
            elif isinstance(image_path_or_array, np.ndarray):
                # numpy array를 QPixmap으로 변환
                if len(image_path_or_array.shape) == 3:
                    height, width, channel = image_path_or_array.shape
                    bytes_per_line = 3 * width
                    q_image = QImage(image_path_or_array.data, width, height, 
                                   bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped()
                else:
                    height, width = image_path_or_array.shape
                    bytes_per_line = width
                    q_image = QImage(image_path_or_array.data, width, height, 
                                   bytes_per_line, QImage.Format.Format_Grayscale8)
                pixmap = QPixmap.fromImage(q_image)
            else:
                return
                
            self.original_pixmap = pixmap
            self.update_display()
            
        except Exception as e:
            print(f"이미지 로드 오류: {e}")
    
    def update_display(self):
        if self.original_pixmap:
            # 비율을 유지하면서 크기 조정
            scaled_pixmap = self.original_pixmap.scaled(
                self.size(), 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            )
            self.setPixmap(scaled_pixmap)
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.original_pixmap:
            self.update_display()


# ===== ProcessingThread 클래스 - 워터마크 마스크 생성 =====
class ProcessingThread(QThread):
    finished = Signal(np.ndarray)
    error = Signal(str)
    progress = Signal(str)
    
    def __init__(self, image_path, session):
        super().__init__()
        self.image_path = image_path
        self.session = session
        
    def run(self):
        try:
            self.progress.emit("AI 모델 로딩 중...")
            
            with open(self.image_path, 'rb') as f:
                input_data = f.read()
            
            self.progress.emit("배경 제거 처리 중...")
            result = remove(input_data, session=self.session)
            
            self.progress.emit("워터마크 마스크 생성 중...")
            img_rgba = Image.open(io.BytesIO(result)).convert("RGBA")
            alpha = np.array(img_rgba.split()[-1])
            
            # ===== 핵심 수정: 워터마크용 마스크 생성 =====
            alpha_threshold = config.get('alpha_threshold', 45)
            
            # 워터마크 마스크 생성: 객체 부분은 흰색(255), 배경은 검은색(0)
            mask = np.where(alpha > alpha_threshold, 255, 0).astype(np.uint8)
            
            # ===== 마스크 품질 개선 =====
            if config.get('improve_mask_quality', True):
                self.progress.emit("마스크 품질 개선 중...")
                kernel_size = config.get('mask_kernel_size', 3)
                iterations = config.get('mask_iterations', 1)
                mask = improve_mask_quality(mask, kernel_size, iterations)
            
            # RGB 채널로 변환
            mask_rgb = cv2.merge([mask, mask, mask])
            
            # ===== 마스크 통계 계산 및 로깅 =====
            object_pixels = np.sum(mask > 0)
            total_pixels = mask.size
            object_ratio = (object_pixels / total_pixels) * 100
            
            self.progress.emit(f"워터마크 처리 완료! (객체 영역: {object_ratio:.1f}%)")
            
            # 디버깅 정보 로그
            print(f"[DEBUG] 워터마크 마스크 생성 완료:")
            print(f"  - 이미지 크기: {mask.shape}")
            print(f"  - 객체 픽셀: {object_pixels:,}")
            print(f"  - 전체 픽셀: {total_pixels:,}")
            print(f"  - 객체 비율: {object_ratio:.2f}%")
            print(f"  - 알파 임계값: {alpha_threshold}")
            
            self.finished.emit(mask_rgb)
            
        except Exception as e:
            self.error.emit(str(e))


class HanaStudio(QMainWindow):
    """메인 애플리케이션 클래스 - 워터마크 지원"""
    def __init__(self):
        super().__init__()
        self.session = None
        self.current_image_path = None
        self.original_image = None
        self.mask_image = None
        self.composite_image = None
        self.saved_mask_path = None  # 프린터용 워터마크 파일 경로
        self.saved_color_path = None  # 프린터용 투명 배경 처리된 컬러 이미지 경로
        self.ai_result_path = None   # AI 배경 제거 결과 파일 경로 (미리보기용)
        self.ai_result_image = None  # AI 배경 제거 결과 이미지 (opencv 형태)
        
        # 프린터 관련 변수
        self.printer_available = PRINTER_AVAILABLE
        self.printer_dll_path = None
        self.print_mode = "normal"  # "normal" 또는 "watermark"
        
        # 임계값 슬라이더 참조
        self.threshold_slider = None
        self.threshold_label = None
        
        self.init_ui()
        self.init_ai_model()
        self.check_printer_availability()
        
    def init_ui(self):
        """UI 초기화"""
        self.setWindowTitle(f"{AppConstants.APP_NAME} - 워터마크 지원 {AppConstants.APP_DESCRIPTION}")
        
        # 설정에서 윈도우 크기 가져오기
        geometry = config.get('window_geometry')
        default_width = max(geometry.get('width', 1600), 1800)
        default_height = max(geometry.get('height', 900), 1000)
        
        self.setGeometry(
            geometry.get('x', 100), 
            geometry.get('y', 100), 
            default_width, 
            default_height
        )
        self.setMinimumSize(1600, 900)
        
        # 라이트 테마 스타일 설정
        self.setStyleSheet("""
            QMainWindow {
                background-color: #F8F9FA;
            }
            QWidget {
                background-color: #F8F9FA;
                color: #212529;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QSplitter {
                background-color: #F8F9FA;
            }
            QSplitter::handle {
                background-color: #DEE2E6;
            }
            QGroupBox {
                font-weight: 600;
                font-size: 12px;
                color: #495057;
                border: 2px solid #DEE2E6;
                border-radius: 10px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #FFFFFF;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 10px;
                background-color: #FFFFFF;
                color: #495057;
            }
            QProgressBar {
                border: none;
                border-radius: 6px;
                background-color: #E9ECEF;
                height: 12px;
            }
            QProgressBar::chunk {
                background-color: #4A90E2;
                border-radius: 6px;
            }
            QTextEdit {
                background-color: #FFFFFF;
                border: 2px solid #E9ECEF;
                border-radius: 8px;
                padding: 10px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 11px;
                color: #212529;
            }
            QLabel {
                background-color: transparent;
                color: #212529;
            }
            QSlider {
                background-color: transparent;
            }
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: #E9ECEF;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #4A90E2;
                border: 1px solid #357ABD;
                width: 18px;
                margin: -2px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: #5BA0F2;
            }
        """)
        
        # 중앙 위젯
        central_widget = QWidget()
        central_widget.setStyleSheet("background-color: #F8F9FA;")
        self.setCentralWidget(central_widget)
        
        # 메인 레이아웃
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # 헤더
        self.create_header(main_layout)
        
        # 메인 컨텐츠 (좌우 분할)
        content_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(content_splitter)
        
        # 좌측 패널 (컨트롤)
        self.create_left_panel(content_splitter)
        
        # 우측 패널 (이미지 뷰어)
        self.create_right_panel(content_splitter)
        
        # 하단 상태바
        self.create_status_bar(main_layout)
        
        # 분할 비율 설정
        content_splitter.setSizes([500, 1300])
        
    def create_header(self, parent_layout):
        """헤더 생성"""
        header_frame = QFrame()
        header_frame.setFixedHeight(80)
        header_frame.setStyleSheet("""
            QFrame {
                background-color: #FFFFFF;
                border: none;
                border-bottom: 3px solid #4A90E2;
                border-radius: 0;
            }
        """)
        
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(30, 15, 30, 15)
        
        # 로고/제목
        title_label = QLabel("🎨 Hana Studio")
        title_label.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        title_label.setStyleSheet("color: #2C3E50;")
        
        subtitle_label = QLabel("AI 기반 이미지 배경 제거 및 워터마크 카드 인쇄 도구")
        subtitle_label.setFont(QFont("Segoe UI", 11))
        subtitle_label.setStyleSheet("color: #7F8C8D; margin-top: 5px;")
        
        title_layout = QVBoxLayout()
        title_layout.addWidget(title_label)
        title_layout.addWidget(subtitle_label)
        title_layout.setSpacing(0)
        
        header_layout.addLayout(title_layout)
        header_layout.addStretch()
        
        parent_layout.addWidget(header_frame)
        
    def create_left_panel(self, splitter):
        """좌측 패널 생성 - 워터마크 관련 UI로 수정"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # 📁 파일 선택 그룹
        file_group = QGroupBox("📁 파일 선택")
        file_layout = QVBoxLayout(file_group)
        self.select_btn = ModernButton("이미지 선택", primary=True)
        self.select_btn.clicked.connect(self.select_image)
        self.file_label = QLabel("선택된 파일이 없습니다")
        file_layout.addWidget(self.select_btn)
        file_layout.addWidget(self.file_label)
        layout.addWidget(file_group)

        # ⚙️ 처리 옵션 그룹 - 임계값 슬라이더 추가
        option_group = QGroupBox("⚙️ 처리 옵션")
        option_layout = QVBoxLayout(option_group)
        
        # 임계값 조정 슬라이더
        threshold_layout = QHBoxLayout()
        current_threshold = config.get('alpha_threshold', 45)
        self.threshold_label = QLabel(f"알파 임계값: {current_threshold}")
        self.threshold_label.setStyleSheet("font-size: 10px; color: #6C757D;")
        
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(0, 255)
        self.threshold_slider.setValue(current_threshold)
        self.threshold_slider.valueChanged.connect(self.on_threshold_changed)
        
        threshold_layout.addWidget(self.threshold_label)
        threshold_layout.addWidget(self.threshold_slider)
        option_layout.addLayout(threshold_layout)
        
        # 임계값 설명 라벨
        threshold_help = QLabel("💡 값이 높을수록 더 많은 영역이 객체로 인식됩니다")
        threshold_help.setStyleSheet("font-size: 9px; color: #ADB5BD; margin-bottom: 10px;")
        threshold_help.setWordWrap(True)
        option_layout.addWidget(threshold_help)
        
        self.process_btn = ModernButton("워터마크 생성", primary=True)
        self.process_btn.clicked.connect(self.process_image)
        self.process_btn.setEnabled(False)
        self.export_btn = ModernButton("결과 저장")
        self.export_btn.clicked.connect(self.export_results)
        self.export_btn.setEnabled(False)
        option_layout.addWidget(self.process_btn)
        option_layout.addWidget(self.export_btn)
        layout.addWidget(option_group)

        # 📋 인쇄 모드 그룹 - 워터마크 모드로 수정
        mode_group = QGroupBox("📋 인쇄 모드")
        mode_layout = QVBoxLayout(mode_group)
        self.normal_radio = QRadioButton("🖼️ 일반 인쇄")
        self.watermark_radio = QRadioButton("💧 워터마크 인쇄")
        self.normal_radio.setChecked(True)
        self.normal_radio.toggled.connect(self.on_print_mode_changed)
        mode_layout.addWidget(self.normal_radio)
        mode_layout.addWidget(self.watermark_radio)
        
        # 모드 설명 라벨
        self.mode_description_label = QLabel("📖 원본 이미지만 인쇄합니다")
        self.mode_description_label.setStyleSheet("""
            color: #6C757D; font-size: 10px;
            padding: 4px 8px;
            background-color: #F8F9FA;
            border-left: 3px solid #4A90E2;
        """)
        self.mode_description_label.setWordWrap(True)
        mode_layout.addWidget(self.mode_description_label)
        layout.addWidget(mode_group)

        # 🖨️ 프린터 그룹 - 워터마크 프린터로 수정
        printer_group = QGroupBox("🖨 워터마크 프린터 연동")
        printer_layout = QVBoxLayout(printer_group)
        self.printer_status_label = QLabel("프린터 상태 확인 중...")
        self.printer_status_label.setStyleSheet("font-size: 10px; color: #6C757D;")
        self.printer_status_label.setWordWrap(True)
        printer_layout.addWidget(self.printer_status_label)

        self.test_printer_btn = ModernButton("프린터 연결 테스트")
        self.test_printer_btn.clicked.connect(self.test_printer_connection)
        printer_layout.addWidget(self.test_printer_btn)

        self.print_card_btn = ModernButton("카드 인쇄", primary=True)
        self.print_card_btn.clicked.connect(self.print_card)
        self.print_card_btn.setEnabled(False)
        printer_layout.addWidget(self.print_card_btn)
        layout.addWidget(printer_group)

        # 📊 진행 상황
        progress_group = QGroupBox("📊 진행 상황")
        progress_layout = QVBoxLayout(progress_group)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_bar)
        self.status_label = QLabel("대기 중...")
        progress_layout.addWidget(self.status_label)
        layout.addWidget(progress_group)

        # 📝 로그
        log_group = QGroupBox("📝 처리 로그")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        layout.addWidget(log_group)

        layout.addStretch()
        splitter.addWidget(panel)
    
    def create_right_panel(self, parent_splitter):
        """우측 패널 생성 - 워터마크 뷰어로 수정"""
        right_panel = QWidget()
        right_panel.setStyleSheet("background-color: #F8F9FA;")
        right_layout = QVBoxLayout(right_panel)
        
        # 이미지 뷰어 그룹
        viewer_group = QGroupBox("🖼️ 이미지 미리보기")
        viewer_layout = QGridLayout(viewer_group)
        viewer_layout.setSpacing(15)
        
        # 원본 이미지
        self.original_viewer = ImageViewer("📷 원본 이미지")
        viewer_layout.addWidget(self.original_viewer, 0, 0)
        
        # 워터마크 마스크 이미지  
        self.mask_viewer = ImageViewer("💧 워터마크 마스크")
        viewer_layout.addWidget(self.mask_viewer, 0, 1)
        
        # 합성 이미지
        self.composite_viewer = ImageViewer("✨ 합성 미리보기 (워터마크+컬러)")
        viewer_layout.addWidget(self.composite_viewer, 1, 0, 1, 2)
        
        # 그리드 비율 설정
        viewer_layout.setRowStretch(0, 1)
        viewer_layout.setRowStretch(1, 1)
        viewer_layout.setColumnStretch(0, 1)
        viewer_layout.setColumnStretch(1, 1)
        
        right_layout.addWidget(viewer_group)
        parent_splitter.addWidget(right_panel)
        
    def create_status_bar(self, parent_layout):
        """하단 상태바 생성"""
        status_frame = QFrame()
        status_frame.setFixedHeight(40)
        status_frame.setStyleSheet("""
            QFrame {
                background-color: #FFFFFF;
                border: none;
                border-top: 1px solid #DEE2E6;
            }
        """)
        
        status_layout = QHBoxLayout(status_frame)
        status_layout.setContentsMargins(20, 10, 20, 10)
        
        self.status_text = QLabel("준비 완료 | AI 모델 초기화 중...")
        self.status_text.setStyleSheet("color: #6C757D; font-size: 11px;")
        
        status_layout.addWidget(self.status_text)
        status_layout.addStretch()
        
        version_label = QLabel("Hana Studio v1.0 - 워터마크 지원")
        version_label.setStyleSheet("color: #ADB5BD; font-size: 10px;")
        status_layout.addWidget(version_label)
        
        parent_layout.addWidget(status_frame)
    
    def check_printer_availability(self):
        """프린터 사용 가능 여부 확인"""
        if not self.printer_available:
            self.printer_status_label.setText("⚠️ 워터마크 프린터 사용 불가")
            return
        def check():
            try:
                self.printer_dll_path = find_printer_dll()
                if self.printer_dll_path:
                    self.printer_status_label.setText("✅ 워터마크 프린터 사용 가능")
                    self.test_printer_btn.setEnabled(True)
                else:
                    self.printer_status_label.setText("❌ DLL 파일 없음")
                    self.test_printer_btn.setEnabled(False)
            except Exception as e:
                self.printer_status_label.setText("오류 발생")
                self.log(f"❌ 프린터 확인 오류: {e}")
        threading.Thread(target=check, daemon=True).start()

        
    def init_ai_model(self):
        """AI 모델 초기화"""
        def load_model():
            try:
                model_name = config.get('ai_model', 'isnet-general-use')
                self.log(f"AI 모델 초기화 중: {model_name}")
                self.session = new_session(model_name=model_name)
                model_info = config.get_ai_model_info(model_name)
                if model_info:
                    self.log(f"✅ {model_info['name']} 로드 완료!")
                else:
                    self.log("✅ AI 모델 로드 완료!")
                self.status_text.setText("준비 완료 | AI 모델 로드 성공")
            except Exception as e:
                self.log(f"❌ AI 모델 로드 실패: {e}")
                self.status_text.setText("오류 | AI 모델 로드 실패")
        
        # 백그라운드에서 모델 로드
        threading.Thread(target=load_model, daemon=True).start()
    
    def on_print_mode_changed(self):
        """인쇄 모드 변경 - 워터마크 인쇄 방식으로 변경"""
        if self.normal_radio.isChecked():
            self.print_mode = "normal"
            self.mode_description_label.setText("📖 컬러(YMCK)만 인쇄 - 워터마크 없음")
            self.print_card_btn.setText("컬러 카드 인쇄")
        else:
            self.print_mode = "watermark"
            self.mode_description_label.setText("📖 워터마크 배경 위에 컬러 이미지 인쇄")
            self.print_card_btn.setText("워터마크 카드 인쇄")
        self.update_print_button_state()
        
        mode_text = "컬러만 인쇄" if self.print_mode == "normal" else "워터마크 + 컬러"
        self.log(f"인쇄 모드 변경: {mode_text}")

    def update_print_button_state(self):
        """인쇄 버튼 활성화 상태 업데이트"""
        if not self.printer_available or not self.printer_dll_path:
            self.print_card_btn.setEnabled(False)
            return
        
        if self.print_mode == "normal":
            # 일반 인쇄: 원본 이미지만 필요
            can_print = self.current_image_path is not None
            
            if can_print and hasattr(self, 'ai_result_image') and self.ai_result_image is not None:
                self.log("✅ 투명 배경 이미지 + 일반 인쇄 모드: 완벽한 조합!")
                
        else:
            # 워터마크 인쇄: 원본 이미지 + 워터마크 마스크 필요
            can_print = (self.current_image_path is not None and 
                        self.mask_image is not None)
        
        self.print_card_btn.setEnabled(can_print)
    
    def test_printer_connection(self):
        """프린터 연결 테스트"""
        if not self.printer_available or not self.printer_dll_path:
            QMessageBox.warning(self, "경고", "프린터 DLL을 찾을 수 없습니다.")
            return
        
        self.test_printer_btn.setEnabled(False)
        self.test_printer_btn.setText("테스트 중...")
        
        def test_connection():
            try:
                if test_printer_connection():
                    self.log("✅ 워터마크 프린터 연결 테스트 성공!")
                    self.printer_status_label.setText("✅ 워터마크 프린터 연결 가능")
                    self.print_card_btn.setEnabled(True)
                    QMessageBox.information(self, "성공", "워터마크 프린터 연결 테스트가 성공했습니다!")
                else:
                    self.log("❌ 워터마크 프린터 연결 테스트 실패")
                    self.printer_status_label.setText("❌ 프린터 연결 실패")
                    QMessageBox.warning(self, "실패", "프린터를 찾을 수 없습니다.\n프린터가 켜져 있고 네트워크에 연결되어 있는지 확인해주세요.")
            except Exception as e:
                self.log(f"❌ 프린터 테스트 오류: {e}")
                QMessageBox.critical(self, "오류", f"프린터 테스트 중 오류가 발생했습니다:\n{e}")
            finally:
                self.test_printer_btn.setEnabled(True)
                self.test_printer_btn.setText("프린터 연결 테스트")
        
        # 백그라운드에서 테스트 실행
        threading.Thread(target=test_connection, daemon=True).start()
    
    def print_card(self):
        """카드 인쇄 - 워터마크 모드 지원"""
        if not self.printer_available or not self.printer_dll_path:
            QMessageBox.warning(self, "경고", "프린터를 사용할 수 없습니다.")
            return
        
        if not self.current_image_path:
            QMessageBox.warning(self, "경고", "원본 이미지를 먼저 선택해주세요.")
            return
        
        # 인쇄 모드별 확인
        if self.print_mode == "watermark":
            if self.mask_image is None:
                QMessageBox.warning(self, "경고", "워터마크 인쇄를 위해서는 워터마크 생성을 먼저 실행해주세요.")
                return
            
            # 워터마크 마스크가 저장되지 않은 경우 임시 저장
            if not self.saved_mask_path or not os.path.exists(self.saved_mask_path):
                if not self.save_mask_for_printing():
                    return
        
        # 인쇄 확인 다이얼로그
        mode_text = "컬러만 인쇄 (YMCK)" if self.print_mode == "normal" else "워터마크 + 컬러 인쇄"
        detail_text = f"원본 이미지: {os.path.basename(self.current_image_path)}\n"
        
        if self.print_mode == "watermark":
            detail_text += f"워터마크: 배경에 마스킹된 패턴 인쇄\n"
            detail_text += f"컬러: 워터마크 위에 컬러 이미지 인쇄\n"
        else:
            detail_text += f"컬러 인쇄: 원본 이미지 전체 (워터마크 없음)\n"
        
        detail_text += f"인쇄 모드: {mode_text}"
        
        reply = QMessageBox.question(
            self, 
            "카드 인쇄", 
            f"카드 인쇄를 시작하시겠습니까?\n\n{detail_text}\n\n"
            "프린터에 카드가 준비되어 있는지 확인해주세요.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        # 인쇄 시작
        self.print_card_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # 무한 진행바
        
        # 프린터 스레드 시작
        if self.print_mode == "normal":
            # 일반 인쇄: 마스크 없이
            print_image_path = getattr(self, 'saved_color_path', self.current_image_path)
            self.printer_thread = PrinterThread(
                self.printer_dll_path,
                print_image_path,
                None,  # 워터마크 없음
                self.print_mode
            )
        else:
            # 워터마크 인쇄
            print_image_path = getattr(self, 'saved_color_path', self.current_image_path)
            self.printer_thread = PrinterThread(
                self.printer_dll_path,
                print_image_path,
                self.saved_mask_path,  # 워터마크 마스크
                self.print_mode
            )
        
        self.printer_thread.progress.connect(self.on_printer_progress)
        self.printer_thread.finished.connect(self.on_printer_finished)
        self.printer_thread.error.connect(self.on_printer_error)
        self.printer_thread.start()
    
    def save_mask_for_printing(self) -> bool:
        """워터마크 인쇄용 파일 생성"""
        try:
            # temp 폴더에 워터마크 마스크 저장
            temp_dir = config.get('directories.temp', 'temp')
            os.makedirs(temp_dir, exist_ok=True)
            
            base_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
            mask_filename = f"{base_name}_watermark.jpg"
            self.saved_mask_path = os.path.join(temp_dir, mask_filename)
            
            # 워터마크 마스크 저장
            quality = config.get('output_quality', 95)
            cv2.imwrite(self.saved_mask_path, self.mask_image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            
            # 컬러 레이어: 원본 이미지 사용
            self.saved_color_path = self.current_image_path
            
            self.log(f"✅ 워터마크 마스크 저장: {self.saved_mask_path}")
            self.log(f"✅ 컬러 레이어: 원본 이미지")
            return True
            
        except Exception as e:
            self.log(f"❌ 인쇄용 파일 생성 실패: {e}")
            QMessageBox.critical(self, "오류", f"인쇄용 파일 생성 실패:\n{e}")
            return False
    
    def on_printer_progress(self, message):
        """프린터 진행상황 업데이트"""
        self.status_label.setText(message)
        self.log(message)
    
    def on_printer_finished(self, success):
        """프린터 작업 완료"""
        self.progress_bar.setVisible(False)
        self.print_card_btn.setEnabled(True)
        
        mode_text = "일반 인쇄" if self.print_mode == "normal" else "워터마크 인쇄"
        
        if success:
            self.log(f"✅ {mode_text} 완료!")
            self.status_text.setText("인쇄 완료")
            success_msg = f"{mode_text}가 완료되었습니다!"
            if self.print_mode == "watermark":
                success_msg += "\n워터마크 배경 위에 컬러 이미지가 인쇄되었습니다."
            QMessageBox.information(self, "성공", success_msg)
        else:
            self.log(f"❌ {mode_text} 실패")
            self.status_text.setText("인쇄 실패")
    
    def on_printer_error(self, error_message):
        """프린터 오류 처리"""
        self.progress_bar.setVisible(False)
        self.print_card_btn.setEnabled(True)
        self.log(f"❌ 프린터 오류: {error_message}")
        self.status_text.setText("인쇄 오류 발생")
        QMessageBox.critical(self, "인쇄 오류", f"카드 인쇄 중 오류가 발생했습니다:\n\n{error_message}")
        
    def log(self, message):
        """로그 메시지 추가"""
        self.log_text.append(f"[{self.get_timestamp()}] {message}")
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )
        
    def get_timestamp(self):
        """현재 시간 문자열 반환"""
        return datetime.now().strftime("%H:%M:%S")
        
    def select_image(self):
        """이미지 선택"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "이미지 선택", 
            "", 
            config.get_image_filter()
        )
        
        if file_path:
            # 지원되는 형식인지 확인
            if not config.is_supported_image(file_path):
                QMessageBox.warning(self, "경고", "지원하지 않는 이미지 형식입니다.")
                return
            
            # 파일 크기 확인
            max_size_mb = 50
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            if file_size_mb > max_size_mb:
                QMessageBox.warning(self, "경고", f"파일 크기가 너무 큽니다. (최대 {max_size_mb}MB)")
                return
            
            self.current_image_path = file_path
            self.file_label.setText(f"📁 {os.path.basename(file_path)}")
            self.log(f"이미지 선택: {os.path.basename(file_path)} ({file_size_mb:.1f}MB)")
            
            # 원본 이미지 표시
            self.original_viewer.set_image(file_path)
            
            # 이미지 로드
            self.original_image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            
            # 투명 배경 정보 로깅
            if len(self.original_image.shape) == 3 and self.original_image.shape[2] == 4:
                self.log("✅ PNG 투명 배경이 감지되었습니다")
            else:
                self.log("📷 일반 이미지 (투명 배경 없음)")
            
            self.process_btn.setEnabled(True)
            self.status_text.setText("이미지 로드 완료 | 워터마크 생성 대기 중")
            
            # 이전 결과 초기화
            self.mask_image = None
            self.composite_image = None
            self.saved_mask_path = None
            self.saved_color_path = None
            self.ai_result_path = None
            self.ai_result_image = None
            self.export_btn.setEnabled(False)
            
            # 인쇄 버튼 상태 업데이트
            self.update_print_button_state()
            
    def process_image(self):
        """이미지 처리 시작"""
        if not self.current_image_path or not self.session:
            return
            
        self.process_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # 무한 진행바
        
        # 처리 스레드 시작
        self.processing_thread = ProcessingThread(self.current_image_path, self.session)
        self.processing_thread.finished.connect(self.on_processing_finished)
        self.processing_thread.error.connect(self.on_processing_error)
        self.processing_thread.progress.connect(self.on_processing_progress)
        self.processing_thread.start()
        
    def on_processing_progress(self, message):
        """처리 진행상황 업데이트"""
        self.status_label.setText(message)
        self.log(message)
        
    def on_processing_finished(self, mask_array):
        """처리 완료"""
        self.mask_image = mask_array
        
        # 워터마크 마스크 표시
        self.mask_viewer.set_image(mask_array)
        
        # AI 배경 제거 결과 저장
        self.save_ai_result_for_preview()
        
        # 마스크 통계 정보 표시
        self.display_mask_statistics(mask_array)
        
        # 합성 이미지 생성 및 표시
        self.create_composite_preview()
        
        self.progress_bar.setVisible(False)
        self.process_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
        
        # 인쇄 버튼 상태 업데이트
        self.update_print_button_state()
        
        self.log("✅ 워터마크 생성 완료!")
        self.status_text.setText("워터마크 생성 완료 | 결과 저장 및 인쇄 가능")

    def save_ai_result_for_preview(self):
        """AI 배경 제거 결과를 미리보기용으로 저장"""
        try:
            # AI 배경 제거 재실행
            with open(self.current_image_path, 'rb') as f:
                input_data = f.read()
            
            from rembg import remove
            result = remove(input_data, session=self.session)
            
            # 임시 파일로 저장
            temp_dir = config.get('directories.temp', 'temp')
            os.makedirs(temp_dir, exist_ok=True)
            
            base_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
            preview_filename = f"{base_name}_ai_result.png"
            self.ai_result_path = os.path.join(temp_dir, preview_filename)
            
            with open(self.ai_result_path, 'wb') as f:
                f.write(result)
            
            # AI 결과를 opencv로 로드 (미리보기용)
            self.ai_result_image = cv2.imread(self.ai_result_path, cv2.IMREAD_UNCHANGED)
            
            self.log("💡 AI 배경 제거 결과를 미리보기용으로 저장했습니다")
            
        except Exception as e:
            self.log(f"⚠️ AI 결과 저장 실패: {e}")
            self.ai_result_image = None

    def display_mask_statistics(self, mask_array):
        """마스크 통계 정보를 UI에 표시"""
        try:
            # 그레이스케일로 변환 (RGB 마스크인 경우)
            if len(mask_array.shape) == 3:
                mask_gray = cv2.cvtColor(mask_array, cv2.COLOR_RGB2GRAY)
            else:
                mask_gray = mask_array
            
            # 통계 계산
            total_pixels = mask_gray.size
            white_pixels = np.sum(mask_gray > 128)  # 흰색 픽셀 (워터마크 인쇄될 객체 부분)
            black_pixels = total_pixels - white_pixels  # 검은색 픽셀 (배경 부분)
            
            white_ratio = (white_pixels / total_pixels) * 100
            black_ratio = (black_pixels / total_pixels) * 100
            
            # 로그에 상세 정보 출력
            self.log(f"📊 워터마크 마스크 통계:")
            self.log(f"   전체 픽셀: {total_pixels:,}")
            self.log(f"   객체 영역 (워터마크): {white_pixels:,} ({white_ratio:.1f}%)")
            self.log(f"   배경 영역 (투명): {black_pixels:,} ({black_ratio:.1f}%)")
            
            # 마스크 품질 평가
            if white_ratio < 5:
                self.log("⚠️ 객체 영역이 매우 작습니다. 임계값을 낮춰보세요.")
            elif white_ratio > 80:
                self.log("⚠️ 객체 영역이 매우 큽니다. 임계값을 높이거나 다른 이미지를 사용해보세요.")
            else:
                self.log("✅ 적절한 객체 영역 비율입니다.")
                
        except Exception as e:
            self.log(f"마스크 통계 계산 오류: {e}")
        
    def on_processing_error(self, error_message):
        """처리 오류 처리"""
        self.progress_bar.setVisible(False)
        self.process_btn.setEnabled(True)
        self.log(f"❌ 워터마크 생성 오류: {error_message}")
        self.status_text.setText("오류 발생 | 다시 시도해주세요")
        
    def create_composite_preview(self):
        """합성 미리보기 생성 - 워터마크 시각화"""
        if self.mask_image is not None:
            try:
                # AI 배경 제거 결과 이미지 사용
                if hasattr(self, 'ai_result_image') and self.ai_result_image is not None:
                    # AI 결과 이미지가 있으면 사용
                    if self.ai_result_image.shape[2] == 4:
                        # BGRA → BGR 변환 (투명 부분은 검은색으로)
                        bgr = self.ai_result_image[:, :, :3]
                        alpha = self.ai_result_image[:, :, 3]
                        
                        # 투명 부분을 검은색으로 변환
                        transparent_mask = alpha < 128
                        display_image = bgr.copy()
                        display_image[transparent_mask] = [0, 0, 0]  # 검은색
                        
                        self.log("🎨 AI 배경 제거 결과로 미리보기 생성")
                    else:
                        display_image = self.ai_result_image.copy()
                else:
                    # AI 결과가 없으면 원본 사용
                    display_image = self.original_image.copy()
                
                # 워터마크 시각화를 위한 합성 미리보기
                composite = display_image.copy()
                
                # 마스크를 파란색으로 오버레이하여 워터마크 영역 표시
                mask_gray = cv2.cvtColor(self.mask_image, cv2.COLOR_RGB2GRAY)
                
                # 워터마크 인쇄될 영역(흰색 부분)을 파란색으로 표시
                blue_overlay = np.zeros_like(composite)
                blue_overlay[:, :, 0] = mask_gray  # 파란색 채널에 마스크 적용
                
                # 반투명 오버레이
                composite = cv2.addWeighted(composite, 0.7, blue_overlay, 0.3, 0)
                
                self.composite_image = composite
                self.composite_viewer.set_image(composite)
                
                self.log("🎨 합성 미리보기: 파란색=워터마크영역")
                
            except Exception as e:
                self.log(f"합성 미리보기 생성 오류: {e}")

    def on_threshold_changed(self, value):
        """임계값 변경 시 호출"""
        config.set('alpha_threshold', value)
        
        # 라벨 업데이트
        if self.threshold_label:
            self.threshold_label.setText(f"알파 임계값: {value}")
        
        self.log(f"임계값 변경: {value}")
        
        # 이미지가 로드되어 있으면 자동 재처리 (선택사항)
        if self.current_image_path and config.get('auto_reprocess', False):
            self.process_image()
            
    def export_results(self):
        """결과 내보내기"""
        if self.mask_image is None:
            return
            
        # 저장 폴더 선택
        output_dir = config.get('directories.output', 'output')
        folder_path = QFileDialog.getExistingDirectory(
            self, 
            "저장 폴더 선택", 
            output_dir
        )
        if not folder_path:
            return
            
        try:
            base_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
            quality = config.get('output_quality', 95)
            
            # 워터마크 마스크 저장
            mask_path = os.path.join(folder_path, f"{base_name}_watermark.jpg")
            cv2.imwrite(mask_path, self.mask_image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            
            # 합성 이미지 저장 (있는 경우)
            if self.composite_image is not None:
                composite_path = os.path.join(folder_path, f"{base_name}_composite.jpg")
                cv2.imwrite(composite_path, self.composite_image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            
            # 원본 이미지도 복사 (선택사항)
            if config.get('auto_save_original', False):
                import shutil
                original_path = os.path.join(folder_path, f"{base_name}_original{Path(self.current_image_path).suffix}")
                shutil.copy2(self.current_image_path, original_path)
            
            self.log(f"✅ 워터마크 결과 저장 완료: {folder_path}")
            self.status_text.setText("저장 완료")
            
            # 성공 메시지 표시
            QMessageBox.information(
                self, 
                "저장 완료", 
                f"워터마크 처리 결과가 저장되었습니다.\n위치: {folder_path}"
            )
            
        except Exception as e:
            self.log(f"❌ 저장 오류: {e}")
            QMessageBox.critical(self, "오류", f"저장 중 오류가 발생했습니다:\n{e}")
                              
def main():
    """메인 함수"""
    # 필요한 디렉토리 생성
    import os
    for dir_name in ['temp', 'output', 'models', 'logs']:
        os.makedirs(dir_name, exist_ok=True)
    
    # DPI 스케일링 문제 해결
    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "0"
    os.environ["QT_SCALE_FACTOR_ROUNDING_POLICY"] = "Floor"
    os.environ["QT_FONT_DPI"] = "96"
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "0"
    
    app = QApplication(sys.argv)
    
    # 고해상도 DPI 지원 설정 (안전한 방식)
    try:
        app.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, False)
        app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)
    except AttributeError:
        # 구 버전 PySide6에서는 일부 속성이 없을 수 있음
        print("일부 DPI 속성을 설정할 수 없습니다. 버전 호환성 문제일 수 있습니다.")
    
    # 앱 정보 설정
    app.setApplicationName(AppConstants.APP_NAME)
    app.setApplicationVersion(AppConstants.APP_VERSION)
    app.setOrganizationName(AppConstants.APP_AUTHOR)
    
    # 라이트 테마 팔레트 강제 설정
    light_palette = QPalette()
    light_palette.setColor(QPalette.ColorRole.Window, QColor("#F8F9FA"))
    light_palette.setColor(QPalette.ColorRole.WindowText, QColor("#212529"))
    light_palette.setColor(QPalette.ColorRole.Base, QColor("#FFFFFF"))
    light_palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#E9ECEF"))
    light_palette.setColor(QPalette.ColorRole.ToolTipBase, QColor("#FFFFFF"))
    light_palette.setColor(QPalette.ColorRole.ToolTipText, QColor("#212529"))
    light_palette.setColor(QPalette.ColorRole.Text, QColor("#212529"))
    light_palette.setColor(QPalette.ColorRole.Button, QColor("#F8F9FA"))
    light_palette.setColor(QPalette.ColorRole.ButtonText, QColor("#212529"))
    light_palette.setColor(QPalette.ColorRole.BrightText, QColor("#DC3545"))
    light_palette.setColor(QPalette.ColorRole.Link, QColor("#4A90E2"))
    light_palette.setColor(QPalette.ColorRole.Highlight, QColor("#4A90E2"))
    light_palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#FFFFFF"))
    app.setPalette(light_palette)
    
    # Fusion 스타일 적용
    app.setStyle('Fusion')
    
    # 설정 검증
    if not config.validate_settings():
        print("⚠️ 설정 검증 실패, 기본값으로 복원합니다.")
        config.reset_to_defaults()
    
    # 메인 윈도우 생성 및 표시
    window = HanaStudio()
    window.show()
    
    # 종료 시 설정 저장
    def save_window_geometry():
        geometry = window.geometry()
        config.set('window_geometry.x', geometry.x())
        config.set('window_geometry.y', geometry.y())
        config.set('window_geometry.width', geometry.width())
        config.set('window_geometry.height', geometry.height())
        config.save_settings()
    
    app.aboutToQuit.connect(save_window_geometry)
    
    # 애플리케이션 실행
    try:
        sys.exit(app.exec())
    except KeyboardInterrupt:
        print("\n프로그램이 사용자에 의해 중단되었습니다.")
        sys.exit(0)


if __name__ == "__main__":
    main()