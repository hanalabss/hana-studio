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
                              QMessageBox,QRadioButton,QButtonGroup)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap, QFont, QColor, QPalette, QImage

import cv2
import numpy as np
from PIL import Image
from rembg import remove, new_session

# 프로젝트 모듈들
from config import config, AppConstants

# 프린터 모듈 (선택적 import)
try:
    from printer_integration import PrinterThread, find_printer_dll, test_printer_connection
    PRINTER_AVAILABLE = True
    print("✅ 프린터 모듈 로드 성공")
except ImportError as e:
    PRINTER_AVAILABLE = False
    print(f"⚠️ 프린터 모듈 로드 실패: {e}")


class ModernButton(QPushButton):
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
            
            self.progress.emit("마스크 생성 중...")
            img_rgba = Image.open(io.BytesIO(result)).convert("RGBA")
            alpha = np.array(img_rgba.split()[-1])
            
            # 실루엣 마스크 생성 (배경은 흰색, 객체는 검은색)
            alpha_threshold = config.get('alpha_threshold', 45)
            mask = np.where(alpha > alpha_threshold, 0, 255).astype(np.uint8)
            mask_rgb = cv2.merge([mask, mask, mask])
            
            self.progress.emit("처리 완료!")
            self.finished.emit(mask_rgb)
            
        except Exception as e:
            self.error.emit(str(e))


class HanaStudio(QMainWindow):
    def __init__(self):
        super().__init__()
        self.session = None
        self.current_image_path = None
        self.original_image = None
        self.mask_image = None
        self.composite_image = None
        self.saved_mask_path = None  # 프린터용 마스크 파일 경로
        
        # 프린터 관련 변수
        self.printer_available = PRINTER_AVAILABLE
        self.printer_dll_path = None
        self.print_mode = "normal"  # "normal" 또는 "layered" - 새로 추가
        
        self.init_ui()
        self.init_ai_model()
        self.check_printer_availability()
        
    def init_ui(self):
        self.setWindowTitle(f"{AppConstants.APP_NAME} - {AppConstants.APP_DESCRIPTION}")
        
        # 설정에서 윈도우 크기 가져오기 (크기 증가)
        geometry = config.get('window_geometry')
        # 기본 크기를 더 크게 설정
        default_width = max(geometry.get('width', 1600), 1800)  # 최소 1800px
        default_height = max(geometry.get('height', 900), 1000)  # 최소 1000px
        
        self.setGeometry(
            geometry.get('x', 100), 
            geometry.get('y', 100), 
            default_width, 
            default_height
        )
        # 최소 크기 더 크게 설정
        self.setMinimumSize(1600, 900)  # 1200x800 -> 1600x900으로 증가
        
        # 라이트 테마 스타일 설정 (검은 배경 문제 해결)
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
        
        # 분할 비율 설정 (왼쪽 패널에 더 많은 공간 할당)
        content_splitter.setSizes([500, 1300])  # 450 -> 500으로 더 증가
        
    def create_header(self, parent_layout):
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
        
        subtitle_label = QLabel("AI 기반 이미지 배경 제거 및 카드 인쇄 도구")
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

        # ⚙️ 처리 옵션 그룹
        option_group = QGroupBox("⚙️ 처리 옵션")
        option_layout = QVBoxLayout(option_group)
        self.process_btn = ModernButton("배경 제거 시작", primary=True)
        self.process_btn.clicked.connect(self.process_image)
        self.process_btn.setEnabled(False)
        self.export_btn = ModernButton("결과 저장")
        self.export_btn.clicked.connect(self.export_results)
        self.export_btn.setEnabled(False)
        option_layout.addWidget(self.process_btn)
        option_layout.addWidget(self.export_btn)
        layout.addWidget(option_group)

        # 📋 인쇄 모드 그룹
        mode_group = QGroupBox("📋 인쇄 모드")
        mode_layout = QVBoxLayout(mode_group)
        self.normal_radio = QRadioButton("일반 인쇄")
        self.layered_radio = QRadioButton("레이어 인쇄")
        self.normal_radio.setChecked(True)
        self.normal_radio.toggled.connect(self.on_print_mode_changed)
        mode_layout.addWidget(self.normal_radio)
        mode_layout.addWidget(self.layered_radio)
        
        # ⬇️ 여기서 mode_description_label 추가됨
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

        # 🖨️ 프린터 그룹
        printer_group = QGroupBox("🖨 프린터 연동")
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


    def create_print_mode_group(self, parent_layout):
        """인쇄 모드 선택 그룹 생성 (프린터와 분리)"""
        mode_group = QGroupBox("📋 인쇄 모드 선택")
        mode_group.setMaximumHeight(110)  # 120 -> 110으로 축소
        mode_layout = QVBoxLayout(mode_group)
        mode_layout.setSpacing(5)  # 8 -> 5로 축소
        
        # 라디오 버튼 그룹
        self.print_mode_group = QButtonGroup()
        
        # 일반 인쇄 모드
        self.normal_print_radio = QRadioButton("🖼️ 일반 인쇄")
        self.normal_print_radio.setToolTip("원본 이미지만 인쇄 (기본 모드)")
        self.normal_print_radio.setChecked(True)  # 기본 선택
        self.normal_print_radio.toggled.connect(self.on_print_mode_changed)
        
        # 레이어 인쇄 모드
        self.layered_print_radio = QRadioButton("🎭 레이어 인쇄 (YMCW)")
        self.layered_print_radio.setToolTip("원본 이미지 + 마스크 워터마크 레이어 인쇄")
        self.layered_print_radio.toggled.connect(self.on_print_mode_changed)
        
        # 버튼 그룹에 추가
        self.print_mode_group.addButton(self.normal_print_radio, 0)
        self.print_mode_group.addButton(self.layered_print_radio, 1)
        
        # 라디오 버튼 스타일링 (크기 축소)
        radio_style = """
            QRadioButton {
                font-size: 11px;
                font-weight: 500;
                color: #495057;
                padding: 5px;
                spacing: 8px;
            }
            QRadioButton::indicator {
                width: 14px;
                height: 14px;
            }
            QRadioButton::indicator:unchecked {
                border: 2px solid #6C757D;
                border-radius: 7px;
                background-color: white;
            }
            QRadioButton::indicator:checked {
                border: 2px solid #4A90E2;
                border-radius: 7px;
                background-color: #4A90E2;
            }
            QRadioButton::indicator:checked:hover {
                background-color: #357ABD;
                border-color: #357ABD;
            }
            QRadioButton:hover {
                color: #4A90E2;
            }
        """
        self.normal_print_radio.setStyleSheet(radio_style)
        self.layered_print_radio.setStyleSheet(radio_style)
        
        mode_layout.addWidget(self.normal_print_radio)
        mode_layout.addWidget(self.layered_print_radio)
        
        # 모드 설명 라벨 (폰트 크기 축소)
        self.mode_description_label = QLabel("📖 원본 이미지만 인쇄합니다")
        self.mode_description_label.setStyleSheet("""
            color: #6C757D; 
            font-size: 9px; 
            padding: 3px 8px;
            background-color: #F8F9FA;
            border-radius: 4px;
            border-left: 3px solid #4A90E2;
        """)
        self.mode_description_label.setWordWrap(True)
        mode_layout.addWidget(self.mode_description_label)
        
        parent_layout.addWidget(mode_group)
    
    def create_printer_group(self, parent_layout):
        """프린터 연동 그룹 생성 (프린터 제어만)"""
        self.printer_group = QGroupBox("🖨️ 프린터 연동")
        self.printer_group.setMaximumHeight(120)  # 140 -> 120으로 축소
        printer_layout = QVBoxLayout(self.printer_group)
        printer_layout.setSpacing(5)  # 8 -> 5로 축소
        
        # 프린터 상태 라벨 (폰트 크기 축소)
        self.printer_status_label = QLabel("프린터 상태 확인 중...")
        self.printer_status_label.setStyleSheet("color: #6C757D; font-size: 9px; padding: 3px;")  # 폰트 크기 축소
        self.printer_status_label.setWordWrap(True)
        printer_layout.addWidget(self.printer_status_label)
        
        # 프린터 테스트 버튼 (크기 축소)
        self.test_printer_btn = ModernButton("프린터 연결 테스트")
        self.test_printer_btn.setFixedHeight(30)  # 버튼 높이 축소
        self.test_printer_btn.clicked.connect(self.test_printer_connection)
        printer_layout.addWidget(self.test_printer_btn)
        
        # 카드 인쇄 버튼 (크기 축소)
        self.print_card_btn = ModernButton("카드 인쇄 시작", primary=True)
        self.print_card_btn.setFixedHeight(35)  # 버튼 높이 축소
        self.print_card_btn.clicked.connect(self.print_card)
        self.print_card_btn.setEnabled(False)
        printer_layout.addWidget(self.print_card_btn)
        
        parent_layout.addWidget(self.printer_group)
        
        # 프린터가 사용 불가능한 경우 그룹 비활성화
        if not self.printer_available:
            self.printer_group.setEnabled(False)
            self.printer_status_label.setText("⚠️ 프린터 모듈을 사용할 수 없습니다")
    
    def create_right_panel(self, parent_splitter):
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
        
        # 마스크 이미지  
        self.mask_viewer = ImageViewer("🎭 마스크 이미지")
        viewer_layout.addWidget(self.mask_viewer, 0, 1)
        
        # 합성 이미지
        self.composite_viewer = ImageViewer("✨ 합성 미리보기")
        viewer_layout.addWidget(self.composite_viewer, 1, 0, 1, 2)
        
        # 그리드 비율 설정
        viewer_layout.setRowStretch(0, 1)
        viewer_layout.setRowStretch(1, 1)
        viewer_layout.setColumnStretch(0, 1)
        viewer_layout.setColumnStretch(1, 1)
        
        right_layout.addWidget(viewer_group)
        parent_splitter.addWidget(right_panel)
        
    def create_status_bar(self, parent_layout):
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
        
        version_label = QLabel("Hana Studio v1.0")
        version_label.setStyleSheet("color: #ADB5BD; font-size: 10px;")
        status_layout.addWidget(version_label)
        
        parent_layout.addWidget(status_frame)
    
    def check_printer_availability(self):
        if not self.printer_available:
            self.printer_status_label.setText("⚠️ 프린터 사용 불가")
            return
        def check():
            try:
                self.printer_dll_path = find_printer_dll()
                if self.printer_dll_path:
                    self.printer_status_label.setText("✅ 프린터 사용 가능")
                    self.test_printer_btn.setEnabled(True)
                else:
                    self.printer_status_label.setText("❌ DLL 파일 없음")
                    self.test_printer_btn.setEnabled(False)
            except Exception as e:
                self.printer_status_label.setText("오류 발생")
                self.log(f"❌ 프린터 확인 오류: {e}")
        threading.Thread(target=check, daemon=True).start()

        
    def init_ai_model(self):
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
        if self.normal_radio.isChecked():
            self.print_mode = "normal"
            self.mode_description_label.setText("📖 원본 이미지만 인쇄합니다")
            self.print_card_btn.setText("일반 카드 인쇄")
        else:
            self.print_mode = "layered"
            self.mode_description_label.setText("📖 원본 이미지 위에 마스크 워터마크를 포함하여 인쇄합니다")
            self.print_card_btn.setText("레이어 카드 인쇄")
        self.update_print_button_state()

        
        self.log(f"인쇄 모드 변경: {'일반 인쇄' if self.print_mode == 'normal' else '레이어 인쇄(YMCW)'}")

    def update_print_button_state(self):
        """인쇄 버튼 활성화 상태 업데이트"""
        if not self.printer_available or not self.printer_dll_path:
            self.print_card_btn.setEnabled(False)
            return
        
        if self.print_mode == "normal":
            # 일반 인쇄: 원본 이미지만 필요
            can_print = self.current_image_path is not None
        else:
            # 레이어 인쇄: 원본 이미지 + 마스크 필요
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
                    self.log("✅ 프린터 연결 테스트 성공!")
                    self.printer_status_label.setText("✅ 프린터 연결 가능")
                    self.print_card_btn.setEnabled(True)
                    QMessageBox.information(self, "성공", "프린터 연결 테스트가 성공했습니다!")
                else:
                    self.log("❌ 프린터 연결 테스트 실패")
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
        """개선된 카드 인쇄 실행"""
        if not self.printer_available or not self.printer_dll_path:
            QMessageBox.warning(self, "경고", "프린터를 사용할 수 없습니다.")
            return
        
        if not self.current_image_path:
            QMessageBox.warning(self, "경고", "원본 이미지를 먼저 선택해주세요.")
            return
        
        # 인쇄 모드별 확인
        if self.print_mode == "layered":
            if self.mask_image is None:
                QMessageBox.warning(self, "경고", "레이어 인쇄를 위해서는 배경 제거를 먼저 실행해주세요.")
                return
            
            # 마스크 이미지가 저장되지 않은 경우 임시 저장
            if not self.saved_mask_path or not os.path.exists(self.saved_mask_path):
                if not self.save_mask_for_printing():
                    return
        
        # 인쇄 확인 다이얼로그
        mode_text = "일반 인쇄" if self.print_mode == "normal" else "레이어 인쇄 (YMCW)"
        detail_text = f"원본 이미지: {os.path.basename(self.current_image_path)}\n"
        
        if self.print_mode == "layered":
            detail_text += f"마스크 이미지: {os.path.basename(self.saved_mask_path)}\n"
        
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
        
        # 프린터 스레드 시작 (인쇄 모드에 따라 다른 매개변수)
        if self.print_mode == "normal":
            # 일반 인쇄: 마스크 없이
            self.printer_thread = PrinterThread(
                self.printer_dll_path,
                self.current_image_path,
                None,  # 마스크 없음
                self.print_mode
            )
        else:
            # 레이어 인쇄: 마스크 포함
            self.printer_thread = PrinterThread(
                self.printer_dll_path,
                self.current_image_path,
                self.saved_mask_path,
                self.print_mode
            )
        
        self.printer_thread.progress.connect(self.on_printer_progress)
        self.printer_thread.finished.connect(self.on_printer_finished)
        self.printer_thread.error.connect(self.on_printer_error)
        self.printer_thread.start()
    
    def save_mask_for_printing(self) -> bool:
        """프린터용 마스크 이미지 저장"""
        try:
            # temp 폴더에 마스크 이미지 저장
            temp_dir = config.get('directories.temp', 'temp')
            os.makedirs(temp_dir, exist_ok=True)
            
            base_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
            mask_filename = f"{base_name}_mask_print.jpg"
            self.saved_mask_path = os.path.join(temp_dir, mask_filename)
            
            # 마스크 이미지 저장
            quality = config.get('output_quality', 95)
            cv2.imwrite(self.saved_mask_path, self.mask_image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            
            self.log(f"프린터용 마스크 저장: {self.saved_mask_path}")
            return True
            
        except Exception as e:
            self.log(f"❌ 마스크 저장 실패: {e}")
            QMessageBox.critical(self, "오류", f"마스크 이미지 저장 실패:\n{e}")
            return False
    
    def on_printer_progress(self, message):
        """프린터 진행상황 업데이트"""
        self.status_label.setText(message)
        self.log(message)
    
    def on_printer_finished(self, success):
        """프린터 작업 완료 (개선된 메시지)"""
        self.progress_bar.setVisible(False)
        self.print_card_btn.setEnabled(True)
        
        mode_text = "일반 인쇄" if self.print_mode == "normal" else "레이어 인쇄"
        
        if success:
            self.log(f"✅ {mode_text} 완료!")
            self.status_text.setText("인쇄 완료")
            QMessageBox.information(self, "성공", f"{mode_text}가 완료되었습니다!")
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
        self.log_text.append(f"[{self.get_timestamp()}] {message}")
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )
        
    def get_timestamp(self):
        return datetime.now().strftime("%H:%M:%S")
        
    def select_image(self):
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
            self.original_image = cv2.imread(file_path)
            
            self.process_btn.setEnabled(True)
            self.status_text.setText("이미지 로드 완료 | 처리 대기 중")
            
            # 이전 결과 초기화
            self.mask_image = None
            self.composite_image = None
            self.saved_mask_path = None
            self.export_btn.setEnabled(False)
            
            # 인쇄 버튼 상태 업데이트
            self.update_print_button_state()
            
    def process_image(self):
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
        self.status_label.setText(message)
        self.log(message)
        
    def on_processing_finished(self, mask_array):
        self.mask_image = mask_array
        
        # 마스크 이미지 표시
        self.mask_viewer.set_image(mask_array)
        
        # 합성 이미지 생성 및 표시
        self.create_composite_preview()
        
        self.progress_bar.setVisible(False)
        self.process_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
        
        # 인쇄 버튼 상태 업데이트
        self.update_print_button_state()
        
        self.log("✅ 배경 제거 처리 완료!")
        self.status_text.setText("처리 완료 | 결과 저장 및 인쇄 가능")
        
    def on_processing_error(self, error_message):
        self.progress_bar.setVisible(False)
        self.process_btn.setEnabled(True)
        self.log(f"❌ 처리 오류: {error_message}")
        self.status_text.setText("오류 발생 | 다시 시도해주세요")
        
    def create_composite_preview(self):
        if self.original_image is not None and self.mask_image is not None:
            # 간단한 합성 미리보기 (원본 + 마스크 오버레이)
            composite = self.original_image.copy()
            
            # 마스크를 반투명하게 오버레이
            mask_colored = cv2.applyColorMap(self.mask_image, cv2.COLORMAP_JET)
            composite = cv2.addWeighted(composite, 0.7, mask_colored, 0.3, 0)
            
            self.composite_image = composite
            self.composite_viewer.set_image(composite)
            
    def export_results(self):
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
            
            # 마스크 이미지 저장
            mask_path = os.path.join(folder_path, f"{base_name}_mask.jpg")
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
            
            self.log(f"✅ 결과 저장 완료: {folder_path}")
            self.status_text.setText("저장 완료")
            
            # 성공 메시지 표시
            QMessageBox.information(
                self, 
                "저장 완료", 
                f"처리된 이미지가 저장되었습니다.\n위치: {folder_path}"
            )
            
        except Exception as e:
            self.log(f"❌ 저장 오류: {e}")
            QMessageBox.critical(self, "오류", f"저장 중 오류가 발생했습니다:\n{e}")
                              
def main():
    # DPI 스케일링 문제 해결
    import os
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