"""
Hana Studio 메인 애플리케이션 클래스 - 리팩토링된 버전
"""

import os
import cv2
import numpy as np
import threading
from pathlib import Path

from PySide6.QtWidgets import QMainWindow, QFileDialog, QMessageBox
from PySide6.QtCore import Qt

# 분리된 모듈들 import
from ui import HanaStudioMainWindow, get_app_style
from core import ImageProcessor, ProcessingThread, FileManager
from printer import PrinterThread, find_printer_dll, test_printer_connection
from config import config, AppConstants


class HanaStudio(QMainWindow):
    """Hana Studio 메인 애플리케이션 클래스"""
    
    def __init__(self):
        super().__init__()
        
        # 데이터 속성들
        self.current_image_path = None
        self.original_image = None
        self.mask_image = None
        self.composite_image = None
        self.saved_mask_path = None
        self.print_mode = "normal"
        
        # 코어 모듈들
        self.image_processor = ImageProcessor()
        self.file_manager = FileManager()
        
        # 프린터 관련
        self.printer_available = False
        self.printer_dll_path = None
        
        # UI 초기화
        self.ui = HanaStudioMainWindow(self)
        self._setup_window()
        self._connect_signals()
        self._check_printer_availability()
        
    def _setup_window(self):
        """윈도우 기본 설정"""
        self.setWindowTitle(f"{AppConstants.APP_NAME} - {AppConstants.APP_DESCRIPTION}")
        
        # 윈도우 크기 설정
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
        
        # 스타일 적용
        self.setStyleSheet(get_app_style())
    
    def _connect_signals(self):
        """시그널 연결"""
        components = self.ui.components
        
        # 파일 선택
        components['file_panel'].select_btn.clicked.connect(self.select_image)
        
        # 이미지 처리
        components['processing_panel'].process_requested.connect(self.process_image)
        components['processing_panel'].export_requested.connect(self.export_results)
        
        # 인쇄 모드
        components['print_mode_panel'].mode_changed.connect(self.on_print_mode_changed)
        
        # 프린터
        components['printer_panel'].test_requested.connect(self.test_printer_connection)
        components['printer_panel'].print_requested.connect(self.print_card)
    
    def _check_printer_availability(self):
        """프린터 사용 가능성 확인"""
        def check():
            try:
                self.printer_dll_path = find_printer_dll()
                if self.printer_dll_path:
                    self.printer_available = True
                    self.ui.components['printer_panel'].update_status("✅ 프린터 사용 가능")
                else:
                    self.ui.components['printer_panel'].update_status("❌ DLL 파일 없음")
            except Exception as e:
                self.log(f"❌ 프린터 확인 오류: {e}")
        
        threading.Thread(target=check, daemon=True).start()
    
    def select_image(self):
        """이미지 선택"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "이미지 선택",
            "",
            config.get_image_filter()
        )
        
        if not file_path:
            return
        
        # 이미지 유효성 검사
        is_valid, error_msg = self.image_processor.validate_image(file_path)
        if not is_valid:
            QMessageBox.warning(self, "경고", error_msg)
            return
        
        # 파일 정보 업데이트
        self.current_image_path = file_path
        file_name, file_size_mb = self.file_manager.get_file_info(file_path)
        
        self.ui.components['file_panel'].update_file_info(file_path)
        self.log(f"이미지 선택: {file_name} ({file_size_mb:.1f}MB)")
        
        # 원본 이미지 표시
        self.ui.components['original_viewer'].set_image(file_path)
        self.original_image = cv2.imread(file_path)
        
        # UI 상태 업데이트
        self.ui.components['processing_panel'].set_process_enabled(True)
        self.ui.components['status_text'].setText("이미지 로드 완료 | 처리 대기 중")
        
        # 이전 결과 초기화
        self._reset_processing_results()
    
    def _reset_processing_results(self):
        """처리 결과 초기화"""
        self.mask_image = None
        self.composite_image = None
        self.saved_mask_path = None
        
        self.ui.components['processing_panel'].set_export_enabled(False)
        self.ui.components['mask_viewer'].clear_image()
        self.ui.components['composite_viewer'].clear_image()
        
        self._update_print_button_state()
    
    def process_image(self):
        """이미지 처리 시작"""
        if not self.current_image_path:
            return
        
        # UI 상태 변경
        self.ui.components['processing_panel'].set_process_enabled(False)
        self.ui.components['progress_panel'].show_progress()
        
        # 처리 스레드 시작
        self.processing_thread = ProcessingThread(self.current_image_path, self.image_processor)
        self.processing_thread.finished.connect(self.on_processing_finished)
        self.processing_thread.error.connect(self.on_processing_error)
        self.processing_thread.progress.connect(self.on_processing_progress)
        self.processing_thread.start()
    
    def on_processing_progress(self, message):
        """처리 진행상황 업데이트"""
        self.ui.components['progress_panel'].update_status(message)
        self.log(message)
    
    def on_processing_finished(self, mask_array):
        """처리 완료"""
        self.mask_image = mask_array
        
        # 마스크 이미지 표시
        self.ui.components['mask_viewer'].set_image(mask_array)
        
        # 합성 이미지 생성 및 표시
        if self.original_image is not None:
            self.composite_image = self.image_processor.create_composite_preview(
                self.original_image, self.mask_image
            )
            self.ui.components['composite_viewer'].set_image(self.composite_image)
        
        # UI 상태 업데이트
        self.ui.components['progress_panel'].hide_progress()
        self.ui.components['processing_panel'].set_process_enabled(True)
        self.ui.components['processing_panel'].set_export_enabled(True)
        
        self._update_print_button_state()
        
        self.log("✅ 배경 제거 처리 완료!")
        self.ui.components['status_text'].setText("처리 완료 | 결과 저장 및 인쇄 가능")
    
    def on_processing_error(self, error_message):
        """처리 오류"""
        self.ui.components['progress_panel'].hide_progress()
        self.ui.components['processing_panel'].set_process_enabled(True)
        
        self.log(f"❌ 처리 오류: {error_message}")
        self.ui.components['status_text'].setText("오류 발생 | 다시 시도해주세요")
        
        QMessageBox.critical(self, "처리 오류", f"이미지 처리 중 오류가 발생했습니다:\n\n{error_message}")
    
    def export_results(self):
        """결과 저장"""
        if self.mask_image is None:
            return
        
        # 저장 폴더 선택
        output_dir = config.get('directories.output', 'output')
        folder_path = QFileDialog.getExistingDirectory(self, "저장 폴더 선택", output_dir)
        
        if not folder_path:
            return
        
        # 저장 실행
        success, message = self.file_manager.export_results(
            self.current_image_path,
            self.mask_image,
            self.composite_image,
            folder_path
        )
        
        if success:
            self.log(f"✅ {message}")
            self.ui.components['status_text'].setText("저장 완료")
            QMessageBox.information(self, "저장 완료", message)
        else:
            self.log(f"❌ {message}")
            QMessageBox.critical(self, "저장 오류", message)
    
    def on_print_mode_changed(self, mode):
        """인쇄 모드 변경"""
        self.print_mode = mode
        self.ui.components['printer_panel'].update_print_button_text(mode)
        self._update_print_button_state()
        
        mode_text = '일반 인쇄' if mode == 'normal' else '레이어 인쇄(YMCW)'
        self.log(f"인쇄 모드 변경: {mode_text}")
    
    def _update_print_button_state(self):
        """인쇄 버튼 상태 업데이트"""
        if not self.printer_available or not self.printer_dll_path:
            self.ui.components['printer_panel'].set_print_enabled(False)
            return
        
        if self.print_mode == "normal":
            can_print = self.current_image_path is not None
        else:
            can_print = (self.current_image_path is not None and self.mask_image is not None)
        
        self.ui.components['printer_panel'].set_print_enabled(can_print)
    
    def test_printer_connection(self):
        """프린터 연결 테스트"""
        if not self.printer_available or not self.printer_dll_path:
            QMessageBox.warning(self, "경고", "프린터 DLL을 찾을 수 없습니다.")
            return
        
        self.ui.components['printer_panel'].set_test_enabled(False)
        
        def test_connection():
            try:
                if test_printer_connection():
                    self.log("✅ 프린터 연결 테스트 성공!")
                    self.ui.components['printer_panel'].update_status("✅ 프린터 연결 가능")
                    QMessageBox.information(self, "성공", "프린터 연결 테스트가 성공했습니다!")
                else:
                    self.log("❌ 프린터 연결 테스트 실패")
                    self.ui.components['printer_panel'].update_status("❌ 프린터 연결 실패")
                    QMessageBox.warning(self, "실패", "프린터를 찾을 수 없습니다.\n프린터가 켜져 있고 네트워크에 연결되어 있는지 확인해주세요.")
            except Exception as e:
                self.log(f"❌ 프린터 테스트 오류: {e}")
                QMessageBox.critical(self, "오류", f"프린터 테스트 중 오류가 발생했습니다:\n{e}")
            finally:
                self.ui.components['printer_panel'].set_test_enabled(True)
        
        threading.Thread(target=test_connection, daemon=True).start()
    
    def print_card(self):
        """카드 인쇄"""
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
            
            # 마스크 이미지 저장
            self.saved_mask_path = self.file_manager.save_mask_for_printing(
                self.mask_image, self.current_image_path
            )
            if not self.saved_mask_path:
                QMessageBox.critical(self, "오류", "마스크 이미지 저장에 실패했습니다.")
                return
        
        # 인쇄 확인 다이얼로그
        mode_text = "일반 인쇄" if self.print_mode == "normal" else "레이어 인쇄 (YMCW)"
        file_name, _ = self.file_manager.get_file_info(self.current_image_path)
        
        detail_text = f"원본 이미지: {file_name}\n"
        if self.print_mode == "layered" and self.saved_mask_path:
            mask_name = os.path.basename(self.saved_mask_path)
            detail_text += f"마스크 이미지: {mask_name}\n"
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
        self.ui.components['printer_panel'].set_print_enabled(False)
        self.ui.components['progress_panel'].show_progress()
        
        # 프린터 스레드 시작
        mask_path = self.saved_mask_path if self.print_mode == "layered" else None
        self.printer_thread = PrinterThread(
            self.printer_dll_path,
            self.current_image_path,
            mask_path,
            self.print_mode
        )
        
        self.printer_thread.progress.connect(self.on_printer_progress)
        self.printer_thread.finished.connect(self.on_printer_finished)
        self.printer_thread.error.connect(self.on_printer_error)
        self.printer_thread.start()
    
    def on_printer_progress(self, message):
        """프린터 진행상황 업데이트"""
        self.ui.components['progress_panel'].update_status(message)
        self.log(message)
    
    def on_printer_finished(self, success):
        """프린터 작업 완료"""
        self.ui.components['progress_panel'].hide_progress()
        self.ui.components['printer_panel'].set_print_enabled(True)
        
        mode_text = "일반 인쇄" if self.print_mode == "normal" else "레이어 인쇄"
        
        if success:
            self.log(f"✅ {mode_text} 완료!")
            self.ui.components['status_text'].setText("인쇄 완료")
            QMessageBox.information(self, "성공", f"{mode_text}가 완료되었습니다!")
        else:
            self.log(f"❌ {mode_text} 실패")
            self.ui.components['status_text'].setText("인쇄 실패")
    
    def on_printer_error(self, error_message):
        """프린터 오류 처리"""
        self.ui.components['progress_panel'].hide_progress()
        self.ui.components['printer_panel'].set_print_enabled(True)
        
        self.log(f"❌ 프린터 오류: {error_message}")
        self.ui.components['status_text'].setText("인쇄 오류 발생")
        QMessageBox.critical(self, "인쇄 오류", f"카드 인쇄 중 오류가 발생했습니다:\n\n{error_message}")
    
    def log(self, message):
        """로그 메시지 추가"""
        self.ui.components['log_panel'].add_log(message)
    
    def closeEvent(self, event):
        """애플리케이션 종료 시"""
        # 임시 파일 정리
        self.file_manager.cleanup_temp_files()
        
        # 윈도우 크기 저장
        geometry = self.geometry()
        config.set('window_geometry.x', geometry.x())
        config.set('window_geometry.y', geometry.y())
        config.set('window_geometry.width', geometry.width())
        config.set('window_geometry.height', geometry.height())
        config.save_settings()
        
        event.accept()