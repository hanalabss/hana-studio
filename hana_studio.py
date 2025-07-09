"""
Hana Studio 메인 애플리케이션 클래스 - 수동 마스킹 기능 추가
자동 배경제거 후 사용자가 수동 마스킹 이미지를 업로드할 수 있음
"""

import os
import cv2
import numpy as np
import threading
import time
import tempfile
from pathlib import Path

from PySide6.QtWidgets import QMainWindow, QFileDialog, QMessageBox
from PySide6.QtCore import Qt

# 분리된 모듈들 import
from ui import HanaStudioMainWindow, get_app_style
from core import ImageProcessor, ProcessingThread, FileManager
from printer import PrinterThread, find_printer_dll, test_printer_connection
from printer.printer_thread import print_manager
from printer.printer_discovery import PrinterInfo
from ui.components.printer_selection_dialog import show_printer_selection_dialog
from config import config, AppConstants


class HanaStudio(QMainWindow):
    """Hana Studio 메인 애플리케이션 클래스 - 수동 마스킹 지원"""
    
    def __init__(self):
        super().__init__()
        
        # 데이터 속성들
        self.front_image_path = None
        self.back_image_path = None
        self.front_original_image = None
        self.back_original_image = None
        
        # 자동 배경제거 결과
        self.front_auto_mask_image = None
        self.back_auto_mask_image = None
        
        # 수동 마스킹 이미지 (새로 추가)
        self.front_manual_mask_path = None
        self.back_manual_mask_path = None
        self.front_manual_mask_image = None
        self.back_manual_mask_image = None
        
        # 최종 마스킹 이미지 (자동 또는 수동 중 선택)
        self.front_final_mask_image = None
        self.back_final_mask_image = None
        
        # 프린터용 저장된 마스크 경로
        self.front_saved_mask_path = None
        self.back_saved_mask_path = None
        
        self.print_mode = "normal"
        self.is_dual_side = False
        self.print_quantity = 1
        
        # 코어 모듈들
        self.image_processor = ImageProcessor()
        self.file_manager = FileManager()
        
        # 프린터 관련
        self.printer_available = False
        self.printer_dll_path = None
        self.current_printer_thread = None
        self.selected_printer_info = None
        
        # UI 초기화
        self.ui = HanaStudioMainWindow(self)
        self._setup_window()
        self._connect_signals()
        
        # 프린터 선택 (필수)
        if not self._select_printer_on_startup():
            import sys
            sys.exit(1)
        
        self._check_printer_availability()
        self._setup_manual_mask_viewers()
    
    def _setup_manual_mask_viewers(self):
        """수동 마스킹 뷰어 설정"""
        # 수동 마스킹 뷰어들을 클릭 업로드 모드로 설정
        self.ui.components['front_manual_mask_viewer'].enable_click_upload_mode(True)
        self.ui.components['back_manual_mask_viewer'].enable_click_upload_mode(True)
        
        # 파일 업로드 시그널 연결
        self.ui.components['front_manual_mask_viewer'].file_uploaded.connect(
            lambda path: self.on_manual_mask_uploaded(path, is_front=True)
        )
        self.ui.components['back_manual_mask_viewer'].file_uploaded.connect(
            lambda path: self.on_manual_mask_uploaded(path, is_front=False)
        )
    
    def _select_printer_on_startup(self) -> bool:
        """시작 시 프린터 선택 (필수)"""
        try:
            self.printer_dll_path = find_printer_dll()
            if not self.printer_dll_path:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.critical(
                    None,
                    "DLL 파일 없음",
                    "프린터 DLL 파일(libDSRetransfer600App.dll)을 찾을 수 없습니다.\n\n"
                    "DLL 파일을 다음 위치 중 하나에 배치해주세요:\n"
                    "• 메인 폴더\n"
                    "• dll/ 폴더\n"
                    "• lib/ 폴더"
                )
                return False
            
            selected_printer = show_printer_selection_dialog(self.printer_dll_path, self)
            
            if not selected_printer:
                return False
            
            self.selected_printer_info = selected_printer
            self.printer_available = True
            
            print(f"✅ 프린터 선택 완료: {selected_printer}")
            return True
            
        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(
                None,
                "프린터 선택 오류",
                f"프린터 선택 중 오류가 발생했습니다:\n\n{e}"
            )
            return False
        
    def _setup_window(self):
        """윈도우 기본 설정"""
        self.setWindowTitle(f"{AppConstants.APP_NAME} - 수동 마스킹 지원")
        
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
        
        self.setStyleSheet(get_app_style())
    
    def _connect_signals(self):
        """시그널 연결 - 수동 마스킹 기능 포함"""
        components = self.ui.components
        
        # 파일 선택
        components['file_panel'].front_btn.clicked.connect(self.select_front_image)
        components['file_panel'].back_btn.clicked.connect(self.select_back_image)
        
        # 이미지 처리
        components['processing_panel'].process_requested.connect(self.process_images)
        components['processing_panel'].export_requested.connect(self.export_results)
        
        # 인쇄 모드 - 양면인쇄 시그널 추가
        components['print_mode_panel'].mode_changed.connect(self.on_print_mode_changed)
        components['print_mode_panel'].dual_side_changed.connect(self.on_dual_side_toggled)
        
        # 인쇄 매수
        components['print_quantity_panel'].quantity_changed.connect(self.on_print_quantity_changed)
        
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

    def select_front_image(self):
        """앞면 이미지 선택"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "앞면 이미지 선택",
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
        
        # 앞면 이미지 설정
        self.front_image_path = file_path
        file_name, file_size_mb = self.file_manager.get_file_info(file_path)
        
        # UI 업데이트
        self.ui.components['file_panel'].update_front_file_info(file_path)
        self.log(f"앞면 이미지 선택: {file_name} ({file_size_mb:.1f}MB)")
        
        # ImageViewer에 이미지 설정
        self.ui.components['front_original_viewer'].set_image(file_path)
        
        # OpenCV로 읽기 (기존 로직 유지)
        try:
            self.front_original_image = cv2.imread(file_path)
            if self.front_original_image is None:
                with open(file_path, 'rb') as f:
                    image_data = f.read()
                nparr = np.frombuffer(image_data, np.uint8)
                self.front_original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"[DEBUG] OpenCV 이미지 로드 실패: {e}")
            self.front_original_image = None
        
        self._update_ui_state()
        self._reset_processing_results()
        
    def select_back_image(self):
        """뒷면 이미지 선택"""
        if not self.is_dual_side:
            return
            
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "뒷면 이미지 선택",
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
        
        # 뒷면 이미지 설정
        self.back_image_path = file_path
        file_name, file_size_mb = self.file_manager.get_file_info(file_path)
        
        # UI 업데이트
        self.ui.components['file_panel'].update_back_file_info(file_path)
        self.log(f"뒷면 이미지 선택: {file_name} ({file_size_mb:.1f}MB)")
        
        # ImageViewer에 이미지 설정
        self.ui.components['back_original_viewer'].set_image(file_path)
        
        # OpenCV로 읽기
        try:
            self.back_original_image = cv2.imread(file_path)
            if self.back_original_image is None:
                with open(file_path, 'rb') as f:
                    image_data = f.read()
                nparr = np.frombuffer(image_data, np.uint8)
                self.back_original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"[DEBUG] 뒷면 OpenCV 이미지 로드 실패: {e}")
            self.back_original_image = None
        
        self._update_ui_state()

    def on_dual_side_toggled(self, checked):
        """양면 인쇄 토글 - 인쇄모드 패널에서 이동"""
        self.is_dual_side = checked
        
        # 파일선택 패널에 양면 상태 전달
        self.ui.components['file_panel'].set_dual_side_enabled(checked)
        
        if not checked:
            # 단면 모드로 변경 시 뒷면 데이터 초기화
            self.back_image_path = None
            self.back_original_image = None
            self.back_auto_mask_image = None
            self.back_manual_mask_path = None
            self.back_manual_mask_image = None
            self.back_final_mask_image = None
            self.ui.components['back_original_viewer'].clear_image()
            self.ui.components['back_auto_result_viewer'].clear_image()
            self.ui.components['back_manual_mask_viewer'].clear_image()
        
        # 인쇄 버튼 텍스트 업데이트
        self.ui.components['printer_panel'].update_print_button_text(
            self.print_mode, checked, self.print_quantity
        )
        
        mode_text = "양면 인쇄" if checked else "단면 인쇄"
        self.log(f"인쇄 방식 변경: {mode_text}")
        
        self._update_ui_state()
    
    def on_print_mode_changed(self, mode):
        """인쇄 모드 변경"""
        self.print_mode = mode
        self.ui.components['printer_panel'].update_print_button_text(
            mode, self.is_dual_side, self.print_quantity
        )
        self._update_print_button_state()
        
        mode_text = '일반 인쇄' if mode == 'normal' else '레이어 인쇄(YMCW)'
        self.log(f"인쇄 모드 변경: {mode_text}")
    
    def on_print_quantity_changed(self, quantity):
        """인쇄 매수 변경"""
        self.print_quantity = quantity
        self.ui.components['printer_panel'].update_print_button_text(
            self.print_mode, self.is_dual_side, quantity
        )
        
        self.log(f"인쇄 매수 변경: {quantity}장")
        
        if self.front_image_path:
            if quantity > 1:
                status = f"{'양면' if self.is_dual_side else '단면'} {quantity}장 인쇄 준비 완료"
            else:
                status = f"{'양면' if self.is_dual_side else '단면'} 인쇄 준비 완료"
            self.ui.components['status_text'].setText(status)
    
    def on_manual_mask_uploaded(self, file_path: str, is_front: bool):
        """수동 마스킹 이미지 업로드 처리"""
        try:
            # 이미지 유효성 검사
            is_valid, error_msg = self.image_processor.validate_image(file_path)
            if not is_valid:
                QMessageBox.warning(self, "경고", f"마스킹 이미지 오류: {error_msg}")
                return
            
            # 마스킹 이미지 로드
            mask_image = cv2.imread(file_path)
            if mask_image is None:
                with open(file_path, 'rb') as f:
                    image_data = f.read()
                nparr = np.frombuffer(image_data, np.uint8)
                mask_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if mask_image is None:
                QMessageBox.warning(self, "경고", "마스킹 이미지를 읽을 수 없습니다.")
                return
            
            side_text = "앞면" if is_front else "뒷면"
            file_name = os.path.basename(file_path)
            
            if is_front:
                self.front_manual_mask_path = file_path
                self.front_manual_mask_image = mask_image
                self.front_final_mask_image = mask_image  # 수동 마스킹을 최종으로 설정
                
                self.log(f"✅ {side_text} 수동 마스킹 업로드: {file_name}")
                self.log(f"   {side_text} 마스킹 이미지가 최종 결과로 설정되었습니다.")
            else:
                self.back_manual_mask_path = file_path
                self.back_manual_mask_image = mask_image
                self.back_final_mask_image = mask_image
                
                self.log(f"✅ {side_text} 수동 마스킹 업로드: {file_name}")
                self.log(f"   {side_text} 마스킹 이미지가 최종 결과로 설정되었습니다.")
            
            # UI 상태 업데이트
            self._update_ui_state()
            self._update_print_button_state()
            
        except Exception as e:
            side_text = "앞면" if is_front else "뒷면"
            error_msg = f"{side_text} 수동 마스킹 업로드 실패: {e}"
            self.log(f"❌ {error_msg}")
            QMessageBox.critical(self, "업로드 오류", error_msg)
    
    def _update_ui_state(self):
        """UI 상태 업데이트"""
        # 처리 버튼 활성화 조건
        can_process = self.front_image_path is not None
        
        self.ui.components['processing_panel'].set_process_enabled(can_process)
        self._update_print_button_state()
        
        # 저장 버튼 활성화 조건 (자동 또는 수동 마스킹이 있으면)
        has_front_result = (self.front_auto_mask_image is not None or 
                           self.front_manual_mask_image is not None)
        has_back_result = (not self.is_dual_side or 
                          self.back_auto_mask_image is not None or 
                          self.back_manual_mask_image is not None)
        
        can_export = has_front_result and has_back_result
        self.ui.components['processing_panel'].set_export_enabled(can_export)
        
        # 상태 메시지 업데이트
        if can_process:
            if self.print_quantity > 1:
                status = f"{'양면' if self.is_dual_side else '단면'} {self.print_quantity}장 인쇄 준비 완료"
            elif self.is_dual_side:
                status = "양면 인쇄 준비 완료" if self.back_image_path else "양면 인쇄 준비 (뒷면 이미지 선택사항)"
            else:
                status = "단면 인쇄 준비 완료"
            self.ui.components['status_text'].setText(status)
        else:
            self.ui.components['status_text'].setText("앞면 이미지를 선택해주세요")
    
    def _reset_processing_results(self):
        """처리 결과 초기화"""
        # 자동 배경제거 결과 초기화
        self.front_auto_mask_image = None
        self.back_auto_mask_image = None
        
        # 수동 마스킹은 유지 (사용자가 직접 업로드한 것이므로)
        # self.front_manual_mask_image = None  # 주석 처리
        # self.back_manual_mask_image = None   # 주석 처리
        
        # 최종 마스킹 이미지 초기화 (수동이 있으면 수동 유지)
        self.front_final_mask_image = self.front_manual_mask_image
        self.back_final_mask_image = self.back_manual_mask_image
        
        # 프린터용 저장 경로 초기화
        self.front_saved_mask_path = None
        self.back_saved_mask_path = None
        
        # 자동 배경제거 뷰어만 클리어 (수동 마스킹 뷰어는 유지)
        self.ui.components['front_auto_result_viewer'].clear_image()
        self.ui.components['back_auto_result_viewer'].clear_image()
        
        self._update_print_button_state()
    
    def process_images(self):
        """이미지 처리 시작 - 자동 배경제거"""
        if not self.front_image_path:
            return
        
        # UI 상태 변경
        self.ui.components['processing_panel'].set_process_enabled(False)
        self.ui.components['progress_panel'].show_progress()
        
        # 회전된 이미지 처리
        front_viewer = self.ui.components['front_original_viewer']
        current_front_image = front_viewer.get_current_image_array()
        
        if current_front_image is not None:
            temp_dir = tempfile.gettempdir()
            temp_front_path = os.path.join(temp_dir, f"temp_front_{int(time.time())}.jpg")
            cv2.imwrite(temp_front_path, current_front_image)
            
            self.log("앞면 이미지 자동 배경 제거 시작... (회전 적용됨)")
            self.processing_thread = ProcessingThread(temp_front_path, self.image_processor)
        else:
            self.log("앞면 이미지 자동 배경 제거 시작...")
            self.processing_thread = ProcessingThread(self.front_image_path, self.image_processor)
        
        self.processing_thread.finished.connect(self.on_front_processing_finished)
        self.processing_thread.error.connect(self.on_processing_error)
        self.processing_thread.progress.connect(self.on_processing_progress)
        self.processing_thread.start()
    
    def on_processing_progress(self, message):
        """처리 진행상황 업데이트"""
        self.ui.components['progress_panel'].update_status(message)
        self.log(message)
    
    def on_front_processing_finished(self, mask_array):
        """앞면 자동 배경제거 완료"""
        self.front_auto_mask_image = mask_array
        # 자동 배경제거 결과를 자동 결과 뷰어에 표시
        self.ui.components['front_auto_result_viewer'].set_image(mask_array)
        
        # 수동 마스킹이 없으면 자동 결과를 최종으로 설정
        if self.front_manual_mask_image is None:
            self.front_final_mask_image = mask_array
        
        self.log("✅ 앞면 자동 배경 제거 완료!")
        
        # 뒷면 이미지가 있고 양면 모드인 경우 뒷면도 처리
        if self.is_dual_side and self.back_image_path:
            back_viewer = self.ui.components['back_original_viewer']
            current_back_image = back_viewer.get_current_image_array()
            
            if current_back_image is not None:
                temp_dir = tempfile.gettempdir()
                temp_back_path = os.path.join(temp_dir, f"temp_back_{int(time.time())}.jpg")
                cv2.imwrite(temp_back_path, current_back_image)
                
                self.log("뒷면 이미지 자동 배경 제거 시작... (회전 적용됨)")
                self.back_processing_thread = ProcessingThread(temp_back_path, self.image_processor)
            else:
                self.log("뒷면 이미지 자동 배경 제거 시작...")
                self.back_processing_thread = ProcessingThread(self.back_image_path, self.image_processor)
            
            self.back_processing_thread.finished.connect(self.on_back_processing_finished)
            self.back_processing_thread.error.connect(self.on_processing_error)
            self.back_processing_thread.progress.connect(self.on_processing_progress)
            self.back_processing_thread.start()
        else:
            self.on_all_processing_finished()
    
    def on_back_processing_finished(self, mask_array):
        """뒷면 자동 배경제거 완료"""
        self.back_auto_mask_image = mask_array
        # 자동 배경제거 결과를 자동 결과 뷰어에 표시
        self.ui.components['back_auto_result_viewer'].set_image(mask_array)
        
        # 수동 마스킹이 없으면 자동 결과를 최종으로 설정
        if self.back_manual_mask_image is None:
            self.back_final_mask_image = mask_array
        
        self.log("✅ 뒷면 자동 배경 제거 완료!")
        
        self.on_all_processing_finished()
    
    def on_all_processing_finished(self):
        """모든 이미지 처리 완료"""
        # UI 상태 업데이트
        self.ui.components['progress_panel'].hide_progress()
        self.ui.components['processing_panel'].set_process_enabled(True)
        
        self._update_ui_state()
        self._update_print_button_state()
        
        # 사용자에게 수동 마스킹 옵션 안내
        self.log("✅ 자동 배경 제거 완료!")
        self.log("💡 결과가 만족스럽지 않다면 수동 마스킹 영역을 클릭하여 직접 마스킹된 이미지를 업로드하세요.")
        self.ui.components['status_text'].setText("자동 처리 완료 | 수동 마스킹 업로드 가능")
    
    def on_processing_error(self, error_message):
        """처리 오류"""
        self.ui.components['progress_panel'].hide_progress()
        self.ui.components['processing_panel'].set_process_enabled(True)
        
        self.log(f"❌ 처리 오류: {error_message}")
        self.ui.components['status_text'].setText("오류 발생 | 다시 시도해주세요")
        
        QMessageBox.critical(self, "처리 오류", f"이미지 처리 중 오류가 발생했습니다:\n\n{error_message}")
    
    def export_results(self):
        """결과 저장 - 최종 마스킹 이미지 사용"""
        if self.front_final_mask_image is None:
            QMessageBox.warning(self, "경고", "저장할 결과가 없습니다.")
            return
        
        # 저장 폴더 선택
        output_dir = config.get('directories.output', 'output')
        folder_path = QFileDialog.getExistingDirectory(self, "저장 폴더 선택", output_dir)
        
        if not folder_path:
            return
        
        try:
            # 합성 이미지 생성
            front_composite = None
            back_composite = None
            
            # 앞면 합성
            front_viewer = self.ui.components['front_original_viewer']
            current_front_image = front_viewer.get_current_image_array()
            
            if current_front_image is not None and self.front_final_mask_image is not None:
                front_composite = self.image_processor.create_composite_preview(
                    current_front_image, self.front_final_mask_image
                )
            elif self.front_original_image is not None and self.front_final_mask_image is not None:
                front_composite = self.image_processor.create_composite_preview(
                    self.front_original_image, self.front_final_mask_image
                )
            
            # 뒷면 합성
            if self.is_dual_side and self.back_final_mask_image is not None:
                back_viewer = self.ui.components['back_original_viewer']
                current_back_image = back_viewer.get_current_image_array()
                
                if current_back_image is not None:
                    back_composite = self.image_processor.create_composite_preview(
                        current_back_image, self.back_final_mask_image
                    )
                elif self.back_original_image is not None:
                    back_composite = self.image_processor.create_composite_preview(
                        self.back_original_image, self.back_final_mask_image
                    )
            
            # 결과 저장
            success, message = self.file_manager.export_dual_results(
                front_image_path=self.front_image_path,
                front_mask_image=self.front_final_mask_image,
                back_image_path=self.back_image_path,
                back_mask_image=self.back_final_mask_image,
                front_composite=front_composite,
                back_composite=back_composite,
                output_folder=folder_path
            )
            
            if success:
                # 어떤 마스킹이 사용되었는지 안내
                front_type = "수동" if self.front_manual_mask_image is not None else "자동"
                back_type = "수동" if self.back_manual_mask_image is not None else "자동" if self.back_final_mask_image is not None else "없음"
                
                self.log(f"✅ {message}")
                self.log(f"   앞면: {front_type} 마스킹, 뒷면: {back_type} 마스킹")
                self.ui.components['status_text'].setText("저장 완료")
                QMessageBox.information(self, "저장 완료", message)
            else:
                self.log(f"❌ {message}")
                QMessageBox.critical(self, "저장 오류", message)
                
        except Exception as e:
            error_msg = f"저장 중 오류: {e}"
            self.log(f"❌ {error_msg}")
            QMessageBox.critical(self, "저장 오류", error_msg)
    
    def _update_print_button_state(self):
        """인쇄 버튼 상태 업데이트"""
        if not self.printer_available or not self.printer_dll_path:
            self.ui.components['printer_panel'].set_print_enabled(False)
            return
        
        if print_manager.get_print_status()['is_printing']:
            self.ui.components['printer_panel'].set_print_enabled(False)
            return
        
        if self.print_mode == "normal":
            # 일반 모드: 앞면 이미지만 있으면 인쇄 가능
            can_print = self.front_image_path is not None
        else:
            # 레이어 모드: 앞면 이미지와 최종 마스킹이 있어야 함
            can_print = (self.front_image_path is not None and self.front_final_mask_image is not None)
        
        self.ui.components['printer_panel'].set_print_enabled(can_print)
    
    def test_printer_connection(self):
        """프린터 연결 테스트"""
        if not self.printer_available or not self.printer_dll_path:
            QMessageBox.warning(self, "경고", "프린터 DLL을 찾을 수 없습니다.")
            return
        
        # 테스트 버튼 비활성화
        self.ui.components['printer_panel'].set_test_enabled(False)
        self.ui.components['printer_panel'].update_status("🔄 프린터 테스트 중...")
        
        from PySide6.QtCore import QTimer, Signal, QObject
        
        class PrinterTestWorker(QObject):
            test_finished = Signal(bool, str)
            
            def __init__(self, dll_path):
                super().__init__()
                self.dll_path = dll_path
            
            def test_connection(self):
                try:
                    from printer.r600_printer import R600Printer
                    
                    with R600Printer(self.dll_path) as printer:
                        printer.set_timeout(3000)
                        printers = printer.enum_printers()
                        
                        if len(printers) > 0:
                            self.test_finished.emit(True, f"프린터 발견: {printers[0]}")
                        else:
                            self.test_finished.emit(False, "프린터를 찾을 수 없습니다.")
                            
                except Exception as e:
                    error_msg = f"프린터 테스트 실패: {str(e)[:100]}"
                    self.test_finished.emit(False, error_msg)
        
        self.test_worker = PrinterTestWorker(self.printer_dll_path)
        self.test_worker.test_finished.connect(self._on_printer_test_finished)
        
        QTimer.singleShot(100, self._start_printer_test)

    def _start_printer_test(self):
        """프린터 테스트 시작"""
        try:
            import threading
            test_thread = threading.Thread(
                target=self.test_worker.test_connection,
                daemon=True
            )
            test_thread.start()
            
        except Exception as e:
            self._on_printer_test_finished(False, f"테스트 시작 실패: {e}")

    def _on_printer_test_finished(self, success: bool, message: str):
        """프린터 테스트 결과 처리"""
        try:
            self.ui.components['printer_panel'].set_test_enabled(True)
            
            if success:
                self.log(f"✅ {message}")
                self.ui.components['printer_panel'].update_status("✅ 프린터 연결 가능")
                self.ui.components['status_text'].setText("프린터 테스트 성공")
            else:
                self.log(f"❌ {message}")
                self.ui.components['printer_panel'].update_status("❌ 프린터 연결 실패")
                QMessageBox.warning(
                    self, 
                    "프린터 테스트 실패", 
                    f"프린터 연결을 확인할 수 없습니다.\n\n{message}\n\n"
                    "프린터가 켜져 있고 네트워크에 연결되어 있는지 확인해주세요."
                )
            
            if hasattr(self, 'test_worker'):
                delattr(self, 'test_worker')
                
        except Exception as e:
            self.log(f"❌ 테스트 결과 처리 오류: {e}")
            self.ui.components['printer_panel'].set_test_enabled(True)
            self.ui.components['printer_panel'].update_status("❌ 테스트 오류")
        
    def print_card(self):
        """카드 인쇄 - 최종 마스킹 이미지 사용"""
        if not self.printer_available or not self.printer_dll_path:
            QMessageBox.warning(self, "경고", "프린터를 사용할 수 없습니다.")
            return
        
        if not self.front_image_path:
            QMessageBox.warning(self, "경고", "앞면 이미지를 먼저 선택해주세요.")
            return
        
        if print_manager.get_print_status()['is_printing']:
            QMessageBox.warning(self, "경고", "이미 인쇄가 진행 중입니다.")
            return
        
        # 인쇄 모드별 확인
        if self.print_mode == "layered":
            if self.front_final_mask_image is None:
                QMessageBox.warning(self, "경고", "레이어 인쇄를 위해서는 마스킹 이미지가 필요합니다.\n자동 배경제거를 실행하거나 수동 마스킹을 업로드해주세요.")
                return
            
            # 최종 마스킹 이미지 저장 (자동 또는 수동)
            self.front_saved_mask_path = self.file_manager.save_mask_for_printing(
                self.front_final_mask_image, self.front_image_path, "front"
            )
            if not self.front_saved_mask_path:
                QMessageBox.critical(self, "오류", "앞면 마스크 이미지 저장에 실패했습니다.")
                return
            
            # 뒷면 마스크도 저장 (있는 경우)
            if self.is_dual_side and self.back_final_mask_image is not None and self.back_image_path:
                self.back_saved_mask_path = self.file_manager.save_mask_for_printing(
                    self.back_final_mask_image, self.back_image_path, "back"
                )
                if not self.back_saved_mask_path:
                    self.log("⚠️ 뒷면 마스크 저장 실패, 뒷면은 일반 모드로 인쇄됩니다.")
        
        # 회전된 이미지를 고려한 인쇄 경로 준비
        front_print_path = self.front_image_path
        back_print_path = self.back_image_path
        
        # 앞면이 회전되었다면 임시 파일로 저장
        front_viewer = self.ui.components['front_original_viewer']
        current_front_image = front_viewer.get_current_image_array()
        if current_front_image is not None and front_viewer.get_rotation_angle() != 0:
            temp_dir = tempfile.gettempdir()
            front_print_path = os.path.join(temp_dir, f"print_front_{int(time.time())}.jpg")
            cv2.imwrite(front_print_path, current_front_image)
            self.log(f"앞면 이미지 회전 적용됨 ({front_viewer.get_rotation_angle()}도)")
        
        # 뒷면이 회전되었다면 임시 파일로 저장
        if self.is_dual_side and self.back_image_path:
            back_viewer = self.ui.components['back_original_viewer']
            current_back_image = back_viewer.get_current_image_array()
            if current_back_image is not None and back_viewer.get_rotation_angle() != 0:
                temp_dir = tempfile.gettempdir()
                back_print_path = os.path.join(temp_dir, f"print_back_{int(time.time())}.jpg")
                cv2.imwrite(back_print_path, current_back_image)
                self.log(f"뒷면 이미지 회전 적용됨 ({back_viewer.get_rotation_angle()}도)")
        
        # 인쇄 확인 다이얼로그
        mode_text = "일반 인쇄" if self.print_mode == "normal" else "레이어 인쇄 (YMCW)"
        side_text = "양면" if self.is_dual_side else "단면"
        
        front_name, _ = self.file_manager.get_file_info(self.front_image_path)
        detail_text = f"앞면 이미지: {front_name}\n"
        
        # 마스킹 정보 추가
        if self.print_mode == "layered":
            front_mask_type = "수동 마스킹" if self.front_manual_mask_image is not None else "자동 마스킹"
            detail_text += f"  마스킹: {front_mask_type}\n"
        
        # 회전 정보 추가
        if front_viewer.get_rotation_angle() != 0:
            detail_text += f"  (회전: {front_viewer.get_rotation_angle()}도)\n"
        
        if self.is_dual_side and self.back_image_path:
            back_name, _ = self.file_manager.get_file_info(self.back_image_path)
            detail_text += f"뒷면 이미지: {back_name}\n"
            
            # 뒷면 마스킹 정보
            if self.print_mode == "layered" and self.back_final_mask_image is not None:
                back_mask_type = "수동 마스킹" if self.back_manual_mask_image is not None else "자동 마스킹"
                detail_text += f"  마스킹: {back_mask_type}\n"
            
            # 뒷면 회전 정보
            back_viewer = self.ui.components['back_original_viewer']
            if back_viewer.get_rotation_angle() != 0:
                detail_text += f"  (회전: {back_viewer.get_rotation_angle()}도)\n"
        elif self.is_dual_side:
            detail_text += "뒷면 이미지: 없음 (빈 뒷면으로 인쇄)\n"
        
        detail_text += f"인쇄 방식: {side_text} {mode_text}\n"
        detail_text += f"인쇄 매수: {self.print_quantity}장"
        
        # 예상 시간 계산
        estimated_minutes = (self.print_quantity * 30) // 60
        estimated_seconds = (self.print_quantity * 30) % 60
        if estimated_minutes > 0:
            time_text = f"예상 시간: 약 {estimated_minutes}분 {estimated_seconds}초"
        else:
            time_text = f"예상 시간: 약 {self.print_quantity * 30}초"
        
        reply = QMessageBox.question(
            self,
            "카드 인쇄",
            f"카드 인쇄를 시작하시겠습니까?\n\n{detail_text}\n{time_text}\n\n"
            "프린터에 충분한 카드가 준비되어 있는지 확인해주세요.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        # 인쇄 시작
        self._start_multi_print(front_print_path, back_print_path)
    
    def _start_multi_print(self, front_path=None, back_path=None):
        """여러장 인쇄 시작"""
        try:
            self.ui.components['printer_panel'].set_print_enabled(False)
            self.ui.components['progress_panel'].show_progress()
            
            if front_path is None:
                front_path = self.front_image_path
            if back_path is None:
                back_path = self.back_image_path
            
            # 프린터 스레드 시작
            self.current_printer_thread = print_manager.start_multi_print(
                dll_path=self.printer_dll_path,
                front_image_path=front_path,
                back_image_path=back_path,
                front_mask_path=self.front_saved_mask_path if self.print_mode == "layered" else None,
                back_mask_path=self.back_saved_mask_path if self.print_mode == "layered" else None,
                print_mode=self.print_mode,
                is_dual_side=self.is_dual_side,
                quantity=self.print_quantity
            )
            
            # 시그널 연결
            self.current_printer_thread.progress.connect(self.on_printer_progress)
            self.current_printer_thread.finished.connect(self.on_printer_finished)
            self.current_printer_thread.error.connect(self.on_printer_error)
            self.current_printer_thread.print_progress.connect(self.on_print_progress)
            self.current_printer_thread.card_completed.connect(self.on_card_completed)
            
            self.current_printer_thread.start()
            
            self.log(f"📄 {self.print_quantity}장 인쇄 시작!")
            
        except Exception as e:
            self.ui.components['progress_panel'].hide_progress()
            self.ui.components['printer_panel'].set_print_enabled(True)
            error_msg = f"인쇄 시작 실패: {e}"
            self.log(f"❌ {error_msg}")
            QMessageBox.critical(self, "인쇄 오류", error_msg)
    
    def on_printer_progress(self, message):
        """프린터 진행상황 업데이트"""
        self.ui.components['progress_panel'].update_status(message)
        self.log(message)
    
    def on_print_progress(self, current, total):
        """인쇄 진행률 업데이트"""
        self.ui.components['progress_panel'].update_print_status(current, total, f"📄 {current}/{total} 장 인쇄 중...")
    
    def on_card_completed(self, card_num):
        """개별 카드 완료"""
        self.log(f"✅ {card_num}번째 카드 인쇄 완료!")
        
        if card_num < self.print_quantity:
            self.ui.components['status_text'].setText(f"인쇄 진행 중: {card_num}/{self.print_quantity} 완료")
    
    def on_printer_finished(self, success):
        """프린터 작업 완료"""
        self.ui.components['progress_panel'].hide_progress()
        self.ui.components['printer_panel'].set_print_enabled(True)
        
        mode_text = "일반 인쇄" if self.print_mode == "normal" else "레이어 인쇄"
        side_text = "양면" if self.is_dual_side else "단면"
        
        if success:
            # 사용된 마스킹 타입 정보 추가
            mask_info = ""
            if self.print_mode == "layered":
                front_mask_type = "수동" if self.front_manual_mask_image is not None else "자동"
                back_mask_type = "수동" if self.back_manual_mask_image is not None else "자동" if self.back_final_mask_image is not None else "없음"
                mask_info = f" (앞면: {front_mask_type}, 뒷면: {back_mask_type})"
            
            self.log(f"✅ {side_text} {mode_text} {self.print_quantity}장 완료!{mask_info}")
            self.ui.components['status_text'].setText("인쇄 완료")
            QMessageBox.information(self, "성공", f"{side_text} {mode_text} {self.print_quantity}장이 완료되었습니다!")
        else:
            self.log(f"❌ {side_text} {mode_text} 실패")
            self.ui.components['status_text'].setText("인쇄 실패")
        
        self._update_print_button_state()
    
    def on_printer_error(self, error_message):
        """프린터 오류 처리"""
        self.ui.components['progress_panel'].hide_progress()
        self.ui.components['printer_panel'].set_print_enabled(True)
        
        self.log(f"❌ 프린터 오류: {error_message}")
        self.ui.components['status_text'].setText("인쇄 오류 발생")
        QMessageBox.critical(self, "인쇄 오류", f"카드 인쇄 중 오류가 발생했습니다:\n\n{error_message}")
        
        self._update_print_button_state()
    
    def log(self, message):
        """로그 메시지 추가"""
        self.ui.components['log_panel'].add_log(message)
    
    def closeEvent(self, event):
        """애플리케이션 종료 시"""
        # 진행 중인 인쇄 중단
        if print_manager.get_print_status()['is_printing']:
            reply = QMessageBox.question(
                self,
                "인쇄 진행 중",
                "인쇄가 진행 중입니다. 프로그램을 종료하시겠습니까?\n인쇄가 중단될 수 있습니다.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                print_manager.stop_current_print()
            else:
                event.ignore()
                return
        
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