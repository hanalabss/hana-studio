"""
Hana Studio 메인 애플리케이션 클래스 - final_preview_viewer 제거로 인한 수정
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
from printer.printer_thread import print_manager
from config import config, AppConstants


class HanaStudio(QMainWindow):
    """Hana Studio 메인 애플리케이션 클래스 - 양면 인쇄 및 여러장 인쇄 지원"""
    
    def __init__(self):
        super().__init__()
        
        # 데이터 속성들
        self.front_image_path = None
        self.back_image_path = None
        self.front_original_image = None
        self.back_original_image = None
        self.front_mask_image = None
        self.back_mask_image = None
        self.front_saved_mask_path = None
        self.back_saved_mask_path = None
        self.print_mode = "normal"
        self.is_dual_side = False
        self.print_quantity = 1  # 인쇄 매수 추가
        
        # 코어 모듈들
        self.image_processor = ImageProcessor()
        self.file_manager = FileManager()
        
        # 프린터 관련
        self.printer_available = False
        self.printer_dll_path = None
        self.current_printer_thread = None
        
        # UI 초기화
        self.ui = HanaStudioMainWindow(self)
        self._setup_window()
        self._connect_signals()
        self._check_printer_availability()
        
    def _setup_window(self):
        """윈도우 기본 설정"""
        self.setWindowTitle(f"{AppConstants.APP_NAME} - 양면 및 여러장 카드 인쇄 지원")
        
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
        components['file_panel'].front_btn.clicked.connect(self.select_front_image)
        components['file_panel'].back_btn.clicked.connect(self.select_back_image)
        components['file_panel'].dual_side_check.toggled.connect(self.on_dual_side_toggled)
        
        # 이미지 처리
        components['processing_panel'].process_requested.connect(self.process_images)
        components['processing_panel'].export_requested.connect(self.export_results)
        
        # 인쇄 모드
        components['print_mode_panel'].mode_changed.connect(self.on_print_mode_changed)
        
        # 인쇄 매수 - 새로 추가
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
        
        self.ui.components['file_panel'].update_front_file_info(file_path)
        self.log(f"앞면 이미지 선택: {file_name} ({file_size_mb:.1f}MB)")
        
        # 앞면 이미지 표시
        self.ui.components['front_original_viewer'].set_image(file_path)
        self.front_original_image = cv2.imread(file_path)
        
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
        
        self.ui.components['file_panel'].update_back_file_info(file_path)
        self.log(f"뒷면 이미지 선택: {file_name} ({file_size_mb:.1f}MB)")
        
        # 뒷면 이미지 표시
        self.ui.components['back_original_viewer'].set_image(file_path)
        self.back_original_image = cv2.imread(file_path)
        
        self._update_ui_state()
    
    def on_dual_side_toggled(self, checked):
        """양면 인쇄 토글"""
        self.is_dual_side = checked
        
        if not checked:
            # 단면 모드로 변경 시 뒷면 데이터 초기화
            self.back_image_path = None
            self.back_original_image = None
            self.back_mask_image = None
            self.ui.components['back_original_viewer'].clear_image()
            self.ui.components['back_result_viewer'].clear_image()
        
        # 인쇄 모드 패널에 양면/단면 상태 전달
        self.ui.components['print_mode_panel'].update_dual_side_status(checked)
        
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
        """인쇄 매수 변경 - 새로 추가"""
        self.print_quantity = quantity
        self.ui.components['printer_panel'].update_print_button_text(
            self.print_mode, self.is_dual_side, quantity
        )
        
        self.log(f"인쇄 매수 변경: {quantity}장")
        
        # 상태 메시지 업데이트
        if self.front_image_path:
            if quantity > 1:
                status = f"{'양면' if self.is_dual_side else '단면'} {quantity}장 인쇄 준비 완료"
            else:
                status = f"{'양면' if self.is_dual_side else '단면'} 인쇄 준비 완료"
            self.ui.components['status_text'].setText(status)
    
    def _update_ui_state(self):
        """UI 상태 업데이트"""
        # 처리 버튼 활성화 조건
        can_process = self.front_image_path is not None
        if self.is_dual_side:
            # 양면 모드에서는 뒷면 이미지는 선택사항 (앞면만 있어도 처리 가능)
            pass
        
        self.ui.components['processing_panel'].set_process_enabled(can_process)
        self._update_print_button_state()
        
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
        """처리 결과 초기화 - final_preview_viewer 제거"""
        self.front_mask_image = None
        self.back_mask_image = None
        self.front_saved_mask_path = None
        self.back_saved_mask_path = None
        
        self.ui.components['processing_panel'].set_export_enabled(False)
        self.ui.components['front_result_viewer'].clear_image()
        self.ui.components['back_result_viewer'].clear_image()
        
        # final_preview_viewer는 제거되었으므로 호출하지 않음
        
        self._update_print_button_state()
    
    def process_images(self):
        """이미지 처리 시작"""
        if not self.front_image_path:
            return
        
        # UI 상태 변경
        self.ui.components['processing_panel'].set_process_enabled(False)
        self.ui.components['progress_panel'].show_progress()
        
        # 앞면 이미지 처리 시작
        self.log("앞면 이미지 배경 제거 시작...")
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
        """앞면 처리 완료"""
        self.front_mask_image = mask_array
        self.ui.components['front_result_viewer'].set_image(mask_array)
        self.log("✅ 앞면 배경 제거 완료!")
        
        # 뒷면 이미지가 있고 양면 모드인 경우 뒷면도 처리
        if self.is_dual_side and self.back_image_path:
            self.log("뒷면 이미지 배경 제거 시작...")
            self.back_processing_thread = ProcessingThread(self.back_image_path, self.image_processor)
            self.back_processing_thread.finished.connect(self.on_back_processing_finished)
            self.back_processing_thread.error.connect(self.on_processing_error)
            self.back_processing_thread.progress.connect(self.on_processing_progress)
            self.back_processing_thread.start()
        else:
            # 뒷면 처리가 없으면 바로 완료 처리
            self.on_all_processing_finished()
    
    def on_back_processing_finished(self, mask_array):
        """뒷면 처리 완료"""
        self.back_mask_image = mask_array
        self.ui.components['back_result_viewer'].set_image(mask_array)
        self.log("✅ 뒷면 배경 제거 완료!")
        
        # 모든 처리 완료
        self.on_all_processing_finished()
    
    def on_all_processing_finished(self):
        """모든 이미지 처리 완료 - final_preview 제거"""
        # UI 상태 업데이트 (final_preview 생성 제거)
        self.ui.components['progress_panel'].hide_progress()
        self.ui.components['processing_panel'].set_process_enabled(True)
        self.ui.components['processing_panel'].set_export_enabled(True)
        
        self._update_print_button_state()
        
        self.log("✅ 모든 이미지 처리 완료!")
        self.ui.components['status_text'].setText("처리 완료 | 결과 저장 및 인쇄 가능")
    
    def on_processing_error(self, error_message):
        """처리 오류"""
        self.ui.components['progress_panel'].hide_progress()
        self.ui.components['processing_panel'].set_process_enabled(True)
        
        self.log(f"❌ 처리 오류: {error_message}")
        self.ui.components['status_text'].setText("오류 발생 | 다시 시도해주세요")
        
        QMessageBox.critical(self, "처리 오류", f"이미지 처리 중 오류가 발생했습니다:\n\n{error_message}")
    
    def export_results(self):
        """결과 저장 - 양면 지원"""
        if self.front_mask_image is None:
            QMessageBox.warning(self, "경고", "저장할 결과가 없습니다.")
            return
        
        # 저장 폴더 선택
        output_dir = config.get('directories.output', 'output')
        folder_path = QFileDialog.getExistingDirectory(self, "저장 폴더 선택", output_dir)
        
        if not folder_path:
            return
        
        try:
            # 합성 이미지 생성 (필요시)
            front_composite = None
            back_composite = None
            
            if self.front_original_image is not None and self.front_mask_image is not None:
                front_composite = self.image_processor.create_composite_preview(
                    self.front_original_image, self.front_mask_image
                )
            
            if (self.back_original_image is not None and self.back_mask_image is not None 
                and self.is_dual_side):
                back_composite = self.image_processor.create_composite_preview(
                    self.back_original_image, self.back_mask_image
                )
            
            # 양면 결과 저장
            success, message = self.file_manager.export_dual_results(
                front_image_path=self.front_image_path,
                front_mask_image=self.front_mask_image,
                back_image_path=self.back_image_path,
                back_mask_image=self.back_mask_image,
                front_composite=front_composite,
                back_composite=back_composite,
                output_folder=folder_path
            )
            
            if success:
                self.log(f"✅ {message}")
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
        
        # 인쇄 중인 경우 비활성화
        if print_manager.get_print_status()['is_printing']:
            self.ui.components['printer_panel'].set_print_enabled(False)
            return
        
        if self.print_mode == "normal":
            # 일반 모드: 앞면 이미지만 있으면 인쇄 가능
            can_print = self.front_image_path is not None
        else:
            # 레이어 모드: 앞면 이미지와 마스크가 있어야 함
            can_print = (self.front_image_path is not None and self.front_mask_image is not None)
        
        self.ui.components['printer_panel'].set_print_enabled(can_print)
    
    def test_printer_connection(self):
        """프린터 연결 테스트 - 크래시 방지 버전"""
        if not self.printer_available or not self.printer_dll_path:
            QMessageBox.warning(self, "경고", "프린터 DLL을 찾을 수 없습니다.")
            return
        
        # 테스트 버튼 비활성화
        self.ui.components['printer_panel'].set_test_enabled(False)
        self.ui.components['printer_panel'].update_status("🔄 프린터 테스트 중...")
        
        # 시그널을 사용해서 메인 스레드에서 메시지박스 표시
        from PySide6.QtCore import QTimer, Signal, QObject
        
        class PrinterTestWorker(QObject):
            """프린터 테스트 전용 워커"""
            test_finished = Signal(bool, str)  # 성공여부, 메시지
            
            def __init__(self, dll_path):
                super().__init__()
                self.dll_path = dll_path
            
            def test_connection(self):
                """실제 프린터 테스트 (스레드에서 실행)"""
                try:
                    # 간단한 프린터 연결 테스트만 수행
                    from printer.r600_printer import R600Printer
                    
                    # 매우 짧은 타임아웃으로 빠른 테스트
                    with R600Printer(self.dll_path) as printer:
                        printer.set_timeout(3000)  # 3초 타임아웃
                        printers = printer.enum_printers()
                        
                        if len(printers) > 0:
                            self.test_finished.emit(True, f"프린터 발견: {printers[0]}")
                        else:
                            self.test_finished.emit(False, "프린터를 찾을 수 없습니다.")
                            
                except Exception as e:
                    error_msg = f"프린터 테스트 실패: {str(e)[:100]}"
                    self.test_finished.emit(False, error_msg)
        
        # 워커 생성 및 시그널 연결
        self.test_worker = PrinterTestWorker(self.printer_dll_path)
        self.test_worker.test_finished.connect(self._on_printer_test_finished)
        
        # 타이머로 스레드 시작 (메인 스레드에서 안전하게)
        QTimer.singleShot(100, self._start_printer_test)

    def _start_printer_test(self):
        """프린터 테스트 시작 (메인 스레드에서 실행)"""
        try:
            import threading
            test_thread = threading.Thread(
                target=self.test_worker.test_connection,
                daemon=True
            )
            test_thread.start()
            
            # 타임아웃 타이머 설정 (10초)
            # QTimer.singleShot(10000, self._on_printer_test_timeout)
            
        except Exception as e:
            self._on_printer_test_finished(False, f"테스트 시작 실패: {e}")

    def _on_printer_test_timeout(self):
        """프린터 테스트 타임아웃"""
        if hasattr(self, 'test_worker'):
            self._on_printer_test_finished(False, "프린터 테스트 시간 초과 (10초)")

    def _on_printer_test_finished(self, success: bool, message: str):
        """프린터 테스트 결과 처리 (메인 스레드에서 실행)"""
        try:
            # UI 상태 복원
            self.ui.components['printer_panel'].set_test_enabled(True)
            
            if success:
                self.log(f"✅ {message}")
                self.ui.components['printer_panel'].update_status("✅ 프린터 연결 가능")
                
                # 성공 시에는 간단한 로그만 출력 (메시지박스 없음)
                self.ui.components['status_text'].setText("프린터 테스트 성공")
                
            else:
                self.log(f"❌ {message}")
                self.ui.components['printer_panel'].update_status("❌ 프린터 연결 실패")
                
                # 실패 시에만 메시지박스 표시 (메인 스레드에서)
                QMessageBox.warning(
                    self, 
                    "프린터 테스트 실패", 
                    f"프린터 연결을 확인할 수 없습니다.\n\n{message}\n\n"
                    "프린터가 켜져 있고 네트워크에 연결되어 있는지 확인해주세요."
                )
            
            # 워커 정리
            if hasattr(self, 'test_worker'):
                delattr(self, 'test_worker')
                
        except Exception as e:
            self.log(f"❌ 테스트 결과 처리 오류: {e}")
            self.ui.components['printer_panel'].set_test_enabled(True)
            self.ui.components['printer_panel'].update_status("❌ 테스트 오류")
        
    def print_card(self):
        """카드 인쇄 - 여러장 지원"""
        if not self.printer_available or not self.printer_dll_path:
            QMessageBox.warning(self, "경고", "프린터를 사용할 수 없습니다.")
            return
        
        if not self.front_image_path:
            QMessageBox.warning(self, "경고", "앞면 이미지를 먼저 선택해주세요.")
            return
        
        # 현재 인쇄 중인지 확인
        if print_manager.get_print_status()['is_printing']:
            QMessageBox.warning(self, "경고", "이미 인쇄가 진행 중입니다.")
            return
        
        # 인쇄 모드별 확인
        if self.print_mode == "layered":
            if self.front_mask_image is None:
                QMessageBox.warning(self, "경고", "레이어 인쇄를 위해서는 배경 제거를 먼저 실행해주세요.")
                return
            
            # 앞면 마스크 이미지 저장
            self.front_saved_mask_path = self.file_manager.save_mask_for_printing(
                self.front_mask_image, self.front_image_path, "front"
            )
            if not self.front_saved_mask_path:
                QMessageBox.critical(self, "오류", "앞면 마스크 이미지 저장에 실패했습니다.")
                return
            
            # 뒷면 마스크도 저장 (있는 경우)
            if self.is_dual_side and self.back_mask_image is not None and self.back_image_path:
                self.back_saved_mask_path = self.file_manager.save_mask_for_printing(
                    self.back_mask_image, self.back_image_path, "back"
                )
                if not self.back_saved_mask_path:
                    self.log("⚠️ 뒷면 마스크 저장 실패, 뒷면은 일반 모드로 인쇄됩니다.")
        
        # 인쇄 확인 다이얼로그
        mode_text = "일반 인쇄" if self.print_mode == "normal" else "레이어 인쇄 (YMCW)"
        side_text = "양면" if self.is_dual_side else "단면"
        
        front_name, _ = self.file_manager.get_file_info(self.front_image_path)
        detail_text = f"앞면 이미지: {front_name}\n"
        
        if self.is_dual_side and self.back_image_path:
            back_name, _ = self.file_manager.get_file_info(self.back_image_path)
            detail_text += f"뒷면 이미지: {back_name}\n"
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
        self._start_multi_print()
    
    def _start_multi_print(self):
        """여러장 인쇄 시작"""
        try:
            # UI 상태 변경
            self.ui.components['printer_panel'].set_print_enabled(False)
            self.ui.components['progress_panel'].show_progress()
            
            # 프린터 스레드 시작
            self.current_printer_thread = print_manager.start_multi_print(
                dll_path=self.printer_dll_path,
                front_image_path=self.front_image_path,
                back_image_path=self.back_image_path,
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
        
        # 상태바 업데이트
        if card_num < self.print_quantity:
            self.ui.components['status_text'].setText(f"인쇄 진행 중: {card_num}/{self.print_quantity} 완료")
    
    def on_printer_finished(self, success):
        """프린터 작업 완료"""
        self.ui.components['progress_panel'].hide_progress()
        self.ui.components['printer_panel'].set_print_enabled(True)
        
        mode_text = "일반 인쇄" if self.print_mode == "normal" else "레이어 인쇄"
        side_text = "양면" if self.is_dual_side else "단면"
        
        if success:
            self.log(f"✅ {side_text} {mode_text} {self.print_quantity}장 완료!")
            self.ui.components['status_text'].setText("인쇄 완료")
            QMessageBox.information(self, "성공", f"{side_text} {mode_text} {self.print_quantity}장이 완료되었습니다!")
        else:
            self.log(f"❌ {side_text} {mode_text} 실패")
            self.ui.components['status_text'].setText("인쇄 실패")
        
        # 인쇄 완료 후 상태 재설정
        self._update_print_button_state()
    
    def on_printer_error(self, error_message):
        """프린터 오류 처리"""
        self.ui.components['progress_panel'].hide_progress()
        self.ui.components['printer_panel'].set_print_enabled(True)
        
        self.log(f"❌ 프린터 오류: {error_message}")
        self.ui.components['status_text'].setText("인쇄 오류 발생")
        QMessageBox.critical(self, "인쇄 오류", f"카드 인쇄 중 오류가 발생했습니다:\n\n{error_message}")
        
        # 오류 후 상태 재설정
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