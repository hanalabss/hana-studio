import sys
import os
import io
import threading
import time
import tempfile
from pathlib import Path

from PySide6.QtWidgets import QMainWindow, QFileDialog, QMessageBox, QApplication
from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QIcon

# 분리된 모듈들 import (가벼운 것들만)
from ui import HanaStudioMainWindow, get_app_style
from config import config, AppConstants, get_resource_path

# [START] 무거운 모듈들은 지연 import로 처리
# import cv2, numpy는 필요할 때만
# core, printer 모듈들도 필요할 때만


class HanaStudio(QMainWindow):
    """Hana Studio 메인 애플리케이션 클래스 - 최적화된 지연 로딩"""

    def __init__(self, is_admin: bool = False):
        super().__init__()

        # Admin 권한 저장
        self.is_admin = is_admin

        print(f"[INIT] Starting HanaStudio initialization (Admin: {is_admin})")
        QApplication.processEvents()

        # [TARGET] 윈도우 아이콘 설정
        try:
            self._setup_window_icon()
        except Exception as e:
            print(f"[WARN] Icon setup failed: {e}")

        QApplication.processEvents()

        # 데이터 속성들 초기화
        self._init_data_attributes()
        QApplication.processEvents()

        # UI 초기화 (가벼운 작업만 - 패널은 나중에 지연 생성됨)
        print("[INIT] Creating UI frame...")
        QApplication.processEvents()
        try:
            self.ui = HanaStudioMainWindow(self)
            QApplication.processEvents()
            self._setup_window()
            QApplication.processEvents()
            # 시그널 연결은 패널 생성 후로 지연
            self._signals_connected = False
            print("[INIT] UI frame created successfully")
        except Exception as e:
            print(f"[ERROR] UI creation failed: {e}")
            raise

        # [START] 무거운 작업들은 나중에 지연 초기화
        self._lazy_init_scheduled = False
        QTimer.singleShot(500, self._lazy_initialize)  # 500ms로 늘림

        self.adjusted_x = 0
        self.adjusted_y = 0

        QApplication.processEvents()
        print("[INIT] HanaStudio initialization complete")
    
    def _init_data_attributes(self):
        """데이터 속성들 초기화"""
        # 데이터 속성들
        self.front_image_path = None
        self.back_image_path = None
        self.front_original_image = None
        self.back_original_image = None
        
        # 자동 배경제거 결과
        self.front_auto_mask_image = None
        self.back_auto_mask_image = None
        
        # 수동 마스킹 이미지
        self.front_manual_mask_path = None
        self.back_manual_mask_path = None
        self.front_manual_mask_image = None
        self.back_manual_mask_image = None
        
        # 프린터용 저장된 마스크 경로
        self.front_saved_mask_path = None
        self.back_saved_mask_path = None
        
        # 개별 면 방향 설정
        self.front_orientation = "portrait"
        self.back_orientation = "portrait"
        
        self.print_mode = "normal"
        self.is_dual_side = False
        self.print_quantity = 1
        self.card_orientation = "portrait"
        
        # [START] 코어 모듈들은 지연 초기화
        self.image_processor = None
        self.file_manager = None
        
        # 프린터 관련
        self.printer_available = False
        self.printer_dll_path = None
        self.current_printer_thread = None
        self.selected_printer_info = None

        # 인쇄 큐 관련
        self._print_queue = None
        self._queue_initialized = False
    
    def _lazy_initialize(self):
        """무거운 모듈들을 지연 초기화"""
        try:
            # 패널이 모두 생성될 때까지 대기
            if not hasattr(self.ui, '_panels_initialized') or not self.ui._panels_initialized:
                # 패널 초기화가 아직 안 끝났으면 다시 예약
                QTimer.singleShot(100, self._lazy_initialize)
                return

            # 뷰어도 모두 생성될 때까지 대기
            if not hasattr(self.ui, '_viewers_initialized') or not self.ui._viewers_initialized:
                QTimer.singleShot(100, self._lazy_initialize)
                return

            # 이미 초기화됐으면 스킵
            if self._lazy_init_scheduled:
                return
            self._lazy_init_scheduled = True

            # 시그널 연결 (패널 생성 완료 후)
            if not self._signals_connected:
                self._connect_signals()
                self._signals_connected = True
                print("[OK] 시그널 연결 완료")

            print("[SYSTEM] 시스템 준비 중...")
            QApplication.processEvents()

            # 무거운 모듈들을 여기서 import (model_loader 제외 - rembg가 무거움)
            from core import ImageProcessor, FileManager

            QApplication.processEvents()
            self.image_processor = ImageProcessor()
            QApplication.processEvents()
            self.file_manager = FileManager()
            QApplication.processEvents()

            print("[SYSTEM] 시스템 준비 완료")

            # 메뉴바 설정
            self._setup_menubar()

            # 프린터 초기화 (더 나중에)
            QTimer.singleShot(500, self._lazy_init_printer)

            # AI 모델 로딩 시그널 연결은 첫 사용 시 수행
            QTimer.singleShot(2000, self._connect_model_loader_signals)

        except Exception as e:
            print(f"[ERROR] 시스템 준비 실패: {e}")

    def _connect_model_loader_signals(self):
        """AI 모델 로더 시그널 연결 (별도 스레드에서 import)"""
        import threading

        def connect_signals():
            try:
                from core.model_loader import get_model_loader
                model_loader = get_model_loader()
                # 시그널 연결은 메인 스레드에서 해야 함
                QTimer.singleShot(0, lambda: self._setup_model_signals(model_loader))
            except Exception as e:
                print(f"[WARN] AI 모델 로더 연결 실패: {e}")

        thread = threading.Thread(target=connect_signals, daemon=True)
        thread.start()

    def _setup_model_signals(self, model_loader):
        """메인 스레드에서 시그널 연결 및 자동 선로딩"""
        try:
            # 부모 위젯 설정 (다운로드 다이얼로그용)
            model_loader.set_parent_widget(self)

            model_loader.loading_progress.connect(self._on_model_loading_progress)
            model_loader.loading_completed.connect(self._on_model_loading_completed)
            model_loader.loading_failed.connect(self._on_model_loading_failed)
            print("[AI] 모델 로더 시그널 연결 완료")

            # ✨ UI 준비 완료 후 자동으로 AI 모델 백그라운드 선로딩 시작
            if not model_loader.is_loaded and not model_loader.is_loading:
                print("[AI] UI 준비 완료 - AI 모델 백그라운드 선로딩 시작...")
                model_loader.start_background_loading()
            else:
                print(f"[AI] 모델 상태: is_loaded={model_loader.is_loaded}, is_loading={model_loader.is_loading}")

        except Exception as e:
            print(f"[WARN] 시그널 연결 실패: {e}")
    
    def _lazy_init_printer(self):
        """프린터 관련 지연 초기화"""
        try:
            print("🖨️ 프린터 연결 준비 중...")

            from printer import find_printer_dll
            from printer.printer_thread import print_manager

            # 프린터 DLL 확인
            self.printer_dll_path = find_printer_dll()
            if self.printer_dll_path:
                print(f"[OK] 프린터 DLL 발견: {self.printer_dll_path}")
                # 자동으로 프린터 선택 대화상자 표시
                QTimer.singleShot(1000, self._auto_show_printer_dialog)
            else:
                print("[WARN] 프린터 DLL을 찾을 수 없음")

            self._setup_manual_mask_viewers()

            # 인쇄 큐 초기화
            self._init_print_queue()

            print("[OK] 프린터 초기화 완료")

        except Exception as e:
            print(f"[ERROR] 프린터 초기화 실패: {e}")

    def _init_print_queue(self):
        """인쇄 큐 시스템 초기화"""
        if self._queue_initialized:
            return

        try:
            from printer.print_queue import print_queue
            self._print_queue = print_queue

            # 큐 시그널 연결
            self._print_queue.job_added.connect(self._on_queue_job_added)
            self._print_queue.job_started.connect(self._on_queue_job_started)
            self._print_queue.job_progress.connect(self._on_queue_job_progress)
            self._print_queue.job_completed.connect(self._on_queue_job_completed)
            self._print_queue.all_jobs_completed.connect(self._on_all_jobs_completed)
            self._print_queue.queue_updated.connect(self._on_queue_updated)
            self._print_queue.log_message.connect(self._on_queue_log)

            # 대기열 보기 버튼 연결
            self.ui.components['progress_panel'].queue_btn.clicked.connect(self._show_queue_list_dialog)

            self._queue_initialized = True
            print("[OK] 인쇄 큐 시스템 초기화 완료")

        except Exception as e:
            print(f"[ERROR] 인쇄 큐 초기화 실패: {e}")
    
    def _auto_show_printer_dialog(self):
        """프린터 선택 대화상자 자동 표시"""
        try:
            if self.printer_dll_path and not self.selected_printer_info:
                from ui.components.printer_selection_dialog import PrinterSelectionDialog
                # 올바른 순서: dll_path가 먼저, parent가 나중
                dialog = PrinterSelectionDialog(self.printer_dll_path, parent=self)
                if dialog.exec():
                    self.selected_printer_info = dialog.get_selected_printer()
                    if self.selected_printer_info:
                        self.printer_available = True
                        print(f"[OK] 프린터 선택됨: {self.selected_printer_info.name}")
                        self.ui.components['printer_panel'].update_status(f"✅ 프린터 연결됨: {self.selected_printer_info.name}")
                        self._update_print_button_state()
        except Exception as e:
            print(f"[ERROR] 프린터 대화상자 표시 실패: {e}")
    
    def _on_model_loading_progress(self, message: str):
        """AI 모델 로딩 진행 상황 처리"""
        self.log(message)
        # UI 상태 업데이트 - 모델 로딩 중임을 표시
        self.ui.components['progress_panel'].update_status(message)
    
    def _on_model_loading_completed(self, session):
        """AI 모델 로딩 완료 처리"""
        self.log("[OK] 배경제거 기능 사용 가능!")
        # UI 상태 업데이트 - 준비 완료
        if hasattr(self, 'front_image_path') and self.front_image_path:
            self._update_ui_state()
        else:
            self.ui.components['progress_panel'].update_status("[INFO] 이미지를 선택해주세요")
    
    def _on_model_loading_failed(self, error_message: str):
        """AI 모델 로딩 실패 처리"""
        self.log(f"[ERROR] {error_message}")
        self.ui.components['progress_panel'].update_status("❌ 배경제거 기능 사용 불가")
    
    # === 지연 로딩을 위한 getter 메서드들 ===
    
    def get_image_processor(self):
        """ImageProcessor 인스턴스 반환 (지연 로딩)"""
        if self.image_processor is None:
            from core import ImageProcessor
            self.image_processor = ImageProcessor()
        return self.image_processor
    
    def get_file_manager(self):
        """FileManager 인스턴스 반환 (지연 로딩)"""
        if self.file_manager is None:
            from core import FileManager
            self.file_manager = FileManager()
        return self.file_manager
    
    def _setup_window_icon(self):
        """윈도우 아이콘 설정"""
        try:
            icon_path = get_resource_path("hana.ico")
            if os.path.exists(icon_path):
                self.setWindowIcon(QIcon(icon_path))
                print(f"[OK] 윈도우 아이콘 설정: {icon_path}")
            else:
                print(f"[WARN] 아이콘 파일 없음: {icon_path}")
        except Exception as e:
            print(f"[WARN] 윈도우 아이콘 설정 실패: {e}")
        
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
            from printer import find_printer_dll
            from ui.components.printer_selection_dialog import show_printer_selection_dialog
            
            self.printer_dll_path = find_printer_dll()
            if not self.printer_dll_path:
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
            
            print(f"[OK] 프린터 선택 완료: {selected_printer}")
            return True
            
        except Exception as e:
            QMessageBox.critical(
                None,
                "프린터 선택 오류",
                f"프린터 선택 중 오류가 발생했습니다:\n\n{e}"
            )
            return False
        
    def _setup_window(self):
        """윈도우 기본 설정"""
        self.setWindowTitle(f"{AppConstants.APP_NAME}")
        
        # geometry가 None인 경우를 처리
        geometry = config.get('window_geometry', {})
        if geometry is None:
            geometry = {}
            
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
        """시그널 연결 - 위치 조정 시그널 추가"""
        components = self.ui.components
        
        # 기존 시그널들...
        components['file_panel'].front_btn.clicked.connect(self.select_front_image)
        components['file_panel'].back_btn.clicked.connect(self.select_back_image)
        
        # 개별 배경제거
        components['front_original_viewer'].process_requested.connect(
            lambda threshold: self.process_single_image(is_front=True, threshold=threshold)
        )
        components['back_original_viewer'].process_requested.connect(
            lambda threshold: self.process_single_image(is_front=False, threshold=threshold)
        )
        
        # 개별 면 방향 변경
        components['front_original_viewer'].orientation_changed.connect(
            lambda orientation: self.on_front_orientation_changed(orientation)
        )
        components['back_original_viewer'].orientation_changed.connect(
            lambda orientation: self.on_back_orientation_changed(orientation)
        )
        
        # 임계값 변경
        components['front_original_viewer'].threshold_changed.connect(
            lambda value: self.log(f"앞면 임계값 변경: {value}")
        )
        components['back_original_viewer'].threshold_changed.connect(
            lambda value: self.log(f"뒷면 임계값 변경: {value}")
        )
        
        # 기존 시그널들
        components['print_mode_panel'].mode_changed.connect(self.on_print_mode_changed)
        components['print_mode_panel'].dual_side_changed.connect(self.on_dual_side_toggled)
        components['print_quantity_panel'].quantity_changed.connect(self.on_print_quantity_changed)
        components['printer_panel'].test_requested.connect(self.test_printer_connection)
        components['print_quantity_panel'].print_requested.connect(self.print_card)

        # 탭 변경 시그널
        if 'image_tab_widget' in components:
            components['image_tab_widget'].tab_changed.connect(self.on_image_tab_changed)

        # ✨ 위치 조정 시그널 연결 (float 타입)
        components['position_panel'].position_changed.connect(self.on_position_changed)

    def on_position_changed(self, x: float, y: float):
        """위치 조정값 변경 처리 (float)"""
        self.adjusted_x = x
        self.adjusted_y = y
        
        # 로그에 위치 변경 기록 (개발자용)
        if x == 0.0 and y == 0.0:
            self.log("📐 카드 위치 초기화됨")
        else:
            self.log(f"📐 카드 위치 조정: X={x:+.1f}mm, Y={y:+.1f}mm")
    
    def get_position_adjustment(self):
        """현재 위치 조정값 반환 (float)"""
        return self.adjusted_x, self.adjusted_y
    
    def set_position_adjustment(self, x: float, y: float):
        """위치 조정값 설정 (float)"""
        self.ui.components['position_panel'].set_position(x, y)
    
    def _start_multi_print(self, front_path=None, back_path=None):
        """여러장 인쇄 시작 - 큐 시스템 사용"""
        try:
            # 큐 시스템 초기화 확인
            if not self._queue_initialized:
                self._init_print_queue()

            # [TARGET] 진행 상황 표시 시작
            self.ui.components['progress_panel'].show_progress()
            # 큐 시스템: 버튼은 계속 활성화 (추가 요청 가능)

            # [TARGET] 사용자 친화적 인쇄 시작 메시지
            if self.print_quantity > 1:
                self.log(f"📄 카드 {self.print_quantity}장 인쇄 요청!")
            else:
                self.log(f"📄 카드 인쇄 요청!")

            if front_path is None:
                front_path = self.front_image_path
            if back_path is None:
                back_path = self.back_image_path

            # 인쇄 큐에 작업 추가
            job_id = self._print_queue.add_job(
                dll_path=self.printer_dll_path,
                front_image_path=front_path,
                back_image_path=back_path,
                front_mask_path=self.front_saved_mask_path if self.print_mode == "layered" else None,
                back_mask_path=self.back_saved_mask_path if self.print_mode == "layered" else None,
                print_mode=self.print_mode,
                is_dual_side=self.is_dual_side,
                quantity=self.print_quantity,
                front_orientation=self.front_orientation,
                back_orientation=self.back_orientation,
                adjusted_x=self.adjusted_x,
                adjusted_y=self.adjusted_y
            )

            self.log(f"[OK] 인쇄 작업 #{job_id} 큐에 추가됨")

        except Exception as e:
            self.ui.components['progress_panel'].hide_progress()
            error_msg = f"인쇄 시작 실패: {e}"
            self.log(f"[ERROR] {error_msg}")
            QMessageBox.critical(self, "인쇄 오류", error_msg)
    
    def print_card(self):
        """카드 인쇄 - 위치 조정 정보 포함된 확인 다이얼로그 (큐 시스템 지원)"""
        # 기존 검증 코드들...
        if not self.printer_available or not self.printer_dll_path:
            QMessageBox.warning(self, "경고", "프린터를 사용할 수 없습니다.")
            return

        if not self.front_image_path:
            QMessageBox.warning(self, "경고", "앞면 이미지를 먼저 선택해주세요.")
            return

        # 큐 시스템: 인쇄 중이어도 새 작업 추가 가능 (큐에 쌓임)
        # 사용자에게 현재 대기열 상태 알림
        if self._queue_initialized and self._print_queue:
            queue_status = self._print_queue.get_queue_status()
            if queue_status['is_processing']:
                # 인쇄 중이면 큐에 추가할지 확인
                queue_size = queue_status['queue_size']
                reply = QMessageBox.question(
                    self, "인쇄 대기열",
                    f"현재 인쇄가 진행 중입니다.\n대기열에 {queue_size}개의 작업이 있습니다.\n\n새 인쇄 작업을 대기열에 추가하시겠습니까?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes
                )
                if reply == QMessageBox.No:
                    return
        
        # 레이어 모드 검증...
        if self.print_mode == "layered":
            front_mask = self.ui.components['front_unified_mask_viewer'].get_current_mask()
            if front_mask is None:
                QMessageBox.warning(self, "경고", "레이어 인쇄를 위해서는 마스킹 이미지가 필요합니다.\n개별 배경제거를 실행하거나 수동 마스킹을 업로드해주세요.")
                return
            
            # 마스크 저장...
            file_manager = self.get_file_manager()
            self.front_saved_mask_path = file_manager.save_mask_for_printing(
                front_mask, self.front_image_path, "front"
            )
            if not self.front_saved_mask_path:
                QMessageBox.critical(self, "오류", "앞면 마스크 이미지 저장에 실패했습니다.")
                return
            
            if self.is_dual_side and self.back_image_path:
                back_mask = self.ui.components['back_unified_mask_viewer'].get_current_mask()
                if back_mask is not None:
                    self.back_saved_mask_path = file_manager.save_mask_for_printing(
                        back_mask, self.back_image_path, "back"
                    )
                    if not self.back_saved_mask_path:
                        self.log("[WARNING] 뒷면 마스크 저장 실패, 뒷면은 일반 모드로 인쇄됩니다.")
        
        # 인쇄 경로 설정
        front_print_path = self.front_image_path
        back_print_path = self.back_image_path
        
        # 인쇄 확인 다이얼로그 - 위치 조정 정보 포함
        mode_text = "일반 인쇄" if self.print_mode == "normal" else "레이어 인쇄 (YMCW)"
        side_text = "양면" if self.is_dual_side else "단면"
        
        # 개별 면 방향 정보
        front_orientation_text = "세로형" if self.front_orientation == "portrait" else "가로형"
        back_orientation_text = "세로형" if self.back_orientation == "portrait" else "가로형"
        
        file_manager = self.get_file_manager()
        front_name, _ = file_manager.get_file_info(self.front_image_path)
        detail_text = f"앞면 이미지: {front_name} ({front_orientation_text})\n"
        
        # 마스킹 정보 추가
        if self.print_mode == "layered":
            front_mask_type = self.ui.components['front_unified_mask_viewer'].get_mask_type()
            front_mask_text = "수동 마스킹" if front_mask_type == "manual" else "자동 마스킹"
            detail_text += f"  마스킹: {front_mask_text}\n"
        
        if self.is_dual_side and self.back_image_path:
            back_name, _ = file_manager.get_file_info(self.back_image_path)
            detail_text += f"뒷면 이미지: {back_name} ({back_orientation_text})\n"
            
            if self.print_mode == "layered":
                back_mask_type = self.ui.components['back_unified_mask_viewer'].get_mask_type()
                if back_mask_type:
                    back_mask_text = "수동 마스킹" if back_mask_type == "manual" else "자동 마스킹"
                    detail_text += f"  마스킹: {back_mask_text}\n"
            
        elif self.is_dual_side:
            detail_text += f"뒷면 이미지: 없음 (빈 뒷면으로 인쇄, {back_orientation_text})\n"
        
        detail_text += f"인쇄 방식: {side_text} {mode_text}\n"
        detail_text += f"인쇄 매수: {self.print_quantity}장\n"
        
        # ✨ 위치 조정 정보 추가 (float 형식)
        if self.adjusted_x != 0.0 or self.adjusted_y != 0.0:
            detail_text += f"위치 조정: X{self.adjusted_x:+.1f}mm, Y{self.adjusted_y:+.1f}mm\n"

        reply = QMessageBox.question(
            self,
            "카드 인쇄",
            f"카드 인쇄를 시작하시겠습니까?\n\n{detail_text}\n"
            "프린터에 충분한 카드가 준비되어 있는지 확인해주세요.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        # 인쇄 시작
        self._start_multi_print(front_print_path, back_print_path)

    def on_image_tab_changed(self, tab_index: int):
        """이미지 탭 변경 시 처리"""
        tab_name = "앞면" if tab_index == 0 else "뒷면"
        self.log(f"📑 {tab_name} 탭으로 전환")

    def on_front_orientation_changed(self, orientation: str):
        """앞면 방향 변경 처리"""
        self.front_orientation = orientation
        
        # 통합 마스킹 뷰어에 방향 적용
        self.ui.components['front_unified_mask_viewer'].set_card_orientation(orientation)
        
        orientation_text = "세로형" if orientation == "portrait" else "가로형"
        self.log(f"앞면 출력 방향 변경: {orientation_text}")
        
        # 상태 업데이트
        self._update_ui_state()
        self._update_print_button_state()
    
    def on_back_orientation_changed(self, orientation: str):
        """뒷면 방향 변경 처리"""
        self.back_orientation = orientation
        
        # 통합 마스킹 뷰어에 방향 적용
        self.ui.components['back_unified_mask_viewer'].set_card_orientation(orientation)
        
        orientation_text = "세로형" if orientation == "portrait" else "가로형"
        self.log(f"뒷면 출력 방향 변경: {orientation_text}")
        
        # 상태 업데이트
        self._update_ui_state()
        self._update_print_button_state()

    def on_card_orientation_changed(self, orientation: str):
        """전역 카드 방향 변경 처리 (사용되지 않음 - 하위 호환성만)"""
        pass
            
    def _check_printer_availability(self):
        """프린터 사용 가능성 확인"""
        def check():
            try:
                from printer import find_printer_dll
                self.printer_dll_path = find_printer_dll()
                if self.printer_dll_path:
                    self.printer_available = True
                    self.ui.components['printer_panel'].update_status("🔌 프린터 연결 테스트를 눌러주세요")
                else:
                    self.ui.components['printer_panel'].update_status("❌ DLL 파일 없음")
            except Exception as e:
                self.log(f"[ERROR] 프린터 확인 오류: {e}")
                self.ui.components['printer_panel'].update_status("❌ 프린터 확인 실패")

        threading.Thread(target=check, daemon=True).start()

    def select_front_image(self):
        """앞면 이미지 선택 - PyInstaller 호환"""
        import sys
        
        # PyInstaller 호환 초기 디렉토리 설정
        if getattr(sys, 'frozen', False):
            # 실행파일과 같은 디렉토리에서 시작
            initial_dir = os.path.dirname(sys.executable)
        else:
            initial_dir = os.getcwd()
        
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "앞면 이미지 선택",
            initial_dir,  # 초기 디렉토리 지정
            config.get_image_filter()
        )
        
        if not file_path:
            return
        
        # 절대 경로로 변환
        file_path = os.path.abspath(file_path)
        print(f"[DEBUG] 선택된 앞면 파일: {file_path}")
        print(f"[DEBUG] 파일 존재 여부: {os.path.exists(file_path)}")
        
        # 파일 존재 확인
        if not os.path.exists(file_path):
            QMessageBox.warning(self, "오류", f"파일을 찾을 수 없습니다:\n{file_path}")
            return
        
        # [START] 지연 로딩 적용
        image_processor = self.get_image_processor()
        file_manager = self.get_file_manager()
        
        # 이미지 유효성 검사
        is_valid, error_msg = image_processor.validate_image(file_path)
        if not is_valid:
            QMessageBox.warning(self, "경고", error_msg)
            return
        
        # 앞면 이미지 설정
        self.front_image_path = file_path
        file_name, file_size_mb = file_manager.get_file_info(file_path)
        
        # UI 업데이트
        self.ui.components['file_panel'].update_front_file_info(file_path)
        self.log(f"앞면 이미지 선택: {file_name} ({file_size_mb:.1f}MB)")
        
        # ImageViewer에 이미지 설정
        self.ui.components['front_original_viewer'].set_image(file_path)
        
        # ✨ 앞면 탭으로 자동 전환
        self.ui.set_current_tab(0)
        
        # OpenCV로 읽기 (지연 로딩)
        try:
            self.front_original_image = file_manager._safe_imread(file_path)
            if self.front_original_image is None:
                print(f"[WARNING] OpenCV 이미지 로드 실패: {file_path}")
        except Exception as e:
            print(f"[DEBUG] OpenCV 이미지 로드 실패: {e}")
            self.front_original_image = None
        
        self._update_ui_state()
        self._reset_front_processing_results()

    def select_back_image(self):
        """뒷면 이미지 선택 - PyInstaller 호환"""
        if not self.is_dual_side:
            return
        
        import sys
        
        # PyInstaller 호환 초기 디렉토리 설정
        if getattr(sys, 'frozen', False):
            initial_dir = os.path.dirname(sys.executable)
        else:
            initial_dir = os.getcwd()
            
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "뒷면 이미지 선택",
            initial_dir,
            config.get_image_filter()
        )
        
        if not file_path:
            return
        
        # 절대 경로로 변환
        file_path = os.path.abspath(file_path)
        print(f"[DEBUG] 선택된 뒷면 파일: {file_path}")
        print(f"[DEBUG] 파일 존재 여부: {os.path.exists(file_path)}")
        
        # 파일 존재 확인
        if not os.path.exists(file_path):
            QMessageBox.warning(self, "오류", f"파일을 찾을 수 없습니다:\n{file_path}")
            return
        
        # [START] 지연 로딩 적용
        image_processor = self.get_image_processor()
        file_manager = self.get_file_manager()
        
        # 이미지 유효성 검사
        is_valid, error_msg = image_processor.validate_image(file_path)
        if not is_valid:
            QMessageBox.warning(self, "경고", error_msg)
            return
        
        # 뒷면 이미지 설정
        self.back_image_path = file_path
        file_name, file_size_mb = file_manager.get_file_info(file_path)
        
        # UI 업데이트
        self.ui.components['file_panel'].update_back_file_info(file_path)
        self.log(f"뒷면 이미지 선택: {file_name} ({file_size_mb:.1f}MB)")
        
        # ImageViewer에 이미지 설정
        self.ui.components['back_original_viewer'].set_image(file_path)
        
        # ✨ 뒷면 탭으로 자동 전환
        if self.is_dual_side:
            self.ui.set_current_tab(1)
        
        # OpenCV로 읽기 (지연 로딩)
        try:
            self.back_original_image = file_manager._safe_imread(file_path)
            if self.back_original_image is None:
                print(f"[WARNING] 뒷면 OpenCV 이미지 로드 실패: {file_path}")
        except Exception as e:
            print(f"[DEBUG] 뒷면 OpenCV 이미지 로드 실패: {e}")
            self.back_original_image = None
        
        self._update_ui_state()
        self._reset_back_processing_results()

    def process_single_image(self, is_front: bool, threshold: int = 200):
        """개별 이미지 배경제거 처리 - AI 모델 대기 기능 포함"""
        if is_front:
            if not self.front_image_path:
                return
            image_path = self.front_image_path
            viewer = self.ui.components['front_original_viewer']
            side_text = "앞면"
        else:
            if not self.back_image_path:
                return
            image_path = self.back_image_path
            viewer = self.ui.components['back_original_viewer']
            side_text = "뒷면"

        # 배경제거 버튼 비활성화
        viewer.set_process_enabled(False)
        self.ui.components['progress_panel'].show_progress()

        # AI 모델 준비 상태 확인
        from core.model_loader import is_ai_model_ready, get_model_loader
        from core.model_downloader import is_model_downloaded, ensure_model_downloaded

        loader = get_model_loader()
        model_name = loader.model_name

        # 모델이 이미 다운로드되어 있는지 확인
        if is_model_downloaded(model_name):
            self.log(f"[AI] 기존 모델 사용: {model_name}")
        else:
            # 모델이 없으면 다운로드
            self.log(f"[AI] 모델 다운로드 시작: {model_name}")
            self.ui.components['progress_panel'].update_status("[DOWNLOAD] AI 모델 다운로드 중...")

            if not ensure_model_downloaded(model_name, self):
                # 다운로드 취소됨
                viewer.set_process_enabled(True)
                self.ui.components['progress_panel'].hide_progress()
                self.log("[CANCEL] 모델 다운로드가 취소되었습니다")
                self.ui.components['progress_panel'].update_status("[CANCEL] 다운로드 취소됨")
                return

            self.log(f"[AI] 모델 다운로드 완료!")

        if not is_ai_model_ready():
            self.log(f"[WAIT] 배경제거 AI 준비 완료 대기 중... {side_text} 처리는 자동으로 시작됩니다")
            self.ui.components['progress_panel'].update_status("[WAIT] 배경제거 AI 준비 중...")

            # 모델 로딩 시작 (다운로드는 이미 완료)
            if not loader.is_loading and not loader.is_loaded:
                loader.start_background_loading()

            # 모델 로딩 완료까지 대기하는 스레드 시작
            import threading
            def wait_and_process():
                import time
                # 전역 로더 다시 가져오기 (스레드 안전성)
                from core.model_loader import get_model_loader
                current_loader = get_model_loader()

                start_time = time.time()
                timeout = 180.0  # 60초 → 180초 (첫 실행 시 ONNX Runtime + numpy/scipy 로딩 시간 확보)

                print(f"[DEBUG] 대기 시작 - is_loaded: {current_loader.is_loaded}, is_loading: {current_loader.is_loading}")

                # 로딩 완료까지 대기
                while not current_loader.is_loaded and (time.time() - start_time) < timeout:
                    time.sleep(0.1)

                elapsed = time.time() - start_time
                print(f"[DEBUG] 대기 종료 - is_loaded: {current_loader.is_loaded}, elapsed: {elapsed:.1f}s")

                # 결과를 메인 스레드에서 처리하도록 invokeMethod 사용
                from PySide6.QtCore import QMetaObject, Qt
                if current_loader.is_loaded:
                    print(f"[DEBUG] 처리 시작 요청")
                    # 메인 스레드에서 실행 - 인자를 인스턴스 변수로 저장
                    self._pending_process_args = (is_front, threshold, image_path, side_text)
                    QMetaObject.invokeMethod(self, "_execute_pending_process", Qt.QueuedConnection)
                else:
                    print(f"[DEBUG] 타임아웃!")
                    self._pending_timeout_args = is_front
                    QMetaObject.invokeMethod(self, "_execute_pending_timeout", Qt.QueuedConnection)

            threading.Thread(target=wait_and_process, daemon=True).start()
            return

        # 모델이 준비된 경우 즉시 처리 시작
        self._start_processing_after_wait(is_front, threshold, image_path, side_text)
    
    def _start_processing_after_wait(self, is_front: bool, threshold: int, image_path: str, side_text: str):
        """AI 모델 로딩 완료 후 실제 배경제거 처리 시작"""
        # 세션 준비 확인
        from core.model_loader import get_model_loader
        loader = get_model_loader()

        if not loader.is_ready():
            # 세션이 준비되지 않음
            viewer = self.ui.components['front_original_viewer'] if is_front else self.ui.components['back_original_viewer']
            self.ui.components['progress_panel'].hide_progress()
            viewer.set_process_enabled(True)

            error_msg = f"{side_text} 배경제거 실패: AI 모델이 아직 준비되지 않았습니다."
            self.log(f"[ERROR] {error_msg}")

            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self,
                "AI 모델 준비 중",
                "배경제거 AI를 준비하는 중입니다.\n잠시 후 다시 시도해주세요."
            )
            return

        # 세션 가져오기 (이미 준비됨 보장)
        session = loader.get_loaded_session()

        # 항상 원본 이미지 경로로 처리
        self.log(f"[SUCCESS] 배경제거 AI 준비 완료! {side_text} 배경제거 시작...")

        # 임계값을 config에 임시 설정
        from config import config
        original_threshold = config.get('alpha_threshold', 200)
        config.set('alpha_threshold', threshold)

        # [START] 지연 로딩 적용
        from core import ProcessingThread
        image_processor = self.get_image_processor()

        # 세션을 명시적으로 전달
        self.processing_thread = ProcessingThread(
            image_path,
            image_processor,
            session,  # 세션 전달
            threshold  # 임계값 전달
        )
        
        # 시그널 연결 (어느 쪽인지 구분)
        if is_front:
            self.processing_thread.finished.connect(
                lambda mask: self.on_front_processing_finished(mask, threshold, original_threshold)
            )
        else:
            self.processing_thread.finished.connect(
                lambda mask: self.on_back_processing_finished(mask, threshold, original_threshold)
            )
        
        self.processing_thread.error.connect(
            lambda error: self.on_processing_error(error, is_front, original_threshold)
        )
        self.processing_thread.progress.connect(self.on_processing_progress)
        self.processing_thread.start()
    
    def _handle_model_timeout(self, is_front: bool):
        """AI 모델 로딩 타임아웃 처리"""
        side_text = "앞면" if is_front else "뒷면"
        viewer = self.ui.components['front_original_viewer'] if is_front else self.ui.components['back_original_viewer']
        
        self.ui.components['progress_panel'].hide_progress()
        viewer.set_process_enabled(True)
        
        error_msg = f"{side_text} 배경제거 실패: AI 모델 로딩 타임아웃"
        self.log(f"[ERROR] {error_msg}")
        self.ui.components['progress_panel'].update_status("❌ AI 모델 로딩 실패")
        
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.warning(
            self,
            "AI 모델 로딩 실패",
            f"AI 모델 로딩이 완료되지 않아 배경제거를 실행할 수 없습니다.\n\n"
            "네트워크 연결을 확인하고 다시 시도해주세요."
        )

    @Slot(bool, int, str, str)
    def _on_model_ready_for_processing(self, is_front: bool, threshold: int, image_path: str, side_text: str):
        """모델 로딩 완료 후 처리 시작 (메인 스레드에서 호출)"""
        self._start_processing_after_wait(is_front, threshold, image_path, side_text)

    @Slot(bool)
    def _on_model_timeout_for_processing(self, is_front: bool):
        """모델 로딩 타임아웃 처리 (메인 스레드에서 호출)"""
        self._handle_model_timeout(is_front)

    @Slot()
    def _execute_pending_process(self):
        """대기 중인 처리 실행 (메인 스레드에서 호출)"""
        if hasattr(self, '_pending_process_args'):
            args = self._pending_process_args
            print(f"[DEBUG] _execute_pending_process 호출됨")
            self._on_model_ready_for_processing(*args)
            del self._pending_process_args

    @Slot()
    def _execute_pending_timeout(self):
        """대기 중인 타임아웃 처리 실행 (메인 스레드에서 호출)"""
        if hasattr(self, '_pending_timeout_args'):
            is_front = self._pending_timeout_args
            print(f"[DEBUG] _execute_pending_timeout 호출됨")
            self._on_model_timeout_for_processing(is_front)
            del self._pending_timeout_args

    def on_dual_side_toggled(self, checked):
        """양면 인쇄 토글"""
        self.is_dual_side = checked
        
        # 파일선택 패널에 양면 상태 전달
        self.ui.components['file_panel'].set_dual_side_enabled(checked)
        
        # ✨ 탭 위젯에 양면 상태 전달
        self.ui.set_dual_side_enabled(checked)
        
        if not checked:
            # 단면 모드로 변경 시 뒷면 데이터 초기화
            self.back_image_path = None
            self.back_original_image = None
            self.back_auto_mask_image = None
            self.back_manual_mask_path = None
            self.back_manual_mask_image = None
            self.ui.components['back_original_viewer'].clear_image()
            self.ui.components['back_unified_mask_viewer'].clear_mask()
            self.ui.components['back_manual_mask_viewer'].clear_image()
            
            # ✨ 앞면 탭으로 강제 이동
            self.ui.set_current_tab(0)
        
        # 인쇄 버튼 텍스트 업데이트
        self.ui.components['print_quantity_panel'].update_print_button_text(
            self.print_mode, checked, self.print_quantity
        )
        
        mode_text = "양면 인쇄" if checked else "단면 인쇄"
        self.log(f"인쇄 방식 변경: {mode_text}")
        
        self._update_ui_state()
    
    def on_print_mode_changed(self, mode):
        """인쇄 모드 변경"""
        self.print_mode = mode
        self.ui.components['print_quantity_panel'].update_print_button_text(
            mode, self.is_dual_side, self.print_quantity
        )
        self._update_print_button_state()
        
        mode_text = '일반 인쇄' if mode == 'normal' else '레이어 인쇄(YMCW)'
        self.log(f"인쇄 모드 변경: {mode_text}")
    
    def on_print_quantity_changed(self, quantity):
        """인쇄 매수 변경"""
        self.print_quantity = quantity
        self.ui.components['print_quantity_panel'].update_print_button_text(
            self.print_mode, self.is_dual_side, quantity
        )
        
        self.log(f"인쇄 매수 변경: {quantity}장")
        
        if self.front_image_path:
            # 개별 면 방향 정보로 상태 메시지 구성
            front_orientation_text = "세로형" if self.front_orientation == "portrait" else "가로형"
            
            if quantity > 1:
                if self.is_dual_side and self.back_image_path:
                    back_orientation_text = "세로형" if self.back_orientation == "portrait" else "가로형"
                    status = f"앞면:{front_orientation_text}, 뒷면:{back_orientation_text} {quantity}장 인쇄 준비"
                elif self.is_dual_side:
                    status = f"앞면:{front_orientation_text} 양면 {quantity}장 인쇄 준비 (뒷면 선택사항)"
                else:
                    status = f"앞면:{front_orientation_text} 단면 {quantity}장 인쇄 준비"
            else:
                if self.is_dual_side and self.back_image_path:
                    back_orientation_text = "세로형" if self.back_orientation == "portrait" else "가로형"
                    status = f"앞면:{front_orientation_text}, 뒷면:{back_orientation_text} 인쇄 준비"
                elif self.is_dual_side:
                    status = f"앞면:{front_orientation_text} 양면 인쇄 준비 (뒷면 선택사항)"
                else:
                    status = f"앞면:{front_orientation_text} 단면 인쇄 준비"
            
            self.ui.components['progress_panel'].update_status(status)
    
    def on_manual_mask_uploaded(self, file_path: str, is_front: bool):
        """수동 마스킹 이미지 업로드 처리"""
        try:
            # [START] 지연 로딩 적용
            import cv2
            import numpy as np
            
            image_processor = self.get_image_processor()
            
            # 이미지 유효성 검사
            is_valid, error_msg = image_processor.validate_image(file_path)
            if not is_valid:
                QMessageBox.warning(self, "경고", f"마스킹 이미지 오류: {error_msg}")
                return
            
            # 마스킹 이미지 로드 (한글 경로 대응 - 바이트 방식 우선)
            try:
                with open(file_path, 'rb') as f:
                    image_data = f.read()
                nparr = np.frombuffer(image_data, np.uint8)
                mask_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except:
                # Fallback: cv2.imread 직접 시도 (ASCII 경로인 경우)
                mask_image = cv2.imread(file_path)
            
            if mask_image is None:
                QMessageBox.warning(self, "경고", "마스킹 이미지를 읽을 수 없습니다.")
                return
            
            side_text = "앞면" if is_front else "뒷면"
            file_name = os.path.basename(file_path)
            
            if is_front:
                self.front_manual_mask_path = file_path
                self.front_manual_mask_image = mask_image
                
                # 통합 마스킹 뷰어에 수동 마스킹 설정
                self.ui.components['front_unified_mask_viewer'].set_manual_mask(mask_image)
                
                self.log(f"[OK] {side_text} 수동 마스킹 업로드: {file_name}")
                self.log(f"   {side_text} 통합 미리보기가 수동 마스킹으로 업데이트되었습니다.")
            else:
                self.back_manual_mask_path = file_path
                self.back_manual_mask_image = mask_image
                
                # 통합 마스킹 뷰어에 수동 마스킹 설정
                self.ui.components['back_unified_mask_viewer'].set_manual_mask(mask_image)
                
                self.log(f"[OK] {side_text} 수동 마스킹 업로드: {file_name}")
                self.log(f"   {side_text} 통합 미리보기가 수동 마스킹으로 업데이트되었습니다.")
            
            # UI 상태 업데이트
            self._update_ui_state()
            self._update_print_button_state()
            
        except Exception as e:
            side_text = "앞면" if is_front else "뒷면"
            error_msg = f"{side_text} 수동 마스킹 업로드 실패: {e}"
            self.log(f"[ERROR] {error_msg}")
            QMessageBox.critical(self, "업로드 오류", error_msg)
    
    def _update_ui_state(self):
        """UI 상태 업데이트 - 단순화된 메시지"""
        # 상태 메시지만 업데이트
        if self.front_image_path:
            if self.print_quantity > 1:
                if self.is_dual_side and self.back_image_path:
                    status = f"📋 카드 {self.print_quantity}장 인쇄 준비"
                elif self.is_dual_side:
                    status = f"📋 카드 {self.print_quantity}장 인쇄 준비"
                else:
                    status = f"📋 카드 {self.print_quantity}장 인쇄 준비"
            elif self.is_dual_side:
                if self.back_image_path:
                    status = "📋 카드 인쇄 준비"
                else:
                    status = "📋 카드 인쇄 준비"
            else:
                status = "📋 카드 인쇄 준비"
            
            self.ui.components['progress_panel'].update_status(status)
        else:
            self.ui.components['progress_panel'].update_status("📂 이미지를 선택해주세요")
            
    def _reset_front_processing_results(self):
        """앞면 처리 결과 초기화"""
        self.front_auto_mask_image = None
        self.front_saved_mask_path = None
        
        # 통합 마스킹 뷰어에서 자동 마스킹만 클리어 (수동은 유지)
        if self.front_manual_mask_image is None:
            self.ui.components['front_unified_mask_viewer'].clear_mask()
        
        self._update_print_button_state()
    
    def _reset_back_processing_results(self):
        """뒷면 처리 결과 초기화"""
        self.back_auto_mask_image = None
        self.back_saved_mask_path = None
        
        # 통합 마스킹 뷰어에서 자동 마스킹만 클리어 (수동은 유지)
        if self.back_manual_mask_image is None:
            self.ui.components['back_unified_mask_viewer'].clear_mask()
        
        self._update_print_button_state()
    
    def on_processing_progress(self, message):
        """처리 진행상황 업데이트 - 단순화"""
        # 기술적 메시지를 사용자 친화적으로 변환
        if "AI 모델" in message or "모델" in message:
            simple_message = "🔄 이미지 처리 중..."
        elif "배경 제거" in message or "마스크" in message:
            simple_message = "🔄 이미지 처리 중..."
        elif "완료" in message:
            simple_message = "✅ 이미지 처리 완료!"
        else:
            simple_message = "🔄 이미지 처리 중..."
            
        self.ui.components['progress_panel'].update_status(simple_message)
        # 로그는 기존 메시지 유지 (개발자용)
        self.log(message)
    
    def on_front_processing_finished(self, mask_array, used_threshold, original_threshold):
        """앞면 자동 배경제거 완료 - 임계값 복원"""
        # 임계값 복원
        from config import config
        config.set('alpha_threshold', original_threshold)
        
        self.front_auto_mask_image = mask_array
        
        # 통합 마스킹 뷰어에 자동 마스킹 설정
        self.ui.components['front_unified_mask_viewer'].set_auto_mask(mask_array)
        
        self.log(f"[OK] 앞면 자동 배경 제거 완료! (임계값: {used_threshold})")
        self.log("   앞면 통합 미리보기가 자동 마스킹으로 업데이트되었습니다.")
        
        # UI 정리
        self.ui.components['progress_panel'].hide_progress()
        self.ui.components['front_original_viewer'].set_process_enabled(True)
        
        self._update_ui_state()
        self._update_print_button_state()
        
    def on_back_processing_finished(self, mask_array, used_threshold, original_threshold):
        """뒷면 자동 배경제거 완료 - 임계값 복원"""
        # 임계값 복원
        from config import config
        config.set('alpha_threshold', original_threshold)
        
        self.back_auto_mask_image = mask_array
        
        # 통합 마스킹 뷰어에 자동 마스킹 설정
        self.ui.components['back_unified_mask_viewer'].set_auto_mask(mask_array)
        
        self.log(f"[OK] 뒷면 자동 배경 제거 완뢬! (임계값: {used_threshold})")
        self.log("   뒷면 통합 미리보기가 자동 마스킹으로 업데이트되었습니다.")
        
        # UI 정리
        self.ui.components['progress_panel'].hide_progress()
        self.ui.components['back_original_viewer'].set_process_enabled(True)
        
        self._update_ui_state()
        self._update_print_button_state()
        
    def on_processing_error(self, error_message, is_front: bool, original_threshold):
        """처리 오류 - 임계값 복원"""
        # 임계값 복원
        from config import config
        config.set('alpha_threshold', original_threshold)
        
        side_text = "앞면" if is_front else "뒷면"
        viewer = self.ui.components['front_original_viewer'] if is_front else self.ui.components['back_original_viewer']
        
        self.ui.components['progress_panel'].hide_progress()
        viewer.set_process_enabled(True)
        
        self.log(f"[ERROR] {side_text} 처리 오류: {error_message}")
        self.ui.components['progress_panel'].update_status(f"{side_text} 오류 발생 | 다시 시도해주세요")
        
        QMessageBox.critical(self, "처리 오류", f"{side_text} 이미지 처리 중 오류가 발생했습니다:\n\n{error_message}")
        
    def get_front_threshold(self):
        """앞면 임계값 반환"""
        return self.ui.components['front_original_viewer'].get_threshold_value()

    def get_back_threshold(self):
        """뒷면 임계값 반환"""
        return self.ui.components['back_original_viewer'].get_threshold_value()

    def set_front_threshold(self, value):
        """앞면 임계값 설정"""
        self.ui.components['front_original_viewer'].set_threshold_value(value)
        self.log(f"앞면 임계값 설정: {value}")

    def set_back_threshold(self, value):
        """뒷면 임계값 설정"""
        self.ui.components['back_original_viewer'].set_threshold_value(value)
        self.log(f"뒷면 임계값 설정: {value}")
    
    def _update_print_button_state(self):
        """인쇄 버튼 상태 업데이트 - 큐 시스템에서는 인쇄 중에도 활성화"""
        if not self.printer_available or not self.printer_dll_path:
            self.ui.components['print_quantity_panel'].set_print_enabled(False)
            return

        # 큐 시스템: 인쇄 중에도 추가 요청 가능하므로 is_printing 체크 제거

        if self.print_mode == "normal":
            # 일반 모드: 앞면 이미지만 있으면 인쇄 가능
            can_print = self.front_image_path is not None
        else:
            # 레이어 모드: 앞면 이미지와 마스킹이 있어야 함
            front_mask = self.ui.components['front_unified_mask_viewer'].get_current_mask()
            can_print = (self.front_image_path is not None and front_mask is not None)

        self.ui.components['print_quantity_panel'].set_print_enabled(can_print)
    
    def test_printer_connection(self):
        """프린터 연결 테스트"""
        # DLL이 없으면 경고
        if not self.printer_dll_path:
            QMessageBox.warning(self, "경고", "프린터 DLL을 찾을 수 없습니다.")
            return
        
        # 프린터가 선택되지 않았으면 선택 대화상자 표시
        if not self.selected_printer_info:
            from ui.components.printer_selection_dialog import PrinterSelectionDialog
            dialog = PrinterSelectionDialog(self.printer_dll_path, parent=self)
            if dialog.exec():
                self.selected_printer_info = dialog.get_selected_printer()
                if self.selected_printer_info:
                    self.printer_available = True
                    print(f"[OK] 프린터 선택됨: {self.selected_printer_info.name}")
                    self.ui.components['printer_panel'].update_status(f"✅ 프린터 연결됨: {self.selected_printer_info.name}")
                    self._update_print_button_state()
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
                self.log(f"[OK] {message}")
                self.ui.components['printer_panel'].update_status("✅ 프린터 연결 가능")
                self.ui.components['progress_panel'].update_status("✅ 프린터 테스트 성공")
            else:
                self.log(f"[ERROR] {message}")
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
            self.log(f"[ERROR] 테스트 결과 처리 오류: {e}")
            self.ui.components['printer_panel'].set_test_enabled(True)
            self.ui.components['printer_panel'].update_status("❌ 테스트 오류")
    
    def on_printer_progress(self, message):
        """프린터 진행상황 업데이트 - 상태 메시지만 표시 (진행률은 SDK 콜백에서 처리)"""
        # 프린터 관련 메시지 단순화 (진행률은 on_step_progress에서만 처리)
        if "초기화" in message:
            simple_message = "🔄 프린터 초기화..."
        elif "목록" in message:
            simple_message = "🔍 프린터 검색..."
        elif "선택" in message:
            simple_message = "🖨️ 프린터 연결..."
        elif "인쇄 시작" in message:
            simple_message = "🖨️ 인쇄 준비 중..."
        elif "카드 인쇄 중" in message:
            simple_message = "🖨️ 카드 인쇄 중..."
        elif "장 완료" in message or "다음 카드" in message:
            simple_message = "✅ 카드 완료! 다음 준비..."
        elif "리소스 정리" in message:
            simple_message = "🔄 정리 중..."
        elif "모든" in message and "완료" in message:
            simple_message = "✅ 인쇄 완료!"
        elif "실패" in message or "오류" in message:
            simple_message = "❌ 인쇄 실패"
        elif "중단" in message:
            simple_message = "⏹️ 인쇄 중단됨"
        else:
            simple_message = "🖨️ 작업 중..."

        # 상태 메시지만 업데이트 (진행률은 건드리지 않음)
        self.ui.components['progress_panel'].update_status(simple_message)

        # 로그는 기존 메시지 유지 (개발자용)
        self.log(message)

    def on_print_progress(self, current, total):
        """인쇄 진행률 업데이트 - 상태만 표시 (진행바는 SDK 콜백에서만 처리)"""
        # 진행바는 on_step_progress에서만 업데이트하므로 여기서는 상태만 표시
        pass  # SDK 콜백(on_step_progress)이 정확한 진행률을 제공하므로 중복 업데이트 방지

    def on_step_progress(self, percent: int, message: str):
        """단계별 진행률 업데이트 - SDK 콜백 기반"""
        self.ui.components['progress_panel'].update_step_progress(percent)
        self.ui.components['progress_panel'].update_status(f"🖨️ {message}")
    
    def on_card_completed(self, card_num):
        """개별 카드 완료 - 단순화"""
        self.log(f"[OK] {card_num}번째 카드 인쇄 완료!")
        
        if card_num < self.print_quantity:
            # 사용자에게는 간단한 메시지만 표시
            self.ui.components['progress_panel'].update_status(f"🖨️ 카드 인쇄 중... ({card_num}/{self.print_quantity})")
    
    def on_printer_finished(self, success):
        """프린터 작업 완료 - 100%를 먼저 표시 후 메시지박스"""
        self.ui.components['print_quantity_panel'].set_print_enabled(True)

        if success:
            # 100% 진행률을 먼저 확실히 표시
            self.ui.components['progress_panel'].update_step_progress(100)
            self.ui.components['progress_panel'].update_status("✅ 인쇄 완료!")
            self.log(f"[OK] 카드 {self.print_quantity}장 인쇄 완료!")

            # 메시지박스 표시 (사용자가 확인을 누를 때까지 100%가 보임)
            QMessageBox.information(self, "성공", f"카드 {self.print_quantity}장이 완료되었습니다!")
        else:
            self.log(f"[ERROR] 카드 인쇄 실패")
            self.ui.components['progress_panel'].update_status("❌ 인쇄 실패")

        # 메시지박스 닫은 후 진행바 숨기기
        self.ui.components['progress_panel'].hide_progress()
        self._update_print_button_state()

    def on_printer_error(self, error_message):
        """프린터 오류 처리 - 단순화"""
        self.ui.components['progress_panel'].hide_progress()
        self.ui.components['print_quantity_panel'].set_print_enabled(True)

        self.log(f"[ERROR] 프린터 오류: {error_message}")
        self.ui.components['progress_panel'].update_status("❌ 인쇄 오류 발생")
        QMessageBox.critical(self, "인쇄 오류", f"카드 인쇄 중 오류가 발생했습니다:\n\n{error_message}")

        self._update_print_button_state()

    # ============================================
    # 인쇄 큐 시그널 핸들러
    # ============================================

    def _on_queue_job_added(self, job_id: int, queue_size: int):
        """큐에 작업 추가됨 - 진행 상황 즉시 업데이트"""
        self.log(f"[QUEUE] 작업 #{job_id} 대기열 추가 (대기: {queue_size}개)")

        # 진행 상황 패널 표시 및 업데이트
        self.ui.components['progress_panel'].show_progress()

        # 전체 큐 상태 가져와서 진행률 업데이트
        if self._print_queue:
            progress, status = self._print_queue.get_overall_progress()
            self.ui.components['progress_panel'].update_step_progress(progress)
            self.ui.components['progress_panel'].update_status(f"🖨️ {status}")

    def _on_queue_job_started(self, job_id: int, job_index: int, total_jobs: int):
        """큐 작업 시작"""
        self.log(f"[QUEUE] 작업 #{job_id} 시작 ({job_index}/{total_jobs})")
        self.ui.components['progress_panel'].show_progress()

        # 전체 큐 기준 진행률로 업데이트
        if self._print_queue:
            progress, status = self._print_queue.get_overall_progress()
            self.ui.components['progress_panel'].update_step_progress(progress)
            self.ui.components['progress_panel'].update_status(f"🖨️ {status}")

    def _on_queue_job_progress(self, progress: int, message: str):
        """큐 작업 진행률 업데이트"""
        self.ui.components['progress_panel'].update_step_progress(progress)
        self.ui.components['progress_panel'].update_status(f"🖨️ {message}")

    def _on_queue_job_completed(self, job_id: int, success: bool):
        """개별 작업 완료"""
        if success:
            self.log(f"[QUEUE] 작업 #{job_id} 완료!")
        else:
            self.log(f"[QUEUE] 작업 #{job_id} 실패")

    def _on_all_jobs_completed(self, success_cards: int, failed_cards: int,
                                  success_jobs: int, failed_jobs: int, last_error: str):
        """모든 작업 완료 - 최종 알림 (카드 단위 통계 + 오류 메시지)"""
        # 100% 진행률 표시
        self.ui.components['progress_panel'].update_step_progress(100)

        total_cards = success_cards + failed_cards

        if failed_cards == 0:
            # 모두 성공
            self.ui.components['progress_panel'].update_status("✅ 모든 인쇄 완료!")
            self.log(f"[OK] 모든 인쇄 완료! ({success_cards}장)")
            QMessageBox.information(
                self, "인쇄 완료",
                f"✅ {success_cards}장 인쇄 완료"
            )
        elif success_cards > 0:
            # 일부 성공
            self.ui.components['progress_panel'].update_status("⚠️ 일부 인쇄 실패")
            self.log(f"[WARNING] 일부 인쇄 실패 ({total_cards}장 중 {success_cards}장 완료)")

            error_detail = f"\n\n원인: {last_error}" if last_error else ""
            QMessageBox.warning(
                self, "인쇄 부분 완료",
                f"⚠️ {total_cards}장 중 {success_cards}장 인쇄됨\n"
                f"({failed_cards}장 실패){error_detail}"
            )
        else:
            # 모두 실패
            self.ui.components['progress_panel'].update_status("❌ 인쇄 실패")
            self.log(f"[ERROR] 인쇄 실패 ({failed_cards}장)")

            error_detail = f"\n\n원인: {last_error}" if last_error else ""
            QMessageBox.critical(
                self, "인쇄 실패",
                f"❌ {failed_cards}장 인쇄 실패{error_detail}"
            )

        # 진행바 숨기기
        self.ui.components['progress_panel'].hide_progress()
        self._update_print_button_state()

    def _on_queue_updated(self, remaining_jobs: int):
        """대기열 업데이트"""
        if remaining_jobs > 0:
            self.log(f"[QUEUE] 남은 작업: {remaining_jobs}개")

    def _on_queue_log(self, message: str):
        """큐 로그 메시지"""
        self.log(message)

    def _show_queue_list_dialog(self):
        """인쇄 대기열 목록 다이얼로그 표시"""
        from ui.components import QueueListDialog

        dialog = QueueListDialog(self)

        # 현재 대기열 목록 가져오기
        if self._print_queue:
            jobs = self._print_queue.get_all_jobs_for_display()
            dialog.update_queue_list(jobs)

        dialog.exec()

    def log(self, message):
        """로그 메시지 추가"""
        self.ui.components['log_panel'].add_log(message)

    # ============================================
    # 메뉴바 설정
    # ============================================

    def _setup_menubar(self):
        """메뉴바 설정"""
        try:
            from PySide6.QtGui import QAction, QKeySequence

            menubar = self.menuBar()
            menubar.setStyleSheet("""
                QMenuBar {
                    background-color: #FFFFFF;
                    border-bottom: 1px solid #E5E7EB;
                    padding: 2px 0;
                }
                QMenuBar::item {
                    padding: 6px 12px;
                    background: transparent;
                    color: #374151;
                }
                QMenuBar::item:selected {
                    background-color: #F3F4F6;
                    border-radius: 4px;
                }
                QMenuBar::item:pressed {
                    background-color: #E5E7EB;
                }
                QMenu {
                    background-color: #FFFFFF;
                    border: 1px solid #E5E7EB;
                    border-radius: 8px;
                    padding: 4px;
                }
                QMenu::item {
                    padding: 8px 24px;
                    border-radius: 4px;
                }
                QMenu::item:selected {
                    background-color: #F3F4F6;
                }
                QMenu::separator {
                    height: 1px;
                    background-color: #E5E7EB;
                    margin: 4px 8px;
                }
            """)

            # ========== 파일 메뉴 ==========
            file_menu = menubar.addMenu("파일(&F)")

            # 앞면 이미지 열기
            open_front_action = QAction("앞면 이미지 열기(&O)", self)
            open_front_action.setShortcut(QKeySequence("Ctrl+O"))
            open_front_action.triggered.connect(self._menu_open_front_image)
            file_menu.addAction(open_front_action)

            # 뒷면 이미지 열기
            open_back_action = QAction("뒷면 이미지 열기(&B)", self)
            open_back_action.setShortcut(QKeySequence("Ctrl+Shift+O"))
            open_back_action.triggered.connect(self._menu_open_back_image)
            file_menu.addAction(open_back_action)

            file_menu.addSeparator()

            # 종료
            exit_action = QAction("종료(&X)", self)
            exit_action.setShortcut(QKeySequence("Alt+F4"))
            exit_action.triggered.connect(self.close)
            file_menu.addAction(exit_action)

            # ========== 도움말 메뉴 ==========
            help_menu = menubar.addMenu("도움말(&H)")

            # 버전 정보
            about_action = QAction("버전 정보(&A)", self)
            about_action.setShortcut(QKeySequence("F1"))
            about_action.triggered.connect(self._menu_show_about)
            help_menu.addAction(about_action)

            # ========== 관리 메뉴 (Admin 전용) ==========
            if self.is_admin:
                admin_menu = menubar.addMenu("관리(&A)")

                # 라이선스 관리
                license_action = QAction("라이선스 관리(&L)", self)
                license_action.triggered.connect(self._menu_open_admin_panel)
                admin_menu.addAction(license_action)

            print("[MENU] 메뉴바 설정 완료")

        except Exception as e:
            print(f"[ERROR] 메뉴바 설정 실패: {e}")

    # ============================================
    # 메뉴 액션 핸들러
    # ============================================

    def _menu_open_front_image(self):
        """메뉴: 앞면 이미지 열기"""
        try:
            if hasattr(self.ui, 'file_panel') and self.ui.file_panel:
                self.ui.file_panel.front_btn.click()
        except Exception as e:
            print(f"[ERROR] 앞면 이미지 열기 실패: {e}")

    def _menu_open_back_image(self):
        """메뉴: 뒷면 이미지 열기"""
        try:
            if hasattr(self.ui, 'file_panel') and self.ui.file_panel:
                self.ui.file_panel.back_btn.click()
        except Exception as e:
            print(f"[ERROR] 뒷면 이미지 열기 실패: {e}")

    def _menu_show_about(self):
        """메뉴: 버전 정보"""
        from config import AppConstants
        QMessageBox.about(
            self,
            "Hana Studio 정보",
            f"<h3>Hana Studio</h3>"
            f"<p>버전: {AppConstants.APP_VERSION}</p>"
            f"<p>카드 프린터 전용 이미지 처리 솔루션</p>"
            f"<br>"
            f"<p>© 2025 Hana Labs. All rights reserved.</p>"
        )

    def _menu_open_admin_panel(self):
        """메뉴: Admin Panel 열기"""
        try:
            from licensing.admin_panel import AdminPanel
            panel = AdminPanel(self)
            panel.exec()
        except Exception as e:
            QMessageBox.warning(self, "오류", f"Admin Panel을 열 수 없습니다: {e}")

    def closeEvent(self, event):
        """애플리케이션 종료 시"""
        try:
            # 큐 시스템 확인
            if self._queue_initialized and self._print_queue:
                queue_status = self._print_queue.get_queue_status()
                if queue_status['is_processing'] or queue_status['queue_size'] > 0:
                    total_pending = queue_status['queue_size'] + (1 if queue_status['is_processing'] else 0)
                    reply = QMessageBox.question(
                        self,
                        "인쇄 진행 중",
                        f"인쇄가 진행 중입니다. (대기열: {total_pending}개 작업)\n프로그램을 종료하시겠습니까?\n\n모든 인쇄 작업이 취소됩니다.",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.No
                    )

                    if reply == QMessageBox.StandardButton.Yes:
                        self._print_queue.cancel_all()
                    else:
                        event.ignore()
                        return
        except Exception as e:
            print(f"[WARN] 종료 시 큐 확인 실패: {e}")
        
        # 임시 파일 정리
        if self.file_manager:
            self.file_manager.cleanup_temp_files()
        
        # 윈도우 크기 저장
        geometry = self.geometry()
        config.set('window_geometry.x', geometry.x())
        config.set('window_geometry.y', geometry.y())
        config.set('window_geometry.width', geometry.width())
        config.set('window_geometry.height', geometry.height())
        config.save_settings()
        
        event.accept()