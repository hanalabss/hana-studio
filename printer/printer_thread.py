"""
프린터 작업 스레드
"""

from typing import Optional
from PySide6.QtCore import QThread, Signal
from config import config
from .r600_printer import R600Printer
from .exceptions import R600PrinterError


class PrinterThread(QThread):
    """개선된 프린터 작업 스레드 - 리소스 정리 강화"""
    progress = Signal(str)
    finished = Signal(bool)
    error = Signal(str)
    
    def __init__(self, dll_path: str, original_image_path: str, 
                 mask_image_path: Optional[str] = None, print_mode: str = "normal"):
        super().__init__()
        self.dll_path = dll_path
        self.original_image_path = original_image_path
        self.mask_image_path = mask_image_path
        self.print_mode = print_mode
    
    def run(self):
        printer = None
        try:
            self.progress.emit("프린터 초기화 중...")
            printer = R600Printer(self.dll_path)
            
            self.progress.emit("프린터 목록 조회 중...")
            printers = printer.enum_printers()
            
            if not printers:
                self.error.emit("사용 가능한 프린터가 없습니다.")
                return
            
            self.progress.emit(f"프린터 선택: {printers[0]}")
            printer.select_printer(printers[0])
            
            # 타임아웃 설정 추가
            printer.set_timeout(15000)  # 15초로 증가
            
            # 카드 크기 설정
            card_width = config.get('printer.card_width', 53.98)
            card_height = config.get('printer.card_height', 85.6)
            
            # 인쇄 모드에 따른 처리
            if self.print_mode == "normal":
                self.progress.emit("일반 카드 인쇄 시작...")
                printer.print_normal_card(
                    image_path=self.original_image_path,
                    card_width=card_width,
                    card_height=card_height
                )
                self.progress.emit("✅ 일반 인쇄 완료!")
            else:  # layered
                self.progress.emit("레이어 카드 인쇄 시작...")
                if not self.mask_image_path:
                    self.error.emit("레이어 인쇄를 위해서는 마스크 이미지가 필요합니다.")
                    return
                
                printer.print_layered_card(
                    watermark_path=self.mask_image_path,
                    image_path=self.original_image_path,
                    card_width=card_width,
                    card_height=card_height
                )
                self.progress.emit("✅ 레이어 인쇄 완료!")
            
            self.finished.emit(True)
            
        except Exception as e:
            self.error.emit(f"인쇄 오류: {str(e)}")
        finally:
            # 강화된 리소스 정리
            if printer is not None:
                try:
                    self.progress.emit("프린터 리소스 정리 중...")
                    printer.cleanup_and_close()
                    self.progress.emit("리소스 정리 완료")
                except Exception as cleanup_error:
                    print(f"리소스 정리 중 오류: {cleanup_error}")