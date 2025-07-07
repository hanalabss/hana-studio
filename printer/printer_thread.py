"""
프린터 작업 스레드 - 양면 인쇄 지원
"""

from typing import Optional
from PySide6.QtCore import QThread, Signal
from config import config
from .r600_printer import R600Printer
from .exceptions import R600PrinterError


class PrinterThread(QThread):
    """양면 인쇄 지원 프린터 작업 스레드"""
    progress = Signal(str)
    finished = Signal(bool)
    error = Signal(str)
    
    def __init__(self, dll_path: str, 
                 front_image_path: str, 
                 back_image_path: Optional[str] = None,
                 front_mask_path: Optional[str] = None,
                 back_mask_path: Optional[str] = None,
                 print_mode: str = "normal",
                 is_dual_side: bool = True):
        super().__init__()
        self.dll_path = dll_path
        self.front_image_path = front_image_path
        self.back_image_path = back_image_path
        self.front_mask_path = front_mask_path
        self.back_mask_path = back_mask_path
        self.print_mode = print_mode
        self.is_dual_side = is_dual_side
    
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
            
            # 타임아웃 설정
            printer.set_timeout(15000)  # 15초로 증가
            
            # 카드 크기 설정
            card_width = config.get('printer.card_width', 53.98)
            card_height = config.get('printer.card_height', 85.6)
            
            # 양면/단면 인쇄 처리
            if self.is_dual_side:
                self.progress.emit("양면 카드 인쇄 시작...")
                
                # 양면 인쇄 실행
                if self.print_mode == "layered":
                    printer.print_dual_side_card(
                        front_image_path=self.front_image_path,
                        back_image_path=self.back_image_path,
                        front_watermark_path=self.front_mask_path,
                        back_watermark_path=self.back_mask_path,
                        card_width=card_width,
                        card_height=card_height,
                        print_mode="layered"
                    )
                    self.progress.emit("✅ 양면 레이어 인쇄 완료!")
                else:
                    printer.print_dual_side_card(
                        front_image_path=self.front_image_path,
                        back_image_path=self.back_image_path,
                        front_watermark_path=None,
                        back_watermark_path=None,
                        card_width=card_width,
                        card_height=card_height,
                        print_mode="normal"
                    )
                    self.progress.emit("✅ 양면 일반 인쇄 완료!")
            else:
                # 단면 인쇄 (하위 호환성)
                self.progress.emit("단면 카드 인쇄 시작...")
                
                if self.print_mode == "layered":
                    if not self.front_mask_path:
                        self.error.emit("레이어 인쇄를 위해서는 마스크 이미지가 필요합니다.")
                        return
                    
                    printer.print_single_side_card(
                        image_path=self.front_image_path,
                        watermark_path=self.front_mask_path,
                        card_width=card_width,
                        card_height=card_height,
                        print_mode="layered"
                    )
                    self.progress.emit("✅ 단면 레이어 인쇄 완료!")
                else:
                    printer.print_single_side_card(
                        image_path=self.front_image_path,
                        watermark_path=None,
                        card_width=card_width,
                        card_height=card_height,
                        print_mode="normal"
                    )
                    self.progress.emit("✅ 단면 일반 인쇄 완료!")
            
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