"""
프린터 작업 스레드 - 양면 인쇄 및 여러장 인쇄 지원
"""

import time
from typing import Optional
from PySide6.QtCore import QThread, Signal
from config import config
from .r600_printer import R600Printer
from .exceptions import R600PrinterError


class PrinterThread(QThread):
    """양면 인쇄 및 여러장 인쇄 지원 프린터 작업 스레드"""
    progress = Signal(str)
    finished = Signal(bool)
    error = Signal(str)
    print_progress = Signal(int, int)  # 현재 장수, 전체 장수
    card_completed = Signal(int)  # 완료된 카드 번호
    
    def __init__(self, dll_path: str, 
                 front_image_path: str, 
                 back_image_path: Optional[str] = None,
                 front_mask_path: Optional[str] = None,
                 back_mask_path: Optional[str] = None,
                 print_mode: str = "normal",
                 is_dual_side: bool = True,
                 quantity: int = 1):
        super().__init__()
        self.dll_path = dll_path
        self.front_image_path = front_image_path
        self.back_image_path = back_image_path
        self.front_mask_path = front_mask_path
        self.back_mask_path = back_mask_path
        self.print_mode = print_mode
        self.is_dual_side = is_dual_side
        self.quantity = quantity
        self.should_stop = False
    
    def stop_printing(self):
        """인쇄 중단 요청"""
        self.should_stop = True
    
    def run(self):
        """스레드 실행"""
        printer = None
        successful_prints = 0
        
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
            
            # 여러장 인쇄 시작
            self.progress.emit(f"📄 총 {self.quantity}장 인쇄 시작...")
            
            for card_num in range(1, self.quantity + 1):
                # 중단 요청 확인
                if self.should_stop:
                    self.progress.emit("❌ 사용자에 의해 인쇄가 중단되었습니다.")
                    break
                
                try:
                    # 진행상황 시그널 발송
                    self.print_progress.emit(card_num - 1, self.quantity)
                    
                    # 카드별 인쇄 실행
                    self._print_single_card(printer, card_num, card_width, card_height)
                    
                    successful_prints += 1
                    self.card_completed.emit(card_num)
                    
                    # 마지막 카드가 아닌 경우 잠시 대기
                    if card_num < self.quantity:
                        self.progress.emit(f"✅ {card_num}장 완료! 다음 카드 준비 중...")
                        time.sleep(2)  # 프린터 안정화를 위한 대기
                    
                except R600PrinterError as e:
                    self.progress.emit(f"❌ {card_num}번째 카드 인쇄 실패: {e}")
                    # 개별 카드 실패 시에도 계속 진행할지 결정
                    continue_printing = self._handle_card_error(card_num, e)
                    if not continue_printing:
                        break
                except Exception as e:
                    self.progress.emit(f"❌ {card_num}번째 카드에서 예상치 못한 오류: {e}")
                    break
            
            # 최종 결과 처리
            self._handle_final_result(successful_prints)
            
        except Exception as e:
            self.error.emit(f"인쇄 초기화 오류: {str(e)}")
        finally:
            # 강화된 리소스 정리
            if printer is not None:
                try:
                    self.progress.emit("프린터 리소스 정리 중...")
                    printer.cleanup_and_close()
                    self.progress.emit("리소스 정리 완료")
                except Exception as cleanup_error:
                    print(f"리소스 정리 중 오류: {cleanup_error}")
    
    def _print_single_card(self, printer: R600Printer, card_num: int, 
                          card_width: float, card_height: float):
        """단일 카드 인쇄"""
        side_text = "양면" if self.is_dual_side else "단면"
        mode_text = "레이어" if self.print_mode == "layered" else "일반"
        
        self.progress.emit(f"📄 {card_num}번째 카드 {side_text} {mode_text} 인쇄 중...")
        
        if self.is_dual_side:
            # 양면 인쇄
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
        else:
            # 단면 인쇄
            if self.print_mode == "layered":
                if not self.front_mask_path:
                    raise R600PrinterError("레이어 인쇄를 위해서는 마스크 이미지가 필요합니다.")
                
                printer.print_single_side_card(
                    image_path=self.front_image_path,
                    watermark_path=self.front_mask_path,
                    card_width=card_width,
                    card_height=card_height,
                    print_mode="layered"
                )
            else:
                printer.print_single_side_card(
                    image_path=self.front_image_path,
                    watermark_path=None,
                    card_width=card_width,
                    card_height=card_height,
                    print_mode="normal"
                )
    
    def _handle_card_error(self, card_num: int, error: R600PrinterError) -> bool:
        """개별 카드 오류 처리"""
        self.progress.emit(f"⚠️ {card_num}번째 카드 인쇄 실패, 계속 진행합니다...")
        
        # 현재는 항상 계속 진행하도록 설정
        # 필요시 사용자에게 선택권을 줄 수 있음
        return True
    
    def _handle_final_result(self, successful_prints: int):
        """최종 결과 처리"""
        if self.should_stop:
            self.progress.emit(f"🛑 인쇄 중단됨 - 완료: {successful_prints}/{self.quantity}장")
            self.finished.emit(successful_prints > 0)
        elif successful_prints == self.quantity:
            # 모든 카드 성공
            side_text = "양면" if self.is_dual_side else "단면"
            mode_text = "레이어" if self.print_mode == "layered" else "일반"
            self.progress.emit(f"🎉 모든 카드 인쇄 완료! ({self.quantity}장 {side_text} {mode_text} 인쇄)")
            self.finished.emit(True)
        elif successful_prints > 0:
            # 일부 성공
            self.progress.emit(f"⚠️ 일부 완료 - 성공: {successful_prints}/{self.quantity}장")
            self.finished.emit(True)
        else:
            # 모두 실패
            self.progress.emit("❌ 모든 카드 인쇄 실패")
            self.finished.emit(False)


class MultiCardPrintManager:
    """여러장 인쇄 관리 클래스"""
    
    def __init__(self):
        self.current_thread = None
        self.is_printing = False
    
    def start_multi_print(self, **kwargs) -> PrinterThread:
        """여러장 인쇄 시작"""
        if self.is_printing:
            raise RuntimeError("이미 인쇄가 진행 중입니다.")
        
        self.current_thread = PrinterThread(**kwargs)
        self.is_printing = True
        
        # 완료 시 상태 초기화
        self.current_thread.finished.connect(self._on_print_finished)
        self.current_thread.error.connect(self._on_print_finished)
        
        return self.current_thread
    
    def stop_current_print(self):
        """현재 인쇄 중단"""
        if self.current_thread and self.is_printing:
            self.current_thread.stop_printing()
    
    def _on_print_finished(self):
        """인쇄 완료 시 상태 초기화"""
        self.is_printing = False
        self.current_thread = None
    
    def get_print_status(self) -> dict:
        """현재 인쇄 상태 반환"""
        return {
            'is_printing': self.is_printing,
            'has_thread': self.current_thread is not None
        }


# 전역 인스턴스
print_manager = MultiCardPrintManager()