"""
printer/printer_thread.py 수정
개별 면 방향 지원 - 앞면과 뒷면이 서로 다른 방향을 가질 수 있음
"""

import time
from datetime import datetime
from typing import Any, Callable, List, Optional
from PySide6.QtCore import QThread, Signal
from config import config
from .r600_printer import R600Printer
from .exceptions import R600PrinterError
from .base import (
    CardJob, PrintCancelled, PrinterDevice, PrinterError, ProgressEvent, StructuredProgress,
    as_card_provider, provider_record_id,
)
from .printer_discovery import PrinterInfo


class PrinterThread(QThread):
    """개별 면 방향을 지원하는 프린터 작업 스레드"""
    progress = Signal(str)
    finished = Signal(bool)
    error = Signal(str)
    print_progress = Signal(int, int)  # 현재 장수, 전체 장수
    card_completed = Signal(int)  # 완료된 카드 번호
    step_progress = Signal(int, str)  # 전체 진행률(%), 상태 메시지
    card_result = Signal(int, bool, str)  # 카드 순번(1부터), 성공 여부, 오류 문구
    card_stage = Signal(int, object)  # 카드 순번(1부터), printer.base.ProgressEvent (단계 키·패널 포함)


    def __init__(self, dll_path: str, 
                    front_image_path: str, 
                    back_image_path: Optional[str] = None,
                    front_mask_path: Optional[str] = None,
                    back_mask_path: Optional[str] = None,
                    print_mode: str = "normal",
                    is_dual_side: bool = True,
                    quantity: int = 1,
                    front_orientation: str = "portrait",  # 개별 면 방향 추가
                    back_orientation: str = "portrait",   # 개별 면 방향 추가
                    adjusted_x: float = 0.0,              # 위치 조정값 추가
                    adjusted_y: float = 0.0,              # 위치 조정값 추가
                    selected_printer: Any = None,
                    device: Optional[PrinterDevice] = None,
                    card_jobs: Optional[List[CardJob]] = None,
                    stop_on_error: bool = True,
                    printer_factory: Optional[Callable[[PrinterDevice], Any]] = None,
                    r600_factory: Optional[Callable[[str, Optional[PrinterInfo]], Any]] = None):
            """selected_printer: 사용할 RTAI 프린터(PrinterInfo, PrinterDevice 또는 이름).
            device + card_jobs: CardPrinter 계열 장치로 카드마다 다른 CardJob을 인쇄한다.
              card_jobs는 CardJob 목록이거나 printer.base.CardProvider(카드를 찍기 직전에 만들어 준다).
            printer_factory / r600_factory: 프린터 객체 생성 함수(테스트·가짜 백엔드용)."""
            super().__init__()
            self.dll_path = dll_path
            self.front_image_path = front_image_path
            self.back_image_path = back_image_path
            self.front_mask_path = front_mask_path
            self.back_mask_path = back_mask_path
            self.print_mode = print_mode
            self.is_dual_side = is_dual_side
            self.quantity = quantity
            self.front_orientation = front_orientation  # 개별 면 방향
            self.back_orientation = back_orientation    # 개별 면 방향
            self.adjusted_x = adjusted_x                # 위치 조정값
            self.adjusted_y = adjusted_y                # 위치 조정값
            self.should_stop = False
            self.selected_printer = selected_printer
            self.device = device
            self.card_jobs = as_card_provider(card_jobs) if card_jobs is not None else None
            self.stop_on_error = stop_on_error
            self.printer_factory = printer_factory
            self.r600_factory = r600_factory
            # 카드별 결과 (index, ok, status, message, code, record_id, 시각). status: success|failed|cancelled
            self.results: List[dict] = []
            self._finished_emitted = False
            if self.card_jobs is not None:
                self.quantity = len(self.card_jobs)

    
    def stop_printing(self):
        """인쇄 중단 요청"""
        self.should_stop = True
    
    def _selected_printer_info(self) -> Optional[PrinterInfo]:
        """selected_printer / device 인자를 R600Printer가 받는 PrinterInfo로 바꾼다."""
        chosen = self.selected_printer if self.selected_printer is not None else self.device
        if chosen is None:
            return None
        if isinstance(chosen, PrinterInfo):
            return chosen
        if isinstance(chosen, PrinterDevice):
            connection = "TCP" if chosen.connection in ("tcp", "net") else "USB"
            return PrinterInfo(chosen.address or chosen.name, connection)
        return PrinterInfo(str(chosen), "USB")

    def _record_result(self, index: int, ok: bool, message: str = "", code=None,
                       record_id=None, started_at: Optional[str] = None,
                       status: Optional[str] = None):
        self.results.append({
            "index": index, "ok": ok, "status": status or ("success" if ok else "failed"),
            "message": message, "code": code,
            "record_id": record_id, "started_at": started_at,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
        })
        self.card_result.emit(index, ok, message)

    def _emit_finished(self, success: bool):
        self._finished_emitted = True
        self.finished.emit(success)

    def run(self):
        """스레드 실행 - 개별 면 방향에 따른 크기 계산"""
        if self.card_jobs is not None and self.device is not None:
            try:
                self._run_card_jobs()
            finally:
                if not self._finished_emitted:
                    self._emit_finished(False)
            return

        printer = None
        successful_prints = 0

        try:
            self.progress.emit("프린터 초기화 중...")
            selected_info = self._selected_printer_info()
            if self.r600_factory is not None:
                printer = self.r600_factory(self.dll_path, selected_info)
            elif selected_info is not None:
                printer = R600Printer(self.dll_path, selected_printer=selected_info)
            else:
                printer = R600Printer(self.dll_path)

            if selected_info is not None:
                # 사용자가 고른 프린터를 그대로 쓴다
                self.progress.emit(f"프린터 선택: {selected_info.name}")
                if printer.selected_printer_name != selected_info.name:
                    printer.select_printer(selected_info.name)
            else:
                self.progress.emit("프린터 목록 조회 중...")
                printers = printer.enum_printers()

                if not printers:
                    self.error.emit("사용 가능한 프린터가 없습니다.")
                    return

                self.progress.emit(f"프린터 선택: {printers[0]}")
                printer.select_printer(printers[0])
            printer.set_position_offset(self.adjusted_x, self.adjusted_y)
            
            # 타임아웃 설정
            printer.set_timeout(15000)  # 15초로 증가
            
            # 개별 면 방향 정보 표시
            front_orientation_text = "세로형" if self.front_orientation == "portrait" else "가로형"
            back_orientation_text = "세로형" if self.back_orientation == "portrait" else "가로형"
            
            # 여러장 인쇄 시작
            if self.is_dual_side:
                self.progress.emit(f"📄 총 {self.quantity}장 양면 인쇄 시작 (앞면:{front_orientation_text}, 뒷면:{back_orientation_text})")
            else:
                self.progress.emit(f"📄 총 {self.quantity}장 단면 인쇄 시작 (앞면:{front_orientation_text})")
            
            for card_num in range(1, self.quantity + 1):
                # 중단 요청 확인
                if self.should_stop:
                    self.progress.emit("❌ 사용자에 의해 인쇄가 중단되었습니다.")
                    break
                
                card_started_at = datetime.now().isoformat(timespec="seconds")
                try:
                    # 인쇄 시작 전 진행상황 시그널 발송
                    self.print_progress.emit(card_num - 1, self.quantity)

                    # 카드별 인쇄 실행 - 개별 면 방향 전달
                    self._print_single_card(printer, card_num)

                    successful_prints += 1
                    self._record_result(card_num, True, started_at=card_started_at)

                    # 인쇄 완료 후 진행상황 업데이트
                    self.print_progress.emit(card_num, self.quantity)
                    self.card_completed.emit(card_num)
                    
                    # 마지막 카드가 아닌 경우 잠시 대기
                    if card_num < self.quantity:
                        self.progress.emit(f"✅ {card_num}장 완료! 다음 카드 준비 중...")
                        time.sleep(1)  # 프린터 안정화를 위한 대기
                    
                except R600PrinterError as e:
                    self.progress.emit(f"❌ {card_num}번째 카드 인쇄 실패: {e}")
                    self._record_result(card_num, False, str(e), started_at=card_started_at)
                    # 개별 카드 실패 시에도 계속 진행할지 결정
                    continue_printing = self._handle_card_error(card_num, e)
                    if not continue_printing:
                        break
                except Exception as e:
                    self.progress.emit(f"❌ {card_num}번째 카드에서 예상치 못한 오류: {e}")
                    self._record_result(card_num, False, str(e), started_at=card_started_at)
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
            # 초기화 실패 등으로 finished가 나가지 않았으면 큐가 멈추지 않게 실패로 알린다
            if not self._finished_emitted:
                self._emit_finished(False)

    def _run_card_jobs(self):
        """CardPrinter 계열 장치로 카드마다 다른 CardJob을 차례로 인쇄한다."""
        printer = None
        successful_prints = 0
        total = len(self.card_jobs)
        try:
            self.progress.emit(f"프린터 연결 중: {self.device.name}")
            try:
                if self.printer_factory is not None:
                    printer = self.printer_factory(self.device)
                else:
                    from . import registry
                    printer = registry.open(self.device)
            except Exception as e:
                self.error.emit(f"프린터 연결 실패: {e}")
                return

            self.progress.emit(f"📄 총 {total}장 인쇄 시작")
            cards = self.card_jobs
            for card_num in range(1, total + 1):
                if self.should_stop:
                    self.progress.emit("❌ 사용자에 의해 인쇄가 중단되었습니다.")
                    break

                def on_event(event: ProgressEvent, _num=card_num):
                    overall = int(((_num - 1) + event.percent / 100) / total * 100)
                    self.step_progress.emit(overall, f"[{_num}/{total}] {event.text}")
                    self.card_stage.emit(_num, event)

                started_at = datetime.now().isoformat(timespec="seconds")
                self.print_progress.emit(card_num - 1, total)
                self.progress.emit(f"📄 {card_num}번째 카드 인쇄 중")
                record_id = provider_record_id(cards, card_num - 1)
                try:
                    # 공급자는 여기서(찍기 직전에) 카드를 만든다. 목록이면 그냥 꺼낸다.
                    job = cards[card_num - 1]
                    record_id = job.record_id
                    printer.print_card(job, progress=StructuredProgress(on_event))
                except PrintCancelled as e:
                    self.should_stop = True
                    self.progress.emit(f"⏹️ {card_num}번째 카드는 취소되어 인쇄하지 않았습니다.")
                    self._record_result(card_num, False, str(e), None, record_id, started_at,
                                        status="cancelled")
                    break
                except PrinterError as e:
                    self.progress.emit(f"❌ {card_num}번째 카드 인쇄 실패: {e}")
                    self._record_result(card_num, False, str(e), e.code, record_id, started_at)
                    if self.stop_on_error:
                        break
                    continue
                except Exception as e:
                    self.progress.emit(f"❌ {card_num}번째 카드에서 예상치 못한 오류: {e}")
                    self._record_result(card_num, False, str(e), None, record_id, started_at)
                    break

                successful_prints += 1
                self._record_result(card_num, True, "", None, record_id, started_at)
                self.print_progress.emit(card_num, total)
                self.card_completed.emit(successful_prints)

            self._handle_final_result(successful_prints)
        finally:
            if printer is not None:
                try:
                    printer.close()
                except Exception as cleanup_error:
                    print(f"리소스 정리 중 오류: {cleanup_error}")

    def _print_single_card(self, printer: R600Printer, card_num: int):
        """단일 카드 인쇄 - 개별 면 방향 정보 포함 + 진행률 콜백"""
        front_orientation_text = "세로형" if self.front_orientation == "portrait" else "가로형"
        back_orientation_text = "세로형" if self.back_orientation == "portrait" else "가로형"
        side_text = "양면" if self.is_dual_side else "단면"
        mode_text = "레이어" if self.print_mode == "layered" else "일반"

        if self.is_dual_side:
            self.progress.emit(f"📄 {card_num}번째 {side_text} {mode_text} 카드 인쇄 중 (앞면:{front_orientation_text}, 뒷면:{back_orientation_text})")
        else:
            self.progress.emit(f"📄 {card_num}번째 {side_text} {mode_text} 카드 인쇄 중 (앞면:{front_orientation_text})")

        # 진행률 콜백 함수 생성
        # 전체 진행률 = (완료된 카드 수 + 현재 카드 진행률/100) / 전체 카드 수 * 100
        def progress_callback(card_progress: int, message: str):
            completed_cards = card_num - 1  # 현재 카드는 아직 미완료
            overall_progress = int((completed_cards + card_progress / 100) / self.quantity * 100)
            status_message = f"[{card_num}/{self.quantity}] {message}"
            self.step_progress.emit(overall_progress, status_message)
            # 기존 방식 R600 흐름은 단계 키를 주지 않는다(STAGE_UNKNOWN).
            self.card_stage.emit(card_num, ProgressEvent(max(0, min(100, int(card_progress))), message))

        if self.is_dual_side:
            # 양면 인쇄 - 개별 면 방향 전달 + 콜백
            if self.print_mode == "layered":
                printer.print_dual_side_card(
                    front_image_path=self.front_image_path,
                    back_image_path=self.back_image_path,
                    front_watermark_path=self.front_mask_path,
                    back_watermark_path=self.back_mask_path,
                    front_orientation=self.front_orientation,
                    back_orientation=self.back_orientation,
                    print_mode="layered",
                    progress_callback=progress_callback
                )
            else:
                printer.print_dual_side_card(
                    front_image_path=self.front_image_path,
                    back_image_path=self.back_image_path,
                    front_watermark_path=None,
                    back_watermark_path=None,
                    front_orientation=self.front_orientation,
                    back_orientation=self.back_orientation,
                    print_mode="normal",
                    progress_callback=progress_callback
                )
        else:
            # 단면 인쇄 - 앞면 방향만 전달 + 콜백
            if self.print_mode == "layered":
                if not self.front_mask_path:
                    raise R600PrinterError("레이어 인쇄를 위해서는 마스크 이미지가 필요합니다.")

                printer.print_single_side_card(
                    image_path=self.front_image_path,
                    watermark_path=self.front_mask_path,
                    card_orientation=self.front_orientation,
                    print_mode="layered",
                    progress_callback=progress_callback
                )
            else:
                printer.print_single_side_card(
                    image_path=self.front_image_path,
                    watermark_path=None,
                    card_orientation=self.front_orientation,
                    print_mode="normal",
                    progress_callback=progress_callback
                )
                
    def _handle_card_error(self, card_num: int, error: R600PrinterError) -> bool:
        """개별 카드 오류 처리"""
        self.progress.emit(f"⚠️ {card_num}번째 카드 인쇄 실패, 계속 진행합니다...")
        
        # 현재는 항상 계속 진행하도록 설정
        # 필요시 사용자에게 선택권을 줄 수 있음
        return True
    
    def _handle_final_result(self, successful_prints: int):
        """최종 결과 처리 - 사용자 친화적 메시지"""
        # 최종 진행률 100% 발송
        self.step_progress.emit(100, "인쇄 완료")

        if self.should_stop:
            self.progress.emit(f"⏹️ 인쇄 중단됨 ({successful_prints}장 완료)")
            self._emit_finished(successful_prints > 0)
        elif successful_prints == self.quantity:
            # 모든 카드 성공
            self.progress.emit(f"🎉 {self.quantity}장 인쇄 완료!")
            self._emit_finished(True)
        elif successful_prints > 0:
            # 일부 성공
            self.progress.emit(f"⚠️ {self.quantity}장 중 {successful_prints}장 인쇄됨")
            self._emit_finished(True)
        else:
            # 모두 실패
            self.progress.emit("❌ 인쇄 실패")
            self._emit_finished(False)


class MultiCardPrintManager:
    """여러장 인쇄 관리 클래스 - 개별 면 방향 지원"""
    
    def __init__(self):
        self.current_thread = None
        self.is_printing = False
    
    def start_multi_print(self, **kwargs) -> PrinterThread:
        """여러장 인쇄 시작 - 개별 면 방향 정보 포함"""
        if self.is_printing:
            raise RuntimeError("이미 인쇄가 진행 중입니다.")
        
        # 하위 호환성을 위해 card_orientation이 있으면 변환
        if 'card_orientation' in kwargs:
            card_orientation = kwargs.pop('card_orientation')
            # 기존 전역 방향을 개별 면 방향으로 변환
            if 'front_orientation' not in kwargs:
                kwargs['front_orientation'] = card_orientation
            if 'back_orientation' not in kwargs:
                kwargs['back_orientation'] = card_orientation
        
        self.current_thread = PrinterThread(**kwargs)  # 개별 면 방향 매개변수 포함
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