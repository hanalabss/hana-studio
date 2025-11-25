"""
printer/printer_thread.py 수정
개별 면 방향 지원 - 앞면과 뒷면이 서로 다른 방향을 가질 수 있음
"""

import time
from typing import Optional
from PySide6.QtCore import QThread, Signal
from config import config
from .r600_printer import R600Printer
from .exceptions import R600PrinterError


class PrinterThread(QThread):
    """개별 면 방향을 지원하는 프린터 작업 스레드"""
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
                    quantity: int = 1,
                    front_orientation: str = "portrait",  # 개별 면 방향 추가
                    back_orientation: str = "portrait",   # 개별 면 방향 추가
                    adjusted_x: float = 0.0,              # 위치 조정값 추가
                    adjusted_y: float = 0.0,              # 위치 조정값 추가
                    selected_printer = None):             # 선택된 프린터 정보
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
            self.selected_printer = selected_printer    # 선택된 프린터
            self.should_stop = False

    
    def stop_printing(self):
        """인쇄 중단 요청"""
        self.should_stop = True
    
    def run(self):
        """스레드 실행 - 개별 면 방향에 따른 크기 계산"""
        printer = None
        successful_prints = 0
        
        try:
            self.progress.emit("프린터 초기화 중...")
            printer = R600Printer(self.dll_path, selected_printer=self.selected_printer)

            # 프린터 선택 (enum 후에 해야 안정적)
            self.progress.emit("프린터 목록 조회 중...")
            printers = printer.enum_printers()

            if not printers:
                self.error.emit("사용 가능한 프린터가 없습니다.")
                return

            # selected_printer가 있으면 자동 선택, 없으면 첫 번째 프린터 선택
            if self.selected_printer:
                self.progress.emit(f"프린터 자동 선택: {self.selected_printer.name}")
                success = printer.auto_select_printer()
                if not success:
                    # 자동 선택 실패 시 수동 선택으로 폴백
                    self.progress.emit(f"⚠️ 자동 선택 실패, 수동 선택으로 전환: {printers[0]}")
                    printer.select_printer(printers[0])
                else:
                    self.progress.emit(f"✅ 프린터 자동 선택 성공: {self.selected_printer.name}")
            else:
                self.progress.emit(f"프린터 선택: {printers[0]}")
                printer.select_printer(printers[0])
            
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
                
                try:
                    # 진행상황 시그널 발송
                    self.print_progress.emit(card_num - 1, self.quantity)
                    
                    # 카드별 인쇄 실행 - 개별 면 방향 전달
                    self._print_single_card(printer, card_num)
                    
                    successful_prints += 1
                    self.card_completed.emit(card_num)
                    
                    # 마지막 카드가 아닌 경우 잠시 대기
                    if card_num < self.quantity:
                        self.progress.emit(f"✅ {card_num}장 완료! 다음 카드 준비 중...")
                        time.sleep(1)  # 프린터 안정화를 위한 대기
                    
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
                    
    def _print_single_card(self, printer: R600Printer, card_num: int):
        """단일 카드 인쇄 - 개별 면 방향 정보 포함"""
        front_orientation_text = "세로형" if self.front_orientation == "portrait" else "가로형"
        back_orientation_text = "세로형" if self.back_orientation == "portrait" else "가로형"
        side_text = "양면" if self.is_dual_side else "단면"
        mode_text = "레이어" if self.print_mode == "layered" else "일반"
        
        if self.is_dual_side:
            self.progress.emit(f"📄 {card_num}번째 {side_text} {mode_text} 카드 인쇄 중 (앞면:{front_orientation_text}, 뒷면:{back_orientation_text})")
        else:
            self.progress.emit(f"📄 {card_num}번째 {side_text} {mode_text} 카드 인쇄 중 (앞면:{front_orientation_text})")
        
        if self.is_dual_side:
            # 양면 인쇄 - 개별 면 방향 전달
            if self.print_mode == "layered":
                printer.print_dual_side_card(
                    front_image_path=self.front_image_path,
                    back_image_path=self.back_image_path,
                    front_watermark_path=self.front_mask_path,
                    back_watermark_path=self.back_mask_path,
                    front_orientation=self.front_orientation,  # 개별 면 방향
                    back_orientation=self.back_orientation,    # 개별 면 방향
                    print_mode="layered"
                )
            else:
                printer.print_dual_side_card(
                    front_image_path=self.front_image_path,
                    back_image_path=self.back_image_path,
                    front_watermark_path=None,
                    back_watermark_path=None,
                    front_orientation=self.front_orientation,  # 개별 면 방향
                    back_orientation=self.back_orientation,    # 개별 면 방향
                    print_mode="normal"
                )
        else:
            # 단면 인쇄 - 앞면 방향만 전달
            if self.print_mode == "layered":
                if not self.front_mask_path:
                    raise R600PrinterError("레이어 인쇄를 위해서는 마스크 이미지가 필요합니다.")
                
                printer.print_single_side_card(
                    image_path=self.front_image_path,
                    watermark_path=self.front_mask_path,
                    card_orientation=self.front_orientation,  # 앞면 방향
                    print_mode="layered"
                )
            else:
                printer.print_single_side_card(
                    image_path=self.front_image_path,
                    watermark_path=None,
                    card_orientation=self.front_orientation,  # 앞면 방향
                    print_mode="normal"
                )
                
    def _handle_card_error(self, card_num: int, error: R600PrinterError) -> bool:
        """개별 카드 오류 처리"""
        self.progress.emit(f"⚠️ {card_num}번째 카드 인쇄 실패, 계속 진행합니다...")
        
        # 현재는 항상 계속 진행하도록 설정
        # 필요시 사용자에게 선택권을 줄 수 있음
        return True
    
    def _handle_final_result(self, successful_prints: int):
        """최종 결과 처리 - 사용자 친화적 메시지"""
        if self.should_stop:
            self.progress.emit(f"⏹️ 인쇄 중단됨 - 완료: {successful_prints}/{self.quantity}장")
            self.finished.emit(successful_prints > 0)
        elif successful_prints == self.quantity:
            # 모든 카드 성공 - 단순화
            self.progress.emit(f"🎉 모든 카드 인쇄 완료! ({self.quantity}장)")
            self.finished.emit(True)
        elif successful_prints > 0:
            # 일부 성공
            self.progress.emit(f"⚠️ 일부 완료 - 성공: {successful_prints}/{self.quantity}장")
            self.finished.emit(True)
        else:
            # 모두 실패
            self.progress.emit("❌ 카드 인쇄 실패")
            self.finished.emit(False)


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

    def generate_print_preview(self,
                              original_path: str,
                              mask_path: str,
                              show_layers: bool = True):
        """
        레이아웃 인쇄 미리보기 생성

        Args:
            original_path: 원본 이미지 경로
            mask_path: 마스크 이미지 경로
            show_layers: 레이어 구조 표시 여부

        Returns:
            미리보기 이미지 (3단계 합성) or None (실패 시)
        """
        import numpy as np
        import cv2
        from core.file_manager import FileManager

        try:
            file_mgr = FileManager()

            # 이미지 로드
            original = file_mgr._safe_imread(original_path)
            mask = file_mgr._safe_imread(mask_path)

            if original is None or mask is None:
                raise ValueError("이미지 로드 실패")

            # 크기 검증
            if original.shape[:2] != mask.shape[:2]:
                raise ValueError(f"원본과 마스크 크기 불일치: {original.shape[:2]} vs {mask.shape[:2]}")

            if not show_layers:
                # 단순 합성
                return self._create_simple_composite(original, mask)

            # 레이어 구조 표시 (3단계)
            h, w = original.shape[:2]

            # 캔버스 생성 (가로로 3개 배치 + 설명 공간)
            canvas_h = h + 100  # 상단 설명 공간
            canvas_w = w * 3 + 100  # 3개 이미지 + 간격
            canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

            # 1단계: 마스크 인쇄 (검은색 부분만)
            step1 = self._create_mask_print_preview(mask)
            canvas[100:100+h, 20:20+w] = step1

            # 텍스트 중앙 정렬을 위한 계산
            text1 = "1. Mask"
            text1_sub = "(black only)"
            text_size1 = cv2.getTextSize(text1, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
            text_size1_sub = cv2.getTextSize(text1_sub, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_x1 = 20 + (w - text_size1[0]) // 2
            text_x1_sub = 20 + (w - text_size1_sub[0]) // 2

            cv2.putText(canvas, text1, (text_x1, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
            cv2.putText(canvas, text1_sub, (text_x1_sub, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 2)

            # 2단계: 원본 오버레이
            step2 = original.copy()
            canvas[100:100+h, 40+w:40+w+w] = step2

            text2 = "2. Original"
            text_size2 = cv2.getTextSize(text2, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
            text_x2 = 40 + w + (w - text_size2[0]) // 2

            cv2.putText(canvas, text2, (text_x2, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

            # 3단계: 최종 결과
            step3 = self._create_final_composite(original, mask)
            canvas[100:100+h, 60+w*2:60+w*2+w] = step3

            text3 = "3. Final Result"
            text3_sub = "(mask + original)"
            text_size3 = cv2.getTextSize(text3, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
            text_size3_sub = cv2.getTextSize(text3_sub, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_x3 = 60 + w*2 + (w - text_size3[0]) // 2
            text_x3_sub = 60 + w*2 + (w - text_size3_sub[0]) // 2

            cv2.putText(canvas, text3, (text_x3, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
            cv2.putText(canvas, text3_sub, (text_x3_sub, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 2)

            return canvas

        except Exception as e:
            print(f"[ERROR] 인쇄 미리보기 생성 실패: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _create_mask_print_preview(self, mask):
        """마스크 인쇄 시뮬레이션 (검은색 부분만)"""
        import numpy as np
        import cv2

        # 흰색 배경 생성
        white_bg = np.ones_like(mask) * 255

        # 마스크의 검은색 부분만 표시
        gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        object_mask = (gray_mask < 128)

        result = white_bg.copy()
        result[object_mask] = [0, 0, 0]  # 검은색

        return result

    def _create_final_composite(self, original, mask):
        """최종 합성 이미지 (실제 인쇄 결과: W레이어 + YMC레이어)"""
        import numpy as np
        import cv2

        # 실제 프린터 동작 (YMCW 리본):
        # 1) W (White) 레이어 먼저 인쇄 - 마스크 검은색 부분만 흰색 베이스
        # 2) YMC (컬러) 레이어 나중 인쇄 - 원본 이미지 전체
        #
        # 결과:
        # - 마스크 검은색 영역 = W + YMC = 불투명하게 원본 색상 (100%)
        # - 마스크 흰색 영역 = YMC만 = 반투명하게 원본 색상 (50%)

        # 원본 이미지로 시작
        result = original.copy().astype(np.float32)

        # 마스크의 검은색/흰색 영역 구분
        gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        background_mask = (gray_mask >= 128)  # 흰색 = 배경 = W 레이어 없음

        # 배경 영역(W 레이어 없는 곳)은 반투명 효과
        # 흰색 배경과 블렌딩하여 반투명 효과 시뮬레이션
        white_bg = np.ones_like(result) * 255
        result[background_mask] = cv2.addWeighted(
            result[background_mask], 0.3,  # 원본 30%
            white_bg[background_mask], 0.7,  # 흰색 배경 70%
            0
        )

        return result.astype(np.uint8)

    def _create_simple_composite(self, original, mask):
        """단순 합성 (레이어 구조 없이)"""
        return self._create_final_composite(original, mask)


# 전역 인스턴스
print_manager = MultiCardPrintManager()