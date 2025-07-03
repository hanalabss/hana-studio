"""
RTAI LUKA 프린터 연동 모듈
검증된 작동 코드를 Hana Studio 스타일로 통합
일반 인쇄와 레이어 인쇄(YMCW)를 분리하여 처리
"""

import ctypes
import os
import time
from typing import List, Optional
from PySide6.QtCore import QThread, Signal
from config import config


class R600PrinterError(Exception):
    """R600 프린터 관련 예외"""
    pass


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
                    printer.cleanup_and_close()  # 새로운 정리 메서드
                    self.progress.emit("리소스 정리 완료")
                except Exception as cleanup_error:
                    print(f"리소스 정리 중 오류: {cleanup_error}")

class R600Printer:
    """개선된 R600 프린터 제어 클래스 - 리소스 관리 강화"""
    
    def __init__(self, dll_path: str = './libDSRetransfer600App.dll'):
        """R600 프린터 초기화"""
        self.lib = None
        self.selected_printer = None
        self.committed_img_info = None
        self.is_initialized = False

        try:
            print(f"[DEBUG] DLL 경로 시도: {dll_path}")
            if not os.path.exists(dll_path):
                raise FileNotFoundError(f"DLL 파일이 존재하지 않음: {dll_path}")

            self.lib = ctypes.CDLL(dll_path)
            print("[DEBUG] DLL 로드 성공")

            self._setup_function_signatures()
            self._initialize_library()
            self.is_initialized = True
            
        except Exception as e:
            print(f"[DEBUG] 예외 발생 위치: {type(e).__name__}: {e}")
            raise R600PrinterError(f"프린터 초기화 실패: {e}")

    def cleanup_and_close(self):
        """강화된 리소스 정리"""
        try:
            if not self.is_initialized:
                return
                
            print("[DEBUG] 프린터 리소스 정리 시작...")
            
            # 1. 카드 상태 확인 및 정리
            try:
                # 혹시 카드가 남아있다면 배출 시도
                self.eject_card()
                print("[DEBUG] 카드 배출 완료")
            except:
                print("[DEBUG] 카드 배출 건너뜀 (정상)")
            
            # 2. 라이브러리 정리
            if self.lib:
                ret = self.lib.R600LibClear()
                print(f"[DEBUG] 라이브러리 정리 결과: {ret}")
                
                # 3. 잠깐 대기 (중요!)
                time.sleep(2)  # DLL 내부 정리 시간 확보
                
            # 4. 상태 초기화
            self.selected_printer = None
            self.committed_img_info = None
            self.is_initialized = False
            
            print("[DEBUG] 프린터 리소스 정리 완료")
            
        except Exception as e:
            print(f"[DEBUG] 리소스 정리 중 오류: {e}")
    
    def close(self):
        """기존 close 메서드 - cleanup_and_close 호출"""
        self.cleanup_and_close()

    def print_normal_card(self, image_path: str, card_width: float = 53.98, card_height: float = 85.6):
        """일반 카드 인쇄 - 리소스 정리 강화"""
        try:
            print("=== 일반 카드 인쇄 시작 ===")
            
            # 1. 카드 삽입
            self.inject_card()
            
            # 2. 리본 옵션 설정
            self.set_ribbon_option(ribbon_type=1, key=0, value="2")
            
            # 3. 캔버스 설정
            self.setup_canvas()
            
            # 4. 원본 이미지를 워터마크로 먼저 그리기
            self.draw_watermark(0.0, 0.0, card_width, card_height, "")
            
            # 5. 원본 이미지 그리기
            self.draw_image(0.0, 0.0, card_width, card_height, image_path)
            
            # 6. 캔버스 커밋
            self.commit_canvas()
            
            # 7. 잠시 대기
            time.sleep(1)
            
            # 8. 인쇄 실행
            self.print_draw()
            
            # 9. 인쇄 완료 대기 (추가)
            time.sleep(2)  # 인쇄 완료 대기
            
            # 10. 카드 배출
            self.eject_card()
            
            # 11. 배출 완료 대기 (추가)
            time.sleep(1)  # 배출 완료 대기
            
            print("=== 일반 카드 인쇄 완료 ===")
            
        except R600PrinterError as e:
            print(f"일반 카드 인쇄 중 오류 발생: {e}")
            # 오류 발생 시에도 카드 배출 시도
            try:
                self.eject_card()
            except:
                pass
            raise

    def print_layered_card(self, watermark_path: str, image_path: Optional[str] = None,
                          card_width: float = 53.98, card_height: float = 85.6):
        """레이어 카드 인쇄 - 리소스 정리 강화"""
        try:
            print("=== 레이어 카드 인쇄 시작 (YMCW) ===")
            
            # 1. 카드 삽입
            self.inject_card()
            
            # 2. 리본 옵션 설정
            self.set_ribbon_option(ribbon_type=1, key=0, value="2")
            
            # 3. 캔버스 설정
            self.setup_canvas()
            
            # 4. 워터마크 레이어 그리기
            if watermark_path:
                self.draw_watermark(0.0, 0.0, card_width, card_height, watermark_path)
                
            # 5. 베이스 이미지 그리기
            if image_path:
                self.draw_image(0.0, 0.0, card_width, card_height, image_path)
            
            # 6. 캔버스 커밋
            self.commit_canvas()
            
            # 7. 잠시 대기
            time.sleep(1)
            
            # 8. 인쇄 실행
            self.print_draw()
            
            # 9. 인쇄 완료 대기 (추가)
            time.sleep(2)  # 인쇄 완료 대기
            
            # 10. 카드 배출
            self.eject_card()
            
            # 11. 배출 완료 대기 (추가)
            time.sleep(1)  # 배출 완료 대기
            
            print("=== 레이어 카드 인쇄 완료 (YMCW) ===")
            
        except R600PrinterError as e:
            print(f"레이어 카드 인쇄 중 오류 발생: {e}")
            # 오류 발생 시에도 카드 배출 시도
            try:
                self.eject_card()
            except:
                pass
            raise


class R600Printer:
    """개선된 R600 프린터 제어 클래스"""
    
    def __init__(self, dll_path: str = './libDSRetransfer600App.dll'):
        """
        R600 프린터 초기화

        Args:
            dll_path: DLL 파일 경로
        """
        self.lib = None
        self.selected_printer = None
        self.committed_img_info = None

        try:
            print(f"[DEBUG] DLL 경로 시도: {dll_path}")
            if not os.path.exists(dll_path):
                raise FileNotFoundError(f"DLL 파일이 존재하지 않음: {dll_path}")

            self.lib = ctypes.CDLL(dll_path)
            print("[DEBUG] DLL 로드 성공")

            self._setup_function_signatures()
            self._initialize_library()
        except Exception as e:
            print(f"[DEBUG] 예외 발생 위치: {type(e).__name__}: {e}")
            raise R600PrinterError(f"프린터 초기화 실패: {e}")

    def _setup_function_signatures(self):
        """함수 시그니처 정의"""
        # 프린터 열거
        self.lib.R600EnumTcpPrt.argtypes = [
            ctypes.POINTER(ctypes.c_char), 
            ctypes.POINTER(ctypes.c_uint), 
            ctypes.POINTER(ctypes.c_int)
        ]
        self.lib.R600EnumTcpPrt.restype = ctypes.c_uint
        
        # 타임아웃 설정
        self.lib.R600TcpSetTimeout.argtypes = [ctypes.c_int, ctypes.c_int]
        self.lib.R600TcpSetTimeout.restype = ctypes.c_uint
        
        # 프린터 선택
        self.lib.R600SelectPrt.argtypes = [ctypes.POINTER(ctypes.c_char)]
        self.lib.R600SelectPrt.restype = ctypes.c_uint
        
        # 카드 관련
        self.lib.R600CardInject.argtypes = [ctypes.c_int]
        self.lib.R600CardInject.restype = ctypes.c_uint
        
        self.lib.R600CardEject.argtypes = [ctypes.c_int]
        self.lib.R600CardEject.restype = ctypes.c_uint
        
        # 리본 옵션
        self.lib.R600SetRibbonOpt.argtypes = [
            ctypes.c_ubyte, ctypes.c_uint, ctypes.c_char_p, ctypes.c_uint
        ]
        self.lib.R600SetRibbonOpt.restype = ctypes.c_uint
        
        # 캔버스 관련
        self.lib.R600SetCanvasPortrait.argtypes = [ctypes.c_int]
        self.lib.R600SetCanvasPortrait.restype = ctypes.c_uint
        
        self.lib.R600PrepareCanvas.argtypes = [ctypes.c_int, ctypes.c_int]
        self.lib.R600PrepareCanvas.restype = ctypes.c_uint
        
        self.lib.R600CommitCanvas.argtypes = [
            ctypes.POINTER(ctypes.c_char), 
            ctypes.POINTER(ctypes.c_uint)
        ]
        self.lib.R600CommitCanvas.restype = ctypes.c_uint
        
        # 이미지 관련
        self.lib.R600SetImagePara.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_float]
        self.lib.R600SetImagePara.restype = ctypes.c_uint
        
        self.lib.R600DrawWaterMark.argtypes = [
            ctypes.c_double, ctypes.c_double, ctypes.c_double, 
            ctypes.c_double, ctypes.c_char_p
        ]
        self.lib.R600DrawWaterMark.restype = ctypes.c_uint
        
        self.lib.R600DrawImage.argtypes = [
            ctypes.c_double, ctypes.c_double, ctypes.c_double, 
            ctypes.c_double, ctypes.c_char_p, ctypes.c_int
        ]
        self.lib.R600DrawImage.restype = ctypes.c_uint
        
        self.lib.R600DrawLayerWhite.argtypes = [
            ctypes.c_double, ctypes.c_double, ctypes.c_double, 
            ctypes.c_double, ctypes.c_char_p
        ]
        self.lib.R600DrawLayerWhite.restype = ctypes.c_uint
        
        # 인쇄
        self.lib.R600PrintDraw.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        self.lib.R600PrintDraw.restype = ctypes.c_uint
    
    def _initialize_library(self):
        """라이브러리 초기화"""
        ret = self.lib.R600LibInit()
        if ret != 0:
            raise R600PrinterError(f"라이브러리 초기화 실패: {ret}")
        print(f"라이브러리 초기화 성공: {ret}")
    
    def _check_result(self, result: int, operation: str):
        """결과 코드 확인"""
        if result == 0:
            print(f"{operation} 성공")
        else:
            raise R600PrinterError(f"{operation} 실패: 오류 코드 {result}")
    
    def enum_printers(self) -> List[str]:
        """TCP 프린터 목록 조회"""
        list_buffer_size = 1024
        printer_list_buffer = ctypes.create_string_buffer(list_buffer_size)
        enum_list_len = ctypes.c_uint(list_buffer_size)
        num_printers = ctypes.c_int()
        
        ret = self.lib.R600EnumTcpPrt(
            printer_list_buffer, 
            ctypes.byref(enum_list_len), 
            ctypes.byref(num_printers)
        )
        
        if ret != 0:
            raise R600PrinterError(f"프린터 열거 실패: {ret}")
        
        actual_len = enum_list_len.value
        printer_count = num_printers.value
        
        if actual_len > 0 and printer_count > 0:
            printer_names_str = printer_list_buffer.value.decode('cp949')
            printer_names = [name.strip() for name in printer_names_str.split('\n') if name.strip()]
            
            print(f"발견된 프린터 수: {printer_count}")
            print("프린터 목록:")
            for name in printer_names:
                print(f"- {name}")
            
            return printer_names
        else:
            print("프린터를 찾을 수 없습니다.")
            return []
    
    def set_timeout(self, timeout_ms: int = 10000):
        """타임아웃 설정"""
        ret = self.lib.R600TcpSetTimeout(timeout_ms, timeout_ms)
        self._check_result(ret, "타임아웃 설정")
    
    def select_printer(self, printer_name: str):
        """프린터 선택"""
        ret = self.lib.R600SelectPrt(printer_name.encode('cp949'))
        self._check_result(ret, f"프린터 선택 ({printer_name})")
        self.selected_printer = printer_name
    
    def inject_card(self):
        """카드 삽입"""
        ret = self.lib.R600CardInject(0)
        self._check_result(ret, "카드 삽입")
    
    def eject_card(self):
        """카드 배출"""
        ret = self.lib.R600CardEject(0)
        self._check_result(ret, "카드 배출")
    
    def set_ribbon_option(self, ribbon_type: int = 1, key: int = 0, value: str = "2"):
        """리본 옵션 설정"""
        value_bytes = value.encode('cp949')
        ret = self.lib.R600SetRibbonOpt(ribbon_type, key, value_bytes, len(value_bytes))
        self._check_result(ret, "리본 옵션 설정")
    
    def setup_canvas(self, portrait: bool = True):
        """캔버스 설정"""
        # 세로 방향 설정
        ret = self.lib.R600SetCanvasPortrait(1 if portrait else 0)
        self._check_result(ret, "캔버스 방향 설정")
        
        # 캔버스 준비
        ret = self.lib.R600PrepareCanvas(0, 0)
        self._check_result(ret, "캔버스 준비")
        
        # 이미지 파라미터 설정
        ret = self.lib.R600SetImagePara(1, 0, 0.0)
        self._check_result(ret, "이미지 파라미터 설정")
    
    def draw_watermark(self, x: float, y: float, width: float, height: float, image_path: str):
        """워터마크 그리기"""
        if not os.path.exists(image_path):
            raise R600PrinterError(f"이미지 파일 {image_path}이 존재하지 않습니다.")
        
        ret = self.lib.R600DrawWaterMark(x, y, width, height, image_path.encode('cp949'))
        self._check_result(ret, f"워터마크 그리기 ({image_path})")
    
    def draw_image(self, x: float, y: float, width: float, height: float, 
                   image_path: str, mode: int = 1):
        """이미지 그리기"""
        if not os.path.exists(image_path):
            raise R600PrinterError(f"이미지 파일 {image_path}이 존재하지 않습니다.")
        
        ret = self.lib.R600DrawImage(x, y, width, height, image_path.encode('cp949'), mode)
        self._check_result(ret, f"이미지 그리기 ({image_path})")
    
    def draw_layer_white(self, x: float, y: float, width: float, height: float, image_path: str):
        """화이트 레이어 그리기"""
        if not os.path.exists(image_path):
            raise R600PrinterError(f"이미지 파일 {image_path}이 존재하지 않습니다.")
        
        ret = self.lib.R600DrawLayerWhite(x, y, width, height, image_path.encode('cp949'))
        self._check_result(ret, f"화이트 레이어 그리기 ({image_path})")
    
    def commit_canvas(self) -> str:
        """캔버스 커밋"""
        img_info_buffer_size = 200
        img_info_buffer = ctypes.create_string_buffer(img_info_buffer_size)
        p_img_info_len = ctypes.pointer(ctypes.c_uint(img_info_buffer_size))
        
        ret = self.lib.R600CommitCanvas(img_info_buffer, p_img_info_len)
        if ret != 0:
            raise R600PrinterError(f"캔버스 커밋 실패: {ret}")
        
        self.committed_img_info = img_info_buffer.value.decode('cp949')
        print(f"캔버스 커밋 성공. 이미지 정보: {self.committed_img_info}")
        print(f"실제 이미지 정보 길이: {p_img_info_len.contents.value}")
        
        return self.committed_img_info
    
    def print_draw(self, img_info: Optional[str] = None):
        """인쇄 실행"""
        if img_info is None:
            if self.committed_img_info is None:
                raise R600PrinterError("커밋된 이미지 정보가 없습니다. 먼저 commit_canvas()를 호출하세요.")
            img_info = self.committed_img_info
        
        ret = self.lib.R600PrintDraw(img_info.encode('cp949'), ctypes.c_char_p(None))
        self._check_result(ret, "인쇄 실행")
    
    def print_normal_card(self, image_path: str, card_width: float = 53.98, card_height: float = 85.6):
        """
        일반 카드 인쇄 - 원본 이미지를 워터마크로도 사용 (타임아웃 문제 해결)
        
        Args:
            image_path: 원본 이미지 경로
            card_width: 카드 너비 (mm)
            card_height: 카드 높이 (mm)
        """
        try:
            print("=== 일반 카드 인쇄 시작 ===")
            
            # 1. 카드 삽입
            self.inject_card()
            
            # 2. 리본 옵션 설정
            self.set_ribbon_option(ribbon_type=1, key=0, value="2")  # YMCW 사용
            
            # 3. 캔버스 설정
            self.setup_canvas()
            
            # 4. 원본 이미지를 워터마크로 먼저 그리기 (타임아웃 방지)
            # self.draw_watermark(0.0, 0.0, card_width, card_height, image_path)
            
            # 5. 원본 이미지 그리기
            self.draw_image(0.0, 0.0, card_width, card_height, image_path)
            
            # 6. 캔버스 커밋
            self.commit_canvas()
            
            # 7. 잠시 대기
            time.sleep(1)
            
            # 8. 인쇄 실행
            self.print_draw()
            
            # 9. 카드 배출
            self.eject_card()
            
            print("=== 일반 카드 인쇄 완료 ===")
            
        except R600PrinterError as e:
            print(f"일반 카드 인쇄 중 오류 발생: {e}")
            raise
        
    def print_layered_card(self, watermark_path: str, image_path: Optional[str] = None,
                          card_width: float = 53.98, card_height: float = 85.6):
        """
        레이어 카드 인쇄 - 원본 이미지 + 마스크 워터마크 레이어
        
        Args:
            watermark_path: 워터마크(마스크) 이미지 경로
            image_path: 원본 이미지 경로 (선택사항)
            card_width: 카드 너비 (mm)
            card_height: 카드 높이 (mm)
        """
        try:
            print("=== 레이어 카드 인쇄 시작 (YMCW) ===")
            
            # 1. 카드 삽입
            self.inject_card()
            
            # 2. 리본 옵션 설정 (레이어 인쇄용 - YMCW 모두 사용)
            self.set_ribbon_option(ribbon_type=1, key=0, value="2")  # YMCW 사용
            
            # 3. 캔버스 설정
            self.setup_canvas()
            
            # 3. 워터마크 레이어 그리기 (마스크)
            if watermark_path:
                self.draw_watermark(0.0, 0.0, card_width, card_height, watermark_path)
                
            # 4. 베이스 이미지 그리기 (원본 이미지가 있는 경우)
            if image_path:
                self.draw_image(0.0, 0.0, card_width, card_height, image_path)
            
            # 5. 캔버스 커밋
            self.commit_canvas()
            
            # 6. 잠시 대기
            time.sleep(1)
            
            # 7. 인쇄 실행
            self.print_draw()
            
            # 8. 카드 배출
            self.eject_card()
            
            print("=== 레이어 카드 인쇄 완료 (YMCW) ===")
            
        except R600PrinterError as e:
            print(f"레이어 카드 인쇄 중 오류 발생: {e}")
            raise
    
    # 기존 print_card 메서드는 하위 호환성을 위해 유지 (deprecated)
    def print_card(self, watermark_path: Optional[str] = None, image_path: Optional[str] = None,
                   card_width: float = 53.98, card_height: float = 85.6):
        """
        기존 카드 인쇄 메서드 (하위 호환성용)
        워터마크가 있으면 레이어 인쇄, 없으면 일반 인쇄
        
        [DEPRECATED] print_normal_card() 또는 print_layered_card() 사용 권장
        """
        if watermark_path:
            self.print_layered_card(watermark_path, image_path, card_width, card_height)
        else:
            self.print_normal_card(image_path, card_width, card_height)
    
    def close(self):
        """라이브러리 정리"""
        if self.lib:
            ret = self.lib.R600LibClear()
            print(f"라이브러리 정리: {ret}")
    
    def __enter__(self):
        """컨텍스트 매니저 진입"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 매니저 종료"""
        self.close()


# Hana Studio 연동을 위한 유틸리티 함수들
def check_printer_dll(dll_path: str) -> bool:
    """DLL 파일 존재 여부 확인"""
    return os.path.exists(dll_path)


def get_default_dll_paths() -> List[str]:
    """기본 DLL 경로 목록 반환"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return [
        os.path.join(base_dir, 'libDSRetransfer600App.dll'),
        os.path.join(base_dir, 'dll', 'libDSRetransfer600App.dll'),
        os.path.join(base_dir, 'lib', 'libDSRetransfer600App.dll'),
        config.get('printer.dll_path', os.path.join(base_dir, 'libDSRetransfer600App.dll'))
    ]


def find_printer_dll() -> Optional[str]:
    """사용 가능한 프린터 DLL 찾기"""
    for path in get_default_dll_paths():
        if check_printer_dll(path):
            print(f"✅ 프린터 DLL 발견: {path}")
            return path
    print("❌ 프린터 DLL을 찾을 수 없습니다.")
    return None


def test_printer_connection() -> bool:
    """프린터 연결 테스트"""
    try:
        dll_path = find_printer_dll()
        if not dll_path:
            return False
        
        with R600Printer(dll_path) as printer:
            printers = printer.enum_printers()
            return len(printers) > 0
    except Exception as e:
        print(f"프린터 연결 테스트 실패: {e}")
        return False


def test_print_modes():
    """인쇄 모드 테스트 함수"""
    try:
        dll_path = find_printer_dll()
        if not dll_path:
            print("❌ DLL 파일을 찾을 수 없습니다.")
            return
        
        with R600Printer(dll_path) as printer:
            printers = printer.enum_printers()
            if not printers:
                print("❌ 사용 가능한 프린터가 없습니다.")
                return
            
            printer.set_timeout(10000)
            printer.select_printer(printers[0])
            
            # 테스트 이미지 경로 (실제 파일로 변경 필요)
            test_image = "test_image.jpg"
            test_mask = "test_mask.jpg"
            
            if os.path.exists(test_image):
                print("🖼️ 일반 인쇄 테스트...")
                printer.print_normal_card(test_image)
                
                if os.path.exists(test_mask):
                    print("🎭 레이어 인쇄 테스트...")
                    printer.print_layered_card(test_mask, test_image)
                else:
                    print("⚠️ 마스크 이미지가 없어 레이어 인쇄 테스트를 건너뜁니다.")
            else:
                print("⚠️ 테스트 이미지가 없어 인쇄 테스트를 건너뜁니다.")
                
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")


# 원본 코드의 메인 함수 (테스트용)
def main():
    """메인 실행 함수 - 개선된 테스트"""
    print("=== Hana Studio 프린터 연동 테스트 ===")
    
    # 기본 연결 테스트
    if test_printer_connection():
        print("✅ 프린터 연결 테스트 성공!")
        
        # 인쇄 모드 테스트 (선택사항)
        choice = input("인쇄 모드 테스트를 실행하시겠습니까? (y/N): ")
        if choice.lower() == 'y':
            test_print_modes()
    else:
        print("❌ 프린터 연결 테스트 실패!")


if __name__ == "__main__":
    main()