"""
프린터 연동 모듈 (향후 확장용)
R600 프린터와의 연동을 위한 기본 구조
"""

import ctypes
import os
import time
from typing import List, Optional, Tuple
from PySide6.QtCore import QThread, Signal


class R600PrinterError(Exception):
    """R600 프린터 관련 예외"""
    pass


class PrinterThread(QThread):
    """프린터 작업을 백그라운드에서 처리하는 스레드"""
    progress = Signal(str)
    finished = Signal(bool)
    error = Signal(str)
    
    def __init__(self, dll_path: str, image_path: str, mask_path: str):
        super().__init__()
        self.dll_path = dll_path
        self.image_path = image_path
        self.mask_path = mask_path
    
    def run(self):
        try:
            self.progress.emit("프린터 연결 중...")
            printer = R600Printer(self.dll_path)
            
            self.progress.emit("프린터 목록 조회 중...")
            printers = printer.enum_printers()
            
            if not printers:
                self.error.emit("사용 가능한 프린터가 없습니다.")
                return
            
            self.progress.emit(f"프린터 선택: {printers[0]}")
            printer.select_printer(printers[0])
            
            self.progress.emit("카드 인쇄 시작...")
            printer.print_card(
                watermark_path=self.mask_path,
                image_path=self.image_path
            )
            
            self.progress.emit("인쇄 완료!")
            self.finished.emit(True)
            
        except Exception as e:
            self.error.emit(f"인쇄 오류: {str(e)}")
        finally:
            if 'printer' in locals():
                printer.close()


class R600Printer:
    """R600 프린터 제어 클래스 (기존 ctypes_dev.py 기반)"""
    
    def __init__(self, dll_path: str = './libDSRetransfer600App.dll'):
        """
        R600 프린터 초기화
        
        Args:
            dll_path: DLL 파일 경로
        """
        self.lib = None
        self.selected_printer = None
        self.committed_img_info = None
        
        if not os.path.exists(dll_path):
            raise R600PrinterError(f"DLL 파일을 찾을 수 없습니다: {dll_path}")
        
        try:
            self.lib = ctypes.CDLL(dll_path)
            self._setup_function_signatures()
            self._initialize_library()
        except Exception as e:
            raise R600PrinterError(f"프린터 초기화 실패: {e}")
    
    def _setup_function_signatures(self):
        """함수 시그니처 정의"""
        # 라이브러리 초기화/정리
        self.lib.R600LibInit.restype = ctypes.c_uint
        self.lib.R600LibClear.restype = ctypes.c_uint
        
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
        
        # 캔버스 관련
        self.lib.R600PrepareCanvas.argtypes = [ctypes.c_int, ctypes.c_int]
        self.lib.R600PrepareCanvas.restype = ctypes.c_uint
        
        self.lib.R600CommitCanvas.argtypes = [
            ctypes.POINTER(ctypes.c_char), 
            ctypes.POINTER(ctypes.c_uint)
        ]
        self.lib.R600CommitCanvas.restype = ctypes.c_uint
        
        # 이미지 관련
        self.lib.R600DrawImage.argtypes = [
            ctypes.c_double, ctypes.c_double, ctypes.c_double, 
            ctypes.c_double, ctypes.c_char_p, ctypes.c_int
        ]
        self.lib.R600DrawImage.restype = ctypes.c_uint
        
        # 인쇄
        self.lib.R600PrintDraw.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        self.lib.R600PrintDraw.restype = ctypes.c_uint
    
    def _initialize_library(self):
        """라이브러리 초기화"""
        ret = self.lib.R600LibInit()
        if ret != 0:
            raise R600PrinterError(f"라이브러리 초기화 실패: {ret}")
    
    def _check_result(self, result: int, operation: str):
        """결과 코드 확인"""
        if result != 0:
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
            return printer_names
        else:
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
    
    def setup_canvas(self):
        """캔버스 설정"""
        ret = self.lib.R600PrepareCanvas(0, 0)
        self._check_result(ret, "캔버스 준비")
    
    def draw_image(self, x: float, y: float, width: float, height: float, 
                   image_path: str, mode: int = 1):
        """이미지 그리기"""
        if not os.path.exists(image_path):
            raise R600PrinterError(f"이미지 파일 {image_path}이 존재하지 않습니다.")
        
        ret = self.lib.R600DrawImage(x, y, width, height, image_path.encode('cp949'), mode)
        self._check_result(ret, f"이미지 그리기 ({image_path})")
    
    def commit_canvas(self) -> str:
        """캔버스 커밋"""
        img_info_buffer_size = 3072
        img_info_buffer = ctypes.create_string_buffer(img_info_buffer_size)
        p_img_info_len = ctypes.pointer(ctypes.c_uint(img_info_buffer_size))
        
        ret = self.lib.R600CommitCanvas(img_info_buffer, p_img_info_len)
        if ret != 0:
            raise R600PrinterError(f"캔버스 커밋 실패: {ret}")
        
        self.committed_img_info = img_info_buffer.value.decode('cp949')
        return self.committed_img_info
    
    def print_draw(self, img_info: Optional[str] = None):
        """인쇄 실행"""
        if img_info is None:
            if self.committed_img_info is None:
                raise R600PrinterError("커밋된 이미지 정보가 없습니다.")
            img_info = self.committed_img_info
        
        ret = self.lib.R600PrintDraw(img_info.encode('cp949'), ctypes.c_char_p(None))
        self._check_result(ret, "인쇄 실행")
    
    def print_card(self, watermark_path: Optional[str] = None, image_path: Optional[str] = None,
                   card_width: float = 53.98, card_height: float = 85.6):
        """
        카드 인쇄 전체 프로세스
        
        Args:
            watermark_path: 워터마크 이미지 경로 (마스크 이미지)
            image_path: 메인 이미지 경로 (원본 이미지)
            card_width: 카드 너비 (mm)
            card_height: 카드 높이 (mm)
        """
        try:
            # 1. 타임아웃 설정
            self.set_timeout(10000)
            
            # 2. 카드 삽입
            self.inject_card()
            time.sleep(2)
            
            # 3. 캔버스 설정
            self.setup_canvas()
            
            # 4. 워터마크 그리기 (마스크 이미지)
            if watermark_path and os.path.exists(watermark_path):
                self.draw_image(0.0, 0.0, card_width, card_height, watermark_path)
            
            # 5. 메인 이미지 그리기
            if image_path and os.path.exists(image_path):
                self.draw_image(0.0, 0.0, card_width, card_height, image_path)
            
            # 6. 캔버스 커밋
            self.commit_canvas()
            time.sleep(1)
            
            # 7. 인쇄 실행
            self.print_draw()
            
            # 8. 카드 배출
            self.eject_card()
            
        except R600PrinterError as e:
            raise e
        except Exception as e:
            raise R600PrinterError(f"카드 인쇄 중 예상치 못한 오류: {e}")
    
    def close(self):
        """라이브러리 정리"""
        if self.lib:
            self.lib.R600LibClear()
    
    def __enter__(self):
        """컨텍스트 매니저 진입"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 매니저 종료"""
        self.close()


# 프린터 연동을 위한 유틸리티 함수들
def check_printer_dll(dll_path: str) -> bool:
    """DLL 파일 존재 여부 확인"""
    return os.path.exists(dll_path)


def get_default_dll_paths() -> List[str]:
    """기본 DLL 경로들 반환"""
    return [
        './libDSRetransfer600App.dll',
        './dll/libDSRetransfer600App.dll',
        './lib/libDSRetransfer600App.dll',
        os.path.join(os.path.dirname(__file__), 'libDSRetransfer600App.dll')
    ]


def find_printer_dll() -> Optional[str]:
    """사용 가능한 프린터 DLL 찾기"""
    for path in get_default_dll_paths():
        if check_printer_dll(path):
            return path
    return None