"""
프린터 탐지 및 선택 모듈 - TCP/USB 프린터 모두 지원
"""

import ctypes
from typing import List, Dict, Optional, Tuple
from .exceptions import R600PrinterError, DLLNotFoundError


class PrinterInfo:
    """프린터 정보 클래스"""
    
    def __init__(self, name: str, connection_type: str, index: int = 0):
        self.name = name.strip()
        self.connection_type = connection_type  # "TCP" 또는 "USB"
        self.index = index
        
    def __str__(self):
        return f"{self.name} ({self.connection_type})"
    
    def __repr__(self):
        return f"PrinterInfo(name='{self.name}', type='{self.connection_type}', index={self.index})"


class PrinterDiscovery:
    """프린터 탐지 및 관리 클래스"""
    
    def __init__(self, dll_path: str):
        """프린터 탐지 초기화"""
        self.dll_path = dll_path
        self.lib = None
        self._load_dll()
        self._setup_function_signatures()
    
    def _load_dll(self):
        """DLL 로드"""
        try:
            self.lib = ctypes.CDLL(self.dll_path)
        except Exception as e:
            raise DLLNotFoundError(f"DLL 로드 실패: {e}")
    
    def _setup_function_signatures(self):
        """함수 시그니처 정의"""
        # TCP 프린터 열거
        self.lib.R600EnumTcpPrt.argtypes = [
            ctypes.POINTER(ctypes.c_char), 
            ctypes.POINTER(ctypes.c_uint), 
            ctypes.POINTER(ctypes.c_int)
        ]
        self.lib.R600EnumTcpPrt.restype = ctypes.c_uint
        
        # USB 프린터 열거
        self.lib.R600EnumUsbPrt.argtypes = [
            ctypes.POINTER(ctypes.c_char), 
            ctypes.POINTER(ctypes.c_uint), 
            ctypes.POINTER(ctypes.c_int)
        ]
        self.lib.R600EnumUsbPrt.restype = ctypes.c_uint
        
        # 라이브러리 초기화
        self.lib.R600LibInit.argtypes = []
        self.lib.R600LibInit.restype = ctypes.c_uint
    
    def initialize_library(self):
        """라이브러리 초기화"""
        try:
            ret = self.lib.R600LibInit()
            if ret != 0:
                raise R600PrinterError(f"라이브러리 초기화 실패: {ret}")
            print("✅ 프린터 라이브러리 초기화 성공")
            return True
        except Exception as e:
            print(f"❌ 라이브러리 초기화 실패: {e}")
            return False
    
    def discover_tcp_printers(self) -> List[PrinterInfo]:
        """TCP 프린터 탐지"""
        printers = []
        
        try:
            list_buffer_size = 1024
            printer_list_buffer = ctypes.create_string_buffer(list_buffer_size)
            enum_list_len = ctypes.c_uint(list_buffer_size)
            num_printers = ctypes.c_int()
            
            ret = self.lib.R600EnumTcpPrt(
                printer_list_buffer, 
                ctypes.byref(enum_list_len), 
                ctypes.byref(num_printers)
            )
            
            if ret == 0:  # 성공
                actual_len = enum_list_len.value
                printer_count = num_printers.value
                
                if actual_len > 0 and printer_count > 0:
                    printer_names_str = printer_list_buffer.value.decode('cp949')
                    printer_names = [name.strip() for name in printer_names_str.split('\n') if name.strip()]
                    
                    for i, name in enumerate(printer_names):
                        printers.append(PrinterInfo(name, "TCP", i))
                    
                    print(f"📡 TCP 프린터 {len(printers)}대 발견: {[p.name for p in printers]}")
                else:
                    print("📡 TCP 프린터를 찾을 수 없습니다.")
            else:
                print(f"📡 TCP 프린터 탐지 실패: 오류코드 {ret}")
        
        except Exception as e:
            print(f"❌ TCP 프린터 탐지 중 오류: {e}")
        
        return printers
    
    def discover_usb_printers(self) -> List[PrinterInfo]:
        """USB 프린터 탐지"""
        printers = []
        
        try:
            list_buffer_size = 1024
            printer_list_buffer = ctypes.create_string_buffer(list_buffer_size)
            enum_list_len = ctypes.c_uint(list_buffer_size)
            num_printers = ctypes.c_int()
            
            ret = self.lib.R600EnumUsbPrt(
                printer_list_buffer, 
                ctypes.byref(enum_list_len), 
                ctypes.byref(num_printers)
            )
            
            if ret == 0:  # 성공
                actual_len = enum_list_len.value
                printer_count = num_printers.value
                
                if actual_len > 0 and printer_count > 0:
                    printer_names_str = printer_list_buffer.value.decode('cp949')
                    printer_names = [name.strip() for name in printer_names_str.split('\n') if name.strip()]
                    
                    for i, name in enumerate(printer_names):
                        printers.append(PrinterInfo(name, "USB", i))
                    
                    print(f"🔌 USB 프린터 {len(printers)}대 발견: {[p.name for p in printers]}")
                else:
                    print("🔌 USB 프린터를 찾을 수 없습니다.")
            else:
                print(f"🔌 USB 프린터 탐지 실패: 오류코드 {ret}")
        
        except Exception as e:
            print(f"❌ USB 프린터 탐지 중 오류: {e}")
        
        return printers
    
    def discover_all_printers(self) -> List[PrinterInfo]:
        """모든 프린터 탐지 (TCP + USB)"""
        all_printers = []
        
        print("🔍 프린터 탐지 시작...")
        
        # 라이브러리 초기화
        if not self.initialize_library():
            return []
        
        # TCP 프린터 탐지
        tcp_printers = self.discover_tcp_printers()
        all_printers.extend(tcp_printers)
        
        # USB 프린터 탐지
        usb_printers = self.discover_usb_printers()
        all_printers.extend(usb_printers)
        
        print(f"🎯 총 {len(all_printers)}대의 프린터 발견")
        
        return all_printers


def discover_available_printers(dll_path: str) -> Tuple[List[PrinterInfo], str]:
    """
    사용 가능한 프린터 탐지 (편의 함수)
    
    Returns:
        Tuple[List[PrinterInfo], str]: (프린터 목록, 요약 메시지)
    """
    discovery = PrinterDiscovery(dll_path)
    
    try:
        printers = discovery.discover_all_printers()
        
        if not printers:
            summary = "❌ 연결된 프린터가 없습니다."
        else:
            tcp_count = len([p for p in printers if p.connection_type == "TCP"])
            usb_count = len([p for p in printers if p.connection_type == "USB"])
            
            summary = f"✅ 총 {len(printers)}대 프린터 발견\n"
            if tcp_count > 0:
                summary += f"📡 TCP: {tcp_count}대\n"
            if usb_count > 0:
                summary += f"🔌 USB: {usb_count}대"
        
        return printers, summary
    
    except Exception as e:
        error_msg = f"❌ 프린터 탐지 실패: {e}"
        print(error_msg)
        return [], error_msg


def get_printer_display_name(printer: PrinterInfo) -> str:
    """프린터 표시용 이름 생성"""
    icon = "📡" if printer.connection_type == "TCP" else "🔌"
    return f"{icon} {printer.name} ({printer.connection_type})"