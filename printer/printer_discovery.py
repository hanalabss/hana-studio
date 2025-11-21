"""
프린터 탐지 및 선택 모듈 - TCP/USB 프린터 모두 지원
"""

import os
import ctypes
import ctypes.util
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
        if not dll_path:
            raise DLLNotFoundError("DLL 경로가 제공되지 않았습니다")

        if not ctypes.util.find_library(dll_path) and not os.path.exists(dll_path):
            raise DLLNotFoundError(f"DLL 파일을 찾을 수 없습니다: {dll_path}")

        self.dll_path = dll_path
        self.lib = None
        self._original_cwd = None
        self._safe_cwd = None
        self._load_dll()
        self._setup_function_signatures()

    def __enter__(self):
        """컨텍스트 관리자 진입 - 작업 디렉토리를 DLL 위치로 변경"""
        dll_dir = os.path.dirname(self.dll_path)
        self._original_cwd = os.getcwd()
        self._safe_cwd = dll_dir

        print(f"[PrinterDiscovery] 🔄 작업 디렉토리 변경: {self._original_cwd} → {dll_dir}")
        os.chdir(dll_dir)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 관리자 종료 - 원래 작업 디렉토리로 복원"""
        if self._original_cwd:
            os.chdir(self._original_cwd)
            print(f"[PrinterDiscovery] ✓ 작업 디렉토리 복원: {self._original_cwd}")
        return False

    def _load_dll(self):
        """DLL 로드 - 한글 경로 안전성 검증 + 작업 디렉토리 변경"""
        try:
            print(f"[PrinterDiscovery] DLL 로딩 시도: {self.dll_path}")

            # ASCII 경로 검증
            try:
                self.dll_path.encode('ascii')
                print(f"[PrinterDiscovery] ✓ ASCII 경로 확인")
            except UnicodeEncodeError:
                print(f"[PrinterDiscovery] ⚠ 한글 경로 감지: {self.dll_path}")
                print(f"[PrinterDiscovery] 경고: 이 경로는 이미 안전한 경로여야 합니다")

            # 🎯 핵심: DLL이 있는 디렉토리로 작업 디렉토리 임시 변경
            dll_dir = os.path.dirname(self.dll_path)
            original_cwd = os.getcwd()

            try:
                print(f"[PrinterDiscovery] 🔄 DLL 로딩 시 작업 디렉토리 변경: {dll_dir}")
                os.chdir(dll_dir)

                # DLL 로드
                self.lib = ctypes.CDLL(self.dll_path)
                print(f"[PrinterDiscovery] ✓ DLL 로드 성공")

            finally:
                # 원래 작업 디렉토리로 복원
                os.chdir(original_cwd)
                print(f"[PrinterDiscovery] ✓ 작업 디렉토리 복원: {original_cwd}")

        except Exception as e:
            print(f"[PrinterDiscovery] ✗ DLL 로드 실패: {e}")
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
            print("[DEBUG] USB 프린터 탐지 시작...")
            list_buffer_size = 1024
            printer_list_buffer = ctypes.create_string_buffer(list_buffer_size)
            enum_list_len = ctypes.c_uint(list_buffer_size)
            num_printers = ctypes.c_int()

            print(f"[DEBUG] R600EnumUsbPrt 호출 (버퍼 크기: {list_buffer_size})")
            ret = self.lib.R600EnumUsbPrt(
                printer_list_buffer,
                ctypes.byref(enum_list_len),
                ctypes.byref(num_printers)
            )

            print(f"[DEBUG] R600EnumUsbPrt 반환값: {ret}")
            print(f"[DEBUG] enum_list_len: {enum_list_len.value}")
            print(f"[DEBUG] num_printers: {num_printers.value}")

            if ret == 0:  # 성공
                actual_len = enum_list_len.value
                printer_count = num_printers.value

                print(f"[DEBUG] 실제 버퍼 길이: {actual_len}, 프린터 개수: {printer_count}")

                if actual_len > 0 and printer_count > 0:
                    raw_data = printer_list_buffer.raw[:actual_len]
                    print(f"[DEBUG] 원시 데이터 (hex): {raw_data.hex()}")
                    print(f"[DEBUG] 원시 데이터 (repr): {repr(raw_data)}")

                    try:
                        printer_names_str = printer_list_buffer.value.decode('cp949')
                        print(f"[DEBUG] 디코딩된 문자열: {repr(printer_names_str)}")
                    except:
                        # cp949 실패 시 utf-8 시도
                        try:
                            printer_names_str = printer_list_buffer.value.decode('utf-8')
                            print(f"[DEBUG] UTF-8 디코딩 성공: {repr(printer_names_str)}")
                        except:
                            printer_names_str = printer_list_buffer.value.decode('latin-1')
                            print(f"[DEBUG] Latin-1 디코딩: {repr(printer_names_str)}")

                    printer_names = [name.strip() for name in printer_names_str.split('\n') if name.strip()]
                    print(f"[DEBUG] 분리된 프린터 이름들: {printer_names}")

                    for i, name in enumerate(printer_names):
                        printers.append(PrinterInfo(name, "USB", i))

                    print(f"🔌 USB 프린터 {len(printers)}대 발견: {[p.name for p in printers]}")
                else:
                    print("🔌 USB 프린터를 찾을 수 없습니다.")
                    print(f"[DEBUG] actual_len={actual_len}, printer_count={printer_count}")
            else:
                print(f"🔌 USB 프린터 탐지 실패: 오류코드 {ret}")
                print("[DEBUG] 가능한 원인:")
                print("  - USB 드라이버가 설치되지 않음")
                print("  - 프린터가 USB에 연결되지 않음")
                print("  - 관리자 권한 필요")

        except Exception as e:
            print(f"❌ USB 프린터 탐지 중 오류: {e}")
            import traceback
            traceback.print_exc()

        return printers
    
    def discover_all_printers(self) -> List[PrinterInfo]:
        """모든 프린터 탐지 (TCP + USB) - 개선된 한글 경로 대응"""
        all_printers = []

        print("🔍 프린터 탐지 시작...")
        print(f"[DEBUG] DLL 디렉토리: {os.path.dirname(self.dll_path)}")
        print(f"[DEBUG] 현재 작업 디렉토리: {os.getcwd()}")

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
    try:
        # 컨텍스트 관리자 사용하여 작업 디렉토리 안전하게 관리
        with PrinterDiscovery(dll_path) as discovery:
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
        import traceback
        traceback.print_exc()
        return [], error_msg


def get_printer_display_name(printer: PrinterInfo) -> str:
    """프린터 표시용 이름 생성"""
    icon = "📡" if printer.connection_type == "TCP" else "🔌"
    return f"{icon} {printer.name} ({printer.connection_type})"