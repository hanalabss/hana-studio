"""
프린터 연동 모듈 - 하위 호환성을 위한 래퍼
기존 코드와의 호환성을 유지하면서 새로운 모듈 구조를 사용
"""

# 새로운 모듈에서 필요한 것들을 import
from printer import (
    R600PrinterError,
    PrinterConnectionError,
    PrinterInitializationError,
    PrintingError,
    DLLNotFoundError,
    R600Printer,
    PrinterThread,
    find_printer_dll,
    test_printer_connection,
    test_print_modes
)

# 하위 호환성을 위해 기존 import 스타일 지원
__all__ = [
    'R600PrinterError',
    'PrinterThread', 
    'find_printer_dll',
    'test_printer_connection'
]

# 기존 코드에서 직접 사용할 수 있도록 모든 항목을 노출
PRINTER_AVAILABLE = True

try:
    # DLL 파일 존재 여부 확인
    dll_path = find_printer_dll()
    if not dll_path:
        PRINTER_AVAILABLE = False
        print("⚠️ 프린터 DLL을 찾을 수 없습니다")
    else:
        print("✅ 프린터 모듈 로드 성공")
except Exception as e:
    PRINTER_AVAILABLE = False
    print(f"⚠️ 프린터 모듈 로드 실패: {e}")