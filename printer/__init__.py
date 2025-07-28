"""
프린터 모듈 초기화
"""

from .exceptions import (
    R600PrinterError, 
    PrinterConnectionError, 
    PrinterInitializationError, 
    PrintingError, 
    DLLNotFoundError
)
from .r600_printer import R600Printer
from .printer_thread import PrinterThread
from .printer_utils import (
    check_printer_dll,
    get_default_dll_paths,
    find_printer_dll,
    test_printer_connection,
    test_print_modes
)
from .printer_discovery import (
    PrinterInfo,
    PrinterDiscovery,
    discover_available_printers,
    get_printer_display_name
)

__all__ = [
    # 예외 클래스들
    'R600PrinterError',
    'PrinterConnectionError', 
    'PrinterInitializationError',
    'PrintingError',
    'DLLNotFoundError',
    
    # 메인 클래스들
    'R600Printer',
    'PrinterThread',
    
    # 유틸리티 함수들
    'check_printer_dll',
    'get_default_dll_paths',
    'find_printer_dll',
    'test_printer_connection',
    'test_print_modes',
    
    # 프린터 탐지 관련
    'PrinterInfo',
    'PrinterDiscovery',
    'discover_available_printers',
    'get_printer_display_name'
]

# 모듈 정보
__version__ = '1.0.0'
__author__ = 'Hana Studio Team'
__description__ = 'RTAI LUKA R600 프린터 연동 모듈 - TCP/USB 지원'