"""
[EMOJI] [EMOJI] [EMOJI]
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
    # [EMOJI] [EMOJI]
    'R600PrinterError',
    'PrinterConnectionError', 
    'PrinterInitializationError',
    'PrintingError',
    'DLLNotFoundError',
    
    # [EMOJI] [EMOJI]
    'R600Printer',
    'PrinterThread',
    
    # [EMOJI] [EMOJI]
    'check_printer_dll',
    'get_default_dll_paths',
    'find_printer_dll',
    'test_printer_connection',
    'test_print_modes',
    
    # [EMOJI] [EMOJI] [EMOJI]
    'PrinterInfo',
    'PrinterDiscovery',
    'discover_available_printers',
    'get_printer_display_name'
]

# [EMOJI] [EMOJI]
__version__ = '1.0.0'
__author__ = 'Hana Studio Team'
__description__ = 'RTAI LUKA R600 [EMOJI] [EMOJI] [EMOJI] - TCP/USB [EMOJI]'