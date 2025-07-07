"""
프린터 관련 예외 클래스들
"""


class R600PrinterError(Exception):
    """R600 프린터 관련 예외"""
    pass


class PrinterConnectionError(R600PrinterError):
    """프린터 연결 실패 예외"""
    pass


class PrinterInitializationError(R600PrinterError):
    """프린터 초기화 실패 예외"""
    pass


class PrintingError(R600PrinterError):
    """인쇄 과정 중 발생하는 예외"""
    pass


class DLLNotFoundError(R600PrinterError):
    """DLL 파일을 찾을 수 없을 때 발생하는 예외"""
    pass