"""
[EMOJI] [EMOJI] [EMOJI] [EMOJI]
"""


class R600PrinterError(Exception):
    """R600 [EMOJI] [EMOJI] [EMOJI]"""
    pass


class PrinterConnectionError(R600PrinterError):
    """[EMOJI] [EMOJI] [EMOJI] [EMOJI]"""
    pass


class PrinterInitializationError(R600PrinterError):
    """[EMOJI] [EMOJI] [EMOJI] [EMOJI]"""
    pass


class PrintingError(R600PrinterError):
    """[EMOJI] [EMOJI] [EMOJI] [EMOJI] [EMOJI]"""
    pass


class DLLNotFoundError(R600PrinterError):
    """DLL [EMOJI] [EMOJI] [EMOJI] [EMOJI] [EMOJI] [EMOJI] [EMOJI]"""
    pass