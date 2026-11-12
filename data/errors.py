class DataError(Exception):
    """사용자에게 그대로 보여줄 수 있는 한국어 메시지를 담는 오류."""


class RangeError(DataError, ValueError):
    pass
