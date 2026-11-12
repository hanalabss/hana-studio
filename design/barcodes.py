"""바코드·QR 생성. 모듈을 정수 픽셀 폭으로 직접 그려 가장자리가 번지지 않는다."""

from __future__ import annotations

from functools import lru_cache

from PySide6.QtCore import QRect
from PySide6.QtGui import QColor, QImage, QPainter

SYMBOLOGIES = ("code128", "code39", "ean13")
SYMBOLOGY_LABELS = {"code128": "Code 128", "code39": "Code 39", "ean13": "EAN-13"}
QR_LEVELS = ("L", "M", "Q", "H")

_CODE39_CHARS = set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-. $/+%")
_QUIET_ZONE = {"code128": (10, 10), "code39": (10, 10), "ean13": (11, 7)}


class BarcodeError(ValueError):
    """사용자에게 그대로 보여줄 수 있는 한국어 메시지를 담는다."""


class BarcodeSizeError(BarcodeError):
    """요청한 픽셀 크기에 모듈이 다 들어가지 않는다."""

    def __init__(self, message: str, min_width_px: int) -> None:
        super().__init__(message)
        self.min_width_px = min_width_px


def ean13_check_digit(first12: str) -> int:
    total = sum(int(ch) * (3 if i % 2 else 1) for i, ch in enumerate(first12))
    return (10 - total % 10) % 10


def normalize(symbology: str, data: str) -> str:
    """데이터를 검사하고 실제로 인코딩될 문자열을 돌려준다(EAN-13은 체크 숫자 포함 13자리)."""
    if symbology not in SYMBOLOGIES:
        raise BarcodeError(f"지원하지 않는 바코드 종류입니다: {symbology}")
    text = "" if data is None else str(data)
    if text == "":
        raise BarcodeError("바코드 데이터가 비어 있습니다.")
    label = SYMBOLOGY_LABELS[symbology]

    if symbology == "ean13":
        if not (text.isascii() and text.isdigit()):
            raise BarcodeError("EAN-13은 숫자만 사용할 수 있습니다.")
        if len(text) == 12:
            return text + str(ean13_check_digit(text))
        if len(text) != 13:
            raise BarcodeError(f"EAN-13은 12자리 또는 13자리 숫자여야 합니다. (현재 {len(text)}자리)")
        expected = ean13_check_digit(text[:12])
        if int(text[12]) != expected:
            raise BarcodeError(
                f"EAN-13 체크 숫자가 맞지 않습니다. 마지막 자리는 {expected}이어야 합니다."
            )
        return text

    if symbology == "code39":
        bad = sorted({ch for ch in text if ch not in _CODE39_CHARS})
        if bad:
            raise BarcodeError(
                f"{label}에 쓸 수 없는 문자가 있습니다: {' '.join(bad)} "
                "(숫자, 영문 대문자, - . $ / + % 공백만 가능)"
            )
        return text

    bad = sorted({ch for ch in text if not (32 <= ord(ch) < 127)})
    if bad:
        shown = " ".join(ch if ch.isprintable() else f"U+{ord(ch):04X}" for ch in bad)
        raise BarcodeError(f"{label}에 쓸 수 없는 문자가 있습니다: {shown} (영문·숫자·기호만 가능)")
    return text


@lru_cache(maxsize=256)
def _linear_modules(symbology: str, text: str) -> tuple[bool, ...]:
    import barcode
    from barcode.errors import BarcodeError as _LibError

    try:
        if symbology == "code39":
            code = barcode.get_barcode_class("code39")(text, add_checksum=False)
        else:
            code = barcode.get_barcode_class(symbology)(text)
        rows = code.build()
    except _LibError as exc:
        raise BarcodeError(f"{SYMBOLOGY_LABELS[symbology]} 바코드를 만들 수 없습니다: {exc}") from exc
    return tuple(ch == "1" for ch in rows[0])


def linear_modules(symbology: str, data: str) -> list[bool]:
    """막대(True)/공백(False) 모듈 열. 여백(quiet zone)은 포함하지 않는다."""
    return list(_linear_modules(symbology, normalize(symbology, data)))


@lru_cache(maxsize=128)
def _qr_matrix(text: str, level: str) -> tuple[tuple[bool, ...], ...]:
    import qrcode
    from qrcode import constants
    from qrcode.exceptions import DataOverflowError

    levels = {
        "L": constants.ERROR_CORRECT_L, "M": constants.ERROR_CORRECT_M,
        "Q": constants.ERROR_CORRECT_Q, "H": constants.ERROR_CORRECT_H,
    }
    qr = qrcode.QRCode(error_correction=levels[level], border=0, box_size=1)
    qr.add_data(text)
    try:
        qr.make(fit=True)
    except (DataOverflowError, ValueError) as exc:
        raise BarcodeError("QR 코드에 담기에는 데이터가 너무 깁니다.") from exc
    return tuple(tuple(bool(cell) for cell in row) for row in qr.modules)


def qr_matrix(data: str, error_correction: str = "M") -> list[list[bool]]:
    """QR 모듈 행렬(True=어두운 칸). 여백은 포함하지 않는다."""
    text = "" if data is None else str(data)
    if text == "":
        raise BarcodeError("QR 코드 데이터가 비어 있습니다.")
    level = str(error_correction).upper()
    if level not in QR_LEVELS:
        raise BarcodeError(f"QR 오류 정정 수준은 L, M, Q, H 중 하나여야 합니다: {error_correction}")
    return [list(row) for row in _qr_matrix(text, level)]


def _canvas(width_px: int, height_px: int, background) -> QImage:
    if width_px < 1 or height_px < 1:
        raise BarcodeSizeError("바코드를 그릴 영역의 크기가 0입니다.", 1)
    if background is None:
        image = QImage(width_px, height_px, QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(0)
    else:
        image = QImage(width_px, height_px, QImage.Format.Format_RGB32)
        image.fill(QColor(background))
    return image


def linear_min_width_px(symbology: str, data: str, quiet_zone: bool = True) -> int:
    quiet = sum(_QUIET_ZONE[symbology]) if quiet_zone else 0
    return len(_linear_modules(symbology, normalize(symbology, data))) + quiet


def render_linear(
    symbology: str,
    data: str,
    width_px: int,
    height_px: int,
    *,
    quiet_zone: bool = True,
    foreground="#000000",
    background="#ffffff",
) -> QImage:
    """정확히 width_px × height_px 크기의 1차원 바코드. background=None이면 투명."""
    modules = _linear_modules(symbology, normalize(symbology, data))
    left_quiet, right_quiet = _QUIET_ZONE[symbology] if quiet_zone else (0, 0)
    total = len(modules) + left_quiet + right_quiet
    width_px, height_px = int(width_px), int(height_px)
    module_px = width_px // total
    if module_px < 1:
        raise BarcodeSizeError(
            f"바코드를 그리기에 영역이 너무 좁습니다. 최소 {total}px 너비가 필요합니다. (현재 {width_px}px)",
            total,
        )
    image = _canvas(width_px, height_px, background)
    origin = (width_px - total * module_px) // 2 + left_quiet * module_px
    painter = QPainter(image)
    color = QColor(foreground)
    run_start = None
    for index, dark in enumerate((*modules, False)):
        if dark and run_start is None:
            run_start = index
        elif not dark and run_start is not None:
            painter.fillRect(
                QRect(origin + run_start * module_px, 0, (index - run_start) * module_px, height_px),
                color,
            )
            run_start = None
    painter.end()
    return image


def qr_min_size_px(data: str, error_correction: str = "M", border_modules: int = 2) -> int:
    return len(qr_matrix(data, error_correction)) + 2 * max(0, int(border_modules))


def render_qr(
    data: str,
    size_px: int | tuple[int, int],
    *,
    error_correction: str = "M",
    border_modules: int = 2,
    foreground="#000000",
    background="#ffffff",
) -> QImage:
    """요청한 픽셀 크기의 QR 코드. 칸은 정수 픽셀 정사각형이고 남는 폭은 여백이 된다."""
    matrix = qr_matrix(data, error_correction)
    width_px, height_px = (size_px, size_px) if isinstance(size_px, int) else size_px
    width_px, height_px = int(width_px), int(height_px)
    border = max(0, int(border_modules))
    total = len(matrix) + 2 * border
    module_px = min(width_px, height_px) // total
    if module_px < 1:
        raise BarcodeSizeError(
            f"QR 코드를 그리기에 영역이 너무 작습니다. 최소 {total}px이 필요합니다. "
            f"(현재 {min(width_px, height_px)}px)",
            total,
        )
    image = _canvas(width_px, height_px, background)
    left = (width_px - len(matrix) * module_px) // 2
    top = (height_px - len(matrix) * module_px) // 2
    painter = QPainter(image)
    color = QColor(foreground)
    for row_index, row in enumerate(matrix):
        y = top + row_index * module_px
        run_start = None
        for col_index, dark in enumerate((*row, False)):
            if dark and run_start is None:
                run_start = col_index
            elif not dark and run_start is not None:
                painter.fillRect(
                    QRect(left + run_start * module_px, y, (col_index - run_start) * module_px, module_px),
                    color,
                )
                run_start = None
    painter.end()
    return image
