"""바코드·QR·DataMatrix·PDF417 생성. 모듈을 정수 픽셀 폭으로 직접 그려 가장자리가 번지지 않는다."""

from __future__ import annotations

from functools import lru_cache

from PySide6.QtCore import QRect
from PySide6.QtGui import QColor, QImage, QPainter

SYMBOLOGIES = ("code128", "code128a", "code128b", "code128c", "code39", "code39ext", "code93", "ean13",
               "ean8", "upca", "itf", "codabar")
SYMBOLOGY_LABELS = {
    "code128": "Code 128", "code128a": "Code 128-A", "code128b": "Code 128-B", "code128c": "Code 128-C",
    "code39": "Code 39", "code39ext": "Code 39 Extended", "code93": "Code 93", "ean13": "EAN-13",
    "ean8": "EAN-8", "upca": "UPC-A", "itf": "ITF", "codabar": "Codabar",
}
QR_LEVELS = ("L", "M", "Q", "H")
CODES_2D = ("qr", "datamatrix", "pdf417")
CODE_2D_LABELS = {"qr": "QR 코드", "datamatrix": "DataMatrix", "pdf417": "PDF417"}
_CODABAR_CHARS = set("0123456789-$:/.+")
_CODABAR_ENDS = set("ABCD")

_CODE39_CHARS = set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-. $/+%")
_QUIET_ZONE = {key: (10, 10) for key in SYMBOLOGIES}
_QUIET_ZONE["ean13"] = (11, 7)
_QUIET_ZONE["ean8"] = (7, 7)
_QUIET_ZONE["upca"] = (9, 9)

# Full ASCII(Code 39 Extended, Code 93)에서 기본 문자 밖의 글자를 두 글자로 바꾸는 표.
# 앞 글자가 $ % / + 인 짝이다. Code 93은 이 앞 글자를 전용 시프트 문자로 쓴다.
_FULL_ASCII: dict[str, str] = {chr(0): "%U", chr(127): "%T", " ": " ", "-": "-", ".": ".", "@": "%V", "`": "%W"}
for _i in range(1, 27):
    _FULL_ASCII[chr(_i)] = "$" + chr(64 + _i)
for _code, _pair in zip(range(27, 32), ("%A", "%B", "%C", "%D", "%E")):
    _FULL_ASCII[chr(_code)] = _pair
for _ch, _letter in zip("!\"#$%&'()*+,", "ABCDEFGHIJKL"):
    _FULL_ASCII[_ch] = "/" + _letter
_FULL_ASCII["/"] = "/O"
_FULL_ASCII[":"] = "/Z"
for _ch, _letter in zip(";<=>?", "FGHIJ"):
    _FULL_ASCII[_ch] = "%" + _letter
for _ch, _letter in zip("[\\]^_", "KLMNO"):
    _FULL_ASCII[_ch] = "%" + _letter
for _ch, _letter in zip("{|}~", "PQRS"):
    _FULL_ASCII[_ch] = "%" + _letter
for _ch in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    _FULL_ASCII[_ch] = _ch
for _i in range(26):
    _FULL_ASCII[chr(97 + _i)] = "+" + chr(65 + _i)

# Code 93: 값 순서대로 문자와 9모듈 무늬. 43~46은 시프트 문자 ($) (%) (/) (+).
_CODE93_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-. $/+%"
_CODE93_PATTERNS = (
    "100010100", "101001000", "101000100", "101000010", "100101000", "100100100", "100100010", "101010000",
    "100010010", "100001010", "110101000", "110100100", "110100010", "110010100", "110010010", "110001010",
    "101101000", "101100100", "101100010", "100110100", "100011010", "101011000", "101001100", "101000110",
    "100101100", "100010110", "110110100", "110110010", "110101100", "110100110", "110010110", "110011010",
    "101101100", "101100110", "100110110", "100111010", "100101110", "111010100", "111010010", "111001010",
    "101101110", "101110110", "110101110", "100100110", "111011010", "111010110", "100110010",
)
_CODE93_START_STOP = "101011110"
_CODE93_SHIFT = {"$": 43, "%": 44, "/": 45, "+": 46}


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


def gtin_check_digit(body: str) -> int:
    """EAN-8·UPC-A·EAN-13 공통 체크 숫자: 오른쪽 끝 자리부터 3, 1, 3, 1 … 을 곱한다."""
    total = sum(int(ch) * (3 if i % 2 == 0 else 1) for i, ch in enumerate(reversed(body)))
    return (10 - total % 10) % 10


def _gtin(label: str, text: str, length: int) -> str:
    """숫자 length-1자리(체크 숫자를 붙인다) 또는 length자리(체크 숫자를 확인한다)."""
    if not (text.isascii() and text.isdigit()):
        raise BarcodeError(f"{label}은 숫자만 사용할 수 있습니다.")
    if len(text) == length - 1:
        return text + str(gtin_check_digit(text))
    if len(text) != length:
        raise BarcodeError(f"{label}은 {length - 1}자리 또는 {length}자리 숫자여야 합니다. (현재 {len(text)}자리)")
    expected = gtin_check_digit(text[:-1])
    if int(text[-1]) != expected:
        raise BarcodeError(f"{label} 체크 숫자가 맞지 않습니다. 마지막 자리는 {expected}이어야 합니다.")
    return text


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

    if symbology == "ean8":
        return _gtin(label, text, 8)
    if symbology == "upca":
        return _gtin(label, text, 12)
    if symbology == "itf":
        if not (text.isascii() and text.isdigit()):
            raise BarcodeError("ITF는 숫자만 사용할 수 있습니다.")
        if len(text) % 2:
            raise BarcodeError(f"ITF는 숫자가 짝수 개여야 합니다. 앞에 0을 붙여 보세요. (현재 {len(text)}자리)")
        return text
    if symbology == "codabar":
        upper = text.upper()
        if len(upper) >= 2 and upper[0] in _CODABAR_ENDS and upper[-1] in _CODABAR_ENDS:
            body, start, stop = upper[1:-1], upper[0], upper[-1]
        else:
            body, start, stop = text, "A", "A"          # 시작·끝 문자를 적지 않으면 A로 감싼다
        bad = sorted({ch for ch in body if ch not in _CODABAR_CHARS})
        if bad or not body:
            raise BarcodeError(f"{label}에 쓸 수 없는 문자가 있습니다: {' '.join(bad) or '(비어 있음)'} "
                               "(숫자와 - $ : / . + 만 가능)")
        return start + body + stop

    if symbology == "code39":
        bad = sorted({ch for ch in text if ch not in _CODE39_CHARS})
        if bad:
            raise BarcodeError(
                f"{label}에 쓸 수 없는 문자가 있습니다: {' '.join(bad)} "
                "(숫자, 영문 대문자, - . $ / + % 공백만 가능)"
            )
        return text

    if symbology in ("code39ext", "code93"):
        bad = sorted({ch for ch in text if ch not in _FULL_ASCII})
        if bad:
            shown = " ".join(ch if ch.isprintable() else f"U+{ord(ch):04X}" for ch in bad)
            raise BarcodeError(f"{label}에 쓸 수 없는 문자가 있습니다: {shown} (영문·숫자·기호만 가능)")
        return text

    if symbology == "code128c":
        if not (text.isascii() and text.isdigit()):
            raise BarcodeError("Code 128-C는 숫자만 사용할 수 있습니다.")
        if len(text) % 2:
            raise BarcodeError(f"Code 128-C는 숫자가 짝수 개여야 합니다. (현재 {len(text)}자리)")
        return text

    if symbology == "code128a":
        bad = sorted({ch for ch in text if not (ord(ch) < 96)})
        if bad:
            shown = " ".join(ch if ch.isprintable() else f"U+{ord(ch):04X}" for ch in bad)
            raise BarcodeError(f"{label}에 쓸 수 없는 문자가 있습니다: {shown} (숫자, 영문 대문자, 기호만 가능)")
        return text

    bad = sorted({ch for ch in text if not (32 <= ord(ch) < 127)})
    if bad:
        shown = " ".join(ch if ch.isprintable() else f"U+{ord(ch):04X}" for ch in bad)
        raise BarcodeError(f"{label}에 쓸 수 없는 문자가 있습니다: {shown} (영문·숫자·기호만 가능)")
    return text


def _code128_forced(code_set: str, text: str) -> str:
    """한 가지 문자 집합(A·B·C)만 써서 만든 Code 128 모듈 문자열. 집합을 바꾸지 않는다."""
    from barcode.charsets import code128 as table

    if code_set == "C":
        values = [int(text[i:i + 2]) for i in range(0, len(text), 2)]
    elif code_set == "A":
        # A 집합: 공백~_ 는 0~63, 제어 문자(0~31)는 64~95.
        values = [ord(ch) - 32 if ord(ch) >= 32 else ord(ch) + 64 for ch in text]
    else:
        values = [ord(ch) - 32 for ch in text]
    start = table.START_CODES[code_set]
    checksum = (start + sum(weight * value for weight, value in enumerate(values, start=1))) % 103
    # 표의 STOP은 11모듈이라 끝맺음 막대(2모듈)를 붙여야 13모듈 정지 문자가 된다.
    return "".join(table.CODES[v] for v in (start, *values, checksum)) + table.STOP + "11"


def _code93(text: str) -> str:
    """Code 93 Full ASCII 모듈 문자열(체크 문자 C·K 포함)."""
    values: list[int] = []
    for ch in text:
        pair = _FULL_ASCII[ch]
        if len(pair) == 2:
            values.append(_CODE93_SHIFT[pair[0]])
            values.append(_CODE93_CHARS.index(pair[1]))
        else:
            values.append(_CODE93_CHARS.index(pair))
    for limit in (20, 15):                                    # C(가중치 1~20), K(1~15)
        total = sum(((len(values) - i - 1) % limit + 1) * v for i, v in enumerate(values))
        values.append(total % 47)
    body = "".join(_CODE93_PATTERNS[v] for v in values)
    return _CODE93_START_STOP + body + _CODE93_START_STOP + "1"


@lru_cache(maxsize=256)
def _linear_modules(symbology: str, text: str) -> tuple[bool, ...]:
    if symbology in ("code128a", "code128b", "code128c"):
        return tuple(ch == "1" for ch in _code128_forced(symbology[-1].upper(), text))
    if symbology == "code93":
        return tuple(ch == "1" for ch in _code93(text))
    if symbology == "code39ext":
        text = "".join(_FULL_ASCII[ch] for ch in text)
        symbology = "code39"

    import barcode
    from barcode.errors import BarcodeError as _LibError

    try:
        if symbology == "code39":
            code = barcode.get_barcode_class("code39")(text, add_checksum=False)
        elif symbology in ("ean8", "upca"):
            code = barcode.get_barcode_class(symbology)(text[:-1])      # 체크 숫자는 라이브러리가 다시 붙인다
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


def code2d_label(symbology: str | None) -> str:
    return CODE_2D_LABELS.get(symbology or "qr", "QR 코드")


def matrix_2d(data: str, symbology: str = "qr", error_correction: str = "M") -> list[list[bool]]:
    """2차원 코드(QR·DataMatrix)의 모듈 행렬. DataMatrix는 오류 정정 수준이 정해져 있어 error_correction을 쓰지 않는다."""
    symbology = symbology or "qr"
    if symbology == "qr":
        return qr_matrix(data, error_correction)
    if symbology == "datamatrix":
        from design.datamatrix import DataMatrixError, datamatrix_matrix

        try:
            return datamatrix_matrix(data)
        except DataMatrixError as exc:
            raise BarcodeError(str(exc)) from exc
    if symbology == "pdf417":
        text = "" if data is None else str(data)
        if text == "":
            raise BarcodeError("PDF417 데이터가 비어 있습니다.")
        return [list(row) for row in _pdf417_matrix(text)]
    raise BarcodeError(f"지원하지 않는 2차원 코드입니다: {symbology}")


PDF417_ROW_HEIGHT = 3          # 한 줄의 높이(모듈 수). 표준 권장 최소값.


def _pdf417_security(data_codewords: int) -> int:
    """ISO 15438 권장 오류 정정 수준."""
    for limit, level in ((40, 2), (160, 3), (320, 4)):
        if data_codewords <= limit:
            return level
    return 5


@lru_cache(maxsize=128)
def _pdf417_matrix(text: str) -> tuple[tuple[bool, ...], ...]:
    """가로가 세로의 약 3배가 되도록 열 수를 고른 PDF417 모듈 행렬."""
    import pdf417gen
    from pdf417gen import rendering

    best = None
    try:
        probe = pdf417gen.encode(text, columns=2, security_level=2)     # 열이 적으면 줄 수 하한(3)에 걸리지 않는다
        data_codewords = sum(len(row) - 4 for row in probe) - 8          # 수준 2의 오류 정정 8개를 뺀다
        level = _pdf417_security(data_codewords)
        for columns in range(2, 31):
            try:
                codes = pdf417gen.encode(text, columns=columns, security_level=level)
            except ValueError:
                continue
            width = 17 * (columns + 4) + 1
            height = len(codes) * PDF417_ROW_HEIGHT
            score = abs(width / height - 3.0)
            if best is None or score < best[0]:
                best = (score, codes, width)
    except ValueError as exc:
        raise BarcodeError("PDF417에 담기에는 데이터가 너무 깁니다.") from exc
    if best is None:
        raise BarcodeError("PDF417에 담기에는 데이터가 너무 깁니다.")
    _score, codes, width = best
    grid = [[False] * width for _ in codes]
    for x, y in rendering.modules(codes):
        grid[y][x] = True
    return tuple(tuple(row) for row in grid for _ in range(PDF417_ROW_HEIGHT))


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


def qr_min_size_px(data: str, error_correction: str = "M", border_modules: int = 2, symbology: str = "qr") -> int:
    matrix = matrix_2d(data, symbology, error_correction)
    return max(len(matrix), len(matrix[0])) + 2 * max(0, int(border_modules))


def render_qr(
    data: str,
    size_px: int | tuple[int, int],
    *,
    error_correction: str = "M",
    border_modules: int = 2,
    foreground="#000000",
    background="#ffffff",
    symbology: str = "qr",
) -> QImage:
    """요청한 픽셀 크기의 2차원 코드(QR·DataMatrix). 칸은 정수 픽셀 정사각형이고 남는 폭은 여백이 된다."""
    matrix = matrix_2d(data, symbology, error_correction)
    width_px, height_px = (size_px, size_px) if isinstance(size_px, int) else size_px
    width_px, height_px = int(width_px), int(height_px)
    border = max(0, int(border_modules))
    rows, cols = len(matrix), len(matrix[0])
    total = max(rows, cols) + 2 * border
    module_px = min(width_px // (cols + 2 * border), height_px // (rows + 2 * border))
    if module_px < 1:
        raise BarcodeSizeError(
            f"{code2d_label(symbology)}를 그리기에 영역이 너무 작습니다. 최소 {total}px이 필요합니다. "
            f"(현재 {min(width_px, height_px)}px)",
            total,
        )
    image = _canvas(width_px, height_px, background)
    left = (width_px - cols * module_px) // 2
    top = (height_px - rows * module_px) // 2
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
