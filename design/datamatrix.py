"""DataMatrix(ECC 200) 만들기. ISO/IEC 16022의 규칙을 그대로 옮겼다(외부 라이브러리 없음).

지원 범위: 정사각 10×10 ~ 48×48(오류 정정 블록 하나짜리), ASCII 인코딩.
- 숫자 두 자리는 한 칸(codeword)에 담는다.
- 128 이상의 글자(한글 등)는 ECI 26(UTF-8)을 앞에 붙이고 UTF-8 바이트를 Upper Shift로 담는다.
  리더기가 ECI를 지원해야 한글이 제대로 읽힌다.
48×48(데이터 174칸)보다 긴 내용은 BarcodeError.
"""

from __future__ import annotations

from functools import lru_cache

# (한 변 칸 수, 데이터 칸, 오류 정정 칸, 데이터 영역 한 변). 영역이 둘 이상이면 2×2로 나뉜다.
SYMBOLS = (
    (10, 3, 5, 8), (12, 5, 7, 10), (14, 8, 10, 12), (16, 12, 12, 14), (18, 18, 14, 16),
    (20, 22, 18, 18), (22, 30, 20, 20), (24, 36, 24, 22), (26, 44, 28, 24),
    (32, 62, 36, 14), (36, 86, 42, 16), (40, 114, 48, 18), (44, 144, 56, 20), (48, 174, 68, 22),
)
MAX_DATA = SYMBOLS[-1][1]

_PAD = 129
_UPPER_SHIFT = 235
_ECI = 241
_ECI_UTF8 = 27            # ECI 26 → 26 + 1


class DataMatrixError(ValueError):
    pass


# ---------------------------------------------------------------- GF(256), 원시 다항식 x^8+x^5+x^3+x^2+1 (301)

_EXP = [0] * 512
_LOG = [0] * 256
_value = 1
for _i in range(255):
    _EXP[_i] = _value
    _LOG[_value] = _i
    _value <<= 1
    if _value & 0x100:
        _value ^= 301
for _i in range(255, 512):
    _EXP[_i] = _EXP[_i - 255]


def _mul(a: int, b: int) -> int:
    return 0 if a == 0 or b == 0 else _EXP[_LOG[a] + _LOG[b]]


@lru_cache(maxsize=None)
def _generator(count: int) -> tuple[int, ...]:
    """(x - α^1)(x - α^2)…(x - α^count)의 계수, 높은 차수부터(맨 앞 1은 뺌)."""
    poly = [1]
    for i in range(1, count + 1):
        root = _EXP[i]
        nxt = poly + [0]
        for j in range(len(poly)):
            nxt[j + 1] ^= _mul(poly[j], root)
        poly = nxt
    return tuple(poly[1:])


def reed_solomon(data: list[int], count: int) -> list[int]:
    gen = _generator(count)
    ecc = [0] * count
    for value in data:
        k = ecc[0] ^ value
        ecc = ecc[1:] + [0]
        if k:
            for j in range(count):
                ecc[j] ^= _mul(k, gen[j])
    return ecc


# ---------------------------------------------------------------- 인코딩

def encode_ascii(text: str) -> list[int]:
    codewords: list[int] = []
    if any(ord(ch) > 127 for ch in text):
        codewords += [_ECI, _ECI_UTF8]
        raw = text.encode("utf-8")
    else:
        raw = text.encode("ascii")
    i = 0
    while i < len(raw):
        b = raw[i]
        if 48 <= b <= 57 and i + 1 < len(raw) and 48 <= raw[i + 1] <= 57:
            codewords.append(130 + (b - 48) * 10 + (raw[i + 1] - 48))
            i += 2
            continue
        if b < 128:
            codewords.append(b + 1)
        else:
            codewords += [_UPPER_SHIFT, b - 127]
        i += 1
    return codewords


def _pad(codewords: list[int], capacity: int) -> list[int]:
    result = list(codewords)
    if len(result) < capacity:
        result.append(_PAD)
    while len(result) < capacity:
        position = len(result) + 1
        value = _PAD + ((149 * position) % 253) + 1
        result.append(value - 254 if value > 254 else value)
    return result


# ---------------------------------------------------------------- 배치(ISO/IEC 16022 부속서 F)

def _placement(nrow: int, ncol: int, codewords: list[int], trace: list | None = None) -> list[list[int]]:
    """codewords의 비트를 nrow×ncol 영역에 놓는다. trace를 주면 놓은 칸 (row, col, codeword 번호)을 차례로 담는다."""
    array = [[-1] * ncol for _ in range(nrow)]

    def module(row: int, col: int, index: int, bit: int) -> None:
        if row < 0:
            row += nrow
            col += 4 - ((nrow + 4) % 8)
        if col < 0:
            col += ncol
            row += 4 - ((ncol + 4) % 8)
        array[row][col] = (codewords[index] >> (8 - bit)) & 1
        if trace is not None:
            trace.append((row, col, index))

    def utah(row: int, col: int, index: int) -> None:
        module(row - 2, col - 2, index, 1)
        module(row - 2, col - 1, index, 2)
        module(row - 1, col - 2, index, 3)
        module(row - 1, col - 1, index, 4)
        module(row - 1, col, index, 5)
        module(row, col - 2, index, 6)
        module(row, col - 1, index, 7)
        module(row, col, index, 8)

    def corner(index: int, cells) -> None:
        for bit, (row, col) in enumerate(cells, start=1):
            module(row, col, index, bit)

    index, row, col = 0, 4, 0
    while True:
        if row == nrow and col == 0:
            corner(index, ((nrow - 1, 0), (nrow - 1, 1), (nrow - 1, 2), (0, ncol - 2),
                           (0, ncol - 1), (1, ncol - 1), (2, ncol - 1), (3, ncol - 1)))
            index += 1
        if row == nrow - 2 and col == 0 and ncol % 4:
            corner(index, ((nrow - 3, 0), (nrow - 2, 0), (nrow - 1, 0), (0, ncol - 4),
                           (0, ncol - 3), (0, ncol - 2), (0, ncol - 1), (1, ncol - 1)))
            index += 1
        if row == nrow - 2 and col == 0 and ncol % 8 == 4:
            corner(index, ((nrow - 3, 0), (nrow - 2, 0), (nrow - 1, 0), (0, ncol - 2),
                           (0, ncol - 1), (1, ncol - 1), (2, ncol - 1), (3, ncol - 1)))
            index += 1
        if row == nrow + 4 and col == 2 and ncol % 8 == 0:
            corner(index, ((nrow - 1, 0), (nrow - 1, ncol - 1), (0, ncol - 3), (0, ncol - 2),
                           (0, ncol - 1), (1, ncol - 3), (1, ncol - 2), (1, ncol - 1)))
            index += 1
        while True:                                      # 오른쪽 위로
            if row < nrow and col >= 0 and array[row][col] < 0:
                utah(row, col, index)
                index += 1
            row -= 2
            col += 2
            if not (row >= 0 and col < ncol):
                break
        row += 1
        col += 3
        while True:                                      # 왼쪽 아래로
            if row >= 0 and col < ncol and array[row][col] < 0:
                utah(row, col, index)
                index += 1
            row += 2
            col -= 2
            if not (row < nrow and col >= 0):
                break
        row += 3
        col += 1
        if not (row < nrow or col < ncol):
            break
    if array[nrow - 1][ncol - 1] < 0:
        array[nrow - 1][ncol - 1] = array[nrow - 2][ncol - 2] = 1
        array[nrow - 1][ncol - 2] = array[nrow - 2][ncol - 1] = 0
    return array


def _symbol_for(length: int) -> tuple[int, int, int, int]:
    for symbol in SYMBOLS:
        if symbol[1] >= length:
            return symbol
    raise DataMatrixError(
        f"DataMatrix에 담기에는 내용이 너무 깁니다(최대 {MAX_DATA}칸, 지금 {length}칸). QR 코드를 쓰세요.")


@lru_cache(maxsize=128)
def _matrix(text: str) -> tuple[tuple[bool, ...], ...]:
    data = encode_ascii(text)
    size, capacity, ecc_count, region = _symbol_for(len(data))
    codewords = _pad(data, capacity)
    codewords += reed_solomon(codewords, ecc_count)
    regions = size // (region + 2)                  # 32×32부터 2×2 영역
    mapping = _placement(region * regions, region * regions, codewords)
    block = region + 2
    matrix = [[False] * size for _ in range(size)]
    for by in range(regions):
        for bx in range(regions):
            top, left = by * block, bx * block
            for i in range(block):
                matrix[top + block - 1][left + i] = True           # 아래 실선
                matrix[top + i][left] = True                       # 왼쪽 실선
                matrix[top][left + i] = i % 2 == 0                 # 위 점선
                matrix[top + i][left + block - 1] = i % 2 == 1     # 오른쪽 점선
            matrix[top + block - 1][left + block - 1] = True
            matrix[top][left] = True
            for r in range(region):
                for c in range(region):
                    matrix[top + 1 + r][left + 1 + c] = mapping[by * region + r][bx * region + c] == 1
    return tuple(tuple(row) for row in matrix)


def datamatrix_matrix(text: str) -> list[list[bool]]:
    """DataMatrix 모듈 행렬(True=어두운 칸). 여백(quiet zone)은 포함하지 않는다."""
    if text is None or str(text) == "":
        raise DataMatrixError("DataMatrix 데이터가 비어 있습니다.")
    return [list(row) for row in _matrix(str(text))]
