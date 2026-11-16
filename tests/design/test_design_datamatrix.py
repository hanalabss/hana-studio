"""DataMatrix(ECC 200). 스캐너가 없어 ISO/IEC 16022 예시값과 구조 검사로 확인한다."""

import pytest

from design import barcodes
from design.datamatrix import (
    MAX_DATA, SYMBOLS, DataMatrixError, _EXP, _mul, _placement, datamatrix_matrix, encode_ascii, reed_solomon,
)


def test_iso_example_codewords():
    # ISO/IEC 16022 부속서의 예: "123456" → 데이터 142 164 186, 오류 정정 114 25 5 88 102 (10×10)
    data = encode_ascii("123456")
    assert data == [142, 164, 186]
    assert reed_solomon(data, 5) == [114, 25, 5, 88, 102]
    assert len(datamatrix_matrix("123456")) == 10


def test_ascii_rules():
    assert encode_ascii("A") == [66]
    assert encode_ascii("A1") == [66, 50]               # 숫자 하나는 짝을 이루지 못한다
    assert encode_ascii("0099") == [130, 229]
    # 한글: ECI 26(UTF-8) 다음 UTF-8 바이트를 Upper Shift로
    assert encode_ascii("가")[:2] == [241, 27]
    assert encode_ascii("가")[2:] == [235, 0xEA - 127, 235, 0xB0 - 127, 235, 0x80 - 127]


@pytest.mark.parametrize("size,data,ecc,region", SYMBOLS)
def test_every_module_is_placed_once(size, data, ecc, region):
    n = size // (region + 2) * region
    trace = []
    _placement(n, n, [0] * (data + ecc), trace)
    cells = [(r, c) for r, c, _ in trace]
    assert len(cells) == (data + ecc) * 8 and len(set(cells)) == len(cells)
    holes = n * n - len(cells)
    assert holes in (0, 4)                                 # 남는 네 칸은 고정 무늬


def _read_back(matrix, size, region, count):
    """그려진 기호에서 칸(codeword)을 다시 읽는다(배치 순서는 _placement의 기록을 쓴다)."""
    blocks = size // (region + 2)
    n = blocks * region
    trace = []
    _placement(n, n, [0] * count, trace)
    values = [0] * count
    for bit_index, (r, c, index) in enumerate(trace):
        by, bx = divmod(r, region)[0], divmod(c, region)[0]
        row = by * (region + 2) + 1 + r % region
        col = bx * (region + 2) + 1 + c % region
        bit = 7 - sum(1 for rr, cc, ii in trace[:bit_index] if ii == index)
        values[index] |= int(matrix[row][col]) << bit
    return values


def _syndromes(codewords, count):
    result = []
    for i in range(1, count + 1):
        value = 0
        for cw in codewords:
            value = _mul(value, _EXP[i]) ^ cw
        result.append(value)
    return result


@pytest.mark.parametrize("text", ["123456", "HN260415", "홍길동 HN260415", "https://hanalabs.example/c/0001", "9" * 300])
def test_symbol_reads_back_with_valid_error_correction(text):
    matrix = datamatrix_matrix(text)
    size = len(matrix)
    _, data, ecc, region = next(s for s in SYMBOLS if s[0] == size)
    codewords = _read_back(matrix, size, region, data + ecc)
    assert codewords[:len(encode_ascii(text))] == encode_ascii(text)
    assert _syndromes(codewords, ecc) == [0] * ecc        # 기호 전체가 올바른 RS 부호어


def test_finder_pattern():
    matrix = datamatrix_matrix("HN260415")
    size = len(matrix)
    assert all(matrix[r][0] for r in range(size)) and all(matrix[size - 1])
    assert [matrix[0][c] for c in range(size)] == [c % 2 == 0 for c in range(size)]
    assert [matrix[r][size - 1] for r in range(size)] == [r % 2 == 1 for r in range(size)]
    big = datamatrix_matrix("x" * 60)                     # 32×32: 영역 2×2, 가운데에도 경계선
    assert len(big) == 32 and all(big[r][16] for r in range(32)) and all(big[15])


def test_too_long_or_empty():
    with pytest.raises(DataMatrixError, match="너무 깁니다"):
        datamatrix_matrix("x" * (MAX_DATA + 1))
    with pytest.raises(barcodes.BarcodeError, match="너무 깁니다"):
        barcodes.matrix_2d("x" * (MAX_DATA + 1), "datamatrix")
    with pytest.raises(barcodes.BarcodeError, match="비어"):
        barcodes.matrix_2d("", "datamatrix")
    with pytest.raises(barcodes.BarcodeError, match="지원하지 않는"):
        barcodes.matrix_2d("A", "aztec")


def test_render_and_min_size(qapp):
    assert barcodes.qr_min_size_px("HN260415", symbology="datamatrix", border_modules=2) == 16
    image = barcodes.render_qr("HN260415", (160, 160), symbology="datamatrix", border_modules=2)
    assert image.width() == 160
    # 12×12 기호 + 여백 2칸 = 16칸, 한 칸 10px. 왼쪽 위 기호 시작점은 어둡다.
    assert image.pixelColor(25, 25).value() < 50 and image.pixelColor(5, 5).value() > 200
    with pytest.raises(barcodes.BarcodeSizeError):
        barcodes.render_qr("HN260415", (10, 10), symbology="datamatrix")
