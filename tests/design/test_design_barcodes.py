import cv2
import numpy as np
import pytest

from design import barcodes
from design.barcodes import BarcodeError, BarcodeSizeError

# Code 128 표준 패턴 (값 → 11모듈), 검증용으로 필요한 것만.
CODE128_START_B = "11010010000"
CODE128_STOP = "1100011101011"


def distinct_colors(array: np.ndarray) -> set[tuple[int, ...]]:
    return {tuple(int(v) for v in c) for c in np.unique(array.reshape(-1, array.shape[-1]), axis=0)}


def row_modules(row: np.ndarray, module_px: int, start: int, count: int) -> list[bool]:
    """한 줄의 픽셀에서 모듈 값을 읽고, 모듈 안의 모든 픽셀이 같은 값인지 확인한다."""
    dark = row[:, 0] < 128
    modules = []
    for index in range(count):
        cell = dark[start + index * module_px: start + (index + 1) * module_px]
        assert cell.all() or not cell.any(), f"모듈 {index} 안에 색이 섞여 있습니다"
        modules.append(bool(cell[0]))
    return modules


def test_code128_modules_have_start_and_stop_patterns():
    modules = barcodes.linear_modules("code128", "ABC-123")
    bits = "".join("1" if m else "0" for m in modules)
    assert bits.startswith(CODE128_START_B)
    assert bits.endswith(CODE128_STOP)
    assert (len(bits) - len(CODE128_STOP)) % 11 == 0
    assert len(bits) == 11 * (1 + 7 + 1) + 13


def test_code39_modules_are_framed_by_asterisk_pattern():
    # '*' = 좁은 막대, 넓은 공백, 좁은 막대, 좁은 공백, 넓은 막대, 좁은 공백, 넓은 막대, 좁은 공백, 좁은 막대 (넓음=3)
    star = "100010111011101"
    bits = "".join("1" if m else "0" for m in barcodes.linear_modules("code39", "AB-1"))
    assert bits.startswith(star + "0") and bits.endswith("0" + star)
    assert len(bits) == 6 * 16 - 1


def test_ean13_checksum_and_length_rules():
    assert barcodes.ean13_check_digit("400638133393") == 1
    assert barcodes.normalize("ean13", "400638133393") == "4006381333931"
    assert barcodes.normalize("ean13", "4006381333931") == "4006381333931"
    assert len(barcodes.linear_modules("ean13", "4006381333931")) == 95

    with pytest.raises(BarcodeError, match="체크 숫자.*1"):
        barcodes.normalize("ean13", "4006381333930")
    with pytest.raises(BarcodeError, match="12자리 또는 13자리.*5자리"):
        barcodes.normalize("ean13", "12345")
    with pytest.raises(BarcodeError, match="숫자만"):
        barcodes.normalize("ean13", "40063813A393")
    with pytest.raises(BarcodeError, match="숫자만"):
        barcodes.normalize("ean13", "４００６３８１３３３９３")


@pytest.mark.parametrize("symbology, data, message", [
    ("code128", "", "비어"),
    ("code128", "한글", "쓸 수 없는 문자"),
    ("code128", "a\tb", "U\\+0009"),
    ("code39", "abc", "쓸 수 없는 문자"),
    ("code39", "A*B", "쓸 수 없는 문자"),
    ("pdf417", "A", "지원하지 않는"),
])
def test_invalid_linear_data_raises_korean_error(symbology, data, message):
    with pytest.raises(BarcodeError, match=message):
        barcodes.linear_modules(symbology, data)
    with pytest.raises(BarcodeError, match=message):
        barcodes.render_linear(symbology, data, 600, 100)


@pytest.mark.parametrize("symbology, data", [
    ("code128", "HANA-2026-0001"), ("code39", "EMP 1024"), ("ean13", "8801234567893"),
])
@pytest.mark.parametrize("width", [472, 601, 997])
def test_linear_render_is_exact_size_two_colors_integer_modules(qimage_array, symbology, data, width):
    modules = barcodes.linear_modules(symbology, data)
    image = barcodes.render_linear(symbology, data, width, 90)
    assert (image.width(), image.height()) == (width, 90)

    pixels = qimage_array(image)
    assert distinct_colors(pixels[:, :, :3]) == {(0, 0, 0), (255, 255, 255)}
    assert (pixels == pixels[0]).all(), "모든 줄이 같아야 합니다"

    left_quiet, right_quiet = (11, 7) if symbology == "ean13" else (10, 10)
    total = len(modules) + left_quiet + right_quiet
    module_px = width // total
    assert module_px >= 1
    start = (width - total * module_px) // 2 + left_quiet * module_px
    assert row_modules(pixels[0], module_px, start, len(modules)) == modules
    assert not (pixels[0, :start, 0] < 128).any()
    assert not (pixels[0, start + len(modules) * module_px:, 0] < 128).any()


def test_linear_without_quiet_zone_and_custom_colors(qimage_array):
    modules = barcodes.linear_modules("code128", "A1")
    image = barcodes.render_linear(
        "code128", "A1", len(modules) * 3, 20, quiet_zone=False, foreground="#102030", background="#f0e0d0"
    )
    pixels = qimage_array(image)
    assert distinct_colors(pixels[:, :, :3]) == {(0x10, 0x20, 0x30), (0xF0, 0xE0, 0xD0)}
    assert row_modules(pixels[5], 3, 0, len(modules)) == modules


def test_transparent_background(qimage_array):
    image = barcodes.render_linear("code128", "A1", 400, 20, background=None, foreground="#ffffff")
    pixels = qimage_array(image)
    assert distinct_colors(pixels) == {(0, 0, 0, 0), (255, 255, 255, 255)}


def test_linear_too_narrow_reports_minimum():
    needed = barcodes.linear_min_width_px("code128", "HANA-2026-0001")
    assert needed == len(barcodes.linear_modules("code128", "HANA-2026-0001")) + 20
    with pytest.raises(BarcodeSizeError, match=f"최소 {needed}px") as info:
        barcodes.render_linear("code128", "HANA-2026-0001", needed - 1, 50)
    assert info.value.min_width_px == needed
    assert barcodes.render_linear("code128", "HANA-2026-0001", needed, 50).width() == needed


def test_qr_matrix_shape_and_levels():
    matrix = barcodes.qr_matrix("HELLO", "M")
    assert len(matrix) == 21 and all(len(row) == 21 for row in matrix)
    finder = [row[:7] for row in matrix[:7]]
    assert all(finder[0]) and all(finder[6]) and finder[1][1:6] == [False] * 5
    assert finder[3][2:5] == [True] * 3

    payload = "https://example.com/members/" + "0123456789" * 6
    sizes = [len(barcodes.qr_matrix(payload, level)) for level in ("L", "M", "Q", "H")]
    assert sizes == sorted(sizes) and sizes[0] < sizes[-1]
    assert barcodes.qr_matrix("x", "h") == barcodes.qr_matrix("x", "H")


def test_qr_errors():
    with pytest.raises(BarcodeError, match="비어"):
        barcodes.qr_matrix("")
    with pytest.raises(BarcodeError, match="L, M, Q, H"):
        barcodes.qr_matrix("x", "Z")
    with pytest.raises(BarcodeError, match="너무 깁니다"):
        barcodes.qr_matrix("가" * 3000, "H")
    with pytest.raises(BarcodeSizeError, match="최소 25px") as info:
        barcodes.render_qr("HELLO", 24)
    assert info.value.min_width_px == barcodes.qr_min_size_px("HELLO") == 25


@pytest.mark.parametrize("size", [236, 301, (300, 240)])
def test_qr_render_is_exact_size_two_colors_square_modules(qimage_array, size):
    data = "https://example.com/id/1024"
    matrix = barcodes.qr_matrix(data, "Q")
    image = barcodes.render_qr(data, size, error_correction="Q", border_modules=2)
    width, height = (size, size) if isinstance(size, int) else size
    assert (image.width(), image.height()) == (width, height)

    pixels = qimage_array(image)
    assert distinct_colors(pixels[:, :, :3]) == {(0, 0, 0), (255, 255, 255)}

    n = len(matrix)
    module_px = min(width, height) // (n + 4)
    left, top = (width - n * module_px) // 2, (height - n * module_px) // 2
    dark = pixels[:, :, 0] < 128
    for r in range(n):
        for c in range(n):
            cell = dark[top + r * module_px: top + (r + 1) * module_px, left + c * module_px: left + (c + 1) * module_px]
            assert cell.all() if matrix[r][c] else not cell.any()
    outside = dark.copy()
    outside[top: top + n * module_px, left: left + n * module_px] = False
    assert not outside.any()


def test_qr_is_decodable(qimage_array):
    data = "https://example.com/id/1024?name=hana"
    pixels = qimage_array(barcodes.render_qr(data, 330, error_correction="M"))
    gray = np.ascontiguousarray(pixels[:, :, 0])
    decoded, _, _ = cv2.QRCodeDetector().detectAndDecode(gray)
    assert decoded == data


def test_ean13_is_decodable(qimage_array):
    # OpenCV 바코드 검출기는 EAN/UPC만 지원하고, 모듈 2px 근처에서 가장 안정적으로 읽는다.
    width = barcodes.linear_min_width_px("ean13", "8801234567893") * 2
    pixels = qimage_array(barcodes.render_linear("ean13", "8801234567893", width, 120))
    canvas = np.full((pixels.shape[0] + 120, pixels.shape[1] + 120), 255, dtype=np.uint8)
    canvas[60:-60, 60:-60] = pixels[:, :, 0]
    found, texts, kinds = cv2.barcode.BarcodeDetector().detectAndDecodeWithType(canvas)[:3]
    assert found
    assert (list(texts), list(kinds)) == (["8801234567893"], ["EAN_13"])


# ---------------------------------------------------------------- 다눌과 같은 종류(39 Extended, 93, 128-A/B/C)

def _bits(symbology: str, data: str) -> str:
    return "".join("1" if m else "0" for m in barcodes.linear_modules(symbology, data))


@pytest.mark.parametrize("code_set, data", [("C", "20261117"), ("B", "Hello-b9"), ("A", "\x01AB")])
def test_code128_forced_set_matches_python_barcode(code_set, data):
    """한 집합만 쓰는 입력은 python-barcode가 만드는 것과 모듈 하나까지 같아야 한다."""
    import barcode

    expected = barcode.get_barcode_class("code128")(data).build()[0]
    assert _bits(f"code128{code_set.lower()}", data) == expected


def test_code128_forced_sets_start_with_their_own_start_code():
    from barcode.charsets import code128 as table

    for code_set, data in (("a", "AB12"), ("b", "AB12"), ("c", "1234")):
        start = table.CODES[table.START_CODES[code_set.upper()]]
        assert _bits(f"code128{code_set}", data).startswith(start)
        assert _bits(f"code128{code_set}", data).endswith(CODE128_STOP)


def test_code93_reference_checksum_test93():
    """널리 쓰이는 예: TEST93의 체크 문자는 '+6'이다."""
    bits = _bits("code93", "TEST93")
    body = bits[9:-10]
    chars = [body[i:i + 9] for i in range(0, len(body), 9)]
    decoded = "".join((barcodes._CODE93_CHARS + "????")[barcodes._CODE93_PATTERNS.index(c)] for c in chars)
    assert decoded == "TEST93+6"
    assert bits.startswith(barcodes._CODE93_START_STOP) and bits.endswith(barcodes._CODE93_START_STOP + "1")


def test_code93_patterns_are_well_formed():
    import re

    patterns = barcodes._CODE93_PATTERNS
    assert len(patterns) == 47 and len(set(patterns)) == 47
    for pattern in patterns:
        runs = [len(run) for run in re.findall(r"1+|0+", pattern)]
        assert len(pattern) == 9 and len(runs) == 6 and pattern[0] == "1" and max(runs) <= 4


def test_code39_extended_uses_full_ascii_pairs():
    plain = _bits("code39", "+A/A")
    assert _bits("code39ext", "a!") == plain
    assert barcodes._FULL_ASCII["\\"] == "%L" and barcodes._FULL_ASCII[chr(127)] == "%T"
    assert len(barcodes._FULL_ASCII) == 128


@pytest.mark.parametrize("symbology, data, message", [
    ("code128c", "123", "짝수"),
    ("code128c", "12AB", "숫자만"),
    ("code128a", "abc", "쓸 수 없는"),
    ("code39ext", "한글", "쓸 수 없는"),
    ("code93", "é", "쓸 수 없는"),
])
def test_new_symbologies_reject_bad_data(symbology, data, message):
    with pytest.raises(BarcodeError, match=message):
        barcodes.normalize(symbology, data)


@pytest.mark.parametrize("symbology", ["code128a", "code128b", "code128c", "code39ext", "code93"])
def test_new_symbologies_render_exact_size(qimage_array, symbology):
    data = "1234" if symbology == "code128c" else "AB12"
    image = barcodes.render_linear(symbology, data, 600, 80)
    assert (image.width(), image.height()) == (600, 80)
