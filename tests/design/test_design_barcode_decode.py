"""그린 바코드를 실제 판독기(zxing-cpp)로 다시 읽어 담은 값이 그대로 나오는지 확인한다."""
import numpy as np
import pytest

from design import barcodes

zxingcpp = pytest.importorskip("zxingcpp")


def _array(image):
    """QImage → 회색조 배열. PNG로 한 번 저장해 읽는다(메모리 배치에 기대지 않게)."""
    import io

    from PIL import Image
    from PySide6.QtCore import QBuffer, QIODevice

    buffer = QBuffer()
    buffer.open(QIODevice.WriteOnly)
    image.save(buffer, "PNG")
    return np.array(Image.open(io.BytesIO(bytes(buffer.data()))).convert("L"))


def _read(image, formats):
    return [(r.format.name, r.text) for r in zxingcpp.read_barcodes(_array(image), formats=formats)]


@pytest.mark.parametrize("symbology, data, expected, fmt", [
    ("code128", "Hana-128", "Hana-128", "Code128"),
    ("code128a", "HANA128A", "HANA128A", "Code128"),
    ("code128b", "Hana128b", "Hana128b", "Code128"),
    ("code128c", "12345678", "12345678", "Code128"),
    ("code39", "HANA-39", "HANA-39", "Code39"),
    ("code93", "HANA93", "HANA93", "Code93"),
    ("ean13", "880123456789", "8801234567893", "EAN13"),
    ("ean8", "1234567", "12345670", "EAN8"),
    ("upca", "03600029145", "0036000291452", "UPCA"),      # 판독기는 EAN-13 모양(앞에 0)으로 돌려준다
    ("itf", "12345678", "12345678", "ITF"),
    ("codabar", "12345", "A12345A", "Codabar"),
    ("codabar", "B9876D", "B9876D", "Codabar"),
])
def test_linear_barcodes_read_back(qapp, symbology, data, expected, fmt):
    width = barcodes.linear_min_width_px(symbology, data) * 3
    image = barcodes.render_linear(symbology, data, width, 90)
    assert _read(image, getattr(zxingcpp.BarcodeFormat, fmt)) == [(fmt, expected)]


@pytest.mark.parametrize("symbology, data, fmt", [
    ("qr", "HANA-QR-OK", "QRCode"),
    ("datamatrix", "HANA-DM-123456", "DataMatrix"),
    ("datamatrix", "1234567890" * 6, "DataMatrix"),
    ("pdf417", "HANA-PDF417-0001", "PDF417"),
    ("pdf417", "EMP-0042;김하나;개발팀;2026-11-17" .encode("ascii", "ignore").decode() * 4, "PDF417"),
])
def test_2d_codes_read_back(qapp, symbology, data, fmt):
    matrix = barcodes.matrix_2d(data, symbology)
    size = (len(matrix[0]) + 8) * 4, (len(matrix) + 8) * 4
    image = barcodes.render_qr(data, size, symbology=symbology, border_modules=4)
    assert _read(image, getattr(zxingcpp.BarcodeFormat, fmt)) == [(fmt, data)]


def test_pdf417_is_wide_and_rejects_empty():
    matrix = barcodes.matrix_2d("HANA-PDF417-0001", "pdf417")
    assert len(matrix[0]) > 2 * len(matrix)
    with pytest.raises(barcodes.BarcodeError):
        barcodes.matrix_2d("", "pdf417")


@pytest.mark.parametrize("symbology, data, message", [
    ("ean8", "12345678", "체크 숫자"),
    ("upca", "12345", "11자리 또는 12자리"),
    ("itf", "123", "짝수"),
    ("codabar", "12AB34", "쓸 수 없는 문자"),
])
def test_new_linear_kinds_explain_bad_values(symbology, data, message):
    with pytest.raises(barcodes.BarcodeError, match=message):
        barcodes.normalize(symbology, data)
