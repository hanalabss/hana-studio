"""검정(K 레진) 패널 층: 순수한 검정 글자·선·바코드·QR만 색 그림에서 빼서 따로 만든다."""
import numpy as np

from design import renderer
from design.model import BarcodeItem, Document, LineItem, QRItem, RectItem, TextItem

DPI = 300


def _doc(*items, background="#ffe9a8"):
    document = Document()
    side = document.side("front")
    side.background = background
    for z, item in enumerate(items, start=1):
        item.z = z
        side.add_item(item)
    return document


def _render(document, **kwargs):
    return renderer.render_side(document, "front", record={}, dpi=DPI, design_mode=False, **kwargs)


def _dark(image) -> int:
    return int((np.asarray(image.convert("L")) < 40).sum())


def test_layer_is_off_by_default_and_absent_when_nothing_qualifies():
    document = _doc(TextItem(x=5, y=5, w=60, h=10, text="HONG", font_size_pt=16))
    assert _render(document).black is None
    colored = _doc(TextItem(x=5, y=5, w=60, h=10, text="HONG", color="#c00000"),
                   RectItem(x=5, y=20, w=10, h=10, fill="#000000", stroke_color=None))
    assert _render(colored, black_layer=True).black is None, "색 글자와 도형은 색 패널에 남는다"


def test_black_text_moves_out_of_the_color_image():
    document = _doc(TextItem(x=5, y=5, w=60, h=10, text="HONG GILDONG", font_size_pt=16, bold=True))
    plain, split = _render(document), _render(document, black_layer=True)
    assert _dark(plain.color) > 1000
    assert _dark(split.color) == 0, "색 그림에는 검정 글자가 남지 않는다"
    mask = np.asarray(split.black)
    assert split.black.mode == "L" and split.black.size == split.color.size
    assert set(np.unique(mask)) == {0, 255}, "레진은 중간 농도가 없다"
    assert abs(int((mask == 255).sum()) - _dark(plain.color)) < 0.2 * _dark(plain.color)
    assert tuple(np.asarray(split.color)[10, 10]) == (255, 233, 168), "글자 밑 바탕색은 그대로다"


def test_barcode_and_qr_keep_their_background_in_color_and_send_bars_to_black():
    document = _doc(BarcodeItem(x=5, y=30, w=45, h=14, data="HANA12345"),
                    QRItem(x=60, y=28, w=20, h=20, data="https://example.com"))
    split = _render(document, black_layer=True)
    color = np.asarray(split.color)
    px = round(10 * DPI / 25.4), round(35 * DPI / 25.4)
    assert _dark(split.color) == 0
    assert tuple(color[px[1], px[0]]) == (255, 255, 255), "바코드의 흰 바탕은 색 그림에 남는다"
    assert int((np.asarray(split.black) == 255).sum()) > 5000


def test_item_drawn_above_hides_the_black_ink_beneath_it():
    text = TextItem(x=5, y=5, w=60, h=10, text="HHHHHHHHHHHHHHHH", font_size_pt=16, bold=True)
    cover = RectItem(x=20, y=3, w=20, h=14, fill="#2060ff", stroke_color=None)
    split = _render(_doc(text, cover), black_layer=True)
    mask = np.asarray(split.black)
    s = DPI / 25.4
    covered = mask[round(4 * s):round(16 * s), round(21 * s):round(39 * s)]
    assert int(covered.sum()) == 0, "위에 덮인 부분은 검정 패널에도 찍지 않는다"
    assert int(mask[:, :round(19 * s)].sum()) > 0


def test_line_goes_to_black_and_bleed_trim_keeps_layers_aligned():
    document = _doc(LineItem(x=5, y=20, w=40, h=0, width_mm=0.5))
    document.bleed_mm = 2.0
    split = _render(document, black_layer=True)
    renderer.trim_bleed(split, document, "front")
    assert split.black.size == split.color.size == renderer.cut_pixel_size(document, "front", DPI)
    assert int((np.asarray(split.black) == 255).sum()) > 0


def test_saved_paths_include_the_black_layer(tmp_path):
    document = _doc(TextItem(x=5, y=5, w=60, h=10, text="HONG"))
    paths = _render(document, black_layer=True).save(tmp_path, "card")
    assert paths["black_path"].endswith("card_black.png")
    assert _render(document).save(tmp_path, "plain")["black_path"] is None
