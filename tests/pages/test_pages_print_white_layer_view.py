"""인쇄 미리 보기: 흰색·은색(W·S)이 깔리는 자리를 검게 보여 준다."""
from design.model import RectItem
from design.presets import CR80, new_document
from ui.pages.printing.preview import LAYER_WHITE, render_card_image


def gray(image, x, y):
    return image.pixelColor(x, y).lightness()


def test_white_layer_view_is_dark_where_ink_goes(qapp):
    document = new_document(CR80)
    side = document.side("front")
    side.add_item(RectItem(x=0, y=0, w=85.6, h=54, fill="#2456E6", stroke_color=None))
    side.add_item(RectItem(x=30, y=15, w=25, h=24, fill="#FFFFFF", stroke_color=None, layer="white"))
    color = render_card_image(document, None, None)
    shown = render_card_image(document, None, None, layer=LAYER_WHITE)
    assert shown.size() == color.size()
    assert gray(shown, shown.width() // 2, shown.height() // 2) < 40      # 깔리는 곳은 검게
    assert gray(shown, 10, 10) > 215                                        # 나머지는 희게


def test_no_white_layer_shows_a_blank_card(qapp):
    document = new_document(CR80)
    document.side("front").add_item(RectItem(x=10, y=10, w=20, h=20, fill="#000000", stroke_color=None))
    shown = render_card_image(document, None, None, layer=LAYER_WHITE)
    assert gray(shown, 10, 10) > 215 and gray(shown, shown.width() // 2, shown.height() // 2) > 215
