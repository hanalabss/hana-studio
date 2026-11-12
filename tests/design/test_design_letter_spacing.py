"""글자 사이 간격: '홍길동'을 '홍 길 동'처럼 벌려 쓴다."""
import numpy as np

from design.model import TextItem, item_from_dict
from design.presets import CR80, new_document
from design.renderer import fitted_font_size_pt, render_side


def ink_span(spacing, align="center", shrink=False, text="홍길동"):
    document = new_document(CR80)
    document.side("front").add_item(TextItem(x=10, y=10, w=60, h=12, text=text, font_size_pt=20, h_align=align,
                                             v_align="middle", letter_spacing=spacing, auto_shrink=shrink))
    image = np.array(render_side(document, "front", dpi=300).color.convert("L"))
    columns = np.where((image < 128).sum(axis=0) > 0)[0]
    return columns.min(), columns.max()


def test_spacing_spreads_the_text_and_keeps_it_centred(qapp):
    left0, right0 = ink_span(0)
    left1, right1 = ink_span(100)
    assert (right1 - left1) > (right0 - left0) * 1.5
    assert abs((left1 + right1) / 2 - (left0 + right0) / 2) <= 3        # 가운데 맞춤이 쏠리지 않는다


def test_right_aligned_text_still_ends_at_the_box_edge(qapp):
    assert abs(ink_span(100, "right")[1] - ink_span(0, "right")[1]) <= 3


def test_auto_shrink_takes_spacing_into_account(qapp):
    wide = TextItem(w=30, h=12, text="남궁민수아", font_size_pt=20, auto_shrink=True, letter_spacing=100)
    plain = TextItem(w=30, h=12, text="남궁민수아", font_size_pt=20, auto_shrink=True)
    assert fitted_font_size_pt(wide, wide.text) < fitted_font_size_pt(plain, plain.text)


def test_spacing_is_saved_and_old_files_default_to_zero(qapp):
    item = TextItem(text="가", letter_spacing=50)
    assert item_from_dict(item.to_dict()).letter_spacing == 50
    data = item.to_dict()
    data.pop("letter_spacing")
    assert item_from_dict(data).letter_spacing == 0
