"""다눌에 있던 꾸미기: 글자 취소선·글자 칸 바탕, 사진 둥근 모서리."""

import numpy as np

from design.model import Document, ImageItem, TextItem
from design.renderer import render_side

S = 300 / 25.4
WHITE = (255, 255, 255)


def px(mm: float) -> int:
    return int(round(mm * S))


def _render(*items, **kwargs):
    document = Document()
    for item in items:
        document.side("front").add_item(item)
    return render_side(document, "front", record={}, **kwargs)


def test_text_background_fills_whole_box_behind_text():
    color = np.array(_render(TextItem(x=10, y=10, w=30, h=10, text="가", background="#00ff00")).color)
    assert tuple(color[px(10.5), px(39.5)]) == (0, 255, 0), "글자 없는 칸 구석도 칠한다"
    assert tuple(color[px(9.5), px(20)]) == WHITE, "칸 밖은 그대로"
    assert (color[px(10):px(20), px(10):px(40)].min(axis=2) < 64).any(), "글자는 바탕 위에 보인다"


def test_text_without_background_is_transparent():
    color = np.array(_render(TextItem(x=10, y=10, w=30, h=10, text="가")).color)
    assert tuple(color[px(10.5), px(39.5)]) == WHITE


def test_strikeout_adds_ink_through_middle_of_text():
    plain = np.array(_render(TextItem(x=5, y=5, w=60, h=12, text="ABCDEFG", font_size_pt=24)).color)
    struck = np.array(_render(TextItem(x=5, y=5, w=60, h=12, text="ABCDEFG", font_size_pt=24,
                                       strikeout=True)).color)
    assert (struck.min(axis=2) < 128).sum() > (plain.min(axis=2) < 128).sum()


def test_black_text_on_dark_background_keeps_background_out_of_black_panel():
    rendered = _render(TextItem(x=10, y=10, w=30, h=10, text="가", color="#000000", background="#202020"),
                       black_layer=True)
    black = np.array(rendered.black)
    assert black[px(10.5), px(39.5)] == 0, "바탕은 검정 패널에 넣지 않는다"
    assert black.max() == 255, "글자는 검정 패널로 간다"
    assert tuple(np.array(rendered.color)[px(10.5), px(39.5)]) == (32, 32, 32), "바탕은 색 패널에 남는다"


def test_image_corner_radius_clips_corners_in_color_and_white(make_png):
    photo = make_png(size=(200, 100), color=(200, 30, 30))
    mask = make_png(size=(200, 100), color=255, mode="L")
    item = ImageItem(x=10, y=10, w=40, h=20, fit="stretch", asset=photo, mask_asset=mask, corner_radius_mm=5)
    rendered = _render(item)
    color, white = np.array(rendered.color), np.array(rendered.white_mask)
    assert tuple(color[px(10.4), px(10.4)]) == WHITE and white[px(10.4), px(10.4)] == 0
    assert tuple(color[px(20), px(30)]) == (200, 30, 30) and white[px(20), px(30)] == 255
    assert tuple(color[px(10.2), px(30)]) == (200, 30, 30), "곧은 변은 그대로"


def test_new_fields_round_trip_and_default_off():
    text = TextItem(text="x", strikeout=True, background="#123456")
    again = TextItem.from_dict(text.to_dict())
    assert again.strikeout and again.background == "#123456"
    old = TextItem.from_dict({"kind": "text", "text": "x"})
    assert old.strikeout is False and old.background is None
    image = ImageItem.from_dict(ImageItem(corner_radius_mm=3).to_dict())
    assert image.corner_radius_mm == 3.0
