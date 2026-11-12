import os

import numpy as np
import pytest
from PIL import Image
from PySide6.QtGui import QColor, QImage, QPainter

from design import barcodes
from design.binding import BindingContext, ColumnSource
from design.model import (
    BarcodeItem, Crop, Document, EllipseItem, ImageItem, LineItem, QRItem, RectItem, TextItem,
)
from design.renderer import (
    RenderError, cut_pixel_size, fitted_font_size_pt, image_target_rect, paint_item, paint_item_local,
    render_side, resolve_font_family, side_pixel_size, text_extent, trim_bleed, value_for,
)

S = 300 / 25.4
RED, BLUE, WHITE, BLACK = (255, 0, 0), (0, 0, 255), (255, 255, 255), (0, 0, 0)


def px(mm: float) -> int:
    return int(round(mm * S))


def color_array(document, side_key="front", **kwargs) -> np.ndarray:
    return np.array(render_side(document, side_key, **kwargs).color)


def ink_box_mm(array: np.ndarray, dpi: float) -> tuple[float, float, float, float]:
    dark = array[:, :, :3].min(axis=2) < 128
    rows, cols = np.where(dark.any(axis=1))[0], np.where(dark.any(axis=0))[0]
    assert rows.size and cols.size, "그려진 것이 없습니다"
    k = 25.4 / dpi
    return (cols[0] * k, rows[0] * k, (cols[-1] + 1) * k, (rows[-1] + 1) * k)


def test_output_size_and_modes():
    document = Document()
    rendered = render_side(document, "front")
    assert rendered.color.mode == "RGB"
    assert rendered.color.size == (1011, 638) == side_pixel_size(document, "front")
    assert (rendered.white_mask, rendered.uv, rendered.warnings) == (None, None, [])
    assert rendered.orientation == "landscape" and rendered.dpi == 300
    assert np.array(rendered.color).min() == 255
    assert render_side(document, "front", dpi=150).color.size == (506, 319)
    with pytest.raises(KeyError):
        render_side(document, "옆면")


def test_portrait_swaps_width_and_height():
    document = Document()
    document.side("front").orientation = "portrait"
    document.side("front").add_item(RectItem(x=40, y=70, w=10, h=10, fill="#ff0000", stroke_color=None))
    rendered = render_side(document, "front")
    assert rendered.color.size == (638, 1011)
    assert rendered.orientation == "portrait"
    array = np.array(rendered.color)
    assert tuple(array[px(75), px(45)]) == RED
    assert render_side(document, "back").color.size == (1011, 638)


def test_rect_lands_on_expected_pixels():
    document = Document()
    document.side("front").add_item(RectItem(x=10, y=5, w=20, h=10, fill="#ff0000", stroke_color=None))
    array = color_array(document)
    left, right, top, bottom = 10 * S, 30 * S, 5 * S, 15 * S   # 118.11, 354.33, 59.06, 177.17
    red = (array == RED).all(axis=2)
    white = (array == WHITE).all(axis=2)

    assert red[int(top) + 1: int(bottom), int(left) + 1: int(right)].all()
    outside = np.ones(red.shape, dtype=bool)
    outside[int(top): int(bottom) + 1, int(left): int(right) + 1] = False
    assert white[outside].all()
    # 경계 픽셀까지 합친 잉크 양이 면적과 일치한다(위치·크기가 1px 안쪽으로 정확).
    coverage = (255 - array[:, :, 1].astype(float)).sum() / 255
    assert coverage == pytest.approx((right - left) * (bottom - top), rel=0.002)


def test_stroke_stays_inside_item_box():
    document = Document()
    document.side("front").add_item(
        RectItem(x=10, y=10, w=20, h=10, fill=None, stroke_color="#0000ff", stroke_width_mm=1.0)
    )
    array = color_array(document)
    assert tuple(array[px(10.5), px(20)]) == BLUE
    assert tuple(array[px(19.5), px(20)]) == BLUE
    assert tuple(array[px(15), px(10.5)]) == BLUE
    assert tuple(array[px(15), px(20)]) == WHITE
    assert tuple(array[px(9.8), px(20)]) == WHITE
    assert tuple(array[px(20.2), px(20)]) == WHITE


def test_rounded_rect_and_ellipse_corners_are_empty():
    document = Document()
    side = document.side("front")
    side.add_item(RectItem(x=10, y=10, w=20, h=20, fill="#ff0000", stroke_color=None, corner_radius_mm=5))
    side.add_item(EllipseItem(x=40, y=10, w=30, h=20, fill="#0000ff", stroke_color=None))
    array = color_array(document)
    assert tuple(array[px(10.4), px(10.4)]) == WHITE
    assert tuple(array[px(12), px(12)]) == RED
    assert tuple(array[px(20), px(10.3)]) == RED
    assert tuple(array[px(20), px(55)]) == BLUE
    assert tuple(array[px(11), px(41)]) == WHITE
    assert tuple(array[px(20), px(40.5)]) == BLUE


def test_rotation_is_about_item_center():
    document = Document()
    item = document.side("front").add_item(
        RectItem(x=25, y=20, w=30, h=10, fill="#ff0000", stroke_color=None, rotation=90)
    )
    array = color_array(document)
    cx, cy = 40, 25
    assert tuple(array[px(cy), px(cx)]) == RED
    assert tuple(array[px(cy + 14), px(cx)]) == RED
    assert tuple(array[px(cy - 14), px(cx)]) == RED
    assert tuple(array[px(cy), px(cx + 6)]) == WHITE
    assert tuple(array[px(cy), px(cx - 14)]) == WHITE
    assert ink_box_mm(255 - (array == RED).all(axis=2)[:, :, None] * np.uint8(255), 300) == pytest.approx(
        item.bounds(), abs=0.2
    )


def test_rotation_45_matches_model_bounds():
    document = Document()
    item = document.side("front").add_item(
        RectItem(x=30, y=20, w=20, h=6, fill="#000000", stroke_color=None, rotation=45)
    )
    array = color_array(document)
    assert ink_box_mm(array, 300) == pytest.approx(item.bounds(), abs=0.2)
    assert tuple(array[px(23), px(40)]) == BLACK
    assert tuple(array[px(23), px(33)]) == WHITE


def test_line_item():
    document = Document()
    side = document.side("front")
    side.add_item(LineItem(x=10, y=10, w=40, h=0, width_mm=1.0, color="#ff0000"))
    side.add_item(LineItem(x=10, y=20, w=20, h=20, width_mm=1.0, color="#0000ff", flip=True))
    array = color_array(document)
    assert tuple(array[px(10), px(30)]) == RED
    assert tuple(array[px(10.8), px(30)]) == WHITE
    assert tuple(array[px(30), px(20)]) == BLUE
    assert tuple(array[px(25), px(15)]) == WHITE
    assert tuple(array[px(35), px(15)]) == BLUE


def test_z_order_decides_overlap():
    document = Document()
    side = document.side("front")
    low = side.add_item(RectItem(x=10, y=10, w=20, h=20, fill="#ff0000", stroke_color=None))
    side.add_item(RectItem(x=20, y=10, w=20, h=20, fill="#0000ff", stroke_color=None))
    assert tuple(color_array(document)[px(20), px(25)]) == BLUE
    side.bring_to_front([low.id])
    assert tuple(color_array(document)[px(20), px(25)]) == RED


def test_hidden_and_non_printable_items_are_skipped():
    document = Document()
    side = document.side("front")
    side.add_item(RectItem(x=5, y=5, w=10, h=10, fill="#ff0000", stroke_color=None, visible=False))
    side.add_item(RectItem(x=25, y=5, w=10, h=10, fill="#ff0000", stroke_color=None, printable=False))
    side.add_item(RectItem(x=45, y=5, w=10, h=10, fill="#ff0000", stroke_color=None, layer="white", printable=False))
    rendered = render_side(document, "front")
    assert np.array(rendered.color).min() == 255
    assert rendered.white_mask is None


def test_layers_are_separated():
    document = Document()
    side = document.side("front")
    side.add_item(RectItem(x=5, y=5, w=10, h=10, fill="#ff0000", stroke_color=None))
    side.add_item(RectItem(x=25, y=5, w=10, h=10, fill="#00ff00", stroke_color=None, layer="white"))
    side.add_item(EllipseItem(x=45, y=5, w=10, h=10, fill="#123456", stroke_color=None, layer="uv"))
    side.add_item(TextItem(x=5, y=30, w=40, h=10, text="화이트 글자", font_size_pt=20, layer="white"))
    rendered = render_side(document, "front")

    color = np.array(rendered.color)
    assert tuple(color[px(10), px(10)]) == RED
    assert tuple(color[px(10), px(30)]) == WHITE and tuple(color[px(10), px(50)]) == WHITE
    assert color[px(30): px(40), px(5): px(45)].min() == 255

    assert rendered.white_mask.mode == "L" and rendered.white_mask.size == rendered.color.size
    white = np.array(rendered.white_mask)
    assert white[px(10), px(30)] == 255
    assert white[px(10), px(10)] == 0 and white[px(10), px(50)] == 0
    assert white[px(30): px(40), px(5): px(45)].max() == 255, "화이트 레이어 글자는 흰색(잉크)으로 나온다"
    assert white[px(50), px(70)] == 0

    uv = np.array(rendered.uv)
    assert uv.dtype == np.uint8 and rendered.uv.mode == "L"
    assert uv[px(10), px(50)] == 255
    assert uv[px(5.3), px(45.3)] == 0, "타원 모서리는 비어 있다"
    assert uv[px(10), px(30)] == 0 and uv[px(10), px(10)] == 0


def test_mask_asset_goes_to_white_layer_at_item_geometry(tmp_path, make_png):
    photo = make_png(size=(200, 100), color=(10, 120, 200))
    mask = make_png(size=(200, 100), color=0, mode="L",
                    painter=lambda draw, image: draw.rectangle([0, 0, 99, 99], fill=255))
    document = Document()
    document.side("front").add_item(ImageItem(x=20, y=10, w=40, h=20, fit="stretch", asset=photo, mask_asset=mask))
    rendered = render_side(document, "front", record={})

    color = np.array(rendered.color)
    assert tuple(color[px(20), px(40)]) == (10, 120, 200)
    white = np.array(rendered.white_mask)
    assert white[px(20), px(30)] == 255, "마스크의 흰 쪽(왼쪽 절반)"
    assert white[px(20), px(50)] == 0, "마스크의 검은 쪽(오른쪽 절반)"
    assert white[px(20), px(15)] == 0 and white[px(35), px(30)] == 0, "항목 밖은 비어 있다"
    ys, xs = np.where(white == 255)
    assert (xs.min(), xs.max() + 1) == pytest.approx((20 * S, 40 * S), abs=1.5)
    assert (ys.min(), ys.max() + 1) == pytest.approx((10 * S, 30 * S), abs=1.5)
    assert rendered.uv is None and rendered.warnings == []


def test_mask_follows_rotation_and_crop(make_png):
    photo = make_png(size=(100, 100), color=(0, 200, 0))
    mask = make_png(size=(100, 100), color=0, mode="L",
                    painter=lambda draw, image: draw.rectangle([0, 0, 99, 49], fill=255))
    document = Document()
    document.side("front").add_item(
        ImageItem(x=30, y=20, w=20, h=20, asset=photo, mask_asset=mask, rotation=90)
    )
    white = np.array(render_side(document, "front", record={}).white_mask)
    # 위쪽 절반이 흰 마스크를 90° 돌리면 오른쪽 절반이 흰색.
    assert white[px(30), px(45)] == 255
    assert white[px(30), px(35)] == 0


def test_black_mask_does_not_erase_other_white_items(make_png):
    photo = make_png(size=(50, 50), color=(0, 0, 0))
    mask = make_png(size=(50, 50), color=0, mode="L")
    document = Document()
    side = document.side("front")
    side.add_item(RectItem(x=10, y=10, w=30, h=30, fill="#000000", stroke_color=None, layer="white"))
    side.add_item(ImageItem(x=20, y=20, w=30, h=30, asset=photo, mask_asset=mask))
    white = np.array(render_side(document, "front", record={}).white_mask)
    assert white[px(30), px(30)] == 255
    assert white[px(45), px(45)] == 0


def test_missing_mask_or_image_is_reported(tmp_path, make_png):
    photo = make_png()
    document = Document()
    side = document.side("front")
    side.add_item(ImageItem(x=5, y=5, w=10, h=10, asset=photo, mask_asset=str(tmp_path / "none.png")))
    side.add_item(ImageItem(x=30, y=5, w=10, h=10, name="증명사진", source=ColumnSource(column="사진")))
    rendered = render_side(document, "front", record={"사진": str(tmp_path / "missing.jpg")})
    assert rendered.white_mask is None
    assert len(rendered.warnings) == 2
    assert any("마스크" in w for w in rendered.warnings) and any("missing.jpg" in w for w in rendered.warnings)
    assert np.array(rendered.color)[px(5): px(15), px(30): px(40)].min() == 255
    with pytest.raises(RenderError, match="missing.jpg"):
        render_side(document, "front", record={"사진": str(tmp_path / "missing.jpg")}, strict=True)

    empty = render_side(document, "front", record={"사진": ""})
    assert len(empty.warnings) == 1, "빈 사진 칸은 경고하지 않는다"


def test_design_placeholders_only_without_record():
    document = Document()
    side = document.side("front")
    side.add_item(ImageItem(x=5, y=5, w=20, h=25, name="사진", source=ColumnSource(column="사진")))
    side.add_item(TextItem(x=30, y=5, w=40, h=10, font_size_pt=16, source=ColumnSource(column="이름")))

    design = color_array(document)
    assert design[px(5): px(30), px(5): px(25)].min() < 255, "사진 자리표시 상자"
    assert design[px(5): px(15), px(30): px(70)].min() < 128, "«이름» 자리표시 글자"

    printed = render_side(document, "front", record={"이름": "", "사진": None})
    assert np.array(printed.color).min() == 255
    assert printed.warnings == []

    forced = color_array(document, record={"이름": "", "사진": None}, design_mode=True)
    assert forced[px(5): px(30), px(5): px(25)].min() < 255


@pytest.mark.parametrize("fit, expected", [
    ("stretch", (0, 0, 20, 20)),
    ("fill", (-10, 0, 40, 20)),
    ("contain", (0, 5, 20, 10)),
])
def test_image_target_rect_fit_modes(fit, expected):
    assert image_target_rect(200, 100, 20, 20, fit) == pytest.approx(expected)


def test_image_target_rect_crop():
    assert image_target_rect(200, 100, 20, 20, "fill", Crop(offset_x=1)) == pytest.approx((0, 0, 40, 20))
    assert image_target_rect(200, 100, 20, 20, "fill", Crop(offset_x=-1)) == pytest.approx((-20, 0, 40, 20))
    assert image_target_rect(200, 100, 20, 20, "fill", Crop(zoom=2)) == pytest.approx((-30, -10, 80, 40))
    assert image_target_rect(200, 100, 20, 20, "fill", Crop(zoom=2, offset_y=1)) == pytest.approx((-30, 0, 80, 40))
    assert image_target_rect(200, 100, 20, 20, "fill", Crop(zoom=0)) == pytest.approx((-10, 0, 40, 20))


def half_and_half(make_png) -> str:
    return make_png(size=(200, 100), color=RED,
                    painter=lambda draw, image: draw.rectangle([100, 0, 199, 99], fill=BLUE))


def test_fit_modes_render(make_png):
    path = half_and_half(make_png)
    document = Document()
    side = document.side("front")
    side.add_item(ImageItem(x=5, y=5, w=20, h=20, fit="fill", asset=path))
    side.add_item(ImageItem(x=30, y=5, w=20, h=20, fit="contain", asset=path))
    side.add_item(ImageItem(x=55, y=5, w=20, h=20, fit="stretch", asset=path))
    array = color_array(document, record={})
    # fill: 가운데 20mm만 보임 → 왼쪽 빨강, 오른쪽 파랑, 위아래 꽉 참
    assert tuple(array[px(6), px(7)]) == RED and tuple(array[px(24), px(23)]) == BLUE
    assert array[px(5) + 2: px(25) - 2, px(25) + 2: px(30) - 2].min() == 255, "상자 밖으로 넘치지 않는다"
    # contain: 위아래 5mm 비고 가운데 띠
    assert tuple(array[px(7), px(40)]) == WHITE and tuple(array[px(23), px(40)]) == WHITE
    assert tuple(array[px(15), px(32)]) == RED and tuple(array[px(15), px(48)]) == BLUE
    # stretch: 전체를 채움
    assert tuple(array[px(6), px(57)]) == RED and tuple(array[px(24), px(73)]) == BLUE


def test_crop_can_differ_per_record(make_png):
    path = half_and_half(make_png)
    document = Document()
    item = document.side("front").add_item(
        ImageItem(x=10, y=10, w=20, h=20, fit="fill", source=ColumnSource(column="사진"))
    )
    item.set_record_crop(7, Crop(offset_x=1.0))
    item.set_record_crop(8, Crop(offset_x=-1.0))
    record = {"사진": path}

    def pixels(key):
        array = color_array(document, record=record, context=BindingContext(record_key=key))
        return tuple(array[px(20), px(12)]), tuple(array[px(20), px(28)])

    assert pixels(None) == (RED, BLUE)
    assert pixels("7") == (RED, RED)
    assert pixels("8") == (BLUE, BLUE)
    assert pixels("9") == (RED, BLUE)


def test_large_photo_is_downscaled_smoothly(make_png):
    def stripes(draw, image):
        for x in range(0, image.width, 2):
            draw.line([x, 0, x, image.height], fill=BLACK)

    path = make_png(size=(2000, 1000), color=WHITE, painter=stripes)
    document = Document()
    document.side("front").add_item(ImageItem(x=10, y=10, w=20, h=10, fit="stretch", asset=path))
    region = color_array(document, record={})[px(11): px(19), px(11): px(29), 0].astype(float)
    assert 110 < region.mean() < 145
    assert region.std() < 12, "1px 줄무늬는 고르게 회색이 되어야 한다(계단 현상 없음)"


def test_value_for_resolves_document_assets(tmp_path):
    document = Document(base_dir=str(tmp_path))
    fixed = ImageItem(asset="assets/a.png")
    bound = ImageItem(source=ColumnSource(column="사진"))
    assert value_for(document, fixed) == os.path.join(str(tmp_path), "assets/a.png")
    assert value_for(document, bound, {"사진": "p.jpg"}) == "p.jpg"
    assert value_for(document, bound, {"사진": "p.jpg"}, BindingContext(photo_base_dir="D:/p")) == os.path.join("D:/p", "p.jpg")
    assert value_for(document, bound) is None
    assert value_for(document, TextItem(source=ColumnSource(column="이름"))) == "«이름»"


def test_paint_item_accepts_pil_and_qimage_values():
    canvas = QImage(300, 200, QImage.Format.Format_RGB32)
    canvas.fill(QColor("white"))
    source = QImage(10, 10, QImage.Format.Format_RGB32)
    source.fill(QColor("blue"))
    painter = QPainter(canvas)
    paint_item(painter, ImageItem(x=1, y=1, w=5, h=5), Image.new("RGB", (10, 10), RED), 10.0)
    paint_item(painter, ImageItem(x=10, y=1, w=5, h=5), source, 10.0)
    painter.end()
    assert canvas.pixelColor(30, 30) == QColor("red")
    assert canvas.pixelColor(125, 30) == QColor("blue")
    assert canvas.pixelColor(80, 30) == QColor("white")


def test_paint_item_local_matches_paint_item():
    item = RectItem(x=4, y=3, w=10, h=6, fill="#ff0000", stroke_color="#000000", stroke_width_mm=0.5, rotation=30)

    def draw(use_local: bool) -> bytes:
        canvas = QImage(240, 160, QImage.Format.Format_RGB32)
        canvas.fill(QColor("white"))
        painter = QPainter(canvas)
        if use_local:
            painter.translate(90, 60)
            painter.rotate(30)
            painter.translate(-50, -30)
            paint_item_local(painter, item, None, 10.0)
        else:
            paint_item(painter, item, None, 10.0)
        painter.end()
        return bytes(canvas.constBits())

    assert draw(True) == draw(False)


def test_painter_state_is_restored():
    canvas = QImage(100, 100, QImage.Format.Format_RGB32)
    painter = QPainter(canvas)
    before = (painter.worldTransform(), painter.pen(), painter.brush(), painter.hasClipping())
    for item, value in (
        (TextItem(text="가", rotation=10), "가"), (ImageItem(), Image.new("RGB", (4, 4))),
        (BarcodeItem(data="A1"), "A1"), (QRItem(data="x"), "x"), (RectItem(rotation=5), None),
    ):
        paint_item(painter, item, value, 5.0)
    assert (painter.worldTransform(), painter.pen(), painter.brush(), painter.hasClipping()) == before
    painter.end()


# ---------------------------------------------------------------- 글자

def test_text_is_drawn_inside_box_with_alignment():
    document = Document()
    side = document.side("front")
    side.add_item(TextItem(x=10, y=10, w=60, h=12, text="홍길동", font_size_pt=14, color="#000000"))
    side.add_item(TextItem(x=10, y=30, w=60, h=12, text="홍길동", font_size_pt=14, h_align="right", v_align="bottom"))
    side.add_item(TextItem(x=10, y=45, w=60, h=8, text="홍길동", font_size_pt=14, h_align="center", v_align="middle"))
    array = color_array(document)

    left, top, right, bottom = ink_box_mm(array[: px(25)], 300)
    assert 10 <= left < 11.5 and 10 <= top < 12.5
    assert 12 < right - left < 18, "14pt 한글 세 글자는 약 15mm"
    assert 3.5 < bottom - top < 5.5

    region = array[px(28): px(43)]
    r_left, r_top, r_right, r_bottom = ink_box_mm(region, 300)
    assert 69 < r_right <= 70.05 and 12.5 < r_bottom + 0 <= 14.05
    assert r_right - r_left == pytest.approx(right - left, abs=0.2)

    c_left, c_top, c_right, c_bottom = ink_box_mm(array[px(44):], 300)
    assert (c_left + c_right) / 2 == pytest.approx(40, abs=0.4)
    assert (c_top + c_bottom) / 2 == pytest.approx(5, abs=0.8)


def test_text_color_bold_and_missing_font_fallback():
    document = Document()
    side = document.side("front")
    side.add_item(TextItem(x=5, y=5, w=70, h=10, text="가나다 ABC", font_size_pt=18, color="#ff0000"))
    side.add_item(TextItem(x=5, y=20, w=70, h=10, text="가나다 ABC", font_size_pt=18, bold=True))
    side.add_item(TextItem(x=5, y=35, w=70, h=10, text="가나다 ABC", font_size_pt=18, font_family="세상에없는글꼴"))
    array = color_array(document).astype(int)

    first = array[px(5): px(17)]
    assert ((first[:, :, 0] > 200) & (first[:, :, 1] < 60) & (first[:, :, 2] < 60)).any()
    assert not ((first[:, :, 0] < 100)).any(), "빨간 글자에는 어두운 빨강 성분이 없다"

    def ink(block):
        return (255 - block[:, :, 1]).sum()

    normal, bold, fallback = ink(first), ink(array[px(19): px(32)]), ink(array[px(34): px(47)])
    assert bold > normal * 1.1
    assert fallback == pytest.approx(normal, rel=0.02), "없는 글꼴은 기본 한글 글꼴로 대체된다"
    assert resolve_font_family("세상에없는글꼴") == resolve_font_family(None)
    assert resolve_font_family(None)


def test_auto_shrink_fits_width_and_wrap_fits_height():
    long_text = "아주 긴 부서 이름을 가진 사람의 직함"
    plain = TextItem(x=5, y=5, w=30, h=8, text=long_text, font_size_pt=16)
    shrink = TextItem(x=5, y=25, w=30, h=8, text=long_text, font_size_pt=16, auto_shrink=True)
    assert fitted_font_size_pt(plain, long_text) == 16
    fitted = fitted_font_size_pt(shrink, long_text)
    assert 4.0 <= fitted < 16
    assert fitted_font_size_pt(shrink, "짧음") == 16

    document = Document()
    document.side("front").add_item(plain)
    document.side("front").add_item(shrink)
    array = color_array(document)
    assert ink_box_mm(array[: px(20)], 300)[2] > 36, "축소하지 않으면 상자를 넘친다"
    left, _, right, _ = ink_box_mm(array[px(22):], 300)
    assert 5 <= left and right <= 35.1
    assert right - left > 24, "들어가는 가장 큰 크기를 고른다"

    wrapped = TextItem(x=5, y=5, w=30, h=40, text=long_text, font_size_pt=16, wrap=True)
    document = Document()
    document.side("front").add_item(wrapped)
    left, top, right, bottom = ink_box_mm(color_array(document), 300)
    assert right <= 35.1 and bottom - top > 10, "줄바꿈으로 여러 줄이 된다"

    wrap_shrink = TextItem(w=30, h=12, text=long_text, font_size_pt=16, wrap=True, auto_shrink=True)
    assert fitted < fitted_font_size_pt(wrap_shrink, long_text) < 16

    floor = TextItem(w=3, h=2, text=long_text, font_size_pt=16, auto_shrink=True, min_font_size_pt=6)
    assert fitted_font_size_pt(floor, long_text) == 6


def test_text_layout_is_identical_across_resolutions():
    text = "하나 스튜디오 카드 디자인 줄바꿈 확인용 문장입니다"
    document = Document()
    document.side("front").add_item(
        TextItem(x=8, y=6, w=37, h=40, text=text, font_size_pt=11, wrap=True, h_align="center")
    )
    boxes = [ink_box_mm(color_array(document, dpi=dpi), dpi) for dpi in (96, 300, 600)]
    for box in boxes[1:]:
        assert box == pytest.approx(boxes[0], abs=0.35)


def test_multiline_text_without_wrap_keeps_explicit_newlines():
    document = Document()
    document.side("front").add_item(TextItem(x=5, y=5, w=60, h=30, text="첫 줄\n둘째 줄\n셋째 줄", font_size_pt=12))
    _, top, _, bottom = ink_box_mm(color_array(document), 300)
    assert bottom - top > 12


# ---------------------------------------------------------------- 바코드 / QR

def code_region(array, item) -> np.ndarray:
    left, top = round(item.x * S), round(item.y * S)
    return array[top: top + round(item.h * S), left: left + round(item.w * S)]


@pytest.mark.parametrize("x, y", [(10.0, 8.0), (10.37, 8.61)])
def test_barcode_in_card_is_crisp_and_matches_modules(x, y):
    item = BarcodeItem(x=x, y=y, w=50, h=12, symbology="code128", data="HANA-0001", show_text=False)
    document = Document()
    document.side("front").add_item(item)
    array = color_array(document)

    colors = {tuple(c) for c in np.unique(array.reshape(-1, 3), axis=0).tolist()}
    assert colors == {BLACK, WHITE}

    dark = array[:, :, 0] < 128
    rows = np.where(dark.any(axis=1))[0]
    cols = np.where(dark.any(axis=0))[0]
    assert rows[0] == pytest.approx(y * S, abs=0.51) and rows[-1] + 1 == pytest.approx((y + 12) * S, abs=1.01)
    assert (dark[rows[0]: rows[-1] + 1] == dark[rows[0]]).all()

    modules = barcodes.linear_modules("code128", "HANA-0001")
    width_px = round(50 * S)
    module_px = width_px // (len(modules) + 20)
    assert module_px >= 2
    line = dark[rows[0], cols[0]: cols[0] + len(modules) * module_px]
    assert line.reshape(len(modules), module_px).all(axis=1).tolist() == modules
    assert cols[-1] + 1 - cols[0] == len(modules) * module_px
    assert x * S <= cols[0] and cols[-1] < (x + 50) * S + 1


def test_barcode_rotated_90_stays_crisp():
    document = Document()
    document.side("front").add_item(
        BarcodeItem(x=20.3, y=10.2, w=40, h=10, data="ROT-90", show_text=False, rotation=90)
    )
    array = color_array(document)
    colors = {tuple(c) for c in np.unique(array.reshape(-1, 3), axis=0).tolist()}
    assert colors == {BLACK, WHITE}
    dark = array[:, :, 0] < 128
    cols = np.where(dark.any(axis=0))[0]
    rows = np.where(dark.any(axis=1))[0]
    assert cols[-1] + 1 - cols[0] == pytest.approx(10 * S, abs=1.01), "세로로 선 바코드의 폭은 10mm"
    assert rows[-1] - rows[0] > 30 * S * 0.6
    assert (dark[:, cols[0]: cols[-1] + 1] == dark[:, cols[0]: cols[0] + 1]).all()


def test_barcode_human_readable_text_is_below_bars():
    with_text = BarcodeItem(x=10, y=10, w=50, h=14, data="HANA-0001", show_text=True)
    document = Document()
    document.side("front").add_item(with_text)
    array = color_array(document)
    dark = array[:, :, 0] < 128
    bar_rows = dark[:, round(10 * S): round(60 * S)].sum(axis=1)
    full = np.where(bar_rows == bar_rows.max())[0]
    assert full[0] == pytest.approx(10 * S, abs=0.6)
    bars_bottom = full[-1] + 1
    assert bars_bottom < (10 + 14) * S - 20, "막대 아래에 글자 자리가 남는다"
    text_zone = dark[bars_bottom + 2: round(24 * S)]
    assert text_zone.any()
    cols = np.where(text_zone.any(axis=0))[0]
    assert (cols[0] + cols[-1]) / 2 == pytest.approx(35 * S, abs=12), "글자는 가운데 정렬"


def test_qr_in_card_is_crisp_and_decodable():
    import cv2

    data = "https://example.com/m/000123"
    item = QRItem(x=30.2, y=15.7, w=22, h=22, data=data, error_correction="Q")
    document = Document()
    document.side("front").add_item(item)
    array = color_array(document)
    colors = {tuple(c) for c in np.unique(array.reshape(-1, 3), axis=0).tolist()}
    assert colors == {BLACK, WHITE}
    decoded, _, _ = cv2.QRCodeDetector().detectAndDecode(np.ascontiguousarray(array[:, :, 0]))
    assert decoded == data


def test_qr_follows_record_binding():
    import cv2

    document = Document()
    document.side("front").add_item(QRItem(x=30, y=15, w=22, h=22, source=ColumnSource(column="회원번호")))
    array = color_array(document, record={"회원번호": "M-000777"})
    decoded, _, _ = cv2.QRCodeDetector().detectAndDecode(np.ascontiguousarray(array[:, :, 0]))
    assert decoded == "M-000777"


def test_invalid_barcode_is_reported_not_printed():
    document = Document()
    document.side("front").add_item(
        BarcodeItem(x=10, y=10, w=50, h=12, name="상품 바코드", symbology="ean13", source=ColumnSource(column="코드"))
    )
    printed = render_side(document, "front", record={"코드": "4006381333930"})
    assert np.array(printed.color).min() == 255
    assert len(printed.warnings) == 1
    assert "상품 바코드" in printed.warnings[0] and "체크 숫자" in printed.warnings[0]
    with pytest.raises(RenderError, match="체크 숫자"):
        render_side(document, "front", record={"코드": "4006381333930"}, strict=True)

    good = render_side(document, "front", record={"코드": "4006381333931"})
    assert good.warnings == [] and np.array(good.color).min() == 0

    design = render_side(document, "front")
    assert design.warnings == []
    assert np.array(design.color)[px(10): px(22), px(10): px(60)].min() < 255, "디자인 화면에는 안내 상자"


def test_barcode_too_small_for_print_is_reported():
    document = Document()
    document.side("front").add_item(BarcodeItem(x=10, y=10, w=8, h=10, data="TOO-LONG-FOR-8MM-0123456789"))
    printed = render_side(document, "front", record={})
    assert len(printed.warnings) == 1 and "너무 좁습니다" in printed.warnings[0]
    assert np.array(printed.color).min() == 255
    preview = render_side(document, "front")
    assert preview.warnings == []
    assert np.array(preview.color)[px(10): px(20), px(10): px(18)].min() < 200, "미리보기는 줄여서라도 보여준다"


def test_barcode_on_white_layer_becomes_mask():
    document = Document()
    document.side("front").add_item(
        BarcodeItem(x=10, y=10, w=50, h=12, data="WHITE-1", show_text=False, layer="white")
    )
    rendered = render_side(document, "front")
    assert np.array(rendered.color).min() == 255
    assert set(np.unique(np.array(rendered.white_mask)).tolist()) == {0, 255}


# ---------------------------------------------------------------- 배경 / 도련 / 저장

def test_background_color_and_asset(make_png):
    document = Document()
    document.side("front").background = "#102030"
    assert tuple(color_array(document)[300, 500]) == (0x10, 0x20, 0x30)

    document.side("front").background = half_and_half(make_png)
    array = color_array(document)
    assert tuple(array[300, 100]) == RED and tuple(array[300, 900]) == BLUE

    document.side("front").background = "assets/none.png"
    rendered = render_side(document, "front")
    assert "배경" in rendered.warnings[0]


def test_crop_bleed_removes_margin():
    document = Document(card_width_mm=90, card_height_mm=58, bleed_mm=2)
    side = document.side("front")
    side.orientation = "portrait"
    side.add_item(RectItem(x=0, y=0, w=58, h=90, fill="#ff0000", stroke_color=None))
    side.add_item(RectItem(x=2, y=2, w=54, h=86, fill="#0000ff", stroke_color=None))
    side.add_item(RectItem(x=0, y=0, w=58, h=90, fill="#ffffff", stroke_color=None, layer="white"))

    full = render_side(document, "front")
    assert full.color.size == (685, 1063)
    cropped = render_side(document, "front", crop_bleed=True)
    assert cropped.color.size == (638, 1016) == cropped.white_mask.size == cut_pixel_size(document, "front")
    array = np.array(cropped.color)
    assert (array[2:-2, 2:-2] == BLUE).all()
    assert Document().bleed_mm == 0 and render_side(Document(), "front", crop_bleed=True).color.size == (1011, 638)


@pytest.mark.parametrize("orientation", ["landscape", "portrait"])
@pytest.mark.parametrize("dpi", [96, 150, 300, 600])
@pytest.mark.parametrize("size_mm, bleed", [((90.0, 58.0), 2.0), ((91.6, 59.98), 3.0), ((88.0, 56.5), 1.25)])
def test_crop_bleed_is_exactly_the_cut_size_and_centred(orientation, dpi, size_mm, bleed):
    document = Document(card_width_mm=size_mm[0], card_height_mm=size_mm[1], bleed_mm=bleed)
    side = document.side("front")
    side.orientation = orientation
    side.add_item(RectItem(x=0, y=0, w=200, h=200, fill="#ffffff", stroke_color=None, layer="uv"))
    width_mm, height_mm = document.side_size_mm("front")
    expected = (round((width_mm - 2 * bleed) * dpi / 25.4), round((height_mm - 2 * bleed) * dpi / 25.4))

    full = render_side(document, "front", dpi=dpi)
    cropped = render_side(document, "front", dpi=dpi, crop_bleed=True)
    assert cropped.color.size == cropped.uv.size == expected == cut_pixel_size(document, "front", dpi)

    assert trim_bleed(full, document, "front") is True
    assert full.color.tobytes() == cropped.color.tobytes()
    assert trim_bleed(full, document, "front") is False, "이미 칼선 크기면 다시 자르지 않는다"


def test_crop_bleed_keeps_the_centre_of_the_full_render():
    document = Document(card_width_mm=90, card_height_mm=58, bleed_mm=2)
    side = document.side("front")
    side.add_item(RectItem(x=0, y=0, w=90, h=58, fill="#ff0000", stroke_color=None))
    side.add_item(RectItem(x=2, y=2, w=86, h=54, fill="#0000ff", stroke_color=None))
    full = render_side(document, "front")
    cropped = render_side(document, "front", crop_bleed=True)
    assert full.color.size == (1063, 685) and cropped.color.size == (1016, 638)
    left, top = (1063 - 1016) // 2, (685 - 638) // 2
    assert cropped.color.tobytes() == full.color.crop((left, top, left + 1016, top + 638)).tobytes()
    array = np.array(cropped.color)
    assert (array[2:-2, 2:-2] == BLUE).all()


def test_rendered_side_save(tmp_path):
    document = Document()
    document.side("front").add_item(RectItem(x=1, y=1, w=5, h=5, layer="white"))
    paths = render_side(document, "front").save(tmp_path / "out", "card1_front")
    assert paths["orientation"] == "landscape" and paths["uv_path"] is None
    assert os.path.basename(paths["color_path"]) == "card1_front_color.png"
    with Image.open(paths["color_path"]) as color, Image.open(paths["white_mask_path"]) as white:
        assert color.size == white.size == (1011, 638)
        assert (color.mode, white.mode) == ("RGB", "L")
        assert color.info["dpi"] == pytest.approx((300, 300), abs=0.1)


# ── text_extent: 글자가 상자 밖으로 나가는 길이 ─────────────────────────────────
def test_text_extent_is_zero_when_the_text_fits_or_there_is_no_text():
    fits = TextItem(w=60, h=12, text="하나", font_size_pt=10)
    assert text_extent(fits, "하나") == (0.0, 0.0, 0.0, 0.0)
    assert text_extent(fits, "") == text_extent(fits, None) == (0.0, 0.0, 0.0, 0.0)
    assert text_extent(RectItem(w=1, h=1), "긴 글자 긴 글자 긴 글자") == (0.0, 0.0, 0.0, 0.0)
    assert text_extent(QRItem(w=5, h=5), "data") == (0.0, 0.0, 0.0, 0.0)


@pytest.mark.parametrize("h_align, sides", [("left", (False, True)), ("right", (True, False)),
                                            ("center", (True, True))])
def test_text_extent_follows_the_alignment(h_align, sides):
    text = "상자보다 훨씬 긴 한 줄짜리 글자입니다"
    item = TextItem(w=20, h=8, text=text, font_size_pt=12, h_align=h_align, v_align="middle")
    left, top, right, bottom = text_extent(item, text)
    assert (left > 1.0, right > 1.0) == sides and top == bottom == 0.0
    if h_align == "center":
        assert left == pytest.approx(right, abs=0.05)
    scaled = text_extent(item, text, 10.0)
    assert scaled == pytest.approx((left * 10, top * 10, right * 10, bottom * 10))


def test_text_extent_reports_vertical_overflow_and_respects_wrap_and_shrink():
    tall = TextItem(w=40, h=3, text="가", font_size_pt=24, v_align="top")
    left, top, right, bottom = text_extent(tall, "가")
    assert bottom > 3.0 and top == left == right == 0.0
    middle = TextItem(w=40, h=3, text="가", font_size_pt=24, v_align="middle")
    _, over_top, _, over_bottom = text_extent(middle, "가")
    assert over_top > 1.0 and over_top == pytest.approx(over_bottom, abs=0.05)

    long_text = "줄바꿈을 켜면 가로로는 넘치지 않고 아래로 넘친다 " * 3
    wrapped = TextItem(w=25, h=6, text=long_text, font_size_pt=10, wrap=True)
    left, top, right, bottom = text_extent(wrapped, long_text)
    assert right == 0.0 and bottom > 5.0

    shrunk = TextItem(w=25, h=6, text="자동 축소 글자", font_size_pt=30, auto_shrink=True, min_font_size_pt=4)
    assert text_extent(shrunk, "자동 축소 글자") == (0.0, 0.0, 0.0, 0.0)


def test_text_extent_covers_the_ink_that_is_actually_drawn():
    text = "ABCDEFGHIJKLMNOP"
    item = TextItem(x=10, y=10, w=15, h=6, text=text, font_size_pt=11)
    document = Document()
    document.side("front").add_item(item)
    _, _, ink_right, ink_bottom = ink_box_mm(color_array(document), 300)
    left, top, right, bottom = text_extent(item, text)
    assert ink_right > item.x + item.w + 5, "글자는 상자에서 잘리지 않는다"
    assert item.x + item.w + right >= ink_right - 0.3, "넘친 글자는 알려 준 범위 안에 있다"
    assert item.x + item.w + right <= ink_right + 1.5
    assert bottom == 0.0 and ink_bottom <= item.y + item.h + 0.3


def test_text_extent_for_barcode_caption():
    wide = BarcodeItem(w=50, h=14, data="HANA-0001", show_text=True)
    assert text_extent(wide, "HANA-0001") == (0.0, 0.0, 0.0, 0.0)
    narrow = BarcodeItem(w=6, h=14, data="HANA-0001-LONG-CAPTION", show_text=True, text_size_pt=9)
    left, top, right, bottom = text_extent(narrow, "HANA-0001-LONG-CAPTION")
    assert left > 1.0 and left == pytest.approx(right, abs=0.05) and top == 0.0
    assert text_extent(BarcodeItem(w=6, h=14, data="X", show_text=False), "HANA-0001-LONG") == (0.0,) * 4
    assert text_extent(BarcodeItem(w=6, h=14, symbology="ean13", show_text=True), "글자") == (0.0,) * 4


# ── 바코드 글자 바탕의 겹침은 장치 픽셀 기준 ────────────────────────────────────
@pytest.mark.parametrize("zoom", [0.25, 1.0, 3.0])
def test_barcode_caption_background_leaves_no_gap_at_any_painter_scale(zoom):
    item = BarcodeItem(w=50, h=14, data="HANA-0001", show_text=True, background="#ffee88")
    scale = 300 / 25.4
    width, height = round(60 * scale * zoom), round(20 * scale * zoom)
    canvas = QImage(width, height, QImage.Format.Format_RGB32)
    canvas.fill(QColor("#00ff00"))
    painter = QPainter(canvas)
    painter.scale(zoom, zoom)
    painter.translate(5 * scale, 3 * scale)
    paint_item_local(painter, item, "HANA-0001", scale, design_mode=False)
    painter.end()
    array = np.array(_rgb(canvas))
    left, right = round(7 * scale * zoom), round(53 * scale * zoom)
    top, bottom = round(4 * scale * zoom), round(16 * scale * zoom)
    region = array[top:bottom, left:right]
    green = (region[:, :, 0] < 40) & (region[:, :, 1] > 215) & (region[:, :, 2] < 40)
    assert not green.any(), "막대와 글자 바탕 사이로 뒤 배경이 비치지 않는다"


def _rgb(image: QImage) -> Image.Image:
    converted = image.convertToFormat(QImage.Format.Format_RGB888)
    return Image.frombuffer("RGB", (converted.width(), converted.height()), bytes(converted.constBits()),
                            "raw", "RGB", converted.bytesPerLine(), 1).copy()
