import os

import numpy as np
import pytest
from PIL import Image, ImageDraw

from design import cutout, renderer, storage
from design.cutout import CutoutSettings
from design.model import Document, ImageItem, RectItem

DPI = 100
S = DPI / 25.4


def _at(image: Image.Image, x_mm: float, y_mm: float):
    return image.getpixel((round(x_mm * S), round(y_mm * S)))


def _near(color, expected, tolerance=6) -> bool:
    return all(abs(int(a) - int(b)) <= tolerance for a, b in zip(color, expected))


def _save_rgba(path, rgb, alpha) -> str:
    cutout.compose(rgb, alpha).save(path)
    return str(path)


@pytest.fixture
def cut_document(tmp_path):
    """주황 카드 위에 파란 바탕·빨간 주인공 그림. 가운데 절반만 남긴 배경 지운 그림을 갖고 있다."""
    document = Document()
    document.base_dir = storage.new_workdir()
    source = Image.new("RGB", (200, 200), (20, 60, 200))
    ImageDraw.Draw(source).rectangle((50, 50, 149, 149), fill=(220, 30, 30))
    source.save(tmp_path / "photo.png")
    alpha = np.zeros((200, 200), dtype=np.uint8)
    alpha[50:150, 50:150] = 255
    asset = storage.import_asset(document.base_dir, tmp_path / "photo.png")
    cut = storage.import_asset(document.base_dir, _save_rgba(tmp_path / "cut.png", np.array(source), alpha))
    Image.fromarray(cutout.white_mask(alpha), "L").save(tmp_path / "mask.png")
    mask = storage.import_asset(document.base_dir, tmp_path / "mask.png")
    side = document.side("front")
    side.background = "#F08C1E"
    item = side.add_item(ImageItem(name="사진", x=10.0, y=10.0, w=40.0, h=40.0, asset=asset, fit="stretch"))
    yield document, item, cut, mask
    storage.cleanup_workdir(document)


# ---------------------------------------------------------------- 모델·파일

def test_cutout_asset_round_trips_and_is_listed_as_an_asset():
    item = ImageItem(asset="assets/a.png", mask_asset="assets/m.png", cutout_asset="assets/c.png")
    data = item.to_dict()
    assert data["cutout_asset"] == "assets/c.png"
    again = ImageItem.from_dict(data)
    assert again.cutout_asset == "assets/c.png" and again.mask_asset == "assets/m.png"

    document = Document()
    document.side("front").add_item(item)
    assert document.asset_refs() == {"assets/a.png", "assets/m.png", "assets/c.png"}


def test_old_item_without_cutout_field_loads_and_keeps_unknown_fields():
    old = ImageItem(asset="assets/a.png", mask_asset="assets/m.png").to_dict()
    del old["cutout_asset"]                     # 예전 파일에는 이 이름이 없다
    old["나중에_생길_값"] = {"a": 1}
    item = ImageItem.from_dict(old)
    assert item.cutout_asset is None and item.mask_asset == "assets/m.png"
    assert item.extra == {"나중에_생길_값": {"a": 1}}
    saved = item.to_dict()
    assert saved["나중에_생길_값"] == {"a": 1} and saved["cutout_asset"] is None


def test_storage_packs_the_cutout_and_refuses_when_it_is_missing(cut_document, tmp_path):
    document, item, cut, mask = cut_document
    item.cutout_asset, item.mask_asset = cut, mask
    path = storage.save(document, tmp_path / "card.hsd")
    reopened = storage.load(path)
    try:
        photo = next(it for _side, it in reopened.iter_items())
        assert photo.cutout_asset == cut
        assert os.path.isfile(os.path.join(reopened.base_dir, cut))
        assert os.path.isfile(os.path.join(reopened.base_dir, mask))
    finally:
        storage.cleanup_workdir(reopened)

    item.cutout_asset = "assets/none.png"
    with pytest.raises(storage.StorageError):
        storage.save(document, tmp_path / "broken.hsd")


# ---------------------------------------------------------------- 렌더러

def test_cutout_shows_the_card_colour_where_the_background_was(cut_document):
    document, item, cut, _mask = cut_document
    before = renderer.render_side(document, "front", None, dpi=DPI).color
    assert _near(_at(before, 13, 13), (20, 60, 200))            # 지우기 전: 그림의 파란 바탕

    item.cutout_asset = cut
    after = renderer.render_side(document, "front", None, dpi=DPI)
    assert _near(_at(after.color, 13, 13), (240, 140, 30))      # 배경 자리에 카드 색이 보인다
    assert _near(_at(after.color, 30, 30), (220, 30, 30))       # 주인공은 그대로
    assert _near(_at(after.color, 70, 30), (240, 140, 30))      # 그림 밖
    assert after.white_mask is None and after.warnings == []    # 흰색 깔기를 켜지 않았다


def test_cutout_sits_on_top_of_a_shape_behind_it(cut_document):
    document, item, cut, _mask = cut_document
    item.cutout_asset = cut
    item.z = 5
    document.side("front").add_item(RectItem(x=5.0, y=5.0, w=20.0, h=50.0, fill="#00A050", stroke_color=None, z=1))
    rendered = renderer.render_side(document, "front", None, dpi=DPI).color
    assert _near(_at(rendered, 13, 13), (0, 160, 80))           # 지운 자리로 뒤 도형이 보인다
    assert _near(_at(rendered, 30, 30), (220, 30, 30))


def test_white_mask_is_made_only_when_the_option_is_on(cut_document):
    document, item, cut, mask = cut_document
    item.cutout_asset, item.mask_asset = cut, mask
    rendered = renderer.render_side(document, "front", None, dpi=DPI)
    white = np.array(rendered.white_mask)
    assert white[round(30 * S), round(30 * S)] > 200            # 주인공 밑에만 흰 잉크
    assert white[round(13 * S), round(13 * S)] < 50
    assert _near(_at(rendered.color, 13, 13), (240, 140, 30))

    item.mask_asset = None
    assert renderer.render_side(document, "front", None, dpi=DPI).white_mask is None


def test_old_mask_only_item_renders_exactly_as_before(cut_document):
    """예전 파일: 마스크만 있고 배경 지운 그림은 없다. 컬러는 원래 그림 그대로, 흰색 층만 생긴다."""
    document, item, _cut, mask = cut_document
    plain = renderer.render_side(document, "front", None, dpi=DPI)
    item.mask_asset = mask
    legacy = renderer.render_side(document, "front", None, dpi=DPI)
    assert np.array_equal(np.array(plain.color), np.array(legacy.color))
    assert _near(_at(legacy.color, 13, 13), (20, 60, 200))
    assert legacy.white_mask is not None


def test_missing_cutout_file_falls_back_to_the_original_with_a_warning(cut_document):
    document, item, _cut, _mask = cut_document
    item.cutout_asset = "assets/gone.png"
    rendered = renderer.render_side(document, "front", None, dpi=DPI)
    assert _near(_at(rendered.color, 13, 13), (20, 60, 200))
    assert any("배경을 지운 그림" in text for text in rendered.warnings)
    assert renderer.value_for(document, item, cutout=False).endswith(os.path.basename(item.asset))


def test_cutout_edge_is_antialiased_without_a_halo_of_the_old_background():
    """파란 바탕 위 빨간 원: 가장자리는 반투명이고, 그 색에 파란 기운이 남지 않는다."""
    source = Image.new("RGB", (400, 400), (20, 60, 200))
    ImageDraw.Draw(source).ellipse((100, 100, 299, 299), fill=(220, 30, 30))
    source = source.resize((200, 200), Image.Resampling.LANCZOS)      # 가장자리에 파랑·빨강이 섞인다
    model = Image.new("L", (400, 400), 0)
    ImageDraw.Draw(model).ellipse((100, 100, 299, 299), fill=255)
    model = np.array(model.resize((200, 200), Image.Resampling.LANCZOS))

    rgb, alpha = cutout.cut(np.array(source), model, CutoutSettings(threshold=128))
    assert alpha[100, 100] == 255 and alpha[5, 5] == 0
    edge = (alpha > 20) & (alpha < 235)
    assert edge.sum() > 100                                           # 계단이 아니라 부드러운 가장자리
    visible = alpha > 20
    assert rgb[visible][:, 2].max() < 90                              # 보이는 곳 어디에도 파란 테두리가 없다
    assert rgb[visible][:, 0].min() > 150


def test_soft_alpha_keeps_unsure_areas_enclosed_by_the_subject():
    """모델이 망설인 옷 안쪽(알파 중간값)은 반투명해지지 않고 통째로 남는다. 진짜 배경 구멍은 뚫린 채로 둔다."""
    alpha = np.zeros((120, 120), dtype=np.uint8)
    alpha[20:100, 20:100] = 255
    alpha[40:60, 40:60] = 90            # 망설인 곳
    alpha[70:90, 70:90] = 0             # 팔과 몸 사이 같은 진짜 배경
    out = cutout.soft_alpha(alpha, 128)
    assert out[50, 50] == 255 and out[30, 30] == 255
    assert out[80, 80] == 0 and out[5, 5] == 0
    assert cutout.soft_alpha(alpha, 60)[50, 50] == 255
    raised = cutout.soft_alpha(np.where(alpha == 255, 180, alpha).astype(np.uint8), 200)
    assert raised.max() == 0            # 더 지우기: 기준에 못 미치면 모두 지워진다


# ---------------------------------------------------------------- 바탕색 지우기

def _logo() -> np.ndarray:
    image = Image.new("RGB", (240, 160), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.ellipse((30, 30, 129, 129), fill=(200, 20, 40))
    draw.ellipse((60, 60, 99, 99), fill=(255, 255, 255))           # 글자 속 구멍 같은 흰 곳
    draw.rectangle((150, 50, 209, 109), fill=(20, 40, 120))
    return np.array(image.resize((120, 80), Image.Resampling.LANCZOS))


def test_color_key_removes_a_plain_background_and_inner_holes():
    rgb = _logo()
    assert cutout.corner_color(rgb) == (255, 255, 255)
    out, alpha = cutout.cut(rgb, None, CutoutSettings(method="color"))
    assert alpha[2, 2] == 0 and alpha[78, 118] == 0                 # 바탕
    assert alpha[40, 40] == 0                                       # 안쪽 구멍도 지워진다
    assert alpha[40, 22] == 255 and alpha[40, 90] == 255            # 그림은 온전히 남는다
    assert _near(out[40, 22], (200, 20, 40)) and _near(out[40, 90], (20, 40, 120))
    assert 0.25 < cutout.kept_ratio(alpha) < 0.5
    edge = (alpha > 30) & (alpha < 225)
    assert edge.any()
    assert out[edge].min(axis=1).max() < 140                        # 가장자리에 흰 기운이 남지 않는다


def test_color_key_follows_the_chosen_colour_and_tolerance():
    rgb = _logo()
    blue = CutoutSettings(method="color", color=(20, 40, 120))
    _out, alpha = cutout.cut(rgb, None, blue)
    assert alpha[40, 90] == 0 and alpha[2, 2] == 255 and alpha[40, 22] == 255

    noisy = rgb.copy()
    noisy[:10, :10] = (236, 240, 238)                               # 스캔한 종이처럼 살짝 어두운 바탕
    tight = cutout.cut(noisy, None, CutoutSettings(method="color", color=(255, 255, 255), tolerance=4))[1]
    loose = cutout.cut(noisy, None, CutoutSettings(method="color", color=(255, 255, 255), tolerance=40))[1]
    assert tight[5, 5] > 0 and loose[5, 5] == 0


def test_settings_live_in_extra_without_disturbing_other_values():
    extra = {"메모": 1, cutout.THRESHOLD_KEY: 90}
    settings = CutoutSettings.from_extra(extra)
    assert settings.method == "ai" and settings.threshold == 90
    assert settings.to_extra(extra) == extra

    picked = CutoutSettings(method="color", color=(1, 2, 255), tolerance=55)
    stored = picked.to_extra(extra)
    assert stored == {"메모": 1, cutout.METHOD_KEY: "color", cutout.COLOR_KEY: "#0102ff", cutout.TOLERANCE_KEY: 55}
    assert CutoutSettings.from_extra(stored) == picked.with_changes(threshold=cutout.DEFAULT_THRESHOLD)
    assert cutout.without_settings(stored) == {"메모": 1}
    odd = CutoutSettings.from_extra({cutout.COLOR_KEY: "파랑", cutout.TOLERANCE_KEY: "x", cutout.THRESHOLD_KEY: True})
    assert odd == CutoutSettings()


def test_verdict_flags_wipe_out_and_untouched_results():
    assert cutout.verdict(0.49) == "ok"
    assert cutout.verdict(0.05) == "wipeout" and cutout.verdict(0.149) == "wipeout"
    assert cutout.verdict(0.99) == "untouched"
    assert cutout.verdict(0.4, lost=0.5) == "partial" and cutout.verdict(0.4, lost=0.05) == "ok"


def test_lost_ratio_notices_a_logo_whose_inside_was_removed():
    rgb = _logo()
    model = np.zeros(rgb.shape[:2], dtype=np.uint8)
    model[:, :70] = np.where(np.abs(rgb[:, :70].astype(int) - 255).max(axis=2) > 48, 255, 0)   # 빨간 원만 남김
    final = cutout.soft_alpha(model, 128)
    assert cutout.lost_ratio(rgb, model, final) > 0.2                 # 파란 네모가 통째로 지워졌다
    whole = np.where(np.abs(rgb.astype(int) - 255).max(axis=2) > 48, 255, 0).astype(np.uint8)
    assert cutout.lost_ratio(rgb, whole, cutout.soft_alpha(whole, 128)) < 0.05
    photo = np.random.default_rng(1).integers(0, 255, rgb.shape, dtype=np.uint8)
    assert cutout.lost_ratio(photo, model, final) == 0.0              # 바탕이 여러 색인 사진은 판단하지 않는다
    assert cutout.work_size((6000, 3000)) == (2400, 1200) and cutout.work_size((800, 840)) == (800, 840)


def test_enclosed_hole_with_a_colour_unlike_the_background_is_kept():
    """옷 무늬 속처럼 모델이 배경으로 잘못 본 갇힌 곳은, 색이 배경색과 다르면 메운다. 배경색과 같은 구멍은 둔다."""
    rgb = np.full((120, 120, 3), (140, 170, 210), dtype=np.uint8)        # 한 가지 색 배경
    rgb[20:100, 20:100] = (230, 230, 235)                                # 옷
    rgb[40:55, 40:55] = (250, 250, 250)                                  # 무늬 속(배경색과 다름)
    rgb[70:85, 70:85] = (140, 170, 210)                                  # 팔과 몸 사이(진짜 배경)
    model = np.zeros((120, 120), dtype=np.uint8)
    model[20:100, 20:100] = 255
    model[40:55, 40:55] = 0
    model[70:85, 70:85] = 0
    alpha = cutout.soft_alpha(model, 128, rgb)
    assert alpha[47, 47] > 200          # 무늬 속은 남는다
    assert alpha[77, 77] < 50           # 진짜 배경 구멍은 투명
    assert alpha[5, 5] < 50 and alpha[30, 30] > 200
