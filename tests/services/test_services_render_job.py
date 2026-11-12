"""render_job: 사진 조정값 변환, 도련 처리, 지연 렌더링, 임시 파일 정리."""
import os

import pytest
from PIL import Image

from data.store import PhotoAdjustment
from design import presets, renderer
from design.binding import ColumnSource, CounterSource
from design.model import Crop, ImageItem, RectItem, TextItem
from printer.base import CAP_DUAL_SIDE, CAP_MAGNETIC, CAP_UV, CAP_WHITE
from services import render_job as rj
from services.render_job import (
    PrintOptions, RenderJob, adjustment_from_crop, crop_from_adjustment, cut_pixel_size,
)


# ── 사진 조정값 변환 ─────────────────────────────────────────────────────────
def test_adjustment_offset_is_frame_fraction_and_crop_offset_is_slack_fraction():
    # 200x100 사진을 100x100 틀에 채우면 가로로 100이 남는다(한쪽 50).
    crop = crop_from_adjustment(PhotoAdjustment(1.0, 0.25, 0.0), (200, 100), (100, 100), "fill")
    # 틀의 25% = 25 이동, 남는 폭의 절반 50 대비 0.5
    assert crop.offset_x == pytest.approx(0.5)
    assert crop.offset_y == 0.0
    assert crop.zoom == 1.0


def test_converted_crop_moves_the_image_by_the_requested_distance():
    adjustment = PhotoAdjustment(1.5, -0.1, 0.2)
    image, frame = (300, 400), (30.0, 40.0)
    crop = crop_from_adjustment(adjustment, image, frame, "fill")
    x, y, w, h = renderer.image_target_rect(*image, *frame, "fill", crop)
    assert (x + w / 2) - frame[0] / 2 == pytest.approx(adjustment.offset_x * frame[0])
    assert (y + h / 2) - frame[1] / 2 == pytest.approx(adjustment.offset_y * frame[1])
    assert w == pytest.approx(frame[0] * 1.5)


@pytest.mark.parametrize("fit", ["fill", "contain", "stretch"])
def test_adjustment_round_trips_through_crop(fit):
    adjustment = PhotoAdjustment(1.4, 0.03, -0.05)     # 세 방식 모두 남는 폭 안쪽의 이동
    image, frame = (640, 480), (25.0, 32.0)
    crop = crop_from_adjustment(adjustment, image, frame, fit)
    back = adjustment_from_crop(crop, image, frame, fit)
    assert back.zoom == pytest.approx(adjustment.zoom)
    assert back.offset_x == pytest.approx(adjustment.offset_x)
    assert back.offset_y == pytest.approx(adjustment.offset_y)


def test_offset_beyond_the_slack_is_clamped_unless_asked_not_to():
    adjustment = PhotoAdjustment(1.0, 0.9, 0.0)
    assert crop_from_adjustment(adjustment, (200, 100), (100, 100)).offset_x == 1.0
    assert crop_from_adjustment(adjustment, (200, 100), (100, 100), clamp=False).offset_x == pytest.approx(1.8)


def test_axis_without_slack_cannot_be_shifted():
    # 세로는 틀과 딱 맞아 Crop으로는 옮길 수 없다.
    crop = crop_from_adjustment(PhotoAdjustment(1.0, 0.0, 0.3), (200, 100), (100, 100))
    assert crop.offset_y == 0.0


def test_default_adjustment_is_the_default_crop():
    assert crop_from_adjustment(PhotoAdjustment(), (120, 160), (20, 30)) == Crop()


# ── 도련 ────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("orientation, expected", [("landscape", (1016, 638)), ("portrait", (638, 1016))])
def test_bleed_is_removed_once_and_output_is_the_cut_size(orientation, expected):
    document = presets.new_document(presets.CUT_54X86_BLEED, orientation)
    document.side("front").add_item(RectItem(x=0, y=0, w=90, h=90, fill="#336699", stroke_color=None))
    assert expected == (round(86 * 300 / 25.4), round(54 * 300 / 25.4))[:: 1 if orientation == "landscape" else -1]
    job = RenderJob.for_copies(document, 1)
    try:
        card = job.render(0)
        with Image.open(card.front.color_path) as image:
            assert image.size == expected
            assert image.size == cut_pixel_size(document, "front", 300)
            # 드라이버의 기존 도련 제거가 한 번 더 자르지 않는 크기다.
            from utils.bleed_crop import get_bleed_crop_box
            assert get_bleed_crop_box(*image.size) is None
        assert card.front.orientation == orientation
        assert card.options["bleed_handled"] is True
        assert not rj.driver_would_crop_again(document, "front")
    finally:
        job.close()


def test_bleed_trim_keeps_the_area_inside_the_cut_line():
    document = presets.new_document(presets.CUT_54X86_BLEED)
    side = document.side("front")
    side.add_item(RectItem(x=0, y=0, w=90, h=58, fill="#ff0000", stroke_color=None))
    side.add_item(RectItem(x=2, y=2, w=86, h=54, fill="#0000ff", stroke_color=None))
    job = RenderJob.for_copies(document, 1)
    try:
        with Image.open(job.render(0).front.color_path) as image:
            pixels = image.convert("RGB")
            assert pixels.getpixel((3, 3)) == (0, 0, 255)
            assert pixels.getpixel((pixels.width - 4, pixels.height - 4)) == (0, 0, 255)
            assert pixels.getpixel((pixels.width // 2, pixels.height // 2)) == (0, 0, 255)
    finally:
        job.close()


def test_document_without_bleed_is_not_cropped():
    document = presets.new_document(presets.CR80)
    job = RenderJob.for_copies(document, 1)
    try:
        with Image.open(job.render(0).front.color_path) as image:
            assert image.size == renderer.side_pixel_size(document, "front", 300)
    finally:
        job.close()


def test_legacy_ratio_is_only_cropped_again_without_the_bleed_handled_mark():
    document = presets.new_document(presets.CardPreset("legacy", "58x90", 90.0, 58.0))
    assert rj.driver_would_crop_again(document, "front", bleed_handled=False)
    assert not rj.driver_would_crop_again(document, "front"), "RenderJob 카드는 드라이버가 다시 자르지 않는다"


@pytest.mark.parametrize("orientation", ["landscape", "portrait"])
def test_trim_bleed_agrees_with_the_renderer_crop(orientation):
    document = presets.new_document(presets.CUT_54X86_BLEED, orientation)
    document.side("front").add_item(RectItem(x=1, y=1, w=30, h=30, fill="#336699", stroke_color=None))
    ours = renderer.render_side(document, "front", dpi=300)
    rj.trim_bleed(ours, document, "front")
    theirs = renderer.render_side(document, "front", dpi=300, crop_bleed=True)
    assert ours.color.size == theirs.color.size == cut_pixel_size(document, "front", 300)
    assert ours.color.tobytes() == theirs.color.tobytes()


# ── 카드 목록 · 바인딩 ───────────────────────────────────────────────────────
def test_each_record_gets_its_own_binding_context(store, simple_document, monkeypatch):
    seen = []
    original = renderer.render_side

    def spy(document, side_key, record=None, dpi=300, context=None, **kwargs):
        seen.append((record["이름"], context.counter_index, context.record_key, dpi, kwargs))
        return original(document, side_key, record, dpi, context, **kwargs)

    monkeypatch.setattr(renderer, "render_side", spy)
    ids = store.ids()
    job = RenderJob.for_records(simple_document, store, ids[1:4], PrintOptions(counter_offset=10))
    try:
        for index in range(job.total):
            job.render(index)
            job.release(index)
    finally:
        job.close()
    assert [(name, counter, key) for name, counter, key, _, _ in seen] == [
        ("이영희", 10, str(ids[1])), ("박민수", 11, str(ids[2])), ("최지우", 12, str(ids[3])),
    ]
    assert all(dpi == 300 for _, _, _, dpi, _ in seen)
    assert all(kwargs["crop_bleed"] is False and kwargs["design_mode"] is False for *_, kwargs in seen)
    assert [spec.display_name for spec in job.specs] == ["이영희", "박민수", "최지우"]


def test_binding_record_carries_resolved_photo_paths(store, simple_document, photo_dir, monkeypatch):
    records = []
    original = renderer.render_side
    monkeypatch.setattr(renderer, "render_side",
                        lambda d, s, record=None, *a, **k: records.append(record) or original(d, s, record, *a, **k))
    job = RenderJob.for_records(simple_document, store, store.ids()[:1])
    try:
        card = job.render(0)
    finally:
        job.close()
    assert records[0]["사진"] == str(photo_dir / "kim.png")
    assert "_id" not in records[0]
    assert card.warnings == []


def test_missing_photo_is_a_warning_or_a_failure_when_strict(store, simple_document):
    record_id = store.ids()[0]
    store.update(record_id, {"사진": "nobody"})
    job = RenderJob.for_records(simple_document, store, [record_id])
    try:
        assert job.render(0).front is not None
    finally:
        job.close()

    store.update(record_id, {"사진": "C:/없는폴더/nobody.png"})
    simple_document.side("front").items[1].source = ColumnSource(column="이름")
    strict = RenderJob.for_records(simple_document, store, [record_id], PrintOptions(strict=True))
    try:
        with pytest.raises(rj.RenderJobError) as caught:
            strict.render(0)
        assert "김철수" in str(caught.value)
    finally:
        strict.close()


def test_deleted_record_fails_with_a_korean_message(store, simple_document):
    record_id = store.ids()[0]
    job = RenderJob.for_records(simple_document, store, [record_id])
    store.delete([record_id])
    try:
        with pytest.raises(rj.RenderJobError, match="지워진 레코드"):
            job.render(0)
    finally:
        job.close()


def test_render_job_works_on_a_snapshot_of_the_document(store, simple_document):
    job = RenderJob.for_records(simple_document, store, store.ids()[:1])
    try:
        simple_document.side("front").items.clear()
        assert len(job.document.side("front").items) == 2
    finally:
        job.close()


# ── 사진 조정값 적용 ─────────────────────────────────────────────────────────
def _two_tone_photo(path):
    image = Image.new("RGB", (200, 100), (255, 0, 0))
    image.paste((0, 0, 255), (100, 0, 200, 100))
    image.save(path)


def test_store_photo_adjustment_is_applied_to_the_bound_image(store, photo_dir):
    _two_tone_photo(photo_dir / "kim.png")
    document = presets.new_document(presets.CR80)
    item = document.side("front").add_item(
        ImageItem(name="사진", x=10, y=10, w=20, h=20, source=ColumnSource(column="사진")))
    first, second = store.ids()[:2]
    store.update(second, {"사진": "kim"})
    # 틀의 절반만큼 오른쪽으로: 남는 폭 전부 → 왼쪽(빨강) 절반만 보인다.
    store.set_photo_adjustment(first, "사진", PhotoAdjustment(1.0, 0.5, 0.0))

    job = RenderJob.for_records(document, store, [first, second])
    scale = 300 / 25.4
    probe = (round(25 * scale), round(20 * scale))     # 틀의 오른쪽 1/4 지점
    try:
        with Image.open(job.render(0).front.color_path) as adjusted:
            assert adjusted.convert("RGB").getpixel(probe)[0] > 200      # 빨강
        with Image.open(job.render(1).front.color_path) as untouched:
            assert untouched.convert("RGB").getpixel(probe)[2] > 200     # 조정 없음: 파랑
    finally:
        job.close()
    assert item.record_crops == {}
    assert all(i.record_crops == {} for _, i in job.document.iter_items() if i.kind == "image")


def test_design_record_crop_is_kept_when_the_store_has_no_adjustment(store, photo_dir):
    _two_tone_photo(photo_dir / "kim.png")
    document = presets.new_document(presets.CR80)
    item = document.side("front").add_item(
        ImageItem(name="사진", x=10, y=10, w=20, h=20, source=ColumnSource(column="사진")))
    first = store.ids()[0]
    item.set_record_crop(first, Crop(offset_x=1.0))
    job = RenderJob.for_records(document, store, [first])
    scale = 300 / 25.4
    try:
        with Image.open(job.render(0).front.color_path) as image:
            assert image.convert("RGB").getpixel((round(25 * scale), round(20 * scale)))[0] > 200
    finally:
        job.close()


# ── 프린터 기능에 따른 레이어 ─────────────────────────────────────────────────
def _layered_document():
    document = presets.new_document(presets.CR80, dual_side=True)
    front = document.side("front")
    front.add_item(RectItem(x=5, y=5, w=20, h=20, fill="#ff0000"))
    front.add_item(RectItem(x=30, y=5, w=20, h=20, layer="white"))
    front.add_item(TextItem(x=5, y=30, w=40, h=8, text="UV", layer="uv"))
    document.side("back").add_item(RectItem(x=5, y=5, w=20, h=20, fill="#00ff00"))
    return document


def test_white_uv_and_back_are_included_when_the_printer_supports_them():
    job = RenderJob.for_copies(_layered_document(), 1, capabilities={CAP_WHITE, CAP_UV, CAP_DUAL_SIDE})
    try:
        card = job.render(0)
        assert os.path.exists(card.front.white_mask_path)
        assert os.path.exists(card.front.uv_path)
        assert card.back is not None and card.back.white_mask_path is None and card.back.uv_path is None
        assert job.is_dual_side
    finally:
        job.close()


def test_unsupported_layers_and_back_side_are_left_out():
    job = RenderJob.for_copies(_layered_document(), 1, capabilities={"status"})
    try:
        card = job.render(0)
        assert os.path.exists(card.front.color_path)
        assert card.front.white_mask_path is None and card.front.uv_path is None
        assert card.back is None
        assert not job.is_dual_side
        assert sorted(os.listdir(job.temp_dir)) == ["card00000_front_color.png"]
    finally:
        job.close()


def test_plain_design_has_no_mask_files(store, simple_document):
    job = RenderJob.for_records(simple_document, store, store.ids()[:1])
    try:
        card = job.render(0)
        assert card.front.white_mask_path is None and card.front.uv_path is None and card.back is None
    finally:
        job.close()


def test_magnetic_tracks_come_from_the_mapping(store, simple_document):
    options = PrintOptions(magnetic={"track1": "사번", "track2": "부서"}, offset_x_mm=0.5)
    job = RenderJob.for_records(simple_document, store, store.ids()[:1], options, {CAP_MAGNETIC})
    try:
        card_job = job.card_job(0)
        assert card_job.options["magnetic"] == {"track1": "A0001", "track2": "영업부"}
        assert card_job.options["offset_x_mm"] == 0.5
        assert card_job.record_id == store.ids()[0]
    finally:
        job.close()

    simple_document.extra["magnetic"] = {"track2": "사번"}
    from_document = RenderJob.for_records(simple_document, store, store.ids()[:1], None, {CAP_MAGNETIC})
    without = RenderJob.for_records(simple_document, store, store.ids()[:1], options, {"status"})
    try:
        assert from_document.render(0).options["magnetic"] == {"track2": "A0001"}
        assert "magnetic" not in without.render(0).options
    finally:
        from_document.close()
        without.close()


# ── 장수 · 지연 렌더링 · 정리 ─────────────────────────────────────────────────
def test_static_copies_share_one_render():
    document = presets.new_document(presets.CR80)
    document.side("front").add_item(TextItem(x=5, y=5, w=40, h=10, text="방문증"))
    job = RenderJob.for_copies(document, 4)
    try:
        paths = {job.render(i).front.color_path for i in range(4)}
        assert len(paths) == 1 and job.render_count == 1
        assert all(spec.record_id is None for spec in job.specs)
        for index in range(3):
            job.release(index)
        assert os.path.exists(next(iter(paths)))      # 마지막 장이 아직 쓴다
        job.release(3)
        assert not os.path.exists(next(iter(paths)))
    finally:
        job.close()


def test_copies_with_a_counter_are_rendered_separately():
    document = presets.new_document(presets.CR80)
    document.side("front").add_item(
        TextItem(x=5, y=5, w=40, h=10, source=CounterSource(start=1, pad=4)))
    job = RenderJob.for_copies(document, 3)
    try:
        assert len({job.render(i).front.color_path for i in range(3)}) == 3
        assert [spec.counter_index for spec in job.specs] == [0, 1, 2]
    finally:
        job.close()


def test_record_copies_share_the_render_and_the_counter(store, simple_document):
    job = RenderJob.for_records(simple_document, store, store.ids()[:2], PrintOptions(copies=2))
    try:
        assert [(s.record_id, s.counter_index) for s in job.specs] == [
            (store.ids()[0], 0), (store.ids()[0], 0), (store.ids()[1], 1), (store.ids()[1], 1)]
        assert job.render(0) is job.render(1)
    finally:
        job.close()


def test_iterating_renders_one_card_at_a_time_and_cleans_up(store, simple_document):
    job = RenderJob.for_records(simple_document, store, store.ids())
    assert job.temp_dir is None and job.render_count == 0      # 만들기만 해서는 렌더링하지 않는다
    seen = []
    previous = None
    for card_job in job:
        assert os.path.exists(card_job.front.color_path)
        if previous is not None:
            assert not os.path.exists(previous)
        previous = card_job.front.color_path
        seen.append(card_job.record_id)
        assert job.live_count == 1
    assert seen == store.ids()
    assert job.max_live == 1 and job.render_count == 5
    folder = job.temp_dir
    assert os.listdir(folder) == []
    job.close()
    assert not os.path.exists(folder)


def test_close_removes_the_job_folder_even_with_cards_left(store, simple_document):
    job = RenderJob.for_records(simple_document, store, store.ids())
    job.render(0)
    job.render(1)
    folder = job.temp_dir
    assert len(os.listdir(folder)) == 2
    job.close()
    assert not os.path.exists(folder)
    with pytest.raises(rj.RenderJobError):
        job.render(2)


def test_subset_keeps_the_original_counter(store, simple_document):
    job = RenderJob.for_records(simple_document, store, store.ids(), PrintOptions(counter_offset=100))
    retry = job.subset([1, 3])
    try:
        assert [(s.record_id, s.counter_index, s.render_key) for s in retry.specs] == [
            (store.ids()[1], 101, 0), (store.ids()[3], 103, 1)]
    finally:
        job.close()
        retry.close()


def test_print_options_round_trip():
    options = PrintOptions(copies=2, magnetic={"track1": "사번"}, name_column="이름", stop_on_error=False,
                           capabilities={"white"})
    restored = PrintOptions.from_dict(options.to_dict())
    assert restored.copies == 2 and restored.magnetic == {"track1": "사번"}
    assert restored.stop_on_error is False and restored.capabilities is None
    assert "capabilities" not in options.to_dict()


def test_stored_default_adjustment_and_missing_adjustment_both_keep_the_design_crop(store, simple_document):
    photo = next(item for item in simple_document.side("front").items if item.kind == "image")
    photo.crop = Crop(offset_x=0.4, offset_y=0.0, zoom=1.3)
    first, second = store.ids()[:2]
    store.set_photo_adjustment(first, "사진", PhotoAdjustment())      # 기본값을 일부러 저장
    assert store.has_photo_adjustment(first, "사진") and not store.has_photo_adjustment(second, "사진")
    job = RenderJob.for_records(simple_document, store, [first, second])
    try:
        for index in range(job.total):
            job.render(index)
        assert photo.record_crops == {}, "렌더링 뒤에는 문서 사본을 원래대로 돌려놓는다"
        assert job.render_count == 2
    finally:
        job.close()
