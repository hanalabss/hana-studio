import os
import time

import numpy as np
import pytest
from PIL import Image
from pages_design_helpers import *  # noqa: F401,F403

from design import renderer
from ui.pages.design import background


@pytest.fixture
def editor(page, tmp_path):
    page.start_panel.choose("blank")
    page.photo = page.add_image_file(make_image(tmp_path / "portrait.png", size=(80, 100)))
    return page


def _finish(page, qapp, timeout: float = 10.0) -> None:
    section = page.inspector.image
    deadline = time.time() + timeout
    while section.is_busy() and time.time() < deadline:
        section.wait_for_removal(100)
        pump(qapp)
    assert not section.is_busy(), "배경 지우기가 끝나지 않았습니다"
    pump(qapp)


def _mask(page) -> np.ndarray:
    path = renderer.asset_path(page.controller.document, page.photo.mask_asset)
    assert path and os.path.isfile(path)
    return np.array(Image.open(path).convert("L"))


def test_mask_helpers():
    alpha = np.array([[0, 100, 200], [255, 128, 129]], dtype=np.uint8)
    mask = background.mask_from_alpha(alpha, 128)
    assert mask.tolist() == [[0, 0, 255], [255, 0, 255]]        # 흰색 = 사람·물건 = 흰 잉크
    assert background.fit_alpha(alpha, (3, 2)) is not None
    assert background.fit_alpha(alpha, (6, 4)).shape == (4, 6)
    assert background.fit_alpha(alpha, (2, 3)).shape == (3, 2)   # 가로세로가 뒤바뀐 결과도 맞춘다


def test_removal_runs_off_the_gui_thread_and_stores_mask(editor, qapp):
    section = editor.inspector.image
    engine = section.engine
    assert section.remove_button.isVisible() and section.remove_button.isEnabled()
    before = editor.controller.undo_stack.index()

    section.remove_button.click()
    assert section.is_busy()
    assert section.progress_box.isVisible() and not section.remove_button.isVisible()
    assert editor.photo.mask_asset is None                       # 아직 계산 중

    _finish(editor, qapp)
    assert len(engine.calls) == 1 and engine.calls[0].endswith(".png")
    assert editor.photo.mask_asset and editor.photo.mask_asset.startswith("assets/")
    assert editor.controller.undo_stack.index() == before + 1
    mask = _mask(editor)
    assert mask.shape == (100, 80)
    assert mask[50, 40] == 255 and mask[5, 5] == 0               # 가운데(사람)만 흰색
    assert mask[50, 15] == 0                                     # 흐릿한 가장자리(100)는 기준 128에서 지워진다
    assert section.done_row.isVisible() and section.adjust_box.isVisible()
    assert section.threshold_slider.value() == 128
    assert not section.preview.pixmap().isNull()
    assert not section.progress_box.isVisible()


def test_removal_is_undoable_in_one_step(editor, qapp):
    section = editor.inspector.image
    section.start_removal()
    _finish(editor, qapp)
    mask_asset = editor.photo.mask_asset

    editor.undo()
    assert editor.photo.mask_asset is None
    assert section.remove_button.isVisible() and not section.adjust_box.isVisible()
    editor.redo()
    assert editor.photo.mask_asset == mask_asset
    assert section.adjust_box.isVisible()


def test_mask_prints_on_white_layer_under_subject(editor, qapp):
    editor.controller.set_properties(editor.photo.id, {"x": 10.0, "y": 5.0, "w": 24.0, "h": 30.0})
    editor.inspector.image.start_removal()
    _finish(editor, qapp)

    rendered = renderer.render_side(editor.controller.document, "front", None, dpi=100)
    assert rendered.white_mask is not None
    white = np.array(rendered.white_mask)
    scale = 100 / 25.4
    assert white[round(20 * scale), round(22 * scale)] > 200     # 사진 가운데 밑에는 흰 잉크
    assert white[round(7 * scale), round(12 * scale)] < 50       # 사진 가장자리(배경) 밑에는 없음
    assert white[round(45 * scale), round(70 * scale)] == 0      # 사진 밖


def test_threshold_slider_previews_live_and_commits_as_one_step(editor, qapp):
    section = editor.inspector.image
    section.start_removal()
    _finish(editor, qapp)
    first_mask = editor.photo.mask_asset
    before = editor.controller.undo_stack.index()
    preview_before = section.preview.pixmap().toImage()

    section.threshold_slider.setSliderDown(True)                 # 끄는 중: 미리 보기만 바뀐다
    section.threshold_slider.setValue(60)
    assert editor.photo.mask_asset == first_mask
    assert section.preview.pixmap().toImage() != preview_before
    section.threshold_slider.setValue(50)
    section.threshold_slider.setSliderDown(False)
    section.threshold_slider.sliderReleased.emit()

    assert editor.photo.mask_asset != first_mask
    assert _mask(editor)[50, 15] == 255                          # 기준을 낮추면 가장자리도 남는다
    section.set_threshold(40)                                    # 키보드로 조금 더
    assert editor.controller.undo_stack.index() == before + 1

    editor.undo()
    assert editor.photo.mask_asset == first_mask


def test_restore_background_clears_mask(editor, qapp):
    section = editor.inspector.image
    section.start_removal()
    _finish(editor, qapp)
    section.restore_button.click()
    assert editor.photo.mask_asset is None
    assert section.remove_button.isVisible() and not section.done_row.isVisible()
    editor.undo()
    assert editor.photo.mask_asset


def test_engine_failure_is_reported_and_button_returns(editor, qapp):
    section = editor.inspector.image
    section.engine.fail_with = "모델 파일이 없습니다"
    section.start_removal()
    _finish(editor, qapp)
    assert editor.photo.mask_asset is None
    assert section.error_row.isVisible()
    assert "배경을 지우지 못했습니다" in section.error_row.text() and "모델 파일이 없습니다" in section.error_row.text()
    assert section.remove_button.isVisible() and section.remove_button.isEnabled()

    section.engine.fail_with = None
    section.start_removal()
    _finish(editor, qapp)
    assert editor.photo.mask_asset and not section.error_row.isVisible()


def test_declined_preparation_does_nothing(editor):
    section = editor.inspector.image
    section.engine.ready = False                                 # 모델 내려받기를 그만둔 경우
    assert section.start_removal() is False
    assert not section.is_busy() and section.engine.calls == []


def test_cancel_discards_result(editor, qapp):
    section = editor.inspector.image
    section.start_removal()
    job = section._job
    section.cancel_button.click()
    assert not section.is_busy() and section.remove_button.isVisible()
    job.wait(5000)
    pump(qapp)
    assert editor.photo.mask_asset is None


def test_replacing_image_drops_old_cutout(editor, qapp, tmp_path):
    section = editor.inspector.image
    section.start_removal()
    _finish(editor, qapp)
    assert section.replace_image(make_image(tmp_path / "other.png", color=(5, 5, 5)))
    assert editor.photo.mask_asset is None
    assert section.remove_button.isVisible() and not section.adjust_box.isVisible()


def test_existing_mask_without_cached_result_offers_redo(page, qapp, tmp_path):
    page.start_panel.choose("blank")
    photo = page.add_image_file(make_image(tmp_path / "p.png"))
    mask = make_image(tmp_path / "m.png", color=(255, 255, 255))
    from design import storage

    asset = storage.import_asset(page.controller.document.base_dir, mask)
    page.controller.set_properties(photo.id, {"mask_asset": asset})
    section = page.inspector.image
    assert section.done_row.isVisible() and section.redo_button.isVisible()
    assert not section.adjust_box.isVisible() and section.restore_button.isVisible()


def test_default_engine_uses_legacy_ai_stack(monkeypatch, tmp_path):
    """기존 화면이 쓰는 세션(core.model_loader)과 처리기(core.image_processor)를 그대로 쓴다. 모델은 내려받지 않는다."""
    import sys
    import types

    import core.image_processor  # noqa: F401  진짜 처리기를 쓴다(rembg와 세션만 가짜)

    calls = {}

    class Loader:
        model_name = "isnet-general-use"

        def is_ready(self):
            return True

        def get_loaded_session(self):
            return "SESSION"

    def fake_remove(data, session=None):
        import io

        calls["session"] = session
        image = Image.new("RGBA", (8, 6), (0, 0, 0, 0))
        image.putpixel((3, 3), (255, 255, 255, 255))
        buffer = io.BytesIO()
        image.save(buffer, "PNG")
        return buffer.getvalue()

    monkeypatch.setitem(sys.modules, "core.model_loader", types.SimpleNamespace(get_model_loader=lambda: Loader()))
    monkeypatch.setitem(sys.modules, "core.model_downloader", types.SimpleNamespace(
        is_model_downloaded=lambda name: True, ensure_model_downloaded=lambda name, parent=None: True))
    monkeypatch.setitem(sys.modules, "rembg", types.SimpleNamespace(remove=fake_remove))

    engine = background.RembgEngine()
    assert engine.prepare(None) is True
    messages = []
    alpha = engine.compute_alpha(make_image(tmp_path / "x.png", size=(8, 6)), messages.append)
    assert calls["session"] == "SESSION"
    assert alpha.shape == (6, 8) and alpha[3, 3] == 255 and alpha[0, 0] == 0
    assert alpha.dtype == np.uint8
    assert messages


def test_default_engine_hides_processor_errors_behind_plain_wording(monkeypatch, tmp_path):
    import sys
    import types

    import core.image_processor  # noqa: F401

    class Loader:
        def is_ready(self):
            return True

        def get_loaded_session(self):
            return "SESSION"

    def broken_remove(data, session=None):
        raise OSError("onnxruntime: invalid graph")

    monkeypatch.setitem(sys.modules, "core.model_loader", types.SimpleNamespace(get_model_loader=lambda: Loader()))
    monkeypatch.setitem(sys.modules, "rembg", types.SimpleNamespace(remove=broken_remove))

    with pytest.raises(RuntimeError) as caught:
        background.RembgEngine().compute_alpha(make_image(tmp_path / "x.png"), lambda _text: None)
    assert "onnxruntime" not in str(caught.value) and "사진" in str(caught.value)


def test_threshold_is_kept_on_the_item_and_follows_undo(editor, qapp):
    section = editor.inspector.image
    controller = editor.controller
    controller.set_properties(editor.photo.id, {"extra": {"메모": "그대로"}})
    section.start_removal()
    _finish(editor, qapp)
    assert editor.photo.extra == {"메모": "그대로", background.THRESHOLD_KEY: 128}
    first_mask = editor.photo.mask_asset
    before = controller.undo_stack.index()

    section.set_threshold(60)
    assert editor.photo.extra[background.THRESHOLD_KEY] == 60
    assert editor.photo.mask_asset != first_mask
    assert controller.undo_stack.index() == before + 1           # 마스크와 정도가 한 단계
    assert len(section.engine.calls) == 1                        # 기억해 둔 결과를 다시 쓴다

    editor.undo()
    assert editor.photo.mask_asset == first_mask
    assert editor.photo.extra[background.THRESHOLD_KEY] == 128
    assert section.threshold_slider.value() == 128
    editor.redo()
    assert editor.photo.extra[background.THRESHOLD_KEY] == 60 and section.threshold_slider.value() == 60

    editor.undo()
    editor.undo()                                                # 배경 지우기 자체를 되돌린다
    assert editor.photo.mask_asset is None and editor.photo.extra == {"메모": "그대로"}


def test_threshold_survives_save_and_is_reused_when_redoing(editor, qapp, tmp_path):
    from design import storage

    section = editor.inspector.image
    section.start_removal()
    _finish(editor, qapp)
    section.set_threshold(60)
    path = str(tmp_path / "card.hsd")
    editor.controller.save(path)

    reopened = storage.load(path)
    try:
        photo = next(item for _side, item in reopened.iter_items() if item.kind == "image")
        assert background.stored_threshold(photo) == 60 and photo.mask_asset
    finally:
        storage.cleanup_workdir(reopened)

    section._results.clear()                                     # 프로그램을 껐다 켠 것처럼: 계산 결과는 없고 정도만 남는다
    editor.controller.open(path)
    pump(qapp)
    photo = next(item for item in editor.controller.side().items if item.kind == "image")
    editor.controller.select(photo.id)
    pump(qapp)
    assert section.redo_button.isVisible() and not section.adjust_box.isVisible()
    section.redo_button.click()
    _finish(editor, qapp)
    assert section.adjust_box.isVisible() and section.threshold_slider.value() == 60
    assert photo.extra[background.THRESHOLD_KEY] == 60


def test_stored_threshold_ignores_unusable_values():
    from design.model import ImageItem

    item = ImageItem()
    assert background.stored_threshold(item) is None
    for value, expected in ((60, 60), (60.4, 60), (1, 5), (999, 250), ("60", None), (True, None), (None, None)):
        item.extra = {background.THRESHOLD_KEY: value}
        assert background.stored_threshold(item) == expected, value
    item.extra = {"다른 값": 1, background.THRESHOLD_KEY: 60}
    assert background.with_threshold(item, None) == {"다른 값": 1}
    assert background.with_threshold(item, 90) == {"다른 값": 1, background.THRESHOLD_KEY: 90}
    assert item.extra[background.THRESHOLD_KEY] == 60            # 원래 것은 건드리지 않는다


def test_replacing_image_forgets_the_threshold(editor, qapp, tmp_path):
    section = editor.inspector.image
    section.start_removal()
    _finish(editor, qapp)
    section.set_threshold(60)
    assert section.replace_image(make_image(tmp_path / "other.png", color=(5, 5, 5)))
    assert background.THRESHOLD_KEY not in editor.photo.extra
    editor.undo()
    assert editor.photo.extra[background.THRESHOLD_KEY] == 60 and editor.photo.mask_asset
