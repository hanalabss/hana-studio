"""배경 지우기: 카드에서 실제로 배경이 사라지는지, 흰색 깔기·바탕색 지우기·결과 확인·되돌리기."""

import os
import time

import numpy as np
import pytest
from PIL import Image, ImageDraw
from pages_design_helpers import *  # noqa: F401,F403

from design import cutout, renderer
from ui.pages.design import background

SAMPLE_IMAGE = os.environ.get("HANA_CUTOUT_SAMPLE", os.path.expanduser(r"~\Desktop\테스트이미지.png"))


class RatioEngine(FakeEngine):
    """그림의 위쪽 일부만 사람으로 보는 가짜 엔진. ratio로 남기는 넓이를 정한다."""

    def __init__(self, ratio: float) -> None:
        super().__init__()
        self.ratio = ratio

    def compute_alpha(self, image_path, progress):
        self.calls.append(image_path)
        with Image.open(image_path) as image:
            width, height = image.size
        alpha = np.zeros((height, width), dtype=np.uint8)
        alpha[: round(height * self.ratio)] = 255
        return alpha


def make_logo(path) -> str:
    image = Image.new("RGB", (120, 80), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 59, 59), fill=(200, 20, 40))
    draw.rectangle((70, 20, 99, 59), fill=(20, 40, 120))
    image.save(str(path))
    return str(path)


def make_page(qapp, workspace, **kwargs):
    from ui.pages.design import DesignPage

    widget = DesignPage(workspace, **kwargs)
    widget.dialogs = FakeDialogs()
    widget.resize(1196, 612)
    widget.show()
    pump(qapp)
    widget.start_panel.choose("blank")
    return widget


@pytest.fixture
def editor(page, tmp_path):
    page.start_panel.choose("blank")
    page.photo = page.add_image_file(make_image(tmp_path / "portrait.png", size=(80, 100)))
    return page


def finish(page, qapp, timeout: float = 10.0) -> None:
    section = page.inspector.image
    deadline = time.time() + timeout
    while section.is_busy() and time.time() < deadline:
        section.wait_for_removal(100)
        pump(qapp)
    assert not section.is_busy(), "배경 지우기가 끝나지 않았습니다"
    pump(qapp)


def cut_image(page, item) -> np.ndarray:
    path = renderer.asset_path(page.controller.document, item.cutout_asset)
    assert path and os.path.isfile(path)
    return np.array(Image.open(path).convert("RGBA"))


def place(page, item) -> None:
    page.controller.set_properties(item.id, {"x": 10.0, "y": 5.0, "w": 24.0, "h": 30.0, "fit": "stretch"})


# ---------------------------------------------------------------- 사람·물건

def test_removal_makes_the_background_transparent_on_the_card(editor, qapp):
    section = editor.inspector.image
    editor.controller.document.side("front").background = "#F08C1E"
    place(editor, editor.photo)
    before = editor.controller.undo_stack.index()

    section.remove_button.click()
    finish(editor, qapp)
    assert editor.controller.undo_stack.index() == before + 1          # 한 단계
    rgba = cut_image(editor, editor.photo)
    assert rgba.shape == (100, 80, 4)
    assert rgba[50, 40, 3] == 255 and rgba[5, 5, 3] == 0

    scale = 100 / 25.4
    printed = np.array(renderer.render_side(editor.controller.document, "front", None, dpi=100).color)
    corner = printed[round(7 * scale), round(12 * scale)]
    middle = printed[round(20 * scale), round(22 * scale)]
    assert np.abs(corner.astype(int) - (240, 140, 30)).max() <= 6      # 배경 자리: 카드 색
    assert np.abs(middle.astype(int) - (40, 90, 200)).max() <= 6       # 사람: 그대로
    assert "배경을 지웠습니다" in section.done_row.text()
    assert not section.method_box.isVisible() and section.threshold_group.isVisible()
    assert not section.color_group.isVisible()

    editor.undo()
    assert editor.photo.cutout_asset is None and editor.photo.mask_asset is None
    assert section.remove_button.isVisible() and section.method_box.isVisible()
    restored = np.array(renderer.render_side(editor.controller.document, "front", None, dpi=100).color)
    assert np.abs(restored[round(7 * scale), round(12 * scale)].astype(int) - (40, 90, 200)).max() <= 6
    editor.redo()
    assert editor.photo.cutout_asset


def test_threshold_change_and_restore_are_single_undo_steps(editor, qapp):
    section = editor.inspector.image
    section.start_removal()
    finish(editor, qapp)
    first = editor.photo.cutout_asset
    assert cut_image(editor, editor.photo)[50, 15, 3] == 0              # 흐릿한 가장자리(100)는 128에서 지워진다
    before = editor.controller.undo_stack.index()

    section.set_threshold(60)
    section.set_threshold(50)
    assert editor.controller.undo_stack.index() == before + 1
    assert editor.photo.cutout_asset != first
    assert cut_image(editor, editor.photo)[50, 15, 3] == 255            # 더 남기기
    editor.undo()
    assert editor.photo.cutout_asset == first and section.threshold_slider.value() == 128
    editor.redo()

    section.restore_button.click()
    assert editor.controller.undo_stack.index() == before + 2
    assert editor.photo.cutout_asset is None and editor.photo.mask_asset is None
    editor.undo()
    assert editor.photo.cutout_asset and section.adjust_box.isVisible()


def test_slider_drag_leaves_no_stray_files_behind(editor, qapp):
    section = editor.inspector.image
    section.start_removal()
    finish(editor, qapp)
    assets = os.path.join(editor.controller.document.base_dir, "assets")
    first = set(os.listdir(assets))
    for value in (120, 110, 100, 90, 60, 50):
        section.set_threshold(value)
    after = set(os.listdir(assets))
    assert first <= after                                               # 되돌릴 때 쓸 처음 결과는 남는다
    assert len(after - first) <= 2                                      # 마지막 그림과 마스크만 새로 생겼다
    editor.undo()
    assert os.path.isfile(renderer.asset_path(editor.controller.document, editor.photo.cutout_asset))


# ---------------------------------------------------------------- 흰색 깔기

def test_white_underlay_follows_the_printer(qapp, workspace, tmp_path):
    for capabilities, expected, phrase in (
        ({"white", "dual_side"}, True, "투명 카드"),
        ({"dual_side"}, False, "흰색을 찍지 못해"),
        (None, False, "투명 카드"),
    ):
        page = make_page(qapp, workspace, background_engine=FakeEngine(), capabilities=lambda c=capabilities: c)
        photo = page.add_image_file(make_image(tmp_path / "p.png", size=(80, 100)))
        section = page.inspector.image
        section.start_removal()
        finish(page, qapp)
        assert photo.cutout_asset, capabilities
        assert bool(photo.mask_asset) is expected, capabilities
        assert section.white_box.isVisible() and section.white_check.isChecked() is expected
        assert "흰색 깔기" in section.white_check.text() and phrase in section.white_hint.text()
        rendered = renderer.render_side(page.controller.document, "front", None, dpi=100)
        assert (rendered.white_mask is not None) is expected
        page.close()


def test_white_underlay_toggle_is_one_undo_step_and_survives_adjusting(qapp, workspace, tmp_path):
    page = make_page(qapp, workspace, background_engine=FakeEngine(), capabilities=lambda: set())
    photo = page.add_image_file(make_image(tmp_path / "p.png", size=(80, 100)))
    section = page.inspector.image
    section.start_removal()
    finish(page, qapp)
    assert photo.mask_asset is None
    before = page.controller.undo_stack.index()

    section.white_check.click()
    assert photo.mask_asset and page.controller.undo_stack.index() == before + 1
    mask = np.array(Image.open(renderer.asset_path(page.controller.document, photo.mask_asset)).convert("L"))
    assert mask[50, 40] == 255 and mask[5, 5] == 0
    assert "찍지 못합니다" in section.white_hint.text()

    section.set_threshold(50)                                           # 다듬어도 켠 상태가 유지된다
    assert photo.mask_asset
    page.undo()
    page.undo()
    assert photo.mask_asset is None and not section.white_check.isChecked() and photo.cutout_asset
    page.redo()
    section.white_check.click()
    assert photo.mask_asset is None and photo.cutout_asset


def test_old_mask_only_item_is_explained_and_can_be_upgraded(page, qapp, tmp_path):
    from design import storage

    page.start_panel.choose("blank")
    photo = page.add_image_file(make_image(tmp_path / "p.png", size=(80, 100)))
    mask = storage.import_asset(page.controller.document.base_dir, make_image(tmp_path / "m.png", color=(255, 255, 255)))
    page.controller.set_properties(photo.id, {"mask_asset": mask})
    section = page.inspector.image
    assert "그림은 그대로" in section.done_row.text() and section.white_check.isChecked()
    assert section.redo_button.isVisible() and not section.preview.isVisible()

    section.redo_button.click()
    finish(page, qapp)
    assert photo.cutout_asset and photo.mask_asset and photo.mask_asset != mask


# ---------------------------------------------------------------- 바탕색만 지우기

@pytest.fixture
def logo_editor(page, tmp_path):
    page.start_panel.choose("blank")
    page.logo = page.add_image_file(make_logo(tmp_path / "logo.png"))
    return page


def test_method_choice_explains_both_methods(logo_editor):
    section = logo_editor.inspector.image
    assert section.method_box.isVisible() and section.method() == "ai"
    texts = " ".join(widget.text() for widget in section.method_box.findChildren(type(section.cutout_hint)))
    assert "사람·물건 사진" in texts and "로고·도장·서명" in texts
    assert [button.text() for button in section.method_buttons.values()] == ["사람·물건만 남기기", "바탕색만 지우기"]


def test_color_method_removes_the_plain_background_without_ai(logo_editor, qapp):
    section = logo_editor.inspector.image
    section.method_buttons["color"].click()
    before = logo_editor.controller.undo_stack.index()
    section.remove_button.click()
    finish(logo_editor, qapp)

    logo = logo_editor.logo
    assert section.engine.calls == []                                   # AI를 쓰지 않는다
    assert logo_editor.controller.undo_stack.index() == before + 1
    rgba = cut_image(logo_editor, logo)
    assert rgba[5, 5, 3] == 0 and rgba[40, 40, 3] == 255 and rgba[40, 85, 3] == 255
    assert logo.extra[cutout.METHOD_KEY] == "color" and cutout.THRESHOLD_KEY not in logo.extra
    assert section.color_group.isVisible() and not section.threshold_group.isVisible()
    assert section.color_chip.color() == "#ffffff" and not section.corner_button.isVisible()
    assert not section.preview.pixmap().isNull()


def test_color_can_be_picked_from_the_preview_and_reset_to_the_corner(logo_editor, qapp):
    section = logo_editor.inspector.image
    section.set_method("color")
    section.start_removal()
    finish(logo_editor, qapp)
    logo = logo_editor.logo
    work = section._results[logo.id]
    width = max(120, section.preview.width())
    target = work.preview.target_rect(width, 150)
    before = logo_editor.controller.undo_stack.index()

    # 미리 보기에서 파란 네모(그림의 85/120, 40/80 자리)를 누른다
    section.preview.pressed.emit(target.left() + target.width() * 85 / 120, target.top() + target.height() * 0.5)
    assert logo.extra[cutout.COLOR_KEY] == "#142878"
    assert logo_editor.controller.undo_stack.index() == before + 1
    rgba = cut_image(logo_editor, logo)
    assert rgba[40, 85, 3] == 0 and rgba[5, 5, 3] == 255                # 파랑이 지워지고 흰 바탕이 남는다
    assert section.color_chip.color() == "#142878" and section.corner_button.isVisible()

    section.preview.pressed.emit(1.0, 1.0)                              # 그림 밖: 아무 일도 없다
    assert logo_editor.controller.undo_stack.index() == before + 1

    section.corner_button.click()
    assert cutout.COLOR_KEY not in logo.extra and cut_image(logo_editor, logo)[5, 5, 3] == 0
    logo_editor.undo()
    assert logo.extra[cutout.COLOR_KEY] == "#142878"


def test_tolerance_slider_is_live_and_one_undo_step(logo_editor, qapp, tmp_path):
    image = Image.open(make_logo(tmp_path / "scan.png"))
    image.paste((232, 236, 234), (0, 60, 120, 80))                      # 스캔한 종이처럼 얼룩진 아래쪽
    image.save(tmp_path / "scan.png")
    section = logo_editor.inspector.image
    assert section.replace_image(str(tmp_path / "scan.png"))
    section.set_method("color")
    section.start_removal()
    finish(logo_editor, qapp)
    logo = logo_editor.logo
    before = logo_editor.controller.undo_stack.index()
    section.set_tolerance(4)
    assert cut_image(logo_editor, logo)[70, 60, 3] > 0                  # 딱 그 색만: 얼룩이 남는다
    preview_before = section.preview.pixmap().toImage()

    section.tolerance_slider.setSliderDown(True)
    section.tolerance_slider.setValue(60)
    assert section.preview.pixmap().toImage() != preview_before         # 끄는 중에도 미리 보기는 바뀐다
    assert logo.extra[cutout.TOLERANCE_KEY] == 4
    section.tolerance_slider.setSliderDown(False)
    section.tolerance_slider.sliderReleased.emit()
    assert logo.extra[cutout.TOLERANCE_KEY] == 60
    assert cut_image(logo_editor, logo)[70, 60, 3] == 0                 # 비슷한 색까지: 얼룩도 지워진다
    assert logo_editor.controller.undo_stack.index() == before + 1      # 잇따라 고친 것은 한 단계
    logo_editor.undo()
    assert cutout.TOLERANCE_KEY in logo.extra and logo.extra[cutout.TOLERANCE_KEY] not in (4, 60)


# ---------------------------------------------------------------- 이상한 결과는 그냥 넣지 않는다

@pytest.mark.parametrize("ratio, phrase", [(0.05, "주인공을 찾지 못했습니다"), (0.99, "지울 배경을 찾지 못했습니다")])
def test_suspicious_ai_result_is_shown_instead_of_applied(qapp, workspace, tmp_path, ratio, phrase):
    page = make_page(qapp, workspace, background_engine=RatioEngine(ratio))
    logo = page.add_image_file(make_logo(tmp_path / "logo.png"))
    section = page.inspector.image
    before = page.controller.undo_stack.index()
    section.start_removal()
    finish(page, qapp)

    assert logo.cutout_asset is None and logo.mask_asset is None        # 조용히 지워 버리지 않는다
    assert page.controller.undo_stack.index() == before
    assert section.review_row.isVisible() and phrase in section.review_row.text()
    assert "바탕색만 지우기" in section.review_row.text() and section.review_text()
    assert section.preview.isVisible() and not section.preview.pixmap().isNull()
    assert section.try_color_button.isVisible() and not section.remove_button.isVisible()

    section.discard_button.click()
    assert not section.review_row.isVisible() and section.remove_button.isVisible()
    assert logo.cutout_asset is None


def test_suspicious_result_can_be_swapped_for_the_color_method_or_accepted(qapp, workspace, tmp_path):
    page = make_page(qapp, workspace, background_engine=RatioEngine(0.05))
    logo = page.add_image_file(make_logo(tmp_path / "logo.png"))
    section = page.inspector.image
    section.start_removal()
    finish(page, qapp)

    section.try_color_button.click()
    finish(page, qapp)
    assert logo.cutout_asset and logo.extra[cutout.METHOD_KEY] == "color"
    assert not section.review_row.isVisible() and section.color_group.isVisible()
    rgba = cut_image(page, logo)
    assert rgba[5, 5, 3] == 0 and rgba[40, 40, 3] == 255

    page.undo()
    section.set_method("ai")
    section.start_removal()
    finish(page, qapp)
    assert section.review_row.isVisible()
    section.accept_button.click()                                       # 그래도 쓰겠다면 쓴다
    assert logo.cutout_asset and cutout.METHOD_KEY not in logo.extra
    assert cut_image(page, logo)[70, 60, 3] == 0


def test_logo_that_loses_its_inside_is_also_shown_first(qapp, workspace, tmp_path):
    """AI가 도장의 테두리만 남기고 속을 지운 경우: 남은 넓이는 그럴듯해도 그냥 넣지 않는다."""

    class HalfLogo(FakeEngine):
        def compute_alpha(self, image_path, progress):
            rgb = np.array(Image.open(image_path).convert("RGB")).astype(int)
            alpha = np.where(np.abs(rgb - 255).max(axis=2) > 48, 255, 0).astype(np.uint8)
            alpha[:, 65:] = 0                                            # 파란 네모를 배경으로 잘못 봤다
            return alpha

    page = make_page(qapp, workspace, background_engine=HalfLogo())
    logo = page.add_image_file(make_logo(tmp_path / "logo.png"))
    section = page.inspector.image
    section.start_removal()
    finish(page, qapp)
    assert logo.cutout_asset is None
    assert "일부까지 함께 지워졌습니다" in section.review_row.text() and section.try_color_button.isVisible()


# ---------------------------------------------------------------- 그 밖

def test_bound_photo_explains_why_it_cannot_be_cut_out(editor, workspace, people_store):
    workspace.set_store(people_store)
    item = editor.add("image")
    section = editor.inspector.image
    section.source_control.set_current("column")
    assert item.source is not None
    assert not section.remove_button.isEnabled() and not section.method_box.isVisible()
    hint = section.cutout_hint.text()
    assert "사람마다 사진이 달라" in hint and "그림 파일" in hint and "PNG" in hint


def test_replacing_the_image_drops_the_cutout_and_its_settings(logo_editor, qapp, tmp_path):
    section = logo_editor.inspector.image
    section.set_method("color")
    section.start_removal()
    finish(logo_editor, qapp)
    logo = logo_editor.logo
    logo_editor.controller.set_properties(logo.id, {"extra": {**logo.extra, "메모": 1}})
    assert section.replace_image(make_image(tmp_path / "other.png", color=(5, 5, 5)))
    assert logo.cutout_asset is None and logo.extra == {"메모": 1}
    assert section.method_box.isVisible()


def test_cutout_survives_save_and_reopen(editor, qapp, tmp_path):
    from design import storage

    section = editor.inspector.image
    section.start_removal()
    finish(editor, qapp)
    path = str(tmp_path / "card.hsd")
    editor.controller.save(path)
    reopened = storage.load(path)
    try:
        photo = next(item for _side, item in reopened.iter_items() if item.kind == "image")
        cut = renderer.asset_path(reopened, photo.cutout_asset)
        assert os.path.isfile(cut) and renderer.value_for(reopened, photo) == cut
        assert renderer.value_for(reopened, photo, cutout=False) == renderer.asset_path(reopened, photo.asset)
    finally:
        storage.cleanup_workdir(reopened)


def _model_ready() -> bool:
    try:
        from core.model_downloader import is_model_downloaded
        from core.model_loader import get_model_loader

        return bool(is_model_downloaded(get_model_loader().model_name))
    except Exception:
        return False


def test_real_model_keeps_the_person_in_the_sample_picture(qapp):
    """진짜 모델로 확인한다. 모델을 이미 내려받은 컴퓨터에서, 예시 그림이 있을 때만 돈다(내려받지 않는다)."""
    if not os.path.isfile(SAMPLE_IMAGE):
        pytest.skip("예시 그림이 없습니다 (HANA_CUTOUT_SAMPLE로 경로를 줄 수 있습니다)")
    if not _model_ready():
        pytest.skip("배경 지우기 모델을 내려받지 않은 컴퓨터입니다")
    engine = background.RembgEngine()
    alpha = engine.compute_alpha(SAMPLE_IMAGE, lambda _text: None)
    work = background.build_work(SAMPLE_IMAGE, "ai", alpha)
    settings = background.CutoutSettings(threshold=engine.default_threshold())
    rgb, cut = work.cut(settings)
    ratio = cutout.kept_ratio(cut)
    assert 0.30 <= ratio <= 0.70, ratio
    assert cutout.verdict(ratio) == "ok"
    assert cut[5, 5] == 0 and cut[300, 400] == 255                      # 모서리는 배경, 얼굴은 사람
    solid = np.mean((cut == 0) | (cut == 255))
    assert solid > 0.93                                                 # 반투명은 가장자리뿐이다(옷이 비치지 않는다)
