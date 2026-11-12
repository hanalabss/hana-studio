"""디자인 화면이 기대는 편집기 공개 API: 작업 폴더, 글자 확정, extra 저장, 카드 설정과 재배치, 명단 항목 표시."""

import numpy as np
import pytest
from PySide6.QtGui import QColor

from design import renderer, storage
from design.binding import ColumnSource, ConcatPart, ConcatSource, FixedSource
from design.model import Document, ItemGeometry
from ui.editor import EditorController, EditorError, EditorPalette


def _geometry(controller, side_key="front") -> dict[str, ItemGeometry]:
    return {item.id: ItemGeometry.of(item) for item in controller.document.side(side_key).items}


def _inside(controller, side_key="front") -> bool:
    width, height = controller.document.side_size_mm(side_key)
    for item in controller.document.side(side_key).items:
        left, top, right, bottom = item.bounds()
        if left < -1e-6 or top < -1e-6 or right > width + 1e-6 or bottom > height + 1e-6:
            return False
    return True


# ---------------------------------------------------------------- 공개 이름

def test_ensure_workdir_is_public_and_keeps_the_old_name(controller):
    first = controller.ensure_workdir()
    assert first == controller.document.base_dir
    assert controller._ensure_workdir() == first

    document = Document.from_dict(controller.document.to_dict())
    assert document.base_dir is None
    controller.set_document(document)
    created = controller.ensure_workdir()
    assert created and controller.document.base_dir == created
    assert controller.ensure_workdir() == created


def test_commit_text_edit_is_public_and_keeps_the_old_name(editor):
    controller, view = editor.controller, editor.view
    item = controller.add_item("text", text="처음")
    view.commit_text_edit()                                   # 고치는 중이 아니면 아무 일도 없다
    assert item.text == "처음"

    assert view.begin_text_edit(item.id)
    view.text_editor().setPlainText("고친 글자")
    view.commit_text_edit()
    assert item.text == "고친 글자" and not view.is_editing_text()

    assert view.begin_text_edit(item.id)
    view.text_editor().setPlainText("옛 이름으로도")
    view._commit_text_edit()
    assert item.text == "옛 이름으로도"


# ---------------------------------------------------------------- extra

def test_extra_is_stored_undoably_and_saved(controller, tmp_path):
    item = controller.add_item("image")
    before = controller.undo_stack.index()
    assert controller.set_properties(item.id, {"extra": {"cutout_threshold": 120}})
    assert item.extra == {"cutout_threshold": 120}
    assert controller.undo_stack.index() == before + 1
    assert not controller.set_properties(item.id, {"extra": {"cutout_threshold": 120}})   # 같은 값은 기록하지 않는다

    given = {"cutout_threshold": 80, "메모": ["a"]}
    assert controller.set_properties(item.id, {"extra": given, "name": "증명사진"})
    given["메모"].append("b")                                  # 넘긴 dict를 나중에 고쳐도 항목은 그대로
    assert item.extra == {"cutout_threshold": 80, "메모": ["a"]} and item.name == "증명사진"
    assert controller.undo_stack.index() == before + 2

    path = controller.save(str(tmp_path / "extra.hsd"))
    loaded = storage.load(path)
    try:
        assert loaded.find_item(item.id)[1].extra == {"cutout_threshold": 80, "메모": ["a"]}
    finally:
        storage.cleanup_workdir(loaded)

    controller.undo_stack.undo()
    assert item.extra == {"cutout_threshold": 120} and item.name != "증명사진"
    controller.undo_stack.undo()
    assert item.extra == {}
    controller.undo_stack.redo()
    controller.undo_stack.redo()
    assert item.extra == {"cutout_threshold": 80, "메모": ["a"]}


def test_extra_changes_merge_like_other_properties_and_reject_bad_values(controller):
    item = controller.add_item("image")
    before = controller.undo_stack.index()
    for value in (10, 20, 30):
        controller.set_properties(item.id, {"extra": {"cutout_threshold": value}}, merge_key="slide")
    assert item.extra == {"cutout_threshold": 30}
    assert controller.undo_stack.index() == before + 1
    controller.undo_stack.undo()
    assert item.extra == {}

    for bad in (None, [("a", 1)], "a=1", {1: "숫자 이름"}):
        with pytest.raises(EditorError):
            controller.set_properties(item.id, {"extra": bad})
    assert item.extra == {}
    assert not controller.set_properties(item.id, {"id": "x", "kind": "text", "z": 99})   # 여전히 못 바꾸는 것들


# ---------------------------------------------------------------- 카드 설정 + 재배치

@pytest.fixture
def laid_out(controller):
    """가로 카드 구석구석에 놓인 항목들."""
    made = {
        "띠": controller.add_item("rect", w=85.6, h=10.0),
        "오른쪽 아래": controller.add_item("text", w=30.0, h=8.0),
        "가운데": controller.add_item("qr", w=18.0, h=18.0),
        "돌린 것": controller.add_item("rect", w=20.0, h=10.0),
    }
    controller.move_items_to({
        made["띠"].id: (0.0, 0.0), made["오른쪽 아래"].id: (52.0, 44.0),
        made["가운데"].id: (33.8, 17.99), made["돌린 것"].id: (60.0, 20.0),
    })
    controller.set_rotation([made["돌린 것"].id], 30.0)
    controller.clear_selection()
    return made


def test_orientation_change_keeps_items_on_card_in_one_undo_step(controller, laid_out):
    original = _geometry(controller)
    before = controller.undo_stack.index()

    assert controller.set_card_setup(orientation="portrait")
    assert controller.undo_stack.index() == before + 1
    assert controller.side().orientation == "portrait"
    assert controller.side_size_mm() == (53.98, 85.6)
    moved = _geometry(controller)
    assert moved != original
    assert _inside(controller)
    band = laid_out["띠"]
    assert (band.x, band.y, band.w, band.h) == pytest.approx((0.0, 0.0, 53.98, 10.0))   # 양쪽에 걸친 띠는 새 폭으로
    middle = laid_out["가운데"]
    assert (middle.x + middle.w / 2, middle.y + middle.h / 2) == pytest.approx((26.99, 42.8))
    assert (middle.w, middle.h) == (18.0, 18.0)
    assert laid_out["돌린 것"].rotation == 30.0
    canvas = controller.scene.canvas_item(middle.id)
    assert (canvas.pos().x(), canvas.pos().y()) == pytest.approx((middle.x, middle.y))    # 화면도 따라온다

    controller.undo_stack.undo()
    assert controller.undo_stack.index() == before
    assert controller.side().orientation == "landscape"
    assert _geometry(controller) == original
    canvas = controller.scene.canvas_item(middle.id)
    assert (canvas.pos().x(), canvas.pos().y()) == pytest.approx((33.8, 17.99))

    controller.undo_stack.redo()
    assert controller.side().orientation == "portrait"
    assert _geometry(controller) == moved


def test_set_orientation_reflows_too_and_can_be_told_not_to(controller, laid_out):
    original = _geometry(controller)
    assert controller.set_orientation("portrait")
    assert _inside(controller) and _geometry(controller) != original
    controller.undo_stack.undo()
    assert _geometry(controller) == original

    assert controller.set_orientation("portrait", reflow=False)
    assert controller.side().orientation == "portrait" and _geometry(controller) == original
    controller.undo_stack.undo()
    assert controller.set_card_setup(orientation="portrait", reflow=False)
    assert _geometry(controller) == original


def test_card_size_change_reflows_every_side_and_undoes_together(controller, laid_out):
    controller.set_card_setup(dual_side=True)
    controller.set_side("back")
    back = controller.add_item("text", w=30.0, h=8.0)
    controller.move_items_to({back.id: (55.6, 45.98)})                # 뒷면 오른쪽 아래 구석
    controller.set_side("front")
    front_before, back_before = _geometry(controller, "front"), _geometry(controller, "back")
    document = controller.document
    steps = controller.undo_stack.index()

    assert controller.set_card_setup(width_mm=60.0, height_mm=40.0, name="작은 카드")
    assert controller.undo_stack.index() == steps + 1
    assert (document.card_width_mm, document.card_height_mm, document.name) == (60.0, 40.0, "작은 카드")
    assert _inside(controller, "front") and _inside(controller, "back")
    assert (back.x + back.w, back.y + back.h) == pytest.approx((60.0, 40.0))   # 붙어 있던 구석에 그대로 붙는다
    front_after, back_after = _geometry(controller, "front"), _geometry(controller, "back")

    controller.undo_stack.undo()
    assert (document.card_width_mm, document.card_height_mm, document.name) == (85.6, 53.98, "새 디자인")
    assert _geometry(controller, "front") == front_before and _geometry(controller, "back") == back_before
    controller.undo_stack.redo()
    assert (document.card_width_mm, document.card_height_mm) == (60.0, 40.0)
    assert _geometry(controller, "front") == front_after and _geometry(controller, "back") == back_after


def test_setup_changes_that_keep_the_size_leave_items_alone(controller, laid_out):
    original = _geometry(controller)
    announced = []
    controller.properties_changed.connect(announced.append)
    assert controller.set_card_setup(name="이름만", dual_side=True)
    assert _geometry(controller) == original and announced == []
    assert not controller.set_card_setup(orientation="landscape")      # 이미 가로
    assert controller.set_card_setup(orientation="portrait")
    assert announced and set(announced[-1]) <= set(original)            # 옮긴 항목을 속성 패널에 알린다


def test_reflow_moves_locked_items_and_keeps_selection(controller, laid_out):
    corner = laid_out["오른쪽 아래"]
    controller.set_locked([corner.id], True)
    controller.select([corner.id])
    controller.set_card_setup(orientation="portrait")
    assert _inside(controller) and corner.locked
    assert controller.selection == [corner.id]
    controller.undo_stack.undo()
    assert (corner.x, corner.y) == (52.0, 44.0) and controller.selection == [corner.id]


# ---------------------------------------------------------------- 명단 항목 표시(옅은 노랑)

def _yellowish(color: QColor) -> bool:
    return color.red() > 235 and color.green() > 220 and color.blue() < color.red() - 25


@pytest.fixture
def fields(editor):
    controller = editor.controller
    bound = controller.add_bound_field("이름", "text")
    joined = controller.add_item("text", source=ConcatSource(parts=[
        ConcatPart(column="부서"), ConcatPart(text=" / "), ConcatPart(column="직급")]))
    fixed = controller.add_item("text", source=FixedSource(value=""))
    plain = controller.add_item("text", text="")
    places = {bound.id: (5.0, 5.0), joined.id: (5.0, 16.0), fixed.id: (5.0, 27.0), plain.id: (5.0, 38.0)}
    for item_id, (x, y) in places.items():
        controller.set_properties(item_id, {"x": x, "y": y, "w": 40.0, "h": 8.0})
    controller.clear_selection()
    editor.set_px_per_mm(8.0)
    editor.app.processEvents()
    return bound, joined, fixed, plain


def _probe(editor, item) -> QColor:
    """항목 오른쪽 끝 안쪽(글자가 없는 곳)의 화면 색."""
    editor.app.processEvents()
    point = editor.view.mm_to_viewport(item.x + item.w - 1.0, item.y + item.h - 1.0).toPoint()
    return editor.view.viewport().grab().toImage().pixelColor(point)


def test_bound_fields_show_a_soft_yellow_tint_until_selected(editor, fields):
    bound, joined, fixed, plain = fields
    controller = editor.controller
    assert _yellowish(_probe(editor, bound)) and _yellowish(_probe(editor, joined))
    assert _probe(editor, fixed).name() == "#ffffff" and _probe(editor, plain).name() == "#ffffff"

    controller.select([bound.id])
    assert _probe(editor, bound).name() == "#ffffff"           # 고르면 표시를 거둔다
    assert _yellowish(_probe(editor, joined))
    controller.clear_selection()
    assert _yellowish(_probe(editor, bound))

    controller.set_source([bound.id], None)                    # 명단과 끊으면 사라진다
    controller.clear_selection()
    assert _probe(editor, bound).name() == "#ffffff"
    controller.undo_stack.undo()
    controller.clear_selection()
    controller.set_visible([joined.id], False)
    assert _yellowish(_probe(editor, bound)) and _probe(editor, joined).name() == "#ffffff"


def test_tint_is_subtle_in_both_themes_and_can_be_switched_off(editor, fields):
    bound = fields[0]
    for palette in (EditorPalette.light(), EditorPalette.dark()):
        tint = QColor(palette.field_tint)
        assert tint.lightnessF() > 0.8, palette.name            # 옅다: 흰 바탕만 살짝 물든다
        assert tint.red() == 255 and tint.green() > 230 and 150 < tint.blue() < 215
        editor.view.set_palette(palette)
        assert _yellowish(_probe(editor, bound)), palette.name  # 카드는 어느 테마에서나 흰 바탕

    fired = []
    editor.controller.options_changed.connect(lambda: fired.append(1))
    editor.controller.set_field_tint_visible(False)
    assert fired == [1] and not editor.controller.field_tint_visible
    assert _probe(editor, bound).name() == "#ffffff"
    editor.controller.set_field_tint_visible(True)
    assert _yellowish(_probe(editor, bound))


def test_tint_leaves_dark_content_untouched(editor, fields):
    """형광펜처럼 얹으므로 검은 글자·바코드는 물들지 않고 그대로 읽힌다."""
    controller = editor.controller
    block = controller.add_item("rect", fill="#000000", source=ColumnSource(column="색"))
    controller.set_properties(block.id, {"x": 50.0, "y": 5.0, "w": 20.0, "h": 10.0})
    controller.clear_selection()
    editor.app.processEvents()
    middle = editor.view.mm_to_viewport(60.0, 10.0).toPoint()
    assert editor.view.viewport().grab().toImage().pixelColor(middle).name() == "#000000"
    edge = editor.view.mm_to_viewport(50.0, 10.0).toPoint()     # 대신 테두리로 알린다
    assert editor.view.viewport().grab().toImage().pixelColor(edge).name() != "#000000"


def test_tint_never_reaches_print_or_export(editor, fields, qimage_array):
    controller = editor.controller
    assert _yellowish(_probe(editor, fields[0]))               # 화면에는 보이는 상태에서
    record = {"이름": "김하늘", "부서": "개발팀", "직급": "과장"}

    def renders():
        out = []
        for person in (None, record):
            rendered = renderer.render_side(controller.document, "front", person, dpi=200)
            out.append(np.asarray(rendered.color).copy())
            out.append(np.asarray(renderer.render_side(
                controller.document, "front", person, dpi=200, design_mode=True).color).copy())
        return out

    with_tint = renders()
    exported_with = qimage_array(controller.export_preview_image(dpi=200))
    controller.set_field_tint_visible(False)
    editor.app.processEvents()
    without_tint = renders()
    exported_without = qimage_array(controller.export_preview_image(dpi=200))

    for a, b in zip(with_tint, without_tint):
        assert np.array_equal(a, b)
    assert np.array_equal(exported_with, exported_without)

    twin = EditorController(document=Document.from_dict(controller.document.to_dict()))
    try:                                                        # 캔버스를 한 번도 띄우지 않은 문서와도 같다
        twin.document.base_dir = controller.document.base_dir
        for a, b in zip(with_tint, [
            np.asarray(renderer.render_side(twin.document, "front", person, dpi=200, design_mode=mode).color)
            for person in (None, record) for mode in (None, True)
        ]):
            assert np.array_equal(a, b)
    finally:
        twin.document.base_dir = None
        twin.close()

    blank = with_tint[2]                                        # 실제 인쇄 그림(사람 내용)에는 노란 기가 없다
    assert not ((blank[..., 0].astype(int) - blank[..., 2]) > 25).any()
