import json
import os

import pytest
from PySide6.QtGui import QGuiApplication, QImage

from design import presets, storage
from design.binding import ColumnSource, FixedSource
from design.model import BarcodeItem, RectItem, TextItem
from ui.editor import MIME_ITEMS, EditorController, EditorError


def _rect(controller, x, y, w=10.0, h=10.0, **extra):
    item = controller.add_item("rect", w=w, h=h, **extra)
    controller.move_items_to({item.id: (x, y)})
    return item


def test_new_document_is_clean_cr80(controller):
    document = controller.document
    assert (document.card_width_mm, document.card_height_mm) == (85.6, 53.98)
    assert document.base_dir
    assert controller.current_side == "front"
    assert not controller.is_modified
    assert controller.layers() == []


def test_new_document_from_template_and_preset(controller):
    controller.new_document("cut_54x86_bleed", "employee", orientation="portrait")
    document = controller.document
    assert document.bleed_mm == 2.0
    assert document.side_size_mm("front") == (58.0, 90.0)
    names = [layer.name for layer in controller.layers()]
    assert names[0] == "사번 바코드" and names[-1] == "상단 띠"
    assert controller.undo_stack.count() == 0


@pytest.mark.parametrize("kind", ["text", "image", "rect", "ellipse", "line", "barcode", "qr"])
def test_add_item_places_inside_card_and_selects(controller, kind):
    item = controller.add_item(kind)
    width, height = controller.side_size_mm()
    assert item.kind == kind
    assert controller.side().get(item.id) is item
    assert 0 <= item.x and item.x + item.w <= width
    assert 0 <= item.y and item.y + item.h <= height
    assert item.center() == pytest.approx((width / 2, height / 2), abs=0.02)
    assert controller.selection == [item.id]
    assert controller.is_modified
    assert controller.problems() == [] or kind == "image"


def test_add_item_defaults(controller):
    text = controller.add_item("text")
    assert text.text == "글자를 입력하세요" and text.font_size_pt == 14.0
    barcode = controller.add_item("barcode")
    assert barcode.data and barcode.symbology == "code128"
    custom = controller.add_item("text", text="안녕", font_size_pt=20, pos=(40.0, 20.0))
    assert custom.text == "안녕" and custom.center() == pytest.approx((40.0, 20.0), abs=0.01)
    clamped = controller.add_item("rect", pos=(-50.0, 500.0))
    assert clamped.x == 0.0 and clamped.y + clamped.h == pytest.approx(53.98)
    with pytest.raises(EditorError):
        controller.add_item("hologram")
    with pytest.raises(EditorError):
        controller.add_item("rect", font_size_pt=3)


def test_repeated_add_cascades(controller):
    first = controller.add_item("rect")
    second = controller.add_item("rect")
    assert (second.x, second.y) != (first.x, first.y)


def test_undo_redo_add_restores_selection(controller):
    first = controller.add_item("rect")
    second = controller.add_item("ellipse")
    controller.undo_stack.undo()
    assert controller.side().get(second.id) is None
    assert controller.selection == [first.id]
    controller.undo_stack.redo()
    assert controller.side().get(second.id) is second
    assert controller.selection == [second.id]


def test_delete_and_undo_restore_order_and_selection(controller):
    a, b, c = (controller.add_item("rect") for _ in range(3))
    controller.select([b.id])
    assert controller.delete()
    assert [layer.id for layer in controller.layers()] == [c.id, a.id]
    assert controller.selection == []
    controller.undo_stack.undo()
    assert [layer.id for layer in controller.layers()] == [c.id, b.id, a.id]
    assert controller.selection == [b.id]
    assert [item.z for item in controller.side().sorted_items()] == [0, 1, 2]


def test_set_property_and_undo(controller):
    item = controller.add_item("text")
    assert controller.set_property([item.id], "font_size_pt", 22)
    assert item.font_size_pt == 22.0 and isinstance(item.font_size_pt, float)
    assert not controller.set_property([item.id], "font_size_pt", 22)
    assert not controller.set_property([item.id], "no_such_attr", 1)
    controller.undo_stack.undo()
    assert item.font_size_pt == 14.0


def test_set_property_merges_with_same_key(controller):
    item = controller.add_item("rect")
    count = controller.undo_stack.count()
    for width in (12.0, 15.0, 19.5):
        controller.set_property([item.id], "w", width, merge_key="width-slider")
    assert controller.undo_stack.count() == count + 1
    controller.set_property([item.id], "w", 25.0)
    assert controller.undo_stack.count() == count + 2
    controller.undo_stack.undo()
    controller.undo_stack.undo()
    assert item.w == 30.0


def test_merged_change_back_to_start_leaves_no_undo_step(controller):
    item = controller.add_item("rect")
    count = controller.undo_stack.count()
    controller.set_property([item.id], "w", 40.0, merge_key="k")
    controller.set_property([item.id], "w", 30.0, merge_key="k")
    assert controller.undo_stack.count() == count
    assert item.w == 30.0


def test_set_source_and_set_text(controller):
    item = controller.add_item("text")
    assert controller.set_text(item.id, "홍길동") and item.text == "홍길동"
    controller.set_source([item.id], FixedSource(value="고정"))
    assert controller.set_text(item.id, "바뀜") and item.source.value == "바뀜"
    controller.set_source([item.id], ColumnSource(column="이름"))
    assert not controller.set_text(item.id, "x")
    assert controller.layers()[0].bound
    controller.undo_stack.undo()
    assert isinstance(item.source, FixedSource)


def test_align_single_item_uses_card(controller):
    item = _rect(controller, 12.0, 7.0)
    assert controller.align("left")
    assert item.x == pytest.approx(0.0)
    assert controller.align("middle")
    assert item.center()[1] == pytest.approx(53.98 / 2)
    controller.undo_stack.undo()
    controller.undo_stack.undo()
    assert (item.x, item.y) == pytest.approx((12.0, 7.0))


def test_align_and_distribute_selection(controller):
    a = _rect(controller, 5.0, 5.0)
    b = _rect(controller, 22.0, 12.0)
    c = _rect(controller, 60.0, 30.0)
    controller.select([a.id, b.id, c.id])
    assert controller.align("top")
    assert [a.y, b.y, c.y] == pytest.approx([5.0, 5.0, 5.0])
    assert controller.distribute("horizontal")
    assert b.x == pytest.approx(32.5)
    assert controller.selection == [a.id, b.id, c.id]
    controller.undo_stack.undo()
    assert b.x == pytest.approx(22.0)
    controller.select([a.id, b.id])
    assert not controller.distribute("horizontal")
    with pytest.raises(EditorError):
        controller.align("diagonal")


def test_z_order_commands(controller):
    a, b, c = (controller.add_item("rect") for _ in range(3))
    top_down = lambda: [layer.id for layer in controller.layers()]
    assert top_down() == [c.id, b.id, a.id]
    controller.send_to_back([c.id])
    assert top_down() == [b.id, a.id, c.id]
    controller.bring_forward([c.id])
    assert top_down() == [b.id, c.id, a.id]
    controller.bring_to_front([a.id])
    assert top_down() == [a.id, b.id, c.id]
    controller.send_backward([a.id])
    assert top_down() == [b.id, a.id, c.id]
    assert not controller.bring_to_front([b.id])
    controller.set_layer_order([c.id, b.id, a.id])
    assert top_down() == [c.id, b.id, a.id]
    assert [it.z for it in controller.side().sorted_items()] == [0, 1, 2]
    for _ in range(5):
        controller.undo_stack.undo()
    assert top_down() == [c.id, b.id, a.id]
    assert controller.scene.canvas_item(a.id).zValue() == 0


def test_duplicate_sits_above_original(controller):
    a = controller.add_item("rect")
    b = controller.add_item("ellipse")
    controller.set_locked([a.id], True)
    clones = controller.duplicate([a.id])
    assert len(clones) == 1 and not clones[0].locked
    assert (clones[0].x, clones[0].y) == pytest.approx((a.x + 2, a.y + 2))
    assert [layer.id for layer in controller.layers()] == [b.id, clones[0].id, a.id]
    assert controller.selection == [clones[0].id]
    controller.undo_stack.undo()
    assert [layer.id for layer in controller.layers()] == [b.id, a.id]


def test_copy_paste_same_side_offsets_each_time(controller):
    item = _rect(controller, 10.0, 10.0)
    assert controller.copy()
    mime = QGuiApplication.clipboard().mimeData()
    assert mime.hasFormat(MIME_ITEMS)
    assert json.loads(bytes(mime.data(MIME_ITEMS)).decode("utf-8"))["items"][0]["id"] == item.id
    first = controller.paste()[0]
    second = controller.paste()[0]
    assert first.id != item.id and second.id != first.id
    assert (first.x, first.y) == pytest.approx((12.0, 12.0))
    assert (second.x, second.y) == pytest.approx((14.0, 14.0))
    assert controller.selection == [second.id]
    controller.undo_stack.undo()
    assert controller.side().get(second.id) is None


def test_paste_survives_side_switch_and_keeps_position(controller):
    text = controller.add_item("text", text="뒷면에도", pos=(30.0, 20.0))
    controller.copy()
    controller.set_side("back")
    assert controller.selection == []
    pasted = controller.paste()
    assert len(pasted) == 1
    assert controller.document.side("back").get(pasted[0].id) is pasted[0]
    assert pasted[0].text == "뒷면에도"
    assert (pasted[0].x, pasted[0].y) == pytest.approx((text.x, text.y))
    assert controller.document.side("front").items == [text]


def test_cut_then_paste_returns_to_same_place(controller):
    item = _rect(controller, 10.0, 10.0)
    assert controller.cut()
    assert controller.side().items == []
    pasted = controller.paste()[0]
    assert (pasted.x, pasted.y) == pytest.approx((10.0, 10.0))
    controller.undo_stack.undo()
    controller.undo_stack.undo()
    assert controller.side().items == [item]


def test_undo_on_other_side_switches_back(controller):
    item = controller.add_item("rect")
    controller.set_side("back")
    controller.undo_stack.undo()
    assert controller.current_side == "front"
    assert controller.side().get(item.id) is None


def test_lock_visibility_layer_helpers(controller):
    item = controller.add_item("rect")
    changes = []
    controller.items_changed.connect(lambda: changes.append(1))
    controller.set_locked([item.id], True)
    controller.set_visible([item.id], False)
    controller.set_layer([item.id], "white")
    controller.rename(item.id, "배경 상자")
    layer = controller.layers()[0]
    assert (layer.locked, layer.visible, layer.layer, layer.name) == (True, False, "white", "배경 상자")
    assert len(changes) == 4
    with pytest.raises(EditorError):
        controller.set_layer([item.id], "gold")


def test_locked_items_cannot_be_deleted_or_nudged(controller):
    item = controller.add_item("rect")
    x = item.x
    controller.set_locked([item.id], True)
    messages = []
    controller.message.connect(messages.append)
    assert not controller.delete([item.id])
    assert not controller.nudge(1.0, 0.0, [item.id])
    assert item.x == x and controller.side().get(item.id) is item
    assert messages and "잠긴" in messages[0]


def test_selection_signal_and_layer_api_can_select_locked(controller):
    item = controller.add_item("rect")
    controller.set_locked([item.id], True)
    controller.clear_selection()
    seen = []
    controller.selection_changed.connect(seen.append)
    controller.select([item.id, "missing"])
    assert controller.selection == [item.id]
    assert seen == [[item.id]]
    controller.select_all()
    assert controller.selection == []


def test_add_bound_field_kinds(controller):
    name = controller.add_bound_field("이름", "text", pos=(30.0, 20.0))
    assert isinstance(name.source, ColumnSource) and name.source.column == "이름"
    assert name.name == "이름" and name.auto_shrink
    assert name.center() == pytest.approx((30.0, 20.0), abs=0.01)
    photo = controller.add_bound_field("사진", "image")
    code = controller.add_bound_field("사번", "barcode")
    qr = controller.add_bound_field("사번", "qr")
    assert [photo.kind, code.kind, qr.kind] == ["image", "barcode", "qr"]
    assert all(layer.bound for layer in controller.layers())
    with pytest.raises(EditorError):
        controller.add_bound_field("이름", "rect")


def test_add_image_from_file_imports_and_fits(controller, make_png, tmp_path):
    path = make_png(size=(400, 100))
    item = controller.add_image_from_file(path)
    assert item.asset.startswith("assets/")
    assert os.path.isfile(os.path.join(controller.document.base_dir, item.asset))
    assert item.w / item.h == pytest.approx(4.0, rel=0.01)
    assert item.w <= 85.6 * 0.6 + 0.01 and item.h <= 53.98 * 0.6 + 0.01
    assert controller.problems() == []
    with pytest.raises(EditorError):
        controller.add_image_from_file(str(tmp_path / "nope.png"))


def test_problems_outside_card_missing_asset_and_barcode(controller):
    outside = _rect(controller, 80.0, 50.0, name="삐져나온 상자")
    gone = _rect(controller, 200.0, 5.0)
    image = controller.add_item("image", asset="assets/missing.png")
    barcode = controller.add_item("barcode", symbology="ean13", data="12ab")
    hidden = _rect(controller, -5.0, -5.0)
    controller.set_visible([hidden.id], False)
    by_item = {(p.item_id, p.code): p for p in controller.problems()}
    assert (outside.id, "outside_card") in by_item
    assert (gone.id, "off_card") in by_item
    assert (image.id, "missing_asset") in by_item
    assert (barcode.id, "invalid_barcode") in by_item
    assert "EAN-13" in by_item[(barcode.id, "invalid_barcode")].message
    assert by_item[(outside.id, "outside_card")].text.startswith("삐져나온 상자: ")
    assert not any(p.item_id == hidden.id for p in controller.problems())
    assert set(controller.warnings) == {outside.id, gone.id}
    assert controller.item_warnings(outside.id) == ["카드 밖으로 나간 부분은 인쇄되지 않습니다."]
    flagged = {layer.id for layer in controller.layers() if layer.warning}
    assert flagged == {outside.id, gone.id}


def test_cut_line_warning_only_when_edge_stops_inside_bleed(controller):
    controller.new_document(presets.CUT_54X86_BLEED)
    full_bleed = _rect(controller, 0.0, 0.0, w=90.0, h=12.0)
    crossing = _rect(controller, 1.0, 20.0, w=20.0, h=10.0)
    safe = _rect(controller, 2.0, 35.0, w=20.0, h=10.0)
    assert set(controller.warnings) == {crossing.id}
    assert controller.warnings[crossing.id][0][0] == "cut_line"
    assert safe.id not in controller.warnings and full_bleed.id not in controller.warnings


def test_problems_signal_fires_when_item_leaves_card(controller):
    item = _rect(controller, 10.0, 10.0)
    fired = []
    controller.problems_changed.connect(lambda: fired.append(1))
    controller.move_items_to({item.id: (11.0, 10.0)})
    assert fired == []
    controller.move_items_to({item.id: (-3.0, 10.0)})
    assert fired == [1]


def test_preview_record_changes_values_and_problems(controller):
    controller.new_document(template="employee")
    name = next(it for it in controller.side().items if it.name == "이름")
    assert controller.scene.value_of(name)[0] == "«이름»"
    assert controller.problems() == []
    controller.set_preview_record({"이름": "김하늘", "사번": "한글사번"})
    assert controller.scene.value_of(name)[0] == "김하늘"
    codes = {p.code for p in controller.problems()}
    assert codes == {"empty_image", "invalid_barcode"}
    controller.set_preview_record(None)
    assert controller.scene.value_of(name)[0] == "«이름»"


def test_export_preview_image_size_and_content(controller, qimage_array):
    controller.add_item("rect", w=20.0, h=20.0, fill="#ff0000", pos=(42.8, 26.99))
    image = controller.export_preview_image(dpi=100)
    assert isinstance(image, QImage)
    assert (image.width(), image.height()) == (337, 213)
    pixels = qimage_array(image)
    assert tuple(pixels[106, 168]) == (255, 0, 0)
    assert tuple(pixels[5, 5]) == (255, 255, 255)


def test_side_background_and_card_setup_are_undoable(controller, make_png):
    assert controller.set_side_background("#102030")
    assert controller.side().background == "#102030"
    assert controller.set_side_background_image(make_png())
    assert controller.side().background.startswith("assets/")
    controller.undo_stack.undo()
    controller.undo_stack.undo()
    assert controller.side().background is None

    fired = []
    controller.card_changed.connect(lambda: fired.append(1))
    assert controller.set_card_setup(preset="cut_54x86_bleed", dual_side=True, name="새 이름")
    document = controller.document
    assert (document.card_width_mm, document.bleed_mm, document.dual_side, document.name) == (90.0, 2.0, True, "새 이름")
    assert controller.set_orientation("portrait")
    assert controller.side_size_mm() == (58.0, 90.0)
    assert controller.scene.card_rect().height() == 90.0
    assert not controller.set_orientation("portrait")
    controller.undo_stack.undo()
    controller.undo_stack.undo()
    assert (document.card_width_mm, document.bleed_mm, document.dual_side, document.name) == (85.6, 0.0, False, "새 디자인")
    assert controller.side_size_mm() == (85.6, 53.98)
    assert len(fired) == 4


def test_save_open_round_trip_is_identical(controller, tmp_path, make_png):
    controller.new_document("cut_54x86_bleed", "employee", dual_side=True)
    controller.add_item("text", text="추가 글자")
    controller.add_image_from_file(make_png())
    rect = controller.add_item("rect")
    controller.set_rotation([rect.id], 33.0)
    controller.set_side("back")
    controller.add_bound_field("사번", "qr")
    controller.set_side_background("#eeeeee")
    assert controller.is_modified
    modified = []
    controller.modified_changed.connect(modified.append)

    path = tmp_path / "card.hsd"
    controller.save(path)
    assert controller.path == str(path)
    assert not controller.is_modified and modified == [False]
    saved = controller.document
    snapshot = saved.to_dict()

    other = EditorController()
    try:
        opened = other.open(path)
        assert opened == saved
        assert opened.to_dict() == snapshot
        assert other.path == str(path) and not other.is_modified
        assert len(other.layers("front")) == len(controller.layers("front"))
        again = tmp_path / "again.hsd"
        other.save(again)
        assert storage.load(again) == saved
    finally:
        other.close()


def test_save_without_path_raises(controller):
    with pytest.raises(EditorError):
        controller.save()


def test_set_document_with_existing_model(controller):
    document = presets.new_document()
    document.side("front").add_item(TextItem(text="밖에서 만든 문서"))
    document.side("front").add_item(BarcodeItem(data="ABC"))
    controller.set_document(document)
    assert controller.document is document
    assert [layer.kind for layer in controller.layers()] == ["barcode", "text"]
    assert all(controller.scene.canvas_item(item.id) for item in document.side("front").items)


def test_refresh_after_direct_model_edit(controller):
    item = controller.add_item("rect")
    controller.side().add_item(RectItem(name="직접 추가", x=1, y=1))
    item.x = 3.0
    controller.refresh()
    assert controller.scene.canvas_item(item.id).pos().x() == 3.0
    assert controller.layers()[0].name == "직접 추가"
