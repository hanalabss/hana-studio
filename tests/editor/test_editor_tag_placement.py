from PySide6.QtCore import QRectF
from PySide6.QtGui import QColor

from design.binding import ColumnSource, ConcatPart, ConcatSource
from ui.editor import EditorPalette
from ui.editor.tag_placement import ABOVE, BELOW, INSIDE, overlap_area, place_tag

TAG = (100.0, 20.0)
VIEW = (0.0, 0.0, 800.0, 600.0)


def _overlaps(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


# ---------------------------------------------------------------- 자리 고르기(순수 함수)

def test_tag_goes_above_when_nothing_is_there():
    where, rect = place_tag((200.0, 200.0, 500.0, 240.0), TAG, viewport=VIEW)
    assert where == ABOVE
    assert rect == (200.0, 173.0, 300.0, 193.0)


def test_tag_moves_below_when_another_item_sits_above():
    box = (200.0, 200.0, 500.0, 240.0)
    line_above = (200.0, 160.0, 500.0, 195.0)
    where, rect = place_tag(box, TAG, [line_above], viewport=VIEW)
    assert where == BELOW
    assert rect[1] == 249.0 and rect[0] == 200.0
    assert not _overlaps(rect, line_above) and not _overlaps(rect, box)


def test_tag_moves_below_when_there_is_no_room_above():
    where, rect = place_tag((200.0, 10.0, 500.0, 50.0), TAG, viewport=VIEW)
    assert where == BELOW and rect[1] >= 50.0


def test_tag_goes_inside_tall_box_when_both_sides_are_taken():
    box = (200.0, 200.0, 400.0, 400.0)
    above, below = (200.0, 150.0, 400.0, 195.0), (200.0, 405.0, 400.0, 450.0)
    where, rect = place_tag(box, TAG, [above, below], viewport=VIEW)
    assert where == INSIDE
    assert rect == (204.0, 204.0, 304.0, 224.0)


def test_short_box_never_gets_the_tag_inside():
    """글자 한 줄짜리 칸은 안쪽에 넣으면 내용이 가려지므로, 덜 가리는 바깥 자리를 쓴다."""
    box = (200.0, 200.0, 500.0, 240.0)
    above = (200.0, 160.0, 500.0, 195.0)          # 꼬리표가 통째로 얹힌다
    below = (150.0, 245.0, 260.0, 285.0)          # 절반쯤만 걸친다
    where, rect = place_tag(box, TAG, [above, below], viewport=VIEW)
    assert where == BELOW
    assert overlap_area(rect, below) < overlap_area((200.0, 173.0, 300.0, 193.0), above)


def test_large_backdrop_does_not_push_the_tag_away():
    """선택한 항목 밑에 깔린 배경은 어디에 두든 가려지므로 걸림돌로 치지 않는다."""
    box = (200.0, 200.0, 500.0, 240.0)
    backdrop = (50.0, 50.0, 750.0, 550.0)
    assert place_tag(box, TAG, [backdrop], viewport=VIEW)[0] == ABOVE
    band = (0.0, 150.0, 800.0, 199.0)              # 선택한 항목을 품지 않은 위쪽 띠는 걸림돌이다
    assert place_tag(box, TAG, [backdrop, band], viewport=VIEW)[0] == BELOW


def test_rotate_handle_area_is_never_covered():
    box = (200.0, 200.0, 340.0, 240.0)
    handle = (262.0, 166.0, 278.0, 204.0)          # 위쪽 가운데의 회전 손잡이와 줄기
    where, rect = place_tag(box, TAG, viewport=VIEW, keep_clear=[handle])
    assert where == BELOW and not _overlaps(rect, handle)

    wide = (200.0, 200.0, 600.0, 240.0)
    handle = (392.0, 166.0, 408.0, 204.0)
    assert place_tag(wide, TAG, viewport=VIEW, keep_clear=[handle])[0] == ABOVE


def test_tag_is_pulled_back_into_view_sideways():
    where, rect = place_tag((760.0, 200.0, 790.0, 240.0), TAG, viewport=VIEW)
    assert where == ABOVE and rect[2] == 800.0 and rect[0] == 700.0
    where, rect = place_tag((-40.0, 200.0, 30.0, 240.0), TAG, viewport=VIEW)
    assert rect[0] == 0.0


def test_lines_are_not_given_an_inside_tag_and_thin_obstacles_count():
    thin = (200.0, 185.0, 500.0, 185.0)             # 굵기가 없는 선도 가려지면 걸림돌이다
    assert overlap_area((200.0, 173.0, 300.0, 193.0), thin) == 200.0
    where, _ = place_tag((200.0, 200.0, 500.0, 200.0), TAG, [thin], viewport=VIEW, allow_inside=False)
    assert where == BELOW


def test_everything_blocked_still_returns_the_least_bad_spot():
    box = (0.0, 0.0, 800.0, 30.0)                   # 위는 화면 밖, 아래는 다른 항목
    below = (0.0, 35.0, 200.0, 65.0)
    where, rect = place_tag(box, TAG, [below], viewport=VIEW)
    assert where == BELOW and rect[3] <= VIEW[3]


# ---------------------------------------------------------------- 캔버스에서

def _bound_text(controller, column, x, y):
    item = controller.add_bound_field(column, "text")
    controller.set_properties(item.id, {"x": x, "y": y, "w": 46.0, "h": 8.0})
    return item


def _view_rect(view, item) -> QRectF:
    left, top, right, bottom = item.bounds()
    return QRectF(view.mm_to_viewport(left, top), view.mm_to_viewport(right, bottom))


def test_selected_bound_item_tag_avoids_the_item_above(editor):
    controller, view = editor.controller, editor.view
    upper = _bound_text(controller, "부서", 20.0, 20.0)
    lower = _bound_text(controller, "직급", 20.0, 29.0)      # 바로 아래 줄
    editor.set_px_per_mm(8.0)

    controller.select([lower.id])
    text, where, rect = view.selection_tag()
    assert text == "명단: 직급"
    assert where == "below"
    assert not rect.intersects(_view_rect(view, upper))
    assert rect.top() > _view_rect(view, lower).bottom()

    controller.select([upper.id])                             # 위가 비어 있으면 원래대로 위에
    assert view.selection_tag()[1] == "above"
    assert not view.selection_tag()[2].intersects(_view_rect(view, lower))


def test_tag_lists_every_column_and_goes_inside_a_tall_photo(editor):
    controller, view = editor.controller, editor.view
    title = _bound_text(controller, "이름", 20.0, 2.0)
    photo = controller.add_bound_field("사진", "image")
    controller.set_properties(photo.id, {"x": 20.0, "y": 11.0, "w": 24.0, "h": 30.0})
    caption = _bound_text(controller, "부서", 20.0, 42.0)
    controller.set_source(caption.id, ConcatSource(parts=[
        ConcatPart(column="부서"), ConcatPart(text=" "), ConcatPart(column="직급"),
    ]))
    editor.set_px_per_mm(8.0)

    controller.select([photo.id])
    text, where, rect = view.selection_tag()
    assert (text, where) == ("명단: 사진", "inside")
    assert _view_rect(view, photo).contains(rect)
    assert not rect.intersects(_view_rect(view, title)) and not rect.intersects(_view_rect(view, caption))

    controller.select([caption.id])
    assert view.selection_tag()[0] == "명단: 부서, 직급"


def test_tag_is_painted_where_the_layout_says(editor):
    controller, view = editor.controller, editor.view
    _bound_text(controller, "부서", 20.0, 20.0)
    lower = _bound_text(controller, "직급", 20.0, 29.0)
    editor.set_px_per_mm(8.0)
    controller.select([lower.id])
    editor.app.processEvents()
    _, _, rect = view.selection_tag()
    image = view.viewport().grab().toImage()
    probe = image.pixelColor(round(rect.left()) + 3, round(rect.center().y()))
    assert probe.name() == view.palette_colors.tag_bg.lower()

    for palette in (EditorPalette.light(), EditorPalette.dark()):   # 두 테마 모두 바탕과 글자가 뚜렷이 갈린다
        background, foreground = QColor(palette.tag_bg), QColor(palette.tag_text)
        assert abs(background.lightnessF() - foreground.lightnessF()) > 0.35, palette.name


def test_no_tag_for_plain_items_and_locked_items_say_so(editor):
    controller, view = editor.controller, editor.view
    plain = controller.add_item("rect")
    controller.select([plain.id])
    assert view.selection_tag() is None
    controller.set_locked([plain.id], True)
    controller.select([plain.id])
    assert view.selection_tag()[0] == "잠김"
    other = controller.add_item("text", source=ColumnSource(column="이름"))
    controller.select([plain.id, other.id])
    assert view.selection_tag() is None
