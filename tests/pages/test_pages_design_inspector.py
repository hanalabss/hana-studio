import pytest
from pages_design_helpers import *  # noqa: F401,F403

from design.binding import ColumnSource, ConcatPart, ConcatSource, CounterSource, DateSource, FixedSource


@pytest.fixture
def editor(page):
    page.start_panel.choose("blank")
    return page


def _steps(page) -> int:
    return page.controller.undo_stack.index()


# ---------------------------------------------------------------- 어떤 구역이 보이는가

def test_section_follows_selection(editor):
    inspector, controller = editor.inspector, editor.controller
    assert inspector.current_section() == "card"
    assert not inspector.detail.isVisible() and not inspector.delete_button.isVisible()

    expected = {"text": "text", "image": "image", "rect": "shape", "ellipse": "shape", "line": "line",
                "barcode": "code", "qr": "code"}
    made = {}
    for kind, section in expected.items():
        made[kind] = editor.add(kind)
        assert inspector.current_section() == section
        assert inspector.sections[section].isVisible()
        assert inspector.detail.isVisible()
        assert not inspector.detail.box.is_open()      # 자세한 설정은 접혀 있다

    controller.select([made["text"].id, made["rect"].id])
    assert inspector.current_section() == "multi"
    assert inspector.title_label.text() == "2개를 골랐습니다"
    controller.clear_selection()
    assert inspector.current_section() == "card"
    assert inspector.title_label.text() == "카드 설정"


def test_header_names_the_item_and_delete_button_removes_it(editor, workspace, people_store):
    workspace.set_store(people_store)
    item = editor.add_field("이름")
    assert editor.inspector.title_label.text() == "글자"
    assert "사람마다 바뀌는 칸" in editor.inspector.subtitle_label.text()
    editor.inspector.delete_button.click()
    assert editor.controller.item(item.id) is None
    assert editor.inspector.current_section() == "card"


# ---------------------------------------------------------------- 글자

def test_text_typing_is_one_undo_step(editor):
    item = editor.add("text")
    section = editor.inspector.text
    before = _steps(editor)
    for text in ("하", "하나", "하나스튜디오"):
        section.text_edit.setPlainText(text)
    assert item.text == "하나스튜디오" and item.source is None
    assert _steps(editor) == before + 1
    editor.undo()
    assert item.text == "글자를 입력하세요"
    assert section.text_edit.toPlainText() == "글자를 입력하세요"


def test_text_size_stepper_merges_and_new_gesture_starts_new_step(editor):
    item = editor.add("text")
    section = editor.inspector.text
    before = _steps(editor)
    for _ in range(5):
        section.size_field.step_by(1)
    assert item.font_size_pt == 19.0
    assert _steps(editor) == before + 1

    editor.session.close()                      # 잠깐 쉬었다가 다시 누른 경우
    section.size_field.step_by(-1)
    assert _steps(editor) == before + 2
    editor.undo()
    editor.undo()
    assert item.font_size_pt == 14.0
    assert section.size_field.value() == 14.0


def test_text_style_controls_edit_model(editor):
    item = editor.add("text")
    section = editor.inspector.text
    before = _steps(editor)

    section.bold_toggle.click()
    section.italic_toggle.click()
    section.underline_toggle.click()
    section.align_control.set_current("center")
    section.shrink_check.setChecked(True)
    section.color_picker.set_current(section.color_picker.colors()[3])
    family = section.font_combo.itemData(section.font_combo.count() - 1)
    pick(section.font_combo, family)
    section.v_align_control.set_current("bottom")
    section.wrap_check.setChecked(True)

    assert (item.bold, item.italic, item.underline) == (True, True, True)
    assert (item.h_align, item.v_align, item.auto_shrink, item.wrap) == ("center", "bottom", True, True)
    assert item.color.upper() == section.color_picker.colors()[3]
    assert item.font_family == family
    assert _steps(editor) == before + 9
    for _ in range(9):
        editor.undo()
    assert (item.bold, item.italic, item.underline, item.h_align) == (False, False, False, "left")
    assert not section.bold_toggle.isChecked() and section.align_control.current() == "left"


def test_text_detail_extras_live_inside_detail_section(editor):
    editor.add("text")
    detail = editor.inspector.detail
    detail.box.set_open(True)
    assert editor.inspector.text.v_align_control.isVisible()
    editor.add("rect")
    assert not editor.inspector.text.v_align_control.isVisible()


def test_text_source_switching_produces_right_sources(editor, workspace, people_store):
    workspace.set_store(people_store)
    item = editor.add("text")
    section = editor.inspector.text
    assert section.mode_control.current() == "fixed"

    section.mode_control.set_current("column")
    assert item.source == ColumnSource(column="이름")
    assert [section.column_combo.itemData(i) for i in range(section.column_combo.count())] == \
        ["이름", "부서", "직급", "사번"]                       # 사진 열은 글자 칸에 쓰지 않는다
    pick(section.column_combo, "부서")
    assert item.source == ColumnSource(column="부서")

    section.mode_control.set_current("auto")
    assert isinstance(item.source, DateSource) and item.source.format == "%Y.%m.%d"
    section.auto_control.set_current("counter")
    assert item.source == CounterSource(start=1, pad=4)

    section.mode_control.set_current("fixed")
    assert item.source is None
    assert section.text_edit.isVisible() and not section.column_combo.isVisible()

    editor.undo()
    assert isinstance(item.source, CounterSource) and section.mode_control.current() == "auto"


def test_fixed_source_text_is_kept_when_switching_to_direct_input(editor):
    item = editor.add("text")
    editor.controller.set_source(item.id, FixedSource(value="고정 문구"))
    section = editor.inspector.text
    assert section.mode_control.current() == "fixed"
    assert section.text_edit.toPlainText() == "고정 문구"
    section.text_edit.setPlainText("바꾼 문구")
    assert item.source == FixedSource(value="바꾼 문구")


def test_column_mode_without_list_uses_placeholder_column(editor):
    item = editor.add("text")
    section = editor.inspector.text
    section.mode_control.set_current("column")
    assert item.source == ColumnSource(column="이름")
    assert "명단을 넣으면" in section.column_hint.text()


def test_missing_column_is_flagged(editor, workspace, people_store):
    workspace.set_store(people_store)
    item = editor.add("text")
    editor.controller.set_source(item.id, ColumnSource(column="학번"))
    section = editor.inspector.text
    assert section.column_combo.column() == "학번"
    assert "(명단에 없음)" in section.column_combo.currentText()
    assert "'학번'" in section.column_hint.text()


def test_date_and_counter_settings_merge_into_one_step(editor):
    item = editor.add("text")
    section = editor.inspector.text
    section.mode_control.set_current("auto")
    before = _steps(editor)
    for _ in range(3):
        section.date_offset.step_by(1)
    assert item.source.offset_days == 3 and _steps(editor) == before + 1
    pick(section.date_combo, "%Y-%m-%d")
    assert item.source == DateSource(format="%Y-%m-%d", offset_days=3)

    section.auto_control.set_current("counter")
    before = _steps(editor)
    editor.session.close()
    section.counter_start.set_value(100)
    section.counter_pad.set_value(6)
    section.counter_prefix.setText("No. ")
    section.counter_prefix.textEdited.emit("No. ")
    assert item.source == CounterSource(start=100, pad=6, prefix="No. ")
    assert _steps(editor) == before + 1
    editor.undo()
    assert item.source == CounterSource(start=1, pad=4)


def test_concat_builder_makes_concat_source(editor, workspace, people_store):
    from ui.pages.design.inspector.concat import ConcatBuilder

    workspace.set_store(people_store)
    workspace.set_current_record(people_store.ids()[2])
    item = editor.add_field("부서")
    section = editor.inspector.text

    seed = section.concat_seed()
    assert seed == ConcatSource(parts=[ConcatPart(column="부서")], separator=" ")
    builder = ConcatBuilder(["이름", "부서", "직급"], seed, section.preview_text)
    builder.add_column_part("직급")
    assert builder.preview_label.text() == "경영지원팀 사원"
    builder.add_text_part("님")
    builder.separator_control.set_current("")
    assert builder.preview_label.text() == "경영지원팀사원님"
    builder.rows[2].up_button.click()
    builder.rows[0].prefix_edit.setText("[")
    builder.rows[0].suffix_edit.setText("]")
    assert builder.preview_label.text() == "[경영지원팀]님사원"
    builder.rows[1].remove_button.click()
    builder.separator_control.set_current("custom")
    builder.separator_edit.setText(" / ")

    result = builder.source()
    assert result == ConcatSource(
        parts=[ConcatPart(column="부서", prefix="[", suffix="]"), ConcatPart(column="직급")], separator=" / ")

    before = _steps(editor)
    section.apply_concat(result)
    assert item.source == result and _steps(editor) == before + 1
    assert section.concat_note.isVisible() and "부서 + 직급" in section.concat_note.text()
    assert not section.column_combo.isVisible()

    section.concat_single_button.click()
    assert item.source == ColumnSource(column="부서")
    assert section.column_combo.isVisible() and not section.concat_note.isVisible()


def test_concat_dialog_needs_at_least_one_list_field(editor):
    from ui.pages.design.inspector.concat import ConcatDialog

    dialog = ConcatDialog(["이름"], ConcatSource(separator=" "), None)
    assert not dialog.apply_button.isEnabled()
    dialog.builder.add_text_part("글자만")
    assert not dialog.apply_button.isEnabled()
    dialog.builder.add_column_part("이름")
    assert dialog.apply_button.isEnabled()
    dialog.close()


# ---------------------------------------------------------------- 사진

def test_image_fit_and_replace_are_undoable(editor, tmp_path):
    item = editor.add_image_file(make_image(tmp_path / "a.png"))
    section = editor.inspector.image
    first_asset = item.asset
    assert section.replace_button.text() == "그림 바꾸기"

    before = _steps(editor)
    section.fit_control.set_current("contain")
    assert item.fit == "contain" and _steps(editor) == before + 1
    assert "전체가 보이도록" in section.fit_hint.text()

    editor.dialogs.image_path = make_image(tmp_path / "b.png", color=(250, 10, 10))
    section.replace_button.click()
    assert item.asset and item.asset != first_asset
    assert _steps(editor) == before + 2
    editor.undo()
    assert item.asset == first_asset
    editor.undo()
    assert item.fit == "fill" and section.fit_control.current() == "fill"


def test_empty_image_frame_asks_for_a_picture(editor):
    editor.add("image")
    section = editor.inspector.image
    assert section.replace_button.text() == "그림 고르기"
    assert section.file_label.isVisible()
    assert not section.remove_button.isEnabled()


def test_image_binding_to_list_photo_column(editor, workspace, people_store):
    workspace.set_store(people_store)
    item = editor.add("image")
    section = editor.inspector.image

    section.source_control.set_current("column")
    assert item.source == ColumnSource(column="사진")
    assert section.column_box.isVisible() and not section.file_box.isVisible()
    assert [section.column_combo.itemData(i) for i in range(section.column_combo.count())] == ["사진"]
    assert not section.remove_button.isEnabled()
    assert "사람마다 사진이 달라" in section.cutout_hint.text()

    section.source_control.set_current("file")
    assert item.source is None
    editor.undo()
    assert item.source == ColumnSource(column="사진")
    assert section.source_control.current() == "column"


# ---------------------------------------------------------------- 도형 / 선

def test_shape_fill_stroke_and_radius(editor):
    item = editor.add("rect")
    section = editor.inspector.shape
    assert section.fill_check.isChecked() and not section.stroke_check.isChecked()
    assert not section.stroke_width_row.isVisible() and section.radius_row.isVisible()

    before = _steps(editor)
    section.fill_picker.set_current(section.fill_picker.colors()[3])
    assert item.fill.upper() == section.fill_picker.colors()[3]

    section.stroke_check.setChecked(True)
    assert item.stroke_color and item.stroke_width_mm == 0.3
    assert section.stroke_width_row.isVisible()
    editor.session.close()
    for _ in range(4):
        section.stroke_width.step_by(1)
    assert item.stroke_width_mm == pytest.approx(0.7)
    editor.session.close()
    for _ in range(3):
        section.radius_field.step_by(1)
    assert item.corner_radius_mm == 1.5
    assert _steps(editor) == before + 4

    editor.undo()
    assert item.corner_radius_mm == 0.0
    editor.undo()
    assert item.stroke_width_mm == pytest.approx(0.3)
    editor.undo()
    assert item.stroke_color is None and not section.stroke_check.isChecked()

    fill = item.fill
    section.fill_check.setChecked(False)
    assert item.fill is None and not section.fill_picker.isVisible()
    section.fill_check.setChecked(True)
    assert item.fill == fill


def test_ellipse_has_no_corner_radius(editor):
    editor.add("ellipse")
    assert not editor.inspector.shape.radius_row.isVisible()


def test_line_colour_width_style(editor):
    item = editor.add("line")
    section = editor.inspector.line
    before = _steps(editor)
    section.color_picker.set_current(section.color_picker.colors()[4])
    editor.session.close()
    for _ in range(5):
        section.width_field.step_by(1)
    section.style_control.set_current("dash")
    assert item.color.upper() == section.color_picker.colors()[4]
    assert item.width_mm == pytest.approx(1.0) and item.style == "dash"
    assert _steps(editor) == before + 3
    editor.undo()
    editor.undo()
    assert item.width_mm == pytest.approx(0.5) and section.width_field.value() == pytest.approx(0.5)


# ---------------------------------------------------------------- 바코드 / QR

def test_barcode_data_symbology_and_validation(editor):
    item = editor.add("barcode")
    section = editor.inspector.code
    assert section.symbology_box.isVisible() and not section.level_box.isVisible()
    assert not section.validation.isVisible()

    before = _steps(editor)
    for text in ("8", "88", "880123"):
        section.data_edit.setText(text)
        section.data_edit.textEdited.emit(text)
    assert item.data == "880123" and _steps(editor) == before + 1

    pick(section.symbology_combo, "ean13")
    assert item.symbology == "ean13"
    assert "숫자 12~13자리" in section.symbology_hint.text()
    assert section.validation.isVisible()
    assert "12자리 또는 13자리" in section.validation.text()
    assert section.validation_message() == section.validation.text()

    editor.session.close()
    section.data_edit.setText("880123456789")
    section.data_edit.textEdited.emit("880123456789")
    assert not section.validation.isVisible()

    section.show_text_check.setChecked(False)
    assert item.show_text is False
    editor.undo()
    editor.undo()
    editor.undo()
    assert item.symbology == "code128" and item.data == "880123"


def test_code39_rejects_lowercase_inline(editor):
    editor.add("barcode")
    section = editor.inspector.code
    pick(section.symbology_combo, "code39")
    section.data_edit.setText("abc")
    section.data_edit.textEdited.emit("abc")
    assert section.validation.isVisible() and "쓸 수 없는 문자" in section.validation.text()


def test_qr_level_and_binding(editor, workspace, people_store):
    workspace.set_store(people_store)
    item = editor.add("qr")
    section = editor.inspector.code
    assert section.level_box.isVisible() and not section.symbology_box.isVisible()
    assert not section.show_text_check.isVisible()

    pick(section.level_combo, "H")
    assert item.error_correction == "H"
    section.source_control.set_current("column")
    assert item.source == ColumnSource(column="이름")
    pick(section.column_combo, "사번")
    assert item.source == ColumnSource(column="사번")
    assert "예시 값" in section.data_label.text()
    section.source_control.set_current("fixed")
    assert item.source is None


def test_bound_barcode_validates_against_previewed_person(editor, workspace, people_store):
    workspace.set_store(people_store)
    item = editor.add_field("이름", "barcode")
    section = editor.inspector.code
    assert not section.validation.isVisible()          # 명단을 보지 않을 때는 예시 값으로 본다
    workspace.set_current_record(people_store.ids()[0])
    assert section.validation.isVisible()              # 한글 이름은 Code 128에 담을 수 없다
    assert item.source == ColumnSource(column="이름")


# ---------------------------------------------------------------- 여러 개 / 자세한 설정

def test_multi_selection_align_distribute_and_common_text(editor):
    controller = editor.controller
    texts = [editor.add("text") for _ in range(3)]
    for index, item in enumerate(texts):
        controller.set_properties(item.id, {"x": 5.0 + index * 6, "y": 4.0 + index * index * 9})
    controller.select([item.id for item in texts])
    section = editor.inspector.multi
    assert section.font_box.isVisible() and section.color_box.isVisible() and not section.mixed_note.isVisible()
    assert not section.distribute_hint.isVisible()

    before = _steps(editor)
    section.align_buttons["left"].click()
    assert len({item.x for item in texts}) == 1
    assert _steps(editor) == before + 1

    section.distribute_buttons["vertical"].click()
    gaps = [texts[i + 1].y - texts[i].y for i in range(2)]
    assert gaps[0] == pytest.approx(gaps[1])

    editor.session.close()
    for _ in range(3):
        section.size_field.step_by(1)
    assert [item.font_size_pt for item in texts] == [17.0, 17.0, 17.0]
    section.color_picker.set_current(section.color_picker.colors()[3])
    assert len({item.color for item in texts}) == 1
    assert _steps(editor) == before + 4
    editor.undo()
    editor.undo()
    assert [item.font_size_pt for item in texts] == [14.0, 14.0, 14.0]


def test_multi_selection_of_mixed_kinds(editor):
    rect, text = editor.add("rect"), editor.add("text")
    editor.controller.select([rect.id, text.id])
    section = editor.inspector.multi
    assert not section.font_box.isVisible() and not section.color_box.isVisible()
    assert section.mixed_note.isVisible()
    assert not section.distribute_buttons["horizontal"].isEnabled() and section.distribute_hint.isVisible()
    assert not editor.inspector.detail.geometry_host.isVisible()

    editor.inspector.detail.box.set_open(True)
    editor.inspector.detail.lock_check.setChecked(True)
    assert rect.locked and text.locked


def test_detail_position_size_rotation_layer(editor):
    item = editor.add("rect")
    detail = editor.inspector.detail
    detail.box.set_open(True)
    assert detail.x_field.text().endswith("mm") and "." in detail.x_field.text()     # mm, 소수 한 자리
    assert detail.x_field.value() == pytest.approx(round(item.x, 1))
    editor.controller.set_properties(item.id, {"x": 10.0, "y": 10.0})

    before = _steps(editor)
    for _ in range(4):
        detail.x_field.step_by(1)
    assert item.x == pytest.approx(12.0) and _steps(editor) == before + 1
    editor.session.close()
    detail.w_field.set_value(40.0)
    detail.h_field.set_value(12.5)
    editor.session.close()
    detail.rotation_field.set_value(-90)
    assert (item.w, item.h, item.rotation) == (40.0, 12.5, 270.0)
    detail.layer_control.set_current("white")
    assert item.layer == "white" and "흰색 잉크" in detail.layer_hint.text()
    detail.printable_check.setChecked(True)
    assert item.printable is False

    for _ in range(6):
        editor.undo()
    assert (item.x, item.w, item.h, item.rotation, item.layer, item.printable) == (10.0, 30.0, 18.0, 0.0, "color", True)
    assert detail.x_field.value() == 10.0 and detail.layer_control.current() == "color"


def test_inspector_follows_canvas_drag(editor):
    item = editor.add("rect")
    editor.inspector.detail.box.set_open(True)
    editor.controller.move_items_to({item.id: (22.0, 7.5)})
    assert editor.inspector.detail.x_field.value() == 22.0
    assert editor.inspector.detail.y_field.value() == 7.5


def test_selected_item_warning_is_shown_on_top(editor):
    item = editor.add("rect")
    assert not editor.inspector.warnings_host.isVisible()
    editor.controller.set_properties(item.id, {"x": -10.0})
    assert editor.inspector.warnings_host.isVisible()
    row = editor.inspector.warnings_box.itemAt(0).widget()
    assert "카드 밖" in row.text()


# ---------------------------------------------------------------- 카드 설정

def test_card_setup_controls(editor, tmp_path):
    controller = editor.controller
    section = editor.inspector.card
    document = controller.document
    assert section.size_combo.currentData() == "cr80"

    before = _steps(editor)
    pick(section.size_combo, "cut_54x86_bleed")
    assert (document.card_width_mm, document.bleed_mm) == (90.0, 2.0)
    section.orientation_control.set_current("portrait")
    assert controller.side().orientation == "portrait"
    assert controller.side_size_mm() == (58.0, 90.0)
    section.sides_control.set_current("dual")
    assert document.dual_side is True and editor.canvas_bar.side_control.isVisible()
    section.background_picker.set_current(section.background_picker.colors()[2])
    assert controller.side().background.upper() == section.background_picker.colors()[2]
    section.name_edit.setText("2026 사원증")
    section.name_edit.editingFinished.emit()
    assert document.name == "2026 사원증"
    assert _steps(editor) == before + 5

    editor.dialogs.image_path = make_image(tmp_path / "bg.png")
    section.background_image_button.click()
    assert controller.side().background.startswith("assets/")
    assert section.background_image_button.text() == "배경 그림 바꾸기"
    section.background_clear_button.click()
    assert controller.side().background is None and not section.background_clear_button.isVisible()

    for _ in range(7):
        editor.undo()
    assert (document.card_width_mm, document.dual_side, document.name) == (85.6, False, "새 디자인")
    assert section.size_combo.currentData() == "cr80" and section.sides_control.current() == "single"


def test_custom_card_size_is_listed(editor):
    editor.controller.set_card_setup(width_mm=100.0, height_mm=70.0)
    section = editor.inspector.card
    assert section.size_combo.currentData() == "custom"
    assert "70 × 100" in section.size_combo.currentText()


def test_changing_orientation_keeps_items_on_the_card_and_undoes_in_one_step(editor):
    controller = editor.controller
    section = editor.inspector.card
    name = editor.add("text")
    photo = editor.add("image")
    controller.set_properties(name.id, {"x": 50.0, "y": 44.0, "w": 34.0, "h": 8.0})    # 가로 카드의 오른쪽 아래
    controller.set_properties(photo.id, {"x": 60.0, "y": 4.0, "w": 22.0, "h": 28.0})   # 오른쪽 위
    controller.clear_selection()
    placed = {item.id: (item.x, item.y, item.w, item.h) for item in controller.side().items}
    before = _steps(editor)

    section.orientation_control.set_current("portrait")
    width, height = controller.side_size_mm()
    assert (width, height) == (53.98, 85.6)
    for item in controller.side().items:
        left, top, right, bottom = item.bounds()
        assert left >= -1e-6 and top >= -1e-6 and right <= width + 1e-6 and bottom <= height + 1e-6, item.kind
    assert not any(problem.code in ("outside_card", "off_card") for problem in controller.problems())
    assert _steps(editor) == before + 1

    editor.undo()
    assert controller.side().orientation == "landscape"
    assert section.orientation_control.current() == "landscape"
    assert {item.id: (item.x, item.y, item.w, item.h) for item in controller.side().items} == placed
    editor.redo()
    assert controller.side().orientation == "portrait" and section.orientation_control.current() == "portrait"
    assert not any(problem.code in ("outside_card", "off_card") for problem in controller.problems())


def test_changing_card_size_keeps_items_on_the_card(editor):
    controller = editor.controller
    section = editor.inspector.card
    corner = editor.add("text")
    controller.set_properties(corner.id, {"x": 55.6, "y": 45.98, "w": 30.0, "h": 8.0})  # 오른쪽 아래 구석에 딱 붙임
    controller.clear_selection()
    before = _steps(editor)

    pick(section.size_combo, "cut_54x86_bleed")                  # 90 × 58 (재단 여백 포함)
    assert (corner.x + corner.w, corner.y + corner.h) == pytest.approx((90.0, 58.0))
    assert _steps(editor) == before + 1
    editor.undo()
    assert (corner.x, corner.y) == (55.6, 45.98) and controller.document.card_width_mm == 85.6


def test_image_fill_card_button_fits_the_card_in_one_undo_step(editor, tmp_path):
    item = editor.add_image_file(make_image(tmp_path / "wide.png", size=(200, 60)))
    section = editor.inspector.image
    original = (item.x, item.y, item.w, item.h, item.fit)
    before = _steps(editor)

    section.fill_card_button.click()

    width, height = editor.controller.side_size_mm()
    assert (item.x, item.y) == (0.0, 0.0)
    assert abs(item.w - width) < 1e-6 and abs(item.h - height) < 1e-6
    assert item.fit == "fill" and _steps(editor) == before + 1

    editor.controller.undo_stack.undo()
    assert (item.x, item.y, item.w, item.h, item.fit) == original


def test_card_background_image_can_become_a_photo_for_cutout(editor, tmp_path):
    controller = editor.controller
    controller.clear_selection()
    section = editor.inspector.card
    section.set_background_image(make_image(tmp_path / "bg.png", size=(300, 200)))
    assert controller.side().background and not controller.side().background.startswith("#")
    count = len(controller.side().items)
    before = _steps(editor)

    item = section.background_to_photo()

    width, height = controller.side_size_mm()
    assert item is not None and item.kind == "image" and item.fit == "fill"
    assert (item.x, item.y) == (0.0, 0.0) and abs(item.w - width) < 1e-6 and abs(item.h - height) < 1e-6
    assert controller.side().background is None
    assert controller.selection == [item.id]
    assert controller.side().sorted_items()[0].id == item.id          # 맨 아래에 깔린다
    assert _steps(editor) == before + 1

    controller.undo_stack.undo()
    assert len(controller.side().items) == count and controller.side().background


def test_manual_mask_goes_to_the_white_layer_and_is_undoable(editor, tmp_path):
    import numpy as np
    from PIL import Image

    from design.renderer import render_side

    item = editor.add_image_file(make_image(tmp_path / "photo.png", size=(100, 100)))
    editor.controller.set_geometry(item.id, {"x": 10.0, "y": 10.0, "w": 20.0, "h": 20.0})
    section = editor.inspector.image
    # 기존 수동 마스킹 방식: 왼쪽 절반이 검정(=흰색을 깔 자리), 오른쪽 절반은 흰색
    mask = np.full((100, 100), 255, np.uint8); mask[:, :50] = 0
    path = tmp_path / "mask.png"; Image.fromarray(mask, "L").save(path)
    before = _steps(editor)

    assert section.set_manual_mask(str(path))
    assert item.mask_asset and section.has_manual_mask() and _steps(editor) == before + 1

    white = np.asarray(render_side(editor.controller.document, "front", dpi=100).white_mask)
    px = 100 / 25.4
    left = white[int(20 * px), int(13 * px)]; right = white[int(20 * px), int(27 * px)]
    assert left > 200 and right < 50                      # 검은 쪽에만 흰색이 깔린다

    assert section.set_manual_mask(str(path), dark_is_ink=False)
    white = np.asarray(render_side(editor.controller.document, "front", dpi=100).white_mask)
    assert white[int(20 * px), int(13 * px)] < 50 and white[int(20 * px), int(27 * px)] > 200

    assert section.clear_manual_mask() and item.mask_asset is None and not section.has_manual_mask()
    editor.controller.undo_stack.undo()
    assert item.mask_asset and section.has_manual_mask()


def test_fill_card_pushes_a_pictures_own_bleed_outside_the_card(editor, tmp_path):
    """58×90 비율(카드 + 사방 2mm)로 만든 시안을 딱 맞춰 넣으면 가장자리 재단선이 카드에 찍힌다(현장 피드백)."""
    controller = editor.controller
    controller.new_document("cut_54x86", None, orientation="portrait")
    item = editor.add_image_file(make_image(tmp_path / "sian.png", size=(685, 1063)))
    editor.inspector.image.fill_card_button.click()
    width, height = controller.side_size_mm()
    assert (item.x, item.y) == (-2.0, -2.0)
    assert (item.w, item.h) == pytest.approx((width + 4.0, height + 4.0))
    assert "재단 여백" in editor.inspector.image.fit_hint.text()


def test_bleed_in_image_only_when_the_ratio_says_so():
    from design.storage import bleed_in_image

    assert bleed_in_image((685, 1063), (54.0, 86.0)) == 2.0        # 58×90
    assert bleed_in_image((685, 1063), (53.98, 85.6)) == 2.0       # 표준 카드에도
    assert bleed_in_image((638, 1016), (54.0, 86.0)) == 0.0        # 카드와 같은 비율
    assert bleed_in_image((200, 60), (85.6, 53.98)) == 0.0
    assert bleed_in_image((0, 10), (54.0, 86.0)) == 0.0
