import os

import pytest
from pages_design_helpers import *  # noqa: F401,F403

from design import storage


@pytest.fixture
def editor(page):
    page.start_panel.size_control.set_current("cut_54x86_bleed")
    page.start_panel.choose("employee")
    return page


@pytest.fixture
def shell(qapp, settings, workspace):
    from PySide6.QtWidgets import QWidget

    from ui.pages.design import DesignPage
    from ui.shell import PAGE_DESIGN, AppShell

    window = AppShell(settings=settings, legacy_factory=QWidget)
    design = DesignPage(workspace, background_engine=FakeEngine())
    design.dialogs = FakeDialogs()
    window.register_page(PAGE_DESIGN, design, "디자인", "design", step_label="카드 꾸미기")
    window.design = design
    window.resize(1280, 800)
    window.show()
    pump(qapp)
    yield window
    design.dialogs.unsaved_answer = "discard"
    window.close()


# ---------------------------------------------------------------- 명단으로 미리 보기

def test_preview_bar_without_list(editor):
    bar = editor.preview_bar
    assert not bar.prev_button.isVisible() and not bar.next_button.isVisible()
    assert bar.empty_label.isVisible() and "명단을 넣으면" in bar.empty_label.text()
    assert not bar.photo_chip.isVisible() and not bar.column_chip.isVisible()


def test_record_navigation_updates_canvas_preview(editor, workspace, people_store, qapp):
    workspace.set_store(people_store)
    bar, controller = editor.preview_bar, editor.controller
    ids = people_store.ids()
    assert bar.record_ids() == ids
    assert bar.position_label.text() == "4명"
    assert controller.preview_record is None
    assert not bar.prev_button.isEnabled() and bar.next_button.isEnabled()

    bar.next_button.click()
    assert workspace.current_record_id == ids[0]
    assert controller.preview_record["이름"] == "강민서"
    assert bar.position_label.text() == "1 / 4  강민서"
    assert not bar.prev_button.isEnabled()

    bar.next_button.click()
    bar.next_button.click()
    assert bar.position_label.text() == "3 / 4  김하늘"
    assert controller.preview_record["부서"] == "경영지원팀"
    assert controller.preview_record["사진"].endswith("HN260417.png")
    name_item = find_item(controller, "이름")
    value, _crop = controller.scene.value_of(name_item)
    assert value == "김하늘"                                  # 캔버스가 그 사람의 내용으로 그린다

    bar.next_button.click()
    bar.next_button.click()                                    # 끝에서 멈춘다
    assert workspace.current_record_id == ids[-1] and not bar.next_button.isEnabled()
    bar.prev_button.click()
    assert workspace.current_record_id == ids[-2]

    bar.placeholder_button.click()
    assert workspace.current_record_id is None and controller.preview_record is None
    assert controller.scene.value_of(name_item)[0] == "«이름»"


def test_preview_follows_list_changes(editor, workspace, people_store):
    workspace.set_store(people_store)
    bar = editor.preview_bar
    people_store.insert({"이름": "새사람", "사번": "HN999999"})
    workspace.notify_records_changed()
    assert len(bar.record_ids()) == 5
    workspace.set_store(None)
    assert bar.record_ids() == [] and bar.empty_label.isVisible()


def test_warning_chips_for_missing_photos_and_columns(editor, workspace, people_store):
    workspace.set_store(people_store)
    bar = editor.preview_bar
    bar.refresh_warnings()
    assert bar.photo_chip.isVisible() and bar.photo_chip.text() == "사진 없는 사람 1명"
    assert not bar.column_chip.isVisible()

    editor.add_field("학번")
    bar.refresh_warnings()
    assert bar.column_chip.isVisible() and "학번" in bar.column_chip.text()

    photo = find_item(editor.controller, "사진")
    editor.controller.delete([photo.id])
    bar.refresh_warnings()
    assert not bar.photo_chip.isVisible()                       # 사진 칸이 없으면 알릴 필요가 없다


def test_thumbnail_strip_renders_only_visible_people(qapp, workspace, tmp_path, theme_manager):
    from data.store import DataStore
    from ui.pages.design import DesignPage

    store = DataStore(str(tmp_path / "many.db"))
    store.add_column("이름")
    store.insert_many({"이름": f"사람{index:03d}"} for index in range(300))
    theme_manager.set_mode("dark", persist=False)
    page = DesignPage(workspace, background_engine=FakeEngine())
    page.dialogs = FakeDialogs()
    page.resize(1196, 700)
    page.show()
    page.start_panel.choose("employee")
    workspace.set_store(store)
    strip = page.preview_bar.strip
    assert strip.isVisible()
    for _ in range(60):
        pump(qapp)
    rendered = strip.rendered_count()
    assert 1 <= rendered <= 12                                   # 화면에 보이는 몇 장만 그린다

    workspace.set_current_record(store.ids()[200])
    for _ in range(60):
        pump(qapp)
    assert strip.rendered_count() > rendered
    assert strip.rendered_count() < 40

    theme_manager.set_mode("light", persist=False)
    assert not strip.isVisible()                                 # 밝은 화면의 보통 너비에서는 숨는다
    page.close()


# ---------------------------------------------------------------- 확인할 것

def test_problem_list_selects_item(editor):
    controller = editor.controller
    card = editor.inspector.card
    assert card.problems() == [] and card.problem_rows == []

    name = find_item(controller, "이름")
    controller.set_properties(name.id, {"x": -200.0})
    controller.clear_selection()
    assert len(card.problem_rows) == 1
    assert "이름" in card.problem_rows[0].text() and "카드 밖" in card.problem_rows[0].text()
    assert "확인할 것이 1개" in editor.footer_hint()

    card.problem_rows[0].clicked.emit()
    assert controller.selection == [name.id]
    assert editor.inspector.current_section() == "text"

    editor.undo()
    controller.clear_selection()
    assert card.problem_rows == [] and "명단을 넣습니다" in editor.footer_hint()


def test_problem_on_back_side_switches_side(editor):
    controller = editor.controller
    editor.add_back_side()
    code = editor.add("barcode")
    controller.set_properties(code.id, {"data": "한글"})
    controller.set_side("front")
    card = editor.inspector.card
    problems = card.problems()
    assert [problem.side for problem in problems] == ["back"]
    assert card.problem_rows[0].text().startswith("뒷면")

    card.problem_rows[0].clicked.emit()
    assert controller.current_side == "back" and controller.selection == [code.id]
    assert editor.canvas_bar.side_control.current() == "back"


# ---------------------------------------------------------------- 테마

def test_theme_switch_updates_canvas_palette(editor, theme_manager):
    light = editor.view.palette_colors
    assert light.name == "light"
    assert light.selection.upper() == theme_manager.theme.hex("accent").upper()
    assert light.canvas_bg.upper() == theme_manager.theme.hex("ground").upper()

    theme_manager.set_mode("dark", persist=False)
    dark = editor.view.palette_colors
    assert dark.name == "dark" and dark != light
    assert dark.canvas_bg.upper() == theme_manager.theme.hex("ground").upper()
    assert dark.selection.upper() == theme_manager.theme.hex("accent").upper()
    assert dark.cut_line.upper() == theme_manager.theme.hex("cutline").upper()
    assert theme_manager.theme.hex("surface") in editor.styleSheet()

    theme_manager.set_mode("light", persist=False)
    assert editor.view.palette_colors == light


# ---------------------------------------------------------------- 파일

def test_save_asks_for_name_once_then_saves_in_place(editor, tmp_path):
    target = str(tmp_path / "card.hsd")
    editor.add("text")
    assert editor.document_state() == ("사원증", False)

    editor.dialogs.save_path = None
    assert editor.save() is False                               # 창을 닫으면 저장하지 않는다
    editor.dialogs.save_path = target
    assert editor.save() is True
    assert os.path.isfile(target) and editor.controller.path == target
    assert editor.document_state() == ("card.hsd", True)
    assert "저장했습니다" in editor.last_message

    editor.add("rect")
    assert editor.document_state() == ("card.hsd", False)
    editor.dialogs.save_path = None                             # 다시 묻지 않는다
    editor.save_action.trigger()
    assert editor.document_state() == ("card.hsd", True)
    assert len(storage.load(target).side("front").items) == len(editor.controller.side().items)


def test_save_as_writes_new_file(editor, tmp_path):
    first, second = str(tmp_path / "a.hsd"), str(tmp_path / "b.hsd")
    editor.dialogs.save_path = first
    assert editor.save()
    editor.dialogs.save_path = second
    editor.save_as_action.trigger()
    assert os.path.isfile(second) and editor.controller.path == second


def test_export_preview_image_uses_previewed_person(editor, workspace, people_store, tmp_path):
    from PySide6.QtGui import QImage

    workspace.set_store(people_store)
    editor.preview_bar.step(1)
    target = str(tmp_path / "preview.png")
    editor.dialogs.export_path = target
    editor.export_action.trigger()
    image = QImage(target)
    assert not image.isNull()
    assert image.width() == round(90 / 25.4 * 300) and image.height() == round(58 / 25.4 * 300)
    assert "내보냈습니다" in editor.last_message


def test_file_shortcuts_are_registered(editor):
    shortcuts = {action.shortcut().toString(): action for action in editor.actions()}
    assert set(shortcuts) >= {"Ctrl+N", "Ctrl+O", "Ctrl+S", "Ctrl+Shift+S", "Ctrl+E"}
    assert shortcuts["Ctrl+S"] is editor.save_action


def test_can_close_asks_only_when_modified(editor, tmp_path):
    dialogs = editor.dialogs
    assert editor.can_close() is True and dialogs.unsaved_asked == 0

    editor.add("text")
    dialogs.unsaved_answer = "cancel"
    assert editor.can_close() is False
    dialogs.unsaved_answer = "discard"
    assert editor.can_close() is True
    dialogs.unsaved_answer = "save"
    dialogs.save_path = None
    assert editor.can_close() is False                          # 저장 창을 닫으면 닫지 않는다
    dialogs.save_path = str(tmp_path / "kept.hsd")
    assert editor.can_close() is True
    assert os.path.isfile(dialogs.save_path) and not editor.controller.is_modified
    assert dialogs.unsaved_asked == 4


# ---------------------------------------------------------------- 캔버스에서 온 신호

def test_messages_from_controller_are_shown(editor):
    editor.controller.message.emit("간격을 맞추려면 3개 이상 선택하세요.")
    assert editor.last_message == "간격을 맞추려면 3개 이상 선택하세요."


def test_double_click_on_bound_text_focuses_content_control(editor, qapp):
    name = find_item(editor.controller, "이름")
    editor.controller.bound_edit_requested.emit(name.id)
    assert editor.controller.selection == [name.id]
    assert editor.inspector.current_section() == "text"
    assert editor.inspector.text.mode_control.current() == "column"


def test_double_click_on_empty_image_opens_picker(editor, tmp_path):
    frame = editor.add("image")
    editor.dialogs.image_path = make_image(tmp_path / "pic.png")
    editor.controller.item_activated.emit(frame.id)
    assert frame.asset


# ---------------------------------------------------------------- 틀과 잇기

def test_shell_registration_wires_document_name_and_footer(shell, qapp):
    design = shell.design
    assert design.shell is shell
    assert shell.current_page_id() == "design"
    assert shell.top_bar.document_name() == "새 카드"
    assert [button.text() for button in shell.footer_buttons()] == ["다음: 명단 넣기"]
    assert "시작할 모양" in shell.footer_hint_text()
    assert not shell.top_bar.undo_button.isEnabled()

    design.start_panel.choose("employee")
    pump(qapp)
    assert shell.top_bar.document_name() == "사원증"
    assert shell.top_bar.document_state_text() == ""
    assert "명단을 넣습니다" in shell.footer_hint_text()

    design.add("text")
    assert shell.top_bar.document_state_text() == "저장 안 한 변경이 있습니다"
    assert shell.top_bar.undo_button.isEnabled() and not shell.top_bar.redo_button.isEnabled()


def test_shell_undo_redo_buttons_drive_editor(shell):
    design = shell.design
    design.start_panel.choose("blank")
    design.add("rect")
    assert len(design.controller.side().items) == 1

    shell.top_bar.undo_button.click()
    assert design.controller.side().items == []
    assert not shell.top_bar.undo_button.isEnabled() and shell.top_bar.redo_button.isEnabled()
    shell.top_bar.redo_button.click()
    assert len(design.controller.side().items) == 1


def test_undo_buttons_are_released_when_leaving_page(shell):
    design = shell.design
    design.start_panel.choose("blank")
    design.add("rect")
    shell.navigate("list")
    assert not shell.top_bar.undo_button.isEnabled()
    shell.undoRequested.emit()                                  # 다른 화면의 되돌리기는 건드리지 않는다
    assert len(design.controller.side().items) == 1
    shell.navigate("design")
    assert shell.top_bar.undo_button.isEnabled()


def test_footer_action_and_links_navigate_to_list(shell):
    design = shell.design
    shell.footer_buttons()[0].click()
    assert shell.current_page_id() == "list"
    shell.navigate("design")
    design.start_panel.choose("blank")
    design.insert_panel.list_link.click()
    assert shell.current_page_id() == "list"


def test_closing_shell_asks_about_unsaved_design(shell, qapp):
    design = shell.design
    design.start_panel.choose("blank")
    design.add("text")
    design.dialogs.unsaved_answer = "cancel"
    shell.close()
    pump(qapp)
    assert shell.isVisible() and design.dialogs.unsaved_asked == 1
    design.dialogs.unsaved_answer = "discard"
    shell.close()
    pump(qapp)
    assert not shell.isVisible()


def test_page_works_at_minimum_shell_size(shell, qapp):
    design = shell.design
    design.start_panel.choose("employee")
    shell.resize(shell.minimumSize())
    pump(qapp)
    assert design.view.width() >= 380
    bar = design.canvas_bar
    assert bar.is_compact() and not bar.zoom_in_button.isVisible() and not design.save_button.isVisible()
    assert bar.zoom_button.geometry().right() <= bar.width()
    first_right = design.file_button.geometry().right()
    assert first_right < bar.align_button.geometry().left()
    assert bar.add_back_button.width() >= bar.add_back_button.sizeHint().width()      # 글자가 잘리지 않는다

    tiles = design.insert_panel.tiles
    assert tiles["text"].geometry().bottom() < tiles["shape"].geometry().top()        # 넣기 버튼이 겹치지 않는다
    assert tiles["shape"].geometry().bottom() < tiles["barcode"].geometry().top()
    assert not design.layers_panel.hint.isVisible()                                   # 낮은 창에서는 설명을 접는다
    assert design.layers_panel.list.height() >= 3 * 34

    shell.resize(1280, 800)
    pump(qapp)
    assert design.layers_panel.hint.isVisible() and design.insert_panel.fields_hint.isVisible()
    assert not bar.is_compact() and bar.zoom_in_button.isVisible() and design.save_button.isVisible()
    assert bar.zoom_in_button.geometry().right() <= bar.width()

    design.add_back_side()                                                            # 앞면/뒷면 전환이 있어도 들어간다
    shell.resize(shell.minimumSize())
    pump(qapp)
    assert bar.side_control.width() >= bar.side_control.minimumSizeHint().width()
    assert bar.zoom_button.geometry().right() <= bar.width()
    assert bar.side_control.geometry().right() < bar.align_button.geometry().left()
