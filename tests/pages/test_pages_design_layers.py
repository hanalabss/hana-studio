import pytest
from pages_design_helpers import *  # noqa: F401,F403


@pytest.fixture
def editor(page):
    page.start_panel.choose("blank")
    for kind in ("rect", "text", "qr"):
        page.add(kind)
    page.controller.clear_selection()
    return page


def _names(panel) -> list[str]:
    return [panel.info(item_id).kind for item_id in panel.ids()]


def test_list_shows_top_item_first(editor):
    panel = editor.layers_panel
    assert _names(panel) == ["qr", "text", "rect"]
    assert panel.list.count() == 3
    assert "끌어서" in panel.hint.text()


def test_empty_list_invites_to_add(page):
    page.start_panel.choose("blank")
    assert page.layers_panel.list.count() == 0
    assert "아직 아무것도 없습니다" in page.layers_panel.hint.text()


def test_reorder_changes_z_order_and_is_one_undo_step(editor):
    panel, controller = editor.layers_panel, editor.controller
    rect = next(item for item in controller.side().items if item.kind == "rect")
    index = controller.undo_stack.index()

    panel.move_layer(rect.id, 0)

    assert _names(panel) == ["rect", "qr", "text"]
    assert [item.kind for item in controller.side().sorted_items()] == ["text", "qr", "rect"]
    assert controller.undo_stack.index() == index + 1
    editor.undo()
    assert _names(panel) == ["qr", "text", "rect"]


def test_drop_inside_list_commits_new_order(editor):
    panel, controller = editor.layers_panel, editor.controller
    entry = panel.list.takeItem(2)
    panel.list.insertItem(1, entry)
    panel.list.dropped.emit()
    assert [item.kind for item in controller.side().sorted_items()] == ["text", "rect", "qr"]


def test_lock_and_visibility_toggles(editor):
    panel, controller = editor.layers_panel, editor.controller
    text = next(item for item in controller.side().items if item.kind == "text")

    panel.toggle_locked(text.id)
    assert text.locked is True and panel.info(text.id).locked is True
    panel.toggle_visible(text.id)
    assert text.visible is False and panel.info(text.id).visible is False

    editor.undo()
    assert text.visible is True and panel.info(text.id).visible is True
    editor.undo()
    assert text.locked is False


def test_eye_and_lock_hit_areas_are_clickable(editor, qapp):
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtWidgets import QApplication

    from ui.pages.design.layers_panel import _button_rects

    panel, controller = editor.layers_panel, editor.controller
    pump(qapp)
    top = controller.side().sorted_items()[-1]
    rect = panel.list.visualItemRect(panel.list.item(0))
    lock, eye = _button_rects(rect)
    for target in (eye, lock):
        point = QPointF(target.center())
        for kind in (QEvent.MouseButtonPress, QEvent.MouseButtonRelease):
            buttons = Qt.LeftButton if kind == QEvent.MouseButtonPress else Qt.NoButton
            QApplication.sendEvent(panel.list.viewport(), QMouseEvent(
                kind, point, QPointF(panel.list.viewport().mapToGlobal(point.toPoint())),
                Qt.LeftButton, buttons, Qt.NoModifier))
        pump(qapp)
    assert top.visible is False
    assert top.locked is True
    assert controller.selection == []          # 버튼을 눌러도 선택은 바뀌지 않는다


def test_rename_through_editor_delegate(editor, qapp):
    panel, controller = editor.layers_panel, editor.controller
    text = next(item for item in controller.side().items if item.kind == "text")
    row = panel.ids().index(text.id)
    entry = panel.list.item(row)

    panel.list.editItem(entry)
    pump(qapp)
    from PySide6.QtWidgets import QLineEdit
    line_edit = panel.list.viewport().findChild(QLineEdit)
    assert line_edit is not None and line_edit.text() == panel.info(text.id).name
    line_edit.setText("회사 이름")
    panel.list.commitData(line_edit)
    panel.list.closeEditor(line_edit, panel.list.itemDelegate().EndEditHint.NoHint)
    pump(qapp)

    assert text.name == "회사 이름"
    assert panel.info(text.id).name == "회사 이름"
    editor.undo()
    pump(qapp)
    assert text.name == ""


def test_selection_is_shared_with_canvas(editor):
    panel, controller = editor.layers_panel, editor.controller
    qr, text, rect = (controller.item(item_id) for item_id in panel.ids())

    controller.select([text.id, rect.id])
    assert [panel.list.item(row).isSelected() for row in range(3)] == [False, True, True]

    panel.list.clearSelection()
    panel.list.item(0).setSelected(True)
    assert controller.selection == [qr.id]
    assert editor.inspector.current_section() == "code"


def test_warning_dot_follows_layer_info(editor):
    panel, controller = editor.layers_panel, editor.controller
    text = next(item for item in controller.side().items if item.kind == "text")
    assert panel.info(text.id).warning is False
    controller.set_properties(text.id, {"x": -20.0})
    assert panel.info(text.id).warning is True
    editor.undo()
    assert panel.info(text.id).warning is False


def test_bound_items_are_marked(editor, workspace, people_store):
    workspace.set_store(people_store)
    item = editor.add_field("이름")
    info = editor.layers_panel.info(item.id)
    assert info.bound is True and info.name == "이름"


def test_delegate_paints_every_state(editor, qapp):
    panel, controller = editor.layers_panel, editor.controller
    ids = panel.ids()
    controller.set_locked(ids[0], True)
    controller.set_visible(ids[1], False)
    controller.set_properties(ids[2], {"x": -100.0})
    controller.select(ids[2])
    pump(qapp)
    image = panel.list.viewport().grab().toImage()
    assert not image.isNull() and image.width() > 50
