from pages_design_helpers import *  # noqa: F401,F403

from design import presets, storage


def test_blank_document_shows_start_panel(page):
    assert not page.is_started()
    assert page.start_panel.isVisible()
    assert not page.editor_host.isVisible()
    assert page.document_state() == ("새 카드", None)
    assert set(page.start_panel.cards) == {template.key for template in presets.TEMPLATES}


def test_thumbnails_are_rendered_for_every_template(page):
    for card in page.start_panel.cards.values():
        pixmap = card.thumb.pixmap()
        assert pixmap is not None and not pixmap.isNull()
    blank = page.start_panel.cards["blank"].thumb.pixmap().toImage()
    employee = page.start_panel.cards["employee"].thumb.pixmap().toImage()
    assert blank != employee


def test_start_with_template_uses_chosen_size_orientation_and_sides(page):
    panel = page.start_panel
    panel.size_control.set_current("cut_54x86_bleed")
    panel.orientation_control.set_current("portrait")
    panel.sides_control.set_current("dual")
    panel.cards["employee"].clicked.emit()

    document = page.controller.document
    assert page.is_started() and page.editor_host.isVisible()
    assert (document.card_width_mm, document.card_height_mm, document.bleed_mm) == (90.0, 58.0, 2.0)
    assert document.dual_side is True
    assert document.side("front").orientation == "portrait"
    assert {item.name for item in document.side("front").items} >= {"사진", "이름", "사번 바코드"}
    assert not page.controller.is_modified
    assert page.canvas_bar.side_control.isVisible()


def test_start_blank_card_opens_empty_editor(page):
    page.start_panel.choose("blank")
    document = page.controller.document
    assert page.is_started()
    assert document.side("front").items == []
    assert (document.card_width_mm, document.card_height_mm) == (presets.CR80.width_mm, presets.CR80.height_mm)
    assert page.inspector.current_section() == "card"
    assert not page.canvas_bar.side_control.isVisible()
    assert page.canvas_bar.add_back_button.isVisible()


def test_size_hint_follows_choice(page):
    panel = page.start_panel
    before = panel.size_hint.text()
    panel.size_control.set_current("cut_54x86_bleed")
    assert panel.size_hint.text() != before
    assert "58 × 90" in panel.size_hint.text()


def test_open_existing_design_file(page, tmp_path):
    document = presets.build_template("visitor", presets.CR80)
    document.base_dir = storage.new_workdir()
    path = storage.save(document, tmp_path / "visit.hsd")
    page.dialogs.design_path = path

    page.start_panel.open_button.click()

    assert page.is_started()
    assert page.controller.path == path
    assert any(item.name == "방문자" for item in page.controller.side().items)
    assert page.document_state() == ("visit.hsd", True)


def test_open_broken_file_reports_and_stays_on_start(page, tmp_path):
    broken = tmp_path / "broken.hsd"
    broken.write_text("이건 디자인 파일이 아닙니다", encoding="utf-8")
    assert page.open_design(str(broken)) is False
    assert not page.is_started()
    assert "열지 못했습니다" in page.last_message


def test_start_from_single_image_makes_one_full_image_item(page, tmp_path):
    page.dialogs.image_path = make_image(tmp_path / "art.png", size=(900, 580))

    page.start_panel.image_button.click()

    document = page.controller.document
    items = document.side("front").items
    assert page.is_started()
    assert len(items) == 1 and items[0].kind == "image"
    width, height = document.side_size_mm("front")
    assert (items[0].x, items[0].y, items[0].w, items[0].h) == (0.0, 0.0, width, height)
    assert document.bleed_mm == 2.0          # 58×90 비율이면 재단 여백 문서
    assert page.inspector.current_section() == "card"


def test_cancelled_file_dialog_changes_nothing(page):
    page.dialogs.design_path = None
    page.dialogs.image_path = None
    assert page.open_design() is False
    assert page.start_from_image() is False
    assert not page.is_started()


def test_new_design_returns_to_start_and_asks_about_unsaved_changes(page):
    page.start_panel.choose("employee")
    page.add("text")
    assert page.controller.is_modified

    page.dialogs.unsaved_answer = "cancel"
    assert page.new_design() is False
    assert page.is_started() and page.dialogs.unsaved_asked == 1

    page.dialogs.unsaved_answer = "discard"
    assert page.new_design() is True
    assert not page.is_started()
    assert page.start_panel.isVisible()
    assert page.controller.document.side("front").items == []
