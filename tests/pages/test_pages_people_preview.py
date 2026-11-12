import pytest
from people_helpers import *  # noqa: F401,F403
from people_helpers import write_png

from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

from data.store import PhotoAdjustment
from design.model import Crop
from ui.pages.people.preview import (
    clamp_adjustment,
    crop_for_adjustment,
    photo_items,
    render_card_image,
)
from ui.pages.people.thumbs import ThumbLoader, read_cover


def id_of(page, name):
    return next(r["_id"] for r in page.store.rows() if r["이름"] == name)


def pixels_differ(a, b) -> bool:
    return a.size() != b.size() or a != b


def send(widget, kind, pos, buttons=Qt.LeftButton):
    button = Qt.LeftButton if kind != QEvent.MouseMove else Qt.NoButton
    held = Qt.NoButton if kind == QEvent.MouseButtonRelease else buttons
    event = QMouseEvent(kind, QPointF(pos), QPointF(widget.mapToGlobal(pos.toPoint())), button, held, Qt.NoModifier)
    QApplication.sendEvent(widget, event)


# ---------------------------------------------------------------- 조정값 계산

def test_crop_conversion_matches_frame_units():
    # 세로로 긴 사진(100x200)을 정사각 틀에 채우면 위아래로 틀 높이의 절반씩 남는다.
    crop = crop_for_adjustment(PhotoAdjustment(1.0, 0.0, 0.25), (100, 200), (30, 30))
    assert (crop.zoom, crop.offset_x, crop.offset_y) == (1.0, 0.0, 0.5)
    # 남는 폭보다 더 옮기면 빈 곳이 생기지 않게 끝에서 멈춘다.
    assert crop_for_adjustment(PhotoAdjustment(1.0, 0.4, 2.0), (100, 200), (30, 30)).offset_y == 1.0
    assert crop_for_adjustment(PhotoAdjustment(0, 0, 0), (100, 200), (30, 30)) == Crop()


def test_clamp_keeps_the_frame_filled():
    clamped = clamp_adjustment(PhotoAdjustment(5.0, 3.0, -3.0), (100, 100), (30, 30))
    assert clamped.zoom == 3.0
    assert clamped.offset_x == pytest.approx(1.0) and clamped.offset_y == pytest.approx(-1.0)
    still = clamp_adjustment(PhotoAdjustment(0.5, 0.2, 0.2), (100, 100), (30, 30))
    assert still == PhotoAdjustment(1.0, 0.0, 0.0)      # 딱 맞는 사진은 옮길 곳이 없다


# ---------------------------------------------------------------- 카드 그리기

def test_render_card_uses_the_record_and_crops_bleed(workspace, people_store, photo_folder):
    document = workspace.editor.document
    workspace.set_store(people_store)
    record = people_store.rows()[0]
    image = render_card_image(document, "front", workspace.binding_record(record), store=people_store,
                              record_id=record["_id"], width_px=430)
    assert abs(image.width() - 430) <= 1
    assert image.height() / image.width() == pytest.approx(54 / 86, abs=0.01)     # 칼선 크기 비율
    other = render_card_image(document, "front", workspace.binding_record(people_store.rows()[1]),
                              store=people_store, record_id=people_store.rows()[1]["_id"], width_px=430)
    assert pixels_differ(image, other)
    placeholder = render_card_image(document, "front", None, width_px=430)
    assert pixels_differ(image, placeholder)
    assert [item.name for item in photo_items(document, "front", people_store)] == ["사진"]


def test_stored_adjustment_changes_the_render_and_leaves_the_design_alone(workspace, people_store, photo_folder):
    write_png(photo_folder / "HN260415.png", size=(120, 300), color=(10, 200, 10))
    from PIL import Image, ImageDraw

    picture = Image.open(photo_folder / "HN260415.png")
    ImageDraw.Draw(picture).rectangle([0, 0, 120, 60], fill=(250, 10, 10))
    picture.save(photo_folder / "HN260415.png")
    workspace.set_store(people_store)
    record = people_store.rows()[0]
    document = workspace.editor.document
    item = photo_items(document, "front", people_store)[0]

    def render():
        return render_card_image(document, "front", workspace.binding_record(record), store=people_store,
                                 record_id=record["_id"], width_px=430)

    before = render()
    people_store.set_photo_adjustment(record["_id"], "사진", PhotoAdjustment(1.0, 0.0, 0.5))
    moved = render()
    assert pixels_differ(before, moved)
    assert item.record_crops == {}                       # 디자인 문서에는 흔적이 남지 않는다
    live = render_card_image(document, "front", workspace.binding_record(record), store=people_store,
                             record_id=record["_id"], width_px=430,
                             adjustments={"사진": PhotoAdjustment(2.0, 0.0, 0.0)})
    assert pixels_differ(moved, live)


# ---------------------------------------------------------------- 오른쪽 미리 보기

def test_panel_follows_the_current_row(page, workspace, qapp):
    panel = page.preview
    assert panel.card.image() is not None
    assert panel.record()["이름"] == "강민서" and panel.position_label.text() == "1 / 8"
    assert not panel.prev_button.isEnabled() and panel.next_button.isEnabled()
    first = panel.card.image().copy()
    page.table.go_to_row(7)
    assert panel.record()["이름"] == "양수아" and panel.position_label.text() == "8 / 8"
    assert not panel.next_button.isEnabled()
    assert pixels_differ(first, panel.card.image())
    assert not panel.side_control.isVisibleTo(panel)     # 한 면짜리 카드


def test_front_back_toggle_for_two_sided_cards(page, workspace, qapp):
    workspace.editor.document.dual_side = True
    page.on_enter()
    panel = page.preview
    assert panel.side_control.isVisibleTo(panel)
    front = panel.card.image().copy()
    panel.side_control.set_current("back")
    assert panel.side() == "back"
    assert pixels_differ(front, panel.card.image())


def patterned_photo(path):
    """확대하거나 옮기면 달라 보이는 사진."""
    from PIL import Image, ImageDraw

    picture = Image.new("RGB", (120, 160), (40, 160, 90))
    draw = ImageDraw.Draw(picture)
    draw.rectangle([0, 0, 120, 30], fill=(230, 40, 40))
    draw.ellipse([40, 60, 80, 100], fill=(250, 250, 250))
    picture.save(path)


def test_photo_adjust_zoom_is_stored_per_person(page, workspace, qapp, photo_folder):
    patterned_photo(photo_folder / "HN260415.png")
    page.table.go_to_row(1)
    page.table.go_to_row(0)
    panel = page.preview
    store = workspace.store
    assert panel.adjust_toggle.isVisibleTo(panel) and not panel.adjust_box.isVisibleTo(panel)
    panel.adjust_toggle.click()
    assert panel.adjust_box.isVisibleTo(panel) and panel.card.frame_rect() is not None
    before = panel.card.image().copy()
    panel.zoom_slider.setValue(180)
    first = id_of(page, "강민서")
    assert store.get_photo_adjustment(first, "사진").zoom == pytest.approx(1.8)
    assert panel.zoom_value.text() == "180%" and pixels_differ(before, panel.card.image())
    assert store.get_photo_adjustment(id_of(page, "고은채"), "사진") == PhotoAdjustment()

    page.table.go_to_row(1)
    assert panel.zoom_slider.value() == 100             # 다른 사람은 따로
    page.table.go_to_row(0)
    assert panel.zoom_slider.value() == 180
    panel.reset_button.click()
    assert store.get_photo_adjustment(first, "사진") == PhotoAdjustment()
    assert not store.has_photo_adjustment(first, "사진") if hasattr(store, "has_photo_adjustment") else True


def test_dragging_the_photo_moves_it(page, workspace, qapp):
    panel = page.preview
    panel.adjust_toggle.click()
    panel.zoom_slider.setValue(200)
    frame = panel.card.frame_rect()
    start = frame.center()
    send(panel.card, QEvent.MouseButtonPress, start)
    send(panel.card, QEvent.MouseMove, start + QPointF(frame.width() * 0.2, -frame.height() * 0.1))
    live = panel.current_adjustment()
    assert live.offset_x == pytest.approx(0.2, abs=0.02) and live.offset_y == pytest.approx(-0.1, abs=0.02)
    stored_while_dragging = workspace.store.get_photo_adjustment(id_of(page, "강민서"), "사진")
    assert stored_while_dragging.offset_x == 0.0        # 놓을 때 저장한다
    send(panel.card, QEvent.MouseButtonRelease, start + QPointF(frame.width() * 5, 0))
    stored = workspace.store.get_photo_adjustment(id_of(page, "강민서"), "사진")
    assert stored.zoom == 2.0 and stored.offset_x == pytest.approx(0.2, abs=0.02)

    send(panel.card, QEvent.MouseButtonPress, start)
    send(panel.card, QEvent.MouseMove, start + QPointF(frame.width() * 9, 0))     # 아주 멀리 끌어도
    send(panel.card, QEvent.MouseButtonRelease, start)
    item = photo_items(workspace.editor.document, "front", workspace.store)[0]
    limit = clamp_adjustment(PhotoAdjustment(2.0, 99.0, 0.0), (60, 76), (item.w, item.h), item.fit).offset_x
    stored = workspace.store.get_photo_adjustment(id_of(page, "강민서"), "사진")
    assert stored.offset_x == pytest.approx(limit) and limit < 1.0      # 빈 곳이 생기기 직전에서 멈춘다


def test_apply_same_adjustment_to_everyone(page, workspace, monkeypatch):
    asked = []
    monkeypatch.setattr("ui.pages.people.page.confirm",
                        lambda parent, title, message, *a, **k: asked.append(message) or True)
    panel = page.preview
    panel.adjust_toggle.click()
    panel.zoom_slider.setValue(150)
    panel.apply_all_button.click()
    assert "8명 모두" in asked[0]
    store = workspace.store
    assert all(store.get_photo_adjustment(i, "사진").zoom == 1.5 for i in store.ids())
    assert page.last_toast().text() == "8명의 사진을 같은 크기와 위치로 맞췄습니다."


def test_no_photo_means_nothing_to_adjust(page, workspace):
    page.table.go_to_row(6)                              # 서지안: 사진 없음
    panel = page.preview
    assert not panel.adjust_toggle.isVisibleTo(panel)
    assert panel.current_adjustment() == PhotoAdjustment()


def test_design_change_is_picked_up_on_enter(page, workspace):
    from design.binding import ColumnSource
    from design.model import TextItem

    assert "unmatched_field" not in {i.kind for i in page.issues()}
    workspace.editor.document.side("front").add_item(
        TextItem(name="연락처", source=ColumnSource(column="연락처"), x=5, y=5, w=30, h=6))
    page.on_enter()
    issue = next(i for i in page.issues() if i.kind == "unmatched_field")
    assert issue.payload == "연락처" and issue.action_text == "맞추기"
    page.match_field("연락처", None)                     # 빈 항목으로 새로 만든다
    assert "연락처" in workspace.store.column_names()
    assert next(i for i in page.issues() if i.kind == "empty_field" and i.payload == "연락처")


# ---------------------------------------------------------------- 작은 사진

def test_thumbnails_load_lazily_and_are_cached(qapp, tmp_path):
    path = write_png(tmp_path / "big.png", size=(900, 1200))
    loader = ThumbLoader()
    done = []
    loader.loaded.connect(lambda: done.append(1))
    assert loader.get(path) is None                      # 처음에는 기다리지 않고 빈 채로 그린다
    assert loader.get(path) is None and len(loader._queue) == 1
    loader.flush()
    image = loader.get(path)
    assert image is not None and (image.width(), image.height()) == (90, 114)
    assert done
    assert loader.get(str(tmp_path / "없음.png")) is None
    loader.flush()
    assert loader.get(str(tmp_path / "없음.png")).isNull()


def test_cover_crop_fills_the_box(tmp_path):
    wide = write_png(tmp_path / "wide.png", size=(800, 200))
    image = read_cover(wide, 90, 114)
    assert (image.width(), image.height()) == (90, 114)


def test_table_paints_thumbnails_and_missing_boxes(page, qapp):
    page.table.grab()
    page.table.thumbs.flush()
    image = page.table.grab().toImage()
    from ui.widgets import theme_color

    def cell_pixels(row):
        rect = page.table.visualRect(page.model.index(row, page.model.column_of("사진")))
        top = page.table.viewport().mapTo(page.table, rect.topLeft())
        return {image.pixelColor(top.x() + x, top.y() + y).name()
                for x in range(4, 34) for y in range(5, 43)}

    assert len(cell_pixels(0) - {theme_color("accent_soft").name()}) >= 1          # 사진이 그려졌다
    missing = cell_pixels(4)
    assert theme_color("warn_mark").name() in missing                              # 점선 빈 상자
