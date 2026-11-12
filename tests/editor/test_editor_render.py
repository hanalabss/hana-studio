import numpy as np
import pytest
from PySide6.QtCore import QRectF
from PySide6.QtGui import QColor, QImage, QPainter

from design import renderer
from design.binding import ColumnSource
from design.model import (
    BarcodeItem, Document, EllipseItem, ImageItem, LineItem, QRItem, RectItem, TextItem,
)
from ui.editor import EditorController

K = renderer.mm_to_px_for(300)


def px(value: float) -> float:
    return value / K


@pytest.fixture
def parity_controller(qapp, make_png):
    """300dpi에서 1000×600px로 떨어지는 카드에 모든 종류의 항목을 올린다."""
    from design import storage

    document = Document(card_width_mm=px(1000), card_height_mm=px(600))
    document.base_dir = storage.new_workdir()
    asset = storage.import_asset(document.base_dir, make_png(size=(240, 160)))
    side = document.side("front")
    side.background = "#F4F1EA"
    side.add_item(RectItem(name="띠", x=0, y=0, w=px(1000), h=px(120), fill="#1f4e79", stroke_color=None))
    side.add_item(TextItem(name="제목", x=px(40), y=px(20), w=px(500), h=px(80), text="하나스튜디오",
                           font_size_pt=16, bold=True, color="#ffffff", v_align="middle"))
    side.add_item(ImageItem(name="그림", x=px(40), y=px(160), w=px(240), h=px(300), asset=asset))
    side.add_item(TextItem(name="이름", x=px(320), y=px(160), w=px(400), h=px(90), font_size_pt=20,
                           source=ColumnSource(column="이름"), auto_shrink=True))
    side.add_item(EllipseItem(name="원", x=px(780), y=px(150), w=px(160), h=px(120), fill="#ffd54f",
                              stroke_color="#333333", stroke_width_mm=0.4))
    side.add_item(LineItem(name="선", x=px(320), y=px(270), w=px(420), h=0.0, color="#c62828", width_mm=0.5))
    side.add_item(RectItem(name="돌린 상자", x=px(760), y=px(330), w=px(180), h=px(60), rotation=25.0,
                           fill="#2e7d32", corner_radius_mm=1.0, stroke_color=None))
    side.add_item(BarcodeItem(name="바코드", x=px(320), y=px(300), w=px(400), h=px(140), data="HN260417"))
    side.add_item(QRItem(name="큐알", x=px(760), y=px(420), w=px(160), h=px(160), data="https://example.com/1"))
    controller = EditorController(document=document)
    yield controller
    storage.cleanup_workdir(document)


def _scene_image(controller, width: int, height: int) -> QImage:
    image = QImage(width, height, QImage.Format.Format_RGB32)
    image.fill(QColor("#FFFFFF"))
    painter = QPainter(image)
    painter.setRenderHints(
        QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing
        | QPainter.RenderHint.SmoothPixmapTransform
    )
    controller.scene.render(painter, QRectF(0, 0, width, height), controller.scene.card_rect())
    painter.end()
    return image


def _item_region(item, scale: float, shape) -> tuple[slice, slice]:
    left, top, right, bottom = item.bounds()
    if item.kind == "line":
        left, top, right, bottom = left - 0.5, top - 0.5, right + 0.5, bottom + 0.5
    y0, y1 = max(0, int(top * scale)), min(shape[0], int(np.ceil(bottom * scale)))
    x0, x1 = max(0, int(left * scale)), min(shape[1], int(np.ceil(right * scale)))
    return slice(y0, y1), slice(x0, x1)


def _difference(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    delta = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=2)
    return float(delta.mean()), float((delta > 48).mean())


@pytest.mark.parametrize("record", [None, {"이름": "김하늘"}])
def test_scene_matches_print_render_per_item(parity_controller, qimage_array, record):
    controller = parity_controller
    controller.set_preview_record(record)
    printed = np.asarray(renderer.render_side(controller.document, "front", record, dpi=300, design_mode=True).color)
    assert printed.shape[:2] == (600, 1000)
    canvas = qimage_array(_scene_image(controller, 1000, 600))

    mean, bad = _difference(canvas, printed)
    assert mean < 1.0 and bad < 0.003
    for item in controller.side().items:
        rows, cols = _item_region(item, K, printed.shape)
        mean, bad = _difference(canvas[rows, cols], printed[rows, cols])
        assert mean < 2.0, item.name
        assert bad < 0.01, item.name
        assert printed[rows, cols].std() > 1.0, item.name


def test_barcode_and_qr_modules_are_pixel_identical(parity_controller, qimage_array):
    controller = parity_controller
    printed = np.asarray(renderer.render_side(controller.document, "front", None, dpi=300).color)
    canvas = qimage_array(_scene_image(controller, 1000, 600))
    for item in controller.side().items:
        if item.kind not in ("barcode", "qr"):
            continue
        rows, cols = _item_region(item, K, printed.shape)
        if item.kind == "barcode":
            rows = slice(rows.start, rows.start + 60)
        assert np.array_equal(canvas[rows, cols], printed[rows, cols]), item.name


def test_view_grab_matches_print_render(parity_controller, qimage_array, qapp):
    from ui.editor import EditorView

    controller = parity_controller
    view = EditorView(controller)
    view.resize(900, 640)
    view.show()
    qapp.processEvents()
    try:
        scale = 8.0
        view.set_zoom(scale / (view.px_per_mm / view.zoom))
        view.centerOn(controller.scene.card_rect().center())
        qapp.processEvents()
        origin = view.mm_to_viewport(0.0, 0.0)
        grabbed = qimage_array(view.viewport().grab().toImage())
        printed = np.asarray(renderer.render_side(
            controller.document, "front", None, dpi=scale * 25.4, design_mode=True,
        ).color)
        ox, oy = round(origin.x()), round(origin.y())
        for item in controller.side().items:
            if item.name in ("띠",):
                continue
            rows, cols = _item_region(item, scale, printed.shape)
            canvas = grabbed[oy + rows.start: oy + rows.stop, ox + cols.start: ox + cols.stop]
            mean, bad = _difference(canvas, printed[rows, cols])
            assert mean < 3.0, (item.name, mean)
            assert bad < 0.03, (item.name, bad)
    finally:
        view.close()
        view.deleteLater()


def test_scene_shows_side_background_and_hidden_items(parity_controller, qimage_array):
    controller = parity_controller
    canvas = qimage_array(_scene_image(controller, 1000, 600))
    assert tuple(canvas[590, 20]) == (0xF4, 0xF1, 0xEA)
    ellipse = next(it for it in controller.side().items if it.name == "원")
    assert tuple(canvas[210, 860]) == (0xFF, 0xD5, 0x4F)
    controller.set_visible([ellipse.id], False)
    canvas = qimage_array(_scene_image(controller, 1000, 600))
    assert tuple(canvas[210, 860]) == (0xF4, 0xF1, 0xEA)
    printed = np.asarray(renderer.render_side(controller.document, "front", None, dpi=300).color)
    assert tuple(printed[210, 860]) == (0xF4, 0xF1, 0xEA)


def test_items_repaint_only_when_content_changes(parity_controller, monkeypatch):
    """끌어서 옮길 때는 캐시된 그림을 그대로 쓴다."""
    controller = parity_controller
    barcode = next(it for it in controller.side().items if it.kind == "barcode")
    canvas_item = controller.scene.canvas_item(barcode.id)
    assert canvas_item.cacheMode() == canvas_item.CacheMode.DeviceCoordinateCache
    updates = []
    monkeypatch.setattr(type(canvas_item), "update", lambda self, *a: updates.append(self.model.id))
    token = controller.new_merge_token()
    for step in range(5):
        controller.move_items_to({barcode.id: (barcode.x + 1.0, barcode.y)}, token)
    assert updates == []
    controller.set_property([barcode.id], "data", "CHANGED")
    assert updates == [barcode.id]
