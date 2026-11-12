import json
import math

import pytest

from design.binding import ColumnSource, ConcatPart, ConcatSource, CounterSource, DateSource, FixedSource
from design.model import (
    CURRENT_VERSION, BarcodeItem, Crop, Document, EllipseItem, ImageItem, LineItem, QRItem,
    RectItem, Side, TextItem, UnknownItem, align_items, distribute_items, item_from_dict,
    rotated_bounds, selection_bounds,
)


def full_document() -> Document:
    document = Document(name="사원증", dual_side=True, bleed_mm=2.0)
    front = document.side("front")
    front.orientation = "portrait"
    front.background = "#fafafa"
    front.add_item(TextItem(
        text="홍길동", font_family="Malgun Gothic", font_size_pt=14.5, bold=True, italic=True,
        underline=True, color="#112233", h_align="center", v_align="middle", auto_shrink=True,
        wrap=True, x=1.5, y=2.25, w=30, h=8, rotation=15.0, name="이름",
        source=ColumnSource(column="이름"),
    ))
    image = ImageItem(asset="assets/a.png", fit="contain", mask_asset="assets/m.png",
                      crop=Crop(offset_x=0.2, offset_y=-0.4, zoom=1.5),
                      source=ColumnSource(column="사진"))
    image.set_record_crop(7, Crop(offset_x=-1.0, zoom=2.0))
    front.add_item(image)
    front.add_item(RectItem(fill=None, stroke_color="#ff0000", stroke_width_mm=0.4, corner_radius_mm=2.0,
                            layer="white", locked=True))
    front.add_item(EllipseItem(fill="#00ff00", layer="uv", printable=False))
    front.add_item(LineItem(x=1, y=1, w=10, h=5, flip=True, style="dash", visible=False))
    front.add_item(BarcodeItem(symbology="ean13", data="4006381333931", show_text=False,
                               source=CounterSource(start=10, step=5, pad=6)))
    front.add_item(QRItem(data="https://example.com", error_correction="H",
                          source=ConcatSource(separator="-", parts=[
                              ConcatPart(column="부서", prefix="[", suffix="]"), ConcatPart(text="고정"),
                          ])))
    back = document.side("back")
    back.add_item(TextItem(text="발급일", source=DateSource(format="%Y년 %m월 %d일", offset_days=3)))
    back.add_item(TextItem(source=FixedSource(value="고정 문구")))
    return document


def test_item_defaults_follow_contract():
    item = TextItem()
    assert item.kind == "text"
    assert (item.locked, item.visible, item.printable, item.layer, item.source) == (False, True, True, "color", None)
    assert isinstance(item.x, float) and isinstance(item.z, int)
    assert len({TextItem().id for _ in range(50)}) == 50
    kinds = {cls().kind for cls in (TextItem, ImageItem, RectItem, EllipseItem, LineItem, BarcodeItem, QRItem)}
    assert kinds == {"text", "image", "rect", "ellipse", "line", "barcode", "qr"}


def test_document_round_trip_is_lossless_through_json():
    document = full_document()
    encoded = json.dumps(document.to_dict(), ensure_ascii=False)
    restored = Document.from_dict(json.loads(encoded))
    assert restored == document
    assert restored.to_dict() == document.to_dict()
    assert restored.version == CURRENT_VERSION
    image = restored.side("front").items[1]
    assert isinstance(image, ImageItem)
    assert image.crop_for(7).zoom == 2.0
    assert image.crop_for("7").offset_x == -1.0
    assert image.crop_for(8) == Crop(offset_x=0.2, offset_y=-0.4, zoom=1.5)
    assert isinstance(restored.side("front").items[0].source, ColumnSource)


def test_base_dir_is_not_serialized():
    document = Document(base_dir="C:/어딘가")
    assert "base_dir" not in document.to_dict()
    assert Document.from_dict(document.to_dict()).base_dir is None


def test_unknown_keys_are_preserved_at_every_level():
    data = full_document().to_dict()
    data["future_doc"] = {"a": [1, 2]}
    data["sides"]["front"]["future_side"] = "s"
    data["sides"]["front"]["items"][0]["future_item"] = 3
    data["sides"]["front"]["items"][0]["source"]["future_source"] = True
    data["sides"]["front"]["items"][1]["crop"]["future_crop"] = "c"
    data["sides"]["front"]["items"][6]["source"]["parts"][0]["future_part"] = None
    hologram = {"kind": "hologram", "id": "h1", "x": 1, "y": 2, "w": 3, "h": 4, "pattern": "wave", "z": 99}
    data["sides"]["front"]["items"].append(hologram)
    data["sides"]["back"]["items"][0]["source"] = {"type": "lookup", "table": "t", "key": "k"}

    restored = Document.from_dict(json.loads(json.dumps(data)))
    again = restored.to_dict()
    saved_hologram = again["sides"]["front"]["items"].pop()
    data["sides"]["front"]["items"].pop()
    assert again == data
    assert hologram.items() <= saved_hologram.items()
    assert Document.from_dict(restored.to_dict()).to_dict() == restored.to_dict()

    unknown = restored.side("front").items[-1]
    assert isinstance(unknown, UnknownItem)
    assert (unknown.kind, unknown.x, unknown.extra["pattern"]) == ("hologram", 1.0, "wave")
    assert restored.side("back").items[0].source.type == "lookup"


def test_newer_version_is_kept_and_older_is_upgraded():
    data = Document().to_dict()
    data["version"] = CURRENT_VERSION + 5
    newer = Document.from_dict(data)
    assert newer.is_newer_format
    assert newer.to_dict()["version"] == CURRENT_VERSION + 5

    data.pop("version")
    assert Document.from_dict(data).version == CURRENT_VERSION
    assert not Document.from_dict(data).is_newer_format


@pytest.mark.parametrize("patch", [
    {"version": 0}, {"version": "1"}, {"format": "other"}, {"card_width_mm": "넓게"}, {"dual_side": 1},
])
def test_invalid_document_values_are_rejected(patch):
    data = Document().to_dict()
    data.update(patch)
    with pytest.raises(ValueError):
        Document.from_dict(data)


def test_integers_from_json_become_float_mm():
    item = item_from_dict({"kind": "rect", "x": 1, "y": 2, "w": 3, "h": 4, "rotation": 90})
    assert all(isinstance(v, float) for v in (item.x, item.y, item.w, item.h, item.rotation))


def test_side_size_respects_orientation():
    document = Document()
    assert document.side_size_mm("front") == (85.6, 53.98)
    document.side("front").orientation = "portrait"
    assert document.side_size_mm("front") == (53.98, 85.6)
    assert document.side_size_mm("back") == (85.6, 53.98)
    swapped = Document(card_width_mm=54.0, card_height_mm=86.0)
    assert swapped.side_size_mm("front") == (86.0, 54.0)
    with pytest.raises(KeyError):
        document.side("옆면")


def test_active_sides_and_find_item():
    document = full_document()
    assert document.active_side_keys() == ["front", "back"]
    document.dual_side = False
    assert document.active_side_keys() == ["front"]
    target = document.side("back").items[0]
    assert document.find_item(target.id) == ("back", target)
    assert document.find_item("없음") is None
    assert document.asset_refs() == {"assets/a.png", "assets/m.png"}


# ---------------------------------------------------------------- 추가 / 삭제 / 복제 / 순서

def make_side(count: int = 4) -> tuple[Side, list[str]]:
    side = Side()
    ids = [side.add_item(RectItem(name=f"r{i}")).id for i in range(count)]
    return side, ids


def order(side: Side) -> list[str]:
    assert [it.z for it in side.sorted_items()] == list(range(len(side.items)))
    return [it.name for it in side.sorted_items()]


def test_add_remove_duplicate():
    side, ids = make_side(3)
    assert order(side) == ["r0", "r1", "r2"]

    bottom = side.add_item(RectItem(name="bottom"), on_top=False)
    assert order(side) == ["bottom", "r0", "r1", "r2"]

    same_id = RectItem(name="dup-id", id=ids[0])
    side.add_item(same_id)
    assert same_id.id != ids[0]
    assert len({it.id for it in side.items}) == 5

    original = side.get(ids[1])
    original.locked = True
    original.x, original.y = 10.0, 20.0
    clone = side.duplicate_item(ids[1], offset_mm=(3.0, 4.0))
    assert clone.id != original.id and clone.name == "r1"
    assert (clone.x, clone.y, clone.locked) == (13.0, 24.0, False)
    assert order(side) == ["bottom", "r0", "r1", "r1", "r2", "dup-id"]
    assert side.sorted_items()[3] is clone
    clone.name = "changed"
    assert original.name == "r1"

    assert side.remove_item(bottom.id) is bottom
    assert side.remove_item("없음") is None
    assert side.duplicate_item("없음") is None
    assert order(side) == ["r0", "r1", "changed", "r2", "dup-id"]
    assert len(side.remove_items([ids[0], ids[2], "없음"])) == 2
    assert order(side) == ["r1", "changed", "dup-id"]


def test_z_reordering():
    side, ids = make_side(4)
    side.bring_to_front([ids[0]])
    assert order(side) == ["r1", "r2", "r3", "r0"]
    side.send_to_back([ids[3], ids[0]])
    assert order(side) == ["r3", "r0", "r1", "r2"]
    side.bring_forward([ids[3]])
    assert order(side) == ["r0", "r3", "r1", "r2"]
    side.send_backward([ids[2]])
    assert order(side) == ["r0", "r3", "r2", "r1"]


def test_forward_backward_keep_selection_order_and_stop_at_ends():
    side, ids = make_side(4)
    side.bring_forward([ids[0], ids[1]])
    assert order(side) == ["r2", "r0", "r1", "r3"]
    side.bring_forward([ids[0], ids[1]])
    side.bring_forward([ids[0], ids[1]])
    assert order(side) == ["r2", "r3", "r0", "r1"]
    side.send_backward([ids[0], ids[1]])
    side.send_backward([ids[0], ids[1]])
    side.send_backward([ids[0], ids[1]])
    assert order(side) == ["r0", "r1", "r2", "r3"]


def test_sorted_items_uses_z_not_list_position():
    side = Side(items=[RectItem(name="top", z=5), RectItem(name="bottom", z=-1), RectItem(name="mid", z=5)])
    assert [it.name for it in side.sorted_items()] == ["bottom", "top", "mid"]
    side.normalize_z()
    assert [(it.name, it.z) for it in side.items] == [("bottom", 0), ("top", 1), ("mid", 2)]


# ---------------------------------------------------------------- 정렬 / 분배

def test_rotated_bounds():
    assert rotated_bounds(10, 20, 30, 10, 0) == (10, 20, 40, 30)
    left, top, right, bottom = rotated_bounds(10, 20, 30, 10, 90)
    assert (left, top, right, bottom) == pytest.approx((20, 10, 30, 40))
    left, top, right, bottom = rotated_bounds(0, 0, 10, 10, 45)
    assert right - left == pytest.approx(10 * math.sqrt(2))
    assert (left + right) / 2 == pytest.approx(5)


def three_rects() -> list[RectItem]:
    return [
        RectItem(x=10, y=5, w=10, h=4),
        RectItem(x=30, y=20, w=20, h=10),
        RectItem(x=70, y=40, w=6, h=6),
    ]


@pytest.mark.parametrize("mode, attr, expected", [
    ("left", "x", [10, 10, 10]),
    ("right", "x", [66, 56, 70]),
    ("center", "x", [38, 33, 40]),
    ("top", "y", [5, 5, 5]),
    ("bottom", "y", [42, 36, 40]),
    ("middle", "y", [23.5, 20.5, 22.5]),
])
def test_align_to_selection(mode, attr, expected):
    items = three_rects()
    other = "y" if attr == "x" else "x"
    before = [getattr(it, other) for it in items]
    align_items(items, mode)
    assert [getattr(it, attr) for it in items] == pytest.approx(expected)
    assert [getattr(it, other) for it in items] == before


def test_align_uses_rotated_bounds_and_skips_locked():
    rotated = RectItem(x=10, y=20, w=30, h=10, rotation=90)
    plain = RectItem(x=0, y=0, w=10, h=10)
    locked = RectItem(x=50, y=50, w=10, h=10, locked=True)
    moved = align_items([rotated, plain, locked], "left")
    assert rotated.bounds()[0] == pytest.approx(0)
    assert rotated.x == pytest.approx(-10)
    assert (locked.x, locked.y) == (50, 50)
    assert moved == [rotated]


def test_align_to_card_and_single_item():
    document = Document(card_width_mm=90, card_height_mm=58, bleed_mm=2)
    side = document.side("front")
    a = side.add_item(RectItem(x=10, y=10, w=20, h=10))
    b = side.add_item(RectItem(x=50, y=30, w=10, h=10))
    document.align("front", [a.id], "right")
    assert a.x == pytest.approx(90 - 2 - 20)
    document.align("front", [a.id, b.id], "middle", to_card=True)
    assert a.y == pytest.approx(24) and b.y == pytest.approx(24)
    document.align("front", [a.id, b.id], "left")
    assert a.x == b.x == pytest.approx(50)
    with pytest.raises(ValueError):
        align_items([a], "diagonal")
    assert align_items([], "left") == []


def test_distribute_equal_gaps():
    items = [
        RectItem(x=0, y=0, w=10, h=10),
        RectItem(x=12, y=30, w=20, h=10),
        RectItem(x=90, y=7, w=10, h=10),
        RectItem(x=40, y=3, w=20, h=10),
    ]
    moved = distribute_items(items, "horizontal")
    ordered = sorted(items, key=lambda it: it.x)
    gaps = [b.x - (a.x + a.w) for a, b in zip(ordered, ordered[1:])]
    assert gaps == pytest.approx([40 / 3] * 3)
    assert (ordered[0].x, ordered[-1].x + ordered[-1].w) == (0, 100)
    assert len(moved) == 2
    assert [it.y for it in items] == [0, 30, 7, 3]


def test_distribute_vertical_overlapping_and_small_selection():
    items = [RectItem(x=0, y=0, w=5, h=10), RectItem(x=0, y=2, w=5, h=10), RectItem(x=0, y=14, w=5, h=10)]
    distribute_items(items, "vertical")
    assert [it.y for it in items] == pytest.approx([0, 7, 14])
    assert selection_bounds(items) == pytest.approx((0, 0, 5, 24))

    pair = [RectItem(x=0), RectItem(x=50)]
    assert distribute_items(pair, "horizontal") == []
    assert [it.x for it in pair] == [0, 50]
    with pytest.raises(ValueError):
        distribute_items(items, "depth")


def test_line_points_round_trip():
    line = LineItem.from_points(30, 5, 10, 25)
    assert (line.x, line.y, line.w, line.h, line.flip) == (10, 5, 20, 20, True)
    assert line.points() == ((10, 25), (30, 5))
    down = LineItem.from_points(0, 0, 10, 4)
    assert down.points() == ((0, 0), (10, 4)) and not down.flip


def test_integer_mm_values_are_stored_and_saved_as_float():
    item = RectItem(x=5, y=6, w=7, h=8, stroke_width_mm=1)
    assert all(isinstance(v, float) for v in (item.x, item.y, item.w, item.h, item.stroke_width_mm))
    assert isinstance(item.z, int)
    assert isinstance(Crop(zoom=2).zoom, float)
    document = Document(card_width_mm=90, card_height_mm=58, bleed_mm=2)
    assert isinstance(document.card_width_mm, float) and isinstance(document.bleed_mm, float)

    item.x = 12
    saved = item.to_dict()
    assert isinstance(saved["x"], float) and json.dumps(saved["x"]) == "12.0"


def test_model_and_binding_do_not_import_qt():
    import subprocess
    import sys

    code = (
        "import sys, design.model, design.binding, design.presets;"
        "bad = [m for m in sys.modules if m.startswith('PySide6')];"
        "sys.exit(1 if bad else 0)"
    )
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
