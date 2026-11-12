"""카드 디자인 문서 모델. 단위는 mm(float), Qt에 의존하지 않는다."""

from __future__ import annotations

import copy
import math
import uuid
from dataclasses import dataclass, field
from typing import Any, ClassVar, Iterable, Iterator, Mapping, Sequence

from design._serial import Serializable
from design.binding import Source

FORMAT_NAME = "hana-studio-design"
CURRENT_VERSION = 1

LAYERS = ("color", "white", "uv")
ORIENTATIONS = ("landscape", "portrait")
SIDE_KEYS = ("front", "back")
ALIGN_MODES = ("left", "center", "right", "top", "middle", "bottom")
DISTRIBUTE_AXES = ("horizontal", "vertical")

DEFAULT_FONT_FAMILY = "Malgun Gothic"
CR80_WIDTH_MM = 85.6
CR80_HEIGHT_MM = 53.98

Bounds = tuple[float, float, float, float]  # left, top, right, bottom


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def rotated_bounds(x: float, y: float, w: float, h: float, rotation: float) -> Bounds:
    """중심 기준으로 회전한 사각형을 감싸는 축 정렬 경계."""
    angle = math.radians(rotation % 360.0)
    cos_a, sin_a = abs(math.cos(angle)), abs(math.sin(angle))
    half_w = (abs(w) * cos_a + abs(h) * sin_a) / 2.0
    half_h = (abs(w) * sin_a + abs(h) * cos_a) / 2.0
    cx, cy = x + w / 2.0, y + h / 2.0
    return (cx - half_w, cy - half_h, cx + half_w, cy + half_h)


@dataclass(kw_only=True)
class Crop(Serializable):
    """이미지 위치 조정. offset은 -1..1(0=가운데, 남는 범위 대비 비율), zoom은 배율."""

    offset_x: float = 0.0
    offset_y: float = 0.0
    zoom: float = 1.0
    extra: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._floats_from_ints()


@dataclass(kw_only=True)
class Item(Serializable):
    KIND: ClassVar[str] = ""

    id: str = field(default_factory=new_id)
    kind: str = ""
    x: float = 0.0
    y: float = 0.0
    w: float = 20.0
    h: float = 10.0
    rotation: float = 0.0
    z: int = 0
    locked: bool = False
    visible: bool = True
    printable: bool = True
    layer: str = "color"
    name: str = ""
    source: Source | None = None
    extra: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.KIND:
            self.kind = self.KIND
        self._floats_from_ints()

    def _encode(self, name: str, value: Any) -> Any:
        if name == "source":
            return None if value is None else value.to_dict()
        return super()._encode(name, value)

    @classmethod
    def _decode(cls, name: str, value: Any) -> Any:
        if name == "source":
            return None if value is None else Source.from_dict(value)
        return super()._decode(name, value)

    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)

    def bounds(self) -> Bounds:
        return rotated_bounds(self.x, self.y, self.w, self.h, self.rotation)

    def move_by(self, dx: float, dy: float) -> None:
        self.x += dx
        self.y += dy

    def clone(self, *, keep_id: bool = False) -> "Item":
        other = copy.deepcopy(self)
        if not keep_id:
            other.id = new_id()
        return other


@dataclass(kw_only=True)
class TextItem(Item):
    KIND: ClassVar[str] = "text"
    text: str = ""
    font_family: str = DEFAULT_FONT_FAMILY
    font_size_pt: float = 10.0
    bold: bool = False
    italic: bool = False
    underline: bool = False
    color: str = "#000000"
    h_align: str = "left"      # left | center | right
    v_align: str = "top"       # top | middle | bottom
    auto_shrink: bool = False
    wrap: bool = False
    min_font_size_pt: float = 4.0


@dataclass(kw_only=True)
class ImageItem(Item):
    KIND: ClassVar[str] = "image"
    asset: str | None = None
    fit: str = "fill"          # fill(채우고 넘치면 자름) | contain | stretch
    crop: Crop = field(default_factory=Crop)
    record_crops: dict[str, Crop] = field(default_factory=dict)
    mask_asset: str | None = None

    def _encode(self, name: str, value: Any) -> Any:
        if name == "crop":
            return value.to_dict()
        if name == "record_crops":
            return {str(k): v.to_dict() for k, v in value.items()}
        return super()._encode(name, value)

    @classmethod
    def _decode(cls, name: str, value: Any) -> Any:
        if name == "crop":
            return Crop.from_dict(value)
        if name == "record_crops":
            return {str(k): Crop.from_dict(v) for k, v in value.items()}
        return super()._decode(name, value)

    def crop_for(self, record_key: Any = None) -> Crop:
        if record_key is not None:
            found = self.record_crops.get(str(record_key))
            if found is not None:
                return found
        return self.crop

    def set_record_crop(self, record_key: Any, crop: Crop | None) -> None:
        key = str(record_key)
        if crop is None:
            self.record_crops.pop(key, None)
        else:
            self.record_crops[key] = crop


@dataclass(kw_only=True)
class RectItem(Item):
    KIND: ClassVar[str] = "rect"
    fill: str | None = "#d9d9d9"
    stroke_color: str | None = "#000000"
    stroke_width_mm: float = 0.0
    corner_radius_mm: float = 0.0


@dataclass(kw_only=True)
class EllipseItem(Item):
    KIND: ClassVar[str] = "ellipse"
    fill: str | None = "#d9d9d9"
    stroke_color: str | None = "#000000"
    stroke_width_mm: float = 0.0


@dataclass(kw_only=True)
class LineItem(Item):
    """경계 상자의 대각선. flip=False면 좌상→우하, True면 좌하→우상. h=0이면 수평선."""

    KIND: ClassVar[str] = "line"
    h: float = 0.0
    color: str = "#000000"
    width_mm: float = 0.3
    style: str = "solid"       # solid | dash | dot
    flip: bool = False

    def points(self) -> tuple[tuple[float, float], tuple[float, float]]:
        if self.flip:
            return ((self.x, self.y + self.h), (self.x + self.w, self.y))
        return ((self.x, self.y), (self.x + self.w, self.y + self.h))

    @classmethod
    def from_points(cls, x1: float, y1: float, x2: float, y2: float, **kwargs: Any) -> "LineItem":
        if x1 > x2:
            x1, y1, x2, y2 = x2, y2, x1, y1
        return cls(x=x1, y=min(y1, y2), w=x2 - x1, h=abs(y2 - y1), flip=y2 < y1, **kwargs)


@dataclass(kw_only=True)
class BarcodeItem(Item):
    KIND: ClassVar[str] = "barcode"
    w: float = 40.0
    h: float = 12.0
    symbology: str = "code128"  # code128 | code39 | ean13
    data: str = ""
    show_text: bool = True
    color: str = "#000000"
    background: str | None = "#ffffff"
    quiet_zone: bool = True
    text_font_family: str = DEFAULT_FONT_FAMILY
    text_size_pt: float = 7.0


@dataclass(kw_only=True)
class QRItem(Item):
    KIND: ClassVar[str] = "qr"
    w: float = 20.0
    h: float = 20.0
    data: str = ""
    error_correction: str = "M"  # L | M | Q | H
    color: str = "#000000"
    background: str | None = "#ffffff"
    border_modules: int = 2


@dataclass(kw_only=True)
class UnknownItem(Item):
    """이 버전이 모르는 종류. 그리지 않고, 저장할 때 내용을 그대로 돌려준다."""


ITEM_TYPES: dict[str, type[Item]] = {
    c.KIND: c
    for c in (TextItem, ImageItem, RectItem, EllipseItem, LineItem, BarcodeItem, QRItem)
}


def item_from_dict(data: Mapping[str, Any]) -> Item:
    if not isinstance(data, Mapping):
        raise ValueError("항목 형식이 올바르지 않습니다.")
    return ITEM_TYPES.get(data.get("kind"), UnknownItem).from_dict(data)


# ---------------------------------------------------------------- 정렬 / 분배

def selection_bounds(items: Iterable[Item]) -> Bounds:
    boxes = [item.bounds() for item in items]
    if not boxes:
        raise ValueError("정렬할 항목이 없습니다.")
    return (
        min(b[0] for b in boxes), min(b[1] for b in boxes),
        max(b[2] for b in boxes), max(b[3] for b in boxes),
    )


def align_items(items: Sequence[Item], mode: str, reference: Bounds | None = None) -> list[Item]:
    """items를 reference(없으면 선택 전체 경계)에 맞춰 옮긴다. 잠긴 항목은 기준에만 쓰인다.

    옮겨진 항목 목록을 돌려준다.
    """
    if mode not in ALIGN_MODES:
        raise ValueError(f"알 수 없는 정렬 방식입니다: {mode}")
    if not items:
        return []
    left, top, right, bottom = reference if reference is not None else selection_bounds(items)
    moved: list[Item] = []
    for item in items:
        if item.locked:
            continue
        b_left, b_top, b_right, b_bottom = item.bounds()
        dx = dy = 0.0
        if mode == "left":
            dx = left - b_left
        elif mode == "right":
            dx = right - b_right
        elif mode == "center":
            dx = (left + right) / 2.0 - (b_left + b_right) / 2.0
        elif mode == "top":
            dy = top - b_top
        elif mode == "bottom":
            dy = bottom - b_bottom
        else:
            dy = (top + bottom) / 2.0 - (b_top + b_bottom) / 2.0
        if dx or dy:
            item.move_by(dx, dy)
            moved.append(item)
    return moved


def distribute_items(items: Sequence[Item], axis: str) -> list[Item]:
    """양 끝 항목은 두고 사이 항목을 같은 간격으로 놓는다. 3개 이상일 때만 의미가 있다."""
    if axis not in DISTRIBUTE_AXES:
        raise ValueError(f"알 수 없는 분배 방향입니다: {axis}")
    if len(items) < 3:
        return []
    lo, hi = (0, 2) if axis == "horizontal" else (1, 3)
    ordered = sorted(items, key=lambda it: (it.bounds()[lo], it.bounds()[hi]))
    boxes = [it.bounds() for it in ordered]
    span = max(b[hi] for b in boxes) - boxes[0][lo]
    gap = (span - sum(b[hi] - b[lo] for b in boxes)) / (len(ordered) - 1)
    moved: list[Item] = []
    cursor = boxes[0][lo]
    for item, box in zip(ordered, boxes):
        delta = cursor - box[lo]
        cursor += (box[hi] - box[lo]) + gap
        if item.locked or abs(delta) < 1e-9:
            continue
        if axis == "horizontal":
            item.move_by(delta, 0.0)
        else:
            item.move_by(0.0, delta)
        moved.append(item)
    return moved


# ---------------------------------------------------------------- 면 / 문서

@dataclass(kw_only=True)
class Side(Serializable):
    orientation: str = "landscape"
    items: list[Item] = field(default_factory=list)
    background: str | None = None   # "#rrggbb" 또는 assets/ 상대 경로
    extra: dict = field(default_factory=dict, repr=False)

    def _encode(self, name: str, value: Any) -> Any:
        if name == "items":
            return [item.to_dict() for item in value]
        return super()._encode(name, value)

    @classmethod
    def _decode(cls, name: str, value: Any) -> Any:
        if name == "items":
            return [item_from_dict(d) for d in value]
        return super()._decode(name, value)

    def __iter__(self) -> Iterator[Item]:
        return iter(self.items)

    def get(self, item_id: str) -> Item | None:
        for item in self.items:
            if item.id == item_id:
                return item
        return None

    def select(self, ids: Iterable[str]) -> list[Item]:
        wanted = set(ids)
        return [item for item in self.items if item.id in wanted]

    def sorted_items(self) -> list[Item]:
        """아래에서 위 순서."""
        return [it for _, it in sorted(enumerate(self.items), key=lambda p: (p[1].z, p[0]))]

    def normalize_z(self) -> None:
        self._apply_order(self.sorted_items())

    def _apply_order(self, ordered: list[Item]) -> None:
        for index, item in enumerate(ordered):
            item.z = index
        self.items[:] = ordered

    def add_item(self, item: Item, *, on_top: bool = True) -> Item:
        if self.get(item.id) is not None:
            item.id = new_id()
        ordered = self.sorted_items()
        if on_top:
            ordered.append(item)
        else:
            ordered.insert(0, item)
        self._apply_order(ordered)
        return item

    def remove_item(self, item_id: str) -> Item | None:
        item = self.get(item_id)
        if item is None:
            return None
        self._apply_order([it for it in self.sorted_items() if it is not item])
        return item

    def remove_items(self, ids: Iterable[str]) -> list[Item]:
        return [it for it in (self.remove_item(i) for i in list(ids)) if it is not None]

    def duplicate_item(self, item_id: str, offset_mm: tuple[float, float] = (2.0, 2.0)) -> Item | None:
        """복제본을 원본 바로 위에 놓는다."""
        original = self.get(item_id)
        if original is None:
            return None
        clone = original.clone()
        clone.locked = False
        clone.move_by(*offset_mm)
        ordered = self.sorted_items()
        ordered.insert(ordered.index(original) + 1, clone)
        self._apply_order(ordered)
        return clone

    def bring_to_front(self, ids: Iterable[str]) -> None:
        wanted = set(ids)
        ordered = self.sorted_items()
        self._apply_order(
            [it for it in ordered if it.id not in wanted] + [it for it in ordered if it.id in wanted]
        )

    def send_to_back(self, ids: Iterable[str]) -> None:
        wanted = set(ids)
        ordered = self.sorted_items()
        self._apply_order(
            [it for it in ordered if it.id in wanted] + [it for it in ordered if it.id not in wanted]
        )

    def bring_forward(self, ids: Iterable[str]) -> None:
        wanted = set(ids)
        ordered = self.sorted_items()
        for i in range(len(ordered) - 2, -1, -1):
            if ordered[i].id in wanted and ordered[i + 1].id not in wanted:
                ordered[i], ordered[i + 1] = ordered[i + 1], ordered[i]
        self._apply_order(ordered)

    def send_backward(self, ids: Iterable[str]) -> None:
        wanted = set(ids)
        ordered = self.sorted_items()
        for i in range(1, len(ordered)):
            if ordered[i].id in wanted and ordered[i - 1].id not in wanted:
                ordered[i], ordered[i - 1] = ordered[i - 1], ordered[i]
        self._apply_order(ordered)

    def align(self, ids: Iterable[str], mode: str, reference: Bounds | None = None) -> list[Item]:
        return align_items(self.select(ids), mode, reference)

    def distribute(self, ids: Iterable[str], axis: str) -> list[Item]:
        return distribute_items(self.select(ids), axis)


def _default_sides() -> dict[str, Side]:
    return {"front": Side(), "back": Side()}


@dataclass(kw_only=True)
class Document(Serializable):
    """card_width_mm × card_height_mm는 가로 기준 캔버스 크기(도련 포함)."""

    version: int = CURRENT_VERSION
    name: str = "새 디자인"
    card_width_mm: float = CR80_WIDTH_MM
    card_height_mm: float = CR80_HEIGHT_MM
    sides: dict[str, Side] = field(default_factory=_default_sides)
    dual_side: bool = False
    bleed_mm: float = 0.0
    base_dir: str | None = field(default=None, compare=False, metadata={"transient": True})
    extra: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._floats_from_ints()

    def _encode(self, name: str, value: Any) -> Any:
        if name == "sides":
            return {key: side.to_dict() for key, side in value.items()}
        return super()._encode(name, value)

    @classmethod
    def _decode(cls, name: str, value: Any) -> Any:
        if name == "sides":
            return {str(key): Side.from_dict(side) for key, side in value.items()}
        return super()._decode(name, value)

    def to_dict(self) -> dict[str, Any]:
        return {"format": FORMAT_NAME, **super().to_dict()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Document":
        if not isinstance(data, Mapping):
            raise ValueError("문서 형식이 올바르지 않습니다.")
        payload = dict(data)
        fmt = payload.pop("format", FORMAT_NAME)
        if fmt != FORMAT_NAME:
            raise ValueError(f"하나스튜디오 디자인 문서가 아닙니다: {fmt!r}")
        version = payload.get("version", 1)
        if isinstance(version, bool) or not isinstance(version, int) or version < 1:
            raise ValueError(f"문서 버전이 올바르지 않습니다: {version!r}")
        document = super().from_dict(payload)
        # 더 새 버전의 문서는 버전 번호를 그대로 두어 다시 저장해도 낮아지지 않게 한다.
        document.version = max(version, CURRENT_VERSION)
        document.sides.setdefault("front", Side())
        return document

    @property
    def is_newer_format(self) -> bool:
        return self.version > CURRENT_VERSION

    def side(self, side_key: str) -> Side:
        try:
            return self.sides[side_key]
        except KeyError:
            raise KeyError(f"문서에 없는 면입니다: {side_key}") from None

    def ensure_side(self, side_key: str) -> Side:
        return self.sides.setdefault(side_key, Side())

    def active_side_keys(self) -> list[str]:
        keys = ["front"]
        if self.dual_side and "back" in self.sides:
            keys.append("back")
        return keys

    def side_size_mm(self, side_key: str) -> tuple[float, float]:
        """면의 (너비, 높이). portrait면 긴 변이 세로."""
        long_edge = max(self.card_width_mm, self.card_height_mm)
        short_edge = min(self.card_width_mm, self.card_height_mm)
        if self.side(side_key).orientation == "portrait":
            return (short_edge, long_edge)
        return (long_edge, short_edge)

    def side_bounds(self, side_key: str, *, inside_bleed: bool = False) -> Bounds:
        width, height = self.side_size_mm(side_key)
        margin = self.bleed_mm if inside_bleed else 0.0
        return (margin, margin, width - margin, height - margin)

    def iter_items(self) -> Iterator[tuple[str, Item]]:
        for key, side in self.sides.items():
            for item in side.items:
                yield key, item

    def find_item(self, item_id: str) -> tuple[str, Item] | None:
        for key, item in self.iter_items():
            if item.id == item_id:
                return key, item
        return None

    def align(self, side_key: str, ids: Iterable[str], mode: str, *, to_card: bool = False) -> list[Item]:
        """선택 항목 정렬. to_card=True거나 항목이 하나뿐이면 카드(칼선 안쪽) 기준."""
        side = self.side(side_key)
        items = side.select(ids)
        reference = None
        if to_card or len(items) == 1:
            reference = self.side_bounds(side_key, inside_bleed=True)
        return align_items(items, mode, reference)

    def distribute(self, side_key: str, ids: Iterable[str], axis: str) -> list[Item]:
        return self.side(side_key).distribute(ids, axis)

    def asset_refs(self) -> set[str]:
        refs: set[str] = set()
        for side in self.sides.values():
            if side.background and not side.background.startswith("#"):
                refs.add(side.background)
            for item in side.items:
                for attr in ("asset", "mask_asset"):
                    value = getattr(item, attr, None)
                    if value:
                        refs.add(value)
        return refs
