"""문서 + 명단 → 카드별 CardJob.

렌더링은 프린터가 그 카드를 요구할 때 한다. 500장 작업도 첫 장부터 바로 인쇄되고,
임시 파일은 카드 결과가 나오는 대로 지운다.
"""

from __future__ import annotations

import copy
import logging
import os
import shutil
import tempfile
import threading
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Mapping, Sequence

from PySide6.QtCore import QObject, QThread, QTimer, Signal

from data.photos import PhotoResolver
from data.store import DataStore, PhotoAdjustment
from design import renderer
from design.binding import BindingContext, ColumnSource, CounterSource, columns_used, to_text
from design.model import Crop, Document
from printer.base import (
    CAP_BLACK_RESIN, CAP_DUAL_SIDE, CAP_MAGNETIC, CAP_UV, CAP_WHITE, CardJob, PrintCancelled, PrinterError, SideImages,
)

log = logging.getLogger(__name__)

DEFAULT_DPI = 300
NAME_COLUMN_HINTS = ("이름", "성명", "name", "Name", "NAME")
CANCEL_MESSAGE = "인쇄가 취소되었습니다."
_EPS = 1e-9


class RenderJobError(Exception):
    """사용자에게 그대로 보여줄 수 있는 한국어 메시지를 담는다."""


# ---------------------------------------------------------------- 옵션

@dataclass
class PrintOptions:
    """인쇄 옵션. 프로젝트 파일에 그대로 저장된다(capabilities 제외).

    copies: 레코드마다(명단 없는 디자인은 전체) 인쇄할 장수.
    magnetic: {"track1": 열 이름, "track2": ..., "track3": ...}. 없으면 문서의 extra["magnetic"]을 쓴다.
    capabilities: 프린터가 지원하는 기능. None이면 장치 제품군의 기본값으로 판단한다.
    lookahead: 인쇄 중인 카드보다 미리 만들어 둘 카드 수.
    strict: 사진 누락·잘못된 바코드가 있는 카드를 인쇄하지 않고 실패로 처리한다.
    """

    copies: int = 1
    dpi: float = DEFAULT_DPI
    stop_on_error: bool = True
    offset_x_mm: float = 0.0
    offset_y_mm: float = 0.0
    brightness: int = 0          # 색 조정(-100~100, 0이 원래대로). 지원하는 프린터에서만 쓰인다.
    contrast: int = 0
    saturation: int = 0
    gamma: int = 0               # 중간 밝기(-100~100, 0이 원래대로, +가 밝게). RTAI에서만 쓰인다.
    density: int = 0             # 진하기(-100~100, 0이 원래대로, 클수록 진함). 전체·노랑·빨강·파랑.
    density_yellow: int = 0
    density_magenta: int = 0
    density_cyan: int = 0
    black_resin: bool = False    # 검정 글자·선·바코드·QR을 검정(K 레진) 패널로 찍는다. 지원하는 프린터에서만 쓰인다.
    magnetic: dict[str, str] | None = None
    name_column: str | None = None
    counter_offset: int = 0
    job_name: str = ""
    lookahead: int = 1
    strict: bool = False
    card_options: dict[str, Any] = field(default_factory=dict)
    capabilities: set[str] | None = None

    _SAVED = (
        "copies", "dpi", "stop_on_error", "offset_x_mm", "offset_y_mm",
        "brightness", "contrast", "saturation", "gamma",
        "density", "density_yellow", "density_magenta", "density_cyan", "black_resin", "magnetic",
        "name_column", "counter_offset", "job_name", "lookahead", "strict", "card_options",
    )

    def to_dict(self) -> dict[str, Any]:
        return {name: copy.deepcopy(getattr(self, name)) for name in self._SAVED}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "PrintOptions":
        options = cls()
        for name in cls._SAVED:
            if data and name in data:
                setattr(options, name, copy.deepcopy(data[name]))
        return options


# ---------------------------------------------------------------- 사진 조정값 변환

def _drawn_size(image_size, frame_size, fit: str, zoom: float) -> tuple[float, float]:
    _, _, draw_w, draw_h = renderer.image_target_rect(
        float(image_size[0]), float(image_size[1]), float(frame_size[0]), float(frame_size[1]),
        fit, Crop(zoom=zoom),
    )
    return draw_w, draw_h


def crop_from_adjustment(
    adjustment: PhotoAdjustment,
    image_size: Sequence[float],
    frame_size: Sequence[float],
    fit: str = "fill",
    *,
    clamp: bool = True,
) -> Crop:
    """명단의 사진 조정값(data.PhotoAdjustment)을 렌더러가 쓰는 design.Crop으로 바꾼다.

    두 값의 단위가 다르다.
      PhotoAdjustment.offset : 사진을 틀(frame) 크기의 몇 배만큼 옮기는가. +면 오른쪽·아래.
      Crop.offset            : 사진이 틀보다 남는 폭의 절반을 1로 본 비율(-1..1). +면 오른쪽·아래.
    zoom은 둘 다 "틀에 맞춘 크기 대비 배율"이라 그대로 옮긴다.

        이동 거리 = adjustment.offset × 틀 크기
        Crop.offset = 이동 거리 ÷ (|그려진 크기 − 틀 크기| ÷ 2)

    그려진 크기는 사진 비율에 따라 달라지므로 image_size(픽셀)와 frame_size(mm 등, 비율만 맞으면 됨)가 필요하다.
    정확히 옮길 수 없는 경우가 둘 있다.
      - 그 축에 남는 폭이 없으면(사진이 틀과 딱 맞음) Crop은 사진을 옮길 수 없다 → 0.
      - 남는 폭보다 더 옮기면 틀에 빈 곳이 생긴다. clamp=True면 ±1로 잘라 빈 곳이 생기지 않게 한다.
    """
    zoom = float(adjustment.zoom) if adjustment.zoom and adjustment.zoom > 0 else 1.0
    draw_w, draw_h = _drawn_size(image_size, frame_size, fit, zoom)

    def convert(offset: float, drawn: float, frame: float) -> float:
        slack = abs(drawn - frame) / 2.0
        if slack <= _EPS or frame <= 0:
            return 0.0
        value = float(offset) * frame / slack
        return max(-1.0, min(1.0, value)) if clamp else value

    return Crop(
        offset_x=convert(adjustment.offset_x, draw_w, float(frame_size[0])),
        offset_y=convert(adjustment.offset_y, draw_h, float(frame_size[1])),
        zoom=zoom,
    )


def adjustment_from_crop(
    crop: Crop, image_size: Sequence[float], frame_size: Sequence[float], fit: str = "fill"
) -> PhotoAdjustment:
    """crop_from_adjustment의 역변환."""
    zoom = float(crop.zoom) if crop.zoom and crop.zoom > 0 else 1.0
    draw_w, draw_h = _drawn_size(image_size, frame_size, fit, zoom)

    def convert(offset: float, drawn: float, frame: float) -> float:
        if frame <= 0:
            return 0.0
        return float(offset) * (abs(drawn - frame) / 2.0) / frame

    return PhotoAdjustment(
        zoom=zoom,
        offset_x=convert(crop.offset_x, draw_w, float(frame_size[0])),
        offset_y=convert(crop.offset_y, draw_h, float(frame_size[1])),
    )


# ---------------------------------------------------------------- 도련

def cut_pixel_size(document: Document, side_key: str, dpi: float = DEFAULT_DPI) -> tuple[int, int]:
    """칼선(도련 제외) 크기의 픽셀 수. design.renderer.cut_pixel_size와 같다."""
    return renderer.cut_pixel_size(document, side_key, dpi)


def trim_bleed(rendered: renderer.RenderedSide, document: Document, side_key: str) -> None:
    """도련을 가운데 기준으로 잘라 정확히 칼선 크기로 만든다. 도련이 없으면 그대로 둔다.

    render_side(crop_bleed=True)와 같은 결과다(둘 다 design.renderer.trim_bleed).
    """
    renderer.trim_bleed(rendered, document, side_key)


def driver_would_crop_again(document: Document, side_key: str, dpi: float = DEFAULT_DPI, *,
                            bleed_handled: bool = True) -> bool:
    """드라이버의 기존 도련 제거(utils.bleed_crop)가 완성된 이미지를 한 번 더 자를지.

    RenderJob이 만든 카드는 options["bleed_handled"]=True를 달고 나가고 RtaiAdapter·SmartAdapter가
    그 표시를 보고 도련 제거를 건너뛰므로 기본값으로는 항상 False다. bleed_handled=False는
    그 표시 없이 드라이버에 직접 넘길 때(기존 방식) 잘릴 크기인지 알려 준다.
    """
    if bleed_handled:
        return False
    try:
        from utils.bleed_crop import get_bleed_crop_box
    except Exception:
        return False
    width, height = cut_pixel_size(document, side_key, dpi)
    return get_bleed_crop_box(width, height) is not None


# ---------------------------------------------------------------- 카드 목록

@dataclass(frozen=True)
class CardSpec:
    """인쇄할 카드 한 장. render_key가 같은 카드는 같은 이미지를 쓴다(같은 레코드의 여러 장)."""

    record_id: int | None
    counter_index: int
    render_key: int
    display_name: str = ""


@dataclass
class RenderedCard:
    front: SideImages | None
    back: SideImages | None
    options: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def guess_name_column(store: DataStore | None, preferred: str | None = None) -> str | None:
    """진행 표시·안내 문구에 쓸 "사람 이름" 열."""
    if store is None:
        return None
    columns = store.columns()
    names = [c.name for c in columns]
    if preferred and preferred in names:
        return preferred
    for hint in NAME_COLUMN_HINTS:
        if hint in names:
            return hint
    for column in columns:
        if column.type == "text":
            return column.name
    return names[0] if names else None


def display_name(record: Mapping[str, Any] | None, name_column: str | None, fallback: str) -> str:
    if record is not None and name_column:
        text = to_text(record.get(name_column)).strip()
        if text:
            return text
    return fallback


def records_by_id(store: DataStore, record_ids: Iterable[int] | None = None) -> dict[int, dict[str, Any]]:
    """레코드를 한꺼번에 읽는다. record_ids가 None이면 전체."""
    if record_ids is None:
        return {record["_id"]: record for record in store.rows()}
    wanted = list(dict.fromkeys(int(i) for i in record_ids))
    if len(wanted) <= 20:
        found = ((record_id, store.get(record_id)) for record_id in wanted)
        return {record_id: record for record_id, record in found if record is not None}
    lookup = set(wanted)
    return {record["_id"]: record for record in store.rows() if record["_id"] in lookup}


def uses_counter(document: Document) -> bool:
    return any(isinstance(item.source, CounterSource) for _, item in document.iter_items())


def uses_white(document: Document) -> bool:
    return any(
        item.layer == "white" or getattr(item, "mask_asset", None)
        for item in _printable_items(document)
    )


def uses_uv(document: Document) -> bool:
    return any(item.layer == "uv" for item in _printable_items(document))


def uses_black_text(document: Document) -> bool:
    """검정(K 레진) 패널로 따로 찍을 수 있는 순수한 검정 글자·선·바코드·QR이 있는가."""
    return any(renderer.goes_to_black_layer(item) for item in _printable_items(document))


def prints_back(document: Document) -> bool:
    return "back" in document.active_side_keys()


def magnetic_mapping(document: Document, options: PrintOptions | None) -> dict[str, str]:
    mapping = (options.magnetic if options and options.magnetic else None) or document.extra.get("magnetic")
    if not isinstance(mapping, Mapping):
        return {}
    return {str(k): str(v) for k, v in mapping.items() if v}


def _printable_items(document: Document):
    for side_key in document.active_side_keys():
        for item in document.side(side_key).items:
            if item.visible and item.printable:
                yield item


def _job_temp_dir() -> str:
    parent = None
    try:
        from utils.safe_temp_path import get_cached_safe_temp_dir

        parent = get_cached_safe_temp_dir()
    except Exception:
        log.debug("안전한 임시 폴더를 얻지 못해 기본 위치를 씁니다.", exc_info=True)
    return tempfile.mkdtemp(prefix="hana_print_", dir=parent)


class RenderJob:
    """문서 사본과 카드 목록을 들고 있다가 요청받은 카드만 렌더링한다.

    문서는 만들 때 복사해 두므로 인쇄 중에 편집 화면에서 디자인을 고쳐도 영향이 없다.
    render()/release()/close()는 한 스레드(만든 스레드)에서만 부른다.
    """

    def __init__(
        self,
        document: Document,
        store: DataStore | None,
        specs: Iterable[CardSpec],
        options: PrintOptions | None = None,
        capabilities: Iterable[str] | None = None,
        *,
        snapshot: bool = True,
    ) -> None:
        self.document = copy.deepcopy(document) if snapshot else document
        self.store = store
        self.specs: list[CardSpec] = list(specs)
        self.options = options or PrintOptions()
        self.capabilities: set[str] | None = None if capabilities is None else set(capabilities)
        self.temp_dir: str | None = None
        self.render_count = 0
        self.max_live = 0
        self._cards: dict[int, RenderedCard] = {}
        self._pending_users: dict[int, int] = {}
        for spec in self.specs:
            self._pending_users[spec.render_key] = self._pending_users.get(spec.render_key, 0) + 1
        self._resolver: PhotoResolver | None = None
        self._closed = False

    # -- 만들기 ------------------------------------------------------------
    @classmethod
    def for_records(
        cls, document: Document, store: DataStore, record_ids: Iterable[int],
        options: PrintOptions | None = None, capabilities: Iterable[str] | None = None,
    ) -> "RenderJob":
        options = options or PrintOptions()
        copies = max(1, int(options.copies))
        name_column = guess_name_column(store, options.name_column)
        record_ids = [int(record_id) for record_id in record_ids]
        records = records_by_id(store, record_ids) if name_column else {}
        specs = []
        for position, record_id in enumerate(record_ids):
            record = records.get(record_id)
            name = display_name(record, name_column, f"{position + 1}번째 사람")
            for _ in range(copies):
                specs.append(CardSpec(int(record_id), options.counter_offset + position, position, name))
        return cls(document, store, specs, options, capabilities)

    @classmethod
    def for_copies(
        cls, document: Document, copies: int = 1,
        options: PrintOptions | None = None, capabilities: Iterable[str] | None = None,
    ) -> "RenderJob":
        """명단 없이 같은 디자인을 copies장. 일련번호 항목이 있으면 장마다 번호가 올라간다."""
        options = options or PrintOptions()
        copies = max(1, int(copies))
        distinct = uses_counter(document)
        specs = [
            CardSpec(None, options.counter_offset + i, i if distinct else 0, f"{i + 1}번째 카드")
            for i in range(copies)
        ]
        return cls(document, None, specs, options, capabilities)

    def subset(self, indices: Iterable[int]) -> "RenderJob":
        """일부 카드만 다시 인쇄할 작업. 일련번호는 처음 인쇄할 때와 같다."""
        keys: dict[int, int] = {}
        specs = []
        for index in indices:
            spec = self.specs[index]
            key = keys.setdefault(spec.render_key, len(keys))
            specs.append(CardSpec(spec.record_id, spec.counter_index, key, spec.display_name))
        return RenderJob(self.document, self.store, specs, self.options, self.capabilities, snapshot=False)

    # -- 조회 --------------------------------------------------------------
    @property
    def total(self) -> int:
        return len(self.specs)

    @property
    def live_count(self) -> int:
        """렌더링해 두고 아직 지우지 않은 카드 이미지 묶음 수."""
        return len(self._cards)

    def supports(self, capability: str) -> bool:
        return self.capabilities is None or capability in self.capabilities

    @property
    def is_dual_side(self) -> bool:
        return prints_back(self.document) and self.supports(CAP_DUAL_SIDE)

    def is_rendered(self, index: int) -> bool:
        return self.specs[index].render_key in self._cards

    # -- 렌더링 ------------------------------------------------------------
    def render(self, index: int) -> RenderedCard:
        if self._closed:
            raise RenderJobError(CANCEL_MESSAGE)
        spec = self.specs[index]
        card = self._cards.get(spec.render_key)
        if card is None:
            card = self._render(spec)
            self._cards[spec.render_key] = card
            self.render_count += 1
            self.max_live = max(self.max_live, len(self._cards))
        return card

    def card_job(self, index: int) -> CardJob:
        spec = self.specs[index]
        card = self.render(index)
        return CardJob(front=card.front, back=card.back, copies=1,
                       options=dict(card.options), record_id=spec.record_id)

    def __iter__(self) -> Iterator[CardJob]:
        """카드를 하나씩 렌더링해 내준다. 다음 카드를 꺼낼 때 앞 카드의 파일을 지운다."""
        for index in range(self.total):
            try:
                yield self.card_job(index)
            finally:
                self.release(index)

    def _ensure_temp_dir(self) -> str:
        if self.temp_dir is None:
            self.temp_dir = _job_temp_dir()
        return self.temp_dir

    def _binding_record(self, spec: CardSpec) -> dict[str, Any]:
        if spec.record_id is None or self.store is None:
            return {}
        record = self.store.get(spec.record_id)
        if record is None:
            raise RenderJobError(f"{spec.display_name}: 명단에서 지워진 사람이라 인쇄할 수 없습니다.")
        if self._resolver is None:
            self._resolver = PhotoResolver(self.store)
        return self._resolver.binding_record(record)

    def _apply_photo_adjustments(self, spec: CardSpec, record: Mapping[str, Any]) -> list:
        """명단에 저장된 사진 조정값을 문서 사본의 record_crops에 넣는다. 되돌릴 목록을 돌려준다."""
        applied = []
        if spec.record_id is None or self.store is None:
            return applied
        key = str(spec.record_id)
        default = PhotoAdjustment()
        for _, item in self.document.iter_items():
            if item.kind != "image" or not isinstance(item.source, ColumnSource):
                continue
            column = item.source.column
            if not column or not self.store.has_column(column):
                continue
            # 저장된 조정값이 없는 사진은 디자인의 기본 자르기(item.crop)를 그대로 쓴다.
            # (기본값과 같은 조정값을 일부러 저장해 둔 경우는 "조정 없음"과 결과가 같다.)
            adjustment = self.store.get_photo_adjustment(spec.record_id, column, default=None)
            if adjustment is None or adjustment == default:
                continue
            image = renderer.load_image(record.get(column) or "")
            if image is None:
                continue
            applied.append((item, item.record_crops.get(key)))
            item.set_record_crop(key, crop_from_adjustment(
                adjustment, (image.width(), image.height()), (item.w, item.h), item.fit))
        return applied

    def _magnetic(self, record: Mapping[str, Any]) -> dict[str, str]:
        if not self.supports(CAP_MAGNETIC):
            return {}
        tracks = {}
        for track, column in magnetic_mapping(self.document, self.options).items():
            text = to_text(record.get(column)).strip()
            if text:
                tracks[track] = text
        return tracks

    def _render(self, spec: CardSpec) -> RenderedCard:
        renderer.ensure_gui()
        document, options = self.document, self.options
        record = self._binding_record(spec)
        context = BindingContext(
            counter_index=spec.counter_index,
            record_key=None if spec.record_id is None else str(spec.record_id),
        )
        directory = self._ensure_temp_dir()
        card = RenderedCard(front=None, back=None)
        applied = self._apply_photo_adjustments(spec, record)
        try:
            for side_key in document.active_side_keys():
                if side_key == "back" and not self.supports(CAP_DUAL_SIDE):
                    continue
                try:
                    rendered = renderer.render_side(
                        document, side_key, record, options.dpi, context,
                        design_mode=False, crop_bleed=False, strict=options.strict,
                        black_layer=bool(options.black_resin) and self.supports(CAP_BLACK_RESIN),
                    )
                except renderer.RenderError as exc:
                    raise RenderJobError(f"{spec.display_name}: {exc}") from exc
                trim_bleed(rendered, document, side_key)
                if not self.supports(CAP_WHITE):
                    rendered.white_mask = None
                if not self.supports(CAP_UV):
                    rendered.uv = None
                paths = rendered.save(directory, f"card{spec.render_key:05d}_{side_key}")
                images = SideImages(
                    color_path=paths["color_path"], white_mask_path=paths["white_mask_path"],
                    uv_path=paths["uv_path"], black_path=paths["black_path"],
                    orientation=paths["orientation"],
                )
                if side_key == "front":
                    card.front = images
                else:
                    card.back = images
                card.warnings.extend(rendered.warnings)
        finally:
            key = str(spec.record_id)
            for item, previous in applied:
                item.set_record_crop(key, previous)

        card.options = dict(options.card_options)
        card.options["bleed_handled"] = True
        if options.offset_x_mm or options.offset_y_mm:
            card.options["offset_x_mm"] = float(options.offset_x_mm)
            card.options["offset_y_mm"] = float(options.offset_y_mm)
        for name in ("brightness", "contrast", "saturation", "gamma",
                     "density", "density_yellow", "density_magenta", "density_cyan"):
            if getattr(options, name):
                card.options[name] = int(getattr(options, name))
        magnetic = self._magnetic(record)
        if magnetic:
            card.options["magnetic"] = magnetic
        return card

    # -- 정리 --------------------------------------------------------------
    def release(self, index: int) -> None:
        """index번 카드가 끝났다. 같은 이미지를 쓸 카드가 더 없으면 파일을 지운다."""
        key = self.specs[index].render_key
        remaining = self._pending_users.get(key, 0) - 1
        self._pending_users[key] = remaining
        if remaining > 0:
            return
        card = self._cards.pop(key, None)
        if card is None:
            return
        for side in (card.front, card.back):
            if side is None:
                continue
            for path in (side.color_path, side.white_mask_path, side.uv_path, side.black_path):
                if path:
                    try:
                        os.remove(path)
                    except OSError:
                        log.debug("임시 파일을 지우지 못했습니다: %s", path)

    def close(self) -> None:
        self._closed = True
        self._cards.clear()
        if self.temp_dir and os.path.isdir(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        self.temp_dir = None


def needs_records(document: Document) -> bool:
    return bool(columns_used(document))


# ---------------------------------------------------------------- 인쇄 스레드와 잇기

class _Slot:
    __slots__ = ("event", "card", "error")

    def __init__(self) -> None:
        self.event = threading.Event()
        self.card: RenderedCard | None = None
        self.error: Exception | None = None


class FeederCards:
    """CardFeeder를 인쇄 큐의 카드 공급자(printer.base.CardProvider)로 내놓는 얇은 껍데기.

    큐는 길이만 읽고, 인쇄 스레드가 카드를 찍기 직전에 cards[index]로 한 장씩 받아 간다.
    그때 렌더링을 요청하고 끝날 때까지 기다린다. 렌더링 실패는 PrinterError(그 카드만 실패),
    취소는 PrintCancelled(그 카드는 인쇄하지 않고 작업을 멈춤)로 나간다.
    """

    def __init__(self, feeder: "CardFeeder") -> None:
        self._feeder = feeder

    def __len__(self) -> int:
        return self._feeder.job.total

    def __getitem__(self, index: int) -> CardJob:
        specs = self._feeder.job.specs
        if not 0 <= index < len(specs):
            raise IndexError(index)
        card = self._feeder.fetch(index)
        return CardJob(front=card.front, back=card.back, copies=1,
                       options=dict(card.options), record_id=specs[index].record_id)

    @property
    def is_dual_side(self) -> bool:
        return self._feeder.job.is_dual_side

    def record_id(self, index: int) -> Any:
        return self._feeder.job.specs[index].record_id


class CardFeeder(QObject):
    """인쇄 스레드의 요청을 받아 자기 스레드(보통 UI 스레드)에서 렌더링해 넘겨준다.

    렌더러의 이미지 캐시와 편집 화면이 같은 스레드를 쓰므로 렌더링을 인쇄 스레드에서 하지 않는다.
    동시에 살아 있는 카드 이미지는 많아야 1 + lookahead 묶음이다.
    """

    card_rendered = Signal(int)
    _requested = Signal(int)

    def __init__(self, job: RenderJob, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.job = job
        self.lookahead = max(0, int(job.options.lookahead))
        self._slots: dict[int, _Slot] = {}
        self._lock = threading.Lock()
        self._aborted = threading.Event()
        self._abort_message = CANCEL_MESSAGE
        self._released: set[int] = set()
        self._requested.connect(self._on_requested)

    def cards(self) -> FeederCards:
        """print_queue.add_card_job에 넘길 카드 공급자. 카드는 요청받을 때 렌더링한다."""
        return FeederCards(self)

    def _slot(self, index: int) -> _Slot:
        with self._lock:
            slot = self._slots.get(index)
            if slot is None:
                slot = self._slots[index] = _Slot()
            return slot

    # -- 인쇄 스레드 쪽 -----------------------------------------------------
    def fetch(self, index: int) -> RenderedCard:
        """index번 카드의 이미지. 준비될 때까지 기다린다.

        취소(abort)됐으면 PrintCancelled, 렌더링에 실패했으면 PrinterError.
        """
        if self._aborted.is_set():
            raise PrintCancelled(self._abort_message)
        slot = self._slot(index)
        if not slot.event.is_set():
            if QThread.currentThread() is self.thread():
                self._ensure(index)
            else:
                self._requested.emit(index)
                while not slot.event.wait(0.05):
                    if self._aborted.is_set():
                        break
        if self._aborted.is_set():
            raise PrintCancelled(self._abort_message)
        if slot.error is not None:
            raise PrinterError(str(slot.error))
        return slot.card

    # -- 렌더링 스레드 쪽 ---------------------------------------------------
    def _ensure(self, index: int) -> None:
        slot = self._slot(index)
        if slot.event.is_set() or index in self._released:
            return
        try:
            slot.card = self.job.render(index)
        except Exception as exc:
            if not isinstance(exc, RenderJobError):
                log.exception("카드 렌더링 실패 (index=%s)", index)
                exc = RenderJobError(f"카드 이미지를 만들지 못했습니다: {exc}")
            slot.error = exc
        slot.event.set()
        if slot.error is None:
            self.card_rendered.emit(index)

    def _on_requested(self, index: int) -> None:
        if self._aborted.is_set():
            return
        self._ensure(index)
        if self.lookahead:
            QTimer.singleShot(0, lambda: self._render_ahead(index))

    def _render_ahead(self, index: int) -> None:
        for ahead in range(index + 1, min(index + 1 + self.lookahead, self.job.total)):
            if self._aborted.is_set() or self.job.live_count >= 1 + self.lookahead:
                return
            self._ensure(ahead)

    def warnings(self, index: int) -> list[str]:
        """index번 카드를 만들면서 빠진 것(없는 사진 등)의 설명."""
        with self._lock:
            slot = self._slots.get(index)
        return list(slot.card.warnings) if slot is not None and slot.card is not None else []

    def release(self, index: int) -> None:
        if index in self._released:
            return
        self._released.add(index)
        self.job.release(index)
        with self._lock:
            self._slots.pop(index, None)

    def abort(self, message: str = CANCEL_MESSAGE) -> None:
        """아직 프린터로 넘어가지 않은 카드를 더 내주지 않는다."""
        self._abort_message = message
        self._aborted.set()

    @property
    def aborted(self) -> bool:
        return self._aborted.is_set()

    def close(self) -> None:
        self._aborted.set()
        self.job.close()
