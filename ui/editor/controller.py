"""편집기 컨트롤러. 패널과 셸은 이 클래스만 호출한다."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, fields
from typing import Any, Iterable, Mapping

from PySide6.QtCore import QMimeData, QObject, Signal
from PySide6.QtGui import QGuiApplication, QImage, QUndoStack
from shiboken6 import isValid

from design import barcodes, presets, renderer, storage
from design.binding import BindingContext, ColumnSource, FixedSource, Source
from design.model import (
    ALIGN_MODES, DISTRIBUTE_AXES, ITEM_TYPES, LAYERS, ORIENTATIONS, Document, Item, ItemGeometry, Side,
    align_items, distribute_items, item_from_dict,
)
from ui.editor import commands
from ui.editor.scene import EditorScene

MIME_FIELD = "application/x-hanastudio-field"
MIME_ITEMS = "application/x-hanastudio-items"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff")

KIND_LABELS = {
    "text": "글자", "image": "사진", "rect": "사각형", "ellipse": "원", "line": "선",
    "barcode": "바코드", "qr": "QR 코드",
}
FIELD_KINDS = ("text", "image", "barcode", "qr")
_EDGE_EPS = 0.05
_PASTE_OFFSET_MM = 2.0
_CASCADE_MM = 3.0


class EditorError(Exception):
    """사용자에게 그대로 보여줄 수 있는 한국어 메시지를 담는다."""


@dataclass(frozen=True)
class LayerInfo:
    id: str
    name: str
    kind: str
    locked: bool
    visible: bool
    printable: bool
    layer: str
    bound: bool
    selected: bool
    warning: bool


@dataclass(frozen=True)
class Problem:
    side: str
    item_id: str | None
    item_name: str
    code: str      # outside_card | off_card | cut_line | missing_asset | empty_image | invalid_barcode
    message: str

    @property
    def text(self) -> str:
        return f"{self.item_name}: {self.message}" if self.item_name else self.message


def display_name(item: Item) -> str:
    if item.name:
        return item.name
    label = KIND_LABELS.get(item.kind, item.kind or "항목")
    if item.kind == "text" and item.source is None:
        snippet = " ".join(str(getattr(item, "text", "")).split())
        if snippet:
            return snippet if len(snippet) <= 16 else snippet[:15] + "…"
    return label


def bound_columns(item: Item) -> list[str]:
    return sorted(item.source.columns()) if item.source is not None else []


def geometry_warnings(document: Document, side_key: str, item: Item) -> list[tuple[str, str]]:
    """카드 밖으로 나갔거나 칼선에 걸친 항목의 (코드, 설명)."""
    width, height = document.side_size_mm(side_key)
    left, top, right, bottom = item.bounds()
    if right <= _EDGE_EPS or bottom <= _EDGE_EPS or left >= width - _EDGE_EPS or top >= height - _EDGE_EPS:
        return [("off_card", "카드 밖에 있어 인쇄되지 않습니다.")]
    bleed = document.bleed_mm
    if bleed <= 0:
        if left < -_EDGE_EPS or top < -_EDGE_EPS or right > width + _EDGE_EPS or bottom > height + _EDGE_EPS:
            return [("outside_card", "카드 밖으로 나간 부분은 인쇄되지 않습니다.")]
        return []
    # 도련 끝까지 닿은 것은 의도한 배경으로 보고, 도련 중간에서 끝나는 것만 알린다.
    crossings = (
        _EDGE_EPS < left < bleed - _EDGE_EPS,
        _EDGE_EPS < top < bleed - _EDGE_EPS,
        width - bleed + _EDGE_EPS < right < width - _EDGE_EPS,
        height - bleed + _EDGE_EPS < bottom < height - _EDGE_EPS,
    )
    if any(crossings):
        return [("cut_line", "재단선에 걸쳐 있어 잘릴 수 있습니다.")]
    return []


def _same_geometry(a: ItemGeometry, b: ItemGeometry) -> bool:
    return all(
        abs(p - q) < 1e-9 for p, q in ((a.x, b.x), (a.y, b.y), (a.w, b.w), (a.h, b.h), (a.rotation, b.rotation))
    )


class EditorController(QObject):
    document_changed = Signal()
    card_changed = Signal()
    side_changed = Signal(str)
    selection_changed = Signal(list)
    items_changed = Signal()
    properties_changed = Signal(list)
    modified_changed = Signal(bool)
    problems_changed = Signal()
    preview_changed = Signal()
    options_changed = Signal()
    message = Signal(str)
    bound_edit_requested = Signal(str)
    item_activated = Signal(str)

    def __init__(self, parent: QObject | None = None, document: Document | None = None) -> None:
        super().__init__(parent)
        renderer.ensure_gui()
        self.undo_stack = QUndoStack(self)
        self.undo_stack.cleanChanged.connect(self._on_clean_changed)
        self.scene = EditorScene(self)
        self._document: Document | None = None
        self._owns_workdir = False
        self._path: str | None = None
        self._side_key = "front"
        self._selection: list[str] = []
        self._warnings: dict[str, list[tuple[str, str]]] = {}
        self._record: Mapping[str, Any] | None = None
        self._context: BindingContext | None = None
        self._view = None
        self._paste_counts: dict[tuple[str, str], int] = {}
        self._grid_visible = False
        self._grid_step_mm = 1.0
        self._snap_enabled = True
        self._cut_line_visible = True
        self._field_tint_visible = True
        if document is not None:
            self.set_document(document)
        else:
            self.new_document()

    def _on_clean_changed(self, clean: bool) -> None:
        if isValid(self):
            self.modified_changed.emit(not clean)

    # ================================================================ 문서

    @property
    def document(self) -> Document:
        return self._document

    @property
    def path(self) -> str | None:
        return self._path

    @property
    def is_modified(self) -> bool:
        return not self.undo_stack.isClean()

    def new_document(
        self, preset: presets.CardPreset | str | None = None, template: str | None = None, *,
        orientation: str = "landscape", dual_side: bool = False, name: str | None = None,
    ) -> Document:
        if isinstance(preset, str):
            preset = presets.get_preset(preset)
        preset = preset or presets.CR80
        if template and template != "blank":
            document = presets.build_template(template, preset, orientation, name=name)
            document.dual_side = dual_side
        else:
            document = presets.new_document(preset, orientation, dual_side=dual_side, name=name or "새 디자인")
        document.base_dir = storage.new_workdir()
        self._install(document, None, True)
        return document

    def set_document(self, document: Document, path: str | None = None, *, owns_workdir: bool = False) -> None:
        """밖에서 만든 문서를 편집 대상으로 삼는다."""
        self._install(document, path, owns_workdir)

    def open(self, path: str | os.PathLike) -> Document:
        document = storage.load(path)
        self._install(document, os.fspath(path), True)
        return document

    def save(self, path: str | os.PathLike | None = None) -> str:
        target = os.fspath(path) if path is not None else self._path
        if not target:
            raise EditorError("저장할 파일 이름이 없습니다.")
        saved = storage.save(self._document, target)
        self._path = saved
        self.undo_stack.setClean()
        return saved

    def close(self) -> None:
        """문서가 쓰던 작업 폴더를 정리한다."""
        if self._document is not None and self._owns_workdir:
            storage.cleanup_workdir(self._document)
        self._owns_workdir = False

    def _install(self, document: Document, path: str | None, owns_workdir: bool) -> None:
        if self._document is not None and self._document is not document:
            self.close()
        was_modified = self.is_modified
        self._document = document
        self._owns_workdir = owns_workdir
        self._path = path
        document.ensure_side("front")
        self._side_key = "front"
        self._selection = []
        self._paste_counts.clear()
        self.undo_stack.clear()
        self.scene.set_document(document, "front")
        self.scene.set_preview(self._record, self._context)
        self._update_warnings(notify=False)
        self.document_changed.emit()
        self.card_changed.emit()
        self.side_changed.emit("front")
        self.selection_changed.emit([])
        self.items_changed.emit()
        self.problems_changed.emit()
        if was_modified:
            self.modified_changed.emit(False)

    def ensure_workdir(self) -> str:
        """문서의 그림 파일(assets/)을 두는 작업 폴더. 없으면 만들어서 돌려준다."""
        if not self._document.base_dir:
            self._document.base_dir = storage.new_workdir()
            self._owns_workdir = True
        return self._document.base_dir

    _ensure_workdir = ensure_workdir

    # ================================================================ 면

    @property
    def current_side(self) -> str:
        return self._side_key

    def side(self, side_key: str | None = None) -> Side:
        return self._document.side(side_key or self._side_key)

    def set_side(self, side_key: str) -> None:
        if side_key == self._side_key:
            return
        self._document.ensure_side(side_key)
        self._switch_side(side_key)
        self._set_selection([])

    def _switch_side(self, side_key: str) -> None:
        self._side_key = side_key
        self._selection = []
        self.scene.set_side(side_key)
        self._update_warnings(notify=False)
        self.side_changed.emit(side_key)
        self.selection_changed.emit([])
        self.items_changed.emit()
        self.problems_changed.emit()

    def side_size_mm(self, side_key: str | None = None) -> tuple[float, float]:
        return self._document.side_size_mm(side_key or self._side_key)

    # ================================================================ 선택

    @property
    def selection(self) -> list[str]:
        return list(self._selection)

    def selected_items(self) -> list[Item]:
        side = self.side()
        return [item for item in (side.get(i) for i in self._selection) if item is not None]

    def item(self, item_id: str) -> Item | None:
        return self.side().get(item_id)

    def select(self, ids: Iterable[str] | str | None, *, additive: bool = False) -> None:
        """레이어 목록에서 부르는 선택. 잠긴 항목도 고를 수 있다."""
        if ids is None:
            wanted: list[str] = []
        elif isinstance(ids, str):
            wanted = [ids]
        else:
            wanted = list(ids)
        if additive:
            wanted = self._selection + [i for i in wanted if i not in self._selection]
        self._set_selection(wanted)

    def toggle_selected(self, item_id: str) -> None:
        if item_id in self._selection:
            self._set_selection([i for i in self._selection if i != item_id])
        else:
            self._set_selection(self._selection + [item_id])

    def select_all(self) -> None:
        self._set_selection([it.id for it in self.side().sorted_items() if it.visible and not it.locked])

    def clear_selection(self) -> None:
        self._set_selection([])

    def _set_selection(self, ids: Iterable[str] | None) -> None:
        side = self.side()
        seen: set[str] = set()
        valid = []
        for item_id in ids or []:
            if item_id not in seen and side.get(item_id) is not None:
                seen.add(item_id)
                valid.append(item_id)
        if valid != self._selection:
            self._selection = valid
            self.selection_changed.emit(list(valid))
        self._refresh_view()

    def _ids(self, ids: Iterable[str] | str | None) -> list[str]:
        if ids is None:
            return list(self._selection)
        if isinstance(ids, str):
            return [ids]
        return list(ids)

    def _movable(self, ids: Iterable[str] | str | None) -> list[Item]:
        side = self.side()
        items = [side.get(i) for i in self._ids(ids)]
        return [item for item in items if item is not None and not item.locked]

    # ================================================================ 명령 실행

    def _push(self, command) -> bool:
        is_noop = getattr(command, "is_noop", None)
        if is_noop is not None and is_noop():
            return False
        self.undo_stack.push(command)
        return True

    def _after_command(self, command, selection: list[str] | None) -> None:
        if getattr(command, "card_level", False):
            self.scene.rebuild()
            self.card_changed.emit()
        else:
            if command.side_key != self._side_key:
                self._switch_side(command.side_key)
            if command.structural:
                self.scene.sync_structure()
            else:
                self.scene.sync_items(command.affected_ids(), command.changed_attrs())
        if selection is not None:
            self._set_selection(selection)
        self._update_warnings()
        ids = command.affected_ids()
        if ids:
            self.properties_changed.emit(ids)
        if command.structural or command.changed_attrs() & commands.LISTED_ATTRS:
            self.items_changed.emit()
        self._refresh_view()

    def _refresh_view(self) -> None:
        if self._view is not None:
            self._view.refresh_overlay()

    def refresh(self, ids: Iterable[str] | None = None) -> None:
        """명령을 거치지 않고 모델을 직접 고쳤을 때 화면을 다시 맞춘다."""
        self.scene.sync_structure()
        self._set_selection(self._selection)
        self._update_warnings()
        self.items_changed.emit()
        self.properties_changed.emit(list(ids) if ids is not None else [it.id for it in self.side().items])

    def new_merge_token(self) -> object:
        return object()

    def cancel_merged(self, merge_token: Any) -> bool:
        """merge_token으로 합쳐 온 마지막 명령을 되돌린다(끌다가 Esc를 눌렀을 때)."""
        index = self.undo_stack.index()
        if merge_token is None or index <= 0:
            return False
        command = self.undo_stack.command(index - 1)
        if getattr(command, "merge_token", None) is not merge_token:
            return False
        self.undo_stack.undo()
        return True

    # ================================================================ 항목 추가

    def add_item(self, kind: str, *, pos: tuple[float, float] | None = None, **defaults: Any) -> Item:
        """항목을 보이는 화면 가운데(또는 pos 중심)에 놓는다."""
        item = self._make_item(kind, defaults)
        self._place(item, pos)
        self._push(commands.AddItemsCommand(self, f"{KIND_LABELS[kind]} 추가", self._side_key, [item]))
        return item

    def add_image_from_file(self, path: str | os.PathLike, pos: tuple[float, float] | None = None) -> Item:
        source = os.fspath(path)
        image = renderer.load_image(source)
        if image is None:
            raise EditorError(f"이미지를 열 수 없습니다: {os.path.basename(source)}")
        try:
            asset = storage.import_asset(self.ensure_workdir(), source)
        except storage.StorageError as exc:
            raise EditorError(str(exc)) from exc
        left, top, right, bottom = self._document.side_bounds(self._side_key, inside_bleed=True)
        limit_w, limit_h = (right - left) * 0.6, (bottom - top) * 0.6
        scale = min(limit_w / image.width(), limit_h / image.height())
        item = self._make_item("image", {
            "asset": asset, "w": round(image.width() * scale, 2), "h": round(image.height() * scale, 2),
            "name": os.path.splitext(os.path.basename(source))[0][:40],
        })
        self._place(item, pos)
        self._push(commands.AddItemsCommand(self, "이미지 추가", self._side_key, [item]))
        return item

    def add_bound_field(self, column: str, kind: str = "text", pos: tuple[float, float] | None = None) -> Item:
        """명단의 열에 묶인 항목을 추가한다(열 칩을 카드에 끌어다 놓았을 때)."""
        if kind not in FIELD_KINDS:
            raise EditorError(f"명단 항목으로 넣을 수 없는 종류입니다: {kind}")
        if not column:
            raise EditorError("명단 항목 이름이 비어 있습니다.")
        defaults: dict[str, Any] = {"name": column, "source": ColumnSource(column=column)}
        if kind == "text":
            defaults["auto_shrink"] = True
        item = self._make_item(kind, defaults)
        self._place(item, pos)
        self._push(commands.AddItemsCommand(self, f"명단 항목 '{column}' 추가", self._side_key, [item]))
        return item

    def _make_item(self, kind: str, overrides: dict[str, Any]) -> Item:
        cls = ITEM_TYPES.get(kind)
        if cls is None:
            raise EditorError(f"알 수 없는 항목 종류입니다: {kind}")
        left, top, right, bottom = self._document.side_bounds(self._side_key, inside_bleed=True)
        area_w, area_h = right - left, bottom - top
        base: dict[str, Any] = {
            "text": {"text": "글자를 입력하세요", "font_size_pt": 14.0, "w": 46.0, "h": 8.0, "v_align": "middle"},
            "image": {"w": 24.0, "h": 30.0},
            "rect": {"w": 30.0, "h": 18.0, "fill": "#D9DEE7", "stroke_color": None},
            "ellipse": {"w": 20.0, "h": 20.0, "fill": "#D9DEE7", "stroke_color": None},
            "line": {"w": 30.0, "h": 0.0, "width_mm": 0.5},
            "barcode": {"w": 40.0, "h": 12.0, "data": "A1234567"},
            "qr": {"w": 18.0, "h": 18.0, "data": "HANA-0001"},
        }[kind]
        values = {**base, **overrides}
        valid = {f.name for f in fields(cls)}
        unknown = sorted(set(values) - valid)
        if unknown:
            raise EditorError(f"'{KIND_LABELS[kind]}'에 없는 속성입니다: {', '.join(unknown)}")
        item = cls(**values)
        if "w" not in overrides and "h" not in overrides:
            limit_w, limit_h = area_w * 0.9, area_h * 0.9
            if kind in ("image", "qr", "ellipse"):
                scale = min(1.0, limit_w / item.w, limit_h / item.h if item.h else 1.0)
                item.w, item.h = round(item.w * scale, 2), round(item.h * scale, 2)
            else:
                item.w, item.h = min(item.w, round(limit_w, 2)), min(item.h, round(limit_h, 2))
        return item

    def _place(self, item: Item, center: tuple[float, float] | None) -> None:
        left, top, right, bottom = self._document.side_bounds(self._side_key, inside_bleed=True)
        dropped = center is not None
        if center is None:
            center = self._view.visible_center_mm() if self._view is not None else None
            if center is None or not (left <= center[0] <= right and top <= center[1] <= bottom):
                center = ((left + right) / 2.0, (top + bottom) / 2.0)
        x, y = center[0] - item.w / 2.0, center[1] - item.h / 2.0
        x = min(max(x, left), max(left, right - item.w))
        y = min(max(y, top), max(top, bottom - item.h))
        if not dropped:
            taken = {(round(it.x, 1), round(it.y, 1)) for it in self.side().items}
            for _ in range(12):
                if (round(x, 1), round(y, 1)) not in taken:
                    break
                x, y = x + _CASCADE_MM, y + _CASCADE_MM
        item.x, item.y = round(x, 2), round(y, 2)

    # ================================================================ 속성

    def set_property(self, ids: Iterable[str] | str | None, name: str, value: Any, merge_key: Any = None) -> bool:
        """항목 속성 하나를 바꾼다. merge_key가 같은 연속 호출은 실행 취소 한 단계로 합쳐진다."""
        return self.set_properties(ids, {name: value}, merge_key)

    def set_properties(
        self, ids: Iterable[str] | str | None, values: Mapping[str, Any], merge_key: Any = None, *,
        text: str = "속성 변경",
    ) -> bool:
        """여러 속성을 한 번에 바꾼다(실행 취소 한 단계).

        "extra"에는 dict를 통째로 넘긴다. 모델에 없는 부가 값(배경 지우기 기준 등)이 여기 담겨
        파일에 함께 저장된다. 일부만 바꾸려면 {**item.extra, 키: 값}으로 넘긴다.
        """
        side = self.side()
        changes: dict[str, dict[str, Any]] = {}
        for item_id in self._ids(ids):
            item = side.get(item_id)
            if item is None:
                continue
            valid = {f.name for f in fields(item)} - {"id", "kind", "z"}
            attrs = {k: self._coerce(item, k, v) for k, v in values.items() if k in valid}
            if attrs:
                changes[item_id] = attrs
        if not changes:
            return False
        token = None if merge_key is None else ("property", merge_key)
        return self._push(commands.PropertyCommand(
            self, text, self._side_key, changes, merge_token=token, select=False,
        ))

    @staticmethod
    def _coerce(item: Item, name: str, value: Any) -> Any:
        if name == "extra":
            if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
                raise EditorError("부가 정보는 이름과 값의 묶음이어야 합니다.")
            return dict(value)
        current = getattr(item, name)
        if isinstance(current, float) and isinstance(value, int) and not isinstance(value, bool):
            return float(value)
        if name == "layer" and value not in LAYERS:
            raise EditorError(f"인쇄 층은 color, white, uv 중 하나여야 합니다: {value}")
        return value

    def set_source(self, ids: Iterable[str] | str | None, source: Source | None) -> bool:
        return self.set_properties(ids, {"source": source}, text="내용 연결 변경")

    def set_text(self, item_id: str, text: str) -> bool:
        """글자 항목의 고정 문구를 바꾼다."""
        item = self.item(item_id)
        if item is None or item.kind != "text":
            return False
        if isinstance(item.source, FixedSource):
            return self.set_properties(item_id, {"source": FixedSource(value=text)}, text="글자 수정")
        if item.source is not None:
            return False
        return self.set_properties(item_id, {"text": text}, text="글자 수정")

    def set_locked(self, ids: Iterable[str] | str | None, locked: bool) -> bool:
        return self.set_properties(ids, {"locked": bool(locked)}, text="잠금" if locked else "잠금 해제")

    def set_visible(self, ids: Iterable[str] | str | None, visible: bool) -> bool:
        return self.set_properties(ids, {"visible": bool(visible)}, text="보이기" if visible else "숨기기")

    def set_printable(self, ids: Iterable[str] | str | None, printable: bool) -> bool:
        return self.set_properties(ids, {"printable": bool(printable)}, text="인쇄 여부 변경")

    def set_layer(self, ids: Iterable[str] | str | None, layer: str) -> bool:
        return self.set_properties(ids, {"layer": layer}, text="인쇄 층 변경")

    def rename(self, item_id: str, name: str) -> bool:
        return self.set_properties(item_id, {"name": name}, text="이름 변경")

    # ================================================================ 위치 / 크기 / 회전

    def move_items_to(self, positions: Mapping[str, tuple[float, float]], merge_token: Any = None) -> bool:
        changes = {i: {"x": float(x), "y": float(y)} for i, (x, y) in positions.items() if self.item(i)}
        if not changes:
            return False
        return self._push(commands.MoveCommand(self, "이동", self._side_key, changes, merge_token=merge_token))

    def nudge(self, dx: float, dy: float, ids: Iterable[str] | None = None, merge_token: Any = None) -> bool:
        items = self._movable(ids)
        return self.move_items_to({it.id: (it.x + dx, it.y + dy) for it in items}, merge_token)

    def set_geometry(self, item_id: str, values: Mapping[str, Any], merge_token: Any = None) -> bool:
        """캔버스에서 크기를 끌어 바꿀 때 쓴다(x, y, w, h, flip, rotation)."""
        if self.item(item_id) is None:
            return False
        return self._push(commands.ResizeCommand(
            self, "크기 조절", self._side_key, {item_id: dict(values)}, merge_token=merge_token,
        ))

    def set_rotation(self, ids: Iterable[str] | str | None, degrees: float, merge_token: Any = None) -> bool:
        changes = {it.id: {"rotation": float(degrees) % 360.0} for it in self._movable(ids)}
        if not changes:
            return False
        return self._push(commands.RotateCommand(self, "회전", self._side_key, changes, merge_token=merge_token))

    # ================================================================ 정렬 / 순서

    def _simulate(self, ids: list[str]) -> list[Item]:
        return [item.clone(keep_id=True) for item in self.side().select(ids)]

    def _position_changes(self, clones: list[Item]) -> dict[str, dict[str, Any]]:
        side = self.side()
        changes = {}
        for clone in clones:
            original = side.get(clone.id)
            if abs(clone.x - original.x) > 1e-9 or abs(clone.y - original.y) > 1e-9:
                changes[clone.id] = {"x": clone.x, "y": clone.y}
        return changes

    def align(self, mode: str, ids: Iterable[str] | None = None, *, to_card: bool | None = None) -> bool:
        """left|center|right|top|middle|bottom. 하나만 골랐거나 to_card면 카드(칼선 안쪽) 기준."""
        if mode not in ALIGN_MODES:
            raise EditorError(f"알 수 없는 정렬 방식입니다: {mode}")
        clones = self._simulate(self._ids(ids))
        if not clones:
            return False
        reference = None
        if to_card or len(clones) == 1:
            reference = self._document.side_bounds(self._side_key, inside_bleed=True)
        align_items(clones, mode, reference)
        changes = self._position_changes(clones)
        if not changes:
            return False
        return self._push(commands.AlignCommand(self, "정렬", self._side_key, changes, select=False))

    def distribute(self, axis: str, ids: Iterable[str] | None = None) -> bool:
        """horizontal|vertical. 3개 이상 골랐을 때 사이 간격을 같게 한다."""
        if axis not in DISTRIBUTE_AXES:
            raise EditorError(f"알 수 없는 분배 방향입니다: {axis}")
        clones = self._simulate(self._ids(ids))
        if len(clones) < 3:
            self.message.emit("간격을 맞추려면 3개 이상 선택하세요.")
            return False
        distribute_items(clones, axis)
        changes = self._position_changes(clones)
        if not changes:
            return False
        return self._push(commands.DistributeCommand(self, "간격 맞춤", self._side_key, changes, select=False))

    def _reorder(self, operation: str, ids: Iterable[str] | None, text: str) -> bool:
        wanted = self._ids(ids)
        if not wanted:
            return False
        scratch = Side(items=[item.clone(keep_id=True) for item in self.side().sorted_items()])
        getattr(scratch, operation)(wanted)
        return self._push(commands.ZOrderCommand(self, text, self._side_key, commands.order_of(scratch), wanted))

    def bring_to_front(self, ids: Iterable[str] | None = None) -> bool:
        return self._reorder("bring_to_front", ids, "맨 앞으로")

    def send_to_back(self, ids: Iterable[str] | None = None) -> bool:
        return self._reorder("send_to_back", ids, "맨 뒤로")

    def bring_forward(self, ids: Iterable[str] | None = None) -> bool:
        return self._reorder("bring_forward", ids, "앞으로")

    def send_backward(self, ids: Iterable[str] | None = None) -> bool:
        return self._reorder("send_backward", ids, "뒤로")

    def set_layer_order(self, ids_top_to_bottom: Iterable[str]) -> bool:
        """레이어 목록에서 끌어 순서를 바꿨을 때. 위→아래 순서의 id 목록."""
        order = list(reversed(list(ids_top_to_bottom)))
        return self._push(commands.ZOrderCommand(self, "순서 변경", self._side_key, order))

    # ================================================================ 삭제 / 복제 / 클립보드

    def delete(self, ids: Iterable[str] | None = None) -> bool:
        targets = [item.id for item in self._movable(ids)]
        if not targets:
            if self._ids(ids):
                self.message.emit("잠긴 항목은 지울 수 없습니다. 먼저 잠금을 푸세요.")
            return False
        return self._push(commands.RemoveItemsCommand(self, "삭제", self._side_key, targets))

    def duplicate(self, ids: Iterable[str] | None = None) -> list[Item]:
        side = self.side()
        wanted = set(self._ids(ids))
        originals = [item for item in side.sorted_items() if item.id in wanted]
        if not originals:
            return []
        order = commands.order_of(side)
        clones = []
        for original in originals:
            clone = original.clone()
            clone.locked = False
            clone.move_by(_PASTE_OFFSET_MM, _PASTE_OFFSET_MM)
            order.insert(order.index(original.id) + 1, clone.id)
            clones.append(clone)
        self._push(commands.AddItemsCommand(self, "복제", self._side_key, clones, order))
        return clones

    def copy(self, ids: Iterable[str] | None = None) -> bool:
        return self._copy(ids, cut=False)

    def cut(self, ids: Iterable[str] | None = None) -> bool:
        targets = [item.id for item in self._movable(ids)]
        if not targets or not self._copy(targets, cut=True):
            return False
        return self._push(commands.RemoveItemsCommand(self, "잘라내기", self._side_key, targets))

    def _copy(self, ids: Iterable[str] | None, cut: bool) -> bool:
        wanted = set(self._ids(ids))
        items = [item for item in self.side().sorted_items() if item.id in wanted]
        if not items:
            return False
        serial = uuid.uuid4().hex
        payload = {
            "serial": serial, "side": self._side_key,
            "items": [item.to_dict() for item in items],
        }
        mime = QMimeData()
        mime.setData(MIME_ITEMS, json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        QGuiApplication.clipboard().setMimeData(mime)
        self._paste_counts = {(serial, self._side_key): 0 if cut else 1}
        return True

    def can_paste(self) -> bool:
        mime = QGuiApplication.clipboard().mimeData()
        return mime is not None and mime.hasFormat(MIME_ITEMS)

    def paste(self) -> list[Item]:
        mime = QGuiApplication.clipboard().mimeData()
        if mime is None or not mime.hasFormat(MIME_ITEMS):
            return []
        try:
            payload = json.loads(bytes(mime.data(MIME_ITEMS)).decode("utf-8"))
            items = [item_from_dict(data).clone() for data in payload["items"]]
            serial = str(payload.get("serial", ""))
        except (ValueError, KeyError, TypeError):
            self.message.emit("붙여넣을 수 없는 내용입니다.")
            return []
        if not items:
            return []
        key = (serial, self._side_key)
        count = self._paste_counts.get(key, 0)
        self._paste_counts[key] = count + 1
        for item in items:
            item.locked = False
            item.move_by(_PASTE_OFFSET_MM * count, _PASTE_OFFSET_MM * count)
        self._push(commands.PasteCommand(self, "붙여넣기", self._side_key, items))
        return items

    # ================================================================ 면 배경 / 카드 설정

    def set_side_background(self, background: str | None, side_key: str | None = None) -> bool:
        """"#rrggbb" 색 또는 None. 이미지 배경은 set_side_background_image를 쓴다."""
        side_key = side_key or self._side_key
        if self._document.side(side_key).background == background:
            return False
        return self._push(commands.SideBackgroundCommand(self, "배경 변경", side_key, background))

    def set_side_background_image(self, path: str | os.PathLike, side_key: str | None = None) -> bool:
        source = os.fspath(path)
        if renderer.load_image(source) is None:
            raise EditorError(f"이미지를 열 수 없습니다: {os.path.basename(source)}")
        try:
            asset = storage.import_asset(self.ensure_workdir(), source)
        except storage.StorageError as exc:
            raise EditorError(str(exc)) from exc
        return self.set_side_background(asset, side_key)

    def set_card_setup(
        self, *, preset: presets.CardPreset | str | None = None, width_mm: float | None = None,
        height_mm: float | None = None, bleed_mm: float | None = None, dual_side: bool | None = None,
        name: str | None = None, orientation: str | None = None, reflow: bool = True,
    ) -> bool:
        """카드 크기·도련·양면·이름·방향을 한 번에 바꾼다(실행 취소 한 단계).

        크기나 방향이 바뀌면 항목들도 새 카드 안으로 함께 옮긴다(같은 실행 취소 단계).
        reflow=False면 항목은 제자리에 둔다.
        """
        changes: dict[str, Any] = {}
        if isinstance(preset, str):
            preset = presets.get_preset(preset)
        if preset is not None:
            changes.update(card_width_mm=preset.width_mm, card_height_mm=preset.height_mm, bleed_mm=preset.bleed_mm)
        for key, value in (("card_width_mm", width_mm), ("card_height_mm", height_mm), ("bleed_mm", bleed_mm)):
            if value is not None:
                if value < 0 or (key != "bleed_mm" and value < 5):
                    raise EditorError("카드 크기가 올바르지 않습니다.")
                changes[key] = float(value)
        if dual_side is not None:
            changes["dual_side"] = bool(dual_side)
        if name is not None:
            changes["name"] = name
        orientations = None
        if orientation is not None:
            if orientation not in ORIENTATIONS:
                raise EditorError(f"알 수 없는 방향입니다: {orientation}")
            orientations = {key: orientation for key in self._document.sides}
        if changes.get("dual_side"):
            self._document.ensure_side("back")
        geometry = self._reflowed(changes, orientations or {}) if reflow else None
        return self._push(commands.CardSetupCommand(self, "카드 설정", changes, orientations, geometry))

    def _reflowed(
        self, changes: Mapping[str, Any], orientations: Mapping[str, str],
    ) -> dict[str, dict[str, ItemGeometry]]:
        """카드 크기·방향이 바뀐 뒤 항목들이 놓일 자리: 면 → {항목 id: 위치}. 달라지는 항목만 담는다."""
        document = self._document
        size = (
            changes.get("card_width_mm", document.card_width_mm),
            changes.get("card_height_mm", document.card_height_mm),
        )
        moved: dict[str, dict[str, ItemGeometry]] = {}
        for side_key, side in document.sides.items():
            if not side.items:
                continue
            placed = document.reflow_side(side_key, orientation=orientations.get(side_key), card_size_mm=size)
            changed = {
                item.id: placed[item.id] for item in side.items
                if item.id in placed and not _same_geometry(placed[item.id], ItemGeometry.of(item))
            }
            if changed:
                moved[side_key] = changed
        return moved

    def set_orientation(self, orientation: str, side_key: str | None = None, *, reflow: bool = True) -> bool:
        """landscape|portrait. side_key가 None이면 모든 면. 항목도 새 방향의 카드 안으로 옮긴다."""
        if orientation not in ORIENTATIONS:
            raise EditorError(f"알 수 없는 방향입니다: {orientation}")
        keys = [side_key] if side_key else list(self._document.sides)
        orientations = {key: orientation for key in keys}
        geometry = self._reflowed({}, orientations) if reflow else None
        return self._push(commands.CardSetupCommand(self, "카드 방향", {}, orientations, geometry))

    # ================================================================ 목록 / 문제

    def layers(self, side_key: str | None = None) -> list[LayerInfo]:
        """레이어 목록용. 위에 있는 항목부터."""
        side_key = side_key or self._side_key
        current = side_key == self._side_key
        return [
            LayerInfo(
                id=item.id, name=display_name(item), kind=item.kind, locked=item.locked,
                visible=item.visible, printable=item.printable, layer=item.layer,
                bound=bool(bound_columns(item)),
                selected=current and item.id in self._selection,
                warning=current and item.id in self._warnings,
            )
            for item in reversed(self._document.side(side_key).sorted_items())
        ]

    @property
    def warnings(self) -> dict[str, list[tuple[str, str]]]:
        """현재 면에서 카드 밖으로 나갔거나 칼선에 걸친 항목: id → [(코드, 설명)]."""
        return self._warnings

    def item_warnings(self, item_id: str) -> list[str]:
        return [message for _, message in self._warnings.get(item_id, [])]

    def _update_warnings(self, notify: bool = True) -> None:
        found = {}
        for item in self.side().items:
            if not item.visible:
                continue
            issues = geometry_warnings(self._document, self._side_key, item)
            if issues:
                found[item.id] = issues
        if found != self._warnings:
            self._warnings = found
            if notify:
                self.problems_changed.emit()

    def problems(self) -> list[Problem]:
        """인쇄 전에 알려줄 문제 목록(카드 밖, 없는 이미지, 잘못된 바코드)."""
        document = self._document
        context = self._context or BindingContext()
        found: list[Problem] = []
        for side_key in document.active_side_keys():
            side = document.side(side_key)
            if side.background and not side.background.startswith("#"):
                path = renderer.asset_path(document, side.background)
                if not path or not os.path.isfile(path):
                    found.append(Problem(side_key, None, "배경", "missing_asset", "배경 이미지 파일을 찾을 수 없습니다."))
            for item in side.sorted_items():
                if not item.visible or not item.printable:
                    continue
                name = display_name(item)
                for code, message in geometry_warnings(document, side_key, item):
                    found.append(Problem(side_key, item.id, name, code, message))
                for code, message in self._content_problems(item, context):
                    found.append(Problem(side_key, item.id, name, code, message))
        return found

    def _content_problems(self, item: Item, context: BindingContext) -> list[tuple[str, str]]:
        document = self._document
        issues: list[tuple[str, str]] = []
        if item.kind == "image":
            uses_columns = bool(bound_columns(item))
            value = renderer.value_for(document, item, self._record, context)
            if uses_columns and self._record is None:
                pass
            elif not value:
                issues.append((
                    "empty_image",
                    "이 사람의 사진이 없습니다." if uses_columns else "넣을 이미지를 아직 고르지 않았습니다.",
                ))
            elif not os.path.isfile(value):
                issues.append(("missing_asset", f"이미지 파일을 찾을 수 없습니다: {os.path.basename(str(value))}"))
            mask = renderer.asset_path(document, getattr(item, "mask_asset", None))
            if mask and not os.path.isfile(mask):
                issues.append(("missing_asset", "배경 제거 마스크 파일을 찾을 수 없습니다."))
        elif item.kind in ("barcode", "qr"):
            value = renderer.value_for(document, item, self._record, context)
            data = "" if value is None else str(value)
            if data.startswith("«") and self._record is None:
                return issues
            try:
                if item.kind == "barcode":
                    barcodes.normalize(item.symbology, data)
                else:
                    barcodes.qr_matrix(data, item.error_correction)
            except barcodes.BarcodeError as exc:
                issues.append(("invalid_barcode", str(exc)))
        return issues

    # ================================================================ 미리 보기

    @property
    def preview_record(self) -> Mapping[str, Any] | None:
        return self._record

    def set_preview_record(self, record: Mapping[str, Any] | None, context: BindingContext | None = None) -> None:
        """명단의 한 사람을 캔버스에 미리 보여준다. None이면 «열» 자리표시로 돌아간다."""
        self._record = dict(record) if record is not None else None
        self._context = context
        self.scene.set_preview(self._record, context)
        self.preview_changed.emit()
        self.problems_changed.emit()
        self._refresh_view()

    def export_preview_image(self, side: str | None = None, dpi: float = 150.0) -> QImage:
        """지금 미리 보기 중인 레코드로 한 면을 그린 그림(도련 포함)."""
        rendered = renderer.render_side(
            self._document, side or self._side_key, self._record, dpi=dpi, context=self._context,
            design_mode=True,
        )
        rgb = rendered.color.convert("RGB")
        data = rgb.tobytes("raw", "RGB")
        image = QImage(data, rgb.width, rgb.height, rgb.width * 3, QImage.Format.Format_RGB888)
        return image.copy()

    # ================================================================ 보기 옵션

    def attach_view(self, view) -> None:
        self._view = view

    @property
    def view(self):
        return self._view

    @property
    def grid_visible(self) -> bool:
        return self._grid_visible

    def set_grid_visible(self, visible: bool) -> None:
        self._set_option("_grid_visible", bool(visible))

    @property
    def grid_step_mm(self) -> float:
        return self._grid_step_mm

    def set_grid_step(self, step_mm: float) -> None:
        if step_mm <= 0:
            raise EditorError("모눈 간격은 0보다 커야 합니다.")
        self._set_option("_grid_step_mm", float(step_mm))

    @property
    def snap_enabled(self) -> bool:
        return self._snap_enabled

    def set_snap_enabled(self, enabled: bool) -> None:
        self._set_option("_snap_enabled", bool(enabled))

    @property
    def cut_line_visible(self) -> bool:
        return self._cut_line_visible

    def set_cut_line_visible(self, visible: bool) -> None:
        self._set_option("_cut_line_visible", bool(visible))

    @property
    def field_tint_visible(self) -> bool:
        """명단에서 가져오는 항목을 캔버스에서 옅은 노랑으로 알려 줄지. 인쇄와는 상관없다."""
        return self._field_tint_visible

    def set_field_tint_visible(self, visible: bool) -> None:
        self._set_option("_field_tint_visible", bool(visible))

    def _set_option(self, attr: str, value: Any) -> None:
        if getattr(self, attr) != value:
            setattr(self, attr, value)
            self.options_changed.emit()
            self._refresh_view()

    def zoom_fit(self) -> None:
        if self._view is not None:
            self._view.zoom_fit()

    def zoom_actual(self) -> None:
        if self._view is not None:
            self._view.zoom_actual()

    def zoom_to_selection(self) -> None:
        if self._view is not None:
            self._view.zoom_to_selection()


def field_mime(column: str, kind: str = "text") -> QMimeData:
    """열 칩을 끌 때 쓸 QMimeData."""
    mime = QMimeData()
    mime.setData(MIME_FIELD, json.dumps({"column": column, "kind": kind}, ensure_ascii=False).encode("utf-8"))
    return mime


def parse_field_mime(mime: QMimeData) -> tuple[str, str] | None:
    if mime is None or not mime.hasFormat(MIME_FIELD):
        return None
    try:
        payload = json.loads(bytes(mime.data(MIME_FIELD)).decode("utf-8"))
        column = str(payload["column"])
    except (ValueError, KeyError, TypeError):
        return None
    kind = str(payload.get("kind") or "text")
    if not column or kind not in FIELD_KINDS:
        return None
    return column, kind


def image_paths_from_mime(mime: QMimeData) -> list[str]:
    if mime is None or not mime.hasUrls():
        return []
    paths = [url.toLocalFile() for url in mime.urls() if url.isLocalFile()]
    return [p for p in paths if p.lower().endswith(IMAGE_EXTENSIONS)]
