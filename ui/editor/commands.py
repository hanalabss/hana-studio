"""실행 취소 명령. 문서를 바꾸는 일은 모두 여기 명령을 QUndoStack에 넣어서 한다."""

from __future__ import annotations

import copy
from typing import Any, Iterable

from PySide6.QtGui import QUndoCommand

from design.model import ItemGeometry

_MERGE_ID = 0x4853  # 병합 후보끼리 같은 id를 쓰고, 실제 병합 여부는 mergeWith가 정한다.

# 레이어 목록에 보이는 속성. 바뀌면 목록을 다시 그려야 한다.
LISTED_ATTRS = frozenset({"name", "locked", "visible", "printable", "layer", "source", "text", "data"})


def _values_equal(a: Any, b: Any) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        try:
            return abs(float(a) - float(b)) < 1e-9
        except (TypeError, ValueError):
            return False
    return a == b


def order_of(side) -> list[str]:
    return [item.id for item in side.sorted_items()]


def apply_order(side, ids: Iterable[str]) -> None:
    by_id = {item.id: item for item in side.items}
    ordered = [by_id.pop(i) for i in ids if i in by_id]
    ordered += [item for item in side.items if item.id in by_id]
    for index, item in enumerate(ordered):
        item.z = index
    side.items[:] = ordered


class EditorCommand(QUndoCommand):
    """controller._after_command로 화면·선택을 맞춘다."""

    structural = False
    card_level = False

    def __init__(self, controller, text: str, side_key: str | None) -> None:
        super().__init__(text)
        self.controller = controller
        self.side_key = side_key
        self.select_before: list[str] = list(controller.selection)
        self.select_after: list[str] | None = None

    def _side(self):
        return self.controller.document.side(self.side_key)

    def affected_ids(self) -> list[str]:
        return []

    def changed_attrs(self) -> set[str]:
        return set()

    def _finish(self, undone: bool) -> None:
        selection = self.select_before if undone else self.select_after
        self.controller._after_command(self, selection)


class ItemStateCommand(EditorCommand):
    """항목 속성 값의 전/후를 기억한다. merge_token이 같은 연속 명령은 하나로 합쳐진다."""

    def __init__(
        self, controller, text: str, side_key: str, changes: dict[str, dict[str, Any]], *,
        merge_token: Any = None, select: bool = True,
    ) -> None:
        super().__init__(controller, text, side_key)
        side = self._side()
        self.merge_token = merge_token
        self.after = {i: {k: copy.deepcopy(v) for k, v in attrs.items()} for i, attrs in changes.items()}
        self.before = {
            i: {k: copy.deepcopy(getattr(side.get(i), k)) for k in attrs}
            for i, attrs in changes.items()
        }
        if select:
            self.select_after = list(changes)
        else:
            self.select_after = list(self.select_before)

    def affected_ids(self) -> list[str]:
        return list(self.after)

    def changed_attrs(self) -> set[str]:
        return {name for attrs in self.after.values() for name in attrs}

    def is_noop(self) -> bool:
        return all(
            _values_equal(self.before[i][k], v) for i, attrs in self.after.items() for k, v in attrs.items()
        )

    def _apply(self, state: dict[str, dict[str, Any]]) -> None:
        side = self._side()
        for item_id, attrs in state.items():
            item = side.get(item_id)
            if item is None:
                continue
            for name, value in attrs.items():
                setattr(item, name, copy.deepcopy(value))

    def redo(self) -> None:
        self._apply(self.after)
        self._finish(False)

    def undo(self) -> None:
        self._apply(self.before)
        self._finish(True)

    def id(self) -> int:
        return _MERGE_ID if self.merge_token is not None else -1

    def mergeWith(self, other: QUndoCommand) -> bool:
        if type(other) is not type(self) or other.merge_token != self.merge_token:
            return False
        if other.side_key != self.side_key or other.after.keys() != self.after.keys():
            return False
        if any(other.after[i].keys() != self.after[i].keys() for i in self.after):
            return False
        self.after = other.after
        self.select_after = other.select_after
        if self.is_noop():
            self.setObsolete(True)
        return True


class MoveCommand(ItemStateCommand):
    pass


class ResizeCommand(ItemStateCommand):
    pass


class RotateCommand(ItemStateCommand):
    pass


class PropertyCommand(ItemStateCommand):
    pass


class AlignCommand(ItemStateCommand):
    pass


class DistributeCommand(ItemStateCommand):
    pass


class AddItemsCommand(EditorCommand):
    structural = True

    def __init__(self, controller, text: str, side_key: str, items: list, order_after: list[str] | None = None):
        super().__init__(controller, text, side_key)
        self.items = list(items)
        self.order_before = order_of(self._side())
        self.order_after = order_after or self.order_before + [item.id for item in self.items]
        self.select_after = [item.id for item in self.items]

    def affected_ids(self) -> list[str]:
        return [item.id for item in self.items]

    def redo(self) -> None:
        side = self._side()
        present = {item.id for item in side.items}
        side.items.extend(item for item in self.items if item.id not in present)
        apply_order(side, self.order_after)
        self._finish(False)

    def undo(self) -> None:
        side = self._side()
        removed = {item.id for item in self.items}
        side.items[:] = [item for item in side.items if item.id not in removed]
        apply_order(side, self.order_before)
        self._finish(True)


class PasteCommand(AddItemsCommand):
    pass


class RemoveItemsCommand(EditorCommand):
    structural = True

    def __init__(self, controller, text: str, side_key: str, ids: Iterable[str]) -> None:
        super().__init__(controller, text, side_key)
        side = self._side()
        wanted = set(ids)
        self.items = [item for item in side.sorted_items() if item.id in wanted]
        self.order_before = order_of(side)
        self.select_before = [i for i in self.select_before] or [item.id for item in self.items]
        self.select_after = [i for i in controller.selection if i not in wanted]

    def affected_ids(self) -> list[str]:
        return [item.id for item in self.items]

    def redo(self) -> None:
        side = self._side()
        removed = {item.id for item in self.items}
        side.items[:] = [item for item in side.items if item.id not in removed]
        apply_order(side, [i for i in self.order_before if i not in removed])
        self._finish(False)

    def undo(self) -> None:
        side = self._side()
        present = {item.id for item in side.items}
        side.items.extend(item for item in self.items if item.id not in present)
        apply_order(side, self.order_before)
        self._finish(True)


class ZOrderCommand(EditorCommand):
    structural = True

    def __init__(self, controller, text: str, side_key: str, order_after: list[str], ids: Iterable[str] = ()):
        super().__init__(controller, text, side_key)
        self.order_before = order_of(self._side())
        self.order_after = list(order_after)
        self.ids = list(ids)
        self.select_after = self.ids or list(self.select_before)
        if self.ids:
            self.select_before = list(self.select_after)

    def affected_ids(self) -> list[str]:
        return list(self.ids)

    def is_noop(self) -> bool:
        return self.order_before == self.order_after

    def redo(self) -> None:
        apply_order(self._side(), self.order_after)
        self._finish(False)

    def undo(self) -> None:
        apply_order(self._side(), self.order_before)
        self._finish(True)


class SideBackgroundCommand(EditorCommand):
    structural = True

    def __init__(self, controller, text: str, side_key: str, background: str | None) -> None:
        super().__init__(controller, text, side_key)
        self.before = self._side().background
        self.after = background
        self.select_after = list(self.select_before)

    def redo(self) -> None:
        self._side().background = self.after
        self._finish(False)

    def undo(self) -> None:
        self._side().background = self.before
        self._finish(True)


class CardSetupCommand(EditorCommand):
    """카드 크기, 도련, 양면 여부, 이름, 면 방향. 그에 맞춰 옮긴 항목 위치도 함께 되돌린다."""

    structural = True
    card_level = True
    FIELDS = ("name", "card_width_mm", "card_height_mm", "bleed_mm", "dual_side")

    def __init__(
        self, controller, text: str, changes: dict[str, Any], orientations: dict[str, str] | None = None,
        geometry: dict[str, dict[str, ItemGeometry]] | None = None,
    ):
        """geometry: 면 → {항목 id: ItemGeometry}. 카드가 바뀐 뒤 항목이 놓일 자리."""
        super().__init__(controller, text, None)
        document = controller.document
        self.after = {k: v for k, v in changes.items() if k in self.FIELDS}
        self.before = {k: getattr(document, k) for k in self.after}
        self.orient_after = dict(orientations or {})
        self.orient_before = {k: document.side(k).orientation for k in self.orient_after}
        self.geometry_after = {key: dict(placed) for key, placed in (geometry or {}).items()}
        self.geometry_before = {
            key: {
                item_id: ItemGeometry.of(document.side(key).get(item_id))
                for item_id in placed if document.side(key).get(item_id) is not None
            }
            for key, placed in self.geometry_after.items()
        }
        self.select_after = list(self.select_before)

    def affected_ids(self) -> list[str]:
        return list(self.geometry_after.get(self.controller.current_side, {}))

    def changed_attrs(self) -> set[str]:
        return {"x", "y", "w", "h", "rotation"} if self.geometry_after else set()

    def is_noop(self) -> bool:
        return self.after == self.before and self.orient_after == self.orient_before

    def _apply(self, fields: dict[str, Any], orientations: dict[str, str], geometry: dict[str, dict]) -> None:
        document = self.controller.document
        for name, value in fields.items():
            setattr(document, name, value)
        for key, orientation in orientations.items():
            document.side(key).orientation = orientation
        for key, placed in geometry.items():
            side = document.side(key)
            for item_id, place in placed.items():
                item = side.get(item_id)
                if item is not None:
                    place.apply_to(item)

    def redo(self) -> None:
        self._apply(self.after, self.orient_after, self.geometry_after)
        self._finish(False)

    def undo(self) -> None:
        self._apply(self.before, self.orient_before, self.geometry_before)
        self._finish(True)
