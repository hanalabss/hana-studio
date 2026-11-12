"""속성 패널 구역의 공통 틀."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable

from PySide6.QtGui import QColor, QFontDatabase
from PySide6.QtWidgets import QVBoxLayout, QWidget

from design.model import Item
from ui.editor import EditorController, EditorError
from ui.theme import SWATCH_PRESETS
from ui.widgets import ComboBox

from ..common import EditSession

SECTION_SPACING = 14
_PRESET_COLORS = {name: color.upper() for name, color in SWATCH_PRESETS}
WHITE = _PRESET_COLORS["흰색"]
BLACK = _PRESET_COLORS["검정"]
# 패널 폭에 한 줄로 들어가도록 다섯 색만 쓴다. 나머지는 '다른 색'으로 고른다.
SWATCHES = tuple(entry for entry in SWATCH_PRESETS if entry[0] != "초록")


@dataclass
class InspectorContext:
    controller: EditorController
    workspace: Any
    session: EditSession
    go_to_list: Callable[[], None] = lambda: None
    pick_image: Callable[[str], str | None] = lambda title: None
    background_engine: Any = None
    # 지금 프린터가 지원하는 기능(set)을 돌려주는 함수. 프린터가 없으면 None을 돌려준다. 없으면 알 수 없음.
    capabilities: Callable[[], set[str] | None] | None = None
    state: dict = field(default_factory=dict)

    def columns(self) -> list[tuple[str, str]]:
        """열린 명단의 (열 이름, 종류). 명단이 없으면 빈 목록."""
        store = self.workspace.store
        if store is None:
            return []
        return [(column.name, column.type) for column in store.columns()]

    def has_list(self) -> bool:
        return self.workspace.store is not None


class Section(QWidget):
    """선택에 맞춰 보이는 구역 하나. load()로 모델 값을 채우고, 고칠 때는 edit()로 명령을 보낸다."""

    def __init__(self, context: InspectorContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.context = context
        self.controller = context.controller
        self.items: list[Item] = []
        self._loading = 0
        self.column = QVBoxLayout(self)
        self.column.setContentsMargins(0, 0, 0, 0)
        self.column.setSpacing(SECTION_SPACING)

    @property
    def item(self) -> Item | None:
        return self.items[0] if self.items else None

    @property
    def ids(self) -> list[str]:
        return [item.id for item in self.items]

    @contextmanager
    def loading(self):
        self._loading += 1
        try:
            yield
        finally:
            self._loading -= 1

    def set_items(self, items: list[Item]) -> None:
        self.items = list(items)
        self.reload()

    def reload(self) -> None:
        with self.loading():
            self.load()

    def load(self) -> None:
        raise NotImplementedError

    def detail_widgets(self) -> list[QWidget]:
        """'자세한 설정' 안에 함께 넣을 위젯."""
        return []

    def edit(self, values: dict[str, Any], merge: str | None = None, text: str = "속성 변경") -> bool:
        if self._loading or not self.items:
            return False
        key = self.context.session.key(merge) if merge else None
        try:
            return self.controller.set_properties(self.ids, values, key, text=text)
        except EditorError as exc:
            self.controller.message.emit(str(exc))
            return False

    def set_source(self, source) -> bool:
        if self._loading or not self.items:
            return False
        return self.controller.set_source(self.ids, source)


_FAMILIES: list[str] | None = None


def font_families() -> list[str]:
    """설치된 글꼴. 한글을 쓸 수 있는 것이 앞에 온다."""
    global _FAMILIES
    if _FAMILIES is None:
        everything = [name for name in QFontDatabase.families() if not name.startswith("@")]
        korean = [name for name in QFontDatabase.families(QFontDatabase.WritingSystem.Korean)
                  if not name.startswith("@")]
        rest = [name for name in everything if name not in set(korean)]
        _FAMILIES = sorted(korean, key=str.casefold) + sorted(rest, key=str.casefold)
    return list(_FAMILIES)


FONT_LABELS = {
    "Malgun Gothic": "맑은 고딕", "Batang": "바탕", "Gulim": "굴림", "Dotum": "돋움", "Gungsuh": "궁서",
    "NanumGothic": "나눔고딕", "NanumMyeongjo": "나눔명조", "Noto Sans KR": "본고딕 (Noto Sans KR)",
}


class FontCombo(ComboBox):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent=parent)
        self.setMaxVisibleItems(14)
        for family in font_families():
            self.addItem(FONT_LABELS.get(family, family), family)

    def family(self) -> str:
        return self.currentData() or ""

    def set_family(self, family: str) -> None:
        index = self.findData(family)
        if index < 0 and family:
            self.insertItem(0, f"{family} (이 컴퓨터에 없음)", family)
            index = 0
        self.setCurrentIndex(index)


def set_combo_data(combo, data) -> None:
    index = combo.findData(data)
    if index != combo.currentIndex():
        combo.setCurrentIndex(index)


def set_text_if_changed(edit, text: str) -> None:
    current = edit.toPlainText() if hasattr(edit, "toPlainText") else edit.text()
    if current != text:
        if hasattr(edit, "setPlainText"):
            edit.setPlainText(text)
        else:
            edit.setText(text)


def show_color(picker, color: str | None) -> None:
    """색을 고른 상태로 보여 준다. 준비된 색과 거의 같으면(순검정과 검정 등) 그 칸을 고른 것으로 친다."""
    if not color:
        return
    wanted = QColor(color)
    for preset in picker.colors():
        other = QColor(preset)
        if max(abs(wanted.red() - other.red()), abs(wanted.green() - other.green()),
               abs(wanted.blue() - other.blue())) <= 32:
            picker.set_current(preset, emit=False)
            return
    picker.set_current(color, emit=False)
