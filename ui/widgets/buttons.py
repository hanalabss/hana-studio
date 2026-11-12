"""버튼: 주, 보조, 조용한, 위험, 아이콘 전용."""

from __future__ import annotations

from PySide6.QtCore import QSize
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import QPushButton, QWidget

from ui.theme import SIZES

from ._util import FocusRingMixin, on_theme_changed, set_prop
from .icons import themed_icon

VARIANTS = ("primary", "secondary", "quiet", "danger")
BUTTON_SIZES = ("sm", "md", "lg")
_HEIGHT = {"sm": SIZES["control_sm"], "md": SIZES["control_md"], "lg": SIZES["control_lg"]}
_RADIUS = {"sm": 6, "md": 8, "lg": 10}
_ICON = {"sm": 16, "md": 18, "lg": 20}
_ICON_ROLE = {"primary": "primary_text", "danger": "danger_on", "secondary": "text", "quiet": "text"}
# 속이 찬 버튼은 가장자리 링이 묻히므로 글자색 링을 안쪽에 그린다.
_INSET_RING = {"primary": "primary_text", "danger": "danger_on"}


class Button(FocusRingMixin, QPushButton):
    def __init__(self, text: str = "", variant: str = "secondary", size: str = "md",
                 icon: str | None = None, parent: QWidget | None = None):
        super().__init__(text, parent)
        if variant not in VARIANTS:
            raise ValueError(f"variant는 {VARIANTS} 중 하나여야 합니다: {variant}")
        if size not in BUTTON_SIZES:
            raise ValueError(f"size는 {BUTTON_SIZES} 중 하나여야 합니다: {size}")
        self._variant = variant
        self._size = size
        self._icon_name = icon
        self._accent = False
        self.setProperty("variant", variant)
        self.setProperty("sizing", size)
        self.setFixedHeight(_HEIGHT[size])
        self.setAutoDefault(False)
        self._refresh_icon()
        on_theme_changed(self._refresh_icon)

    def variant(self) -> str:
        return self._variant

    def set_variant(self, variant: str) -> None:
        if variant not in VARIANTS:
            raise ValueError(variant)
        self._variant = variant
        set_prop(self, "variant", variant)
        self._refresh_icon()

    def set_accent(self, accent: bool) -> None:
        """조용한 버튼을 링크처럼 강조색 글자로 보이게 한다."""
        self._accent = accent
        set_prop(self, "accent", "true" if accent else "false")
        self._refresh_icon()

    def icon_name(self) -> str | None:
        return self._icon_name

    def set_icon_name(self, name: str | None) -> None:
        self._icon_name = name
        self._refresh_icon()

    def _icon_role(self) -> str:
        if self._accent and self._variant == "quiet":
            return "accent_text"
        return _ICON_ROLE[self._variant]

    def _refresh_icon(self, *_args) -> None:
        if not self._icon_name:
            return
        px = _ICON[self._size]
        self.setIcon(themed_icon(self._icon_name, self._icon_role(), px))
        self.setIconSize(QSize(px, px))

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if self.shows_focus_ring():
            painter = QPainter(self)
            radius = _RADIUS[self._size]
            if self._variant in _INSET_RING:
                self.paint_focus_ring(painter, radius, _INSET_RING[self._variant], inset=2.0)
            else:
                self.paint_focus_ring(painter, radius)
            painter.end()


class PrimaryButton(Button):
    def __init__(self, text: str = "", size: str = "md", icon: str | None = None, parent: QWidget | None = None):
        super().__init__(text, "primary", size, icon, parent)


class SecondaryButton(Button):
    def __init__(self, text: str = "", size: str = "md", icon: str | None = None, parent: QWidget | None = None):
        super().__init__(text, "secondary", size, icon, parent)


class QuietButton(Button):
    def __init__(self, text: str = "", size: str = "md", icon: str | None = None, parent: QWidget | None = None,
                 accent: bool = False):
        super().__init__(text, "quiet", size, icon, parent)
        if accent:
            self.set_accent(True)


class DangerButton(Button):
    def __init__(self, text: str = "", size: str = "md", icon: str | None = None, parent: QWidget | None = None):
        super().__init__(text, "danger", size, icon, parent)


class IconButton(Button):
    """글자 없는 정사각형 버튼. tooltip이 읽기 프로그램용 이름도 된다."""

    def __init__(self, icon: str, tooltip: str, size: str = "md", variant: str = "quiet",
                 parent: QWidget | None = None):
        super().__init__("", variant, size, icon, parent)
        self.setProperty("iconOnly", "true")
        self.setFixedWidth(_HEIGHT[size])
        self.setToolTip(tooltip)
        self.setAccessibleName(tooltip)
