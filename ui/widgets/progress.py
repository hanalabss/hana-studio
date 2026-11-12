"""진행 표시: 가는 막대와 리본 패널별 진행."""

from __future__ import annotations

from typing import Sequence

from PySide6.QtCore import QRectF, QSize, Qt
from PySide6.QtGui import QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QProgressBar, QSizePolicy, QWidget

from ui.theme import FONT_SIZES, ThemeManager

from ._util import theme_color

# 리본 패널: key -> (보이는 이름, 색 역할)
RIBBON_PANELS: dict[str, tuple[str, str]] = {
    "y": ("노랑", "ribbon_y"),
    "m": ("빨강", "ribbon_m"),
    "c": ("파랑", "ribbon_c"),
    "k": ("검정", "ribbon_k"),
    "w": ("화이트", "ribbon_w"),
    "o": ("코팅", "ribbon_w"),
    "uv": ("UV", "ribbon_w"),
}


class ProgressBar(QProgressBar):
    """0~100 가는 막대. set_fraction(0.0~1.0)도 쓸 수 있다."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setRange(0, 100)
        self.setTextVisible(False)
        self.setFixedHeight(8)

    def set_fraction(self, fraction: float) -> None:
        self.setValue(round(max(0.0, min(1.0, fraction)) * 100))


class RibbonProgress(QWidget):
    """카드 한 장을 찍는 동안 리본 패널이 차례로 채워지는 모습.

    set_progress(2, 0.55): 앞의 두 패널은 끝, 세 번째가 55%, 나머지는 아직.
    """

    _BAR = 10
    _GAP = 8
    _LABEL_GAP = 6

    def __init__(self, panels: Sequence[str] = ("y", "m", "c", "k", "w"), parent: QWidget | None = None):
        super().__init__(parent)
        unknown = [p for p in panels if p not in RIBBON_PANELS]
        if unknown:
            raise ValueError(f"모르는 리본 패널: {unknown}")
        self._panels = tuple(panels)
        self._fractions = [0.0] * len(self._panels)
        self._current = -1
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMinimumWidth(len(self._panels) * 44)
        self._describe()

    def sizeHint(self) -> QSize:
        return QSize(len(self._panels) * 64, self._BAR + self._LABEL_GAP + FONT_SIZES["caption"] + 6)

    def minimumSizeHint(self) -> QSize:
        return QSize(len(self._panels) * 44, self.sizeHint().height())

    def panels(self) -> tuple[str, ...]:
        return self._panels

    def fractions(self) -> list[float]:
        return list(self._fractions)

    def current_index(self) -> int:
        """지금 채워지는 패널 번호. 시작 전이면 -1, 다 끝났으면 패널 수."""
        return self._current

    def set_progress(self, index: int, fraction: float = 0.0) -> None:
        index = max(0, min(len(self._panels), index))
        fraction = max(0.0, min(1.0, fraction))
        for i in range(len(self._panels)):
            self._fractions[i] = 1.0 if i < index else (fraction if i == index else 0.0)
        self._current = index
        self._describe()
        self.update()

    def set_panel(self, key: str, fraction: float) -> None:
        self.set_progress(self._panels.index(key), fraction)

    def reset(self) -> None:
        self._fractions = [0.0] * len(self._panels)
        self._current = -1
        self._describe()
        self.update()

    def finish(self) -> None:
        self.set_progress(len(self._panels), 0.0)

    def _describe(self) -> None:
        if self._current < 0:
            text = "리본 인쇄 대기"
        elif self._current >= len(self._panels):
            text = "리본 인쇄 끝"
        else:
            name = RIBBON_PANELS[self._panels[self._current]][0]
            text = f"{name} 인쇄 중 {round(self._fractions[self._current] * 100)}%"
        self.setAccessibleName(text)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        count = len(self._panels)
        width = (self.width() - self._GAP * (count - 1)) / count
        radius = self._BAR / 2
        spec = ThemeManager.instance().font_spec
        for i, key in enumerate(self._panels):
            name, role = RIBBON_PANELS[key]
            x = i * (width + self._GAP)
            bar = QRectF(x, 0.5, width, self._BAR)
            clip = QPainterPath()
            clip.addRoundedRect(bar, radius, radius)
            painter.setPen(Qt.NoPen)
            painter.setBrush(theme_color("ribbon_track"))
            painter.drawPath(clip)
            fraction = self._fractions[i]
            if fraction > 0:
                painter.save()
                painter.setClipPath(clip)
                painter.setBrush(theme_color(role))
                painter.drawRect(QRectF(x, 0.5, width * fraction, self._BAR))
                painter.restore()
                if role in ("ribbon_w", "ribbon_k"):
                    painter.setPen(QPen(theme_color("ribbon_edge"), 1))
                    painter.setBrush(Qt.NoBrush)
                    painter.drawPath(clip)
            is_current = i == self._current and self._current < count
            font = QFont(self.font())
            font.setPixelSize(FONT_SIZES["caption"])
            font.setWeight(QFont.Weight(spec.strong_weight) if is_current else QFont.Weight.Normal)
            painter.setFont(font)
            painter.setPen(theme_color("text" if is_current else "text_muted"))
            text_rect = QRectF(x, self._BAR + self._LABEL_GAP, width, FONT_SIZES["caption"] + 6)
            painter.drawText(text_rect, Qt.AlignLeft | Qt.AlignVCenter, name)
        painter.end()
