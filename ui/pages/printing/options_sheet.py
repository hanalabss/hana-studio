"""인쇄 설정: 장수, 면, 위치 맞추기, 프린터가 할 수 있을 때만 보이는 화이트·UV·마그네틱."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QVBoxLayout, QWidget

from design.model import Document
from printer.base import CAP_DUAL_SIDE, CAP_MAGNETIC, CAP_UV, CAP_WHITE
from services.render_job import PrintOptions, magnetic_mapping, uses_uv, uses_white
from ui.pages.settings.prefs import COPIES_MAX, OFFSET_LIMIT_MM
from ui.theme import SPACE
from ui.widgets import (
    ComboBox, Dialog, Divider, FormRow, MmField, QuietButton, SegmentedControl, StepperField, label,
)

from .style import apply_page_style

TRACKS = (("track1", "1번 줄"), ("track2", "2번 줄"), ("track3", "3번 줄"))
_NONE = "쓰지 않음"


@dataclass
class PrintChoices:
    """인쇄 화면이 들고 있는 설정. options는 서비스에 그대로 넘긴다."""

    options: PrintOptions = field(default_factory=PrintOptions)
    front_only: bool = False
    use_white: bool = True
    use_uv: bool = True

    def clone(self) -> "PrintChoices":
        return PrintChoices(copy.deepcopy(self.options), self.front_only, self.use_white, self.use_uv)

    def document_for(self, document: Document) -> Document:
        """앞면만 뽑기로 했으면 뒷면을 끈 사본(면 내용은 함께 쓴다)."""
        if self.front_only and document.dual_side:
            clone = copy.copy(document)
            clone.dual_side = False
            return clone
        return document

    def capabilities_for(self, capabilities: set[str]) -> set[str]:
        caps = set(capabilities)
        if not self.use_white:
            caps.discard(CAP_WHITE)
        if not self.use_uv:
            caps.discard(CAP_UV)
        return caps


def summary_text(choices: PrintChoices, document: Document, capabilities: set[str], people: bool) -> str:
    """"한 사람에 2장, 양면, 화이트 포함" 같은 한 줄."""
    options = choices.options
    parts: list[str] = []
    if people:
        parts.append(f"한 사람에 {options.copies}장")
    if document.dual_side:
        both = not choices.front_only and CAP_DUAL_SIDE in capabilities
        parts.append("양면" if both else "앞면만")
    else:
        parts.append("한 면")
    if uses_white(document):
        parts.append("화이트 포함" if choices.use_white and CAP_WHITE in capabilities else "화이트 빼고")
    if uses_uv(document):
        parts.append("UV 포함" if choices.use_uv and CAP_UV in capabilities else "UV 빼고")
    if magnetic_mapping(document, options) and CAP_MAGNETIC in capabilities:
        parts.append("마그네틱 기록")
    if options.offset_x_mm or options.offset_y_mm:
        parts.append(f"위치 가로 {options.offset_x_mm:+.1f} 세로 {options.offset_y_mm:+.1f}mm")
    if not options.stop_on_error:
        parts.append("실패해도 계속")
    return ", ".join(parts)


class OptionsSheet(Dialog):
    def __init__(self, choices: PrintChoices, document: Document, capabilities: set[str],
                 columns: list[str], people: bool = True, parent: QWidget | None = None) -> None:
        super().__init__("인쇄 설정", None, parent, width=520)
        apply_page_style(self)
        self._choices = choices.clone()
        self._document = document
        options = self._choices.options
        self.body.setSpacing(SPACE["md"])

        self.copies = StepperField(options.copies, 1, COPIES_MAX, 1, 0, "장")
        self.copies.setFixedWidth(200)
        self.copies.setAccessibleName("한 사람에 몇 장" if people else "뽑을 장수")
        self.body.addWidget(FormRow("한 사람에 몇 장" if people else "몇 장 뽑을까요", self.copies,
                                    orientation="horizontal", label_width=132, control_align="left"))

        self.sides: SegmentedControl | None = None
        if document.dual_side:
            self.sides = SegmentedControl([("both", "앞·뒤 양면"), ("front", "앞면만")],
                                          current="front" if self._choices.front_only else "both")
            hint = None
            if CAP_DUAL_SIDE not in capabilities:
                self.sides.set_current("front", emit=False)
                self.sides.setEnabled(False)
                hint = "지금 고른 프린터는 한 면만 찍을 수 있어 앞면만 뽑습니다."
            self.sides.setAccessibleName("찍을 면")
            self.body.addWidget(FormRow("찍을 면", self.sides, hint, orientation="horizontal",
                                        label_width=132, control_align="left"))

        self.body.addWidget(Divider())
        self.body.addWidget(label("찍히는 위치 맞추기", "title"))
        self.body.addWidget(label(
            "카드에 찍힌 그림이 한쪽으로 치우쳐 나올 때만 바꾸세요. +는 오른쪽·아래쪽, −는 왼쪽·위쪽으로 "
            "옮깁니다. 이 값은 다음 인쇄에도 그대로 씁니다.", "muted", wrap=True))
        position = QHBoxLayout()
        position.setSpacing(SPACE["lg"])
        self.offset_x = MmField(options.offset_x_mm, -OFFSET_LIMIT_MM, OFFSET_LIMIT_MM, 0.1, 1)
        self.offset_y = MmField(options.offset_y_mm, -OFFSET_LIMIT_MM, OFFSET_LIMIT_MM, 0.1, 1)
        position.addWidget(FormRow("가로 (오른쪽 +)", self.offset_x))
        position.addWidget(FormRow("세로 (아래쪽 +)", self.offset_y))
        self.reset_position = QuietButton("0으로 되돌리기", "sm", accent=True)
        self.reset_position.clicked.connect(self._reset_position)
        position.addWidget(self.reset_position, 0, Qt.AlignBottom)
        self.body.addLayout(position)

        self.white: QCheckBox | None = None
        self.uv: QCheckBox | None = None
        self.magnetic: dict[str, ComboBox] = {}
        extras = QVBoxLayout()
        extras.setSpacing(SPACE["sm"])
        if CAP_WHITE in capabilities and uses_white(document):
            self.white = QCheckBox("화이트로 꾸민 부분도 찍기")
            self.white.setChecked(self._choices.use_white)
            extras.addWidget(self.white)
        if CAP_UV in capabilities and uses_uv(document):
            self.uv = QCheckBox("UV(빛을 비추면 보이는 잉크)로 꾸민 부분도 찍기")
            self.uv.setChecked(self._choices.use_uv)
            extras.addWidget(self.uv)
        if CAP_MAGNETIC in capabilities and columns:
            current = magnetic_mapping(document, options)
            extras.addWidget(label("카드 뒷면 마그네틱 띠에 넣을 명단 항목", "label"))
            tracks = QHBoxLayout()
            tracks.setSpacing(SPACE["md"])
            for key, title in TRACKS:
                combo = ComboBox([_NONE, *columns])
                if current.get(key) in columns:
                    combo.setCurrentText(current[key])
                self.magnetic[key] = combo
                tracks.addWidget(FormRow(title, combo), 1)
            extras.addLayout(tracks)
        if extras.count():
            self.body.addWidget(Divider())
            self.body.addWidget(label("이 프린터로 할 수 있는 것", "title"))
            self.body.addLayout(extras)

        self.body.addWidget(Divider())
        self.stop_on_error = QCheckBox("한 장이 실패하면 남은 카드는 찍지 않고 멈추기")
        self.stop_on_error.setChecked(options.stop_on_error)
        self.body.addWidget(self.stop_on_error)
        stop_hint = label("꺼 두면 실패한 카드를 건너뛰고 끝까지 찍습니다. 리본이 떨어지면 카드를 여러 장 "
                          "버릴 수 있으니 켜 두는 편이 안전합니다.", "hint", wrap=True)
        stop_hint.setContentsMargins(28, 0, 0, 0)
        self.body.addWidget(stop_hint)

        self.add_button("그만두기", "secondary", "reject")
        self.apply_button = self.add_button("이 설정으로 하기", "primary", "accept", default=True)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.fit_height()

    def fit_height(self) -> None:
        """줄바꿈된 설명까지 다 보이는 높이로 맞춘다(화면보다 커지지는 않게)."""
        width = max(self.width(), self.minimumWidth())
        height = self.layout().totalHeightForWidth(width)
        screen = self.screen().availableGeometry().height() if self.screen() is not None else 0
        if screen:
            height = min(height, screen - 60)
        if height > self.height():
            self.resize(width, height)

    def _reset_position(self) -> None:
        self.offset_x.set_value(0.0)
        self.offset_y.set_value(0.0)

    def choices(self) -> PrintChoices:
        result = self._choices.clone()
        options = result.options
        options.copies = int(self.copies.value())
        options.offset_x_mm = float(self.offset_x.value())
        options.offset_y_mm = float(self.offset_y.value())
        options.stop_on_error = self.stop_on_error.isChecked()
        if self.sides is not None and self.sides.isEnabled():
            result.front_only = self.sides.current() == "front"
        if self.white is not None:
            result.use_white = self.white.isChecked()
        if self.uv is not None:
            result.use_uv = self.uv.isChecked()
        if self.magnetic:
            mapping = {key: combo.currentText() for key, combo in self.magnetic.items()
                       if combo.currentIndex() > 0}
            # 문서에 정해 둔 줄을 "쓰지 않음"으로 바꾸려면 빈 값이 아니라 빈 묶음이어야 한다.
            options.magnetic = mapping or ({"track1": ""} if magnetic_mapping(self._document, None) else None)
        return result
