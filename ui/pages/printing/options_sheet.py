"""인쇄 설정: 장수, 면, 위치 맞추기, 프린터가 할 수 있을 때만 보이는 화이트·UV·마그네틱."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLineEdit, QVBoxLayout, QWidget

from design.model import Document
from printer.base import (CAP_COLOR_ADJUST, CAP_DENSITY_ADJUST, CAP_DUAL_SIDE, CAP_GAMMA_ADJUST,
                          CAP_BLACK_RESIN, CAP_RIBBON_CHECKED, CAP_MAGNETIC, CAP_MIFARE, CAP_UV, CAP_WHITE, COLOR_ADJUST_MAX, COLOR_ADJUST_MIN, DENSITY_KEYS)

DENSITY_TITLES = {"density": "전체", "density_yellow": "노랑", "density_magenta": "빨강",
                  "density_cyan": "파랑"}
from printer.mifare import DEFAULT_BLOCK, DEFAULT_KEY, MifareError, check_block, check_key
from services.render_job import PrintOptions, magnetic_mapping, mifare_setting, uses_black_text, uses_uv, uses_white
from ui.pages.settings.prefs import COPIES_MAX, OFFSET_LIMIT_MM
from ui.theme import SPACE
from ui.widgets import (
    ComboBox, Dialog, Divider, FormRow, MmField, QuietButton, SegmentedControl, StepperField, label, set_prop,
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
    if options.black_resin and CAP_BLACK_RESIN in capabilities and uses_black_text(document):
        parts.append("검정 글자는 검정 리본으로")
    if magnetic_mapping(document, options) and CAP_MAGNETIC in capabilities:
        parts.append("마그네틱 기록")
    if mifare_setting(document, options) and CAP_MIFARE in capabilities:
        parts.append("IC 카드 기록")
    if options.offset_x_mm or options.offset_y_mm:
        parts.append(f"위치 가로 {options.offset_x_mm:+.1f} 세로 {options.offset_y_mm:+.1f}mm")
    gamma = options.gamma if CAP_GAMMA_ADJUST in capabilities else 0
    if CAP_COLOR_ADJUST in capabilities and (options.brightness or options.contrast or options.saturation
                                             or gamma):
        text = f"색 조정 밝기 {options.brightness:+d} 명암 {options.contrast:+d} 채도 {options.saturation:+d}"
        parts.append(text + (f" 중간 밝기 {gamma:+d}" if gamma else ""))
    if CAP_DENSITY_ADJUST in capabilities and any(getattr(options, key) for key in DENSITY_KEYS):
        parts.append("진하기 " + " ".join(f"{DENSITY_TITLES[key]} {getattr(options, key):+d}"
                                        for key in DENSITY_KEYS if getattr(options, key)))
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
        self.body.addWidget(_note(
            "그림이 한쪽으로 치우쳐 찍힐 때만 바꾸세요.",
            "+는 오른쪽·아래쪽, −는 왼쪽·위쪽으로 옮깁니다. 이 값은 다음 인쇄에도 그대로 씁니다."))
        position = QHBoxLayout()
        position.setSpacing(SPACE["lg"])
        self.offset_x = MmField(options.offset_x_mm, -OFFSET_LIMIT_MM, OFFSET_LIMIT_MM, 0.1, 1, follow_units=False)
        self.offset_y = MmField(options.offset_y_mm, -OFFSET_LIMIT_MM, OFFSET_LIMIT_MM, 0.1, 1, follow_units=False)
        position.addWidget(FormRow("가로 (오른쪽 +)", self.offset_x))
        position.addWidget(FormRow("세로 (아래쪽 +)", self.offset_y))
        self.reset_position = QuietButton("0으로 되돌리기", "sm", accent=True)
        self.reset_position.clicked.connect(self._reset_position)
        position.addWidget(self.reset_position, 0, Qt.AlignBottom)
        self.body.addLayout(position)

        # 색 조정: 프린터가 그림을 받을 때 밝기·명암·채도를 바꾼다(드라이버의 색 조정과 같은 일).
        self.color_fields: dict[str, StepperField] = {}
        if CAP_COLOR_ADJUST in capabilities:
            self.body.addWidget(Divider())
            self.body.addWidget(label("색 맞추기", "title"))
            self.body.addWidget(_note(
                "찍힌 색이 화면보다 어둡거나 밋밋할 때만 바꾸세요.",
                f"0이 원래대로이고 {COLOR_ADJUST_MIN}부터 +{COLOR_ADJUST_MAX}까지 넣을 수 있습니다. 화면의 미리 보기에는 "
                "나타나지 않고 인쇄에만 적용됩니다. 이 값은 다음 인쇄에도 그대로 씁니다."
                + (" '중간 밝기'는 흰색과 검정은 그대로 두고 그 사이의 색만 밝게(+)·어둡게(-) 합니다."
                   if CAP_GAMMA_ADJUST in capabilities else "")))
            colors = QHBoxLayout()
            colors.setSpacing(SPACE["lg"])
            titles = [("brightness", "밝기"), ("contrast", "명암"), ("saturation", "채도")]
            if CAP_GAMMA_ADJUST in capabilities:
                # 흰색·검정은 그대로 두고 중간 밝기만 바꾼다(드라이버의 Gamma).
                titles.append(("gamma", "중간 밝기"))
            for key, title in titles:
                field = StepperField(getattr(options, key), COLOR_ADJUST_MIN, COLOR_ADJUST_MAX, 5, 0)
                field.setAccessibleName(title)
                self.color_fields[key] = field
                colors.addWidget(FormRow(title, field))
            self.reset_color = QuietButton("0으로 되돌리기", "sm", accent=True)
            self.reset_color.clicked.connect(self._reset_color)
            colors.addWidget(self.reset_color, 0, Qt.AlignBottom)
            self.body.addLayout(colors)

        # 진하기: 색 리본을 얼마나 진하게 찍을지(SMART 드라이버의 Color Correction과 같은 값).
        self.density_fields: dict[str, StepperField] = {}
        if CAP_DENSITY_ADJUST in capabilities:
            self.body.addWidget(Divider())
            self.body.addWidget(label("진하기", "title"))
            self.body.addWidget(_note(
                "찍힌 색이 흐리거나 너무 진할 때만 바꾸세요.",
                f"0이 원래대로이고 {COLOR_ADJUST_MIN}부터 +{COLOR_ADJUST_MAX}까지, 클수록 진해집니다. '전체'는 모든 색을, "
                "노랑·빨강·파랑은 그 색만 바꿉니다. 화면의 미리 보기에는 나타나지 않고 인쇄에만 "
                "적용됩니다. 이 값은 다음 인쇄에도 그대로 씁니다."))
            densities = QHBoxLayout()
            densities.setSpacing(SPACE["lg"])
            for key in DENSITY_KEYS:
                field = StepperField(getattr(options, key), COLOR_ADJUST_MIN, COLOR_ADJUST_MAX, 5, 0)
                field.setAccessibleName(f"진하기 {DENSITY_TITLES[key]}")
                self.density_fields[key] = field
                densities.addWidget(FormRow(DENSITY_TITLES[key], field))
            self.reset_density = QuietButton("0으로 되돌리기", "sm", accent=True)
            self.reset_density.clicked.connect(self._reset_density)
            densities.addWidget(self.reset_density, 0, Qt.AlignBottom)
            self.body.addLayout(densities)

        self.white: QCheckBox | None = None
        self.uv: QCheckBox | None = None
        self.black_resin: QCheckBox | None = None
        self.magnetic: dict[str, ComboBox] = {}
        extras = QVBoxLayout()
        extras.setSpacing(SPACE["sm"])
        if CAP_BLACK_RESIN in capabilities and uses_black_text(document):
            self.black_resin = QCheckBox("검정 글자·바코드를 검정 리본으로 또렷하게 찍기")
            self.black_resin.setChecked(bool(options.black_resin))
            self.black_resin.setToolTip(
                "색이 완전한 검정인 글자·선·바코드·QR만 해당합니다. 색 리본 세 가지를 겹쳐 만든 검정보다 "
                "가장자리가 선명합니다. 사진과 색 글자는 그대로 색 리본으로 찍습니다.")
            extras.addWidget(self.black_resin)
            # 리본에 검정 패널이 있는지 프린터에서 확인하지 못했으면, 켰을 때만 한 줄 알린다(RTAI 등).
            self.black_resin_note = label("리본에 검정(K) 패널이 있을 때만 켜세요. 없으면 검정 글자가 제대로 찍히지 "
                                          "않을 수 있습니다.", "hint", wrap=True)
            set_prop(self.black_resin_note, "tone", "warn")
            self.black_resin_note.setContentsMargins(28, 0, 0, 0)
            unchecked = CAP_RIBBON_CHECKED not in capabilities
            self.black_resin_note.setVisible(unchecked and self.black_resin.isChecked())
            self.black_resin.toggled.connect(lambda on: self.black_resin_note.setVisible(unchecked and on))
            extras.addWidget(self.black_resin_note)
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
        self.mifare_column: ComboBox | None = None
        if CAP_MIFARE in capabilities and columns:
            current = mifare_setting(document, options)
            extras.addWidget(label("IC 카드(Mifare)에 넣을 명단 항목", "label"))
            self.mifare_column = ComboBox([_NONE, *columns])
            if current.get("column") in columns:
                self.mifare_column.setCurrentText(current["column"])
            self.mifare_block = StepperField(int(current.get("block", DEFAULT_BLOCK)), 1, 62, 1, 0, "번")
            self.mifare_block.setToolTip("값을 쓸 블록. 3, 7, 11…번은 키가 든 블록이라 쓸 수 없습니다")
            self.mifare_key_type = SegmentedControl([("A", "키 A"), ("B", "키 B")])
            self.mifare_key_type.set_current(str(current.get("key_type", "A")).upper(), emit=False)
            self.mifare_key = QLineEdit(str(current.get("key", DEFAULT_KEY)))
            self.mifare_key.setAccessibleName("IC 카드 키")
            self.mifare_key.setPlaceholderText(DEFAULT_KEY)
            self.mifare_format = ComboBox(["글자 그대로", "16진수"])
            self.mifare_format.setCurrentIndex(1 if current.get("format") == "hex" else 0)
            row = QHBoxLayout()
            row.setSpacing(SPACE["md"])
            row.addWidget(FormRow("항목", self.mifare_column), 2)
            row.addWidget(FormRow("블록", self.mifare_block), 1)
            row.addWidget(FormRow("값 형식", self.mifare_format), 1)
            extras.addLayout(row)
            keys = QHBoxLayout()
            keys.setSpacing(SPACE["md"])
            keys.addWidget(FormRow("키 종류", self.mifare_key_type), 1)
            keys.addWidget(FormRow("키 (16진수 12글자)", self.mifare_key), 2)
            extras.addLayout(keys)
            self.mifare_error = label("", "hint", wrap=True)
            self.mifare_error.setProperty("tone", "warn")
            self.mifare_error.setVisible(False)
            extras.addWidget(self.mifare_error)
            self.mifare_column.setToolTip("한 블록에 영문·숫자 16자(한글 5자 정도)까지 들어갑니다. 쓴 뒤 다시 읽어 "
                                          "확인하고, 다르면 그 카드는 인쇄하지 않습니다.")
        if extras.count():
            self.body.addWidget(Divider())
            self.body.addWidget(label("이 프린터로 할 수 있는 것", "title"))
            self.body.addLayout(extras)

        self.body.addWidget(Divider())
        self.stop_on_error = QCheckBox("한 장이 실패하면 남은 카드는 찍지 않고 멈추기")
        self.stop_on_error.setChecked(options.stop_on_error)
        self.stop_on_error.setToolTip('꺼 두면 실패한 카드를 건너뛰고 끝까지 찍습니다. 리본이 떨어지거나 카드가 걸렸을 때 카드를 여러 장 버리지 않도록 켜 두는 편이 안전합니다.')
        self.body.addWidget(self.stop_on_error)

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

    def _reset_density(self) -> None:
        for field in self.density_fields.values():
            field.set_value(0)

    def _reset_color(self) -> None:
        for field in self.color_fields.values():
            field.set_value(0)

    def choices(self) -> PrintChoices:
        result = self._choices.clone()
        options = result.options
        options.copies = int(self.copies.value())
        options.offset_x_mm = float(self.offset_x.value())
        options.offset_y_mm = float(self.offset_y.value())
        options.stop_on_error = self.stop_on_error.isChecked()
        for key, field in self.color_fields.items():
            setattr(options, key, int(field.value()))
        for key, field in self.density_fields.items():
            setattr(options, key, int(field.value()))
        if self.sides is not None and self.sides.isEnabled():
            result.front_only = self.sides.current() == "front"
        if self.white is not None:
            result.use_white = self.white.isChecked()
        if self.uv is not None:
            result.use_uv = self.uv.isChecked()
        if self.black_resin is not None:
            options.black_resin = self.black_resin.isChecked()
        if self.magnetic:
            mapping = {key: combo.currentText() for key, combo in self.magnetic.items()
                       if combo.currentIndex() > 0}
            # 문서에 정해 둔 줄을 "쓰지 않음"으로 바꾸려면 빈 값이 아니라 빈 묶음이어야 한다.
            options.magnetic = mapping or ({"track1": ""} if magnetic_mapping(self._document, None) else None)
        if self.mifare_column is not None:
            if self.mifare_column.currentIndex() > 0:
                options.mifare = {
                    "column": self.mifare_column.currentText(), "block": int(self.mifare_block.value()),
                    "key": self.mifare_key.text().replace(" ", "").upper() or DEFAULT_KEY,
                    "key_type": self.mifare_key_type.current(),
                    "format": "hex" if self.mifare_format.currentIndex() == 1 else "text",
                }
            else:
                options.mifare = {"column": ""} if mifare_setting(self._document, None) else None
        return result

    def mifare_problem(self) -> str:
        """IC 카드 설정이 잘못됐으면 그 까닭. 없으면 빈 글자."""
        if self.mifare_column is None or self.mifare_column.currentIndex() <= 0:
            return ""
        try:
            check_block(int(self.mifare_block.value()))
            check_key(self.mifare_key.text() or DEFAULT_KEY)
        except MifareError as exc:
            return str(exc)
        return ""

    def accept(self) -> None:
        problem = self.mifare_problem()
        if problem:
            self.mifare_error.setText(problem)
            self.mifare_error.setVisible(True)
            return
        super().accept()


def _note(short: str, more: str) -> QWidget:
    """짧은 한 줄만 보이고, 자세한 설명은 마우스를 올렸을 때 툴팁으로."""
    note = label(short, "muted", wrap=True)
    note.setToolTip(more)
    return note
