"""카드 크기 프리셋과 시작용 템플릿(코드로 직접 만든 것)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

from design.binding import ColumnSource, ConcatPart, ConcatSource, CounterSource, DateSource
from design.model import (
    BarcodeItem, Document, ImageItem, LineItem, QRItem, RectItem, Side, TextItem,
)

DEFAULT_BLEED_MM = 2.0


@dataclass(frozen=True)
class CardPreset:
    """width_mm × height_mm는 가로 기준(긴 변 × 짧은 변), 도련 포함 캔버스 크기."""

    key: str
    label: str
    width_mm: float
    height_mm: float
    bleed_mm: float = 0.0

    @property
    def cut_size_mm(self) -> tuple[float, float]:
        return (self.width_mm - 2 * self.bleed_mm, self.height_mm - 2 * self.bleed_mm)


def with_bleed(preset: CardPreset, bleed_mm: float = DEFAULT_BLEED_MM) -> CardPreset:
    """칼선 크기 프리셋에 사방 도련을 더한 프리셋."""
    cut_w, cut_h = preset.cut_size_mm
    return replace(
        preset,
        key=f"{preset.key.removesuffix('_bleed')}_bleed",
        label=f"{cut_h:g} × {cut_w:g} mm + 도련 {bleed_mm:g} mm",
        width_mm=cut_w + 2 * bleed_mm, height_mm=cut_h + 2 * bleed_mm, bleed_mm=bleed_mm,
    )


CR80 = CardPreset("cr80", "CR80 표준 카드 (85.6 × 53.98 mm)", 85.6, 53.98)
# 기존 앱 규격: 칼선 54×86, 사방 도련 2mm를 더한 디자인 58×90 (config.json printer.design_*_mm, bleed_mm)
CUT_54X86 = CardPreset("cut_54x86", "54 × 86 mm (칼선 크기)", 86.0, 54.0)
CUT_54X86_BLEED = CardPreset("cut_54x86_bleed", "54 × 86 mm + 도련 2 mm (58 × 90 mm)", 90.0, 58.0, 2.0)

PRESETS: tuple[CardPreset, ...] = (CR80, CUT_54X86, CUT_54X86_BLEED)


def get_preset(key: str) -> CardPreset:
    for preset in PRESETS:
        if preset.key == key:
            return preset
    raise KeyError(f"없는 카드 크기입니다: {key}")


def new_document(
    preset: CardPreset = CR80, orientation: str = "landscape", *,
    dual_side: bool = False, name: str = "새 디자인",
) -> Document:
    return Document(
        name=name, card_width_mm=preset.width_mm, card_height_mm=preset.height_mm,
        bleed_mm=preset.bleed_mm, dual_side=dual_side,
        sides={"front": Side(orientation=orientation), "back": Side(orientation=orientation)},
    )


# ---------------------------------------------------------------- 시작 템플릿

class _Layout:
    """칼선 안쪽 영역을 0..1 비율로 다루는 좌표 도우미."""

    def __init__(self, document: Document, side_key: str = "front") -> None:
        self.side = document.side(side_key)
        left, top, right, bottom = document.side_bounds(side_key, inside_bleed=True)
        self.left, self.top = left, top
        self.width, self.height = right - left, bottom - top
        self.full = document.side_bounds(side_key)

    def box(self, fx: float, fy: float, fw: float, fh: float) -> dict[str, float]:
        return {
            "x": round(self.left + fx * self.width, 3), "y": round(self.top + fy * self.height, 3),
            "w": round(fw * self.width, 3), "h": round(fh * self.height, 3),
        }

    def band(self, fy: float, fh: float) -> dict[str, float]:
        """도련까지 덮는 가로 띠."""
        left, top, right, bottom = self.full
        y0 = top if fy <= 0 else self.top + fy * self.height
        y1 = bottom if fy + fh >= 1 else self.top + (fy + fh) * self.height
        return {"x": left, "y": round(y0, 3), "w": right - left, "h": round(y1 - y0, 3)}

    def pt(self, fraction_of_height: float) -> float:
        """영역 높이에 비례한 글자 크기(pt), 0.5pt 단위."""
        return max(5.0, round(fraction_of_height * self.height * 72.0 / 25.4 * 2) / 2)

    def add(self, item):
        return self.side.add_item(item)


def _blank(document: Document) -> None:
    return None


def _employee(document: Document) -> None:
    lay = _Layout(document)
    portrait = lay.side.orientation == "portrait"
    accent = "#1f4e79"
    if portrait:
        lay.add(RectItem(name="상단 띠", fill=accent, stroke_color=None, **lay.band(0.0, 0.16)))
        lay.add(TextItem(name="회사명", text="회사명", color="#ffffff", bold=True, font_size_pt=lay.pt(0.05),
                         h_align="center", v_align="middle", auto_shrink=True, **lay.box(0.06, 0.03, 0.88, 0.10)))
        lay.add(ImageItem(name="사진", source=ColumnSource(column="사진"), **lay.box(0.25, 0.21, 0.50, 0.36)))
        lay.add(TextItem(name="이름", source=ColumnSource(column="이름"), bold=True, font_size_pt=lay.pt(0.065),
                         h_align="center", v_align="middle", auto_shrink=True, **lay.box(0.08, 0.60, 0.84, 0.10)))
        lay.add(TextItem(name="부서 직급", h_align="center", v_align="middle", auto_shrink=True,
                         font_size_pt=lay.pt(0.04), color="#444444",
                         source=ConcatSource(separator=" ", parts=[ConcatPart(column="부서"), ConcatPart(column="직급")]),
                         **lay.box(0.08, 0.71, 0.84, 0.07)))
        lay.add(BarcodeItem(name="사번 바코드", source=ColumnSource(column="사번"), data="A1234567",
                            **lay.box(0.14, 0.82, 0.72, 0.14)))
    else:
        lay.add(RectItem(name="상단 띠", fill=accent, stroke_color=None, **lay.band(0.0, 0.22)))
        lay.add(TextItem(name="회사명", text="회사명", color="#ffffff", bold=True, font_size_pt=lay.pt(0.09),
                         v_align="middle", auto_shrink=True, **lay.box(0.05, 0.04, 0.90, 0.14)))
        lay.add(ImageItem(name="사진", source=ColumnSource(column="사진"), **lay.box(0.05, 0.28, 0.27, 0.64)))
        lay.add(TextItem(name="이름", source=ColumnSource(column="이름"), bold=True, font_size_pt=lay.pt(0.12),
                         v_align="middle", auto_shrink=True, **lay.box(0.37, 0.28, 0.58, 0.17)))
        lay.add(TextItem(name="부서 직급", v_align="middle", auto_shrink=True, font_size_pt=lay.pt(0.07),
                         color="#444444",
                         source=ConcatSource(separator=" ", parts=[ConcatPart(column="부서"), ConcatPart(column="직급")]),
                         **lay.box(0.37, 0.46, 0.58, 0.11)))
        lay.add(BarcodeItem(name="사번 바코드", source=ColumnSource(column="사번"), data="A1234567",
                            **lay.box(0.37, 0.66, 0.58, 0.26)))


def _student(document: Document) -> None:
    lay = _Layout(document)
    portrait = lay.side.orientation == "portrait"
    accent = "#2e7d32"
    if portrait:
        lay.add(TextItem(name="학교명", text="학교명", bold=True, color=accent, font_size_pt=lay.pt(0.055),
                         h_align="center", v_align="middle", auto_shrink=True, **lay.box(0.06, 0.04, 0.88, 0.09)))
        lay.add(LineItem(name="구분선", color=accent, width_mm=0.5, **{**lay.box(0.08, 0.145, 0.84, 0.0), "h": 0.0}))
        lay.add(ImageItem(name="사진", source=ColumnSource(column="사진"), **lay.box(0.25, 0.19, 0.50, 0.36)))
        lay.add(TextItem(name="이름", source=ColumnSource(column="이름"), bold=True, font_size_pt=lay.pt(0.065),
                         h_align="center", v_align="middle", auto_shrink=True, **lay.box(0.08, 0.58, 0.84, 0.10)))
        lay.add(TextItem(name="학과", source=ColumnSource(column="학과"), font_size_pt=lay.pt(0.04),
                         h_align="center", v_align="middle", auto_shrink=True, **lay.box(0.08, 0.69, 0.84, 0.07)))
        lay.add(TextItem(name="학번", source=ConcatSource(parts=[ConcatPart(column="학번", prefix="학번 ")]),
                         font_size_pt=lay.pt(0.04), h_align="center", v_align="middle", auto_shrink=True,
                         **lay.box(0.08, 0.77, 0.84, 0.07)))
        lay.add(RectItem(name="하단 띠", fill=accent, stroke_color=None, **lay.band(0.90, 0.10)))
    else:
        lay.add(TextItem(name="학교명", text="학교명", bold=True, color=accent, font_size_pt=lay.pt(0.10),
                         v_align="middle", auto_shrink=True, **lay.box(0.05, 0.05, 0.90, 0.15)))
        lay.add(LineItem(name="구분선", color=accent, width_mm=0.5, **{**lay.box(0.05, 0.23, 0.90, 0.0), "h": 0.0}))
        lay.add(ImageItem(name="사진", source=ColumnSource(column="사진"), **lay.box(0.05, 0.29, 0.26, 0.58)))
        lay.add(TextItem(name="이름", source=ColumnSource(column="이름"), bold=True, font_size_pt=lay.pt(0.12),
                         v_align="middle", auto_shrink=True, **lay.box(0.36, 0.29, 0.59, 0.17)))
        lay.add(TextItem(name="학과", source=ColumnSource(column="학과"), font_size_pt=lay.pt(0.07),
                         v_align="middle", auto_shrink=True, **lay.box(0.36, 0.48, 0.59, 0.11)))
        lay.add(TextItem(name="학번", source=ConcatSource(parts=[ConcatPart(column="학번", prefix="학번 ")]),
                         font_size_pt=lay.pt(0.07), v_align="middle", auto_shrink=True,
                         **lay.box(0.36, 0.61, 0.59, 0.11)))
        lay.add(RectItem(name="하단 띠", fill=accent, stroke_color=None, **lay.band(0.90, 0.10)))


def _membership(document: Document) -> None:
    lay = _Layout(document)
    portrait = lay.side.orientation == "portrait"
    lay.add(RectItem(name="배경", fill="#263238", stroke_color=None, **lay.band(0.0, 1.0)))
    qr_mm = round(min(lay.width, lay.height) * 0.36, 3)
    if portrait:
        lay.add(TextItem(name="카드 이름", text="MEMBERSHIP", bold=True, color="#ffd54f", h_align="center",
                         v_align="middle", auto_shrink=True, font_size_pt=lay.pt(0.05), **lay.box(0.06, 0.06, 0.88, 0.09)))
        lay.add(TextItem(name="이름", source=ColumnSource(column="이름"), bold=True, color="#ffffff", h_align="center",
                         v_align="middle", auto_shrink=True, font_size_pt=lay.pt(0.065), **lay.box(0.08, 0.30, 0.84, 0.10)))
        lay.add(TextItem(name="회원번호", color="#cfd8dc", h_align="center", v_align="middle", auto_shrink=True,
                         font_size_pt=lay.pt(0.04),
                         source=ConcatSource(parts=[ConcatPart(column="회원번호", prefix="No. ")]),
                         **lay.box(0.08, 0.41, 0.84, 0.07)))
        qr_x, qr_y = lay.left + (lay.width - qr_mm) / 2, lay.top + lay.height * 0.92 - qr_mm
    else:
        lay.add(TextItem(name="카드 이름", text="MEMBERSHIP", bold=True, color="#ffd54f", v_align="middle",
                         auto_shrink=True, font_size_pt=lay.pt(0.10), **lay.box(0.06, 0.08, 0.60, 0.16)))
        lay.add(TextItem(name="이름", source=ColumnSource(column="이름"), bold=True, color="#ffffff",
                         v_align="middle", auto_shrink=True, font_size_pt=lay.pt(0.12), **lay.box(0.06, 0.52, 0.55, 0.18)))
        lay.add(TextItem(name="회원번호", color="#cfd8dc", v_align="middle", auto_shrink=True,
                         font_size_pt=lay.pt(0.07),
                         source=ConcatSource(parts=[ConcatPart(column="회원번호", prefix="No. ")]),
                         **lay.box(0.06, 0.72, 0.55, 0.12)))
        qr_x, qr_y = lay.left + lay.width * 0.94 - qr_mm, lay.top + (lay.height - qr_mm) / 2
    lay.add(QRItem(name="회원 QR", source=ColumnSource(column="회원번호"), data="M000001",
                   x=round(qr_x, 3), y=round(qr_y, 3), w=qr_mm, h=qr_mm))


def _visitor(document: Document) -> None:
    lay = _Layout(document)
    portrait = lay.side.orientation == "portrait"
    accent = "#c62828"
    title_h = 0.20 if portrait else 0.30
    lay.add(RectItem(name="상단 띠", fill=accent, stroke_color=None, **lay.band(0.0, title_h)))
    lay.add(TextItem(name="제목", text="방 문 증", bold=True, color="#ffffff", h_align="center", v_align="middle",
                     auto_shrink=True, font_size_pt=lay.pt(0.09 if portrait else 0.15),
                     **lay.box(0.06, 0.03, 0.88, title_h - 0.06)))
    lay.add(TextItem(name="방문자", source=ColumnSource(column="이름"), bold=True, h_align="center", v_align="middle",
                     auto_shrink=True, font_size_pt=lay.pt(0.07 if portrait else 0.13),
                     **lay.box(0.08, title_h + 0.10, 0.84, 0.14 if portrait else 0.20)))
    lay.add(TextItem(name="방문처", source=ColumnSource(column="방문처"), h_align="center", v_align="middle",
                     auto_shrink=True, color="#444444", font_size_pt=lay.pt(0.04 if portrait else 0.07),
                     **lay.box(0.08, title_h + 0.28 if portrait else title_h + 0.32, 0.84, 0.08 if portrait else 0.12)))
    lay.add(TextItem(name="발급일", source=DateSource(format="%Y.%m.%d"), color="#666666", v_align="middle",
                     font_size_pt=lay.pt(0.03 if portrait else 0.055), auto_shrink=True,
                     **lay.box(0.06, 0.88, 0.50, 0.08)))
    lay.add(TextItem(name="일련번호", source=CounterSource(start=1, pad=4, prefix="No. "), color="#666666",
                     h_align="right", v_align="middle", auto_shrink=True,
                     font_size_pt=lay.pt(0.03 if portrait else 0.055), **lay.box(0.56, 0.88, 0.38, 0.08)))


@dataclass(frozen=True)
class Template:
    key: str
    label: str
    description: str
    builder: Callable[[Document], None]


TEMPLATES: tuple[Template, ...] = (
    Template("blank", "빈 카드", "아무것도 없는 카드에서 시작합니다.", _blank),
    Template("employee", "사원증", "사진, 이름, 부서·직급, 사번 바코드", _employee),
    Template("student", "학생증", "사진, 이름, 학과, 학번", _student),
    Template("membership", "회원 카드", "이름, 회원번호, QR 코드", _membership),
    Template("visitor", "방문증", "방문자 이름, 방문처, 발급일, 일련번호", _visitor),
)


def build_template(
    key: str, preset: CardPreset = CR80, orientation: str = "landscape", *, name: str | None = None,
) -> Document:
    for template in TEMPLATES:
        if template.key == key:
            document = new_document(preset, orientation, name=name or template.label)
            template.builder(document)
            return document
    raise KeyError(f"없는 템플릿입니다: {key}")
