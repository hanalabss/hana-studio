"""다눌(DaNooll-Plus) 디자인 파일(.dnc)을 하나스튜디오 디자인으로 바꾼다.

.dnc는 .NET XmlSerializer가 쓴 평범한 XML이다(루트 <Card>). 항목은 <anyType xsi:type="ControlText"> 같은
모양으로 앞면(FrontSide)·뒷면(BackSide)의 <Controls> 안에 그려지는 순서대로 들어 있다.
좌표 단위는 1/100 인치(카드 340×214)이고 카드 왼쪽 위가 원점이다. 여기서는 카드 크기에 비례해 mm로 바꾼다.

옮기지 못하는 것은 notes로 알려 준다(다눌 고유의 리더기 키 번호, 사진의 사람별 위치 조정 등).
그림은 다눌처럼 파일을 가리키므로, 찾을 수 있으면 디자인 안으로 복사하고 못 찾으면 빈 사진 칸으로 둔다.
"""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

from design import storage
from design.binding import ColumnSource
from design.model import (
    BarcodeItem, Document, EllipseItem, ImageItem, Item, LineItem, QRItem, RectItem, TextItem,
)

EXTENSION = ".dnc"
_XSI_TYPE = "{http://www.w3.org/2001/XMLSchema-instance}type"
_BARCODES = {
    "CODE128": "code128", "CODE128A": "code128a", "CODE128B": "code128b", "CODE128C": "code128c",
    "CODE39": "code39", "CODE39EXTENDED": "code39ext", "CODE39_MOD43": "code39", "CODE93": "code93",
    "EAN13": "ean13",
}
_ALIGN = {"near": "left", "center": "center", "far": "right"}


class DanulError(ValueError):
    """사용자에게 보여 줄 수 있는 한국어 메시지."""


@dataclass
class DanulImport:
    document: Document
    notes: list[str] = field(default_factory=list)
    mdb_path: str = ""            # 다눌이 쓰던 Access 명단 파일(있으면 '명단 가져오기'로 바로 열 수 있다)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child(node: ET.Element | None, name: str) -> ET.Element | None:
    if node is None:
        return None
    for child in node:
        if _local(child.tag) == name:
            return child
    return None


def _text(node: ET.Element | None, name: str, default: str = "") -> str:
    child = _child(node, name)
    return default if child is None or child.text is None else child.text


def _float(node, name: str, default: float = 0.0) -> float:
    try:
        return float(_text(node, name, str(default)))
    except ValueError:
        return default


def _bool(node, name: str, default: bool = False) -> bool:
    value = _text(node, name, "").strip().lower()
    return default if not value else value == "true"


def _color(value: str, default: str | None = "#000000") -> str | None:
    """ColorTranslator.ToHtml 값('Black', '#1a2b3c', 'Transparent'). 투명이면 None."""
    from PySide6.QtGui import QColor

    value = (value or "").strip()
    if not value:
        return default
    if value.lower() == "transparent":
        return None
    color = QColor(value)
    if not color.isValid():
        color = QColor(value.lower())
    return color.name() if color.isValid() else default


def parse_font(value: str) -> tuple[str, float, set[str]]:
    """FontConverter 문자열 '맑은 고딕, 10pt, style=Bold, Italic' → (글꼴, 크기 pt, {모양})."""
    parts = [p.strip() for p in (value or "").split(",")]
    family = parts[0] if parts and parts[0] else "맑은 고딕"
    size = 9.0
    styles: set[str] = set()
    in_style = False
    for part in parts[1:]:
        match = re.fullmatch(r"([\d.]+)\s*pt", part)
        if match and not in_style:
            size = float(match.group(1))
            continue
        if part.startswith("style="):
            in_style = True
            part = part[len("style="):]
        if in_style and part:
            styles.add(part.strip().lower())
    return family, size, styles


def _source(node) -> ColumnSource | None:
    column = _text(node, "FieldName").strip()
    return ColumnSource(column=column) if column else None


def _box(node, k: float) -> tuple[float, float, float, float]:
    rect = _child(node, "Rectangle")
    return (_float(rect, "X") * k, _float(rect, "Y") * k,
            max(0.0, _float(rect, "Width")) * k, max(0.0, _float(rect, "Height")) * k)


def _common(node, k: float, kind_name: str) -> dict:
    x, y, w, h = _box(node, k)
    rotation = 0.0
    try:
        rotation = float(int(_text(node, "totalRotation", "0") or 0) % 360)
    except ValueError:
        pass
    return {"x": x, "y": y, "w": w, "h": h, "rotation": rotation, "locked": _bool(node, "Lock"),
            "name": _text(node, "Name") or kind_name, "source": _source(node)}


def _convert_control(node, kind: str, k: float, document: Document, result: DanulImport, base_dir: str) -> Item | None:
    common = _common(node, k, kind)
    pen = max(0, int(_float(node, "PenWidth", 1)))
    border = _color(_text(node, "BorderColor"), "#000000")
    filled = _bool(node, "Filled")
    fill = _color(_text(node, "FillColor"), None) if filled else None

    if kind == "ControlText":
        family, size, styles = parse_font(_text(node, "Font"))
        return TextItem(**common, text=_text(node, "Text"), font_family=family, font_size_pt=size,
                        bold="bold" in styles, italic="italic" in styles, underline="underline" in styles,
                        strikeout="strikeout" in styles, color=_color(_text(node, "FontColor")) or "#000000",
                        background=fill, h_align=_ALIGN.get(_text(node, "FontAlignment").lower(), "left"),
                        v_align="middle", auto_shrink=True,
                        letter_spacing=30.0 if _bool(node, "IsFontSpace") else 0.0)
    if kind == "ControlImage":
        name, folder = _text(node, "Text"), _text(node, "ImagePath")
        item = ImageItem(**common, fit="stretch" if _bool(node, "Fullsize") else "fill",
                         corner_radius_mm=max(0.0, _float(node, "Radius")) * k if _float(node, "Radius") > 1 else 0.0)
        if item.source is None and name:
            candidates = [os.path.join(folder, name) if folder else name, os.path.join(base_dir, name)]
            found = next((p for p in candidates if p and os.path.isfile(p)), None)
            if found:
                item.asset = storage.import_asset(document.base_dir, found)
            else:
                result.notes.append(f"그림 파일을 찾지 못해 빈 사진 칸으로 두었습니다: {name}")
        return item
    if kind == "ControlRectangle":
        radius = _float(node, "Radius") * k if _bool(node, "IsConnerRound") else 0.0
        return RectItem(**common, fill=fill, stroke_color=border if pen > 0 else None,
                        stroke_width_mm=pen * k, corner_radius_mm=radius)
    if kind == "ControlCircle":
        return EllipseItem(**common, fill=fill, stroke_color=border if pen > 0 else None, stroke_width_mm=pen * k)
    if kind == "ControlLine":
        direction = _text(node, "Direction", "Default").lower()
        x, y, w, h = common["x"], common["y"], common["w"], common["h"]
        if direction == "horizontal":
            common.update(y=y + h / 2.0, h=0.0)
        elif direction == "vertical":
            common.update(x=x + w / 2.0, w=0.0)
        return LineItem(**common, color=border or "#000000", width_mm=max(pen, 1) * k)
    if kind in ("ControlBarCodeImage", "ControlBarCodeFont"):
        symbology = "code39" if kind == "ControlBarCodeFont" else _BARCODES.get(
            _text(node, "BarCodeType", "CODE128").upper().replace("_", "").replace("-", ""), None)
        if symbology is None:
            result.notes.append(f"'{_text(node, 'BarCodeType')}' 바코드는 Code 128로 바꿨습니다.")
            symbology = "code128"
        return BarcodeItem(**common, symbology=symbology, data=_text(node, "Text"),
                           show_text=_bool(node, "IsVisbleLabel") if kind == "ControlBarCodeImage" else False)
    if kind == "ControlQRCode":
        size = max(common["w"], common["h"])
        common.update(w=size, h=size)
        return QRItem(**common, data=_text(node, "Text"))
    if kind == "ControlMagnetic":
        track = f"track{max(1, min(3, int(_float(node, 'TrackNumber', 1))))}"
        column = _text(node, "FieldName").strip()
        if column:
            document.extra.setdefault("magnetic", {})[track] = column
        else:
            result.notes.append(f"마그네틱 {track[-1]}번 줄의 고정 값('{_text(node, 'Text')}')은 옮기지 못했습니다. "
                                "인쇄 설정에서 명단 항목을 고르세요.")
        return None
    if kind == "ControlRFIC":
        column = _text(node, "FieldName").strip()
        try:
            block = int(_text(node, "BlockNumber", "4") or 4)
        except ValueError:
            block = 4
        if column:
            document.extra["mifare"] = {"column": column, "block": block if block % 4 != 3 and block > 0 else 4,
                                        "key_type": "A", "format": "text"}
        result.notes.append("IC 카드(RF) 키는 다눌에서는 리더기에 저장된 키 번호였습니다. 인쇄 설정에서 카드 키를 넣어 주세요.")
        return None
    result.notes.append(f"알 수 없는 항목({kind})은 건너뛰었습니다.")
    return None


def load_dnc(path: str | os.PathLike) -> DanulImport:
    """.dnc 파일을 새 문서로 바꾼다. 문서는 아직 저장되지 않은 상태(작업 폴더만 있음)다."""
    path = os.fspath(path)
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as exc:
        raise DanulError(f"다눌 디자인 파일을 읽지 못했습니다. ({exc})") from exc
    if _local(root.tag) != "Card":
        raise DanulError("다눌 디자인 파일(.dnc)이 아닙니다.")

    rect = _child(root, "CardRectangle")
    width, height = _float(rect, "Width", 340.0), _float(rect, "Height", 214.0)
    if width <= 0 or height <= 0:
        width, height = 340.0, 214.0
    k = 85.6 / max(width, height)
    landscape = width >= height

    document = Document(name=os.path.splitext(os.path.basename(path))[0])
    document.base_dir = storage.new_workdir()
    result = DanulImport(document, mdb_path=_text(root, "MDBFilePath"))
    base_dir = os.path.dirname(os.path.abspath(path))
    sides = _child(root, "Sides")
    for side_node in (list(sides) if sides is not None else []):
        key = {"frontside": "front", "backside": "back"}.get(_text(side_node, "Name").lower())
        if key is None:
            continue
        side = document.ensure_side(key)
        side.orientation = "landscape" if landscape else "portrait"
        controls = _child(side_node, "Controls")
        for node in (list(controls) if controls is not None else []):
            kind = node.get(_XSI_TYPE) or _local(node.tag)
            kind = kind.rsplit(":", 1)[-1]
            item = _convert_control(node, kind, k, document, result, base_dir)
            if item is not None:
                side.add_item(item)
    document.dual_side = _bool(root, "IsDuplex") or bool(document.side("back").items)
    return result
