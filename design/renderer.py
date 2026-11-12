"""QPainter 렌더러. 편집 화면과 인쇄가 같은 `paint_item`을 쓴다."""

from __future__ import annotations

import math
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Mapping

from PIL import Image
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QColor, QFont, QFontDatabase, QFontMetricsF, QGuiApplication, QImage, QImageReader,
    QPainter, QPen, QTransform,
)

from design import barcodes
from design.binding import BindingContext, resolve
from design.model import DEFAULT_FONT_FAMILY, Crop, Document, Item

MM_PER_INCH = 25.4
FALLBACK_FONT_FAMILIES = (
    "Malgun Gothic", "맑은 고딕", "Noto Sans KR", "Noto Sans CJK KR", "NanumGothic",
    "Apple SD Gothic Neo",
)
# 글자 배치는 0.1pt 단위 좌표에서 하고 painter를 축소해 그린다.
# 배율이 달라도 줄바꿈·자동 축소 결과가 같아서 화면과 인쇄가 일치한다.
_TEXT_UNITS_PER_PT = 10.0
_TEXT_UNITS_PER_MM = 72.0 * _TEXT_UNITS_PER_PT / MM_PER_INCH
_SHRINK_STEP_PT = 0.5
_HUGE = 1.0e7

_app_ref: QGuiApplication | None = None
_fonts_ready = False


def mm_to_px_for(dpi: float) -> float:
    return dpi / MM_PER_INCH


def ensure_gui() -> None:
    """QGuiApplication과 글꼴을 준비한다. 화면 없는 환경에서는 offscreen으로 만든다."""
    global _app_ref, _fonts_ready
    if QGuiApplication.instance() is None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        _app_ref = QGuiApplication([])
    if _fonts_ready:
        return
    _fonts_ready = True
    if QFontDatabase.families():
        return
    # offscreen 플랫폼은 시스템 글꼴을 읽지 않으므로 직접 올린다.
    candidates = []
    windows_fonts = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
    for name in ("malgun.ttf", "malgunbd.ttf", "malgunsl.ttf", "gulim.ttc", "arial.ttf", "arialbd.ttf"):
        candidates.append(os.path.join(windows_fonts, name))
    candidates += [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    ]
    for path in candidates:
        if os.path.isfile(path):
            QFontDatabase.addApplicationFont(path)


def resolve_font_family(family: str | None) -> str:
    """실제로 쓰일 글꼴 이름. 없는 글꼴이면 기본 한글 글꼴로 대체한다."""
    ensure_gui()
    for name in (family, *FALLBACK_FONT_FAMILIES):
        if name and QFontDatabase.hasFamily(name):
            return name
    return QFontDatabase.systemFont(QFontDatabase.SystemFont.GeneralFont).family()


def _spacing_units(size_pt: float, letter_spacing: float) -> float:
    """글자 사이에 더 띄우는 폭(글자 단위). letter_spacing은 글자 크기의 %."""
    return size_pt * _TEXT_UNITS_PER_PT * float(letter_spacing or 0.0) / 100.0


def _make_font(family: str | None, size_pt: float, bold=False, italic=False, underline=False,
               letter_spacing: float = 0.0) -> QFont:
    font = QFont()
    names: list[str] = []
    for name in (family or DEFAULT_FONT_FAMILY, *FALLBACK_FONT_FAMILIES):
        if name not in names:
            names.append(name)
    font.setFamilies(names)
    font.setPixelSize(max(1, round(size_pt * _TEXT_UNITS_PER_PT)))
    font.setBold(bool(bold))
    font.setItalic(bool(italic))
    font.setUnderline(bool(underline))
    font.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
    font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
    if letter_spacing:
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, _spacing_units(size_pt, letter_spacing))
    return font


def _color(value: str | None, mask_mode: bool, default: str = "#000000") -> QColor:
    if mask_mode:
        return QColor(255, 255, 255)
    color = QColor(value) if value else QColor(default)
    return color if color.isValid() else QColor(default)


# ---------------------------------------------------------------- 이미지 불러오기

class _LRU(OrderedDict):
    def __init__(self, capacity: int) -> None:
        super().__init__()
        self.capacity = capacity

    def lookup(self, key):
        if key in self:
            self.move_to_end(key)
            return self[key]
        return None

    def store(self, key, value) -> None:
        self[key] = value
        self.move_to_end(key)
        while len(self) > self.capacity:
            self.popitem(last=False)


_image_cache = _LRU(24)
_scaled_cache = _LRU(24)


def clear_image_cache() -> None:
    _image_cache.clear()
    _scaled_cache.clear()


def load_image(path: str | os.PathLike) -> QImage | None:
    """파일을 QImage로 읽는다(EXIF 회전 반영). 못 읽으면 None."""
    path = os.fspath(path)
    try:
        stat = os.stat(path)
    except OSError:
        return None
    key = (os.path.abspath(path), stat.st_mtime_ns, stat.st_size)
    cached = _image_cache.lookup(key)
    if cached is not None:
        return cached
    reader = QImageReader(path)
    reader.setAutoTransform(True)
    image = reader.read()
    if image.isNull():
        return None
    image = image.convertToFormat(QImage.Format.Format_ARGB32_Premultiplied)
    _image_cache.store(key, image)
    return image


def _pil_to_qimage(image: Image.Image) -> QImage:
    rgba = image.convert("RGBA")
    data = rgba.tobytes("raw", "RGBA")
    qimage = QImage(data, rgba.width, rgba.height, rgba.width * 4, QImage.Format.Format_RGBA8888)
    return qimage.convertToFormat(QImage.Format.Format_ARGB32_Premultiplied)


def _as_qimage(value: Any) -> QImage | None:
    if value is None:
        return None
    if isinstance(value, QImage):
        return None if value.isNull() else value
    if isinstance(value, Image.Image):
        return _pil_to_qimage(value)
    if isinstance(value, (str, os.PathLike)):
        return load_image(value)
    return None


def _to_mask_image(image: QImage) -> QImage:
    """밝기를 화이트 잉크 양으로 쓰는 마스크용 그림(투명도는 유지)."""
    key = ("mask", image.cacheKey())
    cached = _scaled_cache.lookup(key)
    if cached is not None:
        return cached
    alpha = image.convertToFormat(QImage.Format.Format_Alpha8)
    gray = image.convertToFormat(QImage.Format.Format_Grayscale8).convertToFormat(
        QImage.Format.Format_ARGB32
    )
    gray.setAlphaChannel(alpha)
    result = gray.convertToFormat(QImage.Format.Format_ARGB32_Premultiplied)
    _scaled_cache.store(key, result)
    return result


def _prescaled(image: QImage, width_px: float, height_px: float) -> QImage:
    """크게 줄여 그릴 때 계단 현상이 없도록 미리 부드럽게 줄인다."""
    target_w, target_h = max(1, math.ceil(width_px)), max(1, math.ceil(height_px))
    if image.width() <= target_w * 1.5 and image.height() <= target_h * 1.5:
        return image
    key = (image.cacheKey(), target_w, target_h)
    cached = _scaled_cache.lookup(key)
    if cached is None:
        cached = image.scaled(
            target_w, target_h,
            Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation,
        )
        _scaled_cache.store(key, cached)
    return cached


def image_target_rect(
    image_w: float, image_h: float, box_w: float, box_h: float, fit: str, crop: Crop | None = None
) -> tuple[float, float, float, float]:
    """상자(0,0,box_w,box_h) 안에서 이미지가 그려질 사각형 (x, y, w, h). 상자 밖은 잘린다."""
    crop = crop or Crop()
    if image_w <= 0 or image_h <= 0:
        return (0.0, 0.0, box_w, box_h)
    if fit == "stretch":
        base_w, base_h = box_w, box_h
    else:
        ratios = (box_w / image_w, box_h / image_h)
        scale = min(ratios) if fit == "contain" else max(ratios)
        base_w, base_h = image_w * scale, image_h * scale
    zoom = crop.zoom if crop.zoom > 0 else 1.0
    draw_w, draw_h = base_w * zoom, base_h * zoom
    center_x = box_w / 2.0 + crop.offset_x * abs(draw_w - box_w) / 2.0
    center_y = box_h / 2.0 + crop.offset_y * abs(draw_h - box_h) / 2.0
    return (center_x - draw_w / 2.0, center_y - draw_h / 2.0, draw_w, draw_h)


# ---------------------------------------------------------------- 글자

@lru_cache(maxsize=1024)
def _fitted_size(
    family: str, bold: bool, italic: bool, text: str, wrap: bool,
    size_pt: float, min_pt: float, width_u: float, height_u: float, letter_spacing: float = 0.0,
) -> float:
    size = size_pt
    floor = max(0.5, min(min_pt, size_pt))
    while True:
        metrics = QFontMetricsF(_make_font(family, size, bold, italic, letter_spacing=letter_spacing))
        # 마지막 글자 뒤에 붙는 간격은 눈에 보이지 않으므로 칸 너비에서 빼지 않는다.
        width_u_now = width_u + _spacing_units(size, letter_spacing)
        if wrap:
            rect = metrics.boundingRect(QRectF(0, 0, width_u_now, _HUGE), _wrap_flags(), text)
        else:
            rect = metrics.boundingRect(QRectF(0, 0, _HUGE, _HUGE), int(Qt.AlignmentFlag.AlignLeft), text)
        if (rect.width() <= width_u_now + 0.5 and rect.height() <= height_u + 0.5) or size <= floor:
            return size
        size = max(floor, size - _SHRINK_STEP_PT)


def _wrap_flags() -> int:
    return int(Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap | Qt.TextFlag.TextWrapAnywhere)


def fitted_font_size_pt(item, text: str) -> float:
    """자동 축소를 적용했을 때 실제로 쓰이는 글자 크기(pt)."""
    ensure_gui()
    if not item.auto_shrink or not text:
        return item.font_size_pt
    return _fitted_size(
        item.font_family, item.bold, item.italic, text, item.wrap,
        float(item.font_size_pt), float(item.min_font_size_pt),
        item.w * _TEXT_UNITS_PER_MM, item.h * _TEXT_UNITS_PER_MM,
        float(getattr(item, "letter_spacing", 0.0) or 0.0),
    )


_H_ALIGN = {
    "left": Qt.AlignmentFlag.AlignLeft, "center": Qt.AlignmentFlag.AlignHCenter,
    "right": Qt.AlignmentFlag.AlignRight,
}
_V_ALIGN = {
    "top": Qt.AlignmentFlag.AlignTop, "middle": Qt.AlignmentFlag.AlignVCenter,
    "bottom": Qt.AlignmentFlag.AlignBottom,
}


def _text_flags(h_align: str, v_align: str, wrap: bool) -> int:
    flags = int(_H_ALIGN.get(h_align, Qt.AlignmentFlag.AlignLeft)) | int(
        _V_ALIGN.get(v_align, Qt.AlignmentFlag.AlignTop)
    ) | int(Qt.TextFlag.TextDontClip)
    if wrap:
        flags |= int(Qt.TextFlag.TextWordWrap | Qt.TextFlag.TextWrapAnywhere)
    return flags


def _draw_text(
    painter: QPainter, text: str, font: QFont, color: QColor,
    x_mm: float, y_mm: float, w_mm: float, h_mm: float, mm_to_px: float,
    h_align: str = "left", v_align: str = "top", wrap: bool = False,
) -> None:
    flags = _text_flags(h_align, v_align, wrap)
    # 글자 사이를 띄우면 마지막 글자 뒤에도 간격이 붙어 가운데·오른쪽 맞춤이 왼쪽으로 쏠린다. 그만큼 되돌린다.
    trailing = font.letterSpacing() if font.letterSpacingType() == QFont.SpacingType.AbsoluteSpacing else 0.0
    shift = trailing / 2.0 if h_align == "center" else trailing if h_align == "right" else 0.0
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.translate(x_mm * mm_to_px, y_mm * mm_to_px)
    painter.scale(mm_to_px / _TEXT_UNITS_PER_MM, mm_to_px / _TEXT_UNITS_PER_MM)
    painter.setFont(font)
    painter.setPen(color)
    painter.drawText(QRectF(shift, 0, w_mm * _TEXT_UNITS_PER_MM, h_mm * _TEXT_UNITS_PER_MM), flags, text)
    painter.restore()


def _paint_text(painter, item, value, s, opts) -> None:
    text = "" if value is None else str(value)
    if not text:
        return
    size = fitted_font_size_pt(item, text)
    font = _make_font(item.font_family, size, item.bold, item.italic, item.underline,
                      letter_spacing=float(getattr(item, "letter_spacing", 0.0) or 0.0))
    _draw_text(
        painter, text, font, _color(item.color, opts.mask_mode), 0.0, 0.0, item.w, item.h, s,
        item.h_align, item.v_align, item.wrap,
    )


def _barcode_text_height_mm(item) -> float:
    """바코드 아래 글자 줄의 높이(mm). 글자를 안 보이면 0."""
    if not item.show_text:
        return 0.0
    return min(item.h * 0.5, item.text_size_pt * MM_PER_INCH / 72.0 * 1.35)


def _text_layout(item, value: Any):
    """(글자, 글꼴, 글자 상자 x·y·w·h(mm), 정렬 플래그). 글자를 그리지 않는 항목이면 None."""
    if item.kind == "text":
        text = "" if value is None else str(value)
        if not text:
            return None
        font = _make_font(
            item.font_family, fitted_font_size_pt(item, text), item.bold, item.italic, item.underline,
            letter_spacing=float(getattr(item, "letter_spacing", 0.0) or 0.0),
        )
        return text, font, (0.0, 0.0, item.w, item.h), _text_flags(item.h_align, item.v_align, item.wrap)
    if item.kind == "barcode" and item.show_text:
        try:
            text = barcodes.normalize(item.symbology, "" if value is None else str(value))
        except barcodes.BarcodeError:
            return None
        height = _barcode_text_height_mm(item)
        font = _make_font(item.text_font_family, item.text_size_pt)
        return text, font, (0.0, item.h - height, item.w, height), _text_flags("center", "middle", False)
    return None


def text_extent(item: Item, value: Any, mm_to_px: float = 1.0) -> tuple[float, float, float, float]:
    """그려진 글자가 항목 상자 밖으로 나가는 길이 (왼쪽, 위, 오른쪽, 아래). 안 나가면 0.

    글자는 상자에서 잘리지 않고 넘쳐 그려진다(줄바꿈 없는 긴 글자, 상자보다 큰 글꼴 등).
    편집 화면이 다시 그릴 영역·선택 테두리를 넓히거나 "글자가 넘칩니다" 안내를 띄울 때 쓴다.
    value는 `value_for`가 준 값, 단위는 mm × mm_to_px(기본 mm). 항목의 회전은 반영하지 않은
    항목 자체 좌표 기준이다. 글자 항목과 글자를 보이는 바코드만 값이 있고 나머지는 모두 0.
    글자 배치 폭(advance) 기준이라 기울임꼴의 삐침 한두 픽셀은 포함하지 않는다.
    """
    ensure_gui()
    layout = _text_layout(item, value)
    if layout is None:
        return (0.0, 0.0, 0.0, 0.0)
    text, font, (x, y, w, h), flags = layout
    u = _TEXT_UNITS_PER_MM
    box = QRectF(x * u, y * u, w * u, h * u)
    drawn = QFontMetricsF(font).boundingRect(box, flags, text)
    k = mm_to_px / u
    return (
        max(0.0, -drawn.left()) * k, max(0.0, -drawn.top()) * k,
        max(0.0, drawn.right() - item.w * u) * k, max(0.0, drawn.bottom() - item.h * u) * k,
    )


# ---------------------------------------------------------------- 도형

def _stroke_pen(color: str | None, width_mm: float, s: float, mask_mode: bool) -> QPen | None:
    if not color or width_mm <= 0:
        return None
    pen = QPen(_color(color, mask_mode), width_mm * s)
    pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
    return pen


def _shape_rect(item, s: float, pen: QPen | None) -> QRectF:
    inset = pen.widthF() / 2.0 if pen else 0.0
    return QRectF(inset, inset, max(0.0, item.w * s - 2 * inset), max(0.0, item.h * s - 2 * inset))


def _prepare_shape(painter, item, s, opts) -> QRectF:
    pen = _stroke_pen(item.stroke_color, item.stroke_width_mm, s, opts.mask_mode)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setPen(pen if pen else Qt.PenStyle.NoPen)
    painter.setBrush(_color(item.fill, opts.mask_mode) if item.fill else Qt.BrushStyle.NoBrush)
    return _shape_rect(item, s, pen)


def _paint_rect(painter, item, value, s, opts) -> None:
    rect = _prepare_shape(painter, item, s, opts)
    radius = min(max(0.0, item.corner_radius_mm) * s, rect.width() / 2.0, rect.height() / 2.0)
    if radius > 0:
        painter.drawRoundedRect(rect, radius, radius)
    else:
        painter.drawRect(rect)


def _paint_ellipse(painter, item, value, s, opts) -> None:
    painter.drawEllipse(_prepare_shape(painter, item, s, opts))


_LINE_STYLES = {
    "solid": Qt.PenStyle.SolidLine, "dash": Qt.PenStyle.DashLine, "dot": Qt.PenStyle.DotLine,
}


def _paint_line(painter, item, value, s, opts) -> None:
    if item.width_mm <= 0:
        return
    pen = QPen(_color(item.color, opts.mask_mode), item.width_mm * s)
    pen.setStyle(_LINE_STYLES.get(item.style, Qt.PenStyle.SolidLine))
    pen.setCapStyle(Qt.PenCapStyle.FlatCap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setPen(pen)
    w, h = item.w * s, item.h * s
    if item.flip:
        painter.drawLine(QPointF(0, h), QPointF(w, 0))
    else:
        painter.drawLine(QPointF(0, 0), QPointF(w, h))


# ---------------------------------------------------------------- 자리표시 / 이미지

def _paint_placeholder(painter, item, s, label: str, error: bool = False) -> None:
    w, h = item.w * s, item.h * s
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
    pen = QPen(QColor("#c0392b" if error else "#8a8f98"), 0)
    pen.setStyle(Qt.PenStyle.DashLine)
    painter.setPen(pen)
    painter.setBrush(QColor(192, 57, 43, 28) if error else QColor(138, 143, 152, 36))
    painter.drawRect(QRectF(0, 0, w, h))
    painter.restore()
    if label:
        size = max(4.0, min(9.0, item.h * 72.0 / MM_PER_INCH * 0.35))
        font = _make_font(DEFAULT_FONT_FAMILY, size)
        _draw_text(
            painter, label, font, QColor("#c0392b" if error else "#5f6672"),
            0.0, 0.0, item.w, item.h, s, "center", "middle", True,
        )


def _device_scale(painter: QPainter) -> float:
    t = painter.worldTransform()
    return math.sqrt(abs(t.m11() * t.m22() - t.m12() * t.m21())) or 1.0


def _paint_image(painter, item, value, s, opts) -> None:
    image = _as_qimage(value)
    w, h = item.w * s, item.h * s
    if image is None:
        if opts.design_mode and not opts.mask_mode:
            _paint_placeholder(painter, item, s, item.name or "이미지")
        elif value is not None and opts.problems is not None:
            opts.problems.append(f"이미지를 찾을 수 없습니다: {value}")
        return
    if opts.mask_mode:
        image = _to_mask_image(image)
    crop = opts.crop if opts.crop is not None else item.crop
    x, y, draw_w, draw_h = image_target_rect(image.width(), image.height(), w, h, item.fit, crop)
    scale = _device_scale(painter)
    image = _prescaled(image, draw_w * scale, draw_h * scale)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
    painter.setClipRect(QRectF(0, 0, w, h), Qt.ClipOperation.IntersectClip)
    painter.drawImage(QRectF(x, y, draw_w, draw_h), image)


# ---------------------------------------------------------------- 바코드 / QR

def _draw_crisp(painter: QPainter, rect: QRectF, make_image: Callable[[int, int], QImage]) -> None:
    """make_image(w_px, h_px) 결과를 장치 픽셀에 맞춰 그린다. 90° 단위 회전까지는 보간 없이 찍힌다."""
    t = painter.worldTransform()
    angle = math.degrees(math.atan2(t.m12(), t.m11()))
    quarter = round(angle / 90.0)
    mirrored = t.m11() * t.m22() - t.m12() * t.m21() < 0
    if abs(angle - quarter * 90.0) < 0.01 and not mirrored:
        device = t.mapRect(rect)
        left, top = round(device.left()), round(device.top())
        dev_w, dev_h = max(1, round(device.width())), max(1, round(device.height()))
        quarter %= 4
        image = make_image(dev_w, dev_h) if quarter % 2 == 0 else make_image(dev_h, dev_w)
        if quarter:
            image = image.transformed(QTransform().rotate(90.0 * quarter))
        painter.save()
        painter.resetTransform()
        painter.drawImage(left, top, image)
        painter.restore()
        return
    scale = math.hypot(t.m11(), t.m12())
    image = make_image(max(1, round(rect.width() * scale)), max(1, round(rect.height() * scale)))
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
    painter.drawImage(rect, image)
    painter.restore()


def _paint_code(painter, item, s, opts, rect: QRectF, make_image, min_image) -> bool:
    """그렸으면 True. 못 그렸으면 자리표시(디자인 화면) 또는 problems 기록 후 False."""
    try:
        _draw_crisp(painter, rect, make_image)
    except barcodes.BarcodeSizeError as exc:
        if not opts.design_mode:
            _report(opts, item, str(exc))
            return False
        # 화면을 축소해 봤을 때만 생기는 경우: 최소 크기로 만들어 줄여서 보여준다.
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.drawImage(rect, min_image(exc.min_width_px))
        painter.restore()
    except barcodes.BarcodeError as exc:
        if opts.design_mode and not opts.mask_mode:
            _paint_placeholder(painter, item, s, str(exc), error=True)
        else:
            _report(opts, item, str(exc))
        return False
    return True


_KIND_LABELS = {"barcode": "바코드", "qr": "QR 코드"}


def _report(opts, item, message: str) -> None:
    if opts.problems is not None:
        label = item.name or _KIND_LABELS.get(item.kind, item.kind)
        opts.problems.append(f"{label}: {message}")


def _paint_barcode(painter, item, value, s, opts) -> None:
    data = "" if value is None else str(value)
    w, h = item.w * s, item.h * s
    foreground = _color(item.color, opts.mask_mode)
    background = None if opts.mask_mode or not item.background else QColor(item.background)
    try:
        shown = barcodes.normalize(item.symbology, data)
    except barcodes.BarcodeError as exc:
        if opts.design_mode and not opts.mask_mode:
            unbound = data.startswith("«")
            _paint_placeholder(painter, item, s, data if unbound else str(exc), error=not unbound)
        else:
            _report(opts, item, str(exc))
        return
    text_h_mm = _barcode_text_height_mm(item)
    bars = QRectF(0, 0, w, max(0.0, (item.h - text_h_mm) * s))

    def make(width_px: int, height_px: int) -> QImage:
        return barcodes.render_linear(
            item.symbology, shown, width_px, height_px,
            quiet_zone=item.quiet_zone, foreground=foreground, background=background,
        )

    if not _paint_code(painter, item, s, opts, bars, make, lambda min_w: make(min_w, 8)):
        return
    if item.show_text:
        if background is not None:
            # 막대 그림은 장치 픽셀에 맞춰 반올림되므로 틈이 생기지 않게 장치 픽셀 1개만큼 겹친다.
            # (인쇄 렌더링은 painter 배율이 1이라 정확히 1.0이고, 화면은 확대·축소 배율을 따라간다.)
            scale = _device_scale(painter)
            overlap = 1.0 if abs(scale - 1.0) < 1e-9 else 1.0 / scale
            top = max(0.0, bars.height() - overlap)
            painter.fillRect(QRectF(0, top, w, h - top), background)
        font = _make_font(item.text_font_family, item.text_size_pt)
        _draw_text(
            painter, shown, font, foreground, 0.0, item.h - text_h_mm, item.w, text_h_mm, s,
            "center", "middle",
        )


def _paint_qr(painter, item, value, s, opts) -> None:
    data = "" if value is None else str(value)
    foreground = _color(item.color, opts.mask_mode)
    background = None if opts.mask_mode or not item.background else QColor(item.background)

    def make(width_px: int, height_px: int) -> QImage:
        return barcodes.render_qr(
            data, (width_px, height_px), error_correction=item.error_correction,
            border_modules=item.border_modules, foreground=foreground, background=background,
        )

    _paint_code(
        painter, item, s, opts, QRectF(0, 0, item.w * s, item.h * s), make,
        lambda min_w: make(min_w, min_w),
    )


_PAINTERS: dict[str, Callable] = {
    "text": _paint_text, "image": _paint_image, "rect": _paint_rect, "ellipse": _paint_ellipse,
    "line": _paint_line, "barcode": _paint_barcode, "qr": _paint_qr,
}


@dataclass
class _Options:
    crop: Crop | None = None
    design_mode: bool = True
    mask_mode: bool = False
    problems: list[str] | None = None


def paint_item_local(
    painter: QPainter, item: Item, value: Any, mm_to_px: float, *,
    crop: Crop | None = None, design_mode: bool = True, mask_mode: bool = False,
    problems: list[str] | None = None,
) -> None:
    """항목을 원점 기준 (0,0)-(w·mm_to_px, h·mm_to_px) 상자에 그린다.

    위치·회전은 호출한 쪽(예: QGraphicsItem)이 건다. painter 좌표 1단위가 1/mm_to_px mm면 되고
    painter에 걸린 확대·축소 배율은 자유다(선 굵기·바코드 겹침은 장치 픽셀 기준으로 맞춘다).
    글자는 상자에서 잘리지 않으므로 넘치는 길이는 `text_extent`로 구한다.
    """
    handler = _PAINTERS.get(item.kind)
    if handler is None:
        return
    painter.save()
    try:
        handler(painter, item, value, mm_to_px, _Options(crop, design_mode, mask_mode, problems))
    finally:
        painter.restore()


def paint_item(
    painter: QPainter, item: Item, value: Any, mm_to_px: float, *,
    crop: Crop | None = None, design_mode: bool = True, mask_mode: bool = False,
    problems: list[str] | None = None,
) -> None:
    """항목을 면 좌표(원점=면 왼쪽 위)에 그린다. 회전은 항목 중심 기준.

    value: `value_for`가 준 값. 이미지는 절대 경로 / QImage / PIL 이미지 / None.
    design_mode: True면 빈 이미지·잘못된 바코드 자리에 안내 상자를 그린다.
    mask_mode: 화이트·UV 마스크용. 모든 색을 흰색(=잉크)으로 그린다.
    problems: 인쇄에서 빠진 것(없는 사진, 잘못된 바코드)의 설명이 쌓인다.
    """
    painter.save()
    try:
        center_x, center_y = (item.x + item.w / 2.0) * mm_to_px, (item.y + item.h / 2.0) * mm_to_px
        painter.translate(center_x, center_y)
        if item.rotation:
            painter.rotate(item.rotation)
        painter.translate(-item.w * mm_to_px / 2.0, -item.h * mm_to_px / 2.0)
        paint_item_local(
            painter, item, value, mm_to_px,
            crop=crop, design_mode=design_mode, mask_mode=mask_mode, problems=problems,
        )
    finally:
        painter.restore()


# ---------------------------------------------------------------- 값 준비 / 면 렌더링

def asset_path(document: Document, relative: str | None) -> str | None:
    if not relative:
        return None
    if os.path.isabs(relative) or not document.base_dir:
        return relative
    return os.path.join(document.base_dir, relative)


def value_for(
    document: Document, item: Item, record: Mapping[str, Any] | None = None,
    context: BindingContext | None = None, *, cutout: bool = True,
) -> Any:
    """바인딩을 풀고, 문서 안 이미지 경로를 절대 경로로 바꾼 `paint_item`용 값.

    배경을 지운 그림(`cutout_asset`)이 있으면 원본 대신 그 경로를 준다. cutout=False면 항상 원본.
    """
    value = resolve(item, record, context)
    if item.kind == "image" and isinstance(value, str):
        from_document = item.source is None or item.source.type == "fixed"
        if from_document:
            if cutout and value == item.asset:
                cut = asset_path(document, getattr(item, "cutout_asset", None))
                if cut and os.path.isfile(cut):
                    return cut
            return asset_path(document, value)
    return value


@dataclass
class RenderedSide:
    color: Image.Image
    white_mask: Image.Image | None = None
    uv: Image.Image | None = None
    orientation: str = "landscape"
    dpi: float = 300.0
    warnings: list[str] = field(default_factory=list)

    def save(self, directory: str | os.PathLike, stem: str) -> dict[str, str | None]:
        """PNG로 저장하고 printer.SideImages에 넣을 경로들을 돌려준다."""
        os.makedirs(directory, exist_ok=True)
        paths: dict[str, str | None] = {"orientation": self.orientation}
        for key, suffix, image in (
            ("color_path", "color", self.color),
            ("white_mask_path", "white", self.white_mask),
            ("uv_path", "uv", self.uv),
        ):
            if image is None:
                paths[key] = None
                continue
            path = os.path.join(os.fspath(directory), f"{stem}_{suffix}.png")
            image.save(path, dpi=(self.dpi, self.dpi))
            paths[key] = path
        return paths


class RenderError(RuntimeError):
    pass


def side_pixel_size(document: Document, side_key: str, dpi: float = 300.0) -> tuple[int, int]:
    width_mm, height_mm = document.side_size_mm(side_key)
    s = mm_to_px_for(dpi)
    return (max(1, round(width_mm * s)), max(1, round(height_mm * s)))


def cut_pixel_size(document: Document, side_key: str, dpi: float = 300.0) -> tuple[int, int]:
    """칼선(도련 제외) 크기의 픽셀 수: round(칼선 mm × dpi ÷ 25.4)."""
    width_mm, height_mm = document.side_size_mm(side_key)
    s = mm_to_px_for(dpi)
    bleed = max(0.0, document.bleed_mm)
    return (max(1, round((width_mm - 2 * bleed) * s)), max(1, round((height_mm - 2 * bleed) * s)))


def trim_bleed(rendered: "RenderedSide", document: Document, side_key: str) -> bool:
    """도련 포함으로 렌더링한 면을 가운데 기준으로 잘라 정확히 칼선 크기로 만든다.

    도련이 없거나 이미 칼선 크기 이하면 그대로 두고 False를 돌려준다.
    """
    if document.bleed_mm <= 0:
        return False
    target_w, target_h = cut_pixel_size(document, side_key, rendered.dpi)
    width, height = rendered.color.size
    if target_w >= width or target_h >= height:
        return False
    left, top = (width - target_w) // 2, (height - target_h) // 2
    box = (left, top, left + target_w, top + target_h)
    rendered.color = rendered.color.crop(box)
    if rendered.white_mask is not None:
        rendered.white_mask = rendered.white_mask.crop(box)
    if rendered.uv is not None:
        rendered.uv = rendered.uv.crop(box)
    return True


def _to_pil(image: QImage, mode: str) -> Image.Image:
    fmt = QImage.Format.Format_RGB888 if mode == "RGB" else QImage.Format.Format_Grayscale8
    converted = image.convertToFormat(fmt)
    data = bytes(converted.constBits())
    return Image.frombuffer(
        mode, (converted.width(), converted.height()), data, "raw", mode, converted.bytesPerLine(), 1
    ).copy()


class _MaskLayer:
    def __init__(self, width: int, height: int) -> None:
        self.size = (width, height)
        self.image: QImage | None = None
        self.painter: QPainter | None = None

    def begin(self) -> QPainter:
        if self.painter is None:
            self.image = QImage(*self.size, QImage.Format.Format_RGB32)
            self.image.fill(QColor(0, 0, 0))
            self.painter = QPainter(self.image)
            # 검은 부분이 앞서 그린 잉크 영역을 지우지 않도록 밝은 쪽만 남긴다.
            self.painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Lighten)
        return self.painter

    def finish(self) -> Image.Image | None:
        if self.painter is None:
            return None
        self.painter.end()
        return _to_pil(self.image, "L")


def render_side(
    document: Document,
    side_key: str,
    record: Mapping[str, Any] | None = None,
    dpi: float = 300,
    context: BindingContext | None = None,
    *,
    design_mode: bool | None = None,
    crop_bleed: bool = False,
    strict: bool = False,
) -> RenderedSide:
    """한 면을 인쇄용 이미지로 만든다.

    design_mode: None이면 record가 없을 때만 자리표시를 그린다.
    crop_bleed: 문서에 도련이 있으면 칼선 안쪽만 남긴다. 결과는 정확히 `cut_pixel_size`다.
    strict: 빠진 사진·잘못된 바코드가 있으면 RenderError를 낸다(기본은 warnings에 기록).
    """
    ensure_gui()
    context = context or BindingContext()
    if design_mode is None:
        design_mode = record is None
    side = document.side(side_key)
    s = mm_to_px_for(dpi)
    width, height = side_pixel_size(document, side_key, dpi)
    problems: list[str] = []

    color_image = QImage(width, height, QImage.Format.Format_RGB32)
    color_image.fill(QColor(255, 255, 255))
    painter = QPainter(color_image)
    masks = {"white": _MaskLayer(width, height), "uv": _MaskLayer(width, height)}
    try:
        if side.background:
            if side.background.startswith("#"):
                painter.fillRect(0, 0, width, height, _color(side.background, False, "#ffffff"))
            else:
                background = load_image(asset_path(document, side.background))
                if background is not None:
                    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
                    painter.drawImage(QRectF(0, 0, width, height), _prescaled(background, width, height))
                else:
                    problems.append(f"배경 이미지를 찾을 수 없습니다: {side.background}")

        for item in side.sorted_items():
            if not item.visible or not item.printable:
                continue
            value = value_for(document, item, record, context)
            crop = item.crop_for(context.record_key) if item.kind == "image" else None
            if item.layer == "color":
                paint_item(
                    painter, item, value, s, crop=crop, design_mode=design_mode, problems=problems
                )
                cut_file = asset_path(document, getattr(item, "cutout_asset", None))
                if item.kind == "image" and cut_file and not os.path.isfile(cut_file):
                    problems.append(f"배경을 지운 그림을 찾을 수 없어 원래 그림으로 그렸습니다: {item.cutout_asset}")
                mask_file = asset_path(document, getattr(item, "mask_asset", None))
                if item.kind == "image" and mask_file:
                    if load_image(mask_file) is None:
                        problems.append(f"마스크 이미지를 찾을 수 없습니다: {item.mask_asset}")
                    else:
                        paint_item(
                            masks["white"].begin(), item, mask_file, s,
                            crop=crop, design_mode=False, mask_mode=True,
                        )
            elif item.layer in masks:
                paint_item(
                    masks[item.layer].begin(), item, value, s,
                    crop=crop, design_mode=design_mode, mask_mode=True, problems=problems,
                )
    finally:
        painter.end()
        white_mask = masks["white"].finish()
        uv = masks["uv"].finish()

    if strict and problems:
        raise RenderError("\n".join(problems))

    result = RenderedSide(
        color=_to_pil(color_image, "RGB"), white_mask=white_mask, uv=uv,
        orientation=side.orientation, dpi=float(dpi), warnings=problems,
    )
    if crop_bleed:
        trim_bleed(result, document, side_key)
    return result
