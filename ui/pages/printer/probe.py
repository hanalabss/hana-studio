"""프린터 점검(카드와 리본을 쓰지 않는다)과 시험 인쇄용 카드."""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import QObject, QThread, Signal

from design import presets
from design.model import Document, LineItem, RectItem, TextItem
from printer import registry
from printer.base import PrinterDevice, PrinterError
from ui.pages.printing.text import capability_words, plain, status_sentence, supply_text
from ui.theme import SWATCH_PRESETS

log = logging.getLogger(__name__)


@dataclass
class CheckLine:
    tone: str        # ok | warn | error | idle
    title: str
    text: str


def run_probe(device: PrinterDevice) -> list[CheckLine]:
    """프린터를 열어 상태와 정보를 읽고 닫는다. 카드를 넣거나 빼지 않는다."""
    lines: list[CheckLine] = []
    try:
        printer = registry.open(device)
    except PrinterError as exc:
        lines.append(CheckLine("error", "연결", f"연결하지 못했습니다. {plain(exc.message)}"))
        lines.append(CheckLine("idle", "해 볼 일", "프린터 전원과 선을 확인한 뒤 다시 점검해 주세요."))
        return lines
    try:
        lines.append(CheckLine("ok", "연결", "프린터와 잘 이어져 있습니다."))
        status = printer.status()
        lines.append(CheckLine("ok" if status.ready else "error", "상태", status_sentence(status)))
        if status.feeder_empty is None:
            lines.append(CheckLine("idle", "카드 넣는 곳", "카드가 있는지 프린터가 알려 주지 않습니다. 직접 봐 주세요."))
        else:
            lines.append(CheckLine("error" if status.feeder_empty else "ok", "카드 넣는 곳",
                                   "비어 있습니다. 카드를 채워 주세요." if status.feeder_empty else "카드가 있습니다."))
        if status.ribbon_remaining is None:
            lines.append(CheckLine("idle", "리본", "남은 양을 프린터가 알려 주지 않습니다."))
        else:
            lines.append(CheckLine(_supply_tone(status.ribbon_remaining, status.ribbon_cards_left), "리본",
                                   supply_text(status.ribbon_remaining, status.ribbon_unit)))
        if status.film_remaining is not None:
            lines.append(CheckLine(_supply_tone(status.film_remaining, status.film_cards_left), "필름",
                                   supply_text(status.film_remaining, status.film_unit)))
        for warning in status.warnings:
            lines.append(CheckLine("warn", "알림", plain(warning)))
        words = capability_words(printer.capabilities())
        if words:
            lines.append(CheckLine("idle", "할 수 있는 일", ", ".join(words)))
        serial = _serial(printer.info())
        if serial:
            lines.append(CheckLine("idle", "기기 번호", serial))
    except PrinterError as exc:
        lines.append(CheckLine("error", "점검", f"점검하다가 문제가 생겼습니다. {plain(exc.message)}"))
    except Exception as exc:
        log.exception("프린터 점검 실패")
        lines.append(CheckLine("error", "점검", f"점검하다가 문제가 생겼습니다. ({exc})"))
    finally:
        try:
            printer.close()
        except Exception:
            log.debug("점검 뒤 프린터를 닫지 못했습니다.", exc_info=True)
    return lines


def _supply_tone(value: int, cards_left: int | None) -> str:
    if value <= 0:
        return "error"
    return "warn" if cards_left is not None and cards_left < 20 else "ok"


def _serial(info: dict[str, Any]) -> str:
    value = info.get("serial")
    if isinstance(value, dict):
        value = value.get("serial") or value.get("sn")
    if not value:
        manufacturer = info.get("manufacturer")
        if isinstance(manufacturer, dict):
            value = manufacturer.get("serial") or manufacturer.get("serial_number")
    return str(value).strip() if value else ""


class ProbeTask(QThread):
    """작업 스레드에서 run_probe를 돌린다. SDK는 한 번에 하나만 쓰도록 서비스의 잠금을 함께 쓴다."""

    finished_lines = Signal(object, object)     # device, list[CheckLine]

    def __init__(self, device: PrinterDevice, lock: Any = None, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._device = device
        self._lock = lock if lock is not None else threading.Lock()

    def run(self) -> None:
        with self._lock:
            try:
                lines = run_probe(self._device)
            except Exception as exc:
                log.exception("프린터 점검 스레드 오류")
                lines = [CheckLine("error", "점검", f"점검하지 못했습니다. ({exc})")]
        self.finished_lines.emit(self._device, lines)


def build_test_document(preset: presets.CardPreset | None = None) -> Document:
    """시험 인쇄용 카드: 테두리, 가운데 십자선, 색 띠. 위치와 색을 눈으로 확인하는 용도."""
    preset = preset or presets.CR80
    document = presets.new_document(preset, name="시험 인쇄")
    side = document.side("front")
    width, height = document.side_size_mm("front")
    bleed = document.bleed_mm
    left, top = bleed + 2.0, bleed + 2.0
    inner_w, inner_h = width - 2 * left, height - 2 * top
    colors = dict(SWATCH_PRESETS)
    black = colors["검정"]
    side.add_item(RectItem(name="테두리", x=left, y=top, w=inner_w, h=inner_h, fill=None,
                           stroke_color=black, stroke_width_mm=0.3))
    side.add_item(LineItem(name="가로 가운데 선", x=left, y=height / 2, w=inner_w, h=0.0))
    side.add_item(LineItem(name="세로 가운데 선", x=width / 2, y=top, w=0.0, h=inner_h))
    side.add_item(TextItem(name="제목", text="하나스튜디오 시험 인쇄", x=left + 3, y=top + 3, w=inner_w - 6, h=8,
                           font_size_pt=12, bold=True, color=black, v_align="middle", auto_shrink=True))
    side.add_item(TextItem(name="안내", text="테두리가 카드 가장자리와 나란한지, 색이 고르게 나왔는지 확인하세요.",
                           x=left + 3, y=top + 12, w=inner_w - 6, h=10, font_size_pt=7, color=black, wrap=True))
    names = ("남색", "파랑", "초록", "빨강", "검정")
    bar_w = (inner_w - 6) / len(names)
    for index, name in enumerate(names):
        side.add_item(RectItem(name=f"색 띠 {name}", x=left + 3 + index * bar_w, y=top + inner_h - 12,
                               w=bar_w, h=8, fill=colors[name], stroke_color=None))
    return document
