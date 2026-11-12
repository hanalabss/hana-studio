"""인쇄 전 점검. 담당자가 바로 조치할 수 있는 한국어 문구로 문제를 알려 준다."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from data.photos import PhotoResolver
from data.store import DataStore
from data.values import is_empty
from design import barcodes, renderer
from design.binding import BindingContext, resolve, to_text
from design.model import Document, Item
from printer.base import (
    CAP_DUAL_SIDE, CAP_MAGNETIC, CAP_UV, CAP_WHITE, PrinterDevice, PrinterStatus,
)
from services.printers import default_capabilities
from services.render_job import (
    DEFAULT_DPI, PrintOptions, guess_name_column, magnetic_mapping,
    prints_back, records_by_id, uses_uv, uses_white,
)

ERROR = "error"
WARNING = "warning"

# 예전 점검 코드. 드라이버가 완성된 이미지를 한 번 더 자르던 때의 경고로, 지금은 RenderJob이
# options["bleed_handled"]를 달고 어댑터가 도련 제거를 건너뛰므로 check()가 더는 내지 않는다.
CODE_BLEED_AMBIGUOUS = "bleed_ambiguous"

EXACT_LIMIT = 500          # 이 건수까지는 바코드를 전부 검사한다
SAMPLE_SIZE = 500          # 그보다 많으면 크기 검사는 이만큼만 뽑아서 한다
NAMES_SHOWN = 3
_EDGE_TOLERANCE_MM = 0.05
_KIND_LABELS = {
    "text": "글자", "image": "이미지", "rect": "사각형", "ellipse": "원", "line": "선",
    "barcode": "바코드", "qr": "QR 코드",
}
_SIDE_LABELS = {"front": "앞면", "back": "뒷면"}


@dataclass
class Finding:
    """점검 결과 한 건. record_ids / item_ids로 화면이 해당 레코드·항목으로 이동할 수 있다."""

    code: str
    severity: str
    message: str
    record_ids: list[int] = field(default_factory=list)
    item_ids: list[str] = field(default_factory=list)
    side: str | None = None

    @property
    def blocking(self) -> bool:
        return self.severity == ERROR


@dataclass
class PreflightReport:
    findings: list[Finding] = field(default_factory=list)
    cards: int = 0

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == ERROR]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == WARNING]

    @property
    def ok(self) -> bool:
        """막는 오류가 없다(경고는 있을 수 있다)."""
        return not self.errors

    def by_code(self, code: str) -> list[Finding]:
        return [f for f in self.findings if f.code == code]

    def codes(self) -> set[str]:
        return {f.code for f in self.findings}

    def error_record_ids(self) -> list[int]:
        """오류가 걸린 레코드. "문제 있는 사람은 빼고 인쇄"에 쓴다."""
        ids: dict[int, None] = {}
        for finding in self.errors:
            for record_id in finding.record_ids:
                ids[record_id] = None
        return list(ids)

    def add(self, code: str, severity: str, message: str, record_ids: Iterable[int] = (),
            item_ids: Iterable[str] = (), side: str | None = None) -> Finding:
        finding = Finding(code, severity, message, list(record_ids), list(item_ids), side)
        self.findings.append(finding)
        return finding


def _label(item: Item) -> str:
    return item.name or _KIND_LABELS.get(item.kind, item.kind)


def _names(records: list[Mapping[str, Any]], name_column: str | None) -> str:
    """'김철수, 이영희, 박민수 외 4명' 꼴."""
    shown = []
    for record in records[:NAMES_SHOWN]:
        text = to_text(record.get(name_column)).strip() if name_column else ""
        shown.append(text or f"{record.get('_id')}번 레코드")
    text = ", ".join(shown)
    rest = len(records) - len(shown)
    return f"{text} 외 {rest}명" if rest > 0 else text


def _active_items(document: Document):
    for side_key in document.active_side_keys():
        for item in document.side(side_key).items:
            if item.visible and item.printable:
                yield side_key, item


def check(
    document: Document,
    store: DataStore | None = None,
    record_ids: Iterable[int] | None = None,
    device: PrinterDevice | None = None,
    printer_status: PrinterStatus | None = None,
    *,
    capabilities: Iterable[str] | None = None,
    options: PrintOptions | None = None,
) -> PreflightReport:
    """인쇄 전에 디자인 · 명단 · 프린터를 점검한다.

    record_ids: 인쇄할 레코드. None이면 명단 전체(명단이 없으면 options.copies장).
    capabilities: 프린터 지원 기능(PrinterService.capabilities()). 없으면 device의 제품군 기본값.
    """
    options = options or PrintOptions()
    report = PreflightReport()
    copies = max(1, int(options.copies))

    columns = set(store.column_names()) if store is not None else set()
    bound = [(side, item) for side, item in _active_items(document) if item.source and item.source.columns()]

    records: list[dict[str, Any]] = []
    if store is not None:
        if record_ids is None:
            records = store.rows()
        else:
            wanted = [int(i) for i in record_ids]
            found = records_by_id(store, wanted)
            records = [found[i] for i in dict.fromkeys(wanted) if i in found]
            gone = [i for i in dict.fromkeys(wanted) if i not in found]
            if gone:
                report.add("record_missing", ERROR,
                           f"명단에서 지워진 레코드가 {len(gone)}건 있습니다. 인쇄 대상을 다시 선택하세요.",
                           record_ids=gone)
    uses_data = bool(bound) or bool(magnetic_mapping(document, options))
    report.cards = len(records) * copies if (store is not None and (uses_data or record_ids is not None)) else copies

    if store is None and bound:
        report.add("no_data", ERROR,
                   "이 디자인은 명단의 값을 넣어 인쇄합니다. 명단을 먼저 열거나 가져오세요.",
                   item_ids=[item.id for _, item in bound])
    elif store is not None and (uses_data or record_ids is not None) and not records:
        report.add("no_records", ERROR, "인쇄할 레코드가 없습니다. 인쇄 대상을 선택하세요.")

    name_column = guess_name_column(store, options.name_column)
    if store is not None:
        _check_columns(report, bound, columns)
        _check_magnetic_columns(report, document, options, columns)
        if records:
            _check_photos(report, store, bound, records, name_column)
            _check_empty_values(report, store, bound, records, name_column, columns)
    _check_codes(report, document, store, records, name_column, columns, options)
    _check_bounds(report, document)
    _check_capabilities(report, document, device, capabilities, options)
    _check_printer(report, document, printer_status)
    return report


# ---------------------------------------------------------------- 명단

def _check_columns(report: PreflightReport, bound, columns: set[str]) -> None:
    missing: dict[str, list[Item]] = {}
    for _, item in bound:
        for column in sorted(item.source.columns()):
            if column not in columns:
                missing.setdefault(column, []).append(item)
    for column, items in missing.items():
        labels = ", ".join(dict.fromkeys(f"'{_label(item)}'" for item in items))
        report.add(
            "missing_column", ERROR,
            f"디자인의 {labels} 항목이 명단에 없는 '{column}' 열을 사용합니다. "
            f"명단에 '{column}' 열을 추가하거나 디자인에서 다른 열을 고르세요.",
            item_ids=[item.id for item in items],
        )


def _check_magnetic_columns(report, document, options, columns: set[str]) -> None:
    for track, column in magnetic_mapping(document, options).items():
        if column not in columns:
            report.add("missing_column", ERROR,
                       f"마그네틱 {track}에 넣을 '{column}' 열이 명단에 없습니다. 인쇄 옵션에서 열을 다시 고르세요.")


def _check_photos(report, store: DataStore, bound, records, name_column) -> None:
    photo_columns = {c.name for c in store.columns() if c.type == "photo"}
    resolver = PhotoResolver(store)
    by_column: dict[str, list[Item]] = {}
    for _, item in bound:
        if item.kind != "image":
            continue
        for column in item.source.columns():
            if column in photo_columns:
                by_column.setdefault(column, []).append(item)
    for column, items in by_column.items():
        missing = [record for record in records if resolver.path(record, column) is None]
        if not missing:
            continue
        report.add(
            "missing_photo", ERROR,
            f"사진이 없는 사람이 {len(missing)}명 있습니다: {_names(missing, name_column)}. "
            f"'{column}' 열의 파일 이름과 사진 폴더를 확인하세요. 그대로 인쇄하면 사진 자리가 비어 나옵니다.",
            record_ids=[record["_id"] for record in missing],
            item_ids=[item.id for item in items],
        )


def _check_empty_values(report, store: DataStore, bound, records, name_column, columns) -> None:
    photo_columns = {c.name for c in store.columns() if c.type == "photo"}
    by_column: dict[str, list[Item]] = {}
    for _, item in bound:
        if item.kind in ("barcode", "qr"):
            continue
        for column in item.source.columns():
            if column in columns and column not in photo_columns:
                by_column.setdefault(column, []).append(item)
    for column, items in by_column.items():
        empty = [record for record in records if is_empty(record.get(column))]
        if not empty:
            continue
        report.add(
            "empty_value", WARNING,
            f"'{column}' 값이 비어 있는 사람이 {len(empty)}명 있습니다: {_names(empty, name_column)}. "
            "그대로 인쇄하면 그 자리가 빈 채로 나옵니다.",
            record_ids=[record["_id"] for record in empty],
            item_ids=[item.id for item in items],
        )


# ---------------------------------------------------------------- 바코드 · QR

def _code_problem(item: Item, value: Any, dpi: float, check_size: bool = True) -> str | None:
    data = "" if value is None else str(value)
    scale = renderer.mm_to_px_for(dpi)
    try:
        if item.kind == "barcode":
            shown = barcodes.normalize(item.symbology, data)
            if check_size:
                need = barcodes.linear_min_width_px(item.symbology, shown, item.quiet_zone)
                have = round(item.w * scale)
                if need > have:
                    return (f"바코드 칸이 좁아 '{shown}'을(를) 담을 수 없습니다. "
                            f"칸 너비를 {need / scale:.1f}mm 이상으로 넓히세요.")
        elif check_size:
            need = barcodes.qr_min_size_px(data, item.error_correction, item.border_modules)
            have = round(min(item.w, item.h) * scale)
            if need > have:
                return (f"QR 코드 칸이 작아 내용을 담을 수 없습니다. "
                        f"한 변을 {need / scale:.1f}mm 이상으로 키우세요.")
        elif data == "":
            return "QR 코드 데이터가 비어 있습니다."
    except barcodes.BarcodeError as exc:
        return str(exc)
    return None


def _sample_positions(values: list[str]) -> set[int]:
    """크기 검사를 할 레코드 위치: 가장 긴 값들 + 고르게 뽑은 표본."""
    total = len(values)
    longest = sorted(range(total), key=lambda i: len(values[i]), reverse=True)[: SAMPLE_SIZE // 5]
    step = max(1, total // (SAMPLE_SIZE - len(longest)))
    return set(longest) | set(range(0, total, step))


def _check_codes(report, document, store, records, name_column, columns, options) -> None:
    dpi = options.dpi or DEFAULT_DPI
    resolver = PhotoResolver(store) if store is not None else None
    bound_records: list[dict[str, Any]] | None = None
    for side_key, item in _active_items(document):
        if item.kind not in ("barcode", "qr"):
            continue
        label = _label(item)
        needs_record = bool(item.source and item.source.columns())
        varies = needs_record or (item.source is not None and item.source.type == "counter")
        if not varies or store is None or not records:
            if needs_record:
                continue
            value = resolve(item, {}, BindingContext(counter_index=options.counter_offset))
            problem = _code_problem(item, value, dpi)
            if problem:
                report.add("invalid_barcode", ERROR, f"'{label}': {problem}", item_ids=[item.id], side=side_key)
            continue
        if not item.source.columns() <= columns and needs_record:
            continue
        if bound_records is None:
            bound_records = [resolver.binding_record(record) for record in records]
        values = [
            to_text(resolve(item, bound_records[i], BindingContext(counter_index=options.counter_offset + i)))
            for i in range(len(records))
        ]
        sampled = len(records) > EXACT_LIMIT
        size_checked = _sample_positions(values) if sampled else None
        bad: list[int] = []
        reasons: dict[str, None] = {}
        cache: dict[tuple[str, bool], str | None] = {}
        for position, value in enumerate(values):
            check_size = size_checked is None or position in size_checked
            key = (value, check_size)
            if key not in cache:
                cache[key] = _code_problem(item, value, dpi, check_size)
            if cache[key]:
                bad.append(position)
                reasons.setdefault(cache[key], None)
        if not bad:
            continue
        bad_records = [records[i] for i in bad]
        note = f" (전체 {len(records)}건 중 일부만 뽑아 크기를 검사했습니다.)" if sampled else ""
        report.add(
            "invalid_barcode", ERROR,
            f"'{label}'에 넣을 수 없는 값이 {len(bad)}명 있습니다: {_names(bad_records, name_column)}. "
            f"{next(iter(reasons))}{note}",
            record_ids=[record["_id"] for record in bad_records], item_ids=[item.id], side=side_key,
        )


# ---------------------------------------------------------------- 디자인

def _check_bounds(report: PreflightReport, document: Document) -> None:
    eps = _EDGE_TOLERANCE_MM
    for side_key, item in _active_items(document):
        left, top, right, bottom = item.bounds()
        c_left, c_top, c_right, c_bottom = document.side_bounds(side_key, inside_bleed=True)
        side_label = _SIDE_LABELS.get(side_key, side_key)
        outside = right <= c_left + eps or bottom <= c_top + eps or left >= c_right - eps or top >= c_bottom - eps
        if outside:
            report.add("item_outside", WARNING,
                       f"{side_label}의 '{_label(item)}' 항목이 카드 밖에 있어 인쇄되지 않습니다. "
                       "카드 안으로 옮기거나 지우세요.", item_ids=[item.id], side=side_key)
            continue
        if item.kind not in ("text", "barcode", "qr"):
            continue
        if left < c_left - eps or top < c_top - eps or right > c_right + eps or bottom > c_bottom + eps:
            report.add("item_clipped", WARNING,
                       f"{side_label}의 '{_label(item)}' 항목이 카드 가장자리를 넘어가 일부가 잘립니다. "
                       "카드 안쪽으로 옮기세요.", item_ids=[item.id], side=side_key)


# ---------------------------------------------------------------- 프린터

def _check_capabilities(report, document, device, capabilities, options) -> None:
    if capabilities is None:
        if device is None:
            return
        capabilities = default_capabilities(device)
    caps = set(capabilities)
    if uses_white(document) and CAP_WHITE not in caps:
        items = [item.id for _, item in _active_items(document)
                 if item.layer == "white" or getattr(item, "mask_asset", None)]
        report.add("no_white", WARNING,
                   "이 프린터는 화이트 인쇄를 지원하지 않습니다. 화이트 레이어는 빼고 컬러만 인쇄됩니다.",
                   item_ids=items)
    if uses_uv(document) and CAP_UV not in caps:
        items = [item.id for _, item in _active_items(document) if item.layer == "uv"]
        report.add("no_uv", WARNING,
                   "이 프린터(또는 장착된 리본)는 UV(형광) 인쇄를 지원하지 않습니다. UV 항목은 인쇄되지 않습니다.",
                   item_ids=items)
    if prints_back(document) and CAP_DUAL_SIDE not in caps:
        report.add("no_dual_side", WARNING,
                   "이 프린터는 양면 인쇄를 지원하지 않습니다. 뒷면은 인쇄되지 않고 앞면만 인쇄됩니다.",
                   item_ids=[item.id for item in document.side("back").items], side="back")
    if magnetic_mapping(document, options) and CAP_MAGNETIC not in caps:
        report.add("no_magnetic", WARNING,
                   "이 프린터에는 마그네틱 기록 장치가 없습니다. 마그네틱 정보는 기록하지 않고 인쇄만 합니다.")


def _check_printer(report: PreflightReport, document: Document, status: PrinterStatus | None) -> None:
    if status is None:
        return
    specific = False
    if status.feeder_empty:
        specific = True
        report.add("feeder_empty", ERROR, "카드 공급함이 비어 있습니다. 카드를 넣은 뒤 다시 시도하세요.")
    if status.ribbon_remaining is not None and status.ribbon_remaining <= 0:
        specific = True
        report.add("ribbon_out", ERROR, "리본을 다 썼습니다. 새 리본으로 교체한 뒤 다시 시도하세요.")
    if not status.ready and not specific:
        report.add("printer_not_ready", ERROR,
                   f"프린터가 인쇄할 수 없는 상태입니다: {status.message or '상태를 확인하세요.'}")
    for warning in status.warnings:
        report.add("printer_warning", WARNING, f"프린터 경고: {warning}")
    # 잔량을 장수로 믿을 수 있을 때만 비교한다. 단위를 모르는 값(RTAI의 SDK 원값 등)이나 %는
    # 카드 수와 견줄 수 없으므로 "모자랄 수 있다"고 말하지 않는다.
    cards_left = status.ribbon_cards_left
    if cards_left is not None and cards_left > 0 and report.cards:
        needed = report.cards * (2 if prints_back(document) else 1)
        if cards_left < needed:
            report.add(
                "ribbon_low", WARNING,
                f"리본이 약 {cards_left}장 분량 남았는데 인쇄할 카드는 {report.cards}장"
                f"{'(양면)' if prints_back(document) else ''}입니다. 도중에 리본을 교체해야 할 수 있습니다.",
            )
