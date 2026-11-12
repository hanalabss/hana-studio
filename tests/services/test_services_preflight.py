"""preflight: 규칙마다 담당자가 조치할 수 있는 한국어 문구와 대상 id가 나오는지."""
import pytest

from design import presets
from design.binding import ColumnSource, ConcatPart, ConcatSource
from design.model import BarcodeItem, ImageItem, QRItem, RectItem, TextItem
from printer.base import PrinterStatus
from services import preflight
from services.preflight import ERROR, WARNING
from services.render_job import PrintOptions


def only(report, code):
    found = report.by_code(code)
    assert len(found) == 1, [f.code for f in report.findings]
    return found[0]


def test_clean_job_has_no_findings(store, document, rtai_device):
    status = PrinterStatus(ready=True, message="인쇄할 수 있습니다.", ribbon_remaining=200, feeder_empty=False)
    report = preflight.check(document, store, store.ids(), rtai_device, status)
    assert report.findings == [] and report.ok and report.cards == 5


# ── 명단 ────────────────────────────────────────────────────────────────────
def test_field_bound_to_a_missing_column(store, simple_document):
    side = simple_document.side("front")
    title = side.add_item(TextItem(name="직함", x=5, y=20, w=40, h=8, source=ColumnSource(column="직함")))
    both = side.add_item(TextItem(name="소속", x=5, y=30, w=40, h=8, source=ConcatSource(
        parts=[ConcatPart(column="부서"), ConcatPart(column="직함")])))
    report = preflight.check(simple_document, store)
    finding = only(report, "missing_column")
    assert finding.severity == ERROR and not report.ok
    assert "'직함' 열" in finding.message and "'직함'" in finding.message and "'소속'" in finding.message
    assert finding.item_ids == [title.id, both.id]


def test_hidden_or_non_printing_items_are_not_checked(store, simple_document):
    simple_document.side("front").add_item(
        TextItem(x=5, y=20, w=40, h=8, source=ColumnSource(column="없는열"), printable=False))
    simple_document.side("back").add_item(
        TextItem(x=5, y=20, w=40, h=8, source=ColumnSource(column="뒷면열")))      # 단면 문서의 뒷면
    assert preflight.check(simple_document, store).ok


def test_design_needing_data_without_a_list(simple_document):
    report = preflight.check(simple_document, None)
    assert only(report, "no_data").severity == ERROR
    assert "명단" in only(report, "no_data").message


def test_empty_selection(store, simple_document):
    report = preflight.check(simple_document, store, [])
    assert only(report, "no_records").severity == ERROR and report.cards == 0


def test_deleted_records_in_the_selection(store, simple_document):
    ids = store.ids()
    store.delete([ids[0]])
    finding = only(preflight.check(simple_document, store, ids), "record_missing")
    assert finding.record_ids == [ids[0]]


def test_missing_photos_are_counted_and_named(store, simple_document):
    ids = store.ids()
    store.update(ids[1], {"사진": "nobody"})
    store.update(ids[3], {"사진": ""})
    report = preflight.check(simple_document, store)
    finding = only(report, "missing_photo")
    assert finding.severity == ERROR
    assert finding.record_ids == [ids[1], ids[3]]
    assert "2명" in finding.message and "이영희" in finding.message and "최지우" in finding.message
    assert finding.item_ids == [simple_document.side("front").items[1].id]
    assert report.error_record_ids() == [ids[1], ids[3]]
    # 문제 있는 사람을 빼면 통과한다.
    rest = [i for i in ids if i not in report.error_record_ids()]
    assert preflight.check(simple_document, store, rest).ok


def test_long_name_lists_are_abbreviated(store, simple_document):
    for record_id in store.ids():
        store.update(record_id, {"사진": ""})
    message = only(preflight.check(simple_document, store), "missing_photo").message
    assert "5명" in message and "김철수, 이영희, 박민수 외 2명" in message


def test_only_selected_records_are_checked(store, simple_document):
    ids = store.ids()
    store.update(ids[4], {"사진": "nobody"})
    assert preflight.check(simple_document, store, ids[:3]).ok


def test_empty_required_values_are_a_warning(store, simple_document):
    ids = store.ids()
    store.update(ids[2], {"이름": ""})
    report = preflight.check(simple_document, store)
    finding = only(report, "empty_value")
    assert finding.severity == WARNING and report.ok
    assert finding.record_ids == [ids[2]] and "'이름'" in finding.message and "1명" in finding.message
    assert finding.item_ids == [simple_document.side("front").items[0].id]


# ── 바코드 · QR ─────────────────────────────────────────────────────────────
def _with_barcode(document, **kwargs):
    kwargs.setdefault("source", ColumnSource(column="사번"))
    return document.side("front").add_item(BarcodeItem(name="사번 바코드", x=5, y=35, w=45, h=12, **kwargs))


def test_invalid_barcode_data_per_record(store, simple_document):
    item = _with_barcode(simple_document, symbology="code39")
    ids = store.ids()
    store.update(ids[0], {"사번": "a-001"})          # Code 39는 소문자를 못 쓴다
    store.update(ids[4], {"사번": ""})
    report = preflight.check(simple_document, store)
    finding = only(report, "invalid_barcode")
    assert finding.severity == ERROR
    assert finding.record_ids == [ids[0], ids[4]] and finding.item_ids == [item.id]
    assert "2명" in finding.message and "김철수" in finding.message and "쓸 수 없는 문자" in finding.message
    assert finding.side == "front"


def test_ean13_check_digit_is_validated(store, simple_document):
    _with_barcode(simple_document, symbology="ean13")
    for record_id, value in zip(store.ids(), ["880123456789", "8801234567893", "8801234567890", "12", "x"]):
        store.update(record_id, {"사번": value})
    finding = only(preflight.check(simple_document, store), "invalid_barcode")
    assert finding.record_ids == store.ids()[2:]


def test_barcode_too_narrow_for_its_data(store, simple_document):
    item = _with_barcode(simple_document)
    item.w = 6.0
    finding = only(preflight.check(simple_document, store), "invalid_barcode")
    assert len(finding.record_ids) == 5 and "넓히세요" in finding.message


def test_static_barcode_and_qr_are_checked_once(simple_document, store):
    bad = _with_barcode(simple_document, source=None, data="한글", symbology="code128")
    qr = simple_document.side("front").add_item(QRItem(name="QR", x=60, y=30, w=2, h=2, data="https://example.com/a/b/c"))
    report = preflight.check(simple_document, store)
    findings = report.by_code("invalid_barcode")
    assert [f.item_ids for f in findings] == [[bad.id], [qr.id]]
    assert all(f.record_ids == [] for f in findings)
    assert "키우세요" in findings[1].message


def test_big_lists_are_sampled_for_size_but_validated_for_content(store, simple_document, monkeypatch):
    from design import barcodes

    item = _with_barcode(simple_document, symbology="code39")
    item.w = 60.0
    template = store.rows()[0]
    rows = [{"이름": f"사람{i}", "사번": f"B{i:05d}", "사진": template["사진"]} for i in range(1200)]
    rows[700]["사번"] = "bad value"                                  # 내용 오류: 표본이 아니어도 잡는다
    rows[900]["사번"] = "LONG" * 20                                   # 크기 오류: 가장 긴 값은 항상 검사한다
    store.insert_many(rows)
    ids = store.ids()

    calls = []
    original = barcodes.linear_min_width_px
    monkeypatch.setattr(barcodes, "linear_min_width_px",
                        lambda *args, **kwargs: calls.append(args) or original(*args, **kwargs))
    report = preflight.check(simple_document, store)
    finding = only(report, "invalid_barcode")
    assert finding.record_ids == [ids[5 + 700], ids[5 + 900]]
    assert "일부만 뽑아" in finding.message
    assert len(calls) <= preflight.SAMPLE_SIZE + 150            # 1205건 전부를 재지는 않는다

    calls.clear()
    exact = preflight.check(simple_document, store, ids[:preflight.EXACT_LIMIT])
    assert exact.ok and len(calls) == preflight.EXACT_LIMIT       # 500건까지는 전부 검사


# ── 디자인 ──────────────────────────────────────────────────────────────────
def test_items_outside_the_card(simple_document, store):
    side = simple_document.side("front")
    gone = side.add_item(RectItem(name="밖의 상자", x=120, y=10, w=10, h=10))
    clipped = side.add_item(TextItem(name="잘리는 글자", x=80, y=45, w=20, h=6, text="주소"))
    side.add_item(RectItem(name="배경", x=-5, y=-5, w=100, h=70))            # 배경은 넘쳐도 정상
    report = preflight.check(simple_document, store)
    outside, cut = only(report, "item_outside"), only(report, "item_clipped")
    assert outside.item_ids == [gone.id] and outside.severity == WARNING and "인쇄되지 않습니다" in outside.message
    assert cut.item_ids == [clipped.id] and "잘립니다" in cut.message and "앞면" in cut.message


def test_text_in_the_bleed_area_is_clipped(store):
    document = presets.new_document(presets.CUT_54X86_BLEED)
    item = document.side("front").add_item(TextItem(name="가장자리", x=0.5, y=10, w=20, h=6, text="A"))
    document.side("front").add_item(RectItem(name="바탕", x=0, y=0, w=90, h=58))
    report = preflight.check(document, None)
    assert only(report, "item_clipped").item_ids == [item.id]
    assert "item_outside" not in report.codes() and "bleed_ambiguous" not in report.codes()


def test_legacy_bleed_ratio_is_no_longer_warned_because_drivers_skip_the_second_crop():
    document = presets.new_document(presets.CardPreset("custom", "58x90", 90.0, 58.0))
    assert preflight.CODE_BLEED_AMBIGUOUS == "bleed_ambiguous"
    assert preflight.CODE_BLEED_AMBIGUOUS not in preflight.check(document).codes()


# ── 프린터 기능 ─────────────────────────────────────────────────────────────
def _fancy_document():
    document = presets.new_document(presets.CR80, dual_side=True)
    front = document.side("front")
    white = front.add_item(RectItem(x=5, y=5, w=20, h=20, layer="white"))
    masked = front.add_item(ImageItem(x=30, y=5, w=20, h=20, asset="assets/a.png", mask_asset="assets/m.png"))
    uv = front.add_item(TextItem(x=5, y=30, w=30, h=8, text="UV", layer="uv"))
    back = document.side("back").add_item(TextItem(x=5, y=5, w=30, h=8, text="뒷면"))
    return document, white, masked, uv, back


def test_capability_mismatches_say_what_will_happen(smart_device):
    document, white, masked, uv, back = _fancy_document()
    options = PrintOptions(magnetic={"track2": "사번"})
    report = preflight.check(document, None, None, None, capabilities={"status"}, options=options)
    assert report.ok                                              # 전부 경고다
    assert only(report, "no_white").item_ids == [white.id, masked.id]
    assert "컬러만 인쇄" in only(report, "no_white").message
    assert only(report, "no_uv").item_ids == [uv.id] and "인쇄되지 않습니다" in only(report, "no_uv").message
    assert only(report, "no_dual_side").item_ids == [back.id] and "앞면만" in only(report, "no_dual_side").message
    assert "기록하지 않고" in only(report, "no_magnetic").message

    by_family = preflight.check(document, None, None, smart_device)
    assert {"no_white", "no_uv"} <= by_family.codes()
    assert "no_dual_side" not in by_family.codes()                # 연결 전에는 플리퍼가 있다고 본다


def test_capable_printer_has_no_mismatch(rtai_device):
    document, *_ = _fancy_document()
    report = preflight.check(document, None, None, rtai_device, options=PrintOptions(magnetic={"track1": "x"}))
    assert not {"no_white", "no_uv", "no_dual_side", "no_magnetic"} & report.codes()


def test_magnetic_mapping_to_a_missing_column(store, simple_document, rtai_device):
    report = preflight.check(simple_document, store, None, rtai_device,
                             options=PrintOptions(magnetic={"track1": "카드번호"}))
    assert "카드번호" in only(report, "missing_column").message


# ── 프린터 상태 ─────────────────────────────────────────────────────────────
def test_feeder_empty_and_ribbon_out(store, simple_document):
    status = PrinterStatus(ready=False, message="카드 공급함이 비어 있습니다.", feeder_empty=True,
                           ribbon_remaining=0)
    report = preflight.check(simple_document, store, None, None, status)
    assert only(report, "feeder_empty").severity == ERROR and "카드를 넣" in only(report, "feeder_empty").message
    assert only(report, "ribbon_out").severity == ERROR and "교체" in only(report, "ribbon_out").message
    assert "printer_not_ready" not in report.codes()


def test_printer_not_ready_for_another_reason(store, simple_document):
    status = PrinterStatus(ready=False, message="덮개가 열려 있습니다.", warnings=["청소가 필요합니다."])
    report = preflight.check(simple_document, store, None, None, status)
    assert "덮개가 열려 있습니다." in only(report, "printer_not_ready").message
    assert only(report, "printer_warning").severity == WARNING


@pytest.mark.parametrize("remaining, dual, copies, expected", [
    (4, False, 1, True), (5, False, 1, False), (9, True, 1, True), (10, True, 1, False), (9, False, 2, True),
])
def test_ribbon_remaining_versus_cards(store, simple_document, remaining, dual, copies, expected):
    simple_document.dual_side = dual
    status = PrinterStatus(ready=True, message="ok", ribbon_remaining=remaining, ribbon_unit="cards")
    report = preflight.check(simple_document, store, None, None, status, options=PrintOptions(copies=copies))
    assert ("ribbon_low" in report.codes()) is expected
    if expected:
        finding = only(report, "ribbon_low")
        assert finding.severity == WARNING and str(remaining) in finding.message
        assert f"{5 * copies}장" in finding.message


@pytest.mark.parametrize("unit", ["unknown", "percent"])
def test_ribbon_level_in_an_unknown_unit_is_never_compared_with_the_card_count(store, simple_document, unit):
    # RTAI의 잔량은 SDK 원값이라 장수인지 알 수 없다. 숫자가 작아도 "모자란다"고 말하지 않는다.
    status = PrinterStatus(ready=True, message="ok", ribbon_remaining=3, ribbon_unit=unit)
    report = preflight.check(simple_document, store, None, None, status)
    assert report.cards == 5 and "ribbon_low" not in report.codes() and report.findings == []
    assert PrinterStatus(ready=True, message="ok", ribbon_remaining=3).ribbon_unit == "unknown"

    empty = PrinterStatus(ready=True, message="ok", ribbon_remaining=0, ribbon_unit=unit)
    assert only(preflight.check(simple_document, store, None, None, empty), "ribbon_out").severity == ERROR


def test_real_adapters_only_trigger_ribbon_low_when_the_unit_is_cards(store, simple_document, rtai_device,
                                                                      rtai_lib, smart_device, smart_lib):
    from printer import registry

    rtai_lib.ribbon_remaining = 2
    smart_lib.ribbon_remain = 2
    with registry.open(rtai_device) as printer:
        rtai_status = printer.status()
    with registry.open(smart_device) as printer:
        smart_status = printer.status()
    assert "ribbon_low" not in preflight.check(simple_document, store, None, rtai_device, rtai_status).codes()
    finding = only(preflight.check(simple_document, store, None, smart_device, smart_status), "ribbon_low")
    assert "약 2장" in finding.message


def test_unknown_ribbon_level_is_not_reported(store, simple_document):
    report = preflight.check(simple_document, store, None, None, PrinterStatus(ready=True, message="ok"))
    assert report.findings == []


def test_static_design_counts_copies(rtai_device):
    document = presets.new_document(presets.CR80)
    document.side("front").add_item(TextItem(x=5, y=5, w=30, h=8, text="방문증"))
    status = PrinterStatus(ready=True, message="ok", ribbon_remaining=3, ribbon_unit="cards")
    report = preflight.check(document, None, None, rtai_device, status, options=PrintOptions(copies=10))
    assert report.cards == 10 and "ribbon_low" in report.codes()
