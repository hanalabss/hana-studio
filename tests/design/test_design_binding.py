import datetime as dt
import os

import pytest

from design.binding import (
    BindingContext, ColumnSource, ConcatPart, ConcatSource, CounterSource, DateSource, FixedSource,
    Source, UnknownSource, columns_used, format_date, resolve, to_text,
)
from design.model import BarcodeItem, Document, ImageItem, QRItem, RectItem, TextItem

RECORD = {"이름": "홍길동", "부서": "개발팀", "직급": "", "사번": 1024, "사진": "hong.jpg", "키": 172.0}
FIXED_NOW = dt.datetime(2026, 3, 5, 14, 7, 9)


def ctx(**kwargs) -> BindingContext:
    return BindingContext(now=lambda: FIXED_NOW, **kwargs)


def test_no_source_uses_item_content():
    assert resolve(TextItem(text="고정"), RECORD, ctx()) == "고정"
    assert resolve(TextItem(text="고정"), None) == "고정"
    assert resolve(BarcodeItem(data="ABC"), RECORD, ctx()) == "ABC"
    assert resolve(QRItem(data="qr"), None, ctx()) == "qr"
    assert resolve(ImageItem(asset="assets/a.png"), None, ctx()) == "assets/a.png"
    assert resolve(ImageItem(), RECORD, ctx()) is None
    assert resolve(RectItem(), RECORD, ctx()) is None


def test_fixed_source_overrides_content():
    assert resolve(TextItem(text="x", source=FixedSource(value="문구")), None, ctx()) == "문구"
    assert resolve(ImageItem(source=FixedSource(value="assets/b.png")), None, ctx()) == "assets/b.png"
    assert resolve(ImageItem(source=FixedSource(value="")), RECORD, ctx()) is None


def test_column_source():
    item = TextItem(source=ColumnSource(column="이름"))
    assert resolve(item, RECORD, ctx()) == "홍길동"
    assert resolve(TextItem(source=ColumnSource(column="사번")), RECORD, ctx()) == "1024"
    assert resolve(TextItem(source=ColumnSource(column="키")), RECORD, ctx()) == "172"
    assert resolve(TextItem(source=ColumnSource(column="없는열")), RECORD, ctx()) == ""
    assert resolve(TextItem(source=ColumnSource(column="직급")), RECORD, ctx()) == ""


def test_placeholder_when_record_is_none():
    assert resolve(TextItem(source=ColumnSource(column="이름")), None, ctx()) == "«이름»"
    assert resolve(ImageItem(source=ColumnSource(column="사진")), None, ctx()) is None
    assert resolve(QRItem(source=ColumnSource(column="사번")), None, ctx()) == "«사번»"
    assert resolve(BarcodeItem(data="SAMPLE1", source=ColumnSource(column="사번")), None, ctx()) == "SAMPLE1"
    assert resolve(BarcodeItem(data="SAMPLE1", source=ColumnSource(column="사번")), RECORD, ctx()) == "1024"
    concat = ConcatSource(parts=[ConcatPart(column="부서", suffix=" / "), ConcatPart(column="직급")])
    assert resolve(TextItem(source=concat), None, ctx()) == "«부서» / «직급»"


def test_concat_hides_affixes_of_empty_parts():
    source = ConcatSource(parts=[
        ConcatPart(column="부서", prefix="[", suffix="]"),
        ConcatPart(column="직급", prefix=" (", suffix=")"),
        ConcatPart(text=" - "),
        ConcatPart(column="이름", suffix=" 님"),
        ConcatPart(column="없는열", prefix="#"),
    ])
    assert resolve(TextItem(source=source), RECORD, ctx()) == "[개발팀] - 홍길동 님"
    blank = dict(RECORD, 직급="   ", 부서=None)
    assert resolve(TextItem(source=source), blank, ctx()) == " - 홍길동 님"


def test_concat_separator_skips_empty_parts():
    source = ConcatSource(separator=" · ", parts=[
        ConcatPart(column="부서"), ConcatPart(column="직급"), ConcatPart(column="이름"),
    ])
    assert resolve(TextItem(source=source), RECORD, ctx()) == "개발팀 · 홍길동"
    assert resolve(TextItem(source=source), dict(RECORD, 직급="과장"), ctx()) == "개발팀 · 과장 · 홍길동"


@pytest.mark.parametrize("index, source, expected", [
    (0, CounterSource(), "1"),
    (4, CounterSource(), "5"),
    (0, CounterSource(start=100, step=10, pad=6), "000100"),
    (3, CounterSource(start=100, step=10, pad=6), "000130"),
    (2, CounterSource(start=5, step=-5, pad=3), "-005"),
    (7, CounterSource(start=1, pad=4, prefix="No. ", suffix="호"), "No. 0008호"),
    (0, CounterSource(start=12345, pad=3), "12345"),
])
def test_counter(index, source, expected):
    item = TextItem(source=source)
    assert resolve(item, RECORD, ctx(counter_index=index)) == expected
    assert resolve(item, None, ctx(counter_index=index)) == expected


@pytest.mark.parametrize("fmt, expected", [
    ("%Y-%m-%d", "2026-03-05"),
    ("%Y년 %m월 %d일", "2026년 03월 05일"),
    ("%Y년 %-m월 %-d일 (%a)", "2026년 3월 5일 (목)"),
    ("%y.%#m.%#d %H:%M:%S", "26.3.5 14:07:09"),
    ("%p %I시 %A", "오후 02시 목요일"),
    ("100%% %j %Q", "100% 064 %Q"),
])
def test_date_format(fmt, expected):
    assert format_date(FIXED_NOW, fmt) == expected
    assert resolve(TextItem(source=DateSource(format=fmt)), None, ctx()) == expected


def test_date_offset_days():
    item = TextItem(source=DateSource(format="%Y-%m-%d", offset_days=365))
    assert resolve(item, RECORD, ctx()) == "2027-03-05"


def test_image_column_uses_photo_base_dir():
    item = ImageItem(source=ColumnSource(column="사진"))
    assert resolve(item, RECORD, ctx()) == "hong.jpg"
    assert resolve(item, RECORD, ctx(photo_base_dir="D:/photos")) == os.path.join("D:/photos", "hong.jpg")
    absolute = os.path.abspath("x.png")
    assert resolve(item, dict(RECORD, 사진=absolute), ctx(photo_base_dir="D:/photos")) == absolute
    assert resolve(item, dict(RECORD, 사진="  "), ctx(photo_base_dir="D:/photos")) is None
    assert resolve(item, dict(RECORD, 사진=None), ctx()) is None
    by_number = ImageItem(source=ConcatSource(parts=[ConcatPart(column="사번", suffix=".png")]))
    assert resolve(by_number, RECORD, ctx(photo_base_dir="p")) == os.path.join("p", "1024.png")
    assert resolve(by_number, None, ctx(photo_base_dir="p")) is None


def test_to_text():
    assert to_text(None) == ""
    assert to_text(3.0) == "3"
    assert to_text(3.25) == "3.25"
    assert to_text(float("nan")) == ""
    assert to_text(dt.date(2026, 1, 2)) == "2026-01-02"
    assert to_text(dt.datetime(2026, 1, 2, 3, 4, 5)) == "2026-01-02 03:04:05"
    assert to_text(True) == "예"


def test_source_dict_dispatch_and_unknown_type():
    for source in (
        FixedSource(value="a"), ColumnSource(column="c"), CounterSource(start=3, step=2, pad=5),
        DateSource(format="%Y"), ConcatSource(parts=[ConcatPart(column="x", prefix="p")], separator=","),
    ):
        restored = Source.from_dict(source.to_dict())
        assert type(restored) is type(source) and restored == source

    unknown = Source.from_dict({"type": "lookup", "table": "t"})
    assert isinstance(unknown, UnknownSource)
    assert unknown.to_dict() == {"type": "lookup", "table": "t"}
    assert resolve(TextItem(text="x", source=unknown), RECORD, ctx()) == ""
    with pytest.raises(ValueError):
        Source.from_dict({"type": "counter", "start": "1"})


def test_columns_used():
    document = Document(dual_side=True)
    front, back = document.side("front"), document.side("back")
    front.add_item(TextItem(source=ColumnSource(column="이름")))
    front.add_item(ImageItem(source=ColumnSource(column="사진")))
    front.add_item(TextItem(text="고정"))
    front.add_item(TextItem(source=CounterSource()))
    front.add_item(TextItem(source=ColumnSource(column="")))
    back.add_item(QRItem(source=ConcatSource(parts=[
        ConcatPart(column="부서"), ConcatPart(text="-"), ConcatPart(column="이름"),
    ])))
    assert columns_used(document) == {"이름", "사진", "부서"}
    assert columns_used(Document()) == set()
