"""다눌 디자인(.dnc) 열기. 예시 파일은 다눌 소스의 XmlSerializer 규칙대로 만든 것이다(실제 다눌 파일로는 아직 확인 전)."""

import shutil
from pathlib import Path

import pytest
from PIL import Image

from design.binding import ColumnSource
from design.danul import DanulError, load_dnc, parse_font

SAMPLE = Path(__file__).parent / "data" / "danul_sample.dnc"


@pytest.fixture
def sample(tmp_path):
    path = tmp_path / "사원증.dnc"
    shutil.copy(SAMPLE, path)
    Image.new("RGB", (40, 16), (10, 20, 200)).save(tmp_path / "logo.png")
    return path


def test_font_string():
    assert parse_font("맑은 고딕, 14pt, style=Bold, Strikeout") == ("맑은 고딕", 14.0, {"bold", "strikeout"})
    assert parse_font("Gulim, 9pt") == ("Gulim", 9.0, set())


def test_portrait_card_and_scaled_items(sample):
    result = load_dnc(sample)
    doc = result.document
    assert doc.name == "사원증" and doc.side("front").orientation == "portrait"
    assert result.mdb_path.endswith("사원.mdb")
    kinds = [it.kind for it in doc.side("front").sorted_items()]
    assert kinds == ["image", "image", "text", "rect", "line", "barcode"]
    k = 85.6 / 340
    logo, photo, name, band, line, code = doc.side("front").sorted_items()
    assert (logo.x, logo.w) == pytest.approx((10 * k, 100 * k))
    assert logo.asset and logo.asset.startswith("assets/") and logo.fit == "stretch" and logo.locked
    assert logo.corner_radius_mm == pytest.approx(10 * k)
    assert isinstance(photo.source, ColumnSource) and photo.source.column == "사진" and photo.rotation == 270
    assert name.text == "홍길동" and name.source.column == "이름" and name.bold and name.strikeout
    assert name.color == "#1a2b3c" and name.background == "#ffe0e0" and name.h_align == "center"
    assert band.fill == "#000080" and band.stroke_color is None and band.corner_radius_mm == pytest.approx(5 * k)
    assert line.h == 0 and line.color == "#ff0000"
    assert code.symbology == "code93" and code.show_text and code.source.column == "사번"
    assert doc.extra["magnetic"] == {"track2": "사번"}
    assert doc.extra["mifare"]["column"] == "사번" and doc.extra["mifare"]["block"] == 8
    qr = doc.side("back").items[0]
    assert qr.kind == "qr" and qr.w == qr.h and doc.dual_side
    assert any("키" in note for note in result.notes)


def test_missing_image_becomes_empty_frame(sample):
    (sample.parent / "logo.png").unlink()
    result = load_dnc(sample)
    logo = result.document.side("front").sorted_items()[0]
    assert logo.asset is None and any("logo.png" in note for note in result.notes)


def test_converted_design_renders_and_saves(sample, tmp_path):
    from design import renderer, storage

    doc = load_dnc(sample).document
    renderer.render_side(doc, "front", record={"이름": "김하나", "사번": "A002"})
    saved = storage.save(doc, tmp_path / "out.hsd")
    again = storage.load(saved)
    assert len(again.side("front").items) == 6


def test_not_a_danul_file(tmp_path):
    bad = tmp_path / "x.dnc"
    bad.write_text("<Other/>", encoding="utf-8")
    with pytest.raises(DanulError):
        load_dnc(bad)
    bad.write_text("not xml", encoding="utf-8")
    with pytest.raises(DanulError):
        load_dnc(bad)
