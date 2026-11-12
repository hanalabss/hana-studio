import io
import json
import os
import zipfile

import numpy as np
import pytest
from PIL import Image

from design import storage
from design.binding import ColumnSource
from design.model import CURRENT_VERSION, Document, ImageItem, RectItem, TextItem
from design.presets import CR80, CUT_54X86, CUT_54X86_BLEED
from design.renderer import render_side
from design.storage import StorageError


@pytest.fixture
def workdir(tmp_path):
    path = tmp_path / "work"
    (path / "assets").mkdir(parents=True)
    return str(path)


def sample_document(workdir, make_png) -> Document:
    document = Document(name="저장 시험", dual_side=True, base_dir=workdir)
    photo = storage.import_asset(workdir, make_png(size=(60, 40), color=(200, 30, 30)))
    mask = storage.import_asset(workdir, make_png(size=(60, 40), color=255, mode="L"))
    front = document.side("front")
    front.add_item(ImageItem(x=5, y=5, w=30, h=20, asset=photo, mask_asset=mask))
    front.add_item(TextItem(x=40, y=5, w=40, h=8, source=ColumnSource(column="이름")))
    document.side("back").add_item(RectItem(x=1, y=1, w=10, h=10, layer="uv"))
    return document


def write_zip(path, entries: dict[str, bytes]) -> str:
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return str(path)


def document_json(**patch) -> bytes:
    data = Document().to_dict()
    data.update(patch)
    return json.dumps(data, ensure_ascii=False).encode("utf-8")


# ---------------------------------------------------------------- import_asset

def test_import_asset_copies_into_assets_and_dedupes_by_content(workdir, tmp_path, make_png):
    first = make_png(size=(10, 10), color=(1, 2, 3), name="a.png")
    twin = tmp_path / "다른 이름.PNG"
    twin.write_bytes(open(first, "rb").read())
    other = make_png(size=(10, 10), color=(3, 2, 1), name="b.png")

    rel_first = storage.import_asset(workdir, first)
    assert rel_first.startswith("assets/") and rel_first.endswith(".png") and "\\" not in rel_first
    assert open(os.path.join(workdir, rel_first), "rb").read() == open(first, "rb").read()

    assert storage.import_asset(workdir, str(twin)) == rel_first
    assert storage.import_asset(workdir, first) == rel_first
    rel_other = storage.import_asset(workdir, other)
    assert rel_other != rel_first
    assert sorted(os.listdir(os.path.join(workdir, "assets"))) == sorted(
        [os.path.basename(rel_first), os.path.basename(rel_other)]
    )


def test_import_asset_creates_assets_dir_and_rejects_missing_file(tmp_path, make_png):
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    rel = storage.import_asset(fresh, make_png())
    assert (fresh / rel).is_file()
    with pytest.raises(StorageError, match="찾을 수 없습니다"):
        storage.import_asset(fresh, tmp_path / "없는파일.png")


# ---------------------------------------------------------------- save / load

def test_save_writes_contract_layout(workdir, tmp_path, make_png):
    document = sample_document(workdir, make_png)
    target = storage.save(document, tmp_path / "card.hsd")
    with zipfile.ZipFile(target) as archive:
        names = set(archive.namelist())
        assert {"document.json", "thumbnail.png"} <= names
        assert names - {"document.json", "thumbnail.png"} == document.asset_refs()
        saved = json.loads(archive.read("document.json").decode("utf-8"))
        assert saved["version"] == CURRENT_VERSION and saved["name"] == "저장 시험"
        assert "base_dir" not in saved
        with Image.open(io.BytesIO(archive.read("thumbnail.png"))) as thumbnail:
            assert thumbnail.format == "PNG"
            assert thumbnail.width == pytest.approx(storage.THUMBNAIL_WIDTH_PX, abs=1)
            assert np.array(thumbnail.convert("RGB")).min() < 255
    assert [p.name for p in tmp_path.iterdir() if p.suffix == ".tmp"] == []


def test_round_trip_restores_document_and_assets(workdir, tmp_path, make_png):
    document = sample_document(workdir, make_png)
    target = storage.save(document, tmp_path / "card.hsd")
    loaded = storage.load(target)
    try:
        assert loaded == document
        assert loaded.to_dict() == document.to_dict()
        assert loaded.base_dir and os.path.abspath(loaded.base_dir) != os.path.abspath(workdir)
        for ref in document.asset_refs():
            original = open(os.path.join(workdir, ref), "rb").read()
            assert open(os.path.join(loaded.base_dir, ref), "rb").read() == original

        before, after = render_side(document, "front", record={}), render_side(loaded, "front", record={})
        assert before.color.tobytes() == after.color.tobytes()
        assert before.white_mask.tobytes() == after.white_mask.tobytes()
        assert after.warnings == []

        again = storage.save(loaded, tmp_path / "again.hsd")
        with zipfile.ZipFile(target) as a, zipfile.ZipFile(again) as b:
            assert a.read("document.json") == b.read("document.json")
            assert sorted(a.namelist()) == sorted(b.namelist())
    finally:
        base = loaded.base_dir
        storage.cleanup_workdir(loaded)
        assert not os.path.exists(base)


def test_load_into_given_workdir(workdir, tmp_path, make_png):
    target = storage.save(sample_document(workdir, make_png), tmp_path / "card.hsd")
    destination = tmp_path / "opened"
    loaded = storage.load(target, workdir=destination)
    assert os.path.samefile(loaded.base_dir, destination)
    assert len(os.listdir(destination / "assets")) == 2
    storage.cleanup_workdir(loaded)
    assert destination.exists(), "직접 지정한 폴더는 지우지 않는다"


def test_save_overwrites_atomically_and_keeps_old_file_on_failure(workdir, tmp_path, make_png):
    document = sample_document(workdir, make_png)
    target = tmp_path / "card.hsd"
    storage.save(document, target, thumbnail=False)
    original = target.read_bytes()
    with zipfile.ZipFile(target) as archive:
        assert "thumbnail.png" not in archive.namelist()

    document.side("front").add_item(ImageItem(asset="assets/사라진파일.png"))
    with pytest.raises(StorageError, match="사라진파일.png"):
        storage.save(document, target)
    assert target.read_bytes() == original
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".hsd_")] == []

    document.side("front").items.pop()
    document.name = "새 이름"
    storage.save(document, target)
    assert storage.load(target, workdir=tmp_path / "opened").name == "새 이름"


def test_document_without_assets_saves_without_workdir(tmp_path):
    document = Document(name="글자만")
    document.side("front").add_item(TextItem(text="안녕하세요", x=5, y=5, w=40, h=10))
    target = storage.save(document, tmp_path / "sub" / "text.hsd")
    loaded = storage.load(target)
    assert loaded == document and os.path.isdir(os.path.join(loaded.base_dir, "assets"))
    assert storage.read_thumbnail(target).startswith(b"\x89PNG")
    storage.cleanup_workdir(loaded)


def test_unknown_fields_and_newer_version_survive_file_round_trip(tmp_path):
    data = Document().to_dict()
    data["version"] = CURRENT_VERSION + 3
    data["future_setting"] = {"encoding": "contactless"}
    data["sides"]["front"]["items"] = [
        {"kind": "hologram", "id": "h1", "x": 1, "y": 2, "w": 3, "h": 4, "pattern": "wave"},
        {"kind": "text", "id": "t1", "text": "가", "future_text_option": [1, 2, 3]},
    ]
    source = write_zip(tmp_path / "future.hsd", {
        "document.json": json.dumps(data).encode("utf-8"),
        "future/extra.bin": b"ignored",
    })
    loaded = storage.load(source)
    assert loaded.is_newer_format and loaded.version == CURRENT_VERSION + 3
    assert render_side(loaded, "front").warnings == []

    resaved = storage.save(loaded, tmp_path / "resaved.hsd")
    with zipfile.ZipFile(resaved) as archive:
        again = json.loads(archive.read("document.json").decode("utf-8"))
    assert again["version"] == CURRENT_VERSION + 3
    assert again["future_setting"] == {"encoding": "contactless"}
    items = again["sides"]["front"]["items"]
    assert (items[0]["kind"], items[0]["pattern"]) == ("hologram", "wave")
    assert items[1]["future_text_option"] == [1, 2, 3]
    storage.cleanup_workdir(loaded)


# ---------------------------------------------------------------- 안전성

@pytest.mark.parametrize("evil", [
    "../evil.txt",
    "assets/../../evil.txt",
    "assets/../evil.txt",
    "/abs/evil.txt",
    "C:/evil.txt",
    "assets\\..\\..\\evil.txt",
    "..\\evil.txt",
    "assets/sub/../../../evil.txt",
    "assets/x.png:stream",
])
def test_path_traversal_entries_are_rejected(tmp_path, evil):
    container = tmp_path / "inner" / "deep"
    container.mkdir(parents=True)
    source = write_zip(container / "evil.hsd", {"document.json": document_json(), evil: b"pwned"})
    destination = container / "work"
    with pytest.raises(StorageError, match="안전하지 않은 경로"):
        storage.load(source, workdir=destination)
    leaked = [p for p in tmp_path.rglob("*") if p.is_file() and p.name != "evil.hsd"]
    assert leaked == []

    with pytest.raises(StorageError, match="안전하지 않은 경로"):
        storage.load(source)


def test_safe_nested_asset_is_extracted(tmp_path):
    source = write_zip(tmp_path / "ok.hsd", {"document.json": document_json(), "assets/sub/a.bin": b"data"})
    loaded = storage.load(source, workdir=tmp_path / "w")
    assert (tmp_path / "w" / "assets" / "sub" / "a.bin").read_bytes() == b"data"
    assert loaded.base_dir


def test_broken_files_raise_korean_errors(tmp_path):
    not_zip = tmp_path / "plain.hsd"
    not_zip.write_text("그냥 글자", encoding="utf-8")
    with pytest.raises(StorageError, match="손상"):
        storage.load(not_zip)
    with pytest.raises(StorageError, match="찾을 수 없습니다"):
        storage.load(tmp_path / "없음.hsd")
    with pytest.raises(StorageError, match="document.json 없음"):
        storage.load(write_zip(tmp_path / "empty.hsd", {"assets/a.png": b"x"}))
    with pytest.raises(StorageError, match="문서 내용을 읽을 수 없습니다"):
        storage.load(write_zip(tmp_path / "badjson.hsd", {"document.json": b"{not json"}))
    with pytest.raises(StorageError, match="문서 내용을 읽을 수 없습니다"):
        storage.load(write_zip(tmp_path / "badtype.hsd", {"document.json": document_json(card_width_mm="넓게")}))
    with pytest.raises(StorageError, match="문서 내용을 읽을 수 없습니다"):
        storage.load(write_zip(tmp_path / "badsides.hsd", {"document.json": document_json(sides=[1, 2])}))
    with pytest.raises(StorageError, match="문서 내용을 읽을 수 없습니다"):
        storage.load(write_zip(tmp_path / "list.hsd", {"document.json": b"[1, 2, 3]"}))
    assert storage.read_thumbnail(not_zip) is None
    assert storage.read_thumbnail(write_zip(tmp_path / "nothumb.hsd", {"document.json": document_json()})) is None


def test_oversized_archive_is_rejected(tmp_path, monkeypatch):
    source = write_zip(tmp_path / "big.hsd", {"document.json": document_json(), "assets/a.bin": b"0" * 5000})
    monkeypatch.setattr(storage, "_MAX_TOTAL_BYTES", 1000)
    with pytest.raises(StorageError, match="너무 커서"):
        storage.load(source)


def test_cleanup_only_removes_own_temp_dirs(tmp_path):
    keep = tmp_path / "내 문서"
    keep.mkdir()
    storage.cleanup_workdir(str(keep))
    assert keep.exists()
    temp = storage.new_workdir()
    assert os.path.isdir(os.path.join(temp, "assets"))
    storage.cleanup_workdir(Document(base_dir=temp))
    assert not os.path.exists(temp)
    storage.cleanup_workdir(Document())


# ---------------------------------------------------------------- 기존 방식(이미지 한 장) 변환

def test_single_image_with_bleed_ratio_becomes_bleed_document(tmp_path, make_png):
    image = make_png(size=(580, 900), color=(0, 128, 255))
    document = storage.document_from_single_image(image, workdir=tmp_path / "w")
    assert (document.card_width_mm, document.card_height_mm, document.bleed_mm) == (90.0, 58.0, 2.0)
    assert document.name == os.path.splitext(os.path.basename(image))[0]
    assert not document.dual_side
    front = document.side("front")
    assert front.orientation == "portrait" and len(front.items) == 1
    item = front.items[0]
    assert isinstance(item, ImageItem)
    assert (item.x, item.y, item.w, item.h) == (0.0, 0.0, 58.0, 90.0)
    assert os.path.isfile(os.path.join(document.base_dir, item.asset))

    rendered = render_side(document, "front", record={}, crop_bleed=True)
    assert rendered.color.size == (round(58 * 300 / 25.4) - 48, round(90 * 300 / 25.4) - 48)
    assert np.unique(np.array(rendered.color).reshape(-1, 3), axis=0).tolist() == [[0, 128, 255]]
    assert rendered.warnings == []


def test_single_image_other_ratio_uses_cut_size_and_landscape(tmp_path, make_png):
    image = make_png(size=(860, 540))
    document = storage.document_from_single_image(image, workdir=tmp_path / "w", name="가로 카드")
    assert (document.card_width_mm, document.card_height_mm, document.bleed_mm) == (86.0, 54.0, 0.0)
    assert document.name == "가로 카드"
    front = document.side("front")
    assert front.orientation == "landscape"
    assert (front.items[0].w, front.items[0].h, front.items[0].fit) == (86.0, 54.0, "stretch")


def test_single_image_with_back_mask_and_preset(tmp_path, make_png):
    front_image = make_png(size=(300, 500), color=(255, 0, 0))
    back_image = make_png(size=(300, 500), color=(0, 0, 255))
    mask = make_png(size=(300, 500), color=255, mode="L")
    document = storage.document_from_single_image(
        front_image, back_path=back_image, mask_path=mask, preset=CR80, fit="fill", workdir=tmp_path / "w",
    )
    assert document.dual_side and document.active_side_keys() == ["front", "back"]
    assert (document.card_width_mm, document.card_height_mm) == (CR80.width_mm, CR80.height_mm)
    front, back = document.side("front").items[0], document.side("back").items[0]
    assert front.mask_asset and back.mask_asset is None and front.fit == "fill"
    assert (front.w, front.h) == (53.98, 85.6) == (back.w, back.h)

    rendered = render_side(document, "front", record={})
    assert rendered.white_mask is not None and np.array(rendered.white_mask).min() == 255
    assert tuple(np.array(render_side(document, "back", record={}).color)[500, 300]) == (0, 0, 255)

    saved = storage.save(document, tmp_path / "legacy.hsd")
    reloaded = storage.load(saved, workdir=tmp_path / "r")
    assert reloaded == document


def test_single_image_creates_temp_workdir_and_rejects_bad_file(tmp_path, make_png):
    document = storage.document_from_single_image(make_png(size=(540, 860)))
    try:
        assert os.path.basename(document.base_dir).startswith("hsd_")
        assert (document.card_width_mm, document.bleed_mm) == (CUT_54X86.width_mm, 0.0)
        assert CUT_54X86_BLEED.bleed_mm == 2.0
    finally:
        storage.cleanup_workdir(document)
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    with pytest.raises(StorageError, match="이미지를 열 수 없습니다"):
        storage.document_from_single_image(bad, workdir=tmp_path / "w2")
