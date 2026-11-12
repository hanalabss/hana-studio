from pathlib import Path

import pytest

from data.photos import (
    PhotoLocator,
    PhotoResolver,
    apply_adjustment_to_all,
    binding_record,
    clear_adjustments,
    find_missing,
    find_photo,
    get_adjustment,
    locator_for,
    photo_path,
    set_adjustment,
    set_photo_folder,
)
from data.store import DataStore, PhotoAdjustment


@pytest.fixture
def folder(tmp_path):
    photos = tmp_path / "사진 폴더"
    photos.mkdir()
    for name in ["00123.jpg", "00124.PNG", "홍길동.jpeg", "Kim.JPG", "notes.txt", "00130.bmp", "raw"]:
        (photos / name).write_bytes(b"x")
    (photos / "퇴사자").mkdir()
    (photos / "퇴사자" / "00999.jpg").write_bytes(b"x")
    return photos


def test_find_photo_variants(folder):
    assert find_photo("00123", folder) == folder / "00123.jpg"
    assert find_photo("00123.jpg", folder) == folder / "00123.jpg"
    assert find_photo("00123.JPG", folder).name == "00123.jpg"
    assert find_photo("00124", folder).name == "00124.PNG"
    assert find_photo("00124.png", folder).name == "00124.PNG"
    assert find_photo("홍길동", folder).name == "홍길동.jpeg"
    assert find_photo("kim", folder).name == "Kim.JPG"
    assert find_photo("  00123  ", folder).name == "00123.jpg"
    assert find_photo('"00123.jpg"', folder).name == "00123.jpg"


def test_find_photo_wrong_extension_in_value_falls_back_to_other_candidates(folder):
    assert find_photo("00124.jpg", folder).name == "00124.PNG"
    assert find_photo("00130.jpg", folder).name == "00130.bmp"


def test_find_photo_numeric_values(folder):
    (folder / "77.jpg").write_bytes(b"x")
    assert find_photo(77, folder).name == "77.jpg"
    assert find_photo(77.0, folder).name == "77.jpg"


def test_find_photo_absolute_and_relative_sub_paths(folder, tmp_path):
    absolute = folder / "00123.jpg"
    assert find_photo(str(absolute), None) == absolute
    assert find_photo(str(absolute), tmp_path / "다른 폴더") == absolute
    assert find_photo(str(folder / "00123"), None) == absolute
    assert find_photo("퇴사자/00999", folder) == folder / "퇴사자" / "00999.jpg"
    assert find_photo(str(folder / "없음.jpg"), folder) is None


def test_find_photo_misses(folder, tmp_path):
    for value in [None, "", "   ", "99999", "notes", "퇴사자"]:
        assert find_photo(value, folder) is None, value
    assert find_photo("00123", None) is None
    assert find_photo("00123", tmp_path / "없는 폴더") is None


def test_extension_candidates_limit_and_order(folder):
    (folder / "dup.png").write_bytes(b"x")
    (folder / "dup.jpg").write_bytes(b"x")
    assert find_photo("dup", folder, [".png", ".jpg"]).name == "dup.png"
    assert find_photo("dup", folder, ["jpg", "png"]).name == "dup.jpg"
    assert find_photo("00130", folder, [".jpg"]) is None
    assert find_photo("notes", folder, ["txt"]).name == "notes.txt"


def test_locator_caches_listing_until_refresh(folder):
    locator = PhotoLocator(folder)
    assert locator.find("new") is None
    (folder / "new.jpg").write_bytes(b"x")
    assert locator.find("new") is None
    locator.refresh()
    assert locator.find("new").name == "new.jpg"


def make_store(tmp_path, folder):
    store = DataStore(tmp_path / "명단.hsdata")
    store.add_column("이름")
    store.add_column("사진", "photo")
    set_photo_folder(store, "사진", folder)
    store.insert_many(
        [
            {"이름": "김철수", "사진": "00123"},
            {"이름": "이영희", "사진": "00124.png"},
            {"이름": "박민수", "사진": "00125"},
            {"이름": "최지우", "사진": None},
            {"이름": "홍길동", "사진": str(folder / "홍길동.jpeg")},
        ]
    )
    return store


def test_photo_path_and_find_missing(tmp_path, folder):
    with make_store(tmp_path, folder) as store:
        assert store.column("사진").options["folder"] == str(folder)
        rows = store.rows()
        assert photo_path(store, rows[0], "사진") == folder / "00123.jpg"
        assert photo_path(store, rows[2], "사진") is None
        assert [r["이름"] for r in find_missing(store, "사진")] == ["박민수", "최지우"]
        assert [r["이름"] for r in find_missing(store, "사진", include_empty=False)] == ["박민수"]
        assert [r["이름"] for r in find_missing(store, "사진", filter="최지우")] == ["최지우"]

        set_photo_folder(store, "사진", folder, ["bmp"])
        assert store.column("사진").options["extensions"] == [".bmp"]
        assert [r["이름"] for r in find_missing(store, "사진", include_empty=False)] == ["김철수", "박민수"]


def test_find_missing_using_another_column_as_file_name(tmp_path, folder):
    with DataStore(tmp_path / "a.hsdata") as store:
        store.add_column("사번")
        store.set_column_options("사번", {"folder": str(folder)})
        store.insert_many([{"사번": "00123"}, {"사번": "00777"}])
        assert [r["사번"] for r in find_missing(store, "사번")] == ["00777"]


def test_relative_photo_folder_is_resolved_next_to_data_file(tmp_path):
    (tmp_path / "photos").mkdir()
    (tmp_path / "photos" / "a.jpg").write_bytes(b"x")
    with DataStore(tmp_path / "a.hsdata") as store:
        store.add_column("사진", "photo", {"folder": "photos"})
        assert locator_for(store, "사진").find("a") == (tmp_path / "photos" / "a.jpg").resolve()


def test_binding_record_gives_plain_dict_with_file_paths(tmp_path, folder):
    with make_store(tmp_path, folder) as store:
        rows = store.rows()
        assert binding_record(store, rows[0]) == {"이름": "김철수", "사진": str(folder / "00123.jpg")}
        resolver = PhotoResolver(store)
        assert resolver.binding_record(rows[2]) == {"이름": "박민수", "사진": ""}
        assert resolver.binding_record(rows[3]) == {"이름": "최지우", "사진": ""}
        assert resolver.path(rows[4], "사진") == Path(folder / "홍길동.jpeg")


def test_adjustment_helpers(tmp_path, folder):
    with make_store(tmp_path, folder) as store:
        ids = store.ids()
        assert get_adjustment(store, ids[0], "사진") == PhotoAdjustment()
        set_adjustment(store, ids[0], "사진", PhotoAdjustment(1.2, 0.05, -0.1))
        assert get_adjustment(store, ids[0], "사진") == PhotoAdjustment(1.2, 0.05, -0.1)
        assert apply_adjustment_to_all(store, "사진", PhotoAdjustment(1.4, 0.0, 0.1)) == 5
        assert {get_adjustment(store, i, "사진") for i in ids} == {PhotoAdjustment(1.4, 0.0, 0.1)}
        assert apply_adjustment_to_all(store, "사진", PhotoAdjustment(3.0), filter="김철수") == 1
        assert get_adjustment(store, ids[0], "사진").zoom == 3.0
        assert get_adjustment(store, ids[1], "사진").zoom == 1.4
        assert clear_adjustments(store, "사진") == 5
        assert get_adjustment(store, ids[0], "사진") == PhotoAdjustment()

    with DataStore(tmp_path / "명단.hsdata") as reopened:
        set_adjustment(reopened, 1, "사진", PhotoAdjustment(2.0, 0.5, 0.5))
    with DataStore(tmp_path / "명단.hsdata") as again:
        assert get_adjustment(again, 1, "사진") == PhotoAdjustment(2.0, 0.5, 0.5)
