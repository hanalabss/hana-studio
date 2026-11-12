"""가져오기: 사진 폴더만 있으면 파일 이름에서 사람을 찾아 짝짓는다."""
from data import importers
from data.photos import PhotoResolver
from data.store import DataStore
from ui.pages.people.importing import (AUTO_PHOTO_SOURCE, ImportPlan, count_photos, run_import,
                                       suggest_photo_source)

PEOPLE = [("1", "철도1부", "서하늘", "부사장"), ("2", "지반사업부", "이도윤", "부사장"), ("3", "기술지원부", "최가람", "이사")]


def make_folder(tmp_path, skip=()):
    lines = ["연번,부서명,성명,직위,첨부파일"] + [",".join(p) + ",jpg" for p in PEOPLE]
    (tmp_path / "명단.csv").write_text("\n".join(lines), encoding="utf-8-sig")
    for _, part, name, title in PEOPLE:
        if name not in skip:
            (tmp_path / f"{part}_{name}_{title}.jpg").write_bytes(b"x")
    return importers.preview_file(tmp_path / "명단.csv")


def test_auto_is_suggested_when_no_single_column_matches(tmp_path):
    preview = make_folder(tmp_path)
    assert suggest_photo_source(preview, tmp_path) == AUTO_PHOTO_SOURCE
    assert count_photos(preview, AUTO_PHOTO_SOURCE, tmp_path) == (3, [])


def test_a_column_that_matches_exactly_is_still_preferred(tmp_path):
    (tmp_path / "명단.csv").write_text("이름,사번\n서하늘,A01\n이도윤,A02", encoding="utf-8-sig")
    for name in ("A01.jpg", "A02.jpg"):
        (tmp_path / name).write_bytes(b"x")
    assert suggest_photo_source(importers.preview_file(tmp_path / "명단.csv"), tmp_path) == "사번"


def test_import_links_every_person_to_their_photo(tmp_path):
    preview = make_folder(tmp_path, skip=("이도윤",))
    assert count_photos(preview, AUTO_PHOTO_SOURCE, tmp_path) == (2, [1])
    store = DataStore(str(tmp_path / "명단.hsdata"))
    outcome = run_import(store, ImportPlan(
        preview=preview, field_map={}, include_rest=True, mode="append", skip_existing=False, key_field=None,
        photo_field="사진", photo_source=AUTO_PHOTO_SOURCE, photo_folder=str(tmp_path)))
    assert (outcome.photo_found, outcome.photo_missing) == (2, 1)
    resolver = PhotoResolver(store)
    found = {row["성명"]: resolver.path(row, "사진") for row in store.rows()}
    assert found["서하늘"].name == "철도1부_서하늘_부사장.jpg" and found["이도윤"] is None
    store.close()
