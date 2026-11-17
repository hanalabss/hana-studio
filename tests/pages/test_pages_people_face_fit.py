"""명단 화면 '얼굴에 맞추기'."""

from pathlib import Path

from PIL import Image

from people_helpers import *  # noqa: F401,F403
from people_helpers import pump

SAMPLE = Path(__file__).resolve().parents[1] / "services" / "data" / "face_sample.jpg"


def _use_face_photo(store, photo_folder, name="강민서"):
    record = next(r for r in store.rows() if r["이름"] == name)
    Image.open(SAMPLE).save(photo_folder / f"{record['사번']}.png")     # 같은 파일 이름에 얼굴 사진
    return record["_id"]


def test_fit_face_for_the_current_person(qapp, page, people_store, photo_folder):
    record_id = _use_face_photo(people_store, photo_folder)
    page.table.go_to_row(0)
    pump(qapp)
    page.preview.adjust_toggle.setChecked(True)
    assert page.preview.fit_face()
    assert people_store.get_photo_adjustment(record_id, "사진").zoom > 1.0
    assert not page.preview.face_hint.isVisible()


def test_no_face_shows_a_hint(qapp, page, people_store):
    page.table.go_to_row(0)
    pump(qapp)
    page.preview.adjust_toggle.setChecked(True)
    assert not page.preview.fit_face()          # 단색 사진
    assert "얼굴을 찾지 못했습니다" in page.preview.face_hint.text()


def test_fit_faces_for_everyone(qapp, page, people_store, photo_folder):
    record_id = _use_face_photo(people_store, photo_folder)
    result = page.fit_faces_for_all("사진", (25, 30), "fill", ask=False)
    assert result.fitted == [record_id]
    assert len(result.no_face) == 5 and len(result.no_photo) == 2
    assert people_store.get_photo_adjustment(record_id, "사진").zoom > 1.0

