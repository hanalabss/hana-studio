"""사진에서 얼굴을 찾아 사진 칸에 맞추기.

face_sample.jpg는 scikit-image의 astronaut 예제(NASA, 퍼블릭 도메인)를 잘라 줄인 것이다.
"""

import shutil
from pathlib import Path

import pytest
from PIL import Image

from data.photos import set_photo_folder
from data.store import DataStore, PhotoAdjustment
from design import renderer
from services.face_fit import FaceBox, detect_face, fit_face, fit_people
from services.render_job import crop_from_adjustment

SAMPLE = Path(__file__).parent / "data" / "face_sample.jpg"


def placed(face, image_size, frame, adjustment, fit="fill"):
    """조정값대로 그렸을 때 얼굴 가운데(칸 비율)와 얼굴 높이(칸 비율)."""
    crop = crop_from_adjustment(adjustment, image_size, frame, fit)
    x, y, w, _ = renderer.image_target_rect(*image_size, *frame, fit, crop)
    scale = w / image_size[0]
    cx, cy = face.center
    return (x + cx * scale) / frame[0], (y + cy * scale) / frame[1], face.h * scale / frame[1]


def test_finds_the_face_in_a_photo(qapp):
    face = detect_face(SAMPLE)
    assert face is not None
    cx, cy = face.center
    # 사진 위쪽 가운데에 사람 얼굴이 하나 있다.
    assert 70 < cx < 120 and 60 < cy < 110 and 50 < face.h < 110


def test_blank_or_missing_image_has_no_face(qapp, tmp_path):
    blank = tmp_path / "blank.png"
    Image.new("RGB", (200, 240), (200, 210, 230)).save(blank)
    assert detect_face(blank) is None
    assert detect_face(tmp_path / "없는파일.png") is None
    assert detect_face(None) is None


def test_fit_puts_the_face_in_the_middle_at_a_set_size():
    face = FaceBox(177, 66, 94, 94)
    for frame in ((25, 30), (30, 40), (20, 30)):
        adjustment = fit_face(face, (512, 512), frame)
        x, y, height = placed(face, (512, 512), frame, adjustment)
        assert x == pytest.approx(0.5, abs=1e-6)
        assert y == pytest.approx(0.48, abs=1e-6)
        assert height == pytest.approx(0.42, abs=1e-6)


def test_fit_never_leaves_an_empty_edge():
    # 얼굴이 사진 가장자리에 붙어 있으면 가운데까지 옮기지 못하고 끝에서 멈춘다.
    face = FaceBox(0, 0, 60, 60)
    adjustment = fit_face(face, (200, 200), (30, 30))
    crop = crop_from_adjustment(adjustment, (200, 200), (30, 30), "fill", clamp=False)
    assert -1.0 - 1e-9 <= crop.offset_x <= 1.0 + 1e-9 and -1.0 - 1e-9 <= crop.offset_y <= 1.0 + 1e-9
    # 얼굴이 이미 칸보다 크면 줄이지 않는다(배율 1 아래로는 빈 곳이 생긴다).
    big = fit_face(FaceBox(0, 0, 200, 200), (200, 200), (30, 30))
    assert big.zoom == 1.0


def test_fit_people_sets_only_the_photos_with_a_face(qapp, tmp_path):
    folder = tmp_path / "사진"
    folder.mkdir()
    shutil.copy(SAMPLE, folder / "A1.jpg")
    Image.new("RGB", (200, 240), (200, 210, 230)).save(folder / "A2.png")
    store = DataStore(tmp_path / "people.hsdata")
    store.add_column("이름")
    store.add_column("사진", "photo")
    set_photo_folder(store, "사진", folder)
    store.insert_many([{"이름": "얼굴", "사진": "A1"}, {"이름": "빈 사진", "사진": "A2"}, {"이름": "없음", "사진": "A3"}])
    ids = {r["이름"]: r["_id"] for r in store.rows()}
    kept = PhotoAdjustment(1.5, 0.1, 0.0)
    store.set_photo_adjustment(ids["빈 사진"], "사진", kept)

    seen = []
    result = fit_people(store, "사진", (25, 30), progress=lambda done, total: seen.append((done, total)) or True)

    assert result.fitted == [ids["얼굴"]] and result.no_face == [ids["빈 사진"]] and result.no_photo == [ids["없음"]]
    assert store.get_photo_adjustment(ids["얼굴"], "사진").zoom > 1.0
    assert store.get_photo_adjustment(ids["빈 사진"], "사진") == kept       # 못 찾은 사람은 그대로
    assert seen[-1] == (3, 3)
    store.close()


def test_fit_people_stops_when_cancelled(qapp, tmp_path):
    store = DataStore(tmp_path / "people.hsdata")
    store.add_column("사진", "photo")
    store.insert_many([{"사진": "x"}, {"사진": "y"}])
    result = fit_people(store, "사진", (25, 30), progress=lambda done, total: done < 1)
    assert result.total == 1
    store.close()
