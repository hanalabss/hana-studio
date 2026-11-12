import pytest
from openpyxl import load_workbook
from PIL import Image

from data.errors import DataError
from data.importers import import_into, preview_excel, preview_file
from data.photos import PhotoResolver, find_missing, set_photo_folder
from data.samples import (
    DEFAULT_HEADERS,
    EXAMPLE_ID,
    MISSING_PHOTO_ID,
    PEOPLE,
    PHOTO_SIZE,
    free_names,
    is_id_header,
    sample_headers,
    write_sample,
)
from data.store import DataStore


def import_sample(sample, tmp_path, name="sample.hsdata") -> DataStore:
    """앱이 하는 것처럼: 미리보기 -> 가져오기 -> 사번 값을 사진 열에 넣고 사진 폴더를 건다."""
    store = DataStore(tmp_path / name)
    preview = preview_file(sample.workbook_path)
    result = import_into(store, preview)
    assert result.inserted == sample.people and not result.warnings
    store.add_column("사진", "photo")
    for record in store.rows():
        store.update(record["_id"], {"사진": record[sample.id_header]})
    set_photo_folder(store, "사진", sample.photo_dir)
    return store


def test_default_sample_is_written_where_asked(tmp_path):
    sample = write_sample(tmp_path)
    assert sample.workbook_path == tmp_path / "하나스튜디오 명단 예시.xlsx"
    assert sample.photo_dir == tmp_path / "사진 예시"
    assert sample.workbook_path.is_file() and sample.photo_dir.is_dir()
    assert sample.headers == list(DEFAULT_HEADERS) == ["이름", "부서", "직급", "사번"]
    assert sample.id_header == "사번" and sample.people == len(PEOPLE) == 10
    assert sample.missing_photo_id == MISSING_PHOTO_ID and sample.missing_photo_name == "서이슬"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["사진 예시", "하나스튜디오 명단 예시.xlsx"]


def test_workbook_opens_with_the_apps_importer(tmp_path):
    sample = write_sample(tmp_path)
    preview = preview_file(sample.workbook_path)
    assert preview.kind == "excel" and preview.sheet == "명단" and preview.sheets == ["명단", "사용법"]
    assert preview.has_header and not preview.warnings and preview.blank_rows == 0
    assert [(c.name, c.type) for c in preview.columns] == [
        ("이름", "text"), ("부서", "text"), ("직급", "text"), ("사번", "text")]
    assert preview.total_rows == 10
    assert preview.rows[0] == ["김하늘", "경영지원팀", "사원", "HN260411"]
    assert preview_excel(sample.workbook_path, sheet="명단").total_rows == 10

    with DataStore(tmp_path / "list.hsdata") as store:
        result = import_into(store, preview)
        assert result.inserted == 10 and result.skipped == 0 and result.warnings == []
        assert result.created_columns == ["이름", "부서", "직급", "사번"]
        assert store.column_names() == ["이름", "부서", "직급", "사번"]
        assert all(store.column(name).type == "text" for name in store.column_names())
        ids = [record["사번"] for record in store.rows()]
        assert ids == [person.id for person in PEOPLE] and EXAMPLE_ID in ids
        assert all(isinstance(value, str) for value in ids)


def test_id_cells_are_stored_as_text(tmp_path):
    sample = write_sample(tmp_path, ["이름", "사번", "전화번호"])
    workbook = load_workbook(sample.workbook_path)
    sheet = workbook["명단"]
    assert [cell.value for cell in sheet[1]] == ["이름", "사번", "전화번호"]
    for row in sheet.iter_rows(min_row=2):
        for cell in row:
            assert cell.data_type == "s" and cell.number_format == "@" and isinstance(cell.value, str)
    assert sheet["C2"].value == "010-0000-0411"            # 앞자리 0이 남는다
    workbook.close()
    preview = preview_file(sample.workbook_path)
    assert [c.type for c in preview.columns] == ["text", "text", "text"]
    assert preview.rows[0][2] == "010-0000-0411"


def test_header_row_is_styled_and_frozen(tmp_path):
    sample = write_sample(tmp_path)
    workbook = load_workbook(sample.workbook_path)
    sheet = workbook["명단"]
    assert sheet.freeze_panes == "A2" and not sheet.merged_cells.ranges
    for cell in sheet[1]:
        assert cell.font.bold and cell.fill.fill_type == "solid"
    assert all(sheet.column_dimensions[letter].width >= 10 for letter in "ABCD")
    workbook.close()


def test_guide_sheet_explains_in_plain_korean(tmp_path):
    sample = write_sample(tmp_path)
    workbook = load_workbook(sample.workbook_path)
    text = "\n".join(str(row[0].value or "") for row in workbook["사용법"].iter_rows())
    workbook.close()
    assert "첫 줄에는 항목 이름" in text and "한 줄에 한 사람" in text
    assert "칸을 합치지 마세요" in text and ".xlsx" in text and ".csv" in text
    assert "사진 파일 이름을 사번과 같게: HN260417.jpg" in text
    assert "사진 예시" in text and "서이슬" in text
    for word in ("컬럼", "레코드", "필드", "import", "헤더", "셀 병합"):
        assert word not in text


def test_photos_are_portrait_jpegs_named_by_id(tmp_path):
    sample = write_sample(tmp_path)
    files = sorted(sample.photo_dir.iterdir())
    assert files == sorted(sample.photo_paths) and len(files) == 9
    assert {f.name for f in files} == {f"{p.id}.jpg" for p in PEOPLE if p.id != MISSING_PHOTO_ID}
    assert (sample.photo_dir / "HN260417.jpg").is_file()
    looks = set()
    for path in files:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            assert image.format == "JPEG" and image.mode == "RGB"
            assert image.size == PHOTO_SIZE == (480, 640)
            corner, body = image.getpixel((8, 8)), image.getpixel((240, 600))
            assert corner != body                                  # 배경 위에 사람 모양이 있다
            looks.add((corner, body))
    assert len(looks) == 9                                         # 사람마다 색이 다르다


def test_photos_match_everyone_but_the_one_left_out(tmp_path):
    sample = write_sample(tmp_path)
    with import_sample(sample, tmp_path) as store:
        missing = find_missing(store, "사진")
        assert [(r["이름"], r["사번"]) for r in missing] == [("서이슬", MISSING_PHOTO_ID)]
        resolver = PhotoResolver(store)
        for record in store.rows():
            found = resolver.path(record, "사진")
            if record["사번"] == MISSING_PHOTO_ID:
                assert found is None and resolver.binding_record(record)["사진"] == ""
            else:
                assert found == sample.photo_dir / f"{record['사번']}.jpg"


def test_custom_headers_follow_the_card(tmp_path):
    sample = write_sample(tmp_path, ["사진", "성명", "소속", "직위", "사원번호", "연락처", "이메일", "입사일", "별명"],
                          photo_headers=["사진"])
    assert sample.headers == ["성명", "소속", "직위", "사원번호", "연락처", "이메일", "입사일", "별명"]
    assert sample.photo_headers == ["사진"] and sample.id_header == "사원번호"
    preview = preview_file(sample.workbook_path)
    assert [c.name for c in preview.columns] == sample.headers
    assert preview.rows[6] == ["윤보람", "경영지원팀", "대리", "HN260417", "010-0000-0417",
                               "hn260417@example.com", "2026-03-16", "별명 7"]
    assert {c.name: c.type for c in preview.columns}["입사일"] == "date"
    assert not any(c.type == "photo" for c in preview.columns)
    with import_sample(sample, tmp_path) as store:
        assert [r["성명"] for r in find_missing(store, "사진")] == ["서이슬"]


def test_an_id_column_is_added_when_the_card_has_none(tmp_path):
    sample = write_sample(tmp_path, ["이름", "부서", "전화번호", "증명사진"], photo_headers=["증명사진"])
    assert sample.headers == ["이름", "부서", "전화번호", "사번"] and sample.id_header == "사번"
    assert sample.photo_headers == ["증명사진"]
    preview = preview_file(sample.workbook_path)
    assert preview.has_header and [c.name for c in preview.columns] == sample.headers
    assert [row[3] for row in preview.rows] == [p.id for p in PEOPLE]


def test_header_cleanup_and_id_detection():
    assert sample_headers(None) == (["이름", "부서", "직급", "사번"], "사번", [])
    assert sample_headers([]) == (["이름", "부서", "직급", "사번"], "사번", [])
    assert sample_headers([" 이름 ", "이름", "", None, "학번"]) == (["이름", "학번"], "학번", [])
    assert sample_headers(["이름", "사진"]) == (["이름", "사번"], "사번", ["사진"])      # 이름만 봐도 사진 칸이면 뺀다
    assert sample_headers(["얼굴"], photo_headers=["얼굴"]) == (["이름", "부서", "직급", "사번"], "사번", ["얼굴"])
    assert all(is_id_header(name) for name in ("사번", "사원번호", "학번", "회원 번호", "번호", "ID", "No."))
    assert not any(is_id_header(name) for name in ("전화번호", "휴대폰 번호", "이름", "순번", "부서"))


def test_unknown_headers_without_known_words_still_read_as_a_header(tmp_path):
    sample = write_sample(tmp_path, ["별명", "좌우명"])
    preview = preview_file(sample.workbook_path)
    assert preview.has_header and [c.name for c in preview.columns] == ["별명", "좌우명", "사번"]
    assert preview.rows[0] == ["별명 1", "좌우명 1", "HN260411"] and preview.total_rows == 10


def test_existing_files_are_never_overwritten(tmp_path):
    first = write_sample(tmp_path)
    workbook = load_workbook(first.workbook_path)
    workbook["명단"]["A2"] = "내가 고친 이름"
    workbook.save(first.workbook_path)
    (first.photo_dir / "내 사진.txt").write_text("지우면 안 됨", encoding="utf-8")
    before = first.workbook_path.read_bytes()

    second = write_sample(tmp_path)
    assert second.workbook_path.name == "하나스튜디오 명단 예시 (2).xlsx"
    assert second.photo_dir.name == "사진 예시 (2)"
    assert first.workbook_path.read_bytes() == before
    assert (first.photo_dir / "내 사진.txt").read_text(encoding="utf-8") == "지우면 안 됨"
    assert len(list(second.photo_dir.iterdir())) == 9

    # 엑셀 파일만 지워도 사진 폴더가 남아 있으면 그 번호는 건너뛴다(둘은 늘 같은 번호)
    second.workbook_path.unlink()
    third = write_sample(tmp_path)
    assert (third.workbook_path.name, third.photo_dir.name) == ("하나스튜디오 명단 예시 (3).xlsx", "사진 예시 (3)")
    assert free_names(tmp_path) == (tmp_path / "하나스튜디오 명단 예시 (4).xlsx", tmp_path / "사진 예시 (4)")
    text = "\n".join(str(r[0].value or "") for r in load_workbook(third.workbook_path)["사용법"].iter_rows())
    assert "'사진 예시 (3)' 폴더" in text


def test_missing_folder_is_created(tmp_path):
    sample = write_sample(tmp_path / "새 폴더" / "안쪽")
    assert sample.workbook_path.is_file() and sample.photo_dir.is_dir()


def test_unwritable_place_gives_a_plain_message_and_leaves_nothing(tmp_path, monkeypatch):
    blocker = tmp_path / "파일입니다"
    blocker.write_text("x", encoding="utf-8")
    with pytest.raises(DataError) as caught:
        write_sample(blocker)
    assert "저장할 수 없습니다" in str(caught.value) and "Errno" not in str(caught.value)

    # 사진까지 만든 뒤 엑셀 저장에서 막히면 만들던 사진 폴더도 치운다
    def refuse(*_args, **_kwargs):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr("openpyxl.workbook.workbook.Workbook.save", refuse)
    target = tmp_path / "막힌 곳"
    with pytest.raises(DataError, match="다른 폴더를 골라 주세요"):
        write_sample(target)
    assert list(target.iterdir()) == []
