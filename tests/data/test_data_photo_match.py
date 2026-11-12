"""사진 폴더만 고르면 알아서 짝짓기."""
from data.photo_match import AMBIGUOUS, FOUND, MISSING, folder_has_photos, list_photos, match_photos

ROWS = [
    ["1", "철도1부", "서하늘", "부사장", "jpg"],
    ["2", "지반사업부", "이도윤", "부사장", "jpg"],
    ["3", "기술지원부", "최가람", "이사", "jpg"],
]


def names(matches):
    return [m.file_name for m in matches]


def test_file_names_made_of_several_columns_are_matched():
    files = ["기술지원부_최가람_이사.jpg", "지반사업부_이도윤_부사장.jpg", "철도1부_서하늘_부사장.jpg"]
    matches = match_photos(ROWS, files)
    assert [m.status for m in matches] == [FOUND] * 3
    assert names(matches) == ["철도1부_서하늘_부사장.jpg", "지반사업부_이도윤_부사장.jpg", "기술지원부_최가람_이사.jpg"]


def test_a_wrong_title_in_the_file_name_does_not_lose_the_person():
    """명단에는 '이사'인데 파일 이름에는 '부사장'으로 적힌 경우(실제 받은 자료에 있던 일)."""
    files = ["기술지원부_최가람_부사장.jpg", "지반사업부_이도윤_부사장.jpg", "철도1부_서하늘_부사장.jpg"]
    assert names(match_photos(ROWS, files))[2] == "기술지원부_최가람_부사장.jpg"


def test_file_named_exactly_like_one_column_still_works():
    rows = [["강민서", "HN260415"], ["고은채", "HN260416"]]
    assert names(match_photos(rows, ["HN260416.png", "HN260415.JPG"])) == ["HN260415.JPG", "HN260416.png"]


def test_only_department_or_title_in_common_is_not_a_match():
    headers = ["연번", "부서명", "성명", "직위", "첨부파일"]
    matches = match_photos(ROWS, ["철도1부_단체사진.jpg", "부사장.jpg"], headers)
    assert [m.status for m in matches] == [MISSING] * 3


def test_same_name_twice_is_settled_by_the_other_columns():
    rows = [["영업부", "김민준", "대리"], ["개발부", "김민준", "과장"], ["총무부", "박서연", "사원"]]
    files = ["개발부_김민준.jpg", "영업부_김민준.jpg", "박서연.jpg"]
    assert names(match_photos(rows, files, ["부서", "이름", "직급"])) == ["영업부_김민준.jpg", "개발부_김민준.jpg", "박서연.jpg"]


def test_two_equal_candidates_are_left_for_the_user():
    rows = [["김민준", "A01"], ["박서연", "A02"]]
    matches = match_photos(rows, ["김민준_1.jpg", "김민준_2.jpg", "박서연.jpg"])
    assert matches[0].status == AMBIGUOUS and sorted(matches[0].candidates) == ["김민준_1.jpg", "김민준_2.jpg"]
    assert matches[1].file_name == "박서연.jpg"


def test_one_file_wanted_by_two_people_goes_to_neither():
    rows = [["김민준", "A01"], ["김민준", "A02"]]
    matches = match_photos(rows, ["김민준.jpg"], ["이름", "메모"])
    assert [m.status for m in matches] == [AMBIGUOUS, AMBIGUOUS]


def test_spaces_and_brackets_in_file_names():
    rows = [["서 하늘", "1001"], ["이도윤", "1002"]]
    assert names(match_photos(rows, ["서하늘(1001).jpg", "1002 이도윤 증명사진.jpeg"])) == \
        ["서하늘(1001).jpg", "1002 이도윤 증명사진.jpeg"]


def test_list_photos_ignores_other_files(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "명단.xlsx").write_bytes(b"x")
    (tmp_path / "sub").mkdir()
    assert list_photos(tmp_path) == ["a.jpg"] and folder_has_photos(tmp_path)
    assert not folder_has_photos(tmp_path / "sub") and list_photos(None) == []
