import hashlib
import time
from datetime import date, datetime
from datetime import time as dtime

import pytest
from openpyxl import Workbook

from data.errors import DataError
from data.importers import (
    ImportResult,
    detect_delimiter,
    import_into,
    list_sheets,
    preview_csv,
    preview_excel,
    preview_file,
    suggest_column_map,
)
from data.values import guess_type, parse_date, parse_number, parse_time

KOREAN_CSV = (
    "이름,사번,부서,입사일,급여,연락처\r\n"
    "김철수,00123,영업부,2026-01-05,\"3,500,000\",010-1234-5678\r\n"
    "이영희,00124,개발부,2026.1.5,\"4,200,000\",010-2222-3333\r\n"
    "박민수,00125,\"개발부, 2팀\",26/01/05,2800000,02-123-4567\r\n"
)


def write(tmp_path, name, text, encoding):
    path = tmp_path / name
    path.write_bytes(text.encode(encoding))
    return path


def types(preview):
    return {c.name: c.type for c in preview.columns}


# ---- 값 해석 ----


def test_parse_date_korean_formats():
    expected = date(2026, 1, 5)
    for text in ["2026-01-05", "2026.1.5", "2026. 1. 5.", "2026/1/5", "26/01/05", "26.1.5", "2026년 1월 5일",
                 "2026-01-05 00:00:00"]:
        assert parse_date(text) == expected, text
    assert parse_date("99/12/31") == date(1999, 12, 31)
    for text in ["2026-13-01", "2026-02-30", "010-1234-5678", "02-123-4567", "20260105", "abc", "2026-01-05 13:30:00", ""]:
        assert parse_date(text) is None, text


def test_parse_number_and_time():
    assert parse_number("3,500,000") == 3500000
    assert parse_number("-12.5") == -12.5
    assert parse_number("1,23") is None
    assert parse_number("12a") is None
    assert parse_number("9" * 30) is None
    assert parse_time("9:05") == "09:05"
    assert parse_time("오후 2:30") == "14:30"
    assert parse_time("12:00 AM") == "00:00"
    assert parse_time("23:59:58") == "23:59:58"
    assert parse_time("25:00") is None


def test_guess_type_keeps_leading_zero_values_as_text():
    assert guess_type(["00123", "00124"]) == "text"
    assert guess_type(["010-1234-5678", "02-123-4567"]) == "text"
    assert guess_type(["01012345678"]) == "text"
    assert guess_type(["+821012345678"]) == "text"
    assert guess_type(["1234567890123456"]) == "text"
    assert guess_type(["123", "0", "0.5", "1,000"]) == "number"
    assert guess_type(["123", "00124"]) == "text"
    assert guess_type(["2026-01-05", "26/01/05", ""]) == "date"
    assert guess_type(["2026-01-05", "미정"]) == "text"
    assert guess_type(["09:00", "오후 6:30"]) == "time"
    assert guess_type(["a.JPG", "b.png"]) == "photo"
    assert guess_type(["00123"], "사진") == "photo"
    assert guess_type([], "이름") == "text"


# ---- CSV 미리보기 ----


@pytest.mark.parametrize(
    "encoding,expected",
    [("utf-8", "utf-8"), ("utf-8-sig", "utf-8-sig"), ("cp949", "cp949"), ("euc-kr", "cp949"), ("utf-16", "utf-16")],
)
def test_preview_csv_detects_encoding(tmp_path, encoding, expected):
    preview = preview_csv(write(tmp_path, "명단.csv", KOREAN_CSV, encoding))
    assert preview.encoding == expected
    assert preview.delimiter == ","
    assert preview.has_header is True
    assert [c.name for c in preview.columns] == ["이름", "사번", "부서", "입사일", "급여", "연락처"]
    assert preview.total_rows == 3
    assert preview.sample_rows[2] == ["박민수", "00125", "개발부, 2팀", "26/01/05", "2800000", "02-123-4567"]
    assert types(preview) == {
        "이름": "text", "사번": "text", "부서": "text", "입사일": "date", "급여": "number", "연락처": "text",
    }
    assert preview.warnings == []


def test_preview_csv_utf16_without_bom(tmp_path):
    preview = preview_csv(write(tmp_path, "a.csv", "name,dept\nKim,Sales\nLee,Dev\n", "utf-16-le"))
    assert preview.encoding == "utf-16-le"
    assert [c.name for c in preview.columns] == ["name", "dept"]


@pytest.mark.parametrize("delimiter", [";", "\t", "|"])
def test_preview_csv_detects_delimiter(tmp_path, delimiter):
    text = delimiter.join(["이름", "주소", "금액"]) + "\n"
    text += delimiter.join(["김철수", "서울, 강남구", "1,000"]) + "\n"
    text += delimiter.join(["이영희", "부산, 해운대구, 우동", "2,500"]) + "\n"
    preview = preview_csv(write(tmp_path, "a.csv", text, "cp949"))
    assert preview.delimiter == delimiter
    assert [c.name for c in preview.columns] == ["이름", "주소", "금액"]
    assert preview.sample_rows[1][1] == "부산, 해운대구, 우동"
    assert types(preview)["금액"] == "number"


def test_detect_delimiter_single_column():
    assert detect_delimiter("이름\n김철수\n이영희\n") == ","


def test_preview_csv_headerless(tmp_path):
    preview = preview_csv(write(tmp_path, "a.csv", "김철수,00123,2026-01-05\n이영희,00124,2026-02-01\n", "cp949"))
    assert preview.has_header is False
    assert [c.name for c in preview.columns] == ["항목1", "항목2", "항목3"]
    assert preview.total_rows == 2
    assert preview.sample_rows[0] == ["김철수", "00123", "2026-01-05"]
    assert types(preview) == {"항목1": "text", "항목2": "text", "항목3": "date"}


def test_preview_csv_headerless_all_text_repeating_values(tmp_path):
    preview = preview_csv(write(tmp_path, "a.csv", "가나,영업\n다라,개발\n마바,영업\n", "utf-8"))
    assert preview.has_header is False
    assert preview.total_rows == 3


def test_preview_csv_header_override(tmp_path):
    path = write(tmp_path, "a.csv", "김철수,00123\n이영희,00124\n", "utf-8")
    forced = preview_csv(path, has_header=True)
    assert [c.name for c in forced.columns] == ["김철수", "00123"]
    assert forced.total_rows == 1
    assert preview_csv(write(tmp_path, "b.csv", "이름,부서\n가,나\n", "utf-8"), has_header=False).total_rows == 2
    assert preview_csv(path, delimiter=";").delimiter == ";"


def test_preview_csv_messy_headers_and_blank_lines(tmp_path):
    text = "\n\n 이름 ,,이름,_id,메모\n\n김철수,x,철수,7,\n,,,,\n이영희,y,영희,8,\n\n\n"
    preview = preview_csv(write(tmp_path, "a.csv", text, "utf-8"))
    assert preview.has_header is True
    assert [c.name for c in preview.columns] == ["이름", "항목2", "이름 (2)", "id", "메모"]
    assert preview.total_rows == 2
    assert preview.blank_rows == 2
    assert preview.row_numbers == [5, 7]


def test_preview_csv_drops_empty_trailing_columns_and_pads_short_rows(tmp_path):
    preview = preview_csv(write(tmp_path, "a.csv", "이름,부서,,\n김철수,영업,,\n이영희\n", "utf-8"))
    assert [c.name for c in preview.columns] == ["이름", "부서"]
    assert preview.rows == [["김철수", "영업"], ["이영희", ""]]


def test_preview_csv_quoted_newlines_and_quotes(tmp_path):
    text = '이름,메모\n김철수,"첫 줄\n둘째 줄"\n이영희,"그는 ""안녕""이라 했다"\n'
    preview = preview_csv(write(tmp_path, "a.csv", text, "utf-8"))
    assert preview.rows == [["김철수", "첫 줄\n둘째 줄"], ["이영희", '그는 "안녕"이라 했다']]
    assert preview.row_numbers == [3, 4]


def test_preview_csv_errors(tmp_path):
    with pytest.raises(DataError, match="찾을 수 없습니다"):
        preview_csv(tmp_path / "없음.csv")
    with pytest.raises(DataError, match="데이터가 없습니다"):
        preview_csv(write(tmp_path, "empty.csv", "", "utf-8"))
    with pytest.raises(DataError, match="데이터가 없습니다"):
        preview_csv(write(tmp_path, "blank.csv", "\n\n,,\n", "utf-8"))
    with pytest.raises(DataError, match="지원하지 않는 파일"):
        preview_file(tmp_path / "a.pdf")


def test_preview_csv_undecodable_bytes_warn_instead_of_failing(tmp_path):
    path = tmp_path / "a.csv"
    path.write_bytes("이름,부서\n".encode("cp949") + b"\xff\xff,x\n")
    preview = preview_csv(path)
    assert preview.encoding == "cp949"
    assert preview.warnings and "인코딩" in preview.warnings[0]
    assert preview.total_rows == 1


# ---- 가져오기 ----


def test_import_into_empty_store_creates_typed_columns(tmp_path, store):
    path = write(tmp_path, "명단.csv", KOREAN_CSV, "cp949")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    result = import_into(store, preview_csv(path))
    assert isinstance(result, ImportResult)
    assert (result.inserted, result.skipped, result.warnings) == (3, 0, [])
    assert result.created_columns == ["이름", "사번", "부서", "입사일", "급여", "연락처"]
    assert {c.name: c.type for c in store.columns()}["입사일"] == "date"
    rows = store.rows()
    assert rows[0]["사번"] == "00123" and rows[0]["연락처"] == "010-1234-5678"
    assert [r["입사일"] for r in rows] == ["2026-01-05"] * 3
    assert [r["급여"] for r in rows] == [3500000, 4200000, 2800000]
    assert rows[2]["부서"] == "개발부, 2팀"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_import_append_and_replace(tmp_path, store):
    preview = preview_csv(write(tmp_path, "a.csv", KOREAN_CSV, "utf-8"))
    import_into(store, preview)
    store.mark_printed(store.ids())
    assert import_into(store, preview, mode="append").inserted == 3
    assert store.count() == 6
    result = import_into(store, preview, mode="replace")
    assert result.inserted == 3 and result.created_columns == []
    assert store.count() == 3
    assert all(not r["_printed"] for r in store.rows())
    with pytest.raises(DataError):
        import_into(store, preview, mode="merge")


def test_suggest_column_map_fuzzy_matches_existing_columns(tmp_path, store):
    for name in ["이름", "사원번호", "부서", "입사 일자", "사진"]:
        store.add_column(name)
    text = "성 명,사번,소속,입사일자,Photo,비고\n김철수,00123,영업부,2026-01-05,a.jpg,없음\n"
    preview = preview_csv(write(tmp_path, "a.csv", text, "utf-8"))
    assert suggest_column_map(store, preview) == {
        "성 명": "이름",
        "사번": "사원번호",
        "소속": "부서",
        "입사일자": "입사 일자",
        "Photo": "사진",
        "비고": "비고",
    }
    result = import_into(store, preview)
    assert result.created_columns == ["비고"]
    record = store.rows()[0]
    assert record["이름"] == "김철수" and record["사원번호"] == "00123" and record["사진"] == "a.jpg"


def test_suggest_column_map_uses_each_store_column_once(tmp_path, store):
    store.add_column("이름")
    preview = preview_csv(write(tmp_path, "a.csv", "이름,성명\n가,나\n", "utf-8"))
    assert suggest_column_map(store, preview) == {"이름": "이름", "성명": "성명"}


def test_import_with_explicit_map_skips_and_renames(tmp_path, store):
    store.add_column("성명")
    preview = preview_csv(write(tmp_path, "a.csv", KOREAN_CSV, "utf-8"))
    result = import_into(store, preview, {"이름": "성명", "사번": "직원 번호", "부서": None, "입사일": ""})
    assert result.inserted == 3
    assert store.column_names() == ["성명", "직원 번호"]
    assert store.rows()[1] == {
        "_id": 2, "성명": "이영희", "직원 번호": "00124", "_printed": False, "_print_count": 0, "_printed_at": None,
    }


def test_import_column_map_errors_leave_store_untouched(tmp_path, people):
    preview = preview_csv(write(tmp_path, "a.csv", KOREAN_CSV, "utf-8"))
    with pytest.raises(DataError, match="없습니다"):
        import_into(people, preview, {"없는 열": "이름"})
    with pytest.raises(DataError, match="두 개 이상"):
        import_into(people, preview, {"이름": "이름", "사번": "이름"})
    with pytest.raises(DataError, match="가져올 항목"):
        import_into(people, preview, {"이름": None})
    with pytest.raises(DataError):
        import_into(people, preview, {"이름": "이름", "사번": "_id"}, mode="replace")
    assert people.count() == 5
    assert people.column_names() == ["이름", "사번", "부서", "나이", "입사일"]


def test_import_row_level_warnings_in_korean(tmp_path, store):
    store.add_column("이름")
    store.add_column("나이", "number")
    store.add_column("입사일", "date")
    text = "이름,나이,입사일,메모\n김철수,31,2026-01-05,a\n이영희,서른,미정,b\n,,,c\n박민수,007,2026-02-30,d\n"
    preview = preview_csv(write(tmp_path, "a.csv", text, "utf-8"))
    result = import_into(store, preview, {"이름": "이름", "나이": "나이", "입사일": "입사일"})
    assert result.inserted == 3
    assert result.skipped == 1
    assert len(result.warnings) == 5
    assert "3행" in result.warnings[0] and "'나이'" in result.warnings[0] and "'서른'" in result.warnings[0]
    assert "숫자" in result.warnings[0]
    assert "3행" in result.warnings[1] and "날짜" in result.warnings[1]
    assert result.warnings[2] == "4행: 가져올 값이 없어 건너뛰었습니다."
    assert "5행" in result.warnings[3] and "'007'" in result.warnings[3]
    rows = store.rows()
    assert rows[1]["나이"] == "서른" and rows[1]["입사일"] == "미정"
    assert rows[2]["나이"] == "007" and rows[2]["입사일"] == "2026-02-30"


def test_import_caps_warning_list(tmp_path, store):
    store.add_column("나이", "number")
    text = "나이\n" + "".join(f"값{i}\n" for i in range(250))
    result = import_into(store, preview_csv(write(tmp_path, "a.csv", text, "utf-8"), has_header=True))
    assert result.inserted == 250
    assert len(result.warnings) == 101
    assert "150개" in result.warnings[-1]


def test_import_hostile_headers_and_values(tmp_path, store):
    header = ['select', 'a"b', "x'); DROP TABLE records;--", "_id", "이름 "]
    value = "'; DELETE FROM records; --"
    path = tmp_path / "a.csv"
    import csv

    with open(path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerow([value, 'he said "hi"', "=1+1", "7", "김철수"])
    preview = preview_csv(path, has_header=True)
    result = import_into(store, preview)
    assert result.inserted == 1
    assert store.column_names() == ["select", 'a"b', "x'); DROP TABLE records;--", "id", "이름"]
    record = store.rows()[0]
    assert record["select"] == value
    assert record['a"b'] == 'he said "hi"'
    assert record["x'); DROP TABLE records;--"] == "=1+1"
    assert record["id"] == 7 and record["_id"] == 1


def test_import_10k_rows_is_fast(tmp_path, store):
    lines = ["이름,사번,부서,입사일,급여"]
    lines += [f"사원{i},{i:06d},부서{i % 20},2026-01-{i % 28 + 1:02d},\"{1000 + i:,}\"" for i in range(10_000)]
    path = write(tmp_path, "big.csv", "\n".join(lines) + "\n", "cp949")
    started = time.perf_counter()
    preview = preview_csv(path)
    result = import_into(store, preview)
    elapsed = time.perf_counter() - started
    assert result.inserted == 10_000 and result.warnings == []
    assert store.count() == 10_000
    last = store.rows(order=("_id", "desc"), limit=1)[0]
    assert last["사번"] == "009999" and last["급여"] == 10999 and last["입사일"] == "2026-01-04"
    assert elapsed < 5, f"10,000행 가져오기에 {elapsed:.2f}초 걸림"


# ---- Excel ----


def save_workbook(tmp_path, build, name="명단.xlsx"):
    workbook = Workbook()
    build(workbook)
    path = tmp_path / name
    workbook.save(path)
    return path


def test_preview_excel_basic_types_and_dates(tmp_path, store):
    def build(workbook):
        sheet = workbook.active
        sheet.title = "직원"
        sheet.append(["이름", "사번", "입사일", "출근", "키", "급여", "코드"])
        sheet.append(["김철수", "00123", datetime(2026, 1, 5), dtime(9, 0), 175.5, 3500000, 7])
        sheet.append(["이영희", "00124", date(2026, 2, 1), dtime(8, 30), 162.0, 4200000.0, 42])
        sheet["G2"].number_format = "00000"
        sheet["G3"].number_format = "00000"
        sheet.cell(row=50, column=1).value = None
        sheet.cell(row=3, column=12).value = None

    path = save_workbook(tmp_path, build)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    preview = preview_excel(path)
    assert preview.kind == "excel" and preview.sheet == "직원" and preview.sheets == ["직원"]
    assert preview.has_header is True
    assert types(preview) == {
        "이름": "text", "사번": "text", "입사일": "date", "출근": "time", "키": "number", "급여": "number",
        "코드": "text",
    }
    assert preview.total_rows == 2
    assert preview.sample_rows[0] == ["김철수", "00123", "2026-01-05", "09:00", "175.5", "3500000", "00007"]
    assert preview.sample_rows[1][4:6] == ["162", "4200000"]

    result = import_into(store, preview)
    assert result.inserted == 2 and result.warnings == []
    record = store.rows()[0]
    assert record["사번"] == "00123" and record["입사일"] == "2026-01-05" and record["코드"] == "00007"
    assert record["키"] == 175.5 and record["급여"] == 3500000
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_preview_excel_picks_first_non_empty_sheet_and_named_sheet(tmp_path):
    def build(workbook):
        workbook.active.title = "빈 시트"
        people = workbook.create_sheet("명단")
        people.append(["이름", "부서"])
        people.append(["김철수", "영업부"])
        other = workbook.create_sheet("기타")
        other.append(["코드", "값"])
        other.append(["A", 1])
        other.append(["B", 2])

    path = save_workbook(tmp_path, build)
    assert list_sheets(path) == ["빈 시트", "명단", "기타"]
    preview = preview_excel(path)
    assert preview.sheet == "명단" and preview.sheets == ["빈 시트", "명단", "기타"]
    assert preview_excel(path, sheet="기타").total_rows == 2
    assert preview_excel(path, sheet=2).sheet == "기타"
    assert preview_file(path, sheet="기타").sheet == "기타"
    with pytest.raises(DataError, match="시트에 데이터가 없습니다"):
        preview_excel(path, sheet="빈 시트")
    with pytest.raises(DataError, match="찾을 수 없습니다"):
        preview_excel(path, sheet="없는 시트")
    with pytest.raises(DataError):
        preview_excel(path, sheet=9)


def test_preview_excel_title_row_and_merged_two_level_header(tmp_path):
    def build(workbook):
        sheet = workbook.active
        sheet["A1"] = "2026년 사원 명부"
        sheet.merge_cells("A1:E1")
        sheet["A3"] = "이름"
        sheet.merge_cells("A3:A4")
        sheet["B3"] = "소속"
        sheet.merge_cells("B3:C3")
        sheet["B4"] = "부서"
        sheet["C4"] = "팀"
        sheet["D3"] = "연락처"
        sheet.merge_cells("D3:E3")
        sheet["D4"] = "휴대폰"
        sheet["E4"] = "내선"
        sheet.append(["김철수", "영업부", "1팀", "010-1111-2222", "0101"])
        sheet.append(["이영희", "개발부", "2팀", "010-3333-4444", "0102"])
        sheet.merge_cells("B5:B6")

    preview = preview_excel(save_workbook(tmp_path, build))
    assert preview.has_header is True
    assert [c.name for c in preview.columns] == ["이름", "부서", "팀", "휴대폰", "내선"]
    assert preview.rows == [
        ["김철수", "영업부", "1팀", "010-1111-2222", "0101"],
        ["이영희", "영업부", "2팀", "010-3333-4444", "0102"],
    ]
    assert preview.row_numbers == [5, 6]
    assert types(preview)["내선"] == "text"


def test_preview_excel_merged_header_with_duplicate_leaf_names(tmp_path):
    def build(workbook):
        sheet = workbook.active
        sheet["A1"] = "이름"
        sheet.merge_cells("A1:A2")
        sheet["B1"] = "집"
        sheet.merge_cells("B1:C1")
        sheet["D1"] = "회사"
        sheet.merge_cells("D1:E1")
        for column, label in zip("BCDE", ["전화", "주소", "전화", "주소"]):
            sheet[f"{column}2"] = label
        sheet.append(["김철수", "02-1", "서울", "02-2", "부산"])

    preview = preview_excel(save_workbook(tmp_path, build))
    assert [c.name for c in preview.columns] == ["이름", "집 전화", "집 주소", "회사 전화", "회사 주소"]
    assert preview.total_rows == 1


def test_preview_excel_offset_table_blank_rows_and_columns(tmp_path):
    def build(workbook):
        sheet = workbook.active
        sheet["C4"] = "이름"
        sheet["D4"] = "부서"
        sheet["C5"] = "김철수"
        sheet["D5"] = "영업부"
        sheet["C7"] = "이영희"
        sheet["D7"] = "개발부"
        for row in range(8, 40):
            sheet.cell(row=row, column=3).value = "   "
        sheet["H1"] = None

    preview = preview_excel(save_workbook(tmp_path, build))
    assert [c.name for c in preview.columns] == ["이름", "부서"]
    assert preview.rows == [["김철수", "영업부"], ["이영희", "개발부"]]
    assert preview.row_numbers == [5, 7]
    assert preview.blank_rows == 1


def test_preview_excel_headerless(tmp_path):
    def build(workbook):
        sheet = workbook.active
        sheet.append(["김철수", 123, date(2026, 1, 5)])
        sheet.append(["이영희", 124, date(2026, 1, 6)])

    preview = preview_excel(save_workbook(tmp_path, build))
    assert preview.has_header is False
    assert [c.name for c in preview.columns] == ["항목1", "항목2", "항목3"]
    assert types(preview) == {"항목1": "text", "항목2": "number", "항목3": "date"}


def test_preview_excel_errors(tmp_path):
    with pytest.raises(DataError, match="xlsx"):
        preview_excel(tmp_path / "old.xls")
    with pytest.raises(DataError, match="찾을 수 없습니다"):
        preview_excel(tmp_path / "없음.xlsx")
    broken = tmp_path / "broken.xlsx"
    broken.write_bytes(b"not a zip file")
    with pytest.raises(DataError, match="열 수 없습니다"):
        preview_excel(broken)
    with pytest.raises(DataError, match="데이터가 없습니다"):
        preview_excel(save_workbook(tmp_path, lambda workbook: None, "empty.xlsx"))
