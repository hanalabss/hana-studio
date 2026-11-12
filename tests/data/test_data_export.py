import csv
from datetime import datetime

import pytest
from openpyxl import load_workbook

from data.errors import DataError
from data.export import STATUS_HEADERS, export_csv, export_xlsx
from data.filters import Condition, unprinted_only
from data.importers import import_into, preview_csv, preview_excel


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as stream:
        return list(csv.reader(stream))


def test_export_csv_has_bom_and_all_rows(people, tmp_path):
    path = tmp_path / "내보내기" / "명단.csv"
    assert export_csv(people, path) == 5
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")
    rows = read_csv(path)
    assert rows[0] == ["이름", "사번", "부서", "나이", "입사일"]
    assert rows[1] == ["김철수", "00123", "영업부", "31", "2020-03-02"]
    assert rows[4] == ["최지우", "00126", "", "", ""]
    assert len(rows) == 6
    assert [p.name for p in path.parent.iterdir()] == ["명단.csv"]


def test_export_csv_filter_order_columns_and_status(people, tmp_path):
    ids = people.ids()
    people.mark_printed([ids[1]])
    people.mark_printed([ids[1]])
    path = tmp_path / "a.csv"
    count = export_csv(
        people,
        path,
        filter=Condition("부서", "eq", "개발부"),
        order=("나이", "desc"),
        include_print_status=True,
        columns=["사번", "이름"],
    )
    assert count == 2
    rows = read_csv(path)
    assert rows[0] == ["사번", "이름", *STATUS_HEADERS]
    assert rows[1] == ["00125", "박민수", "", "0", ""]
    assert rows[2][:4] == ["00124", "이영희", "인쇄됨", "2"]
    datetime.strptime(rows[2][4], "%Y-%m-%d %H:%M:%S")

    assert export_csv(people, path, filter=unprinted_only()) == 4
    assert len(read_csv(path)) == 5
    with pytest.raises(DataError):
        export_csv(people, path, columns=["없는 열"])


def test_export_csv_escapes_quotes_commas_newlines(store, tmp_path):
    store.add_column('메모 "특이"')
    values = ['그는 "안녕", 이라 했다', "줄\n바꿈", "'; DROP TABLE records;--", "=1+1"]
    store.insert_many({'메모 "특이"': v} for v in values)
    path = tmp_path / "a.csv"
    export_csv(store, path)
    rows = read_csv(path)
    assert rows[0] == ['메모 "특이"']
    assert [r[0] for r in rows[1:]] == values


def test_export_status_headers_do_not_collide_with_user_columns(store, tmp_path):
    store.add_column("인쇄 여부")
    store.insert({"인쇄 여부": "사용자 값"})
    path = tmp_path / "a.csv"
    export_csv(store, path, include_print_status=True)
    rows = read_csv(path)
    assert rows[0] == ["인쇄 여부", "인쇄 여부 (2)", "인쇄 횟수", "마지막 인쇄 시각"]
    assert rows[1] == ["사용자 값", "", "0", ""]


def test_csv_round_trip_through_importer(people, tmp_path):
    path = tmp_path / "a.csv"
    export_csv(people, path)
    preview = preview_csv(path)
    assert preview.encoding == "utf-8-sig"
    assert {c.name: c.type for c in preview.columns} == {
        "이름": "text", "사번": "text", "부서": "text", "나이": "number", "입사일": "date",
    }
    before = [{k: v for k, v in r.items() if not k.startswith("_")} for r in people.rows()]
    import_into(people, preview, mode="replace")
    after = [{k: v for k, v in r.items() if not k.startswith("_")} for r in people.rows()]
    assert after == before


def test_export_xlsx_types_and_text_safety(people, tmp_path):
    people.add_column("메모")
    ids = people.ids()
    people.update(ids[0], {"메모": "=1+1"})
    people.update(ids[1], {"입사일": "미정"})
    people.mark_printed(ids[0])
    path = tmp_path / "명단.xlsx"
    assert export_xlsx(people, path, include_print_status=True) == 5

    sheet = load_workbook(path).active
    assert sheet.title == "데이터"
    assert [c.value for c in sheet[1]] == ["이름", "사번", "부서", "나이", "입사일", "메모", *STATUS_HEADERS]
    first = sheet[2]
    assert first[0].value == "김철수"
    assert first[1].value == "00123" and first[1].data_type == "s"
    assert first[3].value == 31 and first[3].data_type == "n"
    assert first[4].value == datetime(2020, 3, 2) and first[4].is_date
    assert first[5].value == "=1+1" and first[5].data_type == "s"
    assert first[6].value == "인쇄됨" and first[7].value == 1 and first[8].value
    assert sheet[3][4].value == "미정"
    assert sheet[5][2].value is None and sheet[5][3].value is None
    assert sheet.max_row == 6
    assert [p.name for p in tmp_path.iterdir() if p.suffix == ".xlsx"] == ["명단.xlsx"]


def test_xlsx_round_trip_through_importer(people, tmp_path):
    path = tmp_path / "a.xlsx"
    export_xlsx(people, path, filter=Condition("나이", "not-empty"), order=("나이", "asc"))
    preview = preview_excel(path)
    assert preview.total_rows == 4
    assert {c.name: c.type for c in preview.columns}["사번"] == "text"
    assert preview.rows[0] == ["이영희", "00124", "개발부", "27", "2022-07-01"]


def test_export_overwrites_existing_file_and_reports_locked_target(people, tmp_path):
    path = tmp_path / "a.csv"
    path.write_text("old", encoding="utf-8")
    export_csv(people, path)
    assert len(read_csv(path)) == 6

    blocked = tmp_path / "폴더인데.csv"
    blocked.mkdir()
    with pytest.raises(DataError, match="저장할 수 없습니다"):
        export_csv(people, blocked)
    with pytest.raises(DataError, match="저장할 수 없습니다"):
        export_xlsx(people, tmp_path / "폴더인데.csv")
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".~export-")] == []
