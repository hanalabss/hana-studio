"""다눌처럼 엑셀에서 복사한 것을 명단에 붙여 넣기."""

from data.clipboard import copy_rows, looks_like_header, parse_tsv, paste_grid, to_tsv, undo_paste
from data.store import DataStore


def _store(tmp_path):
    store = DataStore(tmp_path / "p.hsdata")
    for name in ("이름", "부서", "사번"):
        store.add_column(name)
    store.insert_many([{"이름": "김하나", "부서": "개발", "사번": "1"},
                       {"이름": "이두리", "부서": "영업", "사번": "2"}])
    return store


def test_parse_excel_text_with_quotes_and_trailing_newline():
    text = '김하나\t"개발\n1팀"\t1\r\n이두리\t영업\t2\r\n'
    assert parse_tsv(text) == [["김하나", "개발\n1팀", "1"], ["이두리", "영업", "2"]]
    assert parse_tsv("") == []


def test_tsv_round_trip():
    rows = [["a", "b\tc", 'd"e'], ["1", "", "x\ny"]]
    assert parse_tsv(to_tsv(rows)) == rows


def test_paste_overwrites_from_start_cell_and_appends_rest(tmp_path):
    store = _store(tmp_path)
    ids = store.ids()
    grid = parse_tsv("인사\t9\r\n총무\t8\r\n기획\t7\r\n")
    result = paste_grid(store, grid, ["부서", "사번"], ids[1:])
    assert store.get(ids[0])["부서"] == "개발", "시작 줄 위는 그대로"
    assert (store.get(ids[1])["부서"], store.get(ids[1])["사번"]) == ("인사", "9")
    assert len(result.inserted) == 2 and len(store.ids()) == 4
    new = [store.get(i) for i in result.inserted]
    assert [(r["부서"], r["사번"], r["이름"]) for r in new] == [("총무", "8", None), ("기획", "7", None)]


def test_paste_skips_header_row_and_reports_cut_columns(tmp_path):
    store = _store(tmp_path)
    grid = parse_tsv("이름\t부서\n박세나\t재무\t남는칸\n")
    assert looks_like_header(grid[0], store.column_names())
    result = paste_grid(store, grid, ["이름", "부서"], [])
    assert result.skipped_header and result.cut_columns == 1
    assert store.get(result.inserted[0])["이름"] == "박세나"


def test_undo_restores_old_values_and_removes_new_people(tmp_path):
    store = _store(tmp_path)
    ids = store.ids()
    result = paste_grid(store, parse_tsv("X\nY\nZ"), ["이름"], ids)
    undo_paste(store, result)
    assert store.ids() == ids
    assert [store.get(i)["이름"] for i in ids] == ["김하나", "이두리"]


def test_copy_rows_with_header():
    text = copy_rows([{"이름": "김하나", "사번": 1.0}], ["이름", "사번"], with_header=True)
    assert parse_tsv(text) == [["이름", "사번"], ["김하나", "1"]]
