import pytest
from people_helpers import *  # noqa: F401,F403
from people_helpers import PEOPLE, spy_rows

from PySide6.QtCore import Qt

from data.filters import Condition, all_of, printed_only, quick_search, unprinted_only
from data.store import DataStore
from ui.pages.people.model import (
    KIND_CHECK,
    KIND_FIELD,
    KIND_STATUS,
    PAGE_SIZE,
    IdsFilter,
    PeopleTableModel,
    PhotoIndex,
    PhotoPathRole,
    RecordRole,
    StatusRole,
)


@pytest.fixture
def big_store(tmp_path):
    store = DataStore(tmp_path / "big.hsdata")
    store.add_column("이름")
    store.add_column("부서")
    store.add_column("번호", "number")
    store.insert_many(
        {"이름": f"사람{i:05d}", "부서": f"{i % 7}팀", "번호": i} for i in range(10_000)
    )
    yield store
    store.close()


@pytest.fixture
def model(qapp, people_store):
    photos = PhotoIndex()
    photos.configure(people_store, {"사진", "이름"})
    m = PeopleTableModel(photos)
    m.set_store(people_store)
    yield m
    people_store.close()


def column(model, name):
    return model.column_of(name)


def test_columns_are_check_then_photo_then_fields_then_status(model):
    kinds = [s.kind for s in model.specs]
    assert kinds[0] == KIND_CHECK and kinds[-1] == KIND_STATUS
    assert [s.name for s in model.specs if s.kind == KIND_FIELD] == ["사진", "이름", "부서", "직급", "사번", "입사일"]
    assert model.columnCount() == 8
    assert model.rowCount() == len(PEOPLE)
    headers = [model.headerData(i, Qt.Horizontal) for i in range(model.columnCount())]
    assert headers == ["", "사진", "이름", "부서", "직급", "사번", "입사일", "인쇄"]


def test_data_shows_values_status_and_photo_paths(model, photo_folder):
    name_col, status_col, photo_col = column(model, "이름"), model.columnCount() - 1, column(model, "사진")
    assert model.data(model.index(0, name_col)) == "강민서"
    assert model.data(model.index(0, 0), Qt.CheckStateRole) == Qt.Unchecked
    assert model.data(model.index(0, 0), RecordRole)["사번"] == "HN260415"
    assert model.data(model.index(0, status_col)) == "아직"
    assert model.data(model.index(0, photo_col), PhotoPathRole) == str(photo_folder / "HN260415.png")

    printed = model.data(model.index(3, status_col))
    assert printed.endswith("1장") and "월" in printed
    assert model.data(model.index(3, status_col), StatusRole)[1] == "ok_text"

    # 문서윤: 파일이 없다. 서지안: 값이 비었다.
    assert model.data(model.index(4, photo_col), PhotoPathRole) == ""
    assert model.data(model.index(4, status_col)) == "사진 없음"
    assert model.data(model.index(6, status_col), StatusRole) == ("사진 없음", "warn_text")
    assert "찾지 못했습니다" in model.data(model.index(4, photo_col), Qt.ToolTipRole)


def test_flags(model):
    assert model.flags(model.index(0, 0)) & Qt.ItemIsUserCheckable
    assert model.flags(model.index(0, column(model, "이름"))) & Qt.ItemIsEditable
    assert not model.flags(model.index(0, model.columnCount() - 1)) & Qt.ItemIsEditable


def test_ten_thousand_rows_are_paged_not_loaded(qapp, big_store):
    calls = spy_rows(big_store)
    model = PeopleTableModel()
    model.set_store(big_store)
    assert model.rowCount() == 10_000
    assert calls == []                       # 줄 수만 세고 아직 아무 줄도 읽지 않는다

    name = model.column_of("이름")
    assert model.data(model.index(0, name)) == "사람00000"
    assert model.data(model.index(PAGE_SIZE - 1, name)) == "사람00199"
    assert calls == [(0, PAGE_SIZE, PAGE_SIZE)]

    assert model.data(model.index(9_999, name)) == "사람09999"
    assert model.data(model.index(5_000, model.column_of("번호"))) == "5000"
    assert len(calls) == 3
    assert all(limit == PAGE_SIZE for _, limit, _ in calls)
    assert model.fetched_rows == 3 * PAGE_SIZE
    assert model.record_at(10_000) is None and model.record_at(-1) is None


def test_scrolling_everything_keeps_a_bounded_cache(qapp, big_store):
    model = PeopleTableModel()
    model.set_store(big_store)
    name = model.column_of("이름")
    for row in range(0, 10_000, 137):
        assert model.data(model.index(row, name)) == f"사람{row:05d}"
    assert len(model._pages) <= 12
    assert sum(len(p) for p in model._pages.values()) <= 12 * PAGE_SIZE


def test_table_view_on_ten_thousand_rows_fetches_only_what_it_shows(qapp, big_store):
    from ui.pages.people.table import PeopleTable

    calls = spy_rows(big_store)
    model = PeopleTableModel()
    model.set_store(big_store)
    table = PeopleTable(model)
    table.resize(900, 600)
    table.show()
    pump(qapp)
    table.grab()
    assert sum(count for _, _, count in calls) <= 2 * PAGE_SIZE
    table.scrollToBottom()
    pump(qapp)
    table.grab()
    assert sum(count for _, _, count in calls) <= 4 * PAGE_SIZE
    assert all(limit == PAGE_SIZE for _, limit, _ in calls)
    table.close()


def test_sort_by_header_column(model):
    name = column(model, "이름")
    model.sort(name, Qt.DescendingOrder)
    assert model.data(model.index(0, name)) == "양수아"
    assert model.view_order == [("이름", "desc")]
    assert model.sort_state() == (name, "desc")
    model.sort(column(model, "부서"), Qt.AscendingOrder)
    assert model.data(model.index(0, column(model, "부서"))) == "개발팀"
    model.sort(model.columnCount() - 1, Qt.DescendingOrder)        # 인쇄한 사람부터
    assert model.data(model.index(0, name)) in ("남지호", "박도윤")
    model.set_order(None)
    assert model.view_order is None and model.sort_state() is None
    assert model.data(model.index(0, name)) == "강민서"


def test_sort_on_ten_thousand_rows_stays_paged(qapp, big_store):
    calls = spy_rows(big_store)
    model = PeopleTableModel()
    model.set_store(big_store)
    model.sort(model.column_of("번호"), Qt.DescendingOrder)
    assert model.data(model.index(0, model.column_of("이름"))) == "사람09999"
    assert calls == [(0, PAGE_SIZE, PAGE_SIZE)]


def test_filters_give_matching_rows_and_counts(model, people_store):
    model.set_view(unprinted_only(), None)
    assert model.rowCount() == 6
    model.set_view(printed_only(), None)
    assert [model.record_at(r)["이름"] for r in range(model.rowCount())] == ["남지호", "박도윤"]
    model.set_view(quick_search("개발팀 사원"), None)
    assert [model.record_at(r)["이름"] for r in range(model.rowCount())] == ["문서윤"]
    model.set_view(all_of(Condition("부서", "contains", "영업"), Condition("직급", "ne", "대리")), None)
    assert [model.record_at(r)["이름"] for r in range(model.rowCount())] == ["박도윤"]
    assert model.ids() == people_store.ids(model.view_filter)


def test_ids_filter(people_store):
    ids = people_store.ids()
    assert people_store.ids(IdsFilter(frozenset(ids[:2]))) == ids[:2]
    assert people_store.ids(IdsFilter(frozenset(ids[:2]), negate=True)) == ids[2:]
    assert people_store.count(IdsFilter(frozenset())) == 0
    assert people_store.count(IdsFilter(frozenset(), negate=True)) == len(ids)
    people_store.close()


def test_checking_rows(model, qapp):
    seen = []
    model.checkedChanged.connect(lambda: seen.append(len(model.checked_ids())))
    first = model.id_at(0)
    assert model.setData(model.index(0, 0), Qt.Checked, Qt.CheckStateRole)
    assert model.checked_ids() == {first}
    assert model.data(model.index(0, 0), Qt.CheckStateRole) == Qt.Checked
    model.toggle(first)
    assert model.checked_ids() == set()
    assert seen == [1, 0]
    model.set_checked([first], False)          # 바뀐 게 없으면 알리지 않는다
    assert seen == [1, 0]


def test_select_all_applies_to_current_filter_only(model):
    model.set_view(unprinted_only(), None)
    assert model.header_check_state() == Qt.Unchecked
    assert model.toggle_all_in_view() is True
    assert len(model.checked_ids()) == 6
    assert model.header_check_state() == Qt.Checked

    model.set_view(None, None)                  # 조건을 풀면 일부만 체크된 상태
    assert model.header_check_state() == Qt.PartiallyChecked
    assert model.checked_in_view() == 6

    model.set_view(printed_only(), None)        # 체크한 사람이 하나도 안 보이는 조건
    assert model.header_check_state() == Qt.Unchecked
    assert model.toggle_all_in_view() is True   # 여기서 모두 고르면 앞서 고른 6명에 2명이 더해진다
    assert len(model.checked_ids()) == 8
    assert model.toggle_all_in_view() is False  # 다시 누르면 보이는 2명만 풀린다
    assert len(model.checked_ids()) == 6


def test_inline_edit_writes_to_store(model, people_store):
    edited = []
    model.recordEdited.connect(lambda record_id, name: edited.append((record_id, name)))
    index = model.index(0, column(model, "부서"))
    assert model.setData(index, "  영업3팀 ", Qt.EditRole)
    record_id = model.id_at(0)
    assert people_store.get(record_id)["부서"] == "영업3팀"
    assert model.data(index) == "영업3팀"
    assert edited == [(record_id, "부서")]
    assert not model.setData(index, "영업3팀", Qt.EditRole)       # 같은 값이면 아무 일도 없다
    assert edited == [(record_id, "부서")]


def test_inline_edit_normalizes_dates_and_updates_photo_state(model, people_store, photo_folder):
    date_index = model.index(0, column(model, "입사일"))
    model.setData(date_index, "2026.4.1", Qt.EditRole)
    assert model.data(date_index) == "2026-04-01"

    status = model.columnCount() - 1
    assert model.photos.missing_ids() == {model.id_at(4), model.id_at(6)}
    model.setData(model.index(6, column(model, "사진")), "HN260420", Qt.EditRole)
    assert model.data(model.index(6, status)) == "아직"
    assert model.photos.missing_ids() == {model.id_at(4)}


def test_hidden_columns_are_left_out(model, people_store):
    people_store.set_column_options("직급", {"hidden": True})
    model.rebuild_columns()
    assert "직급" not in [s.name for s in model.specs]
    assert model.hidden_columns() == ["직급"]


def test_photo_index_prefers_columns_the_design_uses(people_store, photo_folder):
    people_store.add_column("증명사진", "photo")
    photos = PhotoIndex()
    photos.configure(people_store, {"사진"})
    assert photos.columns == ["사진"] and photos.primary == "사진"
    photos.configure(people_store, set())
    assert photos.columns == ["사진", "증명사진"]
    photos.configure(None)
    assert photos.columns == [] and photos.missing_ids() == set()
    people_store.close()


def test_frozen_model_does_not_touch_the_store(qapp, big_store):
    model = PeopleTableModel()
    model.set_store(big_store)
    calls = spy_rows(big_store)
    model.set_frozen(True)
    assert model.data(model.index(4_000, 1)) is None
    assert calls == []
    model.set_frozen(False)
    assert model.data(model.index(4_000, model.column_of("이름"))) == "사람04000"
