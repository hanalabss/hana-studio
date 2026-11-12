import pytest

from data.errors import DataError
from data.filters import (
    OPERATORS,
    Condition,
    Group,
    all_of,
    any_of,
    compile_filter,
    operators_for,
    printed_only,
    quick_search,
    unprinted_only,
)


def names(store, filter, order=None):
    return [r["이름"] for r in store.rows(filter, order)]


def test_equals_and_not_equals(people):
    assert names(people, Condition("부서", "equals", "개발부")) == ["이영희", "박민수"]
    assert names(people, Condition("부서", "not-equals", "개발부")) == ["김철수", "최지우", "Kim Minsu"]
    assert names(people, Condition("이름", "eq", "kim minsu")) == ["Kim Minsu"]
    assert names(people, Condition("나이", "=", "31")) == ["김철수"]
    assert names(people, Condition("나이", "eq", 31.0)) == ["김철수"]
    assert names(people, Condition("사번", "eq", "123")) == []
    assert names(people, Condition("사번", "eq", "00123")) == ["김철수"]


def test_text_matching_ops(people):
    assert names(people, Condition("이름", "contains", "민")) == ["박민수"]
    assert names(people, Condition("이름", "contains", "MIN")) == ["Kim Minsu"]
    assert names(people, Condition("이름", "starts", "김")) == ["김철수"]
    assert names(people, Condition("이름", "ends", "수")) == ["김철수", "박민수"]
    assert names(people, Condition("부서", "not_contains", "개발")) == ["김철수", "최지우", "Kim Minsu"]
    assert names(people, Condition("나이", "contains", "3")) == ["김철수", "Kim Minsu"]


def test_like_wildcards_are_literal(store):
    store.add_column("메모")
    store.insert_many([{"메모": "100% 완료"}, {"메모": "100점 완료"}, {"메모": "a_b"}, {"메모": "axb"}, {"메모": "c:\\x"}])
    assert [r["메모"] for r in store.rows(Condition("메모", "contains", "0%"))] == ["100% 완료"]
    assert [r["메모"] for r in store.rows(Condition("메모", "contains", "a_b"))] == ["a_b"]
    assert [r["메모"] for r in store.rows(Condition("메모", "contains", "\\"))] == ["c:\\x"]
    assert [r["메모"] for r in store.rows(quick_search("_"))] == ["a_b"]


def test_comparisons_on_numbers(people):
    assert names(people, Condition("나이", "gt", 31)) == ["박민수", "Kim Minsu"]
    assert names(people, Condition("나이", "ge", "31")) == ["김철수", "박민수", "Kim Minsu"]
    assert names(people, Condition("나이", "lt", 31)) == ["이영희"]
    assert names(people, Condition("나이", "le", 31)) == ["김철수", "이영희"]
    assert names(people, Condition("나이", "between", (30, 40))) == ["김철수", "Kim Minsu"]
    with pytest.raises(DataError, match="숫자"):
        people.rows(Condition("나이", "gt", "많음"))
    with pytest.raises(DataError):
        people.rows(Condition("나이", "between", 30))


def test_text_stored_in_number_column_does_not_match_comparisons(people):
    record_id = people.ids()[0]
    people.update(record_id, {"나이": "비밀"})
    assert "김철수" not in names(people, Condition("나이", "gt", 0))
    assert "김철수" not in names(people, Condition("나이", "lt", 1000))
    assert names(people, Condition("나이", "eq", "비밀")) == ["김철수"]


def test_comparisons_on_dates_accept_korean_formats(people):
    assert names(people, Condition("입사일", "ge", "2020.1.1")) == ["김철수", "이영희"]
    assert names(people, Condition("입사일", "lt", "2020-01-01")) == ["박민수", "Kim Minsu"]
    assert names(people, Condition("입사일", "between", ("19/01/01", "2020년 12월 31일"))) == ["김철수", "Kim Minsu"]
    assert names(people, Condition("입사일", "eq", "2022/07/01")) == ["이영희"]
    with pytest.raises(DataError, match="날짜"):
        people.rows(Condition("입사일", "gt", "어제"))


def test_empty_and_not_empty(people):
    assert names(people, Condition("부서", "empty")) == ["최지우"]
    assert names(people, Condition("나이", "empty")) == ["최지우"]
    assert len(names(people, Condition("부서", "not-empty"))) == 4


def test_groups_and_or_nesting(people):
    sales_or_old = any_of(Condition("부서", "eq", "영업부"), Condition("나이", "gt", 40))
    assert names(people, sales_or_old) == ["김철수", "박민수", "Kim Minsu"]
    both = all_of(sales_or_old, Condition("나이", "lt", 40))
    assert names(people, both) == ["김철수", "Kim Minsu"]
    assert names(people, Group("AND", [])) == names(people, None)
    combined = (Condition("부서", "eq", "개발부") & Condition("나이", "lt", 30)) | Condition("이름", "starts", "최")
    assert names(people, combined) == ["이영희", "최지우"]
    assert people.count(combined) == 2
    with pytest.raises(DataError):
        people.rows(Group("xor", [Condition("부서", "empty")]))


def test_quick_search_all_columns(people):
    assert names(people, quick_search("개발")) == ["이영희", "박민수"]
    assert names(people, quick_search("00125")) == ["박민수"]
    assert names(people, quick_search("영업 kim")) == ["Kim Minsu"]
    assert names(people, quick_search("2022")) == ["이영희"]
    assert names(people, quick_search("   ")) == names(people, None)
    assert names(people, quick_search("없는말")) == []
    assert names(people, "개발") == ["이영희", "박민수"]


def test_printed_filters(people):
    ids = people.ids()
    people.mark_printed(ids[:2])
    assert names(people, printed_only()) == ["김철수", "이영희"]
    assert names(people, unprinted_only()) == ["박민수", "최지우", "Kim Minsu"]
    assert names(people, all_of(unprinted_only(), Condition("부서", "eq", "개발부"))) == ["박민수"]
    assert names(people, Condition("_printed", "eq", True)) == ["김철수", "이영희"]
    assert names(people, Condition("_print_count", "ge", 1)) == ["김철수", "이영희"]
    assert names(people, Condition("_id", "eq", ids[2])) == ["박민수"]


def test_compiles_to_parameterized_sql_only(people):
    evil = "x' OR '1'='1"
    sql, params = compile_filter(all_of(Condition("이름", "eq", evil), quick_search(evil)), people)
    assert evil not in sql and "OR '1'" not in sql
    assert evil in params
    assert names(people, Condition("이름", "eq", evil)) == []
    assert names(people, Condition("이름", "contains", "'; DROP TABLE records; --")) == []
    assert people.count() == 5
    assert compile_filter(None, people) == ("1 = 1", [])


def test_hostile_column_names_in_filters(store):
    name = 'a" OR 1=1 --'
    store.add_column(name)
    store.insert_many([{name: "x"}, {name: "y"}])
    assert store.count(Condition(name, "eq", "x")) == 1
    with pytest.raises(DataError):
        store.count(Condition('id" OR 1=1 --', "eq", "x"))


def test_bad_operator_missing_value_unknown_column(people):
    with pytest.raises(DataError, match="지원하지 않는"):
        people.rows(Condition("이름", "regex", "x"))
    with pytest.raises(DataError, match="값"):
        people.rows(Condition("이름", "eq", ""))
    with pytest.raises(DataError, match="찾을 수 없습니다"):
        people.rows(Condition("없는 열", "eq", "x"))
    with pytest.raises(DataError):
        people.rows(object())


def test_operator_catalog():
    assert set(operators_for("text")) <= set(OPERATORS)
    assert "contains" in operators_for("text") and "gt" not in operators_for("text")
    assert "between" in operators_for("date") and "contains" not in operators_for("number")
    assert all(isinstance(label, str) and label for label in OPERATORS.values())
