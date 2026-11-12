import pytest

from data.errors import DataError, RangeError
from data.filters import Condition
from data.selection import PrintSelection, format_ranges, parse_ranges


def test_parse_ranges_basic():
    assert parse_ranges("1,3,4-9", 10) == [1, 3, 4, 5, 6, 7, 8, 9]
    assert parse_ranges("5", 5) == [5]
    assert parse_ranges(" 2 , 4 - 6 ", 10) == [2, 4, 5, 6]
    assert parse_ranges("3-3", 3) == [3]


def test_parse_ranges_friendly_variants():
    assert parse_ranges("1~3", 5) == [1, 2, 3]
    assert parse_ranges("1，3 5", 5) == [1, 3, 5]
    assert parse_ranges("4-", 6) == [4, 5, 6]
    assert parse_ranges("-2", 6) == [1, 2]
    assert parse_ranges("1,3,", 6) == [1, 3]
    assert parse_ranges("5,1,3-4,1", 6) == [5, 1, 3, 4]
    assert parse_ranges("2-4,3-5", 6) == [2, 3, 4, 5]


@pytest.mark.parametrize(
    "text,total,fragment",
    [
        ("", 10, "번호를 적어"),
        ("   ", 10, "번호를 적어"),
        (",", 10, "번호를 적어"),
        ("11", 10, "11번은 범위를 벗어났습니다"),
        ("1-20", 10, "20번은 범위를 벗어났습니다"),
        ("0", 10, "1부터"),
        ("0-3", 10, "1부터"),
        ("9-4", 10, "시작 번호(9)가 끝 번호(4)보다"),
        ("abc", 10, "'abc'"),
        ("1,x-3", 10, "'x-3'"),
        ("1-2-3", 10, "'1-2-3'"),
        ("-", 10, "'-'"),
        ("1.5", 10, "'1.5'"),
        ("3", 0, "뽑을 사람이 없습니다"),
    ],
)
def test_parse_ranges_errors_are_korean_and_specific(text, total, fragment):
    with pytest.raises(RangeError) as caught:
        parse_ranges(text, total)
    assert fragment in str(caught.value)
    assert isinstance(caught.value, ValueError) and isinstance(caught.value, DataError)


def test_out_of_range_message_mentions_total():
    with pytest.raises(RangeError, match="모두 10명"):
        parse_ranges("12", 10)


def test_format_ranges():
    assert format_ranges([1, 3, 4, 5, 9]) == "1,3-5,9"
    assert format_ranges([]) == ""
    assert parse_ranges(format_ranges([7, 2, 3, 1]), 10) == [1, 2, 3, 7]


def test_selection_all_and_current(people):
    ids = people.ids()
    assert PrintSelection.all().resolve(people) == ids
    by_age = people.ids(order=("나이", "desc"))
    assert PrintSelection.all().resolve(people, order=("나이", "desc")) == by_age
    dev = Condition("부서", "eq", "개발부")
    assert PrintSelection.all().resolve(people, filter=dev) == ids[1:3]
    assert PrintSelection.current(ids[3]).resolve(people) == [ids[3]]
    with pytest.raises(DataError):
        PrintSelection.current(9999).resolve(people)


def test_selection_ranges_follow_the_visible_order(people):
    ids = people.ids()
    assert PrintSelection.ranges("1,4-5").resolve(people) == [ids[0], ids[3], ids[4]]
    by_age = people.ids(order=("나이", "desc"))
    assert PrintSelection.ranges("1-2").resolve(people, order=("나이", "desc")) == by_age[:2]
    dev = Condition("부서", "eq", "개발부")
    assert PrintSelection.ranges("2").resolve(people, filter=dev) == [ids[2]]
    with pytest.raises(RangeError, match="모두 2명"):
        PrintSelection.ranges("3").resolve(people, filter=dev)


def test_selection_explicit_ids(people):
    ids = people.ids()
    picked = PrintSelection.ids([ids[4], ids[0], ids[4], 777])
    assert picked.resolve(people) == [ids[0], ids[4]]
    assert picked.count(people) == 2
    by_age_desc = picked.resolve(people, order=("나이", "desc"))
    assert by_age_desc == [ids[4], ids[0]]
    people.delete(ids[0])
    assert picked.resolve(people) == [ids[4]]
    assert PrintSelection.ids([]).resolve(people) == []


def test_selection_by_filter_and_unprinted(people):
    ids = people.ids()
    sales = PrintSelection.by_filter(Condition("부서", "eq", "영업부"))
    assert sales.resolve(people) == [ids[0], ids[4]]

    people.mark_printed([ids[0], ids[1]])
    assert PrintSelection.unprinted().resolve(people) == ids[2:]
    dev = Condition("부서", "eq", "개발부")
    assert PrintSelection.unprinted().resolve(people, filter=dev) == [ids[2]]
    people.mark_printed(PrintSelection.unprinted().resolve(people))
    assert PrintSelection.unprinted().resolve(people) == []


def test_selection_descriptions_are_korean(people):
    for selection in [
        PrintSelection.all(),
        PrintSelection.current(1),
        PrintSelection.ranges("1-3"),
        PrintSelection.ids([1, 2]),
        PrintSelection.by_filter(None),
        PrintSelection.unprinted(),
    ]:
        assert selection.describe()
    assert PrintSelection.ids([1, 2]).describe() == "고른 2명"
