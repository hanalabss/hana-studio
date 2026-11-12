import sys
from pathlib import Path

import pytest

ROOT = str(Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.store import DataStore  # noqa: E402


@pytest.fixture
def store(tmp_path):
    with DataStore(tmp_path / "people.hsdata") as opened:
        yield opened


@pytest.fixture
def people(store):
    store.add_column("이름")
    store.add_column("사번")
    store.add_column("부서")
    store.add_column("나이", "number")
    store.add_column("입사일", "date")
    store.insert_many(
        [
            {"이름": "김철수", "사번": "00123", "부서": "영업부", "나이": 31, "입사일": "2020-03-02"},
            {"이름": "이영희", "사번": "00124", "부서": "개발부", "나이": 27, "입사일": "2022.7.1"},
            {"이름": "박민수", "사번": "00125", "부서": "개발부", "나이": 45, "입사일": "15/01/05"},
            {"이름": "최지우", "사번": "00126", "부서": "", "나이": None, "입사일": None},
            {"이름": "Kim Minsu", "사번": "A-7", "부서": "영업부", "나이": 38, "입사일": "2019-11-11"},
        ]
    )
    return store
