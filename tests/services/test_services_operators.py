"""작업자 계정과 인쇄 기록의 작업자."""

import json
import sqlite3

import pytest

from printer.history import JobHistory
from services.operators import OperatorBook, OperatorError


@pytest.fixture
def book(tmp_path):
    return OperatorBook(tmp_path / "operators.json")


def test_without_operators_nothing_changes(book):
    assert not book.enabled and book.current == "" and book.can_manage()


def test_first_operator_is_admin_and_signed_in(book, tmp_path):
    book.add("김하나", "1234")
    first = book.find("김하나")
    assert first.admin and book.current == "김하나" and book.enabled
    raw = (tmp_path / "operators.json").read_text(encoding="utf-8")
    assert "1234" not in raw and json.loads(raw)["operators"][0]["pin_hash"]
    again = OperatorBook(tmp_path / "operators.json")
    assert again.names() == ["김하나"] and again.current == ""        # 다시 켜면 다시 확인


def test_sign_in_and_permissions(book):
    book.add("김하나", "1234")
    book.add("이둘", "5678")
    assert not book.find("이둘").admin
    with pytest.raises(OperatorError, match="맞지 않습니다"):
        book.sign_in("이둘", "0000")
    book.sign_in("이둘", "5678")
    assert book.current == "이둘" and not book.can_manage()
    with pytest.raises(OperatorError, match="관리자만"):
        book.add("삼", "1111")
    with pytest.raises(OperatorError, match="관리자만"):
        book.remove("김하나")
    book.sign_in("김하나", "1234")
    assert book.can_manage()


def test_input_rules(book):
    with pytest.raises(OperatorError, match="이름"):
        book.add("  ", "1234")
    with pytest.raises(OperatorError, match="숫자 4~8자리"):
        book.add("김하나", "12a4")
    with pytest.raises(OperatorError, match="숫자 4~8자리"):
        book.add("김하나", "123")
    book.add("김하나", "1234")
    with pytest.raises(OperatorError, match="이미 있습니다"):
        book.add("김하나", "9999")


def test_last_admin_cannot_be_removed_while_others_remain(book):
    book.add("김하나", "1234")
    book.add("이둘", "5678")
    with pytest.raises(OperatorError, match="관리자가 한 명은"):
        book.remove("김하나")
    book.remove("이둘")
    book.remove("김하나")                       # 혼자 남은 관리자는 지울 수 있다 → 작업자 확인을 끈다
    assert not book.enabled and book.current == ""


def test_change_pin(book):
    book.add("김하나", "1234")
    with pytest.raises(OperatorError, match="지금 비밀번호"):
        book.change_pin("김하나", "0000", "4321")
    book.change_pin("김하나", "1234", "4321")
    assert book.verify("김하나", "4321") and not book.verify("김하나", "1234")


def test_broken_file_is_reported(tmp_path):
    path = tmp_path / "operators.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(OperatorError, match="읽지 못했습니다"):
        OperatorBook(path)


def test_history_records_the_operator(tmp_path, book):
    history = JobHistory(str(tmp_path / "history.sqlite"))
    plain = history.start_job("작업자 없이")
    history.operator_provider = lambda: book.current
    book.add("김하나", "1234")
    job = history.start_job("사원증")
    assert history.get_job(job).operator == "김하나" and history.get_job(plain).operator is None
    assert [j.id for j in history.list_jobs(search="김하나")] == [job]


def test_old_history_file_gets_the_operator_column(tmp_path):
    path = tmp_path / "old.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE jobs (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL DEFAULT '', "
                           "family TEXT NOT NULL DEFAULT '', device_name TEXT NOT NULL DEFAULT '', "
                           "device_address TEXT NOT NULL DEFAULT '', total_cards INTEGER NOT NULL DEFAULT 0, "
                           "success_cards INTEGER NOT NULL DEFAULT 0, failed_cards INTEGER NOT NULL DEFAULT 0, "
                           "status TEXT NOT NULL DEFAULT 'pending', error_text TEXT, options_json TEXT, "
                           "created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT)")
        connection.execute("INSERT INTO jobs (name, created_at) VALUES ('예전', '2026-01-01T00:00:00')")
    history = JobHistory(str(path))
    assert history.list_jobs()[0].operator is None


def test_set_admin_keeps_at_least_one_admin(book):
    book.add("김하나", "1234")
    book.add("이둘", "5678")
    with pytest.raises(OperatorError, match="한 명은"):
        book.set_admin("김하나", False)                     # 하나뿐인 관리자는 뺄 수 없다
    book.set_admin("이둘", True)
    book.set_admin("김하나", False)                         # 다른 관리자가 생기면 뺄 수 있다
    assert book.find("이둘").admin and not book.find("김하나").admin
    assert not book.can_manage()                            # 지금 작업자(김하나)는 이제 관리자가 아니다
    with pytest.raises(OperatorError, match="관리자만"):
        book.set_admin("김하나", True)
    assert OperatorBook(book.path).find("이둘").admin       # 파일에 남는다
