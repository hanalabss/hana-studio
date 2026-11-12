"""인쇄 기록에 그때의 사람 이름과 명단을 함께 남긴다. 예전 기록 파일은 그대로 열린다."""
import sqlite3

from printer.base import PrinterDevice
from printer.history import SCHEMA_VERSION, JobHistory

_OLD_SCHEMA = """
CREATE TABLE jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL DEFAULT '', family TEXT NOT NULL DEFAULT '',
    device_name TEXT NOT NULL DEFAULT '', device_address TEXT NOT NULL DEFAULT '',
    total_cards INTEGER NOT NULL DEFAULT 0, success_cards INTEGER NOT NULL DEFAULT 0,
    failed_cards INTEGER NOT NULL DEFAULT 0, status TEXT NOT NULL DEFAULT 'pending', error_text TEXT,
    options_json TEXT, created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT);
CREATE TABLE cards (
    id INTEGER PRIMARY KEY AUTOINCREMENT, job_id INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    card_index INTEGER NOT NULL, record_id TEXT, status TEXT NOT NULL, error_text TEXT, error_code INTEGER,
    started_at TEXT, finished_at TEXT);
"""


def test_version_1_file_gains_columns_and_keeps_its_rows(tmp_path):
    path = tmp_path / "old.sqlite"
    connection = sqlite3.connect(path)
    connection.executescript(_OLD_SCHEMA)
    connection.execute("INSERT INTO jobs (name, total_cards, success_cards, status, created_at) "
                       "VALUES ('예전 인쇄', 1, 1, 'completed', '2026-01-05T09:00:00')")
    connection.execute("INSERT INTO cards (job_id, card_index, record_id, status) VALUES (1, 1, '7', 'success')")
    connection.execute("PRAGMA user_version = 1")
    connection.commit()
    connection.close()

    history = JobHistory(str(path))
    job = history.get_job(1)
    assert (job.name, job.status, job.list_id, job.list_name) == ("예전 인쇄", "completed", None, None)
    card = history.list_cards(1)[0]
    assert (card.record_id, card.status, card.record_name) == ("7", "success", None)
    connection = sqlite3.connect(path)
    assert connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION == 2
    connection.close()

    JobHistory(str(path))                              # 두 번 열어도 다시 고치지 않는다
    new_job = history.start_job("새 인쇄", PrinterDevice("rtai", "R600", "usb", "R600"), 1)
    history.record_card(new_job, 1, "success", record_id=7, record_name="김하나")
    assert history.list_cards(new_job)[0].record_name == "김하나"


def test_names_and_list_identity_are_stored_and_searchable(tmp_path):
    history = JobHistory(str(tmp_path / "history.sqlite"))
    job_id = history.start_job("3월 사원증", None, 2)
    history.record_card(job_id, 1, "success", record_id=1)
    history.record_card(job_id, 2, "failed", record_id=2, error_text="리본 없음")
    assert history.set_card_name(job_id, 1, "강민서") == 1
    assert history.set_card_name(job_id, 2, "고은채") == 1
    assert history.set_card_name(job_id, 9, "없는 카드") == 0
    history.set_job_list(job_id, "abc123", "신입사원_명단")
    other = history.start_job("방문증", None, 1)
    history.record_card(other, 1, "success")

    job = history.get_job(job_id)
    assert (job.list_id, job.list_name) == ("abc123", "신입사원_명단")
    assert [card.record_name for card in history.list_cards(job_id)] == ["강민서", "고은채"]
    assert history.list_cards(other)[0].record_name is None
    assert [found.id for found in history.list_jobs(search="은채")] == [job_id]
    assert history.count_jobs(search="은채") == 1
    assert [found.id for found in history.list_jobs(search="방문")] == [other]      # 인쇄 이름으로도 그대로 찾는다
    assert [found.status for found in history.list_jobs()] == ["pending", "pending"]
