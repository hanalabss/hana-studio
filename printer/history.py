"""
printer/history.py
인쇄 작업 이력(SQLite). 작업 한 건과 그 안의 카드별 결과를 남겨 앱을 다시 켜도 볼 수 있게 한다.

연결은 호출마다 열고 닫으므로 어느 스레드에서 불러도 된다.
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .base import PrinterDevice

SCHEMA_VERSION = 2

JOB_PENDING = "pending"
JOB_PRINTING = "printing"
JOB_COMPLETED = "completed"       # 모든 카드 성공
JOB_PARTIAL = "partial"           # 일부만 성공
JOB_FAILED = "failed"
JOB_CANCELLED = "cancelled"

CARD_SUCCESS = "success"
CARD_FAILED = "failed"
CARD_CANCELLED = "cancelled"      # 중단·취소로 인쇄하지 않음

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT NOT NULL DEFAULT '',
    family        TEXT NOT NULL DEFAULT '',
    device_name   TEXT NOT NULL DEFAULT '',
    device_address TEXT NOT NULL DEFAULT '',
    total_cards   INTEGER NOT NULL DEFAULT 0,
    success_cards INTEGER NOT NULL DEFAULT 0,
    failed_cards  INTEGER NOT NULL DEFAULT 0,
    status        TEXT NOT NULL DEFAULT 'pending',
    error_text    TEXT,
    options_json  TEXT,
    created_at    TEXT NOT NULL,
    started_at    TEXT,
    finished_at   TEXT
);
CREATE TABLE IF NOT EXISTS cards (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id      INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    card_index  INTEGER NOT NULL,
    record_id   TEXT,
    status      TEXT NOT NULL,
    error_text  TEXT,
    error_code  INTEGER,
    started_at  TEXT,
    finished_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_cards_job ON cards(job_id, card_index);
CREATE INDEX IF NOT EXISTS idx_cards_record ON cards(record_id);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at);
"""

# 버전 2에서 더한 열. 예전 파일에는 ALTER TABLE로 붙이고, 예전 줄의 값은 NULL로 남는다.
#   cards.record_name : 인쇄할 때의 사람 이름(명단이 바뀌거나 다른 명단을 열어도 그대로 보여 준다)
#   jobs.list_id/name : 어느 명단으로 뽑았는지(DataStore.list_id()). record_id는 그 명단 안에서만 뜻이 있다
_ADDED_COLUMNS = (
    ("cards", "record_name", "TEXT"),
    ("jobs", "list_id", "TEXT"),
    ("jobs", "list_name", "TEXT"),
)


@dataclass
class JobRecord:
    id: int
    name: str
    family: str
    device_name: str
    device_address: str
    total_cards: int
    success_cards: int
    failed_cards: int
    status: str
    error_text: Optional[str]
    options: Dict[str, Any]
    created_at: str
    started_at: Optional[str]
    finished_at: Optional[str]
    list_id: Optional[str] = None
    list_name: Optional[str] = None


@dataclass
class CardRecord:
    id: int
    job_id: int
    card_index: int
    record_id: Optional[str]
    status: str
    error_text: Optional[str]
    error_code: Optional[int]
    started_at: Optional[str]
    finished_at: Optional[str]
    record_name: Optional[str] = None


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def default_history_path() -> str:
    """사용자별 쓰기 가능한 위치의 이력 파일 경로."""
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    return os.path.join(base, "HanaStudio", "print_history.sqlite")


class JobHistory:
    def __init__(self, path: Optional[str] = None):
        self.path = path or default_history_path()
        directory = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(directory, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(_SCHEMA)
            self._migrate(connection)

    @staticmethod
    def _migrate(connection: sqlite3.Connection) -> None:
        """빠진 열만 더한다. 있는 기록은 건드리지 않는다."""
        for table, column, kind in _ADDED_COLUMNS:
            existing = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
            if column not in existing:
                connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {kind}")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_cards_name ON cards(record_name)")
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if version < SCHEMA_VERSION:
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    # -- 기록 -------------------------------------------------------------
    def start_job(self, name: str = "", device: Optional[PrinterDevice] = None,
                  total_cards: int = 0, options: Optional[Dict[str, Any]] = None) -> int:
        """작업 한 건을 만들고 이력 ID를 돌려준다(상태: pending)."""
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO jobs (name, family, device_name, device_address, total_cards, "
                "status, options_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (name, device.family if device else "", device.name if device else "",
                 device.address if device else "", int(total_cards), JOB_PENDING,
                 json.dumps(options, ensure_ascii=False, default=str) if options else None,
                 now_text()))
            return int(cursor.lastrowid)

    def mark_job_started(self, job_id: int) -> None:
        with self._connect() as connection:
            connection.execute("UPDATE jobs SET status = ?, started_at = ? WHERE id = ?",
                               (JOB_PRINTING, now_text(), job_id))

    def record_card(self, job_id: int, card_index: int, status: str,
                    error_text: Optional[str] = None, error_code: Optional[int] = None,
                    record_id: Any = None, started_at: Optional[str] = None,
                    finished_at: Optional[str] = None, record_name: Optional[str] = None) -> int:
        """카드 한 장의 결과를 남긴다. card_index는 작업 안에서 1부터 센 순번."""
        finished_at = finished_at or now_text()
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO cards (job_id, card_index, record_id, status, error_text, "
                "error_code, started_at, finished_at, record_name) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (job_id, int(card_index), None if record_id is None else str(record_id), status,
                 error_text, error_code, started_at, finished_at, record_name or None))
            column = {CARD_SUCCESS: "success_cards", CARD_FAILED: "failed_cards"}.get(status)
            if column:
                connection.execute(f"UPDATE jobs SET {column} = {column} + 1 WHERE id = ?",
                                   (job_id,))
            return int(cursor.lastrowid)

    def set_job_list(self, job_id: int, list_id: Optional[str], list_name: Optional[str] = None) -> None:
        """이 작업을 어느 명단으로 뽑았는지 적는다."""
        with self._connect() as connection:
            connection.execute("UPDATE jobs SET list_id = ?, list_name = ? WHERE id = ?",
                               (list_id or None, list_name or None, job_id))

    def set_card_name(self, job_id: int, card_index: int, record_name: Optional[str]) -> int:
        """이미 남긴 카드 결과에 그때의 사람 이름을 적는다. 고친 줄 수를 돌려준다."""
        with self._connect() as connection:
            return connection.execute(
                "UPDATE cards SET record_name = ? WHERE job_id = ? AND card_index = ?",
                (record_name or None, job_id, int(card_index))).rowcount

    def finish_job(self, job_id: int, status: Optional[str] = None,
                   error_text: Optional[str] = None) -> str:
        """작업을 끝낸다. status를 주지 않으면 카드 결과로 정한다. 정해진 상태를 돌려준다."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT total_cards, success_cards, failed_cards FROM jobs WHERE id = ?",
                (job_id,)).fetchone()
            if row is None:
                raise KeyError(f"작업 이력 {job_id}을(를) 찾을 수 없습니다.")
            if status is None:
                if row["success_cards"] and row["success_cards"] >= row["total_cards"]:
                    status = JOB_COMPLETED
                elif row["success_cards"]:
                    status = JOB_PARTIAL
                else:
                    status = JOB_FAILED
            connection.execute(
                "UPDATE jobs SET status = ?, error_text = COALESCE(?, error_text), "
                "finished_at = ? WHERE id = ?", (status, error_text, now_text(), job_id))
            return status

    # -- 조회 -------------------------------------------------------------
    @staticmethod
    def _job_filter(status: Optional[str], since: Optional[str], until: Optional[str],
                    search: Optional[str], family: Optional[str]) -> Tuple[str, List[Any]]:
        clauses: List[str] = []
        params: List[Any] = []
        for column, operator, value in (("status", "=", status), ("family", "=", family),
                                        ("created_at", ">=", since), ("created_at", "<=", until)):
            if value:
                clauses.append(f"{column} {operator} ?")
                params.append(value)
        if search:
            clauses.append("(name LIKE ? OR device_name LIKE ? OR error_text LIKE ? "
                           "OR id IN (SELECT job_id FROM cards WHERE record_name LIKE ?))")
            params.extend([f"%{search}%"] * 4)
        return (" WHERE " + " AND ".join(clauses)) if clauses else "", params

    def list_jobs(self, limit: int = 100, offset: int = 0, status: Optional[str] = None,
                  since: Optional[str] = None, until: Optional[str] = None,
                  search: Optional[str] = None, family: Optional[str] = None) -> List[JobRecord]:
        """최근 작업부터 돌려준다. since/until은 ISO 날짜·시각 문자열."""
        where, params = self._job_filter(status, since, until, search, family)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM jobs{where} ORDER BY id DESC LIMIT ? OFFSET ?",
                params + [int(limit), int(offset)]).fetchall()
        return [self._job(row) for row in rows]

    def count_jobs(self, status: Optional[str] = None, since: Optional[str] = None,
                   until: Optional[str] = None, search: Optional[str] = None,
                   family: Optional[str] = None) -> int:
        where, params = self._job_filter(status, since, until, search, family)
        with self._connect() as connection:
            return int(connection.execute(f"SELECT COUNT(*) FROM jobs{where}", params).fetchone()[0])

    def get_job(self, job_id: int) -> Optional[JobRecord]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return self._job(row) if row else None

    def list_cards(self, job_id: int) -> List[CardRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM cards WHERE job_id = ? ORDER BY card_index, id", (job_id,)).fetchall()
        return [CardRecord(**dict(row)) for row in rows]

    def cards_for_record(self, record_id: Any) -> List[CardRecord]:
        """데이터 레코드 하나가 인쇄된 이력(최근 것부터)."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM cards WHERE record_id = ? ORDER BY id DESC", (str(record_id),)).fetchall()
        return [CardRecord(**dict(row)) for row in rows]

    # -- 정리 -------------------------------------------------------------
    def delete_job(self, job_id: int) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM jobs WHERE id = ?", (job_id,))

    def purge_older_than(self, days: int) -> int:
        """days일보다 오래된 작업을 지우고 지운 건수를 돌려준다."""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
        with self._connect() as connection:
            return connection.execute("DELETE FROM jobs WHERE created_at < ?", (cutoff,)).rowcount

    def close_interrupted_jobs(self) -> int:
        """앱이 인쇄 도중 꺼져 pending/printing으로 남은 작업을 실패로 닫는다."""
        with self._connect() as connection:
            return connection.execute(
                "UPDATE jobs SET status = ?, error_text = COALESCE(error_text, ?), finished_at = ? "
                "WHERE status IN (?, ?)",
                (JOB_FAILED, "프로그램이 인쇄 도중 종료되었습니다.", now_text(),
                 JOB_PENDING, JOB_PRINTING)).rowcount

    @staticmethod
    def _job(row: sqlite3.Row) -> JobRecord:
        data = dict(row)
        raw_options = data.pop("options_json")
        try:
            options = json.loads(raw_options) if raw_options else {}
        except ValueError:
            options = {}
        return JobRecord(options=options, **data)
