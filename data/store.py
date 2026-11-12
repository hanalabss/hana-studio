from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping

from .errors import DataError
from .values import COLUMN_TYPES, normalize_value

Record = dict[str, Any]

SYSTEM_KEYS = ("_id", "_printed", "_print_count", "_printed_at")
_SYSTEM_SQL = {
    "_id": ("id", "number"),
    "_printed": ("printed", "bool"),
    "_print_count": ("print_count", "number"),
    "_printed_at": ("printed_at", "text"),
}
_CHUNK = 500


@dataclass
class Column:
    name: str
    type: str = "text"
    options: dict = field(default_factory=dict)
    id: int | None = None


@dataclass(frozen=True)
class PhotoAdjustment:
    """레코드 한 건의 사진 한 장을 틀(디자인의 이미지 항목) 안에서 어떻게 놓을지.

    zoom      배율(단위 없음). 1.0 = 항목의 맞춤 방식(fill·contain·stretch)대로 틀에 맞춘 크기,
              1.5 = 그보다 1.5배 크게. 0보다 커야 한다(저장할 때 검사한다).
    offset_x  사진을 가로로 옮기는 거리 ÷ 틀의 너비. +면 오른쪽, -면 왼쪽.
              0.1 = 틀 너비의 10%만큼 오른쪽으로. 0 = 가운데.
    offset_y  사진을 세로로 옮기는 거리 ÷ 틀의 높이. +면 아래, -면 위.

    offset은 사진 크기나 픽셀이 아니라 **틀 크기에 대한 비율**이라 디자인의 틀 크기가 바뀌어도
    같은 구도로 남는다. 보통 -1.0 ~ 1.0 안에서 쓰지만 저장소는 범위를 강제하지 않는다. 인쇄할 때
    렌더러의 design.Crop(남는 폭 기준 -1..1)으로 바꾸면서 틀에 빈 곳이 생기지 않게 잘린다
    (services.render_job.crop_from_adjustment).

    PhotoAdjustment()는 "조정 없음"과 같은 모양이다. 저장된 값이 정말 없는지는
    DataStore.has_photo_adjustment 또는 get_photo_adjustment(..., default=None)으로 구별한다.
    """

    zoom: float = 1.0
    offset_x: float = 0.0
    offset_y: float = 0.0


def _migrate_v1(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS columns ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " name TEXT NOT NULL UNIQUE,"
        " type TEXT NOT NULL,"
        " options TEXT NOT NULL DEFAULT '{}',"
        " position INTEGER NOT NULL)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS records ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " printed INTEGER NOT NULL DEFAULT 0,"
        " print_count INTEGER NOT NULL DEFAULT 0,"
        " printed_at TEXT)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS photo_adjust ("
        " record_id INTEGER NOT NULL,"
        " column_id INTEGER NOT NULL,"
        " zoom REAL NOT NULL,"
        " offset_x REAL NOT NULL,"
        " offset_y REAL NOT NULL,"
        " PRIMARY KEY (record_id, column_id)) WITHOUT ROWID"
    )


# 스키마를 바꿀 때는 여기에 함수를 덧붙인다. 목록 길이가 곧 스키마 버전이다.
MIGRATIONS: list[Callable[[sqlite3.Connection], None]] = [_migrate_v1]


def _physical(column_id: int) -> str:
    return f'"c{int(column_id)}"'


class DataStore:
    """명단 파일(.hsdata, SQLite) 하나.

    스레드 규칙 (코드가 보장하는 것):
      - 연결은 하나이고 모든 공개 메서드가 재진입 잠금(RLock) 안에서 그 연결을 쓴다. 그래서
        어느 스레드에서 불러도 되고, 호출은 한 번에 하나씩 차례로 실행된다(동시에 읽지 않는다).
      - transaction() 블록은 끝날 때까지 잠금을 쥔다. 그동안 다른 스레드의 읽기·쓰기는 기다렸다가
        커밋(또는 되돌림)된 뒤의 상태만 본다. 블록 안에서는 오래 걸리는 일(대화상자, 파일 변환,
        네트워크)을 하지 않는다. 같은 스레드에서는 중첩해 쓸 수 있고 가장 바깥 블록이 끝날 때 반영된다.
      - insert_many에 넘긴 이터러블은 그 트랜잭션 안에서(잠금을 쥔 채) 소비된다.
      - rows()/get()이 돌려주는 레코드는 그 시점의 사본(dict)이다. 고쳐도 저장되지 않고,
        다른 스레드가 값을 바꿔도 따라 바뀌지 않는다.
    보장하지 않는 것:
      - 변경 알림은 없다. 다른 스레드가 바꾼 내용은 다시 읽어야 보인다(화면 갱신은 부른 쪽 몫).
      - 메서드 여러 개를 이어 부르는 동안의 일관성. 묶어야 하면 transaction()으로 감싼다.
      - close() 뒤의 호출은 DataError다. 다른 스레드가 쓰는 중에 닫지 않도록 부른 쪽이 순서를 지킨다.
      - 같은 파일을 DataStore 여러 개(또는 다른 프로세스)로 동시에 여는 경우의 쓰기 충돌.
        한 파일은 DataStore 하나로 열어 공유한다.
    앱에서는 화면(UI) 스레드가 명단을 읽고 쓰고, 인쇄 표시(mark_printed)도 UI 스레드의
    PrintService가 남긴다. 가져오기처럼 오래 걸리는 쓰기만 작업 스레드에서 한다.
    """

    def __init__(self, path: str | Path):
        self.path = str(path)
        self._lock = threading.RLock()
        self._depth = 0
        self._conn: sqlite3.Connection | None = None
        try:
            if self.path != ":memory:":
                Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False)
            self._conn = conn
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._migrate()
        except sqlite3.Error as exc:
            self.close()
            raise DataError(f"데이터 파일을 열 수 없습니다: {self.path}") from exc
        except DataError:
            self.close()
            raise

    # ---- 수명 ----

    def close(self) -> None:
        with self._lock:
            conn, self._conn = self._conn, None
            if conn is None:
                return
            try:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except sqlite3.Error:
                pass
            conn.close()

    def __enter__(self) -> "DataStore":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    @property
    def schema_version(self) -> int:
        row = self._db.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
        return int(row[0]) if row else 0

    @property
    def _db(self) -> sqlite3.Connection:
        if self._conn is None:
            raise DataError("데이터 파일이 닫혀 있습니다.")
        return self._conn

    def _migrate(self) -> None:
        conn = self._db
        conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
        current = self.schema_version
        target = len(MIGRATIONS)
        if current > target:
            raise DataError(
                "이 데이터 파일은 더 최신 버전의 프로그램에서 만들어졌습니다. 프로그램을 업데이트한 뒤 다시 열어 주세요."
            )
        for version in range(current, target):
            with self.transaction():
                MIGRATIONS[version](conn)
                conn.execute(
                    "INSERT INTO meta (key, value) VALUES ('schema_version', ?)"
                    " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (str(version + 1),),
                )

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """여러 변경을 한 번에 반영한다. 도중에 오류가 나면 전부 되돌린다. 중첩 가능."""
        with self._lock:
            conn = self._db
            if self._depth:
                self._depth += 1
                try:
                    yield
                finally:
                    self._depth -= 1
                return
            conn.execute("BEGIN IMMEDIATE")
            self._depth = 1
            try:
                yield
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            else:
                conn.execute("COMMIT")
            finally:
                self._depth = 0

    # ---- 열 ----

    def columns(self) -> list[Column]:
        with self._lock:
            rows = self._db.execute(
                "SELECT id, name, type, options FROM columns ORDER BY position, id"
            ).fetchall()
        return [Column(name, type_, json.loads(options or "{}"), id_) for id_, name, type_, options in rows]

    def column_names(self) -> list[str]:
        return [c.name for c in self.columns()]

    def column(self, name: str) -> Column:
        with self._lock:
            row = self._db.execute(
                "SELECT id, name, type, options FROM columns WHERE name = ?", (name,)
            ).fetchone()
        if row is None:
            raise DataError(f"'{name}' 열을 찾을 수 없습니다.")
        return Column(row[1], row[2], json.loads(row[3] or "{}"), row[0])

    def has_column(self, name: str) -> bool:
        with self._lock:
            return self._db.execute("SELECT 1 FROM columns WHERE name = ?", (name,)).fetchone() is not None

    def _check_name(self, name: Any, ignore_id: int | None = None) -> str:
        if not isinstance(name, str) or name.strip() == "":
            raise DataError("열 이름을 입력하세요.")
        name = name.strip()
        if name.startswith("_"):
            raise DataError(f"열 이름은 밑줄(_)로 시작할 수 없습니다: {name}")
        row = self._db.execute("SELECT id FROM columns WHERE name = ?", (name,)).fetchone()
        if row is not None and row[0] != ignore_id:
            raise DataError(f"'{name}' 열이 이미 있습니다. 다른 이름을 사용하세요.")
        return name

    @staticmethod
    def _check_type(type_: str) -> str:
        if type_ not in COLUMN_TYPES:
            raise DataError(f"지원하지 않는 열 종류입니다: {type_}")
        return type_

    def add_column(
        self, name: str, type: str = "text", options: Mapping | None = None, position: int | None = None
    ) -> Column:
        with self.transaction():
            conn = self._db
            name = self._check_name(name)
            self._check_type(type)
            end = conn.execute("SELECT COALESCE(MAX(position), -1) + 1 FROM columns").fetchone()[0]
            cursor = conn.execute(
                "INSERT INTO columns (name, type, options, position) VALUES (?, ?, ?, ?)",
                (name, type, json.dumps(dict(options or {}), ensure_ascii=False), end),
            )
            column_id = cursor.lastrowid
            conn.execute(f"ALTER TABLE records ADD COLUMN {_physical(column_id)}")
            if position is not None:
                names = [c.name for c in self.columns() if c.id != column_id]
                names.insert(max(0, min(int(position), len(names))), name)
                self._set_order(names)
            return self.column(name)

    def rename_column(self, old: str, new: str) -> None:
        with self.transaction():
            column = self.column(old)
            new = self._check_name(new, ignore_id=column.id)
            self._db.execute("UPDATE columns SET name = ? WHERE id = ?", (new, column.id))

    def remove_column(self, name: str) -> None:
        with self.transaction():
            column = self.column(name)
            conn = self._db
            conn.execute(f"ALTER TABLE records DROP COLUMN {_physical(column.id)}")
            conn.execute("DELETE FROM columns WHERE id = ?", (column.id,))
            conn.execute("DELETE FROM photo_adjust WHERE column_id = ?", (column.id,))
            self._set_order(self.column_names())

    def reorder_columns(self, names: Iterable[str]) -> None:
        """names 순서대로 앞에 놓는다. 빠진 열은 원래 순서대로 뒤에 붙는다."""
        with self.transaction():
            wanted = list(names)
            existing = self.column_names()
            for name in wanted:
                if name not in existing:
                    raise DataError(f"'{name}' 열을 찾을 수 없습니다.")
            if len(set(wanted)) != len(wanted):
                raise DataError("열 순서에 같은 열이 두 번 들어 있습니다.")
            self._set_order(wanted + [n for n in existing if n not in set(wanted)])

    def move_column(self, name: str, position: int) -> None:
        with self.transaction():
            self.column(name)
            names = [n for n in self.column_names() if n != name]
            names.insert(max(0, min(int(position), len(names))), name)
            self._set_order(names)

    def _set_order(self, names: list[str]) -> None:
        self._db.executemany(
            "UPDATE columns SET position = ? WHERE name = ?", [(i, n) for i, n in enumerate(names)]
        )

    def set_column_options(self, name: str, options: Mapping | None) -> None:
        with self.transaction():
            column = self.column(name)
            self._db.execute(
                "UPDATE columns SET options = ? WHERE id = ?",
                (json.dumps(dict(options or {}), ensure_ascii=False), column.id),
            )

    def set_column_type(self, name: str, type: str, options: Mapping | None = None) -> int:
        """열 종류를 바꾸고 기존 값을 새 종류에 맞게 변환한다. 변환하지 못해 그대로 둔 값의 개수를 돌려준다."""
        with self.transaction():
            column = self.column(name)
            self._check_type(type)
            conn = self._db
            physical = _physical(column.id)
            failed = 0
            updates = []
            for record_id, value in conn.execute(
                f"SELECT id, {physical} FROM records WHERE {physical} IS NOT NULL"
            ).fetchall():
                converted, ok = normalize_value(type, value)
                if not ok:
                    failed += 1
                if converted != value or converted.__class__ is not value.__class__:
                    updates.append((converted, record_id))
            conn.executemany(f"UPDATE records SET {physical} = ? WHERE id = ?", updates)
            new_options = column.options if options is None else dict(options)
            conn.execute(
                "UPDATE columns SET type = ?, options = ? WHERE id = ?",
                (type, json.dumps(new_options, ensure_ascii=False), column.id),
            )
            return failed

    # ---- 필터/정렬용 SQL 조각 (data.filters가 사용) ----

    def sql_column(self, name: str) -> tuple[str, str]:
        if name in _SYSTEM_SQL:
            return _SYSTEM_SQL[name]
        column = self.column(name)
        return _physical(column.id), column.type

    def sql_columns(self) -> list[tuple[str, str]]:
        return [(_physical(c.id), c.type) for c in self.columns()]

    def _where(self, filter: Any) -> tuple[str, list]:
        if filter is None:
            return "", []
        if isinstance(filter, str):
            from .filters import quick_search

            filter = quick_search(filter)
        if not hasattr(filter, "to_sql"):
            raise DataError("올바른 필터가 아닙니다.")
        sql, params = filter.to_sql(self)
        return f" WHERE {sql}", list(params)

    def _order(self, order: Any) -> str:
        if order is None:
            return " ORDER BY id"
        items = [order] if isinstance(order, (str, tuple)) else list(order)
        parts = []
        for item in items:
            if isinstance(item, str):
                name, direction = item, "asc"
            else:
                name, direction = item
            direction = str(direction).lower()
            if direction not in ("asc", "desc"):
                raise DataError(f"정렬 방향은 asc 또는 desc여야 합니다: {direction}")
            expr, _ = self.sql_column(name)
            parts.append(f"{expr} IS NULL, {expr} COLLATE NOCASE {direction.upper()}")
        parts.append("id")
        return " ORDER BY " + ", ".join(parts)

    # ---- 레코드 조회 ----

    def count(self, filter: Any = None) -> int:
        with self._lock:
            where, params = self._where(filter)
            return self._db.execute(f"SELECT COUNT(*) FROM records{where}", params).fetchone()[0]

    def rows(self, filter: Any = None, order: Any = None, offset: int = 0, limit: int | None = None) -> list[Record]:
        with self._lock:
            columns = self.columns()
            where, params = self._where(filter)
            select = "".join(f", {_physical(c.id)}" for c in columns)
            sql = (
                f"SELECT id, printed, print_count, printed_at{select} FROM records"
                f"{where}{self._order(order)} LIMIT ? OFFSET ?"
            )
            params += [-1 if limit is None else max(0, int(limit)), max(0, int(offset))]
            fetched = self._db.execute(sql, params).fetchall()
        return [self._record(columns, row) for row in fetched]

    @staticmethod
    def _record(columns: list[Column], row: tuple) -> Record:
        record: Record = {"_id": row[0]}
        for column, value in zip(columns, row[4:]):
            record[column.name] = value
        record["_printed"] = bool(row[1])
        record["_print_count"] = row[2]
        record["_printed_at"] = row[3]
        return record

    def ids(self, filter: Any = None, order: Any = None) -> list[int]:
        with self._lock:
            where, params = self._where(filter)
            sql = f"SELECT id FROM records{where}{self._order(order)}"
            return [row[0] for row in self._db.execute(sql, params)]

    def get(self, record_id: int) -> Record | None:
        with self._lock:
            columns = self.columns()
            select = "".join(f", {_physical(c.id)}" for c in columns)
            row = self._db.execute(
                f"SELECT id, printed, print_count, printed_at{select} FROM records WHERE id = ?",
                (int(record_id),),
            ).fetchone()
        return None if row is None else self._record(columns, row)

    # ---- 레코드 변경 ----

    def _column_map(self, values: Mapping, columns: dict[str, Column]) -> None:
        for key in values:
            if key not in columns and not (isinstance(key, str) and key.startswith("_")):
                raise DataError(f"'{key}' 열을 찾을 수 없습니다.")

    def insert(self, values: Mapping[str, Any]) -> int:
        with self.transaction():
            columns = {c.name: c for c in self.columns()}
            self._column_map(values, columns)
            used = [c for c in columns.values() if c.name in values]
            if not used:
                return self._db.execute("INSERT INTO records DEFAULT VALUES").lastrowid
            names = ", ".join(_physical(c.id) for c in used)
            marks = ", ".join("?" for _ in used)
            params = [normalize_value(c.type, values[c.name])[0] for c in used]
            return self._db.execute(f"INSERT INTO records ({names}) VALUES ({marks})", params).lastrowid

    def insert_many(
        self,
        rows: Iterable[Mapping[str, Any]],
        on_invalid: Callable[[int, str, Any], None] | None = None,
    ) -> int:
        """한 트랜잭션으로 여러 레코드를 넣는다. 종류에 맞지 않는 값은 글자 그대로 저장하고 on_invalid(행 순번, 열 이름, 값)을 부른다."""
        with self.transaction():
            columns = self.columns()
            by_name = {c.name: c for c in columns}
            inserted = 0

            def generate():
                nonlocal inserted
                for index, values in enumerate(rows):
                    self._column_map(values, by_name)
                    params = []
                    for column in columns:
                        value, ok = normalize_value(column.type, values.get(column.name))
                        if not ok and on_invalid is not None:
                            on_invalid(index, column.name, values.get(column.name))
                        params.append(value)
                    inserted += 1
                    yield params

            if columns:
                names = ", ".join(_physical(c.id) for c in columns)
                marks = ", ".join("?" for _ in columns)
                self._db.executemany(f"INSERT INTO records ({names}) VALUES ({marks})", generate())
            else:
                for _ in generate():
                    self._db.execute("INSERT INTO records DEFAULT VALUES")
            return inserted

    def update(self, record_id: int, values: Mapping[str, Any]) -> bool:
        with self.transaction():
            columns = {c.name: c for c in self.columns()}
            self._column_map(values, columns)
            used = [c for c in columns.values() if c.name in values]
            if not used:
                return self._db.execute("SELECT 1 FROM records WHERE id = ?", (int(record_id),)).fetchone() is not None
            assignments = ", ".join(f"{_physical(c.id)} = ?" for c in used)
            params = [normalize_value(c.type, values[c.name])[0] for c in used]
            cursor = self._db.execute(
                f"UPDATE records SET {assignments} WHERE id = ?", params + [int(record_id)]
            )
            return cursor.rowcount > 0

    @staticmethod
    def _id_list(ids: int | Iterable[int]) -> list[int]:
        if isinstance(ids, int):
            return [ids]
        return list(dict.fromkeys(int(i) for i in ids))

    def _chunks(self, ids: int | Iterable[int]) -> Iterator[tuple[str, list[int]]]:
        id_list = self._id_list(ids)
        for start in range(0, len(id_list), _CHUNK):
            chunk = id_list[start : start + _CHUNK]
            yield ", ".join("?" for _ in chunk), chunk

    def delete(self, ids: int | Iterable[int]) -> int:
        with self.transaction():
            deleted = 0
            for marks, chunk in self._chunks(ids):
                deleted += self._db.execute(f"DELETE FROM records WHERE id IN ({marks})", chunk).rowcount
                self._db.execute(f"DELETE FROM photo_adjust WHERE record_id IN ({marks})", chunk)
            return deleted

    def clear(self) -> int:
        """모든 레코드를 지운다. 열 구성은 그대로 둔다."""
        with self.transaction():
            self._db.execute("DELETE FROM photo_adjust")
            return self._db.execute("DELETE FROM records").rowcount

    # ---- 인쇄 상태 ----

    def mark_printed(self, ids: int | Iterable[int], printed: bool = True) -> int:
        with self.transaction():
            changed = 0
            stamp = datetime.now().isoformat(sep=" ", timespec="seconds")
            for marks, chunk in self._chunks(ids):
                if printed:
                    cursor = self._db.execute(
                        "UPDATE records SET printed = 1, print_count = print_count + 1, printed_at = ?"
                        f" WHERE id IN ({marks})",
                        [stamp] + chunk,
                    )
                else:
                    cursor = self._db.execute(f"UPDATE records SET printed = 0 WHERE id IN ({marks})", chunk)
                changed += cursor.rowcount
            return changed

    def reset_print_status(self, ids: int | Iterable[int] | None = None) -> int:
        with self.transaction():
            sql = "UPDATE records SET printed = 0, print_count = 0, printed_at = NULL"
            if ids is None:
                return self._db.execute(sql).rowcount
            changed = 0
            for marks, chunk in self._chunks(ids):
                changed += self._db.execute(f"{sql} WHERE id IN ({marks})", chunk).rowcount
            return changed

    # ---- 사진 조정값 ----

    def get_photo_adjustment(
        self, record_id: int, column: str, default: PhotoAdjustment | None = PhotoAdjustment()
    ) -> PhotoAdjustment | None:
        """저장된 사진 조정값. 저장된 것이 없으면 default(기본은 PhotoAdjustment() = 조정 없음).

        default=None을 주면 "저장된 값 없음"(None)과 "기본값과 같은 값을 일부러 저장함"
        (PhotoAdjustment())을 구별할 수 있다.
        """
        column_id = self.column(column).id
        with self._lock:
            row = self._db.execute(
                "SELECT zoom, offset_x, offset_y FROM photo_adjust WHERE record_id = ? AND column_id = ?",
                (int(record_id), column_id),
            ).fetchone()
        return PhotoAdjustment(*row) if row else default

    def has_photo_adjustment(self, record_id: int, column: str) -> bool:
        """이 레코드·열에 사진 조정값이 저장되어 있는지(값이 기본값과 같아도 저장되어 있으면 True)."""
        return self.get_photo_adjustment(record_id, column, default=None) is not None

    @staticmethod
    def _check_adjustment(adjustment: PhotoAdjustment) -> tuple[float, float, float]:
        zoom, x, y = float(adjustment.zoom), float(adjustment.offset_x), float(adjustment.offset_y)
        if not zoom > 0:
            raise DataError("사진 확대 배율은 0보다 커야 합니다.")
        return zoom, x, y

    def set_photo_adjustment(self, record_id: int, column: str, adjustment: PhotoAdjustment) -> None:
        with self.transaction():
            column_id = self.column(column).id
            zoom, x, y = self._check_adjustment(adjustment)
            if self._db.execute("SELECT 1 FROM records WHERE id = ?", (int(record_id),)).fetchone() is None:
                raise DataError("레코드를 찾을 수 없습니다.")
            self._db.execute(
                "INSERT OR REPLACE INTO photo_adjust (record_id, column_id, zoom, offset_x, offset_y)"
                " VALUES (?, ?, ?, ?, ?)",
                (int(record_id), column_id, zoom, x, y),
            )

    def apply_photo_adjustment_to_all(self, column: str, adjustment: PhotoAdjustment, filter: Any = None) -> int:
        with self.transaction():
            column_id = self.column(column).id
            zoom, x, y = self._check_adjustment(adjustment)
            where, params = self._where(filter)
            cursor = self._db.execute(
                "INSERT OR REPLACE INTO photo_adjust (record_id, column_id, zoom, offset_x, offset_y)"
                f" SELECT id, ?, ?, ?, ? FROM records{where}",
                [column_id, zoom, x, y] + params,
            )
            return cursor.rowcount

    def clear_photo_adjustments(self, column: str, ids: int | Iterable[int] | None = None) -> int:
        with self.transaction():
            column_id = self.column(column).id
            if ids is None:
                return self._db.execute("DELETE FROM photo_adjust WHERE column_id = ?", (column_id,)).rowcount
            removed = 0
            for marks, chunk in self._chunks(ids):
                removed += self._db.execute(
                    f"DELETE FROM photo_adjust WHERE column_id = ? AND record_id IN ({marks})",
                    [column_id] + chunk,
                ).rowcount
            return removed
