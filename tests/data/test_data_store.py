import sqlite3
import threading

import pytest

from data import store as store_module
from data.errors import DataError
from data.store import Column, DataStore, PhotoAdjustment

HOSTILE_NAMES = [
    "select",
    "이름 (한글)",
    '따옴표 "큰" 과 \'작은\'',
    'x"; DROP TABLE records; --',
    "c1",
    "id",
    "printed",
    "100% [대괄호] `백틱`",
    "줄\n바꿈",
]
HOSTILE_VALUE = "Robert'); DROP TABLE records;-- \"quoted\" %_\\"


def test_new_store_is_empty_and_versioned(store):
    assert store.columns() == []
    assert store.count() == 0
    assert store.rows() == []
    assert store.schema_version == len(store_module.MIGRATIONS)


def test_add_columns_with_types_and_options(store):
    store.add_column("이름")
    store.add_column("직급", "list", {"choices": ["사원", "대리"]})
    store.add_column("사진", "photo", {"folder": "C:/사진", "extensions": [".jpg", ".png"]})
    columns = store.columns()
    assert [(c.name, c.type) for c in columns] == [("이름", "text"), ("직급", "list"), ("사진", "photo")]
    assert columns[1].options == {"choices": ["사원", "대리"]}
    assert columns[2].options["extensions"] == [".jpg", ".png"]
    assert isinstance(columns[0], Column)


def test_add_column_rejects_bad_names(store):
    store.add_column("이름")
    for bad in ["", "   ", "_id", "_printed", "_무엇"]:
        with pytest.raises(DataError):
            store.add_column(bad)
    with pytest.raises(DataError, match="이미"):
        store.add_column("이름")
    with pytest.raises(DataError):
        store.add_column("새열", "geometry")
    assert store.column_names() == ["이름"]


def test_record_shape_matches_contract(store):
    store.add_column("이름")
    record_id = store.insert({"이름": "홍길동"})
    record = store.rows()[0]
    assert record == {"_id": record_id, "이름": "홍길동", "_printed": False, "_print_count": 0, "_printed_at": None}
    assert {k: v for k, v in record.items() if not k.startswith("_")} == {"이름": "홍길동"}
    assert store.get(record_id) == record
    assert store.get(9999) is None


def test_hostile_column_names_and_values_are_safe(store):
    for name in HOSTILE_NAMES:
        store.add_column(name)
    assert store.column_names() == HOSTILE_NAMES
    record_id = store.insert({name: HOSTILE_VALUE for name in HOSTILE_NAMES})
    record = store.get(record_id)
    for name in HOSTILE_NAMES:
        assert record[name] == HOSTILE_VALUE
    store.update(record_id, {"select": "바뀜"})
    assert store.get(record_id)["select"] == "바뀜"
    assert store.rows(order=('x"; DROP TABLE records; --', "desc"))[0]["_id"] == record_id
    for name in HOSTILE_NAMES:
        store.rename_column(name, name + " 2")
    assert store.get(record_id)["id 2"] == HOSTILE_VALUE
    for name in HOSTILE_NAMES:
        store.remove_column(name + " 2")
    assert store.count() == 1
    tables = {
        row[0]
        for row in sqlite3.connect(store.path).execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert {"records", "columns", "meta"} <= tables


def test_rename_remove_reorder_preserve_data(people):
    before = people.rows()
    people.rename_column("부서", "소속 부서")
    people.remove_column("사번")
    people.add_column("메모", position=0)
    after = people.rows()
    assert people.column_names() == ["메모", "이름", "소속 부서", "나이", "입사일"]
    for old, new in zip(before, after):
        assert new["_id"] == old["_id"]
        assert new["이름"] == old["이름"]
        assert new["소속 부서"] == old["부서"]
        assert new["나이"] == old["나이"]
        assert "사번" not in new
        assert new["메모"] is None

    people.reorder_columns(["입사일", "이름"])
    assert people.column_names() == ["입사일", "이름", "메모", "소속 부서", "나이"]
    people.move_column("나이", 0)
    assert people.column_names()[0] == "나이"
    assert list(people.rows()[0])[:3] == ["_id", "나이", "입사일"]
    assert [r["이름"] for r in people.rows()] == [r["이름"] for r in before]

    with pytest.raises(DataError):
        people.reorder_columns(["없는 열"])
    with pytest.raises(DataError):
        people.rename_column("이름", "나이")
    with pytest.raises(DataError):
        people.remove_column("없는 열")


def test_removed_column_name_can_be_reused_without_old_data(people):
    people.remove_column("부서")
    people.add_column("부서")
    assert all(r["부서"] is None for r in people.rows())


def test_values_are_normalized_by_type(people):
    rows = people.rows()
    assert rows[0]["사번"] == "00123"
    assert rows[0]["나이"] == 31
    assert [r["입사일"] for r in rows] == ["2020-03-02", "2022-07-01", "2015-01-05", None, "2019-11-11"]
    assert rows[3]["부서"] is None

    people.update(rows[0]["_id"], {"나이": "1,234", "입사일": "미정"})
    changed = people.get(rows[0]["_id"])
    assert changed["나이"] == 1234
    assert changed["입사일"] == "미정"

    people.update(rows[1]["_id"], {"나이": "007"})
    assert people.get(rows[1]["_id"])["나이"] == "007"


def test_insert_update_unknown_column(people):
    with pytest.raises(DataError, match="없는"):
        people.insert({"없는": 1})
    with pytest.raises(DataError):
        people.update(1, {"없는": 1})
    assert people.update(99999, {"이름": "x"}) is False
    record = people.rows()[0]
    record["이름"] = "수정됨"
    assert people.update(record["_id"], record) is True
    assert people.get(record["_id"])["이름"] == "수정됨"
    assert people.get(record["_id"])["_printed"] is False


def test_rows_order_offset_limit(people):
    assert [r["나이"] for r in people.rows(order="나이")] == [27, 31, 38, 45, None]
    assert [r["나이"] for r in people.rows(order=[("나이", "desc")])] == [45, 38, 31, 27, None]
    assert [r["이름"] for r in people.rows(order=[("부서", "asc"), ("이름", "desc")])][:2] == ["이영희", "박민수"]
    page = people.rows(order="_id", offset=1, limit=2)
    assert [r["이름"] for r in page] == ["이영희", "박민수"]
    assert people.rows(offset=10) == []
    assert people.ids(order=("나이", "desc"))[0] == people.rows(order=("나이", "desc"))[0]["_id"]
    with pytest.raises(DataError):
        people.rows(order=("나이", "sideways"))
    with pytest.raises(DataError):
        people.rows(order="없는 열")


def test_delete_and_ids_are_not_reused(people):
    ids = people.ids()
    assert people.delete([ids[0], ids[-1], 424242]) == 2
    assert people.delete(ids[1]) == 1
    assert people.count() == 2
    new_id = people.insert({"이름": "새사람"})
    assert new_id > max(ids)
    assert people.clear() == 3
    assert people.column_names() == ["이름", "사번", "부서", "나이", "입사일"]


def test_print_status_lifecycle(people):
    ids = people.ids()
    assert people.mark_printed([ids[0], ids[1], ids[0]]) == 2
    first = people.get(ids[0])
    assert first["_printed"] is True
    assert first["_print_count"] == 1
    assert first["_printed_at"] is not None and len(first["_printed_at"]) == 19
    assert people.get(ids[2])["_printed"] is False

    people.mark_printed(ids[0])
    assert people.get(ids[0])["_print_count"] == 2

    people.mark_printed([ids[0]], printed=False)
    unmarked = people.get(ids[0])
    assert unmarked["_printed"] is False
    assert unmarked["_print_count"] == 2
    assert unmarked["_printed_at"] is not None

    assert people.reset_print_status([ids[1]]) == 1
    assert people.get(ids[1]) == {**people.get(ids[1]), "_printed": False, "_print_count": 0, "_printed_at": None}
    assert people.get(ids[0])["_print_count"] == 2

    people.mark_printed(ids)
    assert people.reset_print_status() == 5
    assert all(not r["_printed"] and r["_print_count"] == 0 and r["_printed_at"] is None for r in people.rows())


def test_mark_printed_many_ids(store):
    store.add_column("n", "number")
    store.insert_many({"n": i} for i in range(3000))
    ids = store.ids()
    assert store.mark_printed(ids) == 3000
    assert store.count(store_filter_printed()) == 3000


def store_filter_printed():
    from data.filters import printed_only

    return printed_only()


def test_reopen_keeps_everything_and_closes_cleanly(tmp_path):
    path = tmp_path / "하위 폴더" / "명단.hsdata"
    with DataStore(path) as first:
        first.add_column("이름")
        record_id = first.insert({"이름": "홍길동"})
        first.mark_printed(record_id)
    assert not path.with_name(path.name + "-wal").exists()
    with pytest.raises(DataError):
        first.count()
    first.close()

    with DataStore(path) as second:
        assert second.column_names() == ["이름"]
        record = second.get(record_id)
        assert record["이름"] == "홍길동" and record["_print_count"] == 1
        assert second.schema_version == len(store_module.MIGRATIONS)
        mode = sqlite3.connect(path).execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"


def test_migration_hook_runs_once_on_reopen(tmp_path, monkeypatch):
    path = tmp_path / "m.hsdata"
    DataStore(path).close()
    calls = []

    def migrate_v2(conn):
        calls.append(1)
        conn.execute("CREATE TABLE extra (x)")

    monkeypatch.setattr(store_module, "MIGRATIONS", store_module.MIGRATIONS + [migrate_v2])
    with DataStore(path) as upgraded:
        assert upgraded.schema_version == 2
    with DataStore(path) as again:
        assert again.schema_version == 2
    assert calls == [1]


def test_newer_file_and_garbage_file_are_rejected(tmp_path, monkeypatch):
    path = tmp_path / "new.hsdata"
    monkeypatch.setattr(store_module, "MIGRATIONS", store_module.MIGRATIONS + [lambda conn: None])
    DataStore(path).close()
    monkeypatch.undo()
    with pytest.raises(DataError, match="최신 버전"):
        DataStore(path)

    garbage = tmp_path / "garbage.hsdata"
    garbage.write_bytes(b"this is not a database" * 100)
    with pytest.raises(DataError, match="열 수 없습니다"):
        DataStore(garbage)
    assert garbage.read_bytes().startswith(b"this is not")


def test_transaction_rolls_back_on_error(people):
    with pytest.raises(RuntimeError):
        with people.transaction():
            people.add_column("임시")
            people.insert({"이름": "유령"})
            raise RuntimeError("boom")
    assert "임시" not in people.column_names()
    assert people.count() == 5

    def bad_rows():
        yield {"이름": "하나"}
        yield {"없는 열": "둘"}

    with pytest.raises(DataError):
        people.insert_many(bad_rows())
    assert people.count() == 5


def test_insert_many_reports_invalid_values(store):
    store.add_column("나이", "number")
    store.add_column("날짜", "date")
    problems = []
    inserted = store.insert_many(
        [{"나이": "12", "날짜": "2026-01-05"}, {"나이": "열두살", "날짜": "언젠가"}],
        on_invalid=lambda index, column, value: problems.append((index, column, value)),
    )
    assert inserted == 2
    assert problems == [(1, "나이", "열두살"), (1, "날짜", "언젠가")]
    assert store.rows()[1]["나이"] == "열두살"


def test_set_column_type_converts_existing_values(store):
    store.add_column("값")
    store.insert_many([{"값": "1,500"}, {"값": "abc"}, {"값": "0042"}, {"값": None}])
    assert store.set_column_type("값", "number") == 2
    assert [r["값"] for r in store.rows()] == [1500, "abc", "0042", None]
    assert store.column("값").type == "number"
    assert store.set_column_type("값", "text") == 0
    assert [r["값"] for r in store.rows()] == ["1500", "abc", "0042", None]


def test_photo_adjustments_follow_records_and_columns(people):
    people.add_column("사진", "photo")
    ids = people.ids()
    assert people.get_photo_adjustment(ids[0], "사진") == PhotoAdjustment(1.0, 0.0, 0.0)
    people.set_photo_adjustment(ids[0], "사진", PhotoAdjustment(1.5, 0.1, -0.2))
    people.rename_column("사진", "증명사진")
    assert people.get_photo_adjustment(ids[0], "증명사진") == PhotoAdjustment(1.5, 0.1, -0.2)
    with pytest.raises(DataError):
        people.set_photo_adjustment(ids[0], "증명사진", PhotoAdjustment(zoom=0))
    with pytest.raises(DataError):
        people.set_photo_adjustment(987654, "증명사진", PhotoAdjustment())

    assert people.apply_photo_adjustment_to_all("증명사진", PhotoAdjustment(2.0, 0.0, 0.3)) == 5
    assert all(people.get_photo_adjustment(i, "증명사진").zoom == 2.0 for i in ids)
    assert people.clear_photo_adjustments("증명사진", [ids[0]]) == 1
    assert people.get_photo_adjustment(ids[0], "증명사진") == PhotoAdjustment()
    people.delete(ids[1])
    count = sqlite3.connect(people.path).execute("SELECT COUNT(*) FROM photo_adjust").fetchone()[0]
    assert count == 3


def test_usable_from_another_thread(people):
    seen = []
    worker = threading.Thread(target=lambda: seen.append(people.count()))
    worker.start()
    worker.join()
    assert seen == [5]


# ── 사진 조정값: 저장 여부 구별 ───────────────────────────────────────────────
def test_stored_default_adjustment_is_distinguishable_from_nothing_stored(tmp_path):
    with DataStore(tmp_path / "adjust.hsdata") as store:
        store.add_column("사진", "photo")
        first = store.insert({"사진": "a.png"})
        second = store.insert({"사진": "b.png"})

        assert store.get_photo_adjustment(first, "사진") == PhotoAdjustment()
        assert store.get_photo_adjustment(first, "사진", default=None) is None
        assert store.get_photo_adjustment(first, "사진", PhotoAdjustment(2.0)) == PhotoAdjustment(2.0)
        assert store.has_photo_adjustment(first, "사진") is False

        store.set_photo_adjustment(first, "사진", PhotoAdjustment())      # 기본값을 일부러 저장
        assert store.has_photo_adjustment(first, "사진") is True
        assert store.get_photo_adjustment(first, "사진", default=None) == PhotoAdjustment()
        assert store.has_photo_adjustment(second, "사진") is False

        assert store.clear_photo_adjustments("사진", [first]) == 1
        assert store.has_photo_adjustment(first, "사진") is False
        with pytest.raises(DataError):
            store.has_photo_adjustment(first, "없는 열")


def test_offsets_are_stored_as_given_and_only_zoom_is_validated(tmp_path):
    # offset은 틀 크기에 대한 비율(+ = 오른쪽·아래)이고 저장소는 범위를 강제하지 않는다.
    with DataStore(tmp_path / "range.hsdata") as store:
        store.add_column("사진", "photo")
        record = store.insert({"사진": "a.png"})
        store.set_photo_adjustment(record, "사진", PhotoAdjustment(0.25, 1.5, -2.0))
        assert store.get_photo_adjustment(record, "사진") == PhotoAdjustment(0.25, 1.5, -2.0)
        for bad in (0, -1.0, float("nan")):
            with pytest.raises(DataError):
                store.set_photo_adjustment(record, "사진", PhotoAdjustment(zoom=bad))


# ── 스레드 규칙(DataStore 설명서에 적은 보장) ─────────────────────────────────
def test_calls_from_several_threads_are_serialized(tmp_path):
    import threading

    with DataStore(tmp_path / "threads.hsdata") as store:
        store.add_column("이름")
        errors = []

        def writer(tag):
            try:
                for index in range(40):
                    store.insert({"이름": f"{tag}-{index}"})
                    store.count()
            except Exception as exc:          # pragma: no cover - 실패하면 아래에서 드러난다
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(tag,)) for tag in "abcd"]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(30)
        assert errors == [] and store.count() == 160
        assert len({record["이름"] for record in store.rows()}) == 160


def test_other_threads_wait_for_an_open_transaction_and_see_only_the_committed_state(tmp_path):
    import threading

    with DataStore(tmp_path / "tx.hsdata") as store:
        store.add_column("이름")
        inside, seen = threading.Event(), []

        def reader():
            inside.wait(10)
            seen.append(store.count())          # 트랜잭션이 끝날 때까지 기다린다

        thread = threading.Thread(target=reader)
        thread.start()
        with store.transaction():
            store.insert({"이름": "하나"})
            inside.set()
            thread.join(0.2)
            assert thread.is_alive() and seen == [], "다른 스레드는 중간 상태를 읽지 못한다"
            store.insert({"이름": "둘"})
        thread.join(10)
        assert seen == [2]

        snapshot = store.rows()
        snapshot[0]["이름"] = "바꿔도 저장되지 않는다"
        assert store.rows()[0]["이름"] == "하나"
