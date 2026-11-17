"""다눌이 쓰던 Access 명단(.mdb) 가져오기. Access 엔진이 있는 PC에서만 실제 파일로 돈다."""

import subprocess
import sys

import pytest

from data import access
from data.errors import DataError
from data.importers import import_into, preview_file

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows 전용")

_MAKE = r"""
$ErrorActionPreference = 'Stop'
$path = $env:MDB_PATH
$cat = New-Object -ComObject ADOX.Catalog
$cat.Create("Provider=Microsoft.ACE.OLEDB.12.0;Data Source=$path;Jet OLEDB:Engine Type=5") | Out-Null
$c = New-Object System.Data.OleDb.OleDbConnection("Provider=Microsoft.ACE.OLEDB.12.0;Data Source=$path")
$c.Open()
$cmd = $c.CreateCommand()
$cmd.CommandText = "CREATE TABLE [사원] ([이름] TEXT(50), [부서] TEXT(50), [사번] TEXT(20), [입사일] DATETIME, [사진] TEXT(100))"
[void]$cmd.ExecuteNonQuery()
$cmd.CommandText = "INSERT INTO [사원] VALUES ('홍길동', '개발팀', 'A001', #2024-03-02#, 'A001.jpg')"
[void]$cmd.ExecuteNonQuery()
$cmd.CommandText = "INSERT INTO [사원] VALUES ('김하나', '영업팀', 'A002', NULL, '')"
[void]$cmd.ExecuteNonQuery()
$cmd.CommandText = "CREATE TABLE [메모] ([내용] TEXT(50))"
[void]$cmd.ExecuteNonQuery()
$c.Close()
"""


@pytest.fixture
def mdb(tmp_path):
    path = tmp_path / "다눌 명단.mdb"
    done = subprocess.run(["powershell", "-NoProfile", "-Command", _MAKE], capture_output=True,
                          env={**__import__("os").environ, "MDB_PATH": str(path)})
    if done.returncode != 0 or not path.exists():
        pytest.skip("이 PC에는 Access 엔진이 없어 시험용 .mdb를 만들 수 없습니다")
    return path


def test_lists_tables_and_reads_korean_rows(mdb):
    assert set(access.list_tables(mdb)) == {"사원", "메모"}
    table, tables, columns, rows = access.read_table(mdb, "사원")
    assert columns == ["이름", "부서", "사번", "입사일", "사진"]
    assert rows == [["홍길동", "개발팀", "A001", "2024-03-02", "A001.jpg"], ["김하나", "영업팀", "A002", "", ""]]


def test_preview_and_import_into_store(mdb, store):
    preview = preview_file(mdb, sheet="사원")
    assert preview.kind == "access" and preview.sheet == "사원" and set(preview.sheets) == {"사원", "메모"}
    assert [c.name for c in preview.columns] == ["이름", "부서", "사번", "입사일", "사진"]
    result = import_into(store, preview, mode="replace")
    assert result.inserted == 2
    assert {r["이름"] for r in store.rows()} == {"홍길동", "김하나"}


def test_missing_file_is_a_clear_error(tmp_path):
    with pytest.raises(DataError, match="찾을 수 없습니다"):
        access.list_tables(tmp_path / "없음.mdb")


def test_no_engine_message(monkeypatch, tmp_path):
    path = tmp_path / "x.mdb"
    path.write_bytes(b"not an access file")
    monkeypatch.setattr(access, "PROVIDERS", ("Nope.Provider.1",))
    with pytest.raises(DataError, match="Access 데이터베이스 엔진"):
        access.list_tables(path)
