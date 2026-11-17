"""다눌이 쓰던 Access 명단(.mdb/.accdb)을 읽는다.

Windows에 들어 있는 Access 데이터 엔진(ACE OLEDB)을 PowerShell의 System.Data.OleDb로 불러 표를 JSON으로 받는다.
파이썬 패키지를 더 넣지 않아도 되고, 원래 파일은 읽기 전용으로 연다. 엔진이 없는 PC에서는 그렇다고 알리고
Access에서 엑셀로 내보내 가져오도록 안내한다.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from .errors import DataError

ACCESS_SUFFIXES = (".mdb", ".accdb")
PROVIDERS = ("Microsoft.ACE.OLEDB.16.0", "Microsoft.ACE.OLEDB.12.0", "Microsoft.Jet.OLEDB.4.0")
TIMEOUT_SECONDS = 60
NO_ENGINE = ("이 컴퓨터에는 Access 파일을 읽는 프로그램(Access 데이터베이스 엔진)이 없습니다. "
             "Access에서 표를 엑셀(.xlsx)로 내보낸 뒤 그 파일을 가져오세요.")

_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$path = $env:HANA_ACCESS_PATH
$table = $env:HANA_ACCESS_TABLE
$conn = $null
foreach ($provider in $env:HANA_ACCESS_PROVIDERS.Split(';')) {
  try {
    $c = New-Object System.Data.OleDb.OleDbConnection("Provider=$provider;Data Source=$path;Mode=Read;Persist Security Info=False")
    $c.Open(); $conn = $c; break
  } catch { $last = $_.Exception.Message }
}
if ($conn -eq $null) { [Console]::Out.Write((@{ error = 'no_engine'; detail = "$last" } | ConvertTo-Json -Compress)); exit 0 }
try {
  $schema = $conn.GetSchema('Tables')
  $tables = @($schema.Rows | Where-Object { $_.TABLE_TYPE -eq 'TABLE' } | ForEach-Object { $_.TABLE_NAME })
  $result = @{ tables = $tables }
  if ($table) {
    $cmd = $conn.CreateCommand()
    $cmd.CommandText = "SELECT * FROM [" + $table.Replace(']', ']]') + "]"
    $reader = $cmd.ExecuteReader()
    $columns = @(); for ($i = 0; $i -lt $reader.FieldCount; $i++) { $columns += $reader.GetName($i) }
    $rows = New-Object System.Collections.ArrayList
    while ($reader.Read()) {
      $row = @(); for ($i = 0; $i -lt $reader.FieldCount; $i++) {
        $v = $reader.GetValue($i)
        if ($v -is [DBNull]) { $row += '' }
        elseif ($v -is [DateTime]) { $row += $v.ToString('yyyy-MM-dd') }
        elseif ($v -is [byte[]]) { $row += '' }
        else { $row += [string]$v }
      }
      [void]$rows.Add($row)
    }
    $reader.Close()
    $result.columns = $columns
    $result.rows = $rows
  }
  [Console]::Out.Write(($result | ConvertTo-Json -Compress -Depth 4))
} finally { $conn.Close() }
"""


def _run(path: str | Path, table: str | None = None) -> dict:
    path = str(Path(path).resolve())
    if not os.path.isfile(path):
        raise DataError(f"파일을 찾을 수 없습니다: {path}")
    env = dict(os.environ, HANA_ACCESS_PATH=path, HANA_ACCESS_TABLE=table or "",
               HANA_ACCESS_PROVIDERS=";".join(PROVIDERS))
    try:
        done = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", _SCRIPT],
            capture_output=True, env=env, timeout=TIMEOUT_SECONDS,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except FileNotFoundError as exc:
        raise DataError(NO_ENGINE) from exc
    except subprocess.TimeoutExpired as exc:
        raise DataError("Access 파일을 읽는 데 너무 오래 걸립니다. 엑셀로 내보내 가져오세요.") from exc
    text = done.stdout.decode("utf-8", "replace").strip()
    if not text:
        message = done.stderr.decode("utf-8", "replace").strip().splitlines()
        raise DataError("Access 파일을 읽지 못했습니다." + (f" ({message[-1]})" if message else ""))
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise DataError("Access 파일을 읽지 못했습니다.") from exc
    if data.get("error") == "no_engine":
        raise DataError(NO_ENGINE)
    return data


def _as_list(value) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def list_tables(path: str | Path) -> list[str]:
    return [str(t) for t in _as_list(_run(path).get("tables"))]


def read_table(path: str | Path, table: str | None = None) -> tuple[str, list[str], list[str], list[list[str]]]:
    """(표 이름, 모든 표 이름, 열 이름, 행들). table이 없으면 첫 표."""
    tables = list_tables(path)
    if not tables:
        raise DataError("Access 파일에 표가 없습니다.")
    if table is None:
        table = tables[0]
    elif table not in tables:
        raise DataError(f"'{table}' 표를 찾을 수 없습니다.")
    data = _run(path, table)
    columns = [str(c) for c in _as_list(data.get("columns"))]
    rows = []
    for row in _as_list(data.get("rows")):
        cells = row if isinstance(row, list) else [row]
        rows.append(["" if c is None else str(c) for c in cells])
    return table, tables, columns, rows
