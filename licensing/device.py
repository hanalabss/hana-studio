"""
Device identification module for licensing.
Collects hardware identifiers and generates unique device hash.

하드웨어 값은 PowerShell CIM(Get-CimInstance)으로 먼저 조회하고,
실패하면 wmic으로 조회한다. Windows 11 24H2부터 wmic이 기본 제거되었기 때문.
두 방식은 같은 WMI 클래스를 읽으므로 값(→ 해시)이 동일하다.
"""

import os
import json
import subprocess
import hashlib
import winreg
from functools import lru_cache
from typing import Iterable, List, Optional

# GUI(windowed) 빌드에서 콘솔 창이 깜빡이지 않도록
_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

_POWERSHELL = os.path.join(
    os.environ.get("SystemRoot", r"C:\Windows"),
    "System32", "WindowsPowerShell", "v1.0", "powershell.exe"
)

# 클래스별 속성을 배열로 받아 Python에서 wmic 방식과 동일하게 필터링
_CIM_SCRIPT = (
    "$ErrorActionPreference='SilentlyContinue';"
    "$r=@{"
    "uuid=@(Get-CimInstance Win32_ComputerSystemProduct | ForEach-Object UUID);"
    "cpu=@(Get-CimInstance Win32_Processor | ForEach-Object ProcessorId);"
    "disk=@(Get-CimInstance Win32_DiskDrive | ForEach-Object SerialNumber)"
    "};"
    "ConvertTo-Json -InputObject $r -Compress"
)

_INVALID_UUID = {"UUID", "FFFFFFFF-FFFF-FFFF-FFFF-FFFFFFFFFFFF"}
_INVALID_CPU = {"ProcessorId", "0" * 16}
_INVALID_DISK = {"SerialNumber"}


def _first_valid(values: Iterable, invalid: set) -> Optional[str]:
    """빈 값·헤더·무효값을 제외한 첫 번째 값"""
    for value in values:
        if value is None:
            continue
        value = str(value).strip()
        if value and value not in invalid:
            return value
    return None


@lru_cache(maxsize=1)
def _query_cim() -> dict:
    """PowerShell CIM으로 UUID / CPU ID / Disk Serial 일괄 조회 (1회만 실행)"""
    try:
        result = subprocess.run(
            [_POWERSHELL, "-NoProfile", "-NonInteractive", "-Command", _CIM_SCRIPT],
            capture_output=True,
            text=True,
            timeout=20,
            creationflags=_CREATE_NO_WINDOW
        )
        data = json.loads(result.stdout.strip() or "{}")
        # 단일 값이 스칼라로 올 수 있으므로 리스트로 정규화
        return {k: v if isinstance(v, list) else [v] for k, v in data.items()}
    except Exception as e:
        print(f"[LICENSE] CIM 조회 실패: {type(e).__name__}: {e}")
        return {}


def _query_wmic(wmi_alias: str, prop: str) -> List[str]:
    """wmic 조회 (CIM 실패 시 대체)"""
    try:
        result = subprocess.run(
            ["wmic", wmi_alias, "get", prop],
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=_CREATE_NO_WINDOW
        )
        return result.stdout.strip().split("\n")
    except Exception as e:
        print(f"[LICENSE] wmic {wmi_alias} 조회 실패: {type(e).__name__}: {e}")
        return []


def get_mainboard_uuid() -> Optional[str]:
    """
    Mainboard UUID 수집 (Primary 1)
    Win32_ComputerSystemProduct.UUID
    """
    return (_first_valid(_query_cim().get("uuid", []), _INVALID_UUID)
            or _first_valid(_query_wmic("csproduct", "UUID"), _INVALID_UUID))


def get_cpu_id() -> Optional[str]:
    """
    CPU ID 수집 (Primary 2)
    Win32_Processor.ProcessorId
    """
    return (_first_valid(_query_cim().get("cpu", []), _INVALID_CPU)
            or _first_valid(_query_wmic("cpu", "ProcessorId"), _INVALID_CPU))


def get_machine_guid() -> Optional[str]:
    """
    Machine GUID 수집 (Fallback 1)
    Registry: HKLM\\SOFTWARE\\Microsoft\\Cryptography\\MachineGuid
    """
    try:
        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\Microsoft\Cryptography",
            0,
            winreg.KEY_READ
        )
        value, _ = winreg.QueryValueEx(key, "MachineGuid")
        winreg.CloseKey(key)
        if value:
            return value
    except Exception as e:
        print(f"[LICENSE] MachineGuid 조회 실패: {type(e).__name__}: {e}")
    return None


def get_disk_serial() -> Optional[str]:
    """
    Disk Serial 수집 (Fallback 2)
    Win32_DiskDrive.SerialNumber (값이 있는 첫 번째 디스크)
    """
    return (_first_valid(_query_cim().get("disk", []), _INVALID_DISK)
            or _first_valid(_query_wmic("diskdrive", "SerialNumber"), _INVALID_DISK))


def get_device_hash() -> str:
    """
    디바이스 고유 해시 생성

    Slot 1: Mainboard UUID → (실패시) Machine GUID
    Slot 2: CPU ID → (실패시) Disk Serial

    Returns:
        SHA256(Slot1 + ":" + Slot2)

    Raises:
        RuntimeError: 모든 식별자 수집 실패시
    """
    # Slot 1: Mainboard UUID or Machine GUID
    slot1 = get_mainboard_uuid()
    if not slot1:
        slot1 = get_machine_guid()

    # Slot 2: CPU ID or Disk Serial
    slot2 = get_cpu_id()
    if not slot2:
        slot2 = get_disk_serial()

    # 둘 다 실패하면 에러
    if not slot1 or not slot2:
        print(f"[LICENSE] 디바이스 식별자 수집 실패 (slot1={'OK' if slot1 else '없음'}, "
              f"slot2={'OK' if slot2 else '없음'})")
        raise RuntimeError("Failed to collect device identifiers")

    # SHA256 해시 생성
    combined = f"{slot1}:{slot2}"
    device_hash = hashlib.sha256(combined.encode()).hexdigest()

    return device_hash


if __name__ == "__main__":
    # 테스트용
    print(f"Mainboard UUID: {get_mainboard_uuid()}")
    print(f"CPU ID: {get_cpu_id()}")
    print(f"Machine GUID: {get_machine_guid()}")
    print(f"Disk Serial: {get_disk_serial()}")
    print(f"Device Hash: {get_device_hash()}")
