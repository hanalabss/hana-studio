"""
Device identification module for licensing.
Collects hardware identifiers and generates unique device hash.
"""

import subprocess
import hashlib
import winreg
from typing import Optional


def get_mainboard_uuid() -> Optional[str]:
    """
    Mainboard UUID 수집 (Primary 1)
    wmic csproduct get UUID
    """
    try:
        result = subprocess.run(
            ["wmic", "csproduct", "get", "UUID"],
            capture_output=True,
            text=True,
            timeout=10
        )
        lines = result.stdout.strip().split("\n")
        for line in lines:
            line = line.strip()
            # UUID 형식 체크 (빈 값, 헤더, 무효값 제외)
            if line and line != "UUID" and line != "FFFFFFFF-FFFF-FFFF-FFFF-FFFFFFFFFFFF":
                return line
    except Exception:
        pass
    return None


def get_cpu_id() -> Optional[str]:
    """
    CPU ID 수집 (Primary 2)
    wmic cpu get ProcessorId
    """
    try:
        result = subprocess.run(
            ["wmic", "cpu", "get", "ProcessorId"],
            capture_output=True,
            text=True,
            timeout=10
        )
        lines = result.stdout.strip().split("\n")
        for line in lines:
            line = line.strip()
            # 빈 값, 헤더, 무효값 제외
            if line and line != "ProcessorId" and line != "0" * 16:
                return line
    except Exception:
        pass
    return None


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
    except Exception:
        pass
    return None


def get_disk_serial() -> Optional[str]:
    """
    Disk Serial 수집 (Fallback 2)
    wmic diskdrive get SerialNumber (첫 번째 디스크)
    """
    try:
        result = subprocess.run(
            ["wmic", "diskdrive", "get", "SerialNumber"],
            capture_output=True,
            text=True,
            timeout=10
        )
        lines = result.stdout.strip().split("\n")
        for line in lines:
            line = line.strip()
            # 빈 값, 헤더 제외
            if line and line != "SerialNumber":
                return line
    except Exception:
        pass
    return None


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
