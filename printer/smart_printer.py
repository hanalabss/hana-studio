"""
printer/smart_printer.py
IDP SMART 시리즈(SMART-30/50/31/51/81) SmartComm2.dll(x64) ctypes 바인딩.

- 모든 함수는 WINAPI, 문자열은 WCHAR(UTF-16)이다. 마그네틱 ISO 트랙 버퍼만 BYTE다.
- 구조체는 헤더에 #pragma pack이 없으므로 MSVC 기본 정렬(ctypes 기본값과 같음)을 쓴다.
- 인쇄는 드라이버 없이 동작하는 DCL 모드 + SBS(단계별) 제어를 쓴다:
  SmartDCL_OpenDevice2 → SBSStart → CardIn → DrawImage → SmartDCL_Print → DoPrint → CardOut
  → SBSEnd → SmartDCL_CloseDevice
- DLL은 처음 쓸 때 로드한다. 임포트만으로는 아무 것도 로드하지 않는다.
"""

from __future__ import annotations

import ctypes
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .base import PrinterError

DLL_NAME = "SmartComm2.dll"

# ── 상수 (SmartComm2.dll.h) ────────────────────────────────────────────────
SM_SUCCESS = 0x00000000
SM_F_FOUNDNODEV = 0x80000000
SM_F_HOPPEREMPTY = 0x8000000B
SM_F_EMPTYRIBBON = 0x8000001D
SM_F_PRINT = 0x8000001A
SM_F_MAGRW = 0x80000018
MAX_SMART_PRINTER = 32

SMART_DEVICELIST_USB = 0x01
SMART_DEVICELIST_NET = 0x02
SMART_OPENDEVICE_BYID = 0
SMART_OPENDEVICE_BYDESC = 1
DMORIENT_PORTRAIT = 1
DMORIENT_LANDSCAPE = 2

PAGE_FRONT = 0
PAGE_BACK = 1
PANELID_COLOR = 1
PANELID_BLACK = 2
PANELID_OVERLAY = 4
PANELID_UV = 8

CARDPOS_PRINT = 0
CARDPOS_MAGNETIC = 1

MAG_T1, MAG_T2, MAG_T3 = 1, 2, 4
MAG_LOCO = 0x00
MAG_HICO = 0x10
MAG_READ_BUFFER = 1024

GROUP_SMART50 = 50      # SMART-50, SMART-30
GROUP_SMART51 = 51      # SMART-51, SMART-31
GROUP_SMART81 = 81      # SMART-81 (재전사)

SMART_PRINTSIDE_FRONT = 0
PRINTSIDE_BOTH_BY_GROUP = {GROUP_SMART50: 1, GROUP_SMART51: 2, GROUP_SMART81: 2}
DOC_SIZE_BY_GROUP = {GROUP_SMART50: (1012, 636), GROUP_SMART51: (1012, 636),
                     GROUP_SMART81: (1040, 664)}

RI2_RIBBON, RI2_RETRANSFER, RI2_LAMINATE = 0, 1, 2
SYSINFO_VER_1 = 1

RIBBON_NAMES = {
    0: "YMCKO", 1: "YMCKOK", 2: "hYMCKO", 3: "KO", 4: "K", 5: "BO", 6: "B", 7: "BYMCKO",
    8: "YMCKFO", 9: "REWRITABLE", 11: "hYMCKOKO", 12: "YMCKOKR", 14: "YMCK",
    15: "RETRANSFER", 16: "YMCKK", 17: "YMCKF", 18: "YMCKI", 20: "YMCFK",
}
MONO_RIBBONS = {3, 4, 5, 6}
FLUORESCENT_RIBBONS = {8, 17, 20}

# 오류 코드 → (한국어 문구, 사용자가 조치 후 재시도할 수 있는지)
ERROR_TEXT: Dict[int, Tuple[str, bool]] = {
    0xFFFFFFFF: ("프린터 작업에 실패했습니다.", False),
    0x80000000: ("사용할 수 있는 SMART 프린터가 없습니다.", True),
    0x80000001: ("프린터 번호가 범위를 벗어났습니다.", False),
    0x80000002: ("내부 버퍼가 올바르지 않습니다.", False),
    0x80000003: ("프린터가 연결되어 있지 않습니다. 전원과 케이블을 확인해 주세요.", True),
    0x80000004: ("프린터에 전달한 값이 올바르지 않습니다.", False),
    0x80000005: ("프린터를 열지 못했습니다. 다른 프로그램이 사용 중인지 확인해 주세요.", True),
    0x80000006: ("프린터와 통신하지 못했습니다. 케이블 연결을 확인해 주세요.", True),
    0x80000007: ("프린터 드라이버를 찾을 수 없습니다.", False),
    0x80000008: ("프린터 연결이 유효하지 않습니다. 다시 연결해 주세요.", True),
    0x80000009: ("프린터 안에 이미 카드가 있습니다.", True),
    0x8000000A: ("프린터 안에 카드가 없습니다.", True),
    0x8000000B: ("카드 공급함(호퍼)이 비어 있습니다. 카드를 넣어 주세요.", True),
    0x8000000C: ("카드 공급함과 프린터 안 어디에도 카드가 없습니다.", True),
    0x8000000D: ("프린터 응답을 기다리다 시간이 초과되었습니다.", True),
    0x8000000E: ("프린터 덮개가 열려 있습니다. 덮개를 닫아 주세요.", True),
    0x8000000F: ("프린터가 오류 상태입니다. 프린터 화면을 확인해 주세요.", True),
    0x80000010: ("카드를 넣지 못했습니다. 카드 공급함을 확인해 주세요.", True),
    0x80000011: ("카드를 배출하지 못했습니다. 카드 걸림을 확인해 주세요.", True),
    0x80000012: ("카드를 뒤쪽으로 배출하지 못했습니다.", True),
    0x80000013: ("카드를 인코딩 위치로 옮기지 못했습니다.", True),
    0x80000014: ("카드를 IC 위치로 옮기지 못했습니다.", True),
    0x80000015: ("카드를 RF 위치로 옮기지 못했습니다.", True),
    0x80000016: ("카드를 플리퍼로 옮기지 못했습니다.", True),
    0x80000017: ("카드를 플리퍼에서 되돌리지 못했습니다.", True),
    0x80000018: ("마그네틱 읽기/쓰기에 실패했습니다.", True),
    0x80000019: ("프린터가 인쇄 데이터를 받지 못했습니다.", True),
    0x8000001A: ("인쇄에 실패했습니다.", True),
    0x8000001B: ("리본을 찾지 못했습니다. 리본 장착 상태를 확인해 주세요.", True),
    0x8000001C: ("리본을 감지 못했습니다. 리본 장착 상태를 확인해 주세요.", True),
    0x8000001D: ("리본을 모두 사용했습니다. 리본을 교체해 주세요.", True),
    0x8000001E: ("IC 헤드를 올리지 못했습니다.", False),
    0x8000001F: ("IC 헤드를 내리지 못했습니다.", False),
    0x80000020: ("카드를 뒤집지 못했습니다.", True),
    0x80000021: ("카드를 뒤집지 못했습니다.", True),
    0x80000022: ("마그네틱 트랙을 지정하지 않았습니다.", False),
    0x80000023: ("마그네틱 트랙을 하나만 지정해야 합니다.", False),
    0x80000024: ("파일을 찾을 수 없습니다.", False),
    0x80000026: ("이미지 파일을 불러오지 못했습니다.", False),
    0x80000027: ("인쇄 화면(DC)을 만들지 못했습니다.", False),
    0x80000029: ("인쇄 데이터를 만들지 못했습니다.", False),
    0x8000002A: ("프린터를 열지 않은 상태에서 요청했습니다.", False),
    0x8000002B: ("다른 프로그램이 프린터를 사용 중입니다. 잠시 후 다시 시도해 주세요.", True),
    0x8000002C: ("네트워크 연결을 만들지 못했습니다.", True),
    0x8000002D: ("네트워크 프린터에 연결하지 못했습니다.", True),
    0x80000031: ("다른 사용자가 프린터를 예약 중입니다.", True),
    0x80000035: ("네트워크 통신 오류가 발생했습니다.", True),
    0x80000041: ("이 프린터가 아직 지원하지 않는 기능입니다.", False),
    0x80000042: ("내부 버퍼가 부족합니다.", False),
    0x80000206: ("인쇄 헤드가 과열되었습니다. 잠시 후 다시 시도해 주세요.", True),
    0x80000207: ("인쇄 헤드가 장착되어 있지 않습니다.", False),
    0x8000020F: ("프린터 설정을 바꾸지 못했습니다.", False),
    0x80000300: ("네트워크 프린터 인증에 실패했습니다.", False),
    0x8000030D: ("프린터와 인쇄 설정의 해상도가 맞지 않습니다.", False),
    0x8000035D: ("라미네이팅 중 오류가 발생했습니다.", True),
    0x80000380: ("빈(흰색) 이미지입니다.", False),
    0x80000382: ("카드 공급함의 카드가 거의 떨어졌습니다.", True),
    0x80000400: ("카드 공급함에 오류가 있습니다.", True),
    0x80000401: ("프린터에 오류가 있습니다. 프린터 화면을 확인해 주세요.", True),
    0x80000402: ("라미네이터에 오류가 있습니다.", True),
    0x80000403: ("플리퍼에 오류가 있습니다.", True),
}

# 상태 비트 → 한국어 (오류로 볼 것만). 50/51은 64비트 한 개, 81은 [동작·상태, 오류] 두 개.
_STATUS50_ERRORS = {
    0x0000000100000000: "카드를 넣지 못했습니다.",
    0x0000000200000000: "카드를 인쇄 위치로 옮기지 못했습니다.",
    0x0000000400000000: "카드를 배출하지 못했습니다.",
    0x0000000800000000: "카드를 마그네틱 위치로 옮기지 못했습니다.",
    0x0000004000000000: "카드를 플리퍼로 옮기지 못했습니다.",
    0x0000008000000000: "카드를 플리퍼에서 되돌리지 못했습니다.",
    0x0000400000000000: "인쇄에 실패했습니다.",
    0x0000800000000000: "마그네틱 읽기/쓰기에 실패했습니다.",
    0x0001000000000000: "리본을 찾지 못했습니다.",
    0x0002000000000000: "리본 이송 오류입니다.",
    0x0004000000000000: "인쇄 헤드가 장착되어 있지 않습니다.",
    0x0008000000000000: "인쇄 헤드가 과열되었습니다.",
    0x0010000000000000: "리본이 없습니다. 리본을 교체해 주세요.",
    0x0020000000000000: "인쇄 데이터 오류입니다.",
    0x1000000000000000: "프린터가 잠겨 있습니다.",
    0x2000000000000000: "프린터 스풀이 가득 찼습니다.",
}
_STATUS51_ERRORS = {
    0x0000000100000000: "카드를 넣지 못했습니다.",
    0x0000000200000000: "카드를 옮기지 못했습니다.",
    0x0000000400000000: "카드를 외부 장치로 옮기지 못했습니다.",
    0x0000000800000000: "카드를 배출하지 못했습니다.",
    0x0000001000000000: "인쇄 헤드를 움직이지 못했습니다.",
    0x0000004000000000: "리본을 찾지 못했습니다.",
    0x0000008000000000: "리본을 감지 못했습니다.",
    0x0000010000000000: "마그네틱 읽기/쓰기에 실패했습니다.",
    0x0000100000000000: "인쇄 중 오류가 발생했습니다.",
    0x0000200000000000: "프린터 초기화 오류입니다.",
    0x0000400000000000: "외부 장치 연결 오류입니다.",
    0x0000800000000000: "라미네이터 연결 오류입니다.",
    0x0001000000000000: "플리퍼 연결 오류입니다.",
    0x0020000000000000: "리본을 모두 사용했습니다. 리본을 교체해 주세요.",
    0x0040000000000000: "리본이 없습니다. 리본을 장착해 주세요.",
    0x0080000000000000: "인쇄 헤드가 장착되어 있지 않습니다.",
    0x0100000000000000: "인쇄 헤드가 과열되었습니다.",
    0x0200000000000000: "인쇄 데이터 형식이 올바르지 않습니다.",
    0x8000000000000000: "프린터 스풀이 가득 찼습니다.",
}
_STATUS81_ERRORS = {
    0x0000000000000001: "프린터 초기화 오류입니다.",
    0x0000000000000002: "카드를 넣지 못했습니다.",
    0x0000000000000004: "카드를 옮기지 못했습니다.",
    0x0000000000000008: "카드를 외부 장치로 옮기지 못했습니다.",
    0x0000000000000010: "카드를 배출하지 못했습니다.",
    0x0000000000000020: "카드를 뒤집지 못했습니다.",
    0x0000000000000040: "카드를 플리퍼로 보내지 못했습니다.",
    0x0000000000000100: "인쇄에 실패했습니다.",
    0x0000000000000400: "리본을 찾지 못했습니다.",
    0x0000000000000800: "리본을 감지 못했습니다.",
    0x0000000000002000: "필름을 찾지 못했습니다.",
    0x0000000000004000: "필름을 감지 못했습니다.",
    0x0000000000010000: "인쇄 헤드를 움직이지 못했습니다.",
    0x0000000000040000: "재전사 히터가 과열되었습니다.",
    0x0000000000080000: "벤딩 히터가 과열되었습니다.",
    0x0000000000100000: "인쇄 헤드가 과열되었습니다.",
    0x0000000002000000: "마그네틱 읽기/쓰기에 실패했습니다.",
    0x0000000040000000: "외부 장치 연결 오류입니다.",
    0x0000000100000000: "리본이 없습니다. 리본을 장착해 주세요.",
    0x0000000200000000: "리본을 모두 사용했습니다. 리본을 교체해 주세요.",
    0x0000000400000000: "필름이 없습니다. 필름을 장착해 주세요.",
    0x0000000800000000: "필름을 모두 사용했습니다. 필름을 교체해 주세요.",
    0x0000001000000000: "인쇄 헤드가 장착되어 있지 않습니다.",
    0x0000002000000000: "재전사 히터가 장착되어 있지 않습니다.",
    0x0000010000000000: "재전사 히터 오류입니다.",
    0x0001000000000000: "인쇄 데이터를 받는 중 오류가 발생했습니다.",
    0x0002000000000000: "프린터 스풀이 가득 찼습니다.",
}

# 그룹별 상태 비트: (플리퍼 장착, 덮개 열림, 공급함 비어 있음, 마그네틱 장착, 청소 필요, 동작 중)
_S50_HOPPER_HAS_CARD = 0x0000000000040000
_S50_FLIPPER = 0x0000000000800000 | 0x0000000000400000
_S50_COVER_OPEN = 0x0000000010000000
_S50_BUSY = 0x000000002503FFFF
_S51_FLIPPER = 0x0000000000008000 | 0x0000000000004000
_S51_COVER_OPEN = 0x0000000000020000
_S51_CARD_EMPTY = 0x0000000000100000
_S51_MAGNETIC = 0x0000000000002000
_S51_NEED_CLEAN = 0x0000000004000000
_S51_BUSY = 0x00000000200007FF
_S81_SINGLE_ONLY = 0x0000000010000000
_S81_COVER_OPEN = 0x0008000000000000
_S81_HOP_EMPTY = 0x0000400000000000
_S81_MAGNETIC = 0x0000000400000000
_S81_NEED_CLEAN = 0x8000000000000000
_S81_BUSY = 0x0800000003416D7F


# ── 구조체 ─────────────────────────────────────────────────────────────────
class SMART_PRINTER_ITEM(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_wchar * 128),
        ("id", ctypes.c_wchar * 64),
        ("dev", ctypes.c_wchar * 64),
        ("desc", ctypes.c_wchar * 256),
        ("pid", ctypes.c_int),
    ]


class SMART_PRINTER_LIST(ctypes.Structure):
    _fields_ = [
        ("n", ctypes.c_int),
        ("item", SMART_PRINTER_ITEM * MAX_SMART_PRINTER),
    ]


class _SYSINFO_PRINTER(ctypes.Structure):
    _fields_ = [
        ("ver", ctypes.c_wchar * 32),
        ("serial", ctypes.c_wchar * 24),
        ("hserial", ctypes.c_wchar * 24),
        ("cleaningWarning", ctypes.c_int),
        ("cleaning", ctypes.c_int),
        ("totalDensity", ctypes.c_int),
        ("colorDensity", ctypes.c_int),
        ("blackDensity", ctypes.c_int),
        ("overlayDensity", ctypes.c_int),
        ("reserved", ctypes.c_ubyte * 8),
    ]


class _SYSINFO_LAMINATOR(ctypes.Structure):
    _fields_ = [
        ("installed", ctypes.c_int),
        ("ver", ctypes.c_wchar * 32),
        ("serial", ctypes.c_wchar * 24),
        ("reserved", ctypes.c_ubyte * 32),
    ]


class SMART_SYSINFO(ctypes.Structure):
    _fields_ = [
        ("ver", ctypes.c_int),
        ("printer", _SYSINFO_PRINTER),
        ("laminator", _SYSINFO_LAMINATOR),
    ]


_H = ctypes.c_void_p
_INT = ctypes.c_int
_PINT = ctypes.POINTER(ctypes.c_int)
_PBYTE = ctypes.POINTER(ctypes.c_ubyte)
_WSTR = ctypes.c_wchar_p

# 필수 함수. 하나라도 없으면 이 DLL은 쓸 수 없다.
_REQUIRED_SIGNATURES: Dict[str, list] = {
    "SmartComm_GetDeviceList2": [ctypes.POINTER(SMART_PRINTER_LIST)],
    "SmartDCL_OpenDevice2": [ctypes.POINTER(_H), _WSTR, _INT, _INT],
    "SmartDCL_CloseDevice": [_H],
    "SmartComm_GetStatus": [_H, ctypes.POINTER(ctypes.c_int64)],
    "SmartComm_SBSStart": [_H],
    "SmartComm_SBSEnd": [_H],
    "SmartComm_CardIn": [_H],
    "SmartComm_CardOut": [_H],
    "SmartComm_Move": [_H, _INT],
    "SmartComm_DoPrint": [_H],
    "SmartComm_DrawImage": [_H, ctypes.c_ubyte, ctypes.c_ubyte, _INT, _INT, _INT, _INT,
                            _WSTR, ctypes.c_void_p],
    "SmartDCL_Print": [_H, _INT],
    "SmartComm_MagSetAllBuffer": [_H, _INT, _PBYTE, _INT, _INT, _PBYTE, _INT, _INT, _PBYTE, _INT],
    "SmartComm_MagWriteAction": [_H, _INT, _INT],
    "SmartComm_MagReadAction": [_H, _INT],
    "SmartComm_MagGetAllBuffer": [_H, _INT, _PBYTE, _PINT, _INT, _PBYTE, _PINT,
                                  _INT, _PBYTE, _PINT],
}
# 있으면 쓰는 함수 (DLL 버전에 따라 없을 수 있다).
_OPTIONAL_SIGNATURES: Dict[str, list] = {
    "SmartCommEx_GetDeviceList2": [ctypes.POINTER(SMART_PRINTER_LIST), _INT],
    "SmartComm_GetRibbonInfo": [_H, _PINT, _PINT, _PINT, _PINT],
    "SmartComm_GetRibbonInfo2": [_H, _INT, _PINT, _PINT, _PINT, _PINT],
    "SmartComm_GetSystemInfo": [_H, ctypes.POINTER(SMART_SYSINFO)],
    "SmartComm_ClearStatus": [_H],
    "SmartComm_CardOutBack": [_H],
}


@dataclass
class SmartDeviceInfo:
    name: str
    id: str
    dev: str
    desc: str
    pid: int
    connection: str = ""            # "usb" | "net" | "" (구분하지 못함)

    @property
    def group(self) -> int:
        return group_from_pid(self.pid)


@dataclass
class SmartStatus:
    raw: Tuple[int, int]
    group: int
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    has_flipper: bool = False
    has_magnetic: Optional[bool] = None
    cover_open: bool = False
    hopper_empty: bool = False
    busy: bool = False

    @property
    def has_error(self) -> bool:
        return bool(self.errors)


@dataclass
class SmartRibbonInfo:
    type: int
    max: int
    remain: int
    grade: int

    @property
    def type_name(self) -> str:
        return RIBBON_NAMES.get(self.type, f"알 수 없음({self.type})")


def group_from_pid(pid: int) -> int:
    """USB 제품 ID로 기종 그룹을 가른다 (SDK 샘플의 GetDeviceGroup과 같은 규칙)."""
    key = (pid >> 4) & 0xFFF
    if key in (0x385, 0x386, 0x387, 0x388):
        return GROUP_SMART50
    if key == 0x381:
        return GROUP_SMART81
    return GROUP_SMART51


def error_text(code: int) -> Tuple[str, bool]:
    """SDK 반환 코드 → (한국어 문구, 재시도 가능 여부)."""
    code &= 0xFFFFFFFF
    if code in ERROR_TEXT:
        return ERROR_TEXT[code]
    if 0x80100000 <= code <= 0x801000FF:
        return ("스마트카드 리더 오류가 발생했습니다.", True)
    return ("프린터에서 알 수 없는 오류가 발생했습니다.", False)


def decode_status(group: int, raw: Tuple[int, int]) -> SmartStatus:
    """SmartComm_GetStatus 값을 기종 그룹에 맞게 풀어 쓴다."""
    ms = raw[0] & 0xFFFFFFFFFFFFFFFF
    ef = raw[1] & 0xFFFFFFFFFFFFFFFF
    status = SmartStatus(raw=(ms, ef), group=group)
    if group == GROUP_SMART81:
        table, error_word = _STATUS81_ERRORS, ef
        status.has_flipper = not (ms & _S81_SINGLE_ONLY)
        status.has_magnetic = bool(ms & _S81_MAGNETIC)
        status.cover_open = bool(ms & _S81_COVER_OPEN)
        status.hopper_empty = bool(ms & _S81_HOP_EMPTY)
        status.busy = bool(ms & _S81_BUSY)
        need_clean = bool(ms & _S81_NEED_CLEAN)
    elif group == GROUP_SMART50:
        table, error_word = _STATUS50_ERRORS, ms
        status.has_flipper = bool(ms & _S50_FLIPPER)
        status.cover_open = bool(ms & _S50_COVER_OPEN)
        status.hopper_empty = not (ms & _S50_HOPPER_HAS_CARD)
        status.busy = bool(ms & _S50_BUSY)
        need_clean = False
    else:
        table, error_word = _STATUS51_ERRORS, ms
        status.has_flipper = bool(ms & _S51_FLIPPER)
        status.has_magnetic = bool(ms & _S51_MAGNETIC)
        status.cover_open = bool(ms & _S51_COVER_OPEN)
        status.hopper_empty = bool(ms & _S51_CARD_EMPTY)
        status.busy = bool(ms & _S51_BUSY)
        need_clean = bool(ms & _S51_NEED_CLEAN)
    status.errors = [text for bit, text in table.items() if error_word & bit]
    if status.cover_open:
        status.errors.append("프린터 덮개가 열려 있습니다. 덮개를 닫아 주세요.")
    if status.hopper_empty:
        status.warnings.append("카드 공급함(호퍼)이 비어 있습니다.")
    if need_clean:
        status.warnings.append("프린터 청소가 필요합니다.")
    return status


def default_dll_candidates() -> List[str]:
    """SmartComm2.dll을 찾을 위치 목록 (앞에 있을수록 먼저)."""
    candidates: List[str] = []
    try:
        from config import config
        configured = config.get("printer.smart_dll_path", "")
    except Exception:
        configured = ""
    if getattr(sys, "frozen", False):
        base = os.path.dirname(sys.executable)
        roots = [os.path.join(base, "_internal"), base]
    else:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        roots = [base]
    if configured:
        candidates.append(configured if os.path.isabs(configured)
                          else os.path.join(base, configured))
    for root in roots:
        candidates.append(os.path.join(root, "dll", "smart", DLL_NAME))
    return candidates


def find_smart_dll() -> Optional[str]:
    for path in default_dll_candidates():
        if os.path.isfile(path):
            return path
    return None


class SmartPrinter:
    """SmartComm2.dll 얇은 래퍼. 장치 핸들 하나를 가진다.

    lib에 이미 로드된(또는 가짜) 라이브러리 객체를 넘기면 DLL을 로드하지 않는다.
    """

    def __init__(self, dll_path: Optional[str] = None, lib: Any = None):
        self._dll_path = dll_path
        self._lib = lib
        self._handle: Optional[ctypes.c_void_p] = None
        self._optional: Dict[str, bool] = {}
        self.last_list_errors: List[str] = []
        if lib is not None:
            self._configure(lib)

    # -- 로드 -------------------------------------------------------------
    @property
    def lib(self) -> Any:
        if self._lib is None:
            self._lib = self._load()
            self._configure(self._lib)
        return self._lib

    def _load(self) -> Any:
        path = self._dll_path or find_smart_dll()
        if not path or not os.path.isfile(path):
            raise PrinterError("SMART 프린터 라이브러리(SmartComm2.dll)를 찾을 수 없습니다.",
                               detail=f"찾은 위치: {default_dll_candidates()}")
        loader = getattr(ctypes, "WinDLL", None)
        if loader is None:
            raise PrinterError("SMART 프린터는 Windows에서만 사용할 수 있습니다.")
        try:
            lib = loader(os.path.abspath(path))
        except OSError as exc:
            raise PrinterError("SMART 프린터 라이브러리를 불러오지 못했습니다.",
                               detail=f"{path}: {exc}") from exc
        self._dll_path = path
        return lib

    def _configure(self, lib: Any) -> None:
        for name, argtypes in _REQUIRED_SIGNATURES.items():
            try:
                func = getattr(lib, name)
            except AttributeError:
                raise PrinterError("SMART 프린터 라이브러리 버전이 맞지 않습니다.",
                                   detail=f"{name} 함수가 없습니다.") from None
            func.argtypes = argtypes
            func.restype = ctypes.c_uint
        for name, argtypes in _OPTIONAL_SIGNATURES.items():
            try:
                func = getattr(lib, name)
            except AttributeError:
                self._optional[name] = False
                continue
            func.argtypes = argtypes
            func.restype = ctypes.c_uint
            self._optional[name] = True

    def has_function(self, name: str) -> bool:
        _ = self.lib
        return self._optional.get(name, name in _REQUIRED_SIGNATURES)

    @property
    def dll_path(self) -> Optional[str]:
        return self._dll_path

    @property
    def is_open(self) -> bool:
        return self._handle is not None

    def _h(self) -> ctypes.c_void_p:
        if self._handle is None:
            raise PrinterError("SMART 프린터가 열려 있지 않습니다.", recoverable=True)
        return self._handle

    @staticmethod
    def _check(result: int, operation: str) -> None:
        if result != SM_SUCCESS:
            text, recoverable = error_text(result)
            raise PrinterError(f"{operation} 실패: {text}", code=result & 0xFFFFFFFF,
                               recoverable=recoverable)

    # -- 장치 목록 · 열기 ---------------------------------------------------
    def list_devices(self) -> List[SmartDeviceInfo]:
        """연결된 프린터 목록. 가능하면 USB와 네트워크를 따로 조회해 연결 방식을 붙인다."""
        lib = self.lib
        self.last_list_errors = []
        if self._optional.get("SmartCommEx_GetDeviceList2"):
            found: List[SmartDeviceInfo] = []
            for option, connection in ((SMART_DEVICELIST_USB, "usb"), (SMART_DEVICELIST_NET, "net")):
                listing = SMART_PRINTER_LIST()
                result = lib.SmartCommEx_GetDeviceList2(ctypes.byref(listing), option)
                if result != SM_SUCCESS:
                    if result != SM_F_FOUNDNODEV:
                        self.last_list_errors.append(
                            f"{connection} 목록 조회 실패: {error_text(result)[0]} "
                            f"(코드 0x{result & 0xFFFFFFFF:08X})")
                    continue
                found.extend(self._items(listing, connection))
            return found
        listing = SMART_PRINTER_LIST()
        result = lib.SmartComm_GetDeviceList2(ctypes.byref(listing))
        if result == SM_F_FOUNDNODEV:
            return []
        self._check(result, "프린터 목록 조회")
        return self._items(listing, "")

    @staticmethod
    def _items(listing: SMART_PRINTER_LIST, connection: str) -> List[SmartDeviceInfo]:
        count = max(0, min(listing.n, MAX_SMART_PRINTER))
        return [SmartDeviceInfo(name=item.name, id=item.id, dev=item.dev, desc=item.desc,
                                pid=item.pid, connection=connection)
                for item in listing.item[:count]]

    def open(self, device: str, by_description: bool = True,
             orientation: int = DMORIENT_LANDSCAPE) -> None:
        """SmartDCL_OpenDevice2: 드라이버 없이(DCL 모드) 프린터를 연다."""
        if self._handle is not None:
            self.close()
        handle = ctypes.c_void_p()
        result = self.lib.SmartDCL_OpenDevice2(
            ctypes.byref(handle), device,
            SMART_OPENDEVICE_BYDESC if by_description else SMART_OPENDEVICE_BYID, orientation)
        self._check(result, "프린터 연결")
        if not handle.value:
            raise PrinterError("프린터 연결 실패: 핸들을 받지 못했습니다.", recoverable=True)
        self._handle = handle

    def close(self) -> None:
        """SmartDCL_CloseDevice. 열려 있지 않으면 아무 일도 하지 않는다."""
        if self._handle is None:
            return
        handle, self._handle = self._handle, None
        self._check(self.lib.SmartDCL_CloseDevice(handle), "프린터 연결 종료")

    # -- 상태 · 정보 --------------------------------------------------------
    def get_status(self) -> Tuple[int, int]:
        """SmartComm_GetStatus 원본 값. SMART-81은 두 개(동작·상태, 오류)를 채운다."""
        words = (ctypes.c_int64 * 2)()
        self._check(self.lib.SmartComm_GetStatus(self._h(), words), "프린터 상태 조회")
        return words[0], words[1]

    def get_ribbon_info(self) -> SmartRibbonInfo:
        """SmartComm_GetRibbonInfo: 리본 종류·최대·잔량·등급."""
        return self._ribbon("SmartComm_GetRibbonInfo", None)

    def get_ribbon_info2(self, device: int = RI2_RIBBON) -> SmartRibbonInfo:
        """SmartComm_GetRibbonInfo2: 리본(0)·재전사 필름(1)·라미네이트 필름(2) 정보."""
        return self._ribbon("SmartComm_GetRibbonInfo2", device)

    def _ribbon(self, name: str, device: Optional[int]) -> SmartRibbonInfo:
        handle = self._h()
        if not self.has_function(name):
            raise PrinterError("이 라이브러리 버전은 리본 정보 조회를 지원하지 않습니다.")
        values = [ctypes.c_int() for _ in range(4)]
        refs = [ctypes.byref(v) for v in values]
        func = getattr(self.lib, name)
        result = func(handle, *refs) if device is None else func(handle, device, *refs)
        self._check(result, "리본 정보 조회")
        return SmartRibbonInfo(*(v.value for v in values))

    def get_system_info(self) -> Dict[str, Any]:
        """SmartComm_GetSystemInfo: 펌웨어 버전, 시리얼, 헤드 시리얼, 청소 횟수 등."""
        handle = self._h()
        if not self.has_function("SmartComm_GetSystemInfo"):
            raise PrinterError("이 라이브러리 버전은 시스템 정보 조회를 지원하지 않습니다.")
        info = SMART_SYSINFO()
        info.ver = SYSINFO_VER_1
        self._check(self.lib.SmartComm_GetSystemInfo(handle, ctypes.byref(info)), "프린터 정보 조회")
        return {
            "firmware": info.printer.ver,
            "serial": info.printer.serial,
            "head_serial": info.printer.hserial,
            "cleaning_count": info.printer.cleaning,
            "laminator_installed": bool(info.laminator.installed),
            "laminator_firmware": info.laminator.ver,
        }

    def clear_status(self) -> None:
        """SmartComm_ClearStatus: 프린터에 남은 오류 상태를 지운다."""
        handle = self._h()
        if self.has_function("SmartComm_ClearStatus"):
            self._check(self.lib.SmartComm_ClearStatus(handle), "오류 상태 초기화")

    # -- 단계별(SBS) 제어 ----------------------------------------------------
    def sbs_start(self) -> None:
        self._check(self.lib.SmartComm_SBSStart(self._h()), "단계별 제어 시작")

    def sbs_end(self) -> None:
        self._check(self.lib.SmartComm_SBSEnd(self._h()), "단계별 제어 종료")

    def card_in(self) -> None:
        """SmartComm_CardIn: 공급함에서 카드 한 장을 넣는다."""
        self._check(self.lib.SmartComm_CardIn(self._h()), "카드 투입")

    def card_out(self) -> None:
        """SmartComm_CardOut: 안에 있는 카드를 배출한다."""
        self._check(self.lib.SmartComm_CardOut(self._h()), "카드 배출")

    def move(self, position: int) -> None:
        """SmartComm_Move: 카드를 인쇄(0)·마그네틱(1) 등 지정 위치로 옮긴다."""
        self._check(self.lib.SmartComm_Move(self._h(), position), "카드 이동")

    # -- 인쇄 -------------------------------------------------------------
    def draw_image(self, page: int, panel: int, x: int, y: int, width: int, height: int,
                   image_path: str) -> None:
        """SmartComm_DrawImage: 이미지 파일을 지정 면·패널에 올린다. 좌표·크기는 픽셀(300dpi)."""
        result = self.lib.SmartComm_DrawImage(self._h(), page, panel, int(x), int(y),
                                              int(width), int(height),
                                              os.path.abspath(image_path), None)
        self._check(result, "이미지 배치")

    def dcl_print(self, print_side: int) -> None:
        """SmartDCL_Print: 그려 둔 내용을 프린터로 보낸다. SBS 모드에서는 DoPrint가 필요하다."""
        self._check(self.lib.SmartDCL_Print(self._h(), print_side), "인쇄 데이터 전송")

    def do_print(self) -> None:
        """SmartComm_DoPrint: 전송된 데이터를 실제로 인쇄한다."""
        self._check(self.lib.SmartComm_DoPrint(self._h()), "인쇄")

    # -- 마그네틱 ---------------------------------------------------------
    def mag_set_all(self, track1: Optional[bytes], track2: Optional[bytes],
                    track3: Optional[bytes]) -> None:
        """SmartComm_MagSetAllBuffer: 쓸 트랙 데이터를 넣는다. 비울 트랙은 None."""
        args: list = [self._h()]
        for data in (track1, track2, track3):
            data = data or b""
            buffer = (ctypes.c_ubyte * max(len(data), 1)).from_buffer_copy(data or b"\0")
            args.extend((1 if data else 0, buffer, len(data)))
        self._check(self.lib.SmartComm_MagSetAllBuffer(*args), "마그네틱 데이터 설정")

    def mag_write(self, track_mask: int, coercivity: int = MAG_HICO) -> None:
        """SmartComm_MagWriteAction: 버퍼의 데이터를 카드에 쓴다."""
        self._check(self.lib.SmartComm_MagWriteAction(self._h(), track_mask, coercivity),
                    "마그네틱 쓰기")

    def mag_read(self, track_mask: int = MAG_T1 | MAG_T2 | MAG_T3) -> None:
        """SmartComm_MagReadAction: 카드의 마그네틱을 읽어 버퍼에 담는다."""
        self._check(self.lib.SmartComm_MagReadAction(self._h(), track_mask), "마그네틱 읽기")

    def mag_get_all(self) -> Tuple[bytes, bytes, bytes]:
        """SmartComm_MagGetAllBuffer: 읽은 세 트랙을 바이트로 돌려준다."""
        buffers = [(ctypes.c_ubyte * MAG_READ_BUFFER)() for _ in range(3)]
        lengths = [ctypes.c_int(MAG_READ_BUFFER - 1) for _ in range(3)]
        args: list = [self._h()]
        for buffer, length in zip(buffers, lengths):
            args.extend((1, buffer, ctypes.byref(length)))
        self._check(self.lib.SmartComm_MagGetAllBuffer(*args), "마그네틱 데이터 조회")
        return tuple(bytes(buffer[:max(0, min(length.value, MAG_READ_BUFFER))]).rstrip(b"\0")
                     for buffer, length in zip(buffers, lengths))
