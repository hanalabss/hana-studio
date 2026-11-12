"""
printer/fakes.py
실제 DLL 없이 돌아가는 가짜 SDK 라이브러리.

ctypes 라이브러리 객체 자리에 그대로 끼워 넣는다:
    R600Printer.from_library(FakeR600Library())
    RtaiAdapter(lib=FakeR600Library())
    SmartAdapter(lib=FakeSmartLibrary())
호출된 함수 이름과 인자를 calls에 차례로 기록하고, 카드 없음·리본 소진·작업 중 오류를 흉내 낸다.
"""

from __future__ import annotations

import ctypes
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

from . import smart_printer as sp

OUT = "<out>"       # 포인터·출력 버퍼 자리에 기록되는 표시


class RecordedCall(NamedTuple):
    name: str
    args: tuple


def _normalize(arg: Any) -> Any:
    if arg is None or isinstance(arg, (bool, int, float, str, bytes)):
        return arg
    if isinstance(arg, (ctypes.c_char_p, ctypes.c_wchar_p)):
        return arg.value
    if isinstance(arg, ctypes.Array):
        if arg._type_ in (ctypes.c_char, ctypes.c_ubyte):
            return bytes(arg).rstrip(b"\0")
        return OUT
    if isinstance(arg, ctypes._SimpleCData):
        return arg.value
    return OUT


def _target(arg: Any) -> Any:
    inner = getattr(arg, "_obj", None)
    if inner is not None:
        return inner
    if hasattr(arg, "contents"):
        return arg.contents
    return arg


def _set(arg: Any, value: Any) -> None:
    if arg is not None:
        _target(arg).value = value


def _fill(buffer: Any, data: bytes) -> None:
    if buffer is None:
        return
    size = ctypes.sizeof(buffer)
    data = data[:max(size - 1, 0)]
    ctypes.memmove(buffer, data + b"\0", len(data) + 1)


class _FakeFunction:
    def __init__(self, library: "FakeLibrary", name: str):
        self._library = library
        self._name = name
        self.argtypes = None
        self.restype = None

    def __call__(self, *args: Any) -> int:
        return self._library._invoke(self._name, args)


class FakeLibrary:
    """가짜 라이브러리 공통부: 호출 기록과 실패 주입."""

    def __init__(self) -> None:
        self.calls: List[RecordedCall] = []
        self.missing: set = set()
        self._failures: Dict[str, List[List[Any]]] = {}
        self._counts: Dict[str, int] = {}

    def __getattr__(self, name: str) -> _FakeFunction:
        if name.startswith("_") or name in self.__dict__.get("missing", ()):
            raise AttributeError(name)
        function = _FakeFunction(self, name)
        self.__dict__[name] = function
        return function

    def fail(self, name: str, code: int, on_call: Optional[int] = None,
             times: Optional[int] = 1) -> None:
        """name 함수가 code를 돌려주게 한다.

        on_call: 몇 번째 호출에서 실패할지(1부터). None이면 다음 호출부터.
        times: 실패 횟수. None이면 계속 실패.
        """
        self._failures.setdefault(name, []).append([on_call, times, code])

    def clear_failures(self) -> None:
        self._failures.clear()

    def reset_calls(self) -> None:
        self.calls.clear()

    def call_names(self) -> List[str]:
        return [call.name for call in self.calls]

    def calls_named(self, name: str) -> List[RecordedCall]:
        return [call for call in self.calls if call.name == name]

    def _injected_failure(self, name: str, count: int) -> Optional[int]:
        for rule in self._failures.get(name, []):
            on_call, times, code = rule
            if on_call is not None and count != on_call:
                continue
            if times is not None:
                if times <= 0:
                    continue
                rule[1] = times - 1
            return code
        return None

    def _invoke(self, name: str, args: Sequence[Any]) -> int:
        count = self._counts.get(name, 0) + 1
        self._counts[name] = count
        self.calls.append(RecordedCall(name, tuple(_normalize(arg) for arg in args)))
        failure = self._injected_failure(name, count)
        if failure is not None:
            return failure
        handler = getattr(self, "_h_" + name, None)
        return handler(*args) if handler is not None else 0


# ── RTAI / R600 ────────────────────────────────────────────────────────────
R600_ERR_FEEDER_EMPTY = 0x00808002
R600_ERR_RIBBON_OUT = 0x00808010
R600_ERR_PRINT = 0x00808020
# 실기(LUCA-40KM)에서 읽은 상태 값
R600_STATUS_READY = 1004        # 주 상태: Ready
R600_STATUS_NO_ERROR = 3001     # 오류 상태: No Error
R600_STATUS_HEATING = 2002      # 경고 상태: Heating
R600_ERR_SELECT_WITHOUT_ENUM = 8404992   # 0x804000: 목록 조회 없이 R600SelectPrt를 부르면 실기가 돌려주는 값
R600_ERR_NO_PRINTER = 0x00800001


class FakeR600Library(FakeLibrary):
    """libDSRetransfer600App.dll 흉내.

    feeder_empty, ribbon_out 같은 속성을 바꾸거나 fail()로 특정 호출을 실패시킨다.
    """

    def __init__(self, usb_printers: Sequence[str] = ("RTAI R600 FAKE",),
                 tcp_printers: Sequence[str] = ()) -> None:
        super().__init__()
        self.usb_printers = list(usb_printers)
        self.tcp_printers = list(tcp_printers)
        self.selected: Optional[str] = None
        self.feeder_empty = False
        self.ribbon_out = False
        self.card_inside = False
        self.ribbon_remaining = 250
        self.film_remaining = 480
        self.ribbon_type = 1
        self.film_type = 1
        self.error_status = 0
        self.warning_status = 0
        self.printed_cards = 0
        self.canvas_count = 0
        self.mag_tracks: List[bytes] = [b"", b"", b""]
        self.mag_read_override: Optional[Tuple[bytes, bytes, bytes]] = None
        self.status_messages: Dict[int, str] = {
            R600_STATUS_READY: "Ready", R600_STATUS_NO_ERROR: "No Error", R600_STATUS_HEATING: "Heating",
        }
        self.error_messages: Dict[int, str] = {
            R600_ERR_FEEDER_EMPTY: "Card feeder is empty",
            R600_ERR_RIBBON_OUT: "Ribbon is used up",
            R600_ERR_PRINT: "Print failed",
            R600_ERR_NO_PRINTER: "Printer not found",
        }

    def _enum(self, names: Sequence[str], buffer: Any, length_ref: Any, count_ref: Any) -> int:
        data = "\n".join(names).encode("cp949")
        _fill(buffer, data)
        _set(length_ref, len(data))
        _set(count_ref, len(names))
        return 0

    def _h_R600LibInit(self):
        self.enumerated = False      # 새 세션: 목록을 다시 조회해야 선택할 수 있다
        return 0

    def _h_R600LibClear(self):
        self.enumerated = False
        return 0

    def _h_R600EnumUsbPrt(self, buffer, length_ref, count_ref):
        self.enumerated = True
        return self._enum(self.usb_printers, buffer, length_ref, count_ref)

    def _h_R600EnumTcpPrt(self, buffer, length_ref, count_ref):
        self.enumerated = True
        return self._enum(self.tcp_printers, buffer, length_ref, count_ref)

    def _h_R600SelectPrt(self, name):
        # 실제 SDK처럼, 목록을 조회하지 않고 선택하면 실패한다(실기에서 확인된 오류 코드).
        if not getattr(self, "enumerated", False):
            return R600_ERR_SELECT_WITHOUT_ENUM
        text = (_normalize(name) or b"").decode("cp949")
        if text not in self.usb_printers + self.tcp_printers:
            return R600_ERR_NO_PRINTER
        self.selected = text
        return 0

    def _h_R600CardInject(self, position):
        if self.feeder_empty:
            return R600_ERR_FEEDER_EMPTY
        self.card_inside = True
        return 0

    def _h_R600CardEject(self, position):
        self.card_inside = False
        return 0

    def _h_R600CommitCanvas(self, buffer, length_ref):
        self.canvas_count += 1
        info = f"FAKE_CANVAS_{self.canvas_count}".encode("ascii")
        _fill(buffer, info)
        _set(length_ref, len(info))
        return 0

    def _h_R600PrintDraw(self, front, back):
        if self.ribbon_out:
            return R600_ERR_RIBBON_OUT
        self.printed_cards += 1
        self.ribbon_remaining = max(0, self.ribbon_remaining - 1)
        self.film_remaining = max(0, self.film_remaining - 1)
        return 0

    def _h_R600QueryPrtStatus(self, chassis, head, heater, main, sub, error, warning,
                              main_code, sub_code):
        # 실기(LUCA-40KM)는 정상일 때 0이 아니라 고유 번호를 준다: 주 상태 1004(Ready), 오류 3001(No Error).
        for ref, value in ((chassis, 2800), (head, 3800), (heater, 17700),
                           (main, R600_STATUS_READY), (sub, 0x57000000),
                           (error, self.error_status or R600_STATUS_NO_ERROR),
                           (warning, self.warning_status),
                           (main_code, 0), (sub_code, 0)):
            _set(ref, value)
        return 0

    def _message(self, code, buffer, length_ref):
        text = self.error_messages.get(code, "").encode("ascii")
        _fill(buffer, text)
        _set(length_ref, len(text))
        return 0

    def _error_message(self, code, buffer, length_ref):
        # 실기처럼, 오류 코드가 아닌 값(상태 번호 등)에는 중국어로 "그런 오류 코드 없음"을 돌려준다.
        if code in self.error_messages:
            return self._message(code, buffer, length_ref)
        text = "没有该错误码".encode("gbk")
        _fill(buffer, text)
        _set(length_ref, len(text))
        return 0

    def _status_message(self, code, buffer, length_ref):
        text = self.status_messages.get(code)
        if text is None:
            text = self.error_messages.get(code)
        if text is None:
            _fill(buffer, b"")
            return 8421388          # 실기가 모르는 상태 값에 돌려주는 코드
        data = text.encode("ascii")
        _fill(buffer, data)
        _set(length_ref, len(data))
        return 0

    _h_R600GetErrorInfo = _error_message
    _h_R600GetErrorOuterInfo = _error_message
    _h_R600GetCheckStatusMsg = _status_message

    def _h_R600IsFeederNoEmpty(self, flag):
        _set(flag, 0 if self.feeder_empty else 1)
        return 0

    def _h_R600IsPrtHaveCard(self, flag):
        _set(flag, 1 if self.card_inside else 0)
        return 0

    def _h_R600GetCardPos(self, pos):
        _set(pos, 1 if self.card_inside else 0)
        return 0

    def _h_R600GetPrtInfo(self, head_count, card_count, main_fw, sub_fw, fpga):
        for ref, value in ((head_count, 1200), (card_count, self.printed_cards),
                           (main_fw, 0x0102), (sub_fw, 0x0003), (fpga, 0x0001)):
            _set(ref, value)
        return 0

    def _h_R600GetPrtManufInfo(self, *args):
        values = (b"R600 FAKE", b"RTAI", b"FAKE0001", b"2025-01-01", b"1.2.3")
        for index, value in enumerate(values):
            _fill(args[index * 2], value)
            _set(args[index * 2 + 1], len(value))
        return 0

    def _h_R600GetSN(self, device, buffer, length_ref):
        _fill(buffer, b"FAKE0001")
        _set(length_ref, 8)
        return 0

    def _h_R600GetPrtOptionStatus(self, mode, *refs):
        for ref, value in zip(refs, (0, 0, 1, 0, 1, 0, 0)):
            _set(ref, value)
        return 0

    def _h_R600GetRbnAndFilmRemaining(self, ribbon, film):
        _set(ribbon, self.ribbon_remaining)
        _set(film, self.film_remaining)
        return 0

    def _h_R600RibbonSettingsRW(self, mode, ribbon_type, film_type, ribbon_near, film_near):
        for ref, value in ((ribbon_type, self.ribbon_type), (film_type, self.film_type),
                           (ribbon_near, 10), (film_near, 10)):
            _set(ref, value)
        return 0

    def _h_R600MAGCardWrite(self, timeout, t1, n1, t2, n2, t3, n3):
        self.mag_tracks = [bytes(buffer[:length]) for buffer, length in ((t1, n1), (t2, n2), (t3, n3))]
        return 0

    def _h_R600MAGCardRead(self, timeout, t1, n1, t2, n2, t3, n3):
        tracks = self.mag_read_override or self.mag_tracks
        for buffer, length_ref, data in ((t1, n1, tracks[0]), (t2, n2, tracks[1]), (t3, n3, tracks[2])):
            _fill(buffer, data)
            _set(length_ref, len(data))
        return 0

    def _h_R600M1CardGetSN(self, mode, buffer):
        _fill(buffer, b"\x04\xA1\xB2\xC3")
        return 0

    def _h_R600M1CardRead(self, is_hex, block, buffer):
        _fill(buffer, b"00112233445566778899AABBCCDDEEFF")
        return 0


# ── IDP SMART ──────────────────────────────────────────────────────────────
FAKE_SMART_HANDLE = 0x5A000001
SMART_PID_51 = 0x3510
SMART_PID_50 = 0x3850
SMART_PID_81 = 0x3810


class FakeSmartLibrary(FakeLibrary):
    """SmartComm2.dll 흉내. 기본은 플리퍼·마그네틱이 달린 SMART-51 한 대(USB)."""

    def __init__(self, devices: Optional[Sequence[sp.SmartDeviceInfo]] = None) -> None:
        super().__init__()
        self.devices: List[sp.SmartDeviceInfo] = list(devices) if devices is not None else [
            sp.SmartDeviceInfo(name="IDP SMART-51 Card Printer", id="SMART",
                               dev="USB001", desc="SMART-51 FAKE (USB001)",
                               pid=SMART_PID_51, connection="usb"),
        ]
        self.opened: Optional[str] = None
        self.open_count = 0
        self.hopper_empty = False
        self.ribbon_out = False
        self.has_flipper = True
        self.has_magnetic = True
        self.cover_open = False
        self.card_inside = False
        self.sbs_mode = False
        self.ribbon_type = 0
        self.ribbon_max = 250
        self.ribbon_remain = 180
        self.printed_pages = 0
        self.drawn: List[Tuple[int, int, str]] = []
        self.mag_tracks: List[bytes] = [b"", b"", b""]
        self.mag_read_override: Optional[Tuple[bytes, bytes, bytes]] = None
        self.status_override: Optional[Tuple[int, int]] = None

    def _group(self) -> int:
        for device in self.devices:
            if device.desc == self.opened:
                return device.group
        return sp.GROUP_SMART51

    def _list(self, listing_ref, connection: Optional[str]) -> int:
        listing = _target(listing_ref)
        chosen = [d for d in self.devices if connection is None or d.connection == connection]
        listing.n = len(chosen)
        for index, device in enumerate(chosen):
            item = listing.item[index]
            item.name, item.id, item.dev = device.name, device.id, device.dev
            item.desc, item.pid = device.desc, device.pid
        return sp.SM_SUCCESS if chosen else sp.SM_F_FOUNDNODEV

    def _h_SmartComm_GetDeviceList2(self, listing_ref):
        return self._list(listing_ref, None)

    def _h_SmartCommEx_GetDeviceList2(self, listing_ref, option):
        return self._list(listing_ref, "usb" if option == sp.SMART_DEVICELIST_USB else "net")

    def _h_SmartDCL_OpenDevice2(self, handle_ref, device, by, orientation):
        if device not in [d.desc for d in self.devices]:
            return 0x80000003
        self.opened = device
        self.open_count += 1
        _set(handle_ref, FAKE_SMART_HANDLE)
        return 0

    def _h_SmartDCL_CloseDevice(self, handle):
        self.opened = None
        self.sbs_mode = False
        return 0

    def _h_SmartComm_GetStatus(self, handle, words):
        if self.status_override is not None:
            words[0], words[1] = self.status_override
            return 0
        group = self._group()
        ms = ef = 0
        if group == sp.GROUP_SMART81:
            if not self.has_flipper:
                ms |= sp._S81_SINGLE_ONLY
            if self.has_magnetic:
                ms |= sp._S81_MAGNETIC
            if self.hopper_empty:
                ms |= sp._S81_HOP_EMPTY
            if self.cover_open:
                ms |= sp._S81_COVER_OPEN
            if self.ribbon_out:
                ef |= 0x0000000200000000
        elif group == sp.GROUP_SMART50:
            if self.has_flipper:
                ms |= 0x0000000000800000
            if not self.hopper_empty:
                ms |= sp._S50_HOPPER_HAS_CARD
            if self.cover_open:
                ms |= sp._S50_COVER_OPEN
            if self.ribbon_out:
                ms |= 0x0010000000000000
        else:
            if self.has_flipper:
                ms |= 0x0000000000008000
            if self.has_magnetic:
                ms |= sp._S51_MAGNETIC
            if self.hopper_empty:
                ms |= sp._S51_CARD_EMPTY
            if self.cover_open:
                ms |= sp._S51_COVER_OPEN
            if self.ribbon_out:
                ms |= 0x0020000000000000
        words[0] = ms if ms < 2 ** 63 else ms - 2 ** 64
        words[1] = ef
        return 0

    def _ribbon(self, refs):
        for ref, value in zip(refs, (self.ribbon_type, self.ribbon_max, self.ribbon_remain, 0)):
            _set(ref, value)
        return 0

    def _h_SmartComm_GetRibbonInfo(self, handle, *refs):
        return self._ribbon(refs)

    def _h_SmartComm_GetRibbonInfo2(self, handle, device, *refs):
        return self._ribbon(refs)

    def _h_SmartComm_GetSystemInfo(self, handle, info_ref):
        info = _target(info_ref)
        info.printer.ver = "1.02.90"
        info.printer.serial = "FAKE51-0001"
        info.printer.hserial = "HEAD-0001"
        info.printer.cleaning = 3
        return 0

    def _h_SmartComm_SBSStart(self, handle):
        self.sbs_mode = True
        return 0

    def _h_SmartComm_SBSEnd(self, handle):
        self.sbs_mode = False
        return 0

    def _h_SmartComm_CardIn(self, handle):
        if self.hopper_empty:
            return sp.SM_F_HOPPEREMPTY
        self.card_inside = True
        return 0

    def _h_SmartComm_CardOut(self, handle):
        self.card_inside = False
        return 0

    def _h_SmartComm_DrawImage(self, handle, page, panel, x, y, cx, cy, path, area):
        self.drawn.append((page, panel, path))
        return 0

    def _h_SmartComm_DoPrint(self, handle):
        if self.ribbon_out:
            return sp.SM_F_EMPTYRIBBON
        self.printed_pages += 1
        self.ribbon_remain = max(0, self.ribbon_remain - 1)
        return 0

    def _h_SmartComm_MagSetAllBuffer(self, handle, s1, b1, n1, s2, b2, n2, s3, b3, n3):
        self.mag_tracks = [bytes(buffer[:length]) if use else b""
                           for use, buffer, length in ((s1, b1, n1), (s2, b2, n2), (s3, b3, n3))]
        return 0

    def _h_SmartComm_MagGetAllBuffer(self, handle, g1, b1, n1, g2, b2, n2, g3, b3, n3):
        tracks = self.mag_read_override or self.mag_tracks
        for buffer, length_ref, data in ((b1, n1, tracks[0]), (b2, n2, tracks[1]), (b3, n3, tracks[2])):
            _fill(buffer, data)
            _set(length_ref, len(data))
        return 0
