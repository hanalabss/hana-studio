"""
printer/registry.py
제품군을 가리지 않는 프린터 탐지와 열기.

    devices = registry.discover()        # list[PrinterDevice] + devices.diagnostics
    printer = registry.open(devices[0])  # CardPrinter (연결된 상태)

discover()는 DLL이 없거나 장치가 없어도 예외를 내지 않는다. 이유는 diagnostics에 한국어로 남긴다.
"""

from __future__ import annotations

import os
from typing import Callable, Dict, Iterable, List, Optional

from .base import FAMILY_RTAI, FAMILY_SMART, CardPrinter, PrinterDevice, PrinterError

ALL_FAMILIES = (FAMILY_RTAI, FAMILY_SMART)
FAKE_ENV = "HANA_FAKE_PRINTERS"


class DiscoveryResult(List[PrinterDevice]):
    """탐지된 장치 목록. diagnostics에 제품군별 경과(장치 없음, DLL 없음 등)가 들어 있다."""

    def __init__(self, devices: Iterable[PrinterDevice] = (), diagnostics: Iterable[str] = ()):
        super().__init__(devices)
        self.diagnostics: List[str] = list(diagnostics)

    def by_family(self, family: str) -> List[PrinterDevice]:
        return [device for device in self if device.family == family]


_use_fakes: Optional[bool] = None
_fake_libraries: Dict[str, object] = {}
_adapter_factories: Dict[str, Callable[[PrinterDevice], CardPrinter]] = {}


def use_fakes(enabled: bool = True) -> None:
    """실기 대신 가짜 프린터를 쓰게 한다(화면 개발·시연용). 환경 변수 HANA_FAKE_PRINTERS=1과 같다."""
    global _use_fakes
    _use_fakes = enabled
    _fake_libraries.clear()


def fakes_enabled() -> bool:
    if _use_fakes is not None:
        return _use_fakes
    return os.environ.get(FAKE_ENV, "") not in ("", "0", "false", "False")


def register_adapter(family: str, factory: Optional[Callable[[PrinterDevice], CardPrinter]]) -> None:
    """open()이 family 장치에 쓸 어댑터 생성 함수를 바꾼다. None이면 기본값으로 되돌린다."""
    if factory is None:
        _adapter_factories.pop(family, None)
    else:
        _adapter_factories[family] = factory


def _fake_library(family: str):
    from . import fakes
    if family not in _fake_libraries:
        _fake_libraries[family] = (fakes.FakeR600Library() if family == FAMILY_RTAI
                                   else fakes.FakeSmartLibrary())
    return _fake_libraries[family]


def _discover_rtai(diagnostics: List[str]) -> List[PrinterDevice]:
    if fakes_enabled():
        library = _fake_library(FAMILY_RTAI)
        diagnostics.append("RTAI: 가상 프린터를 사용합니다.")
        return [PrinterDevice(FAMILY_RTAI, name, "fake", name) for name in library.usb_printers]
    from .printer_discovery import PrinterDiscovery
    from .printer_utils import find_printer_dll

    dll_path = find_printer_dll()
    if not dll_path:
        diagnostics.append("RTAI: 라이브러리(libDSRetransfer600App.dll)를 찾을 수 없습니다.")
        return []
    with PrinterDiscovery(dll_path) as discovery:
        found = discovery.discover_all_printers()
    devices = [PrinterDevice(FAMILY_RTAI, info.name,
                             "tcp" if info.connection_type == "TCP" else "usb", info.name)
               for info in found]
    diagnostics.append(f"RTAI: 프린터 {len(devices)}대를 찾았습니다." if devices
                       else "RTAI: 연결된 프린터가 없습니다.")
    return devices


def _discover_smart(diagnostics: List[str]) -> List[PrinterDevice]:
    from . import smart_printer as sp

    if fakes_enabled():
        binding = sp.SmartPrinter(lib=_fake_library(FAMILY_SMART))
        diagnostics.append("SMART: 가상 프린터를 사용합니다.")
        connection_override: Optional[str] = "fake"
    else:
        if not sp.find_smart_dll():
            diagnostics.append("SMART: 라이브러리(SmartComm2.dll)를 찾을 수 없습니다.")
            return []
        binding = sp.SmartPrinter()
        connection_override = None
    found = binding.list_devices()
    diagnostics.extend(f"SMART: {message}" for message in binding.last_list_errors)
    devices = [PrinterDevice(FAMILY_SMART, item.name or item.desc,
                             connection_override or item.connection, item.desc,
                             {"id": item.id, "dev": item.dev, "pid": item.pid,
                              "group": item.group})
               for item in found]
    if not fakes_enabled():
        diagnostics.append(f"SMART: 프린터 {len(devices)}대를 찾았습니다." if devices
                           else "SMART: 연결된 프린터가 없습니다.")
    return devices


_DISCOVERERS = {FAMILY_RTAI: _discover_rtai, FAMILY_SMART: _discover_smart}


def discover(families: Iterable[str] = ALL_FAMILIES) -> DiscoveryResult:
    """두 제품군의 프린터를 찾는다. 한쪽이 실패해도 다른 쪽 결과는 돌려준다."""
    result = DiscoveryResult()
    for family in families:
        discoverer = _DISCOVERERS.get(family)
        if discoverer is None:
            result.diagnostics.append(f"알 수 없는 프린터 제품군입니다: {family}")
            continue
        try:
            result.extend(discoverer(result.diagnostics))
        except PrinterError as exc:
            detail = f" ({exc.detail})" if exc.detail else ""
            result.diagnostics.append(f"{family.upper()}: {exc}{detail}")
        except Exception as exc:
            result.diagnostics.append(f"{family.upper()}: 탐지 중 오류가 발생했습니다 - "
                                      f"{type(exc).__name__}: {exc}")
    return result


def create(device: PrinterDevice) -> CardPrinter:
    """device 제품군에 맞는 어댑터를 만든다(아직 연결하지 않는다)."""
    factory = _adapter_factories.get(device.family)
    if factory is not None:
        return factory(device)
    fake = fakes_enabled() or device.connection == "fake"
    if device.family == FAMILY_RTAI:
        from .rtai_adapter import RtaiAdapter
        return RtaiAdapter(lib=_fake_library(FAMILY_RTAI), settle_seconds=0.2) if fake else RtaiAdapter()
    if device.family == FAMILY_SMART:
        from .smart_adapter import SmartAdapter
        return SmartAdapter(lib=_fake_library(FAMILY_SMART)) if fake else SmartAdapter()
    raise PrinterError(f"알 수 없는 프린터 제품군입니다: {device.family}")


def open(device: PrinterDevice) -> CardPrinter:  # noqa: A001 - 계약에 정해진 이름
    """device에 연결된 CardPrinter를 돌려준다. 실패하면 PrinterError."""
    printer = create(device)
    try:
        printer.connect(device)
    except PrinterError:
        printer.close()
        raise
    except Exception as exc:
        printer.close()
        raise PrinterError("프린터에 연결하지 못했습니다.", detail=f"{type(exc).__name__}: {exc}",
                           recoverable=True) from exc
    return printer
