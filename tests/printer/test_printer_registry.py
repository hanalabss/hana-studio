import pytest

from printer import registry
from printer import smart_printer as sp
from printer.base import CardJob, CardPrinter, PrinterDevice, PrinterError, SideImages
from printer.rtai_adapter import RtaiAdapter
from printer.smart_adapter import SmartAdapter


@pytest.fixture(autouse=True)
def reset_registry(monkeypatch):
    monkeypatch.delenv(registry.FAKE_ENV, raising=False)
    registry.use_fakes(None)
    yield
    registry.use_fakes(None)
    registry.register_adapter("rtai", None)
    registry.register_adapter("smart", None)


def test_discover_with_fakes_returns_both_families_and_diagnostics():
    registry.use_fakes(True)
    found = registry.discover()
    assert isinstance(found, list)
    assert [(d.family, d.connection) for d in found] == [("rtai", "fake"), ("smart", "fake")]
    assert found.by_family("smart")[0].extra["pid"] == 0x3510
    assert found.diagnostics == ["RTAI: 가상 프린터를 사용합니다.", "SMART: 가상 프린터를 사용합니다."]


def test_environment_variable_turns_fakes_on(monkeypatch):
    assert not registry.fakes_enabled()
    monkeypatch.setenv(registry.FAKE_ENV, "1")
    assert registry.fakes_enabled()
    assert len(registry.discover()) == 2


def test_discover_one_family_only():
    registry.use_fakes(True)
    assert [d.family for d in registry.discover(["smart"])] == ["smart"]


def test_discover_never_raises_when_both_dlls_are_missing(monkeypatch):
    import printer.printer_utils as utils
    monkeypatch.setattr(utils, "find_printer_dll", lambda: None)
    monkeypatch.setattr(sp, "find_smart_dll", lambda: None)
    found = registry.discover()
    assert found == []
    assert found.diagnostics == [
        "RTAI: 라이브러리(libDSRetransfer600App.dll)를 찾을 수 없습니다.",
        "SMART: 라이브러리(SmartComm2.dll)를 찾을 수 없습니다.",
    ]


def test_discover_keeps_the_other_family_when_one_blows_up(monkeypatch):
    registry.use_fakes(True)

    def broken(diagnostics):
        raise OSError("DLL 로드 실패")

    monkeypatch.setitem(registry._DISCOVERERS, "rtai", broken)
    found = registry.discover()
    assert [d.family for d in found] == ["smart"]
    assert found.diagnostics[0] == "RTAI: 탐지 중 오류가 발생했습니다 - OSError: DLL 로드 실패"


def test_discover_reports_printer_errors_with_detail(monkeypatch):
    def failing(diagnostics):
        raise PrinterError("SMART 프린터 라이브러리를 불러오지 못했습니다.", detail="bad image")

    monkeypatch.setitem(registry._DISCOVERERS, "smart", failing)
    found = registry.discover(["smart"])
    assert found == []
    assert found.diagnostics == ["SMART: SMART 프린터 라이브러리를 불러오지 못했습니다. (bad image)"]


def test_discover_unknown_family_is_a_diagnostic():
    assert registry.discover(["zebra"]).diagnostics == ["알 수 없는 프린터 제품군입니다: zebra"]


def test_unloadable_smart_dll_is_reported_not_raised(monkeypatch, tmp_path):
    import printer.printer_utils as utils
    bogus = tmp_path / "SmartComm2.dll"
    bogus.write_bytes(b"MZ not really a dll")
    monkeypatch.setattr(utils, "find_printer_dll", lambda: None)
    monkeypatch.setattr(sp, "find_smart_dll", lambda: str(bogus))
    found = registry.discover()
    assert found == []
    assert found.diagnostics[1].startswith("SMART: SMART 프린터 라이브러리를 불러오지 못했습니다.")


def test_open_returns_a_connected_adapter_per_family(make_image):
    registry.use_fakes(True)
    rtai, smart = registry.discover()
    with registry.open(rtai) as printer:
        assert isinstance(printer, RtaiAdapter) and printer.is_connected
        assert printer.status().ready
    with registry.open(smart) as printer:
        assert isinstance(printer, SmartAdapter) and printer.is_connected
        printer.print_card(CardJob(front=SideImages(make_image())))
        assert printer.status().ribbon_remaining == 179


def test_open_unknown_family_raises():
    with pytest.raises(PrinterError, match="알 수 없는 프린터 제품군"):
        registry.open(PrinterDevice("zebra", "ZXP"))


def test_open_failure_is_a_printer_error_and_nothing_stays_open():
    registry.use_fakes(True)
    with pytest.raises(PrinterError):
        registry.open(PrinterDevice("smart", "ghost", "fake", "NOT THERE"))
    with pytest.raises(PrinterError, match="연결하지 못했습니다"):
        registry.open(PrinterDevice("rtai", "ghost", "fake", "ghost"))


def test_open_wraps_unexpected_errors(monkeypatch):
    class Exploding(CardPrinter):
        family = "rtai"
        closed = False

        def capabilities(self):
            return set()

        def connect(self, device):
            raise RuntimeError("boom")

        def close(self):
            Exploding.closed = True

        is_connected = False

        def status(self):
            raise NotImplementedError

        def print_card(self, job, progress=None):
            raise NotImplementedError

    registry.register_adapter("rtai", lambda device: Exploding())
    with pytest.raises(PrinterError, match="프린터에 연결하지 못했습니다") as info:
        registry.open(PrinterDevice("rtai", "x"))
    assert "RuntimeError: boom" in info.value.detail
    assert Exploding.closed
