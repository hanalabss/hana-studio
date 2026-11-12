"""PrinterService: 백그라운드 탐지, 마지막 프린터 기억, 한가할 때만 상태 조회."""
import threading

import pytest
from PySide6.QtCore import QThread

from printer import registry
from printer.base import CAP_DUAL_SIDE, CAP_UV, CAP_WHITE, PrinterStatus
from services import printers as printers_module
from services.print_service import PrintService
from services.printers import PrinterService, default_capabilities
from services_helpers import StubPrinter, pump, wait_until


class Signals:
    def __init__(self, service):
        self.devices, self.current, self.status, self.busy = [], [], [], []
        service.devices_changed.connect(self.devices.append)
        service.current_changed.connect(self.current.append)
        service.status_changed.connect(self.status.append)
        service.busy_changed.connect(self.busy.append)


@pytest.fixture
def printers(queue):
    service = PrinterService(queue, poll_interval_ms=200)
    yield service
    service.shutdown()


def test_discovery_runs_in_a_background_thread(app, printers, monkeypatch):
    signals = Signals(printers)
    threads = []
    original = registry.discover
    monkeypatch.setattr(
        registry, "discover",
        lambda *a, **k: threads.append(QThread.currentThread()) or original(*a, **k))
    assert printers.devices() == [] and printers.current() is None

    assert printers.refresh() is True
    assert printers.is_busy and signals.devices == []          # 부른 쪽을 막지 않는다
    wait_until(app, lambda: signals.devices)

    assert threads and threads[0] is not app.thread()
    assert [d.family for d in printers.devices()] == ["rtai", "smart"]
    assert all(d.connection == "fake" for d in printers.devices())
    assert any("가상 프린터" in line for line in printers.diagnostics())
    assert printers.current() == printers.devices()[0] and signals.current == [printers.devices()[0]]
    assert signals.busy == [True, False] and not printers.is_busy


def test_last_used_device_is_remembered_across_restarts(app, tmp_path, queue):
    first = PrinterService(queue)
    first.refresh()
    wait_until(app, lambda: first.devices())
    smart = first.devices()[1]
    first.select(smart)
    first.shutdown()
    settings = tmp_path / "appdata" / "HanaStudio" / "printer.json"
    assert settings.is_file()

    second = PrinterService(queue)
    assert second.remembered() == smart and second.current() is None
    second.refresh()
    wait_until(app, lambda: second.devices())
    assert second.current() == smart
    second.shutdown()


def test_remembered_device_that_is_gone_falls_back_to_the_first(app, queue, tmp_path):
    settings = tmp_path / "printer.json"
    settings.write_text('{"last_device": {"family": "rtai", "name": "없어진 프린터", "address": "X"}}',
                        encoding="utf-8")
    service = PrinterService(queue, str(settings))
    service.refresh()
    wait_until(app, lambda: service.devices())
    assert service.current() == service.devices()[0]
    assert service.remembered().name == "없어진 프린터"        # 다시 연결되면 그 프린터로 돌아간다
    service.shutdown()


def test_capabilities_follow_the_current_device(app, printers):
    assert printers.capabilities() == set()
    printers.refresh()
    wait_until(app, lambda: printers.devices())
    rtai, smart = printers.devices()
    assert {CAP_WHITE, CAP_UV, CAP_DUAL_SIDE} <= printers.capabilities()
    printers.select(smart)
    assert CAP_WHITE not in printers.capabilities() and CAP_DUAL_SIDE in printers.capabilities()
    assert default_capabilities(smart) == printers.capabilities()
    assert default_capabilities(None) == set()


def test_family_defaults_match_the_adapters():
    from printer.rtai_adapter import RtaiAdapter
    from printer.smart_adapter import SmartAdapter

    rtai, smart = registry.discover()
    assert default_capabilities(rtai) == RtaiAdapter(lib=registry._fake_library("rtai")).capabilities()
    assert default_capabilities(smart) == SmartAdapter(lib=registry._fake_library("smart")).capabilities()


def test_status_poll_reports_and_refines_capabilities(app, printers, smart_lib):
    signals = Signals(printers)
    printers.refresh()
    wait_until(app, lambda: printers.devices())
    printers.select(printers.devices()[1])
    smart_lib.has_flipper = False
    smart_lib.ribbon_remain = 42

    assert printers.status() is None
    assert printers.poll_now() is True
    assert printers.poll_now() is False                        # 조회 중에는 겹쳐 하지 않는다
    wait_until(app, lambda: signals.status)
    status = signals.status[0]
    assert isinstance(status, PrinterStatus) and status.ready and status.ribbon_remaining == 42
    assert printers.status() is status
    assert CAP_DUAL_SIDE not in printers.capabilities()        # 플리퍼가 없는 것을 알게 됐다
    assert smart_lib.opened is None                            # 조회 뒤에는 장치를 닫아 둔다

    printers.poll_now()
    wait_until(app, lambda: not printers.is_busy)
    assert len(signals.status) == 1                            # 바뀐 게 없으면 다시 알리지 않는다

    smart_lib.hopper_empty = True
    printers.poll_now()
    wait_until(app, lambda: len(signals.status) == 2)
    assert signals.status[1].feeder_empty and not signals.status[1].ready


def test_periodic_polling_only_when_idle(app, printers, rtai_lib):
    signals = Signals(printers)
    printers.refresh()
    wait_until(app, lambda: printers.devices())
    printers.start_polling(200)
    wait_until(app, lambda: signals.status)
    rtai_lib.feeder_empty = True
    wait_until(app, lambda: len(signals.status) == 2)          # 타이머가 알아서 다시 본다
    assert signals.status[1].feeder_empty
    printers.stop_polling()
    wait_until(app, lambda: not printers.is_busy)
    rtai_lib.feeder_empty = False
    pump(app, 450)
    assert len(signals.status) == 2


def test_connection_failure_becomes_a_not_ready_status(app, printers, rtai_lib):
    signals = Signals(printers)
    printers.refresh()
    wait_until(app, lambda: printers.devices())
    rtai_lib.usb_printers.clear()                              # 케이블이 빠졌다
    printers.poll_now()
    wait_until(app, lambda: signals.status)
    assert not signals.status[0].ready and signals.status[0].message


def test_nothing_touches_the_sdk_while_a_print_is_running(app, store, simple_document, queue, printers,
                                                         stub_device, monkeypatch):
    signals = Signals(printers)
    printers.refresh()
    wait_until(app, lambda: signals.devices)
    stub = StubPrinter(gate_on=1)
    queue.set_printer_factory(lambda device: stub)
    discover_calls, open_calls = [], []
    original_discover, original_open = registry.discover, registry.open
    monkeypatch.setattr(registry, "discover",
                        lambda *a, **k: discover_calls.append(1) or original_discover(*a, **k))
    monkeypatch.setattr(registry, "open", lambda *a, **k: open_calls.append(1) or original_open(*a, **k))

    service = PrintService(queue)
    finished = []
    service.finished.connect(finished.append)
    service.start(simple_document, store, store.ids()[:2], stub_device)
    wait_until(app, lambda: stub.entered.is_set())

    assert printers.printing
    assert printers.refresh() is False and printers.poll_now() is False
    printers.start_polling(200)
    pump(app, 500)
    assert discover_calls == [] and open_calls == [] and not printers.is_busy

    stub.gate.set()
    wait_until(app, lambda: finished)
    wait_until(app, lambda: len(signals.devices) == 2)         # 미뤄 둔 탐지가 인쇄 뒤에 돈다
    assert discover_calls == [1]
    wait_until(app, lambda: signals.status)                    # 상태 조회도 다시 시작된다
    assert open_calls
    printers.stop_polling()


def test_print_thread_waits_for_a_status_query_in_flight(app, store, simple_document, queue, printers,
                                                         rtai_device, monkeypatch):
    """상태 조회가 SDK를 쓰는 동안 시작된 인쇄는 조회가 끝난 뒤에야 프린터를 연다."""
    printers.refresh()
    wait_until(app, lambda: printers.devices())
    inside, release, order = threading.Event(), threading.Event(), []
    original_open = registry.open

    def slow_open(device):
        if not inside.is_set():
            inside.set()
            order.append("status-begin")
            release.wait(10)
            order.append("status-end")
            return original_open(device)
        order.append("print-open")
        return original_open(device)

    monkeypatch.setattr(registry, "open", slow_open)
    assert printers.poll_now()
    wait_until(app, inside.is_set)

    service = PrintService(queue)
    finished = []
    service.finished.connect(finished.append)
    service.start(simple_document, store, store.ids()[:1], rtai_device)
    pump(app, 200)
    assert order == ["status-begin"]                           # 인쇄 스레드는 기다리는 중
    release.set()
    wait_until(app, lambda: finished)
    assert order == ["status-begin", "status-end", "print-open"]
    assert finished[0].success == 1


def test_use_fakes_switch_refreshes_the_list(app, queue, monkeypatch):
    registry.use_fakes(False)
    monkeypatch.setattr(registry, "_DISCOVERERS", {
        family: (lambda diagnostics, _original=function:
                 _original(diagnostics) if registry.fakes_enabled() else [])
        for family, function in registry._DISCOVERERS.items()})
    service = PrinterService(queue)
    signals = Signals(service)
    assert service.use_fakes is False
    service.refresh()
    wait_until(app, lambda: signals.devices)
    assert signals.devices[0] == [] and service.current() is None

    service.set_use_fakes(True)
    wait_until(app, lambda: len(signals.devices) == 2)
    assert service.use_fakes and len(service.devices()) == 2 and service.current() is not None
    service.shutdown()


def test_global_queue_is_the_default_and_can_be_left_unguarded():
    from printer.print_queue import print_queue

    before = (print_queue._printer_factory, print_queue._r600_factory)
    service = PrinterService(guard_queue=False)
    assert service._queue is print_queue
    assert (print_queue._printer_factory, print_queue._r600_factory) == before
    assert printers_module.app_data_dir().endswith("HanaStudio")
    service.shutdown()
