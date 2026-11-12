"""인쇄 큐: 선택한 프린터 사용, 작업 취소·중단, 카드별 CardJob 작업, 이력 기록."""
import threading
import time as real_time

import pytest
from PySide6.QtCore import QEventLoop, QThread

from printer.base import CardJob, CardPrinter, PrinterDevice, PrinterError, PrinterStatus, SideImages
from printer.fakes import R600_ERR_FEEDER_EMPTY, FakeR600Library, FakeSmartLibrary
from printer.history import JobHistory
from printer.print_queue import PrintQueueManager
from printer.printer_discovery import PrinterInfo
from printer.r600_printer import R600Printer
from printer.smart_adapter import SmartAdapter

_monotonic = real_time.monotonic


def wait_until(app, predicate, timeout=15.0):
    deadline = _monotonic() + timeout
    while not predicate():
        app.processEvents(QEventLoop.AllEvents, 20)
        QThread.msleep(2)
        assert _monotonic() < deadline, "시간 안에 끝나지 않았습니다"
    app.processEvents(QEventLoop.AllEvents, 20)


class Recorder:
    """큐 시그널을 받아 적는다."""

    def __init__(self, queue):
        self.finished = []
        self.completed = []
        self.cards = []
        self.cancelled = []
        self.started = []
        self.logs = []
        queue.all_jobs_completed.connect(lambda *args: self.finished.append(args))
        queue.job_completed.connect(lambda job_id, ok: self.completed.append((job_id, ok)))
        queue.card_result.connect(lambda *args: self.cards.append(args))
        queue.job_cancelled.connect(self.cancelled.append)
        queue.job_started.connect(lambda job_id, index, total: self.started.append(job_id))
        queue.log_message.connect(self.logs.append)


@pytest.fixture
def queue(qapp):
    manager = PrintQueueManager()
    yield manager
    manager.cancel_all()


@pytest.fixture
def two_printers(queue):
    """프린터 두 대가 보이는 가짜 R600 라이브러리를 기존 방식 작업에 물린다."""
    lib = FakeR600Library(usb_printers=["FIRST", "SECOND"])

    def factory(dll_path, selected):
        return R600Printer.from_library(lib, selected)

    queue.set_legacy_printer_factory(factory)
    return lib


class StubPrinter(CardPrinter):
    """스레드 흐름만 볼 때 쓰는 가짜 CardPrinter."""

    family = "stub"

    def __init__(self, fail_on=(), gate=None):
        super().__init__()
        self.printed = []
        self.fail_on = set(fail_on)
        self.gate = gate
        self.entered = threading.Event()
        self.closed = False

    def capabilities(self):
        return {"status"}

    def connect(self, device):
        self.device = device

    def close(self):
        self.closed = True

    @property
    def is_connected(self):
        return True

    def status(self):
        return PrinterStatus(True, "ok")

    def print_card(self, job, progress=None):
        number = len(self.printed) + 1
        self.printed.append(job.record_id)
        self.entered.set()
        if self.gate is not None and number == 1:
            self.gate.wait(10)
        if number in self.fail_on:
            raise PrinterError("리본을 모두 사용했습니다.", code=0x8000001D, recoverable=True)
        self._report(100, "완료", progress)


STUB_DEVICE = PrinterDevice("stub", "STUB", "fake", "STUB")


def cards(count):
    return [CardJob(front=SideImages(f"card{i}.png"), record_id=100 + i) for i in range(1, count + 1)]


# ── 기존 방식 작업: 선택한 프린터 ────────────────────────────────────────────
def test_legacy_job_uses_the_selected_printer_not_the_first(qapp, queue, two_printers, make_image):
    recorder = Recorder(queue)
    queue.add_job(dll_path="x.dll", front_image_path=make_image(), is_dual_side=False,
                  selected_printer=PrinterInfo("SECOND", "USB"))
    wait_until(qapp, lambda: recorder.finished)
    assert two_printers.selected == "SECOND"
    assert two_printers.calls_named("R600SelectPrt") == [("R600SelectPrt", (b"SECOND",))]
    assert "R600EnumUsbPrt" not in two_printers.call_names()
    assert recorder.finished == [(1, 0, 1, 0, "")]


def test_legacy_job_accepts_a_printer_device_as_selection(qapp, queue, two_printers, make_image):
    recorder = Recorder(queue)
    queue.add_job(dll_path="x.dll", front_image_path=make_image(), is_dual_side=False,
                  device=PrinterDevice("rtai", "SECOND", "usb", "SECOND"))
    wait_until(qapp, lambda: recorder.finished)
    assert two_printers.selected == "SECOND"


def test_legacy_job_without_selection_keeps_the_old_first_printer_behaviour(
        qapp, queue, two_printers, make_image):
    recorder = Recorder(queue)
    queue.add_job(dll_path="x.dll", front_image_path=make_image(), is_dual_side=False, quantity=2)
    wait_until(qapp, lambda: recorder.finished)
    assert two_printers.selected == "FIRST"
    assert two_printers.printed_cards == 2
    assert recorder.finished == [(2, 0, 1, 0, "")]
    assert [(index, ok) for _, index, ok, _ in recorder.cards] == [(1, True), (2, True)]


def test_selected_printer_that_is_gone_fails_the_job_without_hanging_the_queue(
        qapp, queue, two_printers, make_image):
    recorder = Recorder(queue)
    queue.add_job(dll_path="x.dll", front_image_path=make_image(), is_dual_side=False,
                  selected_printer=PrinterInfo("UNPLUGGED", "USB"))
    queue.add_job(dll_path="x.dll", front_image_path=make_image(), is_dual_side=False,
                  selected_printer=PrinterInfo("FIRST", "USB"))
    wait_until(qapp, lambda: recorder.finished)
    assert recorder.completed == [(1, False), (2, True)]
    success_cards, failed_cards, success_jobs, failed_jobs, last_error = recorder.finished[0]
    assert (success_cards, failed_cards, success_jobs, failed_jobs) == (1, 1, 1, 1)
    assert "인쇄 초기화 오류" in last_error
    assert not queue.get_queue_status()["is_processing"]


def test_no_printer_at_all_no_longer_leaves_the_queue_stuck(qapp, queue, make_image):
    lib = FakeR600Library(usb_printers=[])
    queue.set_legacy_printer_factory(lambda path, selected: R600Printer.from_library(lib, selected))
    recorder = Recorder(queue)
    queue.add_job(dll_path="x.dll", front_image_path=make_image(), is_dual_side=False)
    wait_until(qapp, lambda: recorder.finished)
    assert recorder.finished[0][4] == "사용 가능한 프린터가 없습니다."
    assert recorder.completed == [(1, False)]


def test_legacy_card_failure_is_reported_per_card(qapp, queue, two_printers, make_image):
    two_printers.fail("R600CardInject", R600_ERR_FEEDER_EMPTY, on_call=2)
    recorder = Recorder(queue)
    queue.add_job(dll_path="x.dll", front_image_path=make_image(), is_dual_side=False, quantity=3,
                  selected_printer=PrinterInfo("FIRST", "USB"))
    wait_until(qapp, lambda: recorder.finished)
    assert [(index, ok) for _, index, ok, _ in recorder.cards] == [(1, True), (2, False), (3, True)]
    assert str(R600_ERR_FEEDER_EMPTY) in recorder.cards[1][3]


# ── CardPrinter 작업: 카드마다 다른 내용 ──────────────────────────────────────
def test_card_job_prints_each_card_job_on_the_given_device(qapp, queue):
    printer = StubPrinter()
    opened = []
    queue.set_printer_factory(lambda device: opened.append(device) or printer)
    recorder = Recorder(queue)
    job_id = queue.add_card_job(STUB_DEVICE, cards(3), name="사원증")
    wait_until(qapp, lambda: recorder.finished)
    assert opened == [STUB_DEVICE]
    assert printer.printed == [101, 102, 103] and printer.closed
    assert recorder.cards == [(job_id, 1, True, ""), (job_id, 2, True, ""), (job_id, 3, True, "")]
    assert recorder.finished == [(3, 0, 1, 0, "")]
    assert queue.get_all_jobs_for_display() == []


def test_card_job_runs_through_a_real_adapter_with_variable_data(qapp, queue, make_image):
    lib = FakeSmartLibrary()
    item = lib.devices[0]
    device = PrinterDevice("smart", item.name, "fake", item.desc, {"pid": item.pid})

    def factory(target):
        adapter = SmartAdapter(lib=lib)
        adapter.connect(target)
        return adapter

    queue.set_printer_factory(factory)
    recorder = Recorder(queue)
    jobs = [CardJob(front=SideImages(make_image(color=(i * 40, 0, 0))), record_id=i)
            for i in range(1, 4)]
    queue.add_card_job(device, jobs)
    wait_until(qapp, lambda: recorder.finished)
    assert lib.printed_pages == 3
    assert len(lib.calls_named("SmartComm_DrawImage")) == 3
    assert lib.opened is None
    assert recorder.finished == [(3, 0, 1, 0, "")]


def test_card_job_stops_at_the_first_failure_by_default(qapp, queue):
    printer = StubPrinter(fail_on={2})
    queue.set_printer_factory(lambda device: printer)
    recorder = Recorder(queue)
    job_id = queue.add_card_job(STUB_DEVICE, cards(4))
    wait_until(qapp, lambda: recorder.finished)
    assert printer.printed == [101, 102]
    assert recorder.cards == [
        (job_id, 1, True, ""),
        (job_id, 2, False, "리본을 모두 사용했습니다. (코드 0x8000001D)")]
    success_cards, failed_cards, success_jobs, failed_jobs, last_error = recorder.finished[0]
    assert (success_cards, failed_cards) == (1, 1)
    assert last_error == "리본을 모두 사용했습니다. (코드 0x8000001D)"


def test_card_job_can_continue_after_a_failed_card(qapp, queue):
    printer = StubPrinter(fail_on={2})
    queue.set_printer_factory(lambda device: printer)
    recorder = Recorder(queue)
    queue.add_card_job(STUB_DEVICE, cards(4), stop_on_error=False)
    wait_until(qapp, lambda: recorder.finished)
    assert printer.printed == [101, 102, 103, 104]
    assert [ok for _, _, ok, _ in recorder.cards] == [True, False, True, True]
    assert recorder.finished[0][:2] == (3, 1)


def test_card_job_connection_failure_fails_the_job_and_moves_on(qapp, queue):
    attempts = []

    def factory(device):
        attempts.append(device)
        if len(attempts) == 1:
            raise PrinterError("프린터를 열지 못했습니다.", code=0x80000005)
        return StubPrinter()

    queue.set_printer_factory(factory)
    recorder = Recorder(queue)
    queue.add_card_job(STUB_DEVICE, cards(2))
    queue.add_card_job(STUB_DEVICE, cards(1))
    wait_until(qapp, lambda: recorder.finished)
    assert recorder.completed == [(1, False), (2, True)]
    assert recorder.finished[0][:4] == (1, 2, 1, 1)
    assert "프린터 연결 실패: 프린터를 열지 못했습니다." in recorder.finished[0][4]


def test_add_card_job_rejects_an_empty_list(queue):
    with pytest.raises(ValueError):
        queue.add_card_job(STUB_DEVICE, [])


# ── 취소 · 중단 ────────────────────────────────────────────────────────────
def test_pending_job_can_be_removed_while_another_is_printing(qapp, queue):
    gate = threading.Event()
    printer = StubPrinter(gate=gate)
    queue.set_printer_factory(lambda device: printer)
    recorder = Recorder(queue)
    first = queue.add_card_job(STUB_DEVICE, cards(1))
    second = queue.add_card_job(STUB_DEVICE, cards(2))
    third = queue.add_card_job(STUB_DEVICE, cards(1))
    wait_until(qapp, printer.entered.is_set)

    assert [job.job_id for job in queue.get_pending_jobs()] == [second, third]
    assert queue.cancel_job(second) is True
    assert queue.remove_job(second) is False
    assert queue.remove_job(first) is False            # 진행 중인 작업은 remove로 빼지 못한다
    assert [job.job_id for job in queue.get_pending_jobs()] == [third]
    assert recorder.cancelled == [second]
    display = {job["job_id"]: job["display_status"] for job in queue.get_all_jobs_for_display()}
    assert display == {first: "작업중", second: "취소", third: "대기"}

    gate.set()
    wait_until(qapp, lambda: recorder.finished)
    assert recorder.started == [first, third]
    assert recorder.finished == [(2, 0, 2, 0, "")]
    assert printer.printed == [101, 101]


def test_running_job_stops_cleanly_between_cards(qapp, queue):
    gate = threading.Event()
    printer = StubPrinter(gate=gate)
    queue.set_printer_factory(lambda device: printer)
    recorder = Recorder(queue)
    job_id = queue.add_card_job(STUB_DEVICE, cards(5))
    following = queue.add_card_job(STUB_DEVICE, cards(1))
    wait_until(qapp, printer.entered.is_set)

    assert queue.cancel_job(job_id) is True            # 첫 카드를 찍는 도중에 중단 요청
    assert recorder.cancelled == [job_id]
    gate.set()
    wait_until(qapp, lambda: recorder.finished)

    assert printer.printed == [101, 101]               # 찍던 카드는 끝내고, 다음 작업은 그대로 진행
    assert recorder.cards[0] == (job_id, 1, True, "")
    assert recorder.completed == [(job_id, True), (following, True)]
    assert recorder.finished == [(2, 0, 2, 0, "")]
    assert printer.closed


def test_stop_and_cancel_with_nothing_running_return_false(queue):
    assert queue.stop_current_job() is False
    assert queue.cancel_job(42) is False


def test_legacy_job_stops_between_cards(qapp, queue, two_printers, make_image):
    recorder = Recorder(queue)
    queue.card_result.connect(lambda job_id, index, ok, message: queue.stop_current_job())
    queue.add_job(dll_path="x.dll", front_image_path=make_image(), is_dual_side=False, quantity=50,
                  selected_printer=PrinterInfo("FIRST", "USB"))
    wait_until(qapp, lambda: recorder.finished)
    assert 1 <= two_printers.printed_cards < 50
    assert two_printers.call_names()[-1] == "R600LibClear"


# ── 이력 ──────────────────────────────────────────────────────────────────
def test_queue_writes_job_and_card_history(qapp, tmp_path):
    history = JobHistory(str(tmp_path / "history.sqlite"))
    queue = PrintQueueManager(history=history)
    printer = StubPrinter(fail_on={2})
    queue.set_printer_factory(lambda device: printer)
    recorder = Recorder(queue)
    queue.add_card_job(STUB_DEVICE, cards(3), name="3월 사원증")
    wait_until(qapp, lambda: recorder.finished)

    job = history.list_jobs()[0]
    assert (job.name, job.family, job.device_name, job.total_cards) == ("3월 사원증", "stub", "STUB", 3)
    assert (job.status, job.success_cards, job.failed_cards) == ("partial", 1, 1)
    assert job.error_text == "리본을 모두 사용했습니다. (코드 0x8000001D)"
    assert job.started_at and job.finished_at
    stored = history.list_cards(job.id)
    assert [(c.card_index, c.status, c.record_id, c.error_code) for c in stored] == [
        (1, "success", "101", None), (2, "failed", "102", 0x8000001D)]
    assert stored[1].error_text == "리본을 모두 사용했습니다. (코드 0x8000001D)"
    assert stored[0].started_at and stored[0].finished_at


def test_cancelled_job_is_recorded_as_cancelled(qapp, tmp_path):
    history = JobHistory(str(tmp_path / "history.sqlite"))
    queue = PrintQueueManager(history=history)
    gate = threading.Event()
    printer = StubPrinter(gate=gate)
    queue.set_printer_factory(lambda device: printer)
    recorder = Recorder(queue)
    queue.add_card_job(STUB_DEVICE, cards(4))
    wait_until(qapp, printer.entered.is_set)
    queue.stop_current_job()
    gate.set()
    wait_until(qapp, lambda: recorder.finished)
    job = history.list_jobs()[0]
    assert (job.status, job.success_cards, job.total_cards) == ("cancelled", 1, 4)


def test_legacy_jobs_are_recorded_too(qapp, tmp_path, make_image):
    history = JobHistory(str(tmp_path / "history.sqlite"))
    queue = PrintQueueManager(history=history)
    lib = FakeR600Library()
    queue.set_legacy_printer_factory(lambda path, selected: R600Printer.from_library(lib, selected))
    recorder = Recorder(queue)
    queue.add_job(dll_path="x.dll", front_image_path=make_image(), is_dual_side=False, quantity=2,
                  selected_printer=PrinterInfo("RTAI R600 FAKE", "USB"))
    wait_until(qapp, lambda: recorder.finished)
    job = history.list_jobs()[0]
    assert (job.family, job.device_name, job.status, job.success_cards) == (
        "rtai", "RTAI R600 FAKE", "completed", 2)


def test_history_failure_never_breaks_printing(qapp):
    class BrokenHistory:
        def __getattr__(self, name):
            def explode(*args, **kwargs):
                raise OSError("디스크가 가득 찼습니다")
            return explode

    queue = PrintQueueManager(history=BrokenHistory())
    printer = StubPrinter()
    queue.set_printer_factory(lambda device: printer)
    recorder = Recorder(queue)
    queue.add_card_job(STUB_DEVICE, cards(2))
    wait_until(qapp, lambda: recorder.finished)
    assert recorder.finished == [(2, 0, 1, 0, "")]
    assert any("인쇄 이력 기록 실패" in line for line in recorder.logs)


def test_queue_without_history_records_nothing_and_default_manager_is_lazy(qapp):
    from printer.print_queue import print_queue
    assert PrintQueueManager()._ensure_history() is None
    assert print_queue._history is None                # 임포트만으로는 이력 파일을 만들지 않는다
