"""인쇄 큐: 선택한 프린터 사용, 작업 취소·중단, 카드별 CardJob 작업, 이력 기록."""
import threading
import time as real_time

import pytest
from PySide6.QtCore import QEventLoop, QThread

from printer.base import (
    STAGE_UNKNOWN, CardJob, CardPrinter, CardProvider, PrintCancelled, PrinterDevice, PrinterError,
    PrinterStatus, ProgressEvent, SideImages,
)
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
    manager.wait_idle()


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


# ── 공개 API 보강: job_error, history_id, 카드 공급자, cancel_all, 구조화된 진행 ──────────
def test_connection_failure_is_announced_with_job_error_before_completion(qapp, queue):
    def factory(device):
        raise PrinterError("프린터를 열지 못했습니다.", code=0x80000005)

    queue.set_printer_factory(factory)
    recorder = Recorder(queue)
    events = []
    queue.job_error.connect(lambda job_id, message: events.append(("error", job_id, message)))
    queue.job_completed.connect(lambda job_id, ok: events.append(("completed", job_id, ok)))
    job_id = queue.add_card_job(STUB_DEVICE, cards(2))
    wait_until(qapp, lambda: recorder.finished)
    assert events[0][:2] == ("error", job_id)
    assert "프린터 연결 실패: 프린터를 열지 못했습니다." in events[0][2]
    assert events[1] == ("completed", job_id, False)
    assert recorder.cards == []


def test_legacy_init_failure_also_emits_job_error(qapp, queue, make_image):
    def factory(dll_path, selected):
        raise RuntimeError("DLL 없음")

    queue.set_legacy_printer_factory(factory)
    recorder = Recorder(queue)
    errors = []
    queue.job_error.connect(lambda job_id, message: errors.append((job_id, message)))
    job_id = queue.add_job(dll_path="x.dll", front_image_path=make_image(), is_dual_side=False)
    wait_until(qapp, lambda: recorder.finished)
    assert errors and errors[0][0] == job_id and "DLL 없음" in errors[0][1]


def test_history_id_is_public_and_known_when_the_job_starts(qapp, tmp_path):
    history = JobHistory(str(tmp_path / "history.sqlite"))
    queue = PrintQueueManager(history=history)
    queue.set_printer_factory(lambda device: StubPrinter())
    recorder = Recorder(queue)
    at_start = []
    queue.job_started.connect(lambda job_id, index, total: at_start.append(queue.history_id_for(job_id)))
    job_id = queue.add_card_job(STUB_DEVICE, cards(1), name="이력 확인")
    assert queue.get_job(job_id).history_id == queue.history_id_for(job_id) == at_start[0]
    wait_until(qapp, lambda: recorder.finished)
    stored = history.get_job(at_start[0])
    assert stored is not None and stored.name == "이력 확인" and stored.status == "completed"
    assert queue.history_id_for(job_id) == stored.id, "작업이 끝나 큐가 잊은 뒤에도 찾을 수 있다"
    assert queue.get_job(job_id) is None and queue.get_job(999) is None
    assert queue.history_id_for(999) is None


def test_history_id_is_none_without_a_history_store(qapp, queue):
    queue.set_printer_factory(lambda device: StubPrinter())
    recorder = Recorder(queue)
    job_id = queue.add_card_job(STUB_DEVICE, cards(1))
    assert queue.get_job(job_id).history_id is None and queue.history_id_for(job_id) is None
    wait_until(qapp, lambda: recorder.finished)


class LazyCards:
    """카드를 요청받을 때 만드는 공급자. 누가 언제 무엇을 요청했는지 적는다."""

    def __init__(self, count, fail_on=(), cancel_on=None, dual=False):
        self.count = count
        self.fail_on = set(fail_on)
        self.cancel_on = cancel_on
        self.is_dual_side = dual
        self.requested = []
        self.threads = set()

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        self.requested.append(index)
        self.threads.add(threading.get_ident())
        if index + 1 == self.cancel_on:
            raise PrintCancelled()
        if index + 1 in self.fail_on:
            raise PrinterError("카드 이미지를 만들지 못했습니다.")
        return CardJob(front=SideImages(f"card{index + 1}.png"), record_id=101 + index)

    def record_id(self, index):
        return 101 + index


def test_card_provider_is_asked_one_card_at_a_time_on_the_print_thread(qapp, queue):
    gate = threading.Event()
    printer = StubPrinter(gate=gate)
    queue.set_printer_factory(lambda device: printer)
    recorder = Recorder(queue)
    provider = LazyCards(3, dual=True)
    assert isinstance(provider, CardProvider)
    job_id = queue.add_card_job(STUB_DEVICE, provider)
    assert provider.requested == [], "큐에 넣을 때는 카드를 만들지 않는다"
    job = queue.get_job(job_id)
    assert (job.quantity, job.is_dual_side, job.is_card_job) == (3, True, True)

    wait_until(qapp, printer.entered.is_set)
    assert provider.requested == [0], "지금 찍는 카드만 받아 갔다"
    gate.set()
    wait_until(qapp, lambda: recorder.finished)
    assert provider.requested == [0, 1, 2] and printer.printed == [101, 102, 103]
    assert provider.threads == {next(iter(provider.threads))} != {threading.get_ident()}
    assert recorder.finished == [(3, 0, 1, 0, "")]


def test_dual_side_can_be_stated_explicitly_and_plain_lists_are_not_touched_lazily(qapp, queue):
    queue.set_printer_factory(lambda device: StubPrinter())
    recorder = Recorder(queue)
    first = queue.add_card_job(STUB_DEVICE, LazyCards(1), dual_side=True)
    second = queue.add_card_job(STUB_DEVICE, (card for card in cards(2)))      # 제너레이터도 받는다
    assert queue.get_job(first).is_dual_side is True
    assert queue.get_job(second).quantity == 2 and queue.get_job(second).is_dual_side is False
    wait_until(qapp, lambda: recorder.finished)
    assert recorder.finished == [(3, 0, 2, 0, "")]
    with pytest.raises(ValueError):
        queue.add_card_job(STUB_DEVICE, LazyCards(0))


def test_provider_failure_fails_only_that_card_and_keeps_its_record_id(qapp, tmp_path):
    history = JobHistory(str(tmp_path / "history.sqlite"))
    queue = PrintQueueManager(history=history)
    printer = StubPrinter()
    queue.set_printer_factory(lambda device: printer)
    recorder = Recorder(queue)
    queue.add_card_job(STUB_DEVICE, LazyCards(3, fail_on={2}), stop_on_error=False)
    wait_until(qapp, lambda: recorder.finished)
    assert printer.printed == [101, 103]
    assert [(index, ok) for _, index, ok, _ in recorder.cards] == [(1, True), (2, False), (3, True)]
    stored = history.list_cards(history.list_jobs()[0].id)
    assert [(c.card_index, c.status, c.record_id) for c in stored] == [
        (1, "success", "101"), (2, "failed", "102"), (3, "success", "103")]


def test_cancelled_card_is_recorded_as_cancelled_not_failed(qapp, tmp_path):
    history = JobHistory(str(tmp_path / "history.sqlite"))
    queue = PrintQueueManager(history=history)
    printer = StubPrinter()
    queue.set_printer_factory(lambda device: printer)
    recorder = Recorder(queue)
    finished = []
    queue.card_finished.connect(lambda *args: finished.append(args))
    job_id = queue.add_card_job(STUB_DEVICE, LazyCards(4, cancel_on=2))
    wait_until(qapp, lambda: recorder.finished)

    assert printer.printed == [101], "취소된 카드와 그 뒤 카드는 프린터로 가지 않는다"
    assert [(index, status) for _, index, status, _ in finished] == [(1, "success"), (2, "cancelled")]
    assert recorder.cards[1][:3] == (job_id, 2, False)
    assert recorder.finished == [(1, 0, 1, 0, "")], "취소는 실패 장수에도 오류 문구에도 들어가지 않는다"
    job = history.list_jobs()[0]
    assert (job.status, job.success_cards, job.failed_cards, job.error_text) == ("cancelled", 1, 0, None)
    assert [(c.card_index, c.status, c.record_id) for c in history.list_cards(job.id)] == [
        (1, "success", "101"), (2, "cancelled", "102")]


def test_cancel_all_does_not_block_and_announces_every_job(qapp, queue):
    gate = threading.Event()
    printer = StubPrinter(gate=gate)
    queue.set_printer_factory(lambda device: printer)
    recorder = Recorder(queue)
    first = queue.add_card_job(STUB_DEVICE, cards(3))
    second = queue.add_card_job(STUB_DEVICE, cards(1))
    third = queue.add_card_job(STUB_DEVICE, cards(2))
    wait_until(qapp, printer.entered.is_set)

    started = _monotonic()
    queue.cancel_all()                                   # 첫 카드가 아직 프린터 안에 있다
    assert _monotonic() - started < 1.0, "인쇄 스레드를 기다리지 않는다"
    assert recorder.cancelled == [second, third, first]
    assert queue.get_pending_jobs() == [] and recorder.completed == []
    assert queue.get_queue_status()["is_processing"] is True
    assert queue.wait_idle(50) is False, "찍던 카드는 아직 끝나지 않았다"

    gate.set()
    assert queue.wait_idle(5000) is True
    wait_until(qapp, lambda: recorder.finished)
    assert printer.printed == [101] and recorder.completed == [(first, True)]
    assert recorder.started == [first]
    assert queue.wait_idle() is True


def test_cancel_all_with_nothing_queued_is_harmless(queue):
    recorder = Recorder(queue)
    queue.cancel_all()
    assert recorder.cancelled == [] and queue.wait_idle() is True


def test_card_progress_carries_structured_stages_from_a_real_adapter(qapp, queue, make_image):
    lib = FakeR600Library()
    recorder = Recorder(queue)
    events, steps = [], []
    queue.card_progress.connect(lambda job_id, index, event: events.append((job_id, index, event)))
    queue.job_progress.connect(lambda percent, message: steps.append(message))
    device = PrinterDevice("rtai", "RTAI R600 FAKE", "usb", "RTAI R600 FAKE")
    from printer.rtai_adapter import RtaiAdapter

    def factory(target):
        adapter = RtaiAdapter(lib=lib, settle_seconds=0)
        adapter.connect(target)
        return adapter

    queue.set_printer_factory(factory)
    job_id = queue.add_card_job(device, [CardJob(front=SideImages(make_image(), orientation="portrait")),
                                         CardJob(front=SideImages(make_image(), orientation="portrait"))])
    wait_until(qapp, lambda: recorder.finished)
    assert all(isinstance(event, ProgressEvent) and job == job_id for job, _, event in events)
    assert [event.stage for _, index, event in events if index == 1] == [
        "feed", "setup", "front_image", "print", "wait", "eject", "done"]
    assert [event.stage for _, index, event in events if index == 2][-1] == "done"
    assert all(event.panel is None and event.fraction is None for _, _, event in events)
    assert events[0][2].text == "카드 삽입 중..." and events[0][2].percent == 15
    assert any("카드 삽입 중..." in message for message in steps), "예전 진행 시그널도 그대로 나간다"


def test_card_progress_from_a_driver_without_stage_keys_is_stage_unknown(qapp, queue):
    queue.set_printer_factory(lambda device: StubPrinter())
    recorder = Recorder(queue)
    events = []
    queue.card_progress.connect(lambda job_id, index, event: events.append((index, event)))
    queue.add_card_job(STUB_DEVICE, cards(1))
    wait_until(qapp, lambda: recorder.finished)
    assert events == [(1, ProgressEvent(100, "완료"))]
    assert events[0][1].stage == STAGE_UNKNOWN and events[0][1].panel is None
