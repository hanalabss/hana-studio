"""services 테스트 공용 도우미."""
import os
import threading
import time

from PySide6.QtCore import QEventLoop, QThread

from printer.base import CardPrinter, PrinterDevice, PrinterError, PrinterStatus

_monotonic = time.monotonic


def wait_until(app, predicate, timeout=20.0):
    """pytest-qt 없이 이벤트 루프를 돌리며 조건을 기다린다."""
    deadline = _monotonic() + timeout
    while not predicate():
        app.processEvents(QEventLoop.AllEvents, 20)
        QThread.msleep(2)
        assert _monotonic() < deadline, "시간 안에 끝나지 않았습니다"
    app.processEvents(QEventLoop.AllEvents, 20)


def pump(app, milliseconds=60):
    deadline = _monotonic() + milliseconds / 1000.0
    while _monotonic() < deadline:
        app.processEvents(QEventLoop.AllEvents, 10)
        QThread.msleep(2)


class StubPrinter(CardPrinter):
    """흐름만 볼 때 쓰는 가짜 CardPrinter. 카드를 받을 때 이미지 파일이 실제로 있는지도 적는다."""

    family = "stub"

    def __init__(self, fail_on=(), gate_on=None):
        super().__init__()
        self.printed = []
        self.files = []
        self.options = []
        self.fail_on = set(fail_on)
        self.gate_on = gate_on
        self.gate = threading.Event()
        self.entered = threading.Event()
        self.observed = {}
        self.on_card = None

    def capabilities(self):
        return {"status"}

    def connect(self, device):
        self.device = device

    def close(self):
        pass

    @property
    def is_connected(self):
        return True

    def status(self):
        return PrinterStatus(True, "ok")

    def print_card(self, job, progress=None):
        number = len(self.printed) + 1
        job.validate()
        front = job.front
        self.printed.append(job.record_id)
        self.files.append(front.color_path)
        self.options.append(dict(job.options))
        assert os.path.exists(front.color_path)
        if self.on_card is not None:
            self.on_card(number, job)
        self._report(50, "인쇄 실행 중...", progress)
        if self.gate_on == number:
            self.entered.set()
            self.gate.wait(15)
        if number in self.fail_on:
            raise PrinterError("리본을 모두 사용했습니다.", code=0x8000001D, recoverable=True)
        self._report(100, "완료", progress)


STUB_DEVICE = PrinterDevice("stub", "STUB", "fake", "STUB")


class Recorder:
    """PrintService 시그널을 받아 적는다."""

    def __init__(self, service):
        self.queued, self.started, self.progress = [], [], []
        self.cards, self.finished, self.errors = [], [], []
        service.job_queued.connect(self.queued.append)
        service.job_started.connect(self.started.append)
        service.progress.connect(self.progress.append)
        service.card_result.connect(self.cards.append)
        service.finished.connect(self.finished.append)
        service.error.connect(lambda job, message: self.errors.append((job, message)))
