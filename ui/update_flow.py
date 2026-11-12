"""온라인 업데이트 화면 흐름: 확인(뒤에서) → "새 버전이 있습니다" 묻기 → 받기(진행 표시) → 다시 켜기."""
from __future__ import annotations

import logging
from typing import Callable

from PySide6.QtCore import QObject, QThread, QTimer, Signal
from PySide6.QtWidgets import QApplication, QWidget

from services.updates import AvailableUpdate, UpdateError, Updater
from ui.widgets import ProgressBar, confirm, inform, label
from ui.widgets.dialogs import Dialog

log = logging.getLogger(__name__)


class _Worker(QThread):
    done = Signal(object)
    failed = Signal(str)
    progress = Signal(int)

    def __init__(self, job: Callable[["_Worker"], object], parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._job = job

    def run(self) -> None:
        try:
            self.done.emit(self._job(self))
        except UpdateError as exc:
            self.failed.emit(str(exc))
        except Exception:
            log.exception("업데이트 작업 중 오류")
            self.failed.emit("업데이트 중 문제가 생겼습니다. 잠시 뒤 다시 해 주세요.")


class DownloadDialog(Dialog):
    """새 버전을 받는 동안 보이는 창. 닫을 수 없다(받는 중에 끊기지 않게)."""

    def __init__(self, version: str, parent: QWidget | None = None) -> None:
        super().__init__(f"hanaStudio {version}을 받는 중입니다",
                         "다 받으면 프로그램이 저절로 다시 켜집니다. 잠시만 기다려 주세요.", parent, width=460)
        self.bar = ProgressBar()
        self.bar.setRange(0, 100)
        self.body.addWidget(self.bar)
        self.percent = label("0%", "secondary")
        self.body.addWidget(self.percent)

    def set_progress(self, value: int) -> None:
        value = max(0, min(100, int(value)))
        self.bar.setValue(value)
        self.percent.setText(f"{value}%")

    def reject(self) -> None:      # Esc나 닫기 단추로 닫히지 않게
        pass


class UpdateFlow(QObject):
    """check(silent=True)는 켤 때 조용히, check(silent=False)는 설정의 '업데이트 확인'에서 쓴다."""

    finished = Signal(str)        # "none" | "later" | "failed" | "unsupported" | "restarting"

    def __init__(self, parent: QWidget, updater: Updater | None = None,
                 is_busy: Callable[[], bool] = lambda: False) -> None:
        super().__init__(parent)
        self._parent = parent
        self._updater = updater
        self._is_busy = is_busy
        self._worker: _Worker | None = None
        self._dialog: DownloadDialog | None = None
        self._silent = True

    @property
    def updater(self) -> Updater:
        if self._updater is None:
            self._updater = Updater()
        return self._updater

    def is_running(self) -> bool:
        return self._worker is not None

    def check(self, silent: bool = True) -> None:
        if self._worker is not None:
            return
        self._silent = silent
        # 기록 파일(hana_debug.log)에 남긴다: 업데이트가 안 될 때 원인을 찾는 데 쓴다.
        print(f"[UPDATE] 확인 시작: 설치본={self.updater.supported}, 지금 버전={self.updater.current_version() or '-'}, "
              f"받는 곳={self.updater.source}" + (f", 사유={self.updater.reason}" if not self.updater.supported else ""))
        if not self.updater.supported:
            if not silent:
                inform(self._parent, "업데이트를 확인할 수 없습니다",
                       "온라인 업데이트는 설치 파일로 설치한 hanaStudio에서만 됩니다.")
            self.finished.emit("unsupported")
            return
        self._start(lambda worker: self.updater.check(), self._on_checked)

    def _start(self, job: Callable[[_Worker], object], done: Callable[[object], None]) -> _Worker:
        worker = _Worker(job, self)
        worker.done.connect(done)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(lambda: self._forget(worker))
        self._worker = worker
        worker.start()
        return worker

    def _forget(self, worker: _Worker) -> None:
        if self._worker is worker:
            self._worker = None
        worker.deleteLater()

    def _on_failed(self, message: str) -> None:
        if self._dialog is not None:
            self._dialog.accept()
            self._dialog = None
        print(f"[UPDATE] 실패: {message}")
        if not self._silent:
            inform(self._parent, "업데이트하지 못했습니다", message)
        self.finished.emit("failed")

    def _on_checked(self, update: object) -> None:
        print(f"[UPDATE] 확인 결과: {'새 버전 ' + update.version if isinstance(update, AvailableUpdate) else '최신 버전임'}")
        if not isinstance(update, AvailableUpdate):
            if not self._silent:
                inform(self._parent, "최신 버전입니다",
                       f"지금 쓰는 {self.updater.current_version()}이 가장 새 버전입니다.")
            self.finished.emit("none")
            return
        if self._is_busy():
            # 인쇄 중에는 묻지 않는다. 다음에 켤 때 다시 확인한다.
            if not self._silent:
                inform(self._parent, "인쇄가 끝난 뒤에 업데이트해 주세요",
                       f"새 버전 {update.version}이 있습니다. 인쇄 중에는 업데이트할 수 없습니다.")
            self.finished.emit("later")
            return
        size = f" (받을 크기 약 {update.size_text})" if update.size_text else ""
        message = (f"지금 쓰는 버전은 {self.updater.current_version()}입니다.{size}\n"
                   "업데이트하면 프로그램이 다시 켜집니다. 저장하지 않은 작업은 먼저 저장해 주세요.")
        if not confirm(self._parent, f"새 버전 {update.version}이 있습니다", message,
                       confirm_text="지금 업데이트", cancel_text="나중에"):
            self.finished.emit("later")
            return
        self._download(update)

    def _download(self, update: AvailableUpdate) -> None:
        self._silent = False
        self._dialog = DownloadDialog(update.version, self._parent)
        worker = self._start(
            lambda w: (self.updater.download(update, lambda value: w.progress.emit(int(value))), update)[1],
            self._on_downloaded)
        worker.progress.connect(self._dialog.set_progress)
        self._dialog.open()

    def _on_downloaded(self, update: object) -> None:
        if self._dialog is not None:
            self._dialog.set_progress(100)
        try:
            self.updater.apply_and_restart(update)
        except UpdateError as exc:
            self._on_failed(str(exc))
            return
        print("[UPDATE] 새 버전을 받았습니다. 다시 켭니다.")
        self.finished.emit("restarting")
        QTimer.singleShot(200, self._quit)

    @staticmethod
    def _quit() -> None:
        app = QApplication.instance()
        if app is not None:
            app.closeAllWindows()
            app.quit()


def schedule_startup_check(window: QWidget, is_busy: Callable[[], bool] = lambda: False,
                           delay_ms: int = 4000) -> UpdateFlow:
    """메인 창이 뜬 뒤 조금 있다가 조용히 새 버전을 확인한다. 설치본이 아니면 아무 일도 하지 않는다."""
    flow = UpdateFlow(window, is_busy=is_busy)
    QTimer.singleShot(delay_ms, lambda: flow.check(silent=True))
    return flow
