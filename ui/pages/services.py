"""화면들이 함께 쓰는 프린터·인쇄·기록 서비스 한 벌."""
from __future__ import annotations

import logging
from typing import Any

from printer.history import JobHistory
from services.print_service import PrintService
from services.printers import PrinterService

log = logging.getLogger(__name__)


class AppServices:
    """앱에 하나만 만든다. 테스트와 시연은 인자로 가짜를 넣는다.

        services = AppServices()
        services.start()        # 프린터 찾기와 상태 확인을 시작한다
        ...
        services.shutdown()     # 앱을 닫을 때
    """

    def __init__(self, printers: PrinterService | None = None, printing: PrintService | None = None,
                 history: JobHistory | None = None, queue: Any = None) -> None:
        self.history = history if history is not None else JobHistory()
        if printing is None:
            printing = PrintService(queue, history=self.history)
        self.printing = printing
        self.printers = printers if printers is not None else PrinterService(queue)
        self._closed = False

    def start(self, retention_days: int | None = None) -> None:
        """끊긴 기록을 닫고, 오래된 기록을 지우고, 프린터를 찾기 시작한다."""
        try:
            self.history.close_interrupted_jobs()
            if retention_days:
                self.history.purge_older_than(int(retention_days))
        except Exception:
            log.exception("인쇄 기록을 정리하지 못했습니다.")
        self.printers.refresh()
        self.printers.start_polling()

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.printing.shutdown()
        self.printers.shutdown()
