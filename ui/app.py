"""새 앱 틀에 화면과 서비스를 올려 메인 창을 만든다."""
from __future__ import annotations

import logging

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

log = logging.getLogger(__name__)


def build_main_window(is_admin: bool = False, *, services=None, start_services: bool = True):
    """메인 창(AppShell)을 만들어 돌려준다. QApplication과 테마 적용은 호출하는 쪽에서 먼저 한다."""
    from ui.pages.design import DesignPage
    from ui.pages.history import HistoryPage
    from ui.pages.people import PeoplePage
    from ui.pages.printer import PrinterPage
    from ui.pages.printing import PrintPage
    from ui.pages.services import AppServices
    from ui.pages.settings import Prefs, SettingsPage
    from ui.pages.workspace import Workspace
    from ui.shell import (PAGE_DESIGN, PAGE_HISTORY, PAGE_LIST, PAGE_PRINT, PAGE_PRINTER,
                          PAGE_SETTINGS, AppShell)

    shell = AppShell(is_admin=is_admin)
    try:
        from ui.theme.brand import app_icon
        icon = app_icon()
        if not icon.isNull():
            shell.setWindowIcon(icon)
            if QApplication.instance() is not None:
                QApplication.instance().setWindowIcon(icon)
    except Exception:
        log.exception("앱 아이콘을 불러오지 못했습니다")
    workspace = Workspace(shell)
    services = services or AppServices()

    def printer_capabilities():
        """지금 고른 프린터가 지원하는 기능. 프린터가 없으면 None(알 수 없음)."""
        return services.printers.capabilities() if services.printers.current() is not None else None

    design = DesignPage(workspace, capabilities=printer_capabilities)
    people = PeoplePage(workspace)
    printing = PrintPage(workspace, services)
    history = HistoryPage(workspace, services)

    def print_another_image() -> None:
        """인쇄가 끝난 뒤 '다른 그림으로 뽑기': 그림을 고르면 그 그림 카드로 바꾸고 인쇄 준비로 돌아간다."""
        if design.start_from_image():
            printing.back_to_ready()
            printing.refresh_now()

    printing.otherImageRequested.connect(print_another_image)

    shell.register_page(PAGE_DESIGN, design, "디자인", "design", step_label="카드 꾸미기")
    shell.register_page(PAGE_LIST, people, "명단", "people", step_label="명단 넣기")
    shell.register_page(PAGE_PRINT, printing, "인쇄", "print", step_label="인쇄하기")
    shell.register_page(PAGE_HISTORY, history, "기록", "history")
    shell.register_page(PAGE_PRINTER, PrinterPage(services), "프린터", "printer")
    shell.register_page(PAGE_SETTINGS, SettingsPage(services=services, is_admin=is_admin), "설정", "settings")

    shell.workspace = workspace
    shell.services = services

    done = {"shutdown": False}

    def shutdown() -> None:
        """서비스를 멈추고 명단 파일을 닫는다. 여러 번 불러도 한 번만 한다."""
        if done["shutdown"]:
            return
        done["shutdown"] = True
        try:
            services.shutdown()
        except Exception:
            log.exception("서비스 종료 실패")
        try:
            workspace.close()
        except Exception:
            log.exception("작업 상태 정리 실패")
        app = QApplication.instance()
        if app is not None:
            try:
                app.aboutToQuit.disconnect(shutdown)
            except (RuntimeError, TypeError):
                pass

    shell.shutdown = shutdown       # 창을 직접 없애는 쪽(테스트, 시연)이 부른다

    app = QApplication.instance()
    if app is not None:
        app.aboutToQuit.connect(shutdown)

    if start_services:
        # 창이 뜬 뒤에 프린터를 찾는다(탐지가 창 표시를 늦추지 않게).
        QTimer.singleShot(300, lambda: services.start(Prefs.instance().retention_days()))

    return shell
