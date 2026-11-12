"""앱 틀: 위쪽 줄, 왼쪽 메뉴, 3단계 안내, 화면 영역, 아래 줄."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

from PySide6.QtCore import QRect, QSettings, QSize, Qt, Signal
from PySide6.QtGui import QGuiApplication, QKeySequence, QShortcut
from PySide6.QtWidgets import QFrame, QHBoxLayout, QMainWindow, QStackedWidget, QVBoxLayout, QWidget

from ui.theme import (
    KEY_GEOMETRY,
    KEY_GUIDE_VISIBLE,
    KEY_MAXIMIZED,
    SIZES,
    SPACE,
    ThemeManager,
    mark_themed,
    read_bool,
)
from ui.widgets import Button, Toast, label

from .nav_rail import NavRail
from .page import FooterAction
from .step_guide import GuideStrip
from .top_bar import APP_NAME, PrinterPopover, TopBar

log = logging.getLogger(__name__)

_PRINTER_HELP = {
    "ready": "바로 인쇄할 수 있습니다.",
    "busy": "지금 카드를 찍고 있습니다. 끝날 때까지 프린터 덮개를 열지 마세요.",
    "error": "프린터에 문제가 있습니다. 프린터 화면에서 내용을 확인하세요.",
    "offline": "프린터 전원과 USB 선을 확인한 뒤 프린터 화면에서 다시 찾아 주세요.",
}


def fit_geometry(saved: QRect, screens: list[QRect], minimum: QSize) -> QRect:
    """저장해 둔 창 자리를 지금 모니터에 맞춘다.

    창 제목 줄이 어느 모니터에도 걸치지 않으면 첫 모니터 가운데로 옮기고,
    모니터보다 크면 줄이되 최소 크기 아래로는 줄이지 않는다.
    """
    if not screens:
        return QRect(saved)
    title_bar = QRect(saved.x(), saved.y(), saved.width(), 40)
    screen = next((s for s in screens if s.intersects(title_bar)), None)
    on_screen = screen is not None
    screen = screen or screens[0]
    width = max(minimum.width(), min(saved.width(), screen.width()))
    height = max(minimum.height(), min(saved.height(), screen.height()))
    if on_screen:
        x = max(screen.left(), min(saved.x(), screen.right() - width + 1))
        y = max(screen.top(), min(saved.y(), screen.bottom() - height + 1))
    else:
        x = screen.left() + max(0, (screen.width() - width) // 2)
        y = screen.top() + max(0, (screen.height() - height) // 2)
    return QRect(x, y, width, height)


@dataclass
class _Entry:
    page_id: str
    widget: QWidget
    label: str
    is_step: bool


class AppShell(QMainWindow):
    """화면을 올리는 법:

        shell.register_page("design", DesignPage(), "디자인", "design", step_label="카드 꾸미기")
        shell.navigate("design")

    이미 있는 id로 다시 등록하면 그 자리의 화면만 바뀐다(메뉴와 단축키는 그대로).
    """

    pageChanged = Signal(str)
    undoRequested = Signal()
    redoRequested = Signal()
    guideVisibilityChanged = Signal(bool)

    def __init__(self, settings: QSettings | None = None, with_default_pages: bool = True,
                 legacy_factory: Callable[[], QWidget] | None = None, is_admin: bool = False,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self._theme = ThemeManager.instance()
        self._settings = settings if settings is not None else self._theme.settings
        self._entries: dict[str, _Entry] = {}
        self._current: str | None = None
        self._auto_shortcut = 0
        self._shortcuts: list[QShortcut] = []
        self._document_name = "새 카드"
        self._guide_enabled = read_bool(self._settings, KEY_GUIDE_VISIBLE, True)
        self._printer_page_id = "printer"

        self.setObjectName("AppShell")
        self.setWindowTitle(APP_NAME)
        self.setMinimumSize(1100, 700)
        self.resize(1280, 800)

        central = QWidget(self)
        central.setObjectName("PageArea")
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.top_bar = TopBar(central)
        root.addWidget(self.top_bar)
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        root.addLayout(body, 1)

        self.nav = NavRail(central)
        body.addWidget(self.nav)
        column = QVBoxLayout()
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        body.addLayout(column, 1)

        self.guide = GuideStrip(central)
        column.addWidget(self.guide)
        self.stack = QStackedWidget(central)
        column.addWidget(self.stack, 1)

        self.footer = QFrame(central)
        self.footer.setObjectName("PageFooter")
        self.footer.setFixedHeight(SIZES["footer"])
        footer_layout = QHBoxLayout(self.footer)
        footer_layout.setContentsMargins(SPACE["xl"], 0, SPACE["xl"], 0)
        footer_layout.setSpacing(SPACE["md"])
        self._footer_hint = label("", "secondary")
        footer_layout.addWidget(self._footer_hint)
        footer_layout.addStretch(1)
        self._footer_buttons_box = QHBoxLayout()
        self._footer_buttons_box.setSpacing(SPACE["md"])
        footer_layout.addLayout(self._footer_buttons_box)
        self._footer_buttons: list[Button] = []
        column.addWidget(self.footer)

        # 틀 전체가 아니라 부분마다 테마를 입힌다. 기존 화면을 품는 자리는 빼야 하기 때문이다.
        for part in (self.top_bar, self.nav, self.guide, self.footer):
            mark_themed(part)

        self.printer_popover = PrinterPopover(self)
        self.printer_popover.openPrinterPageRequested.connect(lambda: self.navigate(self._printer_page_id))

        self.top_bar.undoRequested.connect(self.undoRequested)
        self.top_bar.redoRequested.connect(self.redoRequested)
        self.top_bar.themeToggleRequested.connect(self._theme.toggle)
        self.top_bar.printerClicked.connect(self.show_printer_popover)
        self.nav.requested.connect(self.navigate)
        self.guide.stepRequested.connect(self.navigate)
        self.guide.hideRequested.connect(self._hide_guide_from_strip)

        self._add_shortcut("Ctrl+Z", self._shortcut_undo)
        self._add_shortcut("Ctrl+Y", self._shortcut_redo)
        self._add_shortcut("Ctrl+Shift+Z", self._shortcut_redo)
        self._add_shortcut("Ctrl+Shift+L", self._theme.toggle)

        self.set_document(self._document_name, None)
        self.set_printer_status("offline", "프린터 연결 안 됨")
        self._restore_geometry()

        if with_default_pages:
            from .pages import install_default_pages

            install_default_pages(self, legacy_factory, is_admin)
            self.navigate("design")

    # ------------------------------------------------------------------ 화면 등록
    def register_page(self, page_id: str, widget: QWidget, label_text: str, icon: str, *,
                      section: str = "main", step_label: str | None = None,
                      shortcut: str | None = None) -> None:
        """화면을 올린다.

        section: "main" | "extra" | "bottom" (왼쪽 메뉴에서의 자리)
        step_label: 주면 3단계 안내의 한 단계가 된다(등록 순서대로 1, 2, 3).
        shortcut: None이면 Ctrl+1부터 차례로 준다. ""이면 단축키 없음.

        화면에 `themed = False`가 있으면 테마를 입히지 않는다(기존 화면을 품는 자리).
        """
        if hasattr(widget, "shell"):
            widget.shell = self
        if getattr(widget, "themed", True):
            mark_themed(widget)
        signal = getattr(widget, "footerChanged", None)
        if signal is not None:
            signal.connect(lambda pid=page_id: self._on_footer_changed(pid))

        existing = self._entries.get(page_id)
        if existing is not None:
            was_current = self._current == page_id
            if was_current:
                self._call(existing.widget, "on_leave")
            index = self.stack.indexOf(existing.widget)
            self.stack.removeWidget(existing.widget)
            existing.widget.deleteLater()
            self.stack.insertWidget(index, widget)
            existing.widget = widget
            if was_current:
                self.stack.setCurrentWidget(widget)
                self._rebuild_footer()
                self._call(widget, "on_enter")
            return

        self.stack.addWidget(widget)
        self._entries[page_id] = _Entry(page_id, widget, label_text, step_label is not None)
        if shortcut is None and section != "bottom":
            self._auto_shortcut += 1
            shortcut = f"Ctrl+{self._auto_shortcut}" if self._auto_shortcut <= 9 else ""
        tooltip = f"{label_text} ({shortcut})" if shortcut else label_text
        self.nav.add_item(page_id, label_text, icon, section, tooltip)
        if shortcut:
            self._add_shortcut(shortcut, lambda pid=page_id: self.navigate(pid))
        if step_label is not None:
            self.guide.add_step(page_id, step_label)

    def page(self, page_id: str) -> QWidget:
        return self._entries[page_id].widget

    def page_ids(self) -> list[str]:
        return list(self._entries)

    def current_page_id(self) -> str | None:
        return self._current

    def navigate(self, page_id: str) -> bool:
        entry = self._entries.get(page_id)
        if entry is None:
            log.warning("없는 화면으로 가려 했습니다: %s", page_id)
            return False
        if page_id == self._current:
            return True
        if self._current is not None:
            self._call(self._entries[self._current].widget, "on_leave")
        self._current = page_id
        self.stack.setCurrentWidget(entry.widget)
        self.nav.set_current(page_id)
        self.guide.set_current(page_id if entry.is_step else None)
        self._sync_guide_visibility()
        self._rebuild_footer()
        self._sync_title()
        self._call(entry.widget, "on_enter")
        self.pageChanged.emit(page_id)
        return True

    def _call(self, widget: QWidget, method: str) -> None:
        hook = getattr(widget, method, None)
        if not callable(hook):
            return
        try:
            hook()
        except Exception:
            log.exception("%s.%s 실행 중 문제가 생겼습니다", type(widget).__name__, method)

    # ------------------------------------------------------------------ 아래 줄
    def _on_footer_changed(self, page_id: str) -> None:
        if page_id == self._current:
            self._rebuild_footer()

    def _rebuild_footer(self) -> None:
        for button in self._footer_buttons:
            self._footer_buttons_box.removeWidget(button)
            button.hide()
            button.setParent(None)
            button.deleteLater()
        self._footer_buttons = []
        widget = self._entries[self._current].widget if self._current else None
        actions: list[FooterAction] = []
        hint = ""
        if widget is not None:
            get_actions = getattr(widget, "footer_actions", None)
            if callable(get_actions):
                actions = list(get_actions() or [])
            get_hint = getattr(widget, "footer_hint", None)
            if callable(get_hint):
                hint = get_hint() or ""
        for action in actions:
            button = Button(action.text, action.variant, "lg")
            button.setEnabled(action.enabled)
            if action.tooltip:
                button.setToolTip(action.tooltip)
            button.clicked.connect(lambda _checked=False, cb=action.callback: cb())
            self._footer_buttons_box.addWidget(button)
            self._footer_buttons.append(button)
        self._footer_hint.setText(hint)
        self.footer.setVisible(bool(actions) or bool(hint))

    def footer_buttons(self) -> list[Button]:
        return list(self._footer_buttons)

    def footer_hint_text(self) -> str:
        return self._footer_hint.text()

    # ------------------------------------------------------------------ 문서, 되돌리기
    def set_document(self, name: str, saved: bool | None = True) -> None:
        """위쪽 줄에 문서 이름과 저장 상태를 보여 준다. saved=None이면 상태를 숨긴다."""
        self._document_name = name
        self.top_bar.set_document(name, saved)
        self._sync_title()

    def set_undo_enabled(self, enabled: bool) -> None:
        self.top_bar.undo_button.setEnabled(enabled)

    def set_redo_enabled(self, enabled: bool) -> None:
        self.top_bar.redo_button.setEnabled(enabled)

    def _shortcut_undo(self) -> None:
        if self.top_bar.undo_button.isEnabled():
            self.undoRequested.emit()

    def _shortcut_redo(self) -> None:
        if self.top_bar.redo_button.isEnabled():
            self.redoRequested.emit()

    def _sync_title(self) -> None:
        parts = [self._document_name] if self._document_name else []
        if self._current:
            parts.append(self._entries[self._current].label)
        parts.append(APP_NAME)
        self.setWindowTitle(" - ".join(parts))

    # ------------------------------------------------------------------ 프린터
    def set_printer_status(self, state: str, text: str, detail: str | None = None) -> None:
        """state: ready | busy | error | offline. text 예: "RTAI R600 준비됨"."""
        self.top_bar.printer_pill.set_status(state, text)
        self.printer_popover.set_status(text, detail if detail is not None else _PRINTER_HELP[state])

    def set_printer_page(self, page_id: str) -> None:
        self._printer_page_id = page_id

    def show_printer_popover(self) -> None:
        self.printer_popover.show_below(self.top_bar.printer_pill)

    # ------------------------------------------------------------------ 3단계 안내
    def guide_visible(self) -> bool:
        return self._guide_enabled

    def set_guide_visible(self, visible: bool, persist: bool = True) -> None:
        visible = bool(visible)
        if persist:
            self._settings.setValue(KEY_GUIDE_VISIBLE, visible)
            self._settings.sync()
        if visible == self._guide_enabled:
            return
        self._guide_enabled = visible
        self._sync_guide_visibility()
        self.guideVisibilityChanged.emit(visible)

    def set_step_done(self, page_id: str, done: bool = True) -> None:
        self.guide.set_done(page_id, done)

    def _sync_guide_visibility(self) -> None:
        entry = self._entries.get(self._current) if self._current else None
        self.guide.setVisible(self._guide_enabled and entry is not None and entry.is_step)

    def _hide_guide_from_strip(self) -> None:
        self.set_guide_visible(False)
        self.show_toast("순서 안내를 숨겼습니다. 설정에서 다시 켤 수 있습니다.",
                        action_text="다시 보기", on_action=lambda: self.set_guide_visible(True))

    # ------------------------------------------------------------------ 알림
    def show_toast(self, text: str, kind: str = "info", action_text: str | None = None,
                   on_action: Callable[[], None] | None = None, timeout_ms: int | None = None) -> Toast:
        offset = 24 + (self.footer.height() if self.footer.isVisible() else 0)
        return Toast.show_message(self.centralWidget(), text, kind, action_text, on_action,
                                  timeout_ms, bottom_offset=offset)

    # ------------------------------------------------------------------ 창
    def _add_shortcut(self, keys: str, slot: Callable[[], None]) -> None:
        shortcut = QShortcut(QKeySequence(keys), self)
        shortcut.setContext(Qt.WindowShortcut)
        shortcut.activated.connect(slot)
        self._shortcuts.append(shortcut)

    def _restore_geometry(self) -> None:
        try:
            x, y, width, height = (int(v) for v in str(self._settings.value(KEY_GEOMETRY, "")).split(","))
        except ValueError:
            return
        screens = [s.availableGeometry() for s in QGuiApplication.screens()]
        self.setGeometry(fit_geometry(QRect(x, y, width, height), screens, self.minimumSize()))
        if read_bool(self._settings, KEY_MAXIMIZED, False):
            self.setWindowState(self.windowState() | Qt.WindowMaximized)

    def _save_geometry(self) -> None:
        rect = self.normalGeometry() if self.isMaximized() else self.geometry()
        if rect.isValid():
            self._settings.setValue(KEY_GEOMETRY, f"{rect.x()},{rect.y()},{rect.width()},{rect.height()}")
        self._settings.setValue(KEY_MAXIMIZED, self.isMaximized())
        self._settings.sync()

    def closeEvent(self, event) -> None:
        for entry in self._entries.values():
            can_close = getattr(entry.widget, "can_close", None)
            if callable(can_close) and not can_close():
                event.ignore()
                return
        if self._current is not None:
            self._call(self._entries[self._current].widget, "on_leave")
        self._save_geometry()
        super().closeEvent(event)
