import pytest
from PySide6.QtCore import QRect, QSettings, QSize, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QLabel, QMainWindow, QWidget

from ui.shell import (
    PAGE_DESIGN,
    PAGE_HISTORY,
    PAGE_LIST,
    PAGE_PRINT,
    PAGE_PRINTER,
    PAGE_QUICK,
    PAGE_SETTINGS,
    AppShell,
    BasePage,
    FooterAction,
    LegacyHostPage,
    Page,
    PlaceholderPage,
    fit_geometry,
)
from ui.theme import KEY_GEOMETRY, KEY_GUIDE_VISIBLE, KEY_MAXIMIZED


class FakeLegacy(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setCentralWidget(QLabel("기존 화면"))
        self.allow_close = True
        self.close_requests = 0

    def closeEvent(self, event):
        self.close_requests += 1
        if self.allow_close:
            event.accept()
        else:
            event.ignore()


class Factory:
    def __init__(self):
        self.calls = 0
        self.fail = False
        self.window = None

    def __call__(self):
        self.calls += 1
        if self.fail:
            raise RuntimeError("프린터 DLL을 찾지 못했습니다")
        self.window = FakeLegacy()
        return self.window


@pytest.fixture
def factory():
    return Factory()


@pytest.fixture
def shell(settings, factory, show):
    return show(AppShell(settings=settings, legacy_factory=factory))


def labels(buttons):
    return [b.text() for b in buttons]


# ------------------------------------------------------------------ 구성


def test_shell_has_all_pages_and_starts_on_design(shell):
    assert shell.page_ids() == [PAGE_DESIGN, PAGE_LIST, PAGE_PRINT, PAGE_HISTORY, PAGE_QUICK,
                                PAGE_PRINTER, PAGE_SETTINGS]
    assert shell.current_page_id() == PAGE_DESIGN
    assert shell.nav.page_ids() == shell.page_ids()
    assert shell.nav.current() == PAGE_DESIGN
    assert [shell.nav.button(p).text() for p in shell.page_ids()] == [
        "디자인", "명단", "인쇄", "기록", "빠른 인쇄", "프린터", "설정"]
    assert shell.guide.step_ids() == [PAGE_DESIGN, PAGE_LIST, PAGE_PRINT]
    assert [shell.guide.pill(p).text() for p in shell.guide.step_ids()] == ["카드 꾸미기", "명단 넣기", "인쇄하기"]
    assert shell.top_bar.isVisible() and shell.nav.isVisible() and shell.guide.isVisible()


def test_placeholder_pages_have_helpful_korean_copy(shell):
    for page_id in (PAGE_DESIGN, PAGE_LIST, PAGE_PRINT, PAGE_HISTORY, PAGE_PRINTER):
        page = shell.page(page_id)
        assert isinstance(page, PlaceholderPage)
        assert isinstance(page, Page)
        assert page.title
        assert any("가" <= ch <= "힣" for ch in page.empty.title())
        assert len(page.empty.guidance()) >= 20


def test_rail_bottom_items_sit_below_main_items(shell):
    design = shell.nav.button(PAGE_DESIGN).geometry()
    quick = shell.nav.button(PAGE_QUICK).geometry()
    printer = shell.nav.button(PAGE_PRINTER).geometry()
    config = shell.nav.button(PAGE_SETTINGS).geometry()
    assert design.top() < quick.top() < printer.top() < config.top()
    assert printer.top() - quick.bottom() > 100, "프린터와 설정은 맨 아래에 따로 둔다"


# ------------------------------------------------------------------ 이동


def test_navigate_switches_page_and_calls_hooks(shell, spy):
    changed = spy(shell.pageChanged)
    design, people = shell.page(PAGE_DESIGN), shell.page(PAGE_LIST)
    assert (design.enter_count, design.leave_count) == (1, 0)

    assert shell.navigate(PAGE_LIST) is True
    assert shell.current_page_id() == PAGE_LIST
    assert shell.stack.currentWidget() is people
    assert shell.nav.current() == PAGE_LIST
    assert (design.leave_count, people.enter_count) == (1, 1)
    assert changed.calls == [(PAGE_LIST,)]
    assert "명단" in shell.windowTitle() and "hanaStudio" in shell.windowTitle()

    assert shell.navigate(PAGE_LIST) is True
    assert people.enter_count == 1 and len(changed) == 1, "같은 화면으로 다시 가면 아무 일도 없다"
    assert shell.navigate("nowhere") is False
    assert shell.current_page_id() == PAGE_LIST


def test_rail_click_navigates(shell):
    QTest.mouseClick(shell.nav.button(PAGE_HISTORY), Qt.LeftButton)
    assert shell.current_page_id() == PAGE_HISTORY
    QTest.mouseClick(shell.nav.button(PAGE_SETTINGS), Qt.LeftButton)
    assert shell.current_page_id() == PAGE_SETTINGS


@pytest.mark.parametrize("key,page_id", [
    (Qt.Key_1, PAGE_DESIGN), (Qt.Key_2, PAGE_LIST), (Qt.Key_3, PAGE_PRINT),
    (Qt.Key_4, PAGE_HISTORY), (Qt.Key_5, PAGE_QUICK), (Qt.Key_Comma, PAGE_SETTINGS),
])
def test_keyboard_shortcuts_navigate(shell, qapp, key, page_id):
    shell.navigate(PAGE_PRINTER)
    QTest.keyClick(shell, key, Qt.ControlModifier)
    qapp.processEvents()
    assert shell.current_page_id() == page_id


def test_rail_tooltips_teach_the_shortcuts(shell):
    assert shell.nav.button(PAGE_LIST).toolTip() == "명단 (Ctrl+2)"
    assert shell.nav.button(PAGE_SETTINGS).toolTip() == "설정 (Ctrl+,)"
    assert shell.nav.button(PAGE_PRINTER).toolTip() == "프린터"


def test_page_hook_failure_does_not_break_navigation(shell):
    class Broken(BasePage):
        def on_enter(self):
            raise RuntimeError("boom")

    shell.register_page("broken", Broken("고장"), "고장", "warning")
    assert shell.navigate("broken") is True
    assert shell.navigate(PAGE_DESIGN) is True
    assert shell.current_page_id() == PAGE_DESIGN


# ------------------------------------------------------------------ 3단계 안내


def test_step_states_follow_navigation_and_done_flags(shell):
    guide = shell.guide
    assert guide.states() == {PAGE_DESIGN: "current", PAGE_LIST: "upcoming", PAGE_PRINT: "upcoming"}

    shell.set_step_done(PAGE_DESIGN)
    shell.navigate(PAGE_LIST)
    assert guide.states() == {PAGE_DESIGN: "done", PAGE_LIST: "current", PAGE_PRINT: "upcoming"}
    assert "끝남" in guide.pill(PAGE_DESIGN).accessibleName()
    assert "지금 하는 일" in guide.pill(PAGE_LIST).accessibleName()

    shell.set_step_done(PAGE_LIST)
    shell.navigate(PAGE_PRINT)
    assert guide.states() == {PAGE_DESIGN: "done", PAGE_LIST: "done", PAGE_PRINT: "current"}

    shell.navigate(PAGE_DESIGN)
    assert guide.state(PAGE_DESIGN) == "current", "지금 있는 단계는 끝났어도 '지금'으로 보인다"
    assert guide.state(PAGE_LIST) == "done"

    shell.set_step_done(PAGE_LIST, False)
    assert guide.state(PAGE_LIST) == "upcoming"
    with pytest.raises(KeyError):
        shell.set_step_done(PAGE_HISTORY)


def test_clicking_a_step_navigates(shell):
    QTest.mouseClick(shell.guide.pill(PAGE_PRINT), Qt.LeftButton)
    assert shell.current_page_id() == PAGE_PRINT
    QTest.mouseClick(shell.guide.pill(PAGE_DESIGN), Qt.LeftButton)
    assert shell.current_page_id() == PAGE_DESIGN


def test_guide_is_shown_only_on_step_pages(shell):
    for page_id, visible in ((PAGE_LIST, True), (PAGE_HISTORY, False), (PAGE_QUICK, False),
                             (PAGE_SETTINGS, False), (PAGE_PRINT, True)):
        shell.navigate(page_id)
        assert shell.guide.isVisible() is visible, page_id
    shell.navigate(PAGE_HISTORY)
    assert shell.guide.current() is None


def test_step_states_are_painted_differently(shell, theme_manager):
    from tests.ui.test_ui_widgets import count_pixels

    theme = theme_manager.theme
    shell.set_step_done(PAGE_DESIGN)
    shell.navigate(PAGE_LIST)
    done = shell.guide.pill(PAGE_DESIGN).grab().toImage()
    current = shell.guide.pill(PAGE_LIST).grab().toImage()
    upcoming = shell.guide.pill(PAGE_PRINT).grab().toImage()
    assert count_pixels(done, theme.hex("step_done_bg")) > 300
    assert count_pixels(current, theme.hex("step_current_bg")) > 300
    assert count_pixels(upcoming, theme.hex("step_current_bg")) == 0
    assert count_pixels(upcoming, theme.hex("surface")) > 300


def test_hiding_the_guide_persists_and_can_be_undone(shell, settings, factory, show, spy):
    changed = spy(shell.guideVisibilityChanged)
    QTest.mouseClick(shell.guide._hide, Qt.LeftButton)
    assert not shell.guide.isVisible() and shell.guide_visible() is False
    assert changed.calls == [(False,)]
    settings.sync()
    assert str(settings.value(KEY_GUIDE_VISIBLE)).lower() == "false"

    toast = shell.centralWidget()._hana_toast
    assert toast is not None and "설정" in toast.text()

    reopened = show(AppShell(settings=settings, legacy_factory=factory))
    assert reopened.guide_visible() is False and not reopened.guide.isVisible()
    reopened.close()

    QTest.mouseClick(toast.action_button(), Qt.LeftButton)
    assert shell.guide.isVisible() and shell.guide_visible() is True
    assert changed.last == (True,)


def test_settings_page_controls_guide_and_theme(shell, theme_manager):
    shell.navigate(PAGE_SETTINGS)
    page = shell.page(PAGE_SETTINGS)
    assert page.guide_check.isChecked()
    page.guide_check.setChecked(False)
    assert shell.guide_visible() is False
    shell.set_guide_visible(True)
    assert page.guide_check.isChecked(), "다른 곳에서 바꿔도 설정 화면이 따라온다"

    assert page.theme_control.current() == "light"
    QTest.mouseClick(page.theme_control.button("dark"), Qt.LeftButton)
    assert theme_manager.mode == "dark" and theme_manager.theme_name == "dark"
    theme_manager.set_mode("system")
    assert page.theme_control.current() == "system"


# ------------------------------------------------------------------ 아래 줄


def test_footer_shows_next_step_action(shell, qapp):
    assert shell.footer.isVisible()
    assert labels(shell.footer_buttons()) == ["다음: 명단 넣기"]
    assert shell.footer_buttons()[0].variant() == "primary"
    assert shell.footer_hint_text() == "카드를 다 꾸미면 명단을 넣습니다."

    QTest.mouseClick(shell.footer_buttons()[0], Qt.LeftButton)
    assert shell.current_page_id() == PAGE_LIST
    assert labels(shell.footer_buttons()) == ["이전: 카드 꾸미기", "다음: 인쇄하기"]
    assert [b.variant() for b in shell.footer_buttons()] == ["secondary", "primary"]
    qapp.processEvents()
    visible = [b for b in shell.footer.findChildren(type(shell.footer_buttons()[0])) if b.isVisible()]
    assert len(visible) == 2, "이전 화면의 버튼이 남아 있으면 안 된다"

    QTest.mouseClick(shell.footer_buttons()[0], Qt.LeftButton)
    assert shell.current_page_id() == PAGE_DESIGN

    shell.navigate(PAGE_HISTORY)
    assert not shell.footer.isVisible() and shell.footer_buttons() == []


def test_footer_updates_when_page_changes_its_actions(shell):
    hits = []
    page = BasePage("인쇄")
    shell.register_page("custom", page, "시험", "print")
    shell.navigate("custom")
    assert not shell.footer.isVisible()

    page.set_footer_hint("12명을 뽑습니다.")
    page.set_footer_actions([
        FooterAction("이전: 명단 넣기", lambda: hits.append("back"), "secondary"),
        FooterAction("12장 인쇄 시작", lambda: hits.append("print"), enabled=False, tooltip="프린터를 먼저 연결하세요"),
    ])
    assert shell.footer.isVisible()
    assert labels(shell.footer_buttons()) == ["이전: 명단 넣기", "12장 인쇄 시작"]
    start = shell.footer_buttons()[1]
    assert not start.isEnabled() and start.toolTip() == "프린터를 먼저 연결하세요"
    QTest.mouseClick(start, Qt.LeftButton)
    QTest.mouseClick(shell.footer_buttons()[0], Qt.LeftButton)
    assert hits == ["back"]

    shell.navigate(PAGE_DESIGN)
    page.set_footer_actions([])
    assert labels(shell.footer_buttons()) == ["다음: 명단 넣기"], "보이지 않는 화면의 변경은 아래 줄을 건드리지 않는다"


# ------------------------------------------------------------------ 화면 등록


def test_plain_widget_satisfying_protocol_can_be_registered(shell):
    class Plain(QWidget):
        title = "시험 화면"

        def __init__(self):
            super().__init__()
            self.events = []

        def on_enter(self):
            self.events.append("enter")

        def on_leave(self):
            self.events.append("leave")

    page = Plain()
    assert isinstance(page, Page)
    shell.register_page("plain", page, "시험", "info", shortcut="Ctrl+9")
    assert shell.nav.button("plain").toolTip() == "시험 (Ctrl+9)"
    shell.navigate("plain")
    shell.navigate(PAGE_DESIGN)
    assert page.events == ["enter", "leave"]
    assert not shell.guide.step_ids().count("plain")


def test_registering_same_id_replaces_the_page(shell):
    count = shell.stack.count()
    old = shell.page(PAGE_DESIGN)
    new = PlaceholderPage("디자인", "새 편집 화면", "실제 화면이 들어왔습니다. 여기에 카드를 꾸밉니다.", "design")
    new.set_footer_actions([FooterAction("다음: 명단 넣기", lambda: shell.navigate(PAGE_LIST))])
    shell.register_page(PAGE_DESIGN, new, "디자인", "design", step_label="카드 꾸미기")

    assert shell.page(PAGE_DESIGN) is new and shell.stack.count() == count
    assert shell.stack.currentWidget() is new
    assert old.leave_count == 1 and new.enter_count == 1
    assert new.shell is shell
    assert shell.nav.page_ids().count(PAGE_DESIGN) == 1
    assert shell.guide.step_ids() == [PAGE_DESIGN, PAGE_LIST, PAGE_PRINT]
    assert shell.page_ids()[0] == PAGE_DESIGN
    assert labels(shell.footer_buttons()) == ["다음: 명단 넣기"]


# ------------------------------------------------------------------ 위쪽 줄


def test_document_name_and_saved_state(shell):
    bar = shell.top_bar
    shell.set_document("2026 신입 사원증", True)
    assert bar.document_name() == "2026 신입 사원증" and bar.document_state_text() == "저장됨"
    assert shell.windowTitle().startswith("2026 신입 사원증")
    shell.set_document("2026 신입 사원증", False)
    assert bar.document_state_text() == "저장 안 한 변경이 있습니다"
    shell.set_document("새 카드", None)
    assert bar.document_state_text() == ""


def test_undo_redo_slots(shell, spy, qapp):
    undo, redo = spy(shell.undoRequested), spy(shell.redoRequested)
    assert not shell.top_bar.undo_button.isEnabled() and not shell.top_bar.redo_button.isEnabled()
    QTest.keyClick(shell, Qt.Key_Z, Qt.ControlModifier)
    qapp.processEvents()
    assert len(undo) == 0, "되돌릴 일이 없으면 단축키도 조용하다"

    shell.set_undo_enabled(True)
    shell.set_redo_enabled(True)
    QTest.mouseClick(shell.top_bar.undo_button, Qt.LeftButton)
    QTest.keyClick(shell, Qt.Key_Z, Qt.ControlModifier)
    QTest.mouseClick(shell.top_bar.redo_button, Qt.LeftButton)
    QTest.keyClick(shell, Qt.Key_Y, Qt.ControlModifier)
    qapp.processEvents()
    assert len(undo) == 2 and len(redo) == 2


def test_theme_toggle_button_and_shortcut(shell, theme_manager, qapp):
    button = shell.top_bar.theme_button
    assert "어두운 화면" in button.toolTip() and button.icon_name() == "moon"
    QTest.mouseClick(button, Qt.LeftButton)
    assert theme_manager.theme_name == "dark"
    assert "밝은 화면" in button.toolTip() and button.icon_name() == "sun"
    QTest.keyClick(shell, Qt.Key_L, Qt.ControlModifier | Qt.ShiftModifier)
    qapp.processEvents()
    assert theme_manager.theme_name == "light"


def test_theme_switch_repaints_shell(shell, theme_manager, qapp):
    from tests.ui.test_ui_widgets import count_pixels
    from ui.theme import DARK, LIGHT

    light = shell.grab().toImage()
    theme_manager.set_mode("dark")
    qapp.processEvents()
    dark = shell.grab().toImage()
    assert count_pixels(light.scaled(160, 100), LIGHT.hex("ground")) > 2000
    assert count_pixels(dark.scaled(160, 100), DARK.hex("ground")) > 2000
    assert count_pixels(dark.scaled(160, 100), LIGHT.hex("ground")) == 0


def test_printer_pill_and_popover(shell, qapp):
    pill = shell.top_bar.printer_pill
    assert pill.state() == "offline" and pill.text() == "프린터 연결 안 됨"

    shell.set_printer_status("ready", "RTAI R600 준비됨")
    assert pill.state() == "ready" and pill.text() == "RTAI R600 준비됨"
    assert shell.printer_popover.title() == "RTAI R600 준비됨"
    assert shell.printer_popover.detail() == "바로 인쇄할 수 있습니다."

    shell.set_printer_status("error", "리본을 다 썼습니다", "새 리본으로 바꾼 뒤 덮개를 닫아 주세요.")
    assert shell.printer_popover.detail() == "새 리본으로 바꾼 뒤 덮개를 닫아 주세요."

    QTest.mouseClick(pill, Qt.LeftButton)
    qapp.processEvents()
    assert shell.printer_popover.isVisible()
    QTest.mouseClick(shell.printer_popover.open_button(), Qt.LeftButton)
    assert not shell.printer_popover.isVisible()
    assert shell.current_page_id() == PAGE_PRINTER

    extra = QLabel("리본 남은 양 약 310장")
    shell.printer_popover.set_content(extra)
    assert extra.parent() is not None
    shell.printer_popover.set_content(None)
    assert extra.parent() is None
    with pytest.raises(ValueError):
        shell.set_printer_status("sleeping", "?")


def test_shell_toast_sits_above_footer(shell, qapp):
    toast = shell.show_toast("디자인을 저장했습니다.", timeout_ms=0)
    qapp.processEvents()
    footer_top = shell.footer.mapTo(shell.centralWidget(), shell.footer.rect().topLeft()).y()
    assert toast.isVisible() and toast.geometry().bottom() < footer_top


# ------------------------------------------------------------------ 창 크기


def test_window_geometry_is_persisted(settings, factory, show, tmp_path, qapp):
    first = show(AppShell(settings=settings, legacy_factory=factory))
    assert (first.width(), first.height()) == (1280, 800)
    assert first.minimumWidth() == 1100 and first.minimumHeight() == 700
    first.setGeometry(40, 30, 1360, 840)
    qapp.processEvents()
    first.close()

    reopened = QSettings(str(tmp_path / "ui-test.ini"), QSettings.IniFormat)
    assert reopened.value(KEY_GEOMETRY) == "40,30,1360,840"
    assert str(reopened.value(KEY_MAXIMIZED)).lower() == "false"

    second = show(AppShell(settings=reopened, legacy_factory=factory))
    screens = [s.availableGeometry() for s in qapp.screens()]
    expected = fit_geometry(QRect(40, 30, 1360, 840), screens, QSize(1100, 700))
    # 테스트용 가상 화면은 창 최소 크기보다 작아 위치는 플랫폼이 정한다. 크기만 비교한다.
    assert second.size() == expected.size()
    assert second.size() != QSize(1280, 800), "저장한 값을 읽지 않으면 기본 크기로 열린다"


def test_fit_geometry_keeps_a_window_that_fits():
    screen = QRect(0, 0, 1920, 1040)
    assert fit_geometry(QRect(40, 30, 1360, 840), [screen], QSize(1100, 700)) == QRect(40, 30, 1360, 840)


def test_fit_geometry_brings_lost_window_back():
    """모니터를 뗀 뒤에도 창이 화면 밖에서 열리면 안 된다."""
    screen = QRect(0, 0, 1920, 1040)
    lost = QRect(2600, 200, 1360, 840)
    assert fit_geometry(lost, [screen], QSize(1100, 700)) == QRect(280, 100, 1360, 840)
    second = QRect(1920, 0, 1920, 1040)
    assert fit_geometry(lost, [screen, second], QSize(1100, 700)) == QRect(2480, 200, 1360, 840)


def test_fit_geometry_shrinks_to_screen_but_not_below_minimum():
    assert fit_geometry(QRect(0, 0, 2400, 1400), [QRect(0, 0, 1920, 1040)], QSize(1100, 700)) ==         QRect(0, 0, 1920, 1040)
    assert fit_geometry(QRect(0, 0, 1360, 840), [QRect(0, 0, 800, 600)], QSize(1100, 700)).size() ==         QSize(1100, 700)
    assert fit_geometry(QRect(5, 5, 1360, 840), [], QSize(1100, 700)) == QRect(5, 5, 1360, 840)


def test_bad_stored_geometry_is_ignored(settings, factory, show):
    settings.setValue(KEY_GEOMETRY, "not,a,rect")
    shell = show(AppShell(settings=settings, legacy_factory=factory))
    assert (shell.width(), shell.height()) == (1280, 800)


# ------------------------------------------------------------------ 빠른 인쇄(기존 화면)


def test_legacy_screen_is_created_lazily_and_embedded(shell, factory, qapp):
    host = shell.page(PAGE_QUICK)
    assert isinstance(host, LegacyHostPage) and host.title == "빠른 인쇄"
    assert factory.calls == 0 and not host.is_loaded()

    shell.navigate(PAGE_QUICK)
    QTest.qWait(30)
    assert factory.calls == 1 and host.is_loaded()
    window = host.legacy_window()
    assert window is factory.window
    assert not window.isWindow(), "기존 창은 따로 뜨지 않고 틀 안에 들어온다"
    assert window.isVisible() and host.isAncestorOf(window)

    shell.navigate(PAGE_DESIGN)
    shell.navigate(PAGE_QUICK)
    QTest.qWait(30)
    assert factory.calls == 1, "한 번 만든 화면은 다시 만들지 않는다"


def test_legacy_failure_is_contained_and_retryable(shell, factory):
    factory.fail = True
    host = shell.page(PAGE_QUICK)
    shell.navigate(PAGE_QUICK)
    QTest.qWait(30)
    assert not host.is_loaded()
    assert "프린터 DLL을 찾지 못했습니다" in host.error()
    assert host._failed.isVisible() and "프린터 DLL을 찾지 못했습니다" in host._failed.guidance()

    assert shell.navigate(PAGE_LIST) is True, "기존 화면이 실패해도 새 화면은 계속 쓸 수 있다"
    shell.navigate(PAGE_QUICK)
    factory.fail = False
    QTest.mouseClick(host._failed.action_button(), Qt.LeftButton)
    assert host.is_loaded() and host.error() is None
    assert factory.calls >= 2


def test_legacy_screen_is_not_created_twice_when_it_pumps_events(settings, show, qapp):
    """기존 화면은 만드는 도중 processEvents를 부른다. 그 틈에 한 번 더 만들어지면 안 된다."""
    made = []

    def slow_factory():
        qapp.processEvents()
        window = FakeLegacy()
        made.append(window)
        qapp.processEvents()
        return window

    shell = show(AppShell(settings=settings, legacy_factory=slow_factory))
    shell.navigate(PAGE_QUICK)
    host = shell.page(PAGE_QUICK)
    assert host.load_now() is True
    QTest.qWait(30)
    assert len(made) == 1
    assert host.legacy_window() is made[0]


def test_legacy_factory_returning_nothing_counts_as_failure(settings, show):
    shell = show(AppShell(settings=settings, legacy_factory=lambda: None))
    host = shell.page(PAGE_QUICK)
    assert host.load_now() is False
    assert host.error()


def test_close_asks_legacy_screen_first(shell, factory, qapp):
    host = shell.page(PAGE_QUICK)
    assert host.load_now() is True
    factory.window.allow_close = False
    assert shell.close() is False
    assert shell.isVisible(), "기존 화면이 닫기를 거부하면(인쇄 중) 앱도 닫히지 않는다"
    assert factory.window.close_requests == 1

    factory.window.allow_close = True
    assert shell.close() is True
    assert not shell.isVisible()


def test_close_without_legacy_loaded_does_not_create_it(shell, factory):
    assert shell.close() is True
    assert factory.calls == 0
    assert shell.page(PAGE_DESIGN).leave_count == 1


def test_default_legacy_factory_targets_existing_class():
    import inspect

    from ui.shell import default_legacy_factory

    source = inspect.getsource(default_legacy_factory)
    assert "from hana_studio import HanaStudio" in source
    assert "initialize_panels_deferred" in source and "initialize_viewers_deferred" in source


# ------------------------------------------------------------------ 기존 화면은 새 테마를 입지 않는다


class StyledLegacy(FakeLegacy):
    """기존 화면처럼 작은 버튼과 숫자 칸을 스스로 꾸민다."""

    def __init__(self):
        super().__init__()
        from PySide6.QtWidgets import QHBoxLayout, QPushButton, QSpinBox

        body = QWidget()
        row = QHBoxLayout(body)
        self.minus = QPushButton("-")
        self.minus.setFixedSize(24, 24)
        self.minus.setStyleSheet("background: #DC3545; color: white; border: none; border-radius: 4px;")
        self.spin = QSpinBox()
        self.plain = QPushButton("카드 인쇄")
        for widget in (self.minus, self.spin, self.plain):
            row.addWidget(widget)
        self.setCentralWidget(body)


def test_new_theme_does_not_leak_into_legacy_screen(settings, show, theme_manager, qapp):
    from PySide6.QtGui import QPalette
    from PySide6.QtWidgets import QPushButton, QSpinBox

    from ui.theme import DARK, LIGHT, mark_themed

    theme_manager.set_mode("dark")
    made = []

    def factory():
        made.append(StyledLegacy())
        return made[-1]

    shell = show(AppShell(settings=settings, legacy_factory=factory))
    shell.navigate(PAGE_QUICK)
    host = shell.page(PAGE_QUICK)
    assert host.load_now() is True
    qapp.processEvents()
    legacy = made[0]

    reference = show(mark_themed(QWidget()))
    themed_button = QPushButton("카드 인쇄", reference)
    themed_spin = QSpinBox(reference)
    themed_button.show()
    themed_spin.show()
    qapp.processEvents()

    assert (legacy.minus.width(), legacy.minus.height()) == (24, 24), "기존 화면의 작은 버튼이 그대로여야 한다"
    assert themed_button.sizeHint().height() >= 40
    assert legacy.plain.sizeHint().height() < themed_button.sizeHint().height()
    assert legacy.spin.sizeHint().height() < themed_spin.sizeHint().height()

    assert legacy.palette().color(QPalette.Window).name().upper() == LIGHT.hex("ground")
    assert legacy.plain.palette().color(QPalette.ButtonText).name().upper() == LIGHT.hex("text")
    from PySide6.QtGui import QFontDatabase

    system_font = QFontDatabase.systemFont(QFontDatabase.GeneralFont)
    assert legacy.plain.font().family() == system_font.family(), "기존 화면은 원래 쓰던 시스템 글꼴 그대로"
    assert themed_button.font().family() == theme_manager.font_spec.family
    assert shell.palette().color(QPalette.Window).name().upper() == DARK.hex("ground")

    from tests.ui.test_ui_widgets import count_pixels

    assert count_pixels(legacy.plain.grab().toImage(), DARK.hex("surface_raised")) == 0
    assert count_pixels(themed_button.grab().toImage(), DARK.hex("surface_raised")) > 200


def test_shell_parts_and_pages_are_themed_but_shell_itself_is_not(shell):
    assert shell.property("themed") is None, "틀 전체에 입히면 기존 화면까지 물든다"
    for part in (shell.top_bar, shell.nav, shell.guide, shell.footer, shell.printer_popover):
        assert part.property("themed") == "true"
    for page_id in (PAGE_DESIGN, PAGE_LIST, PAGE_PRINT, PAGE_HISTORY, PAGE_PRINTER, PAGE_SETTINGS):
        assert shell.page(page_id).property("themed") == "true"
    quick = shell.page(PAGE_QUICK)
    assert quick.property("themed") is None
    assert quick._waiting.property("themed") == "true" and quick._failed.property("themed") == "true"


def test_windows_opened_by_legacy_screen_stay_light(settings, show, theme_manager, qapp):
    from PySide6.QtGui import QPalette
    from PySide6.QtWidgets import QDialog, QPushButton, QVBoxLayout

    from ui.theme import DARK, LIGHT
    from ui.widgets import Dialog

    theme_manager.set_mode("dark")
    made = []
    shell = show(AppShell(settings=settings, legacy_factory=lambda: made.append(FakeLegacy()) or made[-1]))
    assert shell.page(PAGE_QUICK).load_now() is True

    old_dialog = QDialog(made[0])
    QVBoxLayout(old_dialog).addWidget(QPushButton("프린터 선택"))
    old_dialog.show()
    qapp.processEvents()
    assert old_dialog.palette().color(QPalette.Window).name().upper() == LIGHT.hex("ground")
    assert old_dialog.findChild(QPushButton).palette().color(QPalette.ButtonText).name().upper() == LIGHT.hex("text")

    new_dialog = Dialog("새 대화 상자", parent=shell)
    new_dialog.show()
    qapp.processEvents()
    assert new_dialog.palette().color(QPalette.Window).name().upper() == DARK.hex("surface")
    assert new_dialog.palette().color(QPalette.WindowText).name().upper() == DARK.hex("text")
