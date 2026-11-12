"""기본 화면들. 디자인·명단·인쇄·기록·프린터는 다음 단계에서 실제 화면으로 바뀐다."""

from __future__ import annotations

from typing import Callable

from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QVBoxLayout, QWidget

from ui.theme import SPACE, ThemeManager
from ui.widgets import EmptyState, FormRow, Panel, SegmentedControl, label

from .legacy import LegacyHostPage
from .page import BasePage, FooterAction

PAGE_DESIGN = "design"
PAGE_LIST = "list"
PAGE_PRINT = "print"
PAGE_HISTORY = "history"
PAGE_QUICK = "quick"
PAGE_PRINTER = "printer"
PAGE_SETTINGS = "settings"


class PlaceholderPage(BasePage):
    """빈 화면 안내 하나로 이루어진 화면."""

    def __init__(self, title: str, heading: str, guidance: str, icon: str,
                 action_text: str | None = None, on_action: Callable[[], None] | None = None,
                 parent: QWidget | None = None):
        super().__init__(title, parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.empty = EmptyState(heading, guidance, action_text, icon)
        if on_action is not None:
            self.empty.actionTriggered.connect(on_action)
        layout.addWidget(self.empty)
        self.enter_count = 0
        self.leave_count = 0

    def on_enter(self) -> None:
        self.enter_count += 1

    def on_leave(self) -> None:
        self.leave_count += 1


class SettingsPage(BasePage):
    def __init__(self, shell, parent: QWidget | None = None):
        super().__init__("설정", parent)
        self._shell = shell
        outer = QHBoxLayout(self)
        outer.setContentsMargins(SPACE["xl"], SPACE["xl"], SPACE["xl"], SPACE["xl"])
        outer.addStretch(1)
        column = QVBoxLayout()
        column.setSpacing(SPACE["lg"])
        outer.addLayout(column, 0)
        outer.addStretch(1)

        column.addWidget(label("설정", "heading"))
        panel = Panel("화면")
        panel.setFixedWidth(600)
        panel.body.setSpacing(SPACE["lg"])
        self.theme_control = SegmentedControl(
            [("system", "Windows 설정 따르기"), ("light", "밝게"), ("dark", "어둡게")],
            current=ThemeManager.instance().mode,
        )
        self.theme_control.currentChanged.connect(self._on_theme_mode)
        panel.body.addWidget(FormRow("화면 색", self.theme_control,
                                     "위쪽 줄의 달 모양 버튼으로도 바로 바꿀 수 있습니다."))
        self.guide_check = QCheckBox("카드 꾸미기, 명단 넣기, 인쇄하기 순서 안내 보기")
        self.guide_check.setChecked(shell.guide_visible())
        self.guide_check.toggled.connect(lambda on: shell.set_guide_visible(on))
        guide_row = QVBoxLayout()
        guide_row.setSpacing(4)
        guide_row.addWidget(self.guide_check)
        hint = label("다음에 할 일을 화면 위쪽에 알려 줍니다. 익숙해지면 꺼도 됩니다.", "hint")
        hint.setContentsMargins(28, 0, 0, 0)
        guide_row.addWidget(hint)
        panel.body.addLayout(guide_row)
        column.addWidget(panel)
        column.addStretch(1)

        ThemeManager.instance().modeChanged.connect(self._sync_mode)
        shell.guideVisibilityChanged.connect(self._sync_guide)

    def _on_theme_mode(self, mode: str) -> None:
        ThemeManager.instance().set_mode(mode)

    def _sync_mode(self, mode: str) -> None:
        self.theme_control.set_current(mode, emit=False)

    def _sync_guide(self, visible: bool) -> None:
        if self.guide_check.isChecked() != visible:
            self.guide_check.blockSignals(True)
            self.guide_check.setChecked(visible)
            self.guide_check.blockSignals(False)


def install_default_pages(shell, legacy_factory=None, is_admin: bool = False) -> None:
    """틀에 기본 화면 일곱 개를 올린다. 나중에 같은 id로 register_page를 부르면 실제 화면으로 바뀐다."""

    def go(page_id: str) -> Callable[[], None]:
        return lambda: shell.navigate(page_id)

    design = PlaceholderPage(
        "디자인",
        "여기서 카드를 꾸밉니다",
        "글자, 사진, 바코드를 놓아 카드를 만드는 화면을 준비하고 있습니다. "
        "완성된 그림 파일이 있으면 빠른 인쇄에서 바로 뽑을 수 있습니다.",
        "design", "빠른 인쇄 열기", go(PAGE_QUICK),
    )
    design.set_footer_hint("카드를 다 꾸미면 명단을 넣습니다.")
    design.set_footer_actions([FooterAction("다음: 명단 넣기", go(PAGE_LIST))])

    people = PlaceholderPage(
        "명단",
        "여기에 명단을 넣습니다",
        "엑셀이나 CSV 파일을 가져오면 사람마다 이름과 사진이 바뀐 카드를 만듭니다. "
        "원본 파일은 고치지 않습니다. 이 화면은 준비하고 있습니다.",
        "people",
    )
    people.set_footer_hint("명단을 넣으면 뽑을 사람을 고르고 인쇄합니다.")
    people.set_footer_actions([
        FooterAction("이전: 카드 꾸미기", go(PAGE_DESIGN), "secondary"),
        FooterAction("다음: 인쇄하기", go(PAGE_PRINT)),
    ])

    printing = PlaceholderPage(
        "인쇄",
        "여기서 인쇄를 시작합니다",
        "뽑을 사람을 고르고 한 장씩 미리 본 뒤 인쇄하는 화면을 준비하고 있습니다. "
        "지금 한 장을 뽑으려면 빠른 인쇄를 쓰세요.",
        "print", "빠른 인쇄 열기", go(PAGE_QUICK),
    )
    printing.set_footer_actions([FooterAction("이전: 명단 넣기", go(PAGE_LIST), "secondary")])

    history = PlaceholderPage(
        "기록",
        "아직 인쇄한 기록이 없습니다",
        "카드를 인쇄하면 언제, 누구 카드를, 몇 장 뽑았는지 여기에 남습니다.",
        "history",
    )

    printer = PlaceholderPage(
        "프린터",
        "연결된 프린터를 여기서 확인합니다",
        "프린터 찾기와 리본 남은 양 확인은 준비하고 있습니다. "
        "지금은 빠른 인쇄 화면에서 프린터를 고를 수 있습니다.",
        "printer", "빠른 인쇄 열기", go(PAGE_QUICK),
    )

    shell.register_page(PAGE_DESIGN, design, "디자인", "design", step_label="카드 꾸미기")
    shell.register_page(PAGE_LIST, people, "명단", "people", step_label="명단 넣기")
    shell.register_page(PAGE_PRINT, printing, "인쇄", "print", step_label="인쇄하기")
    shell.register_page(PAGE_HISTORY, history, "기록", "history")
    shell.register_page(PAGE_QUICK, LegacyHostPage(legacy_factory, is_admin), "빠른 인쇄", "bolt",
                        section="extra")
    shell.register_page(PAGE_PRINTER, printer, "프린터", "printer", section="bottom", shortcut="")
    shell.register_page(PAGE_SETTINGS, SettingsPage(shell), "설정", "settings", section="bottom",
                        shortcut="Ctrl+,")
