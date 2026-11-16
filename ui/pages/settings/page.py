"""설정 화면: 화면 색, 순서 안내, 새 카드 크기, 인쇄 기본값, 기록 보관, 파일 위치, 프로그램 정보."""
from __future__ import annotations

import logging
import os
import sys

from PySide6.QtCore import QUrl, Qt
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QCheckBox, QFileDialog, QFrame, QHBoxLayout, QScrollArea, QVBoxLayout, QWidget

from design import presets
from printer.history import default_history_path
from services.printers import app_data_dir
from ui.pages.printing.style import apply_page_style
from ui.shell import BasePage
from ui.theme import SPACE, ThemeManager
from ui.widgets import (
    ComboBox, FormRow, MmField, Panel, PrimaryButton, QuietButton, SecondaryButton, SegmentedControl, StepperField,
    inform, label,
)

from .prefs import COPIES_MAX, OFFSET_LIMIT_MM, RETENTION_CHOICES, Prefs

log = logging.getLogger(__name__)

CONTENT_WIDTH = 680
_LABEL_WIDTH = 150


def version_lines() -> list[tuple[str, str]]:
    """(이름, 값) 목록: 프로그램, 프린터 프로그램 파일, 바탕 기술."""
    lines: list[tuple[str, str]] = []
    try:
        from config import AppConstants

        lines.append(("hanaStudio", AppConstants.APP_VERSION))
    except Exception:
        lines.append(("hanaStudio", "알 수 없음"))
    try:
        from printer.printer_utils import get_default_dll_paths

        rtai = next((path for path in get_default_dll_paths() if os.path.isfile(path)), None)
    except Exception:
        rtai = None
    try:
        from printer.smart_printer import find_smart_dll

        smart = find_smart_dll()
    except Exception:
        smart = None
    lines.append(("RTAI 프린터용 파일", os.path.basename(rtai) + " 있음" if rtai else "없음 (RTAI 프린터를 쓸 수 없습니다)"))
    lines.append(("SMART 프린터용 파일", os.path.basename(smart) + " 있음" if smart else "없음 (SMART 프린터를 쓸 수 없습니다)"))
    try:
        import PySide6

        lines.append(("화면 기술", f"Qt {PySide6.__version__}, Python {sys.version.split()[0]}"))
    except Exception:
        pass
    return lines


class SettingsPage(BasePage):
    def __init__(self, shell=None, services=None, parent: QWidget | None = None, prefs: Prefs | None = None,
                 is_admin: bool = False) -> None:
        """shell: 순서 안내를 켜고 끌 틀(없으면 틀에 올라갈 때 잇는다). services: 기록 정리에 쓴다."""
        super().__init__("설정", parent)
        self.prefs = prefs or Prefs.instance()
        self.services = services
        self.is_admin = bool(is_admin)
        self._bound_shell = None
        self.open_folder = lambda path: QDesktopServices.openUrl(QUrl.fromLocalFile(path))

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        body = QWidget()
        body.setObjectName("ScrollBody")
        outer = QHBoxLayout(body)
        outer.setContentsMargins(SPACE["xl"], SPACE["xl"], SPACE["xl"], SPACE["xl"])
        outer.addStretch(1)
        host = QWidget()
        host.setFixedWidth(CONTENT_WIDTH)
        column = QVBoxLayout(host)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(SPACE["lg"])
        outer.addWidget(host)
        outer.addStretch(1)
        scroll.setWidget(body)
        root.addWidget(scroll)

        column.addWidget(label("설정", "heading"))
        column.addWidget(self._build_screen())
        column.addWidget(self._build_card())
        column.addWidget(self._build_print())
        auto_print = getattr(services, "auto_print", None)
        self.auto_panel = self._build_auto_print(auto_print) if auto_print is not None else None
        if self.auto_panel is not None:
            column.addWidget(self.auto_panel)
        operators = getattr(services, "operators", None)
        self.operator_panel = self._build_operators(operators) if operators is not None else None
        if self.operator_panel is not None:
            column.addWidget(self.operator_panel)
        column.addWidget(self._build_history())
        column.addWidget(self._build_files())
        if self.is_admin:
            column.addWidget(self._build_admin())
        column.addWidget(self._build_about())
        column.addStretch(1)
        apply_page_style(self)

        ThemeManager.instance().modeChanged.connect(self._sync_mode)
        self.prefs.changed.connect(self._load)
        self._load()
        if shell is not None:
            self.shell = shell

    # ------------------------------------------------------------------ 틀
    @property
    def shell(self):
        return self._shell

    @shell.setter
    def shell(self, value) -> None:
        self._shell = value
        if value is None or value is getattr(self, "_bound_shell", None) or not hasattr(self, "guide_check"):
            return
        self._bound_shell = value
        value.guideVisibilityChanged.connect(self._sync_guide)
        self._sync_guide(value.guide_visible())
        self.guide_check.setEnabled(True)

    def on_enter(self) -> None:
        self._load()

    # ------------------------------------------------------------------ 구역
    def _build_screen(self) -> Panel:
        panel = Panel("화면")
        panel.body.setSpacing(SPACE["lg"])
        self.theme_control = SegmentedControl(
            [("system", "시스템 따라가기"), ("light", "밝게"), ("dark", "어둡게")],
            current=ThemeManager.instance().mode)
        self.theme_control.currentChanged.connect(lambda mode: ThemeManager.instance().set_mode(mode))
        self.theme_control.setAccessibleName("화면 색")
        panel.body.addWidget(FormRow("화면 색", self.theme_control,
                                     "'시스템 따라가기'는 Windows 설정의 밝게·어둡게를 따릅니다.",
                                     orientation="horizontal", label_width=_LABEL_WIDTH, control_align="left"))
        from ui import i18n
        self.language_combo = ComboBox([name for _code, name in i18n.LANGUAGES])
        self.language_combo.setAccessibleName("언어 (Language)")
        codes = [code for code, _name in i18n.LANGUAGES]
        self.language_combo.setCurrentIndex(codes.index(i18n.stored_language(self.prefs.settings)))
        self.language_combo.currentIndexChanged.connect(self._on_language)
        panel.body.addWidget(FormRow("언어 (Language)", self.language_combo,
                                     "메뉴·버튼 같은 화면 글자를 바꿉니다. 프로그램을 다시 시작하면 적용됩니다.",
                                     orientation="horizontal", label_width=_LABEL_WIDTH, control_align="left"))
        self.guide_check = QCheckBox("카드 꾸미기, 명단 넣기, 인쇄하기 순서 안내 보기")
        self.guide_check.setEnabled(False)
        self.guide_check.toggled.connect(self._on_guide_toggled)
        guide = QVBoxLayout()
        guide.setSpacing(4)
        guide.addWidget(self.guide_check)
        hint = label("다음에 할 일을 화면 위쪽에 알려 줍니다. 익숙해지면 꺼도 됩니다.", "hint")
        hint.setContentsMargins(28, 0, 0, 0)
        guide.addWidget(hint)
        panel.body.addLayout(guide)
        return panel

    def _on_language(self, index: int) -> None:
        from ui import i18n
        code = i18n.LANGUAGES[index][0]
        if code == i18n.stored_language(self.prefs.settings):
            return
        i18n.save_language(code, self.prefs.settings)
        self._toast_restart()

    def _toast_restart(self) -> None:
        text = "언어를 바꿨습니다. 프로그램을 다시 시작하면 적용됩니다. (Restart to apply)"
        shell = self.window()
        if hasattr(shell, "show_toast"):
            shell.show_toast(text, "info")

    def _build_card(self) -> Panel:
        panel = Panel("새 카드")
        self.card_combo = ComboBox([preset.label for preset in presets.PRESETS])
        self.card_combo.currentIndexChanged.connect(
            lambda index: self.prefs.set_card_preset_key(presets.PRESETS[index].key))
        panel.body.addWidget(FormRow("기본 카드 크기", self.card_combo,
                                     "새 디자인을 만들 때 처음 고르는 크기입니다. 이미 만든 디자인은 바뀌지 않습니다.",
                                     orientation="horizontal", label_width=_LABEL_WIDTH))
        return panel

    def _build_print(self) -> Panel:
        panel = Panel("인쇄 기본값", "인쇄 화면을 열 때 처음 들어가 있는 값입니다. 인쇄할 때마다 바꿀 수 있습니다.")
        panel.body.setSpacing(SPACE["lg"])
        self.copies = StepperField(1, 1, COPIES_MAX, 1, 0, "장")
        self.copies.setFixedWidth(200)
        self.copies.valueChanged.connect(lambda value: self.prefs.set_copies(int(value)))
        self.copies.setAccessibleName("한 사람에 몇 장")
        panel.body.addWidget(FormRow("한 사람에 몇 장", self.copies, orientation="horizontal",
                                     label_width=_LABEL_WIDTH, control_align="left"))
        self.stop_check = QCheckBox("한 장이 실패하면 남은 카드는 찍지 않고 멈추기")
        self.stop_check.toggled.connect(self.prefs.set_stop_on_error)
        stop = QVBoxLayout()
        stop.setSpacing(4)
        stop.addWidget(self.stop_check)
        stop_hint = label("리본이 떨어지거나 카드가 걸렸을 때 카드를 여러 장 버리지 않도록 켜 두는 편이 안전합니다.", "hint", wrap=True)
        stop_hint.setContentsMargins(28, 0, 0, 0)
        stop.addWidget(stop_hint)
        panel.body.addLayout(stop)

        panel.body.addWidget(label("찍히는 위치 맞추기", "strong"))
        self.offset_note = label(
            "카드에 찍힌 그림이 한쪽으로 치우쳐 나올 때만 바꾸세요. 오른쪽이나 아래쪽으로 옮기려면 + 값을, "
            "왼쪽이나 위쪽으로 옮기려면 − 값을 넣습니다. 프린터 화면의 '시험 인쇄 1장'으로 확인할 수 있습니다.",
            "muted", wrap=True)
        panel.body.addWidget(self.offset_note)
        position = QHBoxLayout()
        position.setSpacing(SPACE["lg"])
        self.offset_x = MmField(0.0, -OFFSET_LIMIT_MM, OFFSET_LIMIT_MM, 0.1, 1, follow_units=False)
        self.offset_y = MmField(0.0, -OFFSET_LIMIT_MM, OFFSET_LIMIT_MM, 0.1, 1, follow_units=False)
        self.offset_x.valueChanged.connect(lambda _v: self._save_offsets())
        self.offset_y.valueChanged.connect(lambda _v: self._save_offsets())
        position.addWidget(FormRow("가로 (오른쪽 +)", self.offset_x))
        position.addWidget(FormRow("세로 (아래쪽 +)", self.offset_y))
        self.offset_reset = QuietButton("0으로 되돌리기", accent=True)
        self.offset_reset.clicked.connect(lambda: self.prefs.set_offsets(0.0, 0.0))
        position.addWidget(self.offset_reset, 0, Qt.AlignBottom)
        position.addStretch(1)
        panel.body.addLayout(position)
        self.migrated_note = label("", "hint", wrap=True)
        panel.body.addWidget(self.migrated_note)
        return panel

    # ---- 작업자 ----
    def _build_operators(self, book) -> Panel:
        self.operators = book
        panel = Panel("작업자", "작업자를 더하면 프로그램을 켤 때 누가 쓰는지 확인하고, 인쇄 기록에 작업한 사람을 남깁니다.")
        panel.body.setSpacing(SPACE["md"])
        self.operator_current = label("", "secondary", wrap=True)
        panel.body.addWidget(self.operator_current)
        buttons = QHBoxLayout()
        buttons.setSpacing(SPACE["sm"])
        self.operator_switch = SecondaryButton("작업자 바꾸기", "sm")
        self.operator_switch.clicked.connect(self.switch_operator)
        self.operator_add = SecondaryButton("작업자 더하기", "sm", icon="plus")
        self.operator_add.clicked.connect(self.add_operator)
        buttons.addWidget(self.operator_switch)
        buttons.addWidget(self.operator_add)
        buttons.addStretch(1)
        panel.body.addLayout(buttons)
        self.operator_rows = QVBoxLayout()
        self.operator_rows.setSpacing(2)
        panel.body.addLayout(self.operator_rows)
        self.operator_note = label("비밀번호를 모두 잊었으면 '파일이 있는 곳 > 설정과 인쇄 기록' 폴더의 "
                                   "operators.json을 지우세요. 작업자 확인 없이 시작합니다.", "hint", wrap=True)
        panel.body.addWidget(self.operator_note)
        book.changed.connect(self._sync_operators)
        book.currentChanged.connect(lambda _name: self._sync_operators())
        self._sync_operators()
        return panel

    def _sync_operators(self) -> None:
        book = self.operators
        current = book.current_operator()
        if not book.enabled:
            text = "작업자 확인을 쓰지 않습니다."
        elif current is None:
            text = "지금 작업자가 정해지지 않았습니다. '작업자 바꾸기'로 고르세요."
        else:
            text = f"지금 작업자: {current.name}" + (" (관리자)" if current.admin else "")
        self.operator_current.setText(text)
        manage = book.can_manage()
        self.operator_switch.setVisible(book.enabled)
        self.operator_add.setEnabled(manage)
        self.operator_add.setToolTip("" if manage else "관리자 작업자만 더할 수 있습니다.")
        self.operator_note.setVisible(book.enabled)
        while self.operator_rows.count():
            item = self.operator_rows.takeAt(0)
            if item.layout() is not None:
                while item.layout().count():
                    child = item.layout().takeAt(0)
                    if child.widget() is not None:
                        child.widget().deleteLater()
            elif item.widget() is not None:
                item.widget().deleteLater()
        self.operator_remove_buttons = {}
        for operator in book.operators():
            row = QHBoxLayout()
            row.setSpacing(SPACE["sm"])
            name = label(operator.name + ("  · 관리자" if operator.admin else ""), "strong")
            name.setProperty("i18n_skip", True)
            row.addWidget(name, 1)
            remove = QuietButton("지우기", "sm", icon="trash")
            remove.setEnabled(manage)
            remove.clicked.connect(lambda _checked=False, n=operator.name: self.remove_operator(n))
            row.addWidget(remove)
            self.operator_rows.addLayout(row)
            self.operator_remove_buttons[operator.name] = remove

    def switch_operator(self) -> bool:
        from ui.operators import SignInDialog

        return SignInDialog(self.operators, self, closing_text="그만두기").exec() == SignInDialog.Accepted

    def add_operator(self) -> bool:
        from ui.operators import AddOperatorDialog

        return AddOperatorDialog(self.operators, self).exec() == AddOperatorDialog.Accepted

    def remove_operator(self, name: str, ask: bool = True) -> bool:
        from services.operators import OperatorError
        from ui.widgets import confirm

        if ask and not confirm(self, f"'{name}' 작업자를 지울까요?",
                               "지운 작업자로는 들어올 수 없습니다. 지난 인쇄 기록의 이름은 남습니다.",
                               "지우기", danger=True):
            return False
        try:
            self.operators.remove(name)
        except OperatorError as exc:
            inform(self, "작업자를 지울 수 없습니다", str(exc))
            return False
        return True

    # ---- 폴더 자동 인쇄 ----
    def _build_auto_print(self, auto) -> Panel:
        self.auto = auto
        panel = Panel("폴더 자동 인쇄", "정해 둔 폴더에 명단 파일(CSV·Excel)을 넣으면 고른 디자인으로 바로 인쇄합니다.")
        panel.body.setSpacing(SPACE["md"])
        self.auto_folder_label = label("", "muted", wrap=True)
        self.auto_folder_button = SecondaryButton("폴더 고르기", "sm", icon="folder")
        self.auto_folder_button.clicked.connect(self.choose_auto_folder)
        self.auto_design_label = label("", "muted", wrap=True)
        self.auto_design_button = SecondaryButton("디자인 고르기", "sm", icon="design")
        self.auto_design_button.clicked.connect(self.choose_auto_design)
        for title, path_label, button in (("감시할 폴더", self.auto_folder_label, self.auto_folder_button),
                                          ("인쇄할 디자인", self.auto_design_label, self.auto_design_button)):
            row = QHBoxLayout()
            row.setSpacing(SPACE["md"])
            texts = QVBoxLayout()
            texts.setSpacing(2)
            texts.addWidget(label(title, "strong"))
            texts.addWidget(path_label)
            row.addLayout(texts, 1)
            row.addWidget(button, 0, Qt.AlignTop)
            panel.body.addLayout(row)
        control = QHBoxLayout()
        control.setSpacing(SPACE["md"])
        self.auto_toggle = PrimaryButton("자동 인쇄 켜기", "sm", icon="print")
        self.auto_toggle.clicked.connect(self.toggle_auto_print)
        control.addWidget(self.auto_toggle)
        self.auto_status = label(auto.status, "secondary", wrap=True)
        control.addWidget(self.auto_status, 1)
        panel.body.addLayout(control)
        panel.body.addWidget(label(
            "파일 첫 줄에는 디자인의 칸 이름(이름, 사번 …)을 적습니다. 사진은 같은 폴더에서 찾습니다. "
            "켜기 전에 폴더에 있던 파일은 인쇄하지 않고, 끝난 파일은 '인쇄함', 문제가 생긴 파일은 '실패' 폴더로 옮깁니다. "
            "프로그램을 다시 켜면 자동 인쇄는 꺼진 채로 시작합니다.", "hint", wrap=True))
        self.auto_results = QVBoxLayout()
        self.auto_results.setSpacing(2)
        panel.body.addLayout(self.auto_results)
        auto.statusChanged.connect(self.auto_status.setText)
        auto.runningChanged.connect(lambda _on: self._sync_auto())
        auto.fileFinished.connect(lambda _event: self._sync_auto())
        self._sync_auto()
        return panel

    def _sync_auto(self) -> None:
        auto = self.auto
        folder, design = self.prefs.auto_print_folder(), self.prefs.auto_print_design()
        self.auto_folder_label.setText(folder or "아직 고르지 않았습니다.")
        self.auto_design_label.setText(design or "아직 고르지 않았습니다.")
        self.auto_toggle.setText("자동 인쇄 끄기" if auto.running else "자동 인쇄 켜기")
        self.auto_toggle.setEnabled(auto.running or bool(folder and design))
        self.auto_folder_button.setEnabled(not auto.running)
        self.auto_design_button.setEnabled(not auto.running)
        self.auto_status.setText(auto.status)
        while self.auto_results.count():
            item = self.auto_results.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        for event in auto.history[:5]:
            mark = "✓" if event.ok else "✕"
            self.auto_results.addWidget(label(f"{mark} {event.name} — {event.message}",
                                              "secondary" if event.ok else "hint", wrap=True))

    def choose_auto_folder(self, path: str | None = None) -> None:
        if path is None:
            path = QFileDialog.getExistingDirectory(self, "자동 인쇄할 폴더 고르기", self.prefs.auto_print_folder())
        if path:
            self.prefs.set_auto_print_folder(path)
            self._sync_auto()

    def choose_auto_design(self, path: str | None = None) -> None:
        if path is None:
            path, _ = QFileDialog.getOpenFileName(self, "자동 인쇄할 디자인 고르기", self.prefs.auto_print_design(),
                                                  "하나스튜디오 디자인 (*.hsd)")
        if path:
            self.prefs.set_auto_print_design(path)
            self._sync_auto()

    def toggle_auto_print(self) -> bool:
        from services.auto_print import AutoPrintError

        if self.auto.running:
            self.auto.stop()
            return False
        try:
            self.auto.start(self.prefs.auto_print_folder(), self.prefs.auto_print_design())
        except AutoPrintError as exc:
            inform(self, "자동 인쇄를 켤 수 없습니다", str(exc))
            return False
        return True

    def _build_history(self) -> Panel:
        panel = Panel("인쇄 기록")
        self.retention_combo = ComboBox([text for _, text in RETENTION_CHOICES])
        self.retention_combo.setFixedWidth(240)
        self.retention_combo.currentIndexChanged.connect(
            lambda index: self.prefs.set_retention_days(RETENTION_CHOICES[index][0]))
        self.retention_combo.setAccessibleName("기록을 두는 기간")
        panel.body.addWidget(FormRow("기록을 두는 기간", self.retention_combo,
                                     "이 기간이 지난 기록은 프로그램을 켤 때 지웁니다. 명단에 적힌 '뽑은 날짜'는 지우지 않습니다.",
                                     orientation="horizontal", label_width=_LABEL_WIDTH, control_align="left"))
        return panel

    def _build_files(self) -> Panel:
        panel = Panel("파일이 있는 곳")
        self.folder_rows: dict[str, tuple[str, SecondaryButton]] = {}
        for key, title, path, hint in self.folders():
            row = QFrame()
            row.setObjectName("SettingsRow")
            layout = QHBoxLayout(row)
            layout.setContentsMargins(0, 8, 0, 8)
            layout.setSpacing(SPACE["md"])
            texts = QVBoxLayout()
            texts.setSpacing(2)
            texts.addWidget(label(title, "strong"))
            path_label = label(path, "muted", wrap=True)
            path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            texts.addWidget(path_label)
            if hint:
                texts.addWidget(label(hint, "hint", wrap=True))
            layout.addLayout(texts, 1)
            button = SecondaryButton("폴더 열기", "sm", icon="folder")
            button.clicked.connect(lambda _checked=False, p=path: self._open(p))
            layout.addWidget(button, 0, Qt.AlignTop)
            panel.body.addWidget(row)
            self.folder_rows[key] = (path, button)
        return panel

    @staticmethod
    def folders() -> list[tuple[str, str, str, str]]:
        data = app_data_dir()
        return [
            ("data", "설정과 인쇄 기록", data,
             f"인쇄 기록 파일: {os.path.basename(default_history_path())}"),
            ("documents", "내 문서", os.path.join(os.path.expanduser("~"), "Documents"),
             "디자인과 명단 파일은 저장할 때 고른 곳에 있습니다. 보통 여기에 둡니다."),
        ]

    def _build_admin(self) -> Panel:
        panel = Panel("관리자", "관리자 키로 들어왔을 때만 보입니다.")
        self.license_button = SecondaryButton("라이선스 관리 열기")
        self.license_button.clicked.connect(self.open_license_admin)
        row = QHBoxLayout()
        row.addWidget(self.license_button)
        row.addStretch(1)
        panel.body.addLayout(row)
        return panel

    def open_license_admin(self) -> None:
        """라이선스 키를 만들고 정지하는 관리 창(기존 화면의 '관리 > 라이선스 관리'와 같은 창)."""
        try:
            from licensing.admin_panel import AdminPanel
            AdminPanel(self.window()).exec()
        except Exception as exc:
            log.exception("라이선스 관리 창을 열지 못했습니다")
            inform(self, "라이선스 관리를 열 수 없습니다", str(exc))

    def _build_about(self) -> Panel:
        panel = Panel("프로그램 정보")
        self.version_labels: dict[str, str] = {}
        for name, value in version_lines():
            row = QHBoxLayout()
            row.setSpacing(SPACE["md"])
            title = label(name, "label")
            title.setFixedWidth(_LABEL_WIDTH)
            row.addWidget(title)
            row.addWidget(label(value, None, wrap=True), 1)
            panel.body.addLayout(row)
            self.version_labels[name] = value
        self.update_button = SecondaryButton("업데이트 확인")
        self.update_button.setToolTip("새 버전이 있는지 인터넷에서 확인합니다")
        self.update_button.clicked.connect(self.check_for_updates)
        update_row = QHBoxLayout()
        update_row.addWidget(self.update_button)
        update_row.addStretch(1)
        panel.body.addLayout(update_row)
        self._update_flow = None
        return panel

    def check_for_updates(self) -> None:
        """새 버전이 있는지 확인하고, 있으면 받아서 다시 켤지 묻는다."""
        from ui.update_flow import UpdateFlow

        if self._update_flow is None:
            self._update_flow = UpdateFlow(self.window(), is_busy=self._is_printing)
        self._update_flow.check(silent=False)

    def _is_printing(self) -> bool:
        try:
            return bool(self.services.printing.is_printing)
        except Exception:
            return False

    # ------------------------------------------------------------------ 값
    def _load(self) -> None:
        prefs = self.prefs
        widgets = (self.card_combo, self.copies, self.stop_check, self.offset_x, self.offset_y,
                   self.retention_combo)
        for widget in widgets:
            widget.blockSignals(True)
        keys = [preset.key for preset in presets.PRESETS]
        self.card_combo.setCurrentIndex(keys.index(prefs.card_preset_key()))
        self.copies.set_value(prefs.copies(), emit=False)
        self.stop_check.setChecked(prefs.stop_on_error())
        x, y = prefs.offsets()
        self.offset_x.set_value(x, emit=False)
        self.offset_y.set_value(y, emit=False)
        days = prefs.retention_days()
        self.retention_combo.setCurrentIndex(next(i for i, (d, _) in enumerate(RETENTION_CHOICES) if d == days))
        for widget in widgets:
            widget.blockSignals(False)
        self.offset_reset.setEnabled(bool(x or y))
        self.migrated_note.setText(
            "예전 프로그램에서 쓰던 값을 그대로 가져왔습니다." if prefs.migrated else "")
        self.migrated_note.setVisible(bool(prefs.migrated))
        self._sync_mode(ThemeManager.instance().mode)

    def _save_offsets(self) -> None:
        self.prefs.set_offsets(self.offset_x.value(), self.offset_y.value())

    def _sync_mode(self, mode: str) -> None:
        if self.theme_control.current() != mode:
            self.theme_control.set_current(mode, emit=False)

    def _on_guide_toggled(self, on: bool) -> None:
        if self._bound_shell is not None:
            self._bound_shell.set_guide_visible(on)

    def _sync_guide(self, visible: bool) -> None:
        if self.guide_check.isChecked() != visible:
            self.guide_check.blockSignals(True)
            self.guide_check.setChecked(visible)
            self.guide_check.blockSignals(False)

    def _open(self, path: str) -> None:
        try:
            os.makedirs(path, exist_ok=True)
        except OSError:
            log.info("폴더를 만들지 못했습니다: %s", path)
        self.open_folder(path)
