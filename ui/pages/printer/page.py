"""프린터 화면: 찾은 프린터, 상태와 남은 양, 점검, 시험 인쇄, 못 찾았을 때 확인할 것."""
from __future__ import annotations

import logging

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QPlainTextEdit, QScrollArea, QVBoxLayout, QWidget,
)

from printer.base import FAMILY_RTAI, FAMILY_SMART, PrinterDevice
from services.print_service import PrintServiceError
from services.printers import default_capabilities
from ui.pages.printing.style import apply_page_style
from ui.pages.printing.text import (
    FAMILY_NAMES, brief, capability_words, connection_sentence, pill_state, plain, status_sentence,
    supplies_lines,
)
from ui.pages.settings.prefs import Prefs
from ui.shell import BasePage
from ui.theme import SPACE
from ui.widgets import (
    IconWidget, Panel, PrimaryButton, QuietButton, SecondaryButton, confirm, label, set_prop,
)

from .probe import CheckLine, ProbeTask, build_test_document
from .status import bind_printer_status

log = logging.getLogger(__name__)

_LINE_ICON = {"ok": ("check", "ok_text"), "warn": ("warning", "warn_text"),
              "error": ("error", "danger_text"), "idle": ("info", "text_muted")}
_LIBRARY_NAMES = {FAMILY_RTAI: "RTAI", FAMILY_SMART: "SMART"}
CONTENT_WIDTH = 760


def _same(a: PrinterDevice | None, b: PrinterDevice | None) -> bool:
    if a is None or b is None:
        return False
    return (a.family, a.address or a.name) == (b.family, b.address or b.name)


def checklist_for(diagnostics: list[str]) -> list[str]:
    """프린터를 찾지 못했을 때 사용자가 차례로 확인할 것."""
    items = [
        "프린터 전원이 켜져 있고, 프린터 화면에 오류 표시가 없는지 봐 주세요.",
        "USB 선이 프린터와 컴퓨터 양쪽에 끝까지 꽂혀 있는지 확인해 주세요. "
        "네트워크로 쓰는 프린터라면 랜선과 공유기를 확인해 주세요.",
        "Windows의 '프린터 및 스캐너'에 이 프린터가 보이는지 확인해 주세요. 보이지 않으면 "
        "프린터 제조사의 드라이버를 먼저 설치해 주세요.",
    ]
    text = "\n".join(diagnostics)
    missing = [name for family, name in _LIBRARY_NAMES.items()
               if any(line.upper().startswith(f"{family.upper()}:") and "라이브러리" in line
                      and "찾을 수 없" in line for line in diagnostics)]
    if missing:
        items.append(f"{', '.join(missing)} 프린터를 쓰는 데 필요한 프로그램 파일이 빠져 있습니다. "
                     "hanaStudio를 다시 설치해 주세요.")
    if "오류" in text:
        items.append("프린터를 찾다가 문제가 생겼습니다. 프린터를 껐다가 10초 뒤에 다시 켜 주세요.")
    items.append("확인했으면 위의 '다시 찾기'를 눌러 주세요.")
    return items


class DeviceCard(QFrame):
    """찾은 프린터 한 대."""

    useRequested = Signal(object)
    probeRequested = Signal(object)
    testPrintRequested = Signal(object)

    def __init__(self, device: PrinterDevice, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.device = device
        self.setObjectName("DeviceCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(SPACE["xl"], SPACE["lg"], SPACE["xl"], SPACE["lg"])
        layout.setSpacing(SPACE["sm"])

        top = QHBoxLayout()
        top.setSpacing(SPACE["sm"])
        self.name = label(device.name, "title")
        top.addWidget(self.name)
        self.family_tag = QLabel(FAMILY_NAMES.get(device.family, device.family))
        self.family_tag.setObjectName("Tag")
        top.addWidget(self.family_tag)
        self.current_tag = QLabel("지금 쓰는 프린터")
        self.current_tag.setObjectName("Tag")
        self.current_tag.setProperty("tone", "accent")
        top.addWidget(self.current_tag)
        top.addStretch(1)
        self.state_tag = QLabel("")
        self.state_tag.setObjectName("Tag")
        top.addWidget(self.state_tag)
        layout.addLayout(top)

        self.connection = label(connection_sentence(device), "muted")
        layout.addWidget(self.connection)
        self.status_text = label("", None, wrap=True)
        layout.addWidget(self.status_text)
        self.supplies = label("", "secondary", wrap=True)
        layout.addWidget(self.supplies)
        self.abilities = label("", "secondary", wrap=True)
        layout.addWidget(self.abilities)

        buttons = QHBoxLayout()
        buttons.setSpacing(SPACE["sm"])
        self.use_button = PrimaryButton("이 프린터 쓰기")
        self.probe_button = SecondaryButton("프린터 점검")
        self.probe_button.setToolTip("카드와 리본을 쓰지 않고 상태만 확인합니다")
        self.test_button = SecondaryButton("시험 인쇄 1장", icon="print")
        self.test_button.setToolTip("카드 한 장을 써서 위치와 색을 확인합니다")
        self.use_button.clicked.connect(lambda: self.useRequested.emit(self.device))
        self.probe_button.clicked.connect(lambda: self.probeRequested.emit(self.device))
        self.test_button.clicked.connect(lambda: self.testPrintRequested.emit(self.device))
        for button in (self.use_button, self.probe_button, self.test_button):
            buttons.addWidget(button)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        self.note = label("", "hint", wrap=True)
        layout.addWidget(self.note)

        self.results = QFrame()
        self.results.setObjectName("Block")
        self._results_layout = QGridLayout(self.results)
        self._results_layout.setContentsMargins(SPACE["md"], SPACE["md"], SPACE["md"], SPACE["md"])
        self._results_layout.setHorizontalSpacing(SPACE["sm"])
        self._results_layout.setVerticalSpacing(6)
        self._results_layout.setColumnStretch(2, 1)
        self.results.setVisible(False)
        layout.addWidget(self.results)
        self.result_lines: list[CheckLine] = []

    def update_state(self, current: bool, state: str, state_text: str, status_text: str, supplies: str,
                     abilities: list[str], busy_reason: str) -> None:
        set_prop(self, "current", "true" if current else "false")
        self.current_tag.setVisible(current)
        self.use_button.setVisible(not current)
        self.state_tag.setText(state_text)
        self.state_tag.setVisible(bool(state_text))
        set_prop(self.state_tag, "tone", {"ready": "ok", "busy": "busy", "error": "error"}.get(state, ""))
        self.status_text.setText(status_text)
        self.supplies.setText(supplies)
        self.supplies.setVisible(bool(supplies))
        self.abilities.setText("할 수 있는 일: " + (", ".join(abilities) if abilities else "한 면 컬러 인쇄"))
        for button in (self.use_button, self.probe_button, self.test_button):
            button.setEnabled(not busy_reason)
        self.note.setVisible(False)

    def ability_words(self) -> list[str]:
        text = self.abilities.text().split(": ", 1)[-1]
        return [word.strip() for word in text.split(",") if word.strip()]

    def set_results(self, lines: list[CheckLine], heading: str = "") -> None:
        self.result_lines = list(lines)
        while self._results_layout.count():
            item = self._results_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        row = 0
        if heading:
            self._results_layout.addWidget(label(heading, "label"), row, 0, 1, 3)
            row += 1
        for line in lines:
            icon, role = _LINE_ICON.get(line.tone, _LINE_ICON["idle"])
            self._results_layout.addWidget(IconWidget(icon, 16, role), row, 0, Qt.AlignTop)
            title = label(line.title, "muted")
            title.setMinimumWidth(96)
            self._results_layout.addWidget(title, row, 1, Qt.AlignTop)
            self._results_layout.addWidget(label(line.text, None, wrap=True), row, 2)
            row += 1
        self.results.setVisible(bool(lines))

    def result_texts(self) -> list[str]:
        return [f"{line.title}: {line.text}" for line in self.result_lines]


class PrinterPage(BasePage):
    def __init__(self, services, parent: QWidget | None = None, prefs: Prefs | None = None) -> None:
        super().__init__("프린터", parent)
        self.services = services
        self.prefs = prefs or Prefs.instance()
        self.confirm = confirm
        self.cards: list[DeviceCard] = []
        self._probe: ProbeTask | None = None
        self._probe_device: PrinterDevice | None = None
        self._test_job = None
        self._test_device: PrinterDevice | None = None
        self._test_error = ""

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
        column_host = QWidget()
        column_host.setFixedWidth(CONTENT_WIDTH)
        column = QVBoxLayout(column_host)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(SPACE["lg"])
        outer.addWidget(column_host)
        outer.addStretch(1)
        scroll.setWidget(body)
        root.addWidget(scroll)

        head = QHBoxLayout()
        titles = QVBoxLayout()
        titles.setSpacing(2)
        titles.addWidget(label("프린터", "heading"))
        self.summary = label("", "secondary")
        titles.addWidget(self.summary)
        head.addLayout(titles, 1)
        self.refresh_button = SecondaryButton("다시 찾기", icon="search")
        self.refresh_button.clicked.connect(self.refresh_devices)
        head.addWidget(self.refresh_button, 0, Qt.AlignTop)
        column.addLayout(head)
        self.refresh_reason = label("", "hint", wrap=True)
        column.addWidget(self.refresh_reason)

        self._cards_box = QVBoxLayout()
        self._cards_box.setSpacing(SPACE["md"])
        column.addLayout(self._cards_box)

        self.none_panel = Panel("프린터를 찾지 못했습니다", "아래를 차례로 확인해 주세요.")
        self._checklist_box = QVBoxLayout()
        self._checklist_box.setSpacing(0)
        self.none_panel.body.addLayout(self._checklist_box)
        self.checklist_labels: list[QLabel] = []
        self.detail_toggle = QuietButton("자세한 내용 보기", "sm", accent=True, icon="chevron-down")
        self.detail_toggle.clicked.connect(self._toggle_detail)
        toggle_row = QHBoxLayout()
        toggle_row.addWidget(self.detail_toggle)
        toggle_row.addStretch(1)
        self.none_panel.body.addLayout(toggle_row)
        self.raw_detail = QPlainTextEdit()
        self.raw_detail.setObjectName("RawDetail")
        self.raw_detail.setReadOnly(True)
        self.raw_detail.setFixedHeight(96)
        self.raw_detail.setVisible(False)
        self.raw_detail.setAccessibleName("프린터를 찾은 과정")
        self.none_panel.body.addWidget(self.raw_detail)
        column.addWidget(self.none_panel)

        self.searching_panel = Panel("프린터를 찾고 있습니다", "연결된 프린터를 확인하는 중입니다. 잠시만 기다려 주세요.")
        column.addWidget(self.searching_panel)

        advanced = Panel("고급")
        self.fake_check = QCheckBox("연습 모드 (가짜 프린터)")
        self.fake_check.toggled.connect(self._on_fake_toggled)
        advanced.body.addWidget(self.fake_check)
        self.fake_hint = label(
            "프린터 없이 프로그램을 익힐 때 켭니다. 가짜 프린터 두 대가 나타나고, 인쇄를 눌러도 "
            "카드와 리본을 쓰지 않습니다. 진짜 프린터로 뽑으려면 꺼 주세요.", "hint", wrap=True)
        self.fake_hint.setContentsMargins(28, 0, 0, 0)
        advanced.body.addWidget(self.fake_hint)
        column.addWidget(advanced)
        column.addStretch(1)
        apply_page_style(self)

        printers, printing = services.printers, services.printing
        printers.devices_changed.connect(self._rebuild)
        for signal in (printers.current_changed, printers.status_changed, printers.busy_changed,
                       printing.job_queued, printing.job_started):
            signal.connect(self._sync)
        printing.finished.connect(self._on_job_finished)
        printing.error.connect(self._on_job_error)
        self._rebuild()

    # ------------------------------------------------------------------ 틀
    @property
    def shell(self):
        return self._shell

    @shell.setter
    def shell(self, value) -> None:
        self._shell = value
        if value is not None:
            bind_printer_status(value, self.services)

    def on_enter(self) -> None:
        self._sync()
        self.services.printers.poll_now()

    def _toast(self, text: str, kind: str = "info") -> None:
        if self._shell is not None:
            self._shell.show_toast(text, kind)

    # ------------------------------------------------------------------ 목록
    def _rebuild(self, *_args) -> None:
        for card in self.cards:
            self._cards_box.removeWidget(card)
            card.setParent(None)
            card.deleteLater()
        self.cards = []
        for device in self.services.printers.devices():
            card = DeviceCard(device)
            card.useRequested.connect(self.use_device)
            card.probeRequested.connect(self.probe_device)
            card.testPrintRequested.connect(self.test_print)
            self._cards_box.addWidget(card)
            self.cards.append(card)
        self._fill_checklist()
        self._sync()

    def _fill_checklist(self) -> None:
        diagnostics = self.services.printers.diagnostics()
        while self._checklist_box.count():
            item = self._checklist_box.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self.checklist_labels = []
        for number, text in enumerate(checklist_for(diagnostics), start=1):
            row = QFrame()
            row.setObjectName("CheckItem")
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 10, 0, 10)
            row_layout.setSpacing(SPACE["md"])
            mark = label(f"{number}", "strong")
            mark.setFixedWidth(20)
            mark.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
            row_layout.addWidget(mark, 0, Qt.AlignTop)
            text_label = label(text, None, wrap=True)
            row_layout.addWidget(text_label, 1)
            self._checklist_box.addWidget(row)
            self.checklist_labels.append(text_label)
        self.raw_detail.setPlainText("\n".join(diagnostics) or "남은 내용이 없습니다.")

    def checklist_texts(self) -> list[str]:
        return [item.text() for item in self.checklist_labels]

    def _toggle_detail(self) -> None:
        self.set_detail_visible(self.raw_detail.isHidden())

    def set_detail_visible(self, visible: bool) -> None:
        self.raw_detail.setVisible(visible)
        self.detail_toggle.setText("자세한 내용 접기" if visible else "자세한 내용 보기")
        self.detail_toggle.set_icon_name("chevron-up" if visible else "chevron-down")

    def card_for(self, device: PrinterDevice) -> DeviceCard | None:
        return next((card for card in self.cards if _same(card.device, device)), None)

    def busy_reason(self) -> str:
        if self.services.printing.is_printing or self.services.printers.printing:
            return "인쇄 중에는 프린터를 다시 찾거나 바꾸거나 점검할 수 없습니다. 인쇄가 끝나면 할 수 있습니다."
        if self._probe is not None:
            return "프린터를 점검하고 있습니다."
        return ""

    def _sync(self, *_args) -> None:
        printers, printing = self.services.printers, self.services.printing
        devices = printers.devices()
        current, status = printers.current(), printers.status()
        is_printing = printing.is_printing
        searching = printers.is_busy and not devices
        reason = self.busy_reason()
        for card in self.cards:
            is_current = _same(card.device, current)
            if is_current:
                state, text, _ = pill_state(card.device, status, is_printing)
                short = {"ready": "준비됨", "busy": "인쇄 중", "error": "확인 필요", "offline": "확인 중"}[state]
                sentence = "지금 카드를 찍고 있습니다." if is_printing else status_sentence(status)
                supplies = ", ".join(supplies_lines(status))
                abilities = capability_words(printers.capabilities())
            else:
                state, short = "offline", ""
                sentence = "이 프린터를 쓰기로 하면 상태를 확인합니다."
                supplies = ""
                abilities = capability_words(default_capabilities(card.device))
            card.update_state(is_current, state, short, sentence, supplies, abilities, reason)

        families = sorted({FAMILY_NAMES.get(d.family, d.family) for d in devices})
        if devices:
            self.summary.setText(f"찾은 프린터 {len(devices)}대 ({', '.join(families)})")
        else:
            self.summary.setText("프린터를 찾는 중입니다" if searching else "찾은 프린터가 없습니다")
        self.none_panel.setVisible(not devices and not searching)
        self.searching_panel.setVisible(searching)

        if is_printing or printers.printing:
            refresh_reason = "인쇄 중에는 프린터를 다시 찾을 수 없습니다. 인쇄가 끝나면 누를 수 있습니다."
        elif printers.is_busy:
            refresh_reason = "프린터를 확인하고 있습니다. 잠시만 기다려 주세요."
        elif self._probe is not None:
            refresh_reason = "프린터를 점검하고 있습니다."
        else:
            refresh_reason = ""
        self.refresh_button.setEnabled(not refresh_reason)
        self.refresh_button.setToolTip(refresh_reason)
        self.refresh_reason.setText(reason or (refresh_reason if (is_printing or printers.printing) else ""))
        self.refresh_reason.setVisible(bool(self.refresh_reason.text()))

        fakes = printers.use_fakes
        if self.fake_check.isChecked() != fakes:
            self.fake_check.blockSignals(True)
            self.fake_check.setChecked(fakes)
            self.fake_check.blockSignals(False)
        self.fake_check.setEnabled(not is_printing and not printers.printing)

    # ------------------------------------------------------------------ 할 수 있는 일
    def refresh_devices(self) -> bool:
        started = self.services.printers.refresh()
        self._sync()
        return started

    def use_device(self, device: PrinterDevice) -> None:
        if self.busy_reason():
            return
        self.services.printers.select(device)
        self.services.printers.poll_now()
        self._sync()

    def _on_fake_toggled(self, enabled: bool) -> None:
        if self.services.printing.is_printing:
            self._sync()
            return
        self.services.printers.set_use_fakes(enabled)
        self._sync()
        self._toast("연습 모드를 켰습니다. 카드와 리본을 쓰지 않습니다." if enabled
                    else "연습 모드를 껐습니다. 진짜 프린터를 찾습니다.")

    def probe_device(self, device: PrinterDevice) -> bool:
        """카드와 리본을 쓰지 않는 점검. 인쇄 중이거나 다른 확인이 돌고 있으면 하지 않는다."""
        if self.busy_reason() or self.services.printers.is_busy:
            if self.services.printers.is_busy:
                self._toast("프린터를 확인하고 있습니다. 잠시 뒤에 다시 눌러 주세요.")
            return False
        card = self.card_for(device)
        if card is not None:
            card.set_results([CheckLine("idle", "점검", "프린터에 물어보고 있습니다.")])
        lock = getattr(self.services.printers, "_sdk_lock", None)
        self._probe_device = device
        self._probe = ProbeTask(device, lock, self)
        self._probe.finished_lines.connect(self._on_probe_done)
        self._probe.start()
        self._sync()
        return True

    def is_probing(self) -> bool:
        return self._probe is not None

    def _on_probe_done(self, device: PrinterDevice, lines: list) -> None:
        task, self._probe = self._probe, None
        if task is not None:
            task.wait(5000)
            task.deleteLater()
        card = self.card_for(device)
        if card is not None:
            card.set_results(list(lines), "점검 결과 (카드와 리본은 쓰지 않았습니다)")
        self._sync()
        self.services.printers.poll_now()

    def test_print(self, device: PrinterDevice) -> bool:
        if self.busy_reason():
            return False
        if not self.confirm(
                self, "시험 인쇄를 할까요?",
                f"{device.name}(으)로 카드 한 장을 뽑습니다. 카드 1장과 리본을 씁니다. "
                "테두리와 색 띠가 찍힌 카드가 나오면 위치와 색을 확인하세요.",
                confirm_text="카드 1장 쓰고 시험 인쇄", cancel_text="그만두기"):
            return False
        options = self.prefs.print_options()
        options.copies = 1
        options.job_name = "시험 인쇄"
        try:
            self._test_job = self.services.printing.start(
                build_test_document(self.prefs.card_preset()), None, None, device, options)
        except PrintServiceError as exc:
            self._toast(plain(str(exc)), "error")
            return False
        self._test_device = device
        card = self.card_for(device)
        if card is not None:
            card.set_results([CheckLine("idle", "시험 인쇄", "카드를 찍고 있습니다.")])
        self._sync()
        return True

    def _on_job_error(self, handle, message: str) -> None:
        if handle is self._test_job:
            self._test_error = brief(message)

    def _on_job_finished(self, summary) -> None:
        job = self._test_job
        if job is not None and summary.job_id == job.id:
            self._test_job = None
            card = self.card_for(self._test_device) if self._test_device is not None else None
            if summary.ok:
                line = CheckLine("ok", "시험 인쇄", "카드가 나왔습니다. 테두리가 가장자리와 나란한지, 색이 고른지 확인하세요. "
                                                "한쪽으로 치우쳤다면 설정의 '찍히는 위치 맞추기'에서 고칩니다.")
            else:
                reason = (next((brief(r.message) for r in job.results if r.message), "")
                          or self._test_error or brief(summary.message))
                line = CheckLine("error", "시험 인쇄", f"카드를 뽑지 못했습니다. {reason}".strip())
            if card is not None:
                card.set_results([line])
            self._toast(line.text, "ok" if summary.ok else "error")
            self.services.printing.forget(job)
        self._sync()
