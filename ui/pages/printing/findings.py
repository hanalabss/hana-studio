"""인쇄 전 확인 결과를 사람이 읽는 말과 "고치는 버튼"으로 바꾼다."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from services.preflight import Finding, PreflightReport
from ui.widgets import IconWidget, set_prop

from .text import plain

ACT_EXCLUDE = "exclude"             # payload: 사람 _id 목록
ACT_SHOW_PEOPLE = "show_people"     # payload: 사람 _id 목록
ACT_FIX_DESIGN = "fix_design"       # payload: (item_ids, side)
ACT_OPEN_LIST = "open_list"
ACT_OPEN_PRINTER = "open_printer"
ACT_RECHECK = "recheck"

_TONE_ICON = {"error": ("error", "danger_text"), "warn": ("warning", "warn_text"),
              "ok": ("check", "ok_text"), "info": ("info", "info_text"), "idle": ("info", "text_secondary")}
_PRINTER_CODES = ("feeder_empty", "ribbon_out", "printer_not_ready", "printer_warning")
_DESIGN_CODES = ("missing_column", "item_outside", "item_clipped", "no_white", "no_uv", "no_dual_side")


@dataclass
class Action:
    text: str
    key: str
    payload: Any = None


@dataclass
class Problem:
    code: str
    tone: str                    # error | warn | info | ok | idle
    text: str
    actions: list[Action] = field(default_factory=list)
    record_ids: list[int] = field(default_factory=list)

    @property
    def blocking(self) -> bool:
        return self.tone == "error"


def problem_for(finding: Finding) -> Problem:
    tone = "error" if finding.blocking else "warn"
    actions: list[Action] = []
    count = len(finding.record_ids)
    code = finding.code
    if code == "missing_photo":
        actions.append(Action(f"사진 없는 {count}명 빼고 인쇄", ACT_EXCLUDE, list(finding.record_ids)))
        actions.append(Action("명단에서 고치기", ACT_SHOW_PEOPLE, list(finding.record_ids)))
    elif code == "record_missing":
        actions.append(Action(f"지워진 {count}명 빼기", ACT_EXCLUDE, list(finding.record_ids)))
    elif code == "invalid_barcode":
        if count:
            actions.append(Action(f"이 {count}명 빼고 인쇄", ACT_EXCLUDE, list(finding.record_ids)))
            actions.append(Action("명단에서 고치기", ACT_SHOW_PEOPLE, list(finding.record_ids)))
        if finding.item_ids:
            actions.append(Action("디자인에서 고치기", ACT_FIX_DESIGN, (list(finding.item_ids), finding.side)))
    elif code == "empty_value":
        actions.append(Action("명단에서 고치기", ACT_SHOW_PEOPLE, list(finding.record_ids)))
    elif code == "no_data":
        actions.append(Action("명단 열기", ACT_OPEN_LIST))
    elif code in _DESIGN_CODES and finding.item_ids:
        actions.append(Action("디자인에서 고치기", ACT_FIX_DESIGN, (list(finding.item_ids), finding.side)))
    elif code in _PRINTER_CODES:
        if finding.blocking:
            actions.append(Action("다시 확인", ACT_RECHECK))
        actions.append(Action("프린터 화면 열기", ACT_OPEN_PRINTER))
    return Problem(code, tone, plain(finding.message), actions, list(finding.record_ids))


def problems_from(report: PreflightReport | None) -> list[Problem]:
    if report is None:
        return []
    problems = [problem_for(f) for f in report.errors]
    problems += [problem_for(f) for f in report.warnings]
    return problems


def warning_signature(problems: list[Problem]) -> tuple:
    return tuple(sorted((p.code, p.text) for p in problems if p.tone == "warn"))


class FindingRow(QFrame):
    """문제 한 건: 무슨 일인지와 바로 할 수 있는 일."""

    triggered = Signal(str, object)

    def __init__(self, problem: Problem, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.problem = problem
        self.setObjectName("Finding")
        self.setProperty("tone", problem.tone)
        outer = QHBoxLayout(self)
        outer.setContentsMargins(12, 10, 12, 10)
        outer.setSpacing(10)
        icon_name, icon_role = _TONE_ICON.get(problem.tone, _TONE_ICON["info"])
        holder = QVBoxLayout()
        holder.setContentsMargins(0, 1, 0, 0)
        holder.addWidget(IconWidget(icon_name, 18, icon_role))
        holder.addStretch(1)
        outer.addLayout(holder)
        column = QVBoxLayout()
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(8)
        self._text = QLabel(problem.text)
        self._text.setWordWrap(True)
        self._text.setTextInteractionFlags(Qt.TextSelectableByMouse)
        column.addWidget(self._text)
        self.buttons: list[QPushButton] = []
        if problem.actions:
            stacked = sum(len(a.text) for a in problem.actions) > 16 and len(problem.actions) > 1
            row = QVBoxLayout() if stacked else QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(6)
            for action in problem.actions:
                button = QPushButton(action.text)
                button.setAutoDefault(False)
                button.setCursor(Qt.PointingHandCursor)
                button.clicked.connect(lambda _checked=False, a=action: self.triggered.emit(a.key, a.payload))
                row.addWidget(button, 0, Qt.AlignLeft)
                self.buttons.append(button)
            if not stacked:
                row.addStretch(1)
            column.addLayout(row)
        outer.addLayout(column, 1)

    def text(self) -> str:
        return self._text.text()

    def button(self, key: str) -> QPushButton | None:
        for action, button in zip(self.problem.actions, self.buttons):
            if action.key == key:
                return button
        return None

    def set_tone(self, tone: str) -> None:
        set_prop(self, "tone", tone)
