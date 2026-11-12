"""명단이 아직 없을 때의 화면: 파일 가져오기, 예시 파일 받기, 직접 입력, 최근에 쓴 명단."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QFrame, QHBoxLayout, QSizePolicy, QVBoxLayout, QWidget

from ui.theme import SPACE
from ui.widgets import IconWidget, PrimaryButton, QuietButton, label, paint_icon, theme_color

from .inputs import Rethemed
from .storage import RecentList
from .text import SAMPLE_BUTTON_TEXT, SAMPLE_BUTTON_TIP


class _SheetToCards(Rethemed, QWidget):
    """명단 한 줄이 카드 한 장이 된다는 그림."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setFixedSize(330, 116)
        self.follow_theme()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        line = theme_color("control_line")
        soft = theme_color("surface_pressed")
        field = theme_color("field_bg")

        sheet = QRectF(8.5, 10.5, 128, 96)
        p.setPen(QPen(line, 1))
        p.setBrush(theme_color("surface_raised"))
        p.drawRoundedRect(sheet, 8, 8)
        p.setPen(Qt.NoPen)
        head = QPainterPath()
        head.addRoundedRect(QRectF(sheet.x() + 0.5, sheet.y() + 0.5, sheet.width() - 1, 20), 7.5, 7.5)
        head.addRect(QRectF(sheet.x() + 0.5, sheet.y() + 10, sheet.width() - 1, 10.5))
        p.setBrush(theme_color("surface_sunken"))
        p.drawPath(head.simplified())
        for column, width in ((14, 26), (52, 34), (96, 22)):
            p.setBrush(theme_color("text_disabled"))
            p.drawRoundedRect(QRectF(sheet.x() + column, sheet.y() + 8, width, 5), 2.5, 2.5)
        for row in range(3):
            y = sheet.y() + 31 + row * 22
            first = row == 0
            if first:
                p.setBrush(theme_color("accent_soft"))
                p.drawRect(QRectF(sheet.x() + 1, y - 6, sheet.width() - 2, 21))
            for column, width in ((14, 30), (52, 38), (96, 18)):
                p.setBrush(field if first else soft)
                p.drawRoundedRect(QRectF(sheet.x() + column, y + 1, width, 7), 3.5, 3.5)

        paint_icon(p, "chevron-right", QRectF(152, 46, 24, 24), "text_muted", stroke=2.0)

        for index, (dx, dy) in enumerate(((14, 14), (7, 7), (0, 0))):
            card = QRectF(192.5 + dx, 8.5 + dy, 122, 78)
            p.setPen(QPen(line, 1))
            p.setBrush(theme_color("surface_raised"))
            p.drawRoundedRect(card, 7, 7)
            if index < 2:
                continue
            p.setPen(Qt.NoPen)
            p.setBrush(theme_color("accent"))
            band = QPainterPath()
            band.addRoundedRect(QRectF(card.x() + 0.5, card.y() + 0.5, card.width() - 1, 16), 6.5, 6.5)
            band.addRect(QRectF(card.x() + 0.5, card.y() + 8, card.width() - 1, 8.5))
            p.drawPath(band.simplified())
            p.setBrush(soft)
            p.drawRoundedRect(QRectF(card.x() + 10, card.y() + 25, 30, 40), 3, 3)
            p.setBrush(theme_color("text_disabled"))
            p.drawEllipse(QPointF(card.x() + 25, card.y() + 39), 6, 6)
            p.setBrush(field)
            p.drawRoundedRect(QRectF(card.x() + 50, card.y() + 27, 46, 10), 5, 5)
            p.drawRoundedRect(QRectF(card.x() + 50, card.y() + 43, 60, 7), 3.5, 3.5)
            p.drawRoundedRect(QRectF(card.x() + 50, card.y() + 56, 34, 7), 3.5, 3.5)
        p.end()


class _DropZone(Rethemed, QFrame):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._active = False
        self.follow_theme()

    def set_active(self, active: bool) -> None:
        if active != self._active:
            self._active = active
            self.update()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        rect = QRectF(self.rect()).adjusted(1.5, 1.5, -1.5, -1.5)
        p.setBrush(theme_color("accent_soft" if self._active else "surface"))
        pen = QPen(theme_color("accent" if self._active else "indicator_line"), 1.5, Qt.CustomDashLine)
        pen.setDashPattern([5, 4])
        p.setPen(pen)
        p.drawRoundedRect(rect, 16, 16)
        p.end()


class EmptyView(QWidget):
    importRequested = Signal()
    manualRequested = Signal()
    sampleRequested = Signal()
    openRequested = Signal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(SPACE["xl"], SPACE["xl"], SPACE["xl"], SPACE["xl"])
        outer.addStretch(3)
        row = QHBoxLayout()
        row.addStretch(1)
        column = QVBoxLayout()
        column.setSpacing(SPACE["lg"])
        row.addLayout(column)
        row.addStretch(1)
        outer.addLayout(row)
        outer.addStretch(4)

        self.zone = _DropZone()
        self.zone.setFixedWidth(600)
        zone = QVBoxLayout(self.zone)
        zone.setContentsMargins(40, 34, 40, 30)
        zone.setSpacing(0)
        zone.addWidget(_SheetToCards(), 0, Qt.AlignHCenter)
        zone.addSpacing(SPACE["xl"])
        self.heading = label("엑셀이나 CSV 파일을 가져오세요", "heading")
        self.heading.setAlignment(Qt.AlignHCenter)
        zone.addWidget(self.heading)
        zone.addSpacing(SPACE["sm"])
        self.guidance = label(
            "한 줄에 한 명씩 적힌 파일이면 됩니다. 한 줄이 카드 한 장이 됩니다.\n원래 파일은 고치지 않습니다.",
            "secondary")
        self.guidance.setAlignment(Qt.AlignHCenter)
        zone.addWidget(self.guidance)
        zone.addSpacing(SPACE["xl"])
        self.import_button = PrimaryButton("파일 가져오기", "lg", icon="import")
        self.import_button.setMinimumWidth(220)
        self.import_button.clicked.connect(self.importRequested)
        zone.addWidget(self.import_button, 0, Qt.AlignHCenter)
        zone.addSpacing(SPACE["md"])
        drop_hint = label("파일을 이 화면에 끌어다 놓아도 됩니다.", "hint")
        drop_hint.setAlignment(Qt.AlignHCenter)
        zone.addWidget(drop_hint)
        column.addWidget(self.zone)

        manual = QHBoxLayout()
        manual.setSpacing(SPACE["xs"])
        manual.addStretch(1)
        manual.addWidget(label("어떻게 적는지 궁금하면", "muted"))
        self.sample_button = QuietButton(SAMPLE_BUTTON_TEXT, "sm", accent=True)
        self.sample_button.setAccessibleName(SAMPLE_BUTTON_TEXT)
        self.sample_button.setToolTip(SAMPLE_BUTTON_TIP)
        self.sample_button.clicked.connect(self.sampleRequested)
        manual.addWidget(self.sample_button)
        manual.addSpacing(SPACE["lg"])
        manual.addWidget(label("파일이 없나요?", "muted"))
        self.manual_button = QuietButton("직접 입력해서 시작", "sm", accent=True)
        self.manual_button.clicked.connect(self.manualRequested)
        manual.addWidget(self.manual_button)
        manual.addStretch(1)
        column.addLayout(manual)

        self.recent_box = QWidget()
        self.recent_box.setFixedWidth(600)
        recent = QVBoxLayout(self.recent_box)
        recent.setContentsMargins(0, SPACE["sm"], 0, 0)
        recent.setSpacing(2)
        recent.addWidget(label("최근에 쓴 명단", "label"))
        self._recent_rows = QVBoxLayout()
        self._recent_rows.setSpacing(0)
        recent.addLayout(self._recent_rows)
        column.addWidget(self.recent_box)
        self.recent_buttons: list[QuietButton] = []

        self.details = label("", "hint", wrap=True)
        self.details.setFixedWidth(600)
        self.details.setAlignment(Qt.AlignHCenter)
        self.details.setTextInteractionFlags(Qt.TextSelectableByMouse)
        column.addWidget(self.details)

    def set_drop_active(self, active: bool) -> None:
        self.zone.set_active(active)

    def set_details(self, text: str) -> None:
        self.details.setText(text)

    def set_recent(self, lists: list[RecentList]) -> None:
        while self._recent_rows.count():
            child = self._recent_rows.takeAt(0)
            if child.widget() is not None:
                child.widget().hide()
                child.widget().deleteLater()
        self.recent_buttons = []
        for item in lists:
            row_widget = QWidget()
            row = QHBoxLayout(row_widget)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(SPACE["sm"])
            row.addWidget(IconWidget("people", 18, "text_muted"))
            name = label(item.name, "strong")
            name.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
            row.addWidget(name)
            when = f"{item.modified.month}월 {item.modified.day}일에 고침"
            if item.beside_design:
                when = f"이 카드와 같이 쓰던 명단, {when}"
            row.addWidget(label(when, "muted"))
            row.addStretch(1)
            button = QuietButton("열기", "sm", accent=True)
            button.setAccessibleName(f"{item.name} 명단 열기")
            button.setToolTip(item.path)
            button.clicked.connect(lambda _checked=False, path=item.path: self.openRequested.emit(path))
            row.addWidget(button)
            self.recent_buttons.append(button)
            self._recent_rows.addWidget(row_widget)
        self.recent_box.setVisible(bool(lists))
