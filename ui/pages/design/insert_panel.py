"""왼쪽 패널 위쪽: 넣기 버튼과 명단 항목 칩."""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QAction, QPainter, QPen
from PySide6.QtWidgets import QGridLayout, QMenu, QPushButton, QVBoxLayout, QWidget

from ui.theme import SPACE, ThemeManager
from ui.widgets import FieldChip, FocusRingMixin, label, theme_color

from . import icons
from .common import ChipBox, LinkButton, clear_layout, link_row

EXAMPLE_FIELDS = ("이름", "부서", "사진")
TILE_HEIGHT = 50


class InsertTile(FocusRingMixin, QPushButton):
    """아이콘과 이름이 함께 있는 큰 넣기 버튼."""

    def __init__(self, text: str, icon_name: str, tooltip: str, parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self._icon_name = icon_name
        self.setFixedHeight(TILE_HEIGHT)
        self.setAutoDefault(False)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip(tooltip)
        self.setAccessibleName(f"{text} 넣기")
        self.setAttribute(Qt.WA_Hover, True)
        ThemeManager.instance().themeChanged.connect(lambda *_: self.update())

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        if self.isDown() or (self.menu() is not None and self.menu().isVisible()):
            fill, edge = theme_color("surface_pressed"), theme_color("control_line")
        elif self.underMouse():
            fill, edge = theme_color("surface_hover"), theme_color("control_line")
        else:
            fill, edge = theme_color("surface_raised"), theme_color("line")
        painter.setPen(QPen(edge, 1))
        painter.setBrush(fill)
        painter.drawRoundedRect(rect, 10, 10)
        badge = QRectF(9, (self.height() - 30) / 2, 30, 30)
        painter.setPen(Qt.NoPen)
        painter.setBrush(theme_color("accent_soft"))
        painter.drawRoundedRect(badge, 8, 8)
        icons.paint(painter, self._icon_name, badge.adjusted(6, 6, -6, -6), "accent_text")
        painter.setPen(theme_color("text"))
        painter.setFont(self.font())
        text_rect = QRectF(badge.right() + 8, 0, self.width() - badge.right() - 12, self.height())
        painter.drawText(text_rect, Qt.AlignVCenter | Qt.AlignLeft, self.text())
        if self.menu() is not None:
            icons.paint(painter, "chevron-down", QRectF(self.width() - 19, (self.height() - 11) / 2, 11, 11),
                        "text_muted")
        self.paint_focus_ring(painter, 10)
        painter.end()


class ListFieldChip(FieldChip):
    """명단 항목 칩. 캔버스가 받는 끌기 형식은 공용 FieldChip이 싣는다. 여기서는 도움말만 바꾼다."""

    def __init__(self, name: str, kind: str = "text", parent: QWidget | None = None) -> None:
        super().__init__(name, parent, kind=kind)
        self.kind = kind
        if kind == "image":
            self.setToolTip("카드 위로 끌어다 놓으면 사람마다 바뀌는 사진 칸이 됩니다")
        else:
            self.setToolTip("카드 위로 끌어다 놓으면 사람마다 바뀌는 칸이 됩니다")


class InsertPanel(QWidget):
    addRequested = Signal(str)            # text | image | rect | ellipse | line | barcode | qr
    imageFileRequested = Signal()
    fieldRequested = Signal(str, str)     # 열 이름, kind
    listPageRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        layout.addWidget(label("넣기", "strong"))
        layout.addSpacing(SPACE["sm"])
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(SPACE["sm"])
        self.tiles: dict[str, InsertTile] = {}
        specs = (
            ("text", "글자", "text", "글자를 넣습니다. 넣은 뒤 두 번 누르면 바로 고칠 수 있습니다"),
            ("image", "사진", "image", "로고나 사진 같은 그림을 넣습니다"),
            ("shape", "도형", "shape", "사각형이나 원을 넣습니다"),
            ("line", "선", "line", "구분선을 넣습니다"),
            ("barcode", "바코드", "barcode", "막대 모양 바코드를 넣습니다"),
            ("qr", "QR", "qr", "QR 코드를 넣습니다"),
        )
        for index, (key, text, icon_name, tooltip) in enumerate(specs):
            tile = InsertTile(text, icon_name, tooltip)
            self.tiles[key] = tile
            grid.addWidget(tile, index // 2, index % 2)
        layout.addLayout(grid)
        for key in ("text", "line", "barcode", "qr"):
            self.tiles[key].clicked.connect(lambda _=False, kind=key: self.addRequested.emit(kind))

        image_menu = QMenu(self)
        self.image_file_action = QAction("그림 파일 고르기…", self)
        self.image_file_action.triggered.connect(lambda: self.imageFileRequested.emit())
        self.image_frame_action = QAction("빈 사진 칸 (나중에 명단 사진과 연결)", self)
        self.image_frame_action.triggered.connect(lambda: self.addRequested.emit("image"))
        image_menu.addActions([self.image_file_action, self.image_frame_action])
        self.tiles["image"].setMenu(image_menu)

        shape_menu = QMenu(self)
        self.rect_action = QAction(icons.icon("rect"), "사각형", self)
        self.rect_action.triggered.connect(lambda: self.addRequested.emit("rect"))
        self.ellipse_action = QAction(icons.icon("ellipse"), "원", self)
        self.ellipse_action.triggered.connect(lambda: self.addRequested.emit("ellipse"))
        shape_menu.addActions([self.rect_action, self.ellipse_action])
        self.tiles["shape"].setMenu(shape_menu)
        ThemeManager.instance().themeChanged.connect(self._refresh_menu_icons)

        layout.addSpacing(SPACE["xl"] - 4)
        layout.addWidget(label("명단 항목", "strong"))
        layout.addSpacing(SPACE["xs"])
        self.fields_hint = label("", "hint", wrap=True)
        layout.addWidget(self.fields_hint)
        layout.addSpacing(SPACE["sm"])
        self._chip_host = ChipBox()
        self._chip_flow = self._chip_host.flow
        layout.addWidget(self._chip_host)
        self.list_link = LinkButton("명단 넣으러 가기")
        self.list_link.clicked.connect(self.listPageRequested)
        self._link_row = link_row(self.list_link)
        layout.addSpacing(SPACE["xs"])
        layout.addWidget(self._link_row)
        layout.addStretch(1)

        self.chips: list[ListFieldChip] = []
        self.set_columns(None)

    def _refresh_menu_icons(self, *_args) -> None:
        self.rect_action.setIcon(icons.icon("rect"))
        self.ellipse_action.setIcon(icons.icon("ellipse"))

    def has_list(self) -> bool:
        return self._has_list

    def set_compact(self, compact: bool) -> None:
        """세로 자리가 모자랄 때 설명 글을 접는다(칩의 풍선 도움말에는 그대로 있다)."""
        self.fields_hint.setVisible(not compact)

    def set_columns(self, columns: list[tuple[str, str]] | None) -> None:
        """columns: [(열 이름, kind)] 또는 None(열린 명단 없음)."""
        clear_layout(self._chip_flow)
        self.chips = []
        self._has_list = columns is not None
        if columns is None:
            self.fields_hint.setText("명단을 넣으면 이름, 부서 같은 항목이 여기에 나타납니다. 아래는 예시입니다.")
            for name in EXAMPLE_FIELDS:
                chip = ListFieldChip(name, "image" if name == "사진" else "text")
                chip.setEnabled(False)
                chip.setToolTip("명단을 넣으면 쓸 수 있습니다")
                self._add_chip(chip)
            self._link_row.setVisible(True)
        elif not columns:
            self.fields_hint.setText("열린 명단에 항목이 없습니다. 명단 화면에서 항목을 추가하세요.")
            self._link_row.setVisible(True)
        else:
            self.fields_hint.setText("눌러서 넣거나 카드 위로 끌어다 놓으세요. 사람마다 내용이 바뀝니다.")
            for name, kind in columns:
                chip = ListFieldChip(name, kind)
                chip.clicked.connect(lambda column, k=kind: self.fieldRequested.emit(column, k))
                self._add_chip(chip)
            self._link_row.setVisible(False)
        self._chip_host.updateGeometry()

    def _add_chip(self, chip: ListFieldChip) -> None:
        self._chip_flow.addWidget(chip)
        self.chips.append(chip)
        chip.show()
