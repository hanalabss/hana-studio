"""빈 문서일 때 보이는 시작 화면: 카드 크기·방향·면을 고르고 시작할 모양을 누른다."""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QLabel, QScrollArea, QVBoxLayout, QWidget

from design import presets, renderer
from ui.theme import SPACE, ThemeManager
from ui.widgets import SecondaryButton, SegmentedControl, label, theme_color

from .common import ClickFrame, fit_segments, group

SIZE_OPTIONS = (
    ("cr80", "표준 카드 (CR80)", "신용카드와 같은 크기입니다. 85.6 × 54 mm"),
    ("cut_54x86", "54 × 86 mm", "가장자리에 흰 여백을 조금 두고 인쇄할 때 씁니다."),
    ("cut_54x86_bleed", "54 × 86 mm + 재단 여백",
     "가장자리까지 색을 채울 때 씁니다. 사방 2 mm는 잘려 나가므로 58 × 90 mm로 꾸밉니다."),
)

SAMPLE_RECORD = {
    "이름": "김하늘", "부서": "경영지원팀", "직급": "사원", "사번": "HN260417",
    "학과": "시각디자인과", "학번": "20261234", "회원번호": "M000128", "방문처": "3층 회의실",
}

_THUMB_W, _THUMB_H = 184, 124


def pil_to_qimage(image) -> QImage:
    rgb = image.convert("RGB")
    data = rgb.tobytes("raw", "RGB")
    return QImage(data, rgb.width, rgb.height, rgb.width * 3, QImage.Format.Format_RGB888).copy()


def card_pixmap(document, side_key: str, record, box_w: int, box_h: int, *, context=None,
                dpr: float = 1.0, margin: int = 6) -> QPixmap:
    """한 면을 그려 box 안에 맞춘 둥근 모서리 카드 그림."""
    width_mm, height_mm = document.side_size_mm(side_key)
    scale = min((box_w - 2 * margin) / width_mm, (box_h - 2 * margin) / height_mm)
    dpi = max(40.0, scale * 25.4 * dpr * 1.5)
    rendered = renderer.render_side(document, side_key, record, dpi=dpi, context=context, design_mode=True)
    image = pil_to_qimage(rendered.color)
    pixmap = QPixmap(round(box_w * dpr), round(box_h * dpr))
    pixmap.setDevicePixelRatio(dpr)
    pixmap.fill(Qt.transparent)
    card_w, card_h = width_mm * scale, height_mm * scale
    rect = QRectF((box_w - card_w) / 2, (box_h - card_h) / 2, card_w, card_h)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
    path = QPainterPath()
    radius = 3.0 * scale
    path.addRoundedRect(rect, radius, radius)
    painter.save()
    painter.setClipPath(path)
    painter.drawImage(rect, image)
    painter.restore()
    painter.setPen(QPen(theme_color("control_line"), 1))
    painter.setBrush(Qt.NoBrush)
    painter.drawPath(path)
    painter.end()
    return pixmap


class TemplateCard(ClickFrame):
    def __init__(self, key: str, title: str, description: str, parent: QWidget | None = None) -> None:
        super().__init__("TemplateCard", parent)
        self.key = key
        self.setAccessibleName(f"{title}로 시작")
        self.setToolTip(f"{title}로 시작합니다")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 14)
        layout.setSpacing(4)
        self.thumb = QLabel()
        self.thumb.setFixedHeight(_THUMB_H)
        self.thumb.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.thumb)
        layout.addSpacing(6)
        layout.addWidget(label(title, "strong"))
        text = label(description, "caption", wrap=True)
        text.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        text.setMinimumHeight(34)
        layout.addWidget(text)
        layout.addStretch(1)
        self.setMinimumWidth(168)


class StartPanel(QWidget):
    """startRequested(template_key), openRequested(), imageRequested()를 보낸다."""

    startRequested = Signal(str)
    openRequested = Signal()
    imageRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("StartPanel")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        outer.addWidget(scroll)
        body = QWidget()
        body.setObjectName("StartBody")
        scroll.setWidget(body)
        center = QHBoxLayout(body)
        center.setContentsMargins(SPACE["xxl"], SPACE["xl"], SPACE["xxl"], SPACE["xl"])
        column_host = QWidget()
        column_host.setMaximumWidth(1180)
        center.addWidget(column_host, 1)
        column = QVBoxLayout(column_host)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)

        column.addWidget(label("어떤 카드를 만들까요?", "heading"))
        column.addSpacing(SPACE["xs"])
        column.addWidget(label(
            "크기를 고르고 시작할 모양을 누르면 바로 꾸밀 수 있습니다. 크기와 방향은 나중에도 바꿀 수 있습니다.",
            "secondary", wrap=True))
        column.addSpacing(SPACE["lg"])

        options = QFrame()
        options.setObjectName("StartOptions")
        row = QHBoxLayout(options)
        row.setContentsMargins(SPACE["lg"], SPACE["md"], SPACE["lg"], SPACE["md"])
        row.setSpacing(SPACE["xl"])
        self.size_control = SegmentedControl([(key, text) for key, text, _ in SIZE_OPTIONS], current="cr80")
        self.size_hint = label("", "hint")
        size_box = group("카드 크기", self.size_control)
        size_box.layout().addWidget(self.size_hint)
        row.addWidget(size_box)
        self.orientation_control = SegmentedControl([("landscape", "가로"), ("portrait", "세로")])
        row.addWidget(group("방향", self.orientation_control), 0, Qt.AlignTop)
        self.sides_control = SegmentedControl([("single", "한 면"), ("dual", "양면")])
        row.addWidget(group("인쇄 면", self.sides_control), 0, Qt.AlignTop)
        row.addStretch(1)
        column.addWidget(options)
        column.addSpacing(SPACE["xl"])

        column.addWidget(label("시작할 모양", "title"))
        column.addSpacing(SPACE["sm"] + 2)
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(SPACE["md"])
        self.cards: dict[str, TemplateCard] = {}
        for index, template in enumerate(presets.TEMPLATES):
            card = TemplateCard(template.key, template.label, template.description)
            card.clicked.connect(lambda key=template.key: self.choose(key))
            self.cards[template.key] = card
            grid.addWidget(card, 0, index)
            grid.setColumnStretch(index, 1)
        column.addLayout(grid)
        column.addSpacing(SPACE["xl"])

        existing = QHBoxLayout()
        existing.setSpacing(SPACE["sm"])
        existing.addWidget(label("이미 만든 것이 있다면", "secondary"))
        existing.addSpacing(SPACE["xs"])
        self.open_button = SecondaryButton("저장한 디자인 열기", icon="folder")
        self.open_button.clicked.connect(self.openRequested)
        existing.addWidget(self.open_button)
        self.image_button = SecondaryButton("완성된 그림 한 장으로 시작", icon="image")
        self.image_button.setToolTip("다른 프로그램에서 만든 카드 그림을 그대로 인쇄할 때 씁니다")
        self.image_button.clicked.connect(self.imageRequested)
        existing.addWidget(self.image_button)
        existing.addStretch(1)
        column.addLayout(existing)
        column.addStretch(1)

        for control in (self.size_control, self.orientation_control, self.sides_control):
            fit_segments(control)
        self.size_control.currentChanged.connect(self._on_options_changed)
        self.orientation_control.currentChanged.connect(self._on_options_changed)
        ThemeManager.instance().themeChanged.connect(self.refresh_thumbnails)
        self._thumbs_stale = True
        self._sync_hint()

    # ---- 고른 값 ----
    def preset_key(self) -> str:
        return self.size_control.current()

    def orientation(self) -> str:
        return self.orientation_control.current()

    def dual_side(self) -> bool:
        return self.sides_control.current() == "dual"

    def choose(self, template_key: str) -> None:
        self.startRequested.emit(template_key)

    # ---- 그림 ----
    def _sync_hint(self) -> None:
        self.size_hint.setText(next(hint for key, _, hint in SIZE_OPTIONS if key == self.preset_key()))

    def _on_options_changed(self, *_args) -> None:
        self._sync_hint()
        self._thumbs_stale = True
        if self.isVisible():
            self.refresh_thumbnails()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._thumbs_stale:
            self.refresh_thumbnails()

    def refresh_thumbnails(self, *_args) -> None:
        preset = presets.get_preset(self.preset_key())
        dpr = self.devicePixelRatioF()
        for key, card in self.cards.items():
            document = presets.build_template(key, preset, self.orientation())
            card.thumb.setPixmap(card_pixmap(document, "front", SAMPLE_RECORD, _THUMB_W, _THUMB_H, dpr=dpr))
        self._thumbs_stale = False
