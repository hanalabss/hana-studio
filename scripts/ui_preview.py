"""새 UI 틀 미리 보기. 라이선스 확인도 프린터도 없이 돈다.

    python scripts/ui_preview.py                     틀을 띄운다 (Windows 설정을 따름)
    python scripts/ui_preview.py --theme dark        어두운 화면으로
    python scripts/ui_preview.py --gallery           위젯을 상태별로 모두 본다
    python scripts/ui_preview.py --shot <폴더>        틀과 갤러리를 두 테마로 PNG 저장 (화면 없이 가능)
    python scripts/ui_preview.py --legacy            '빠른 인쇄'에 실제 기존 화면을 넣는다
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="하나스튜디오 새 UI 미리 보기")
    parser.add_argument("--theme", choices=("light", "dark", "system"), default=None)
    parser.add_argument("--gallery", action="store_true", help="위젯 갤러리를 연다")
    parser.add_argument("--shot", metavar="DIR", help="PNG로 저장하고 끝낸다")
    parser.add_argument("--page", default="design", help="처음 열 화면 id")
    parser.add_argument("--font", default=None, help="글꼴을 강제로 고른다 (예: \"Malgun Gothic\")")
    parser.add_argument("--scale", default=None, help="화면 배율 (예: 1.5)")
    parser.add_argument("--legacy", action="store_true", help="빠른 인쇄에 실제 기존 화면을 넣는다")
    return parser.parse_args(argv)


def _prepare_environment(args: argparse.Namespace) -> None:
    if args.shot:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    if args.font:
        os.environ["HANA_STUDIO_FONT"] = args.font
    if args.scale:
        os.environ["QT_SCALE_FACTOR"] = str(args.scale)
    if "HANA_STUDIO_SETTINGS_FILE" not in os.environ:
        folder = Path(tempfile.gettempdir()) / "hana_studio_ui"
        folder.mkdir(parents=True, exist_ok=True)
        name = "shot.ini" if args.shot else "preview.ini"
        if args.shot and (folder / name).exists():
            (folder / name).unlink()
        os.environ["HANA_STUDIO_SETTINGS_FILE"] = str(folder / name)


# --------------------------------------------------------------------------- 갤러리


def _row(*widgets, stretch: bool = True):
    from PySide6.QtWidgets import QHBoxLayout

    box = QHBoxLayout()
    box.setSpacing(12)
    for widget in widgets:
        box.addWidget(widget)
    if stretch:
        box.addStretch(1)
    return box


def _section_buttons():
    from ui.widgets import (DangerButton, IconButton, Panel, PrimaryButton, QuietButton,
                            SecondaryButton, label)

    panel = Panel("버튼", "한 화면에 주 버튼은 하나만 둡니다. 글자는 누르면 일어날 일을 씁니다.")
    for size, name in (("lg", "크게"), ("md", "보통"), ("sm", "작게")):
        panel.body.addWidget(label(name, "label"))
        panel.body.addLayout(_row(
            PrimaryButton("다음: 명단 넣기", size),
            SecondaryButton("디자인 저장", size),
            QuietButton("자세한 설정 열기", size, accent=True),
            QuietButton("건너뛰기", size),
            DangerButton("12명 지우기", size),
        ))
    panel.body.addWidget(label("아이콘과 함께", "label"))
    panel.body.addLayout(_row(
        PrimaryButton("12장 인쇄 시작", icon="print"),
        SecondaryButton("엑셀·CSV 가져오기", icon="import"),
        SecondaryButton("되돌리기", "sm", icon="undo"),
        QuietButton("사진 폴더 고르기", icon="folder"),
        DangerButton("지우기", icon="trash"),
        IconButton("chevron-left", "이전 사람", variant="secondary"),
        IconButton("chevron-right", "다음 사람", variant="secondary"),
        IconButton("settings", "설정"),
    ))
    panel.body.addWidget(label("쓸 수 없을 때", "label"))
    disabled = [
        PrimaryButton("인쇄 시작"), SecondaryButton("디자인 저장"), QuietButton("건너뛰기"),
        DangerButton("지우기"), SecondaryButton("되돌리기", "sm", icon="undo"),
        IconButton("chevron-right", "다음 사람", variant="secondary"),
    ]
    for button in disabled:
        button.setEnabled(False)
    panel.body.addLayout(_row(*disabled))
    return panel


def _section_choices():
    from ui.widgets import FieldChip, FilterChip, Panel, SegmentedControl, StatusPill, label

    panel = Panel("고르기와 상태")
    panel.body.addWidget(label("세그먼트", "label"))
    off = SegmentedControl(["컬러", "화이트", "UV"])
    off.setEnabled(False)
    panel.body.addLayout(_row(
        SegmentedControl(["앞면", "뒷면"]),
        SegmentedControl([("fixed", "직접 입력"), ("column", "명단 항목")], current="column"),
        SegmentedControl(["컬러", "화이트", "UV"]),
        off,
    ))
    panel.body.addWidget(label("걸러 보기 칩", "label"))
    gone = FilterChip("휴직", 0)
    gone.setEnabled(False)
    panel.body.addLayout(_row(
        FilterChip("전체", 128, checked=True),
        FilterChip("아직 안 뽑음", 41),
        FilterChip("인쇄함", 87),
        FilterChip("사진 없음", 4, tone="warn"),
        FilterChip("사진 없음", 4, tone="warn", checked=True),
        gone,
    ))
    panel.body.addWidget(label("사람마다 바뀌는 칸 (카드 위로 끌어다 놓기)", "label"))
    off_chip = FieldChip("입사일")
    off_chip.setEnabled(False)
    panel.body.addLayout(_row(FieldChip("이름"), FieldChip("부서"), FieldChip("직급"), FieldChip("사번"),
                              FieldChip("사진"), off_chip))
    panel.body.addWidget(label("프린터 상태", "label"))
    panel.body.addLayout(_row(
        StatusPill("RTAI R600 준비됨", "ready"),
        StatusPill("RTAI R600 인쇄 중", "busy"),
        StatusPill("리본을 다 썼습니다", "error"),
        StatusPill("프린터 연결 안 됨", "offline"),
    ))
    return panel


def _section_inputs():
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import (QCheckBox, QDoubleSpinBox, QGridLayout, QLineEdit, QPlainTextEdit,
                                   QRadioButton, QSlider, QSpinBox, QVBoxLayout, QWidget)

    from ui.widgets import ComboBox, FormRow, MmField, Panel, SearchField, StepperField, SwatchPicker, label

    panel = Panel("입력")
    grid = QGridLayout()
    grid.setHorizontalSpacing(24)
    grid.setVerticalSpacing(16)

    name = QLineEdit()
    name.setPlaceholderText("예: 2026 신입 사원증")
    filled = QLineEdit("경영지원팀")
    wrong = QLineEdit("HN26041")
    wrong.setProperty("invalid", "true")
    wrong_row = FormRow("사번", wrong)
    wrong_row.set_hint("사번은 8글자여야 합니다. 지금은 7글자입니다.", "error")
    locked = QLineEdit("바꿀 수 없는 값")
    locked.setEnabled(False)
    grid.addWidget(FormRow("디자인 이름", name, "파일 이름으로도 쓰입니다."), 0, 0)
    grid.addWidget(FormRow("부서", filled), 0, 1)
    grid.addWidget(wrong_row, 0, 2)
    grid.addWidget(FormRow("잠긴 칸", locked), 0, 3)

    combo = ComboBox(["명단의 이름", "명단의 부서", "명단의 사번", "직접 입력"])
    combo_off = ComboBox(["맑은 고딕"])
    combo_off.setEnabled(False)
    spin = QSpinBox()
    spin.setRange(1, 99)
    spin.setValue(2)
    spin.setSuffix(" 장")
    dspin = QDoubleSpinBox()
    dspin.setRange(0, 360)
    dspin.setValue(90)
    dspin.setSuffix(" 도")
    grid.addWidget(FormRow("무엇을 보여 줄까요", combo), 1, 0)
    grid.addWidget(FormRow("글꼴", combo_off), 1, 1)
    grid.addWidget(FormRow("한 사람당 장수", spin), 1, 2)
    grid.addWidget(FormRow("회전", dspin), 1, 3)

    mm_off = MmField(8.2)
    mm_off.setEnabled(False)
    grid.addWidget(FormRow("가로 위치", MmField(30.5)), 2, 0)
    grid.addWidget(FormRow("글자 크기", StepperField(25, 6, 96, 1, 0, "pt")), 2, 1)
    grid.addWidget(FormRow("높이", mm_off), 2, 2)
    grid.addWidget(FormRow("명단 찾기", SearchField("이름, 부서, 사번으로 찾기")), 2, 3)

    checks = QWidget()
    checks_box = QVBoxLayout(checks)
    checks_box.setContentsMargins(0, 0, 0, 0)
    checks_box.setSpacing(2)
    on = QCheckBox("이름이 길면 알아서 줄이기")
    on.setChecked(True)
    off = QCheckBox("뒷면도 인쇄")
    on_locked = QCheckBox("화이트 포함 (리본이 지원함)")
    on_locked.setChecked(True)
    on_locked.setEnabled(False)
    off_locked = QCheckBox("UV 인쇄 (리본이 지원하지 않음)")
    off_locked.setEnabled(False)
    for box in (on, off, on_locked, off_locked):
        checks_box.addWidget(box)
    grid.addWidget(checks, 3, 0)

    radios = QWidget()
    radios_box = QVBoxLayout(radios)
    radios_box.setContentsMargins(0, 0, 0, 0)
    radios_box.setSpacing(2)
    first = QRadioButton("선택한 12명")
    first.setChecked(True)
    radios_box.addWidget(first)
    radios_box.addWidget(QRadioButton("아직 안 뽑은 41명 모두"))
    third = QRadioButton("번호로 고르기")
    third.setEnabled(False)
    radios_box.addWidget(third)
    grid.addWidget(radios, 3, 1)

    slider = QSlider(Qt.Horizontal)
    slider.setRange(50, 200)
    slider.setValue(120)
    side = QWidget()
    side_box = QVBoxLayout(side)
    side_box.setContentsMargins(0, 0, 0, 0)
    side_box.setSpacing(16)
    side_box.addWidget(FormRow("화면 크기", slider, "120%"))
    side_box.addStretch(1)
    grid.addWidget(side, 3, 2)

    memo = QPlainTextEdit("뒷면 안내 문구를 여기에 씁니다.\n분실 시 가까운 우체통에 넣어 주세요.")
    memo.setFixedHeight(88)
    grid.addWidget(FormRow("뒷면 문구", memo), 3, 3)

    for column in range(4):
        grid.setColumnStretch(column, 1)
    panel.body.addLayout(grid)
    panel.body.addWidget(label("글자색", "label"))
    panel.body.addWidget(SwatchPicker(current="#16181C"))
    return panel


def _section_feedback():
    from PySide6.QtWidgets import QHBoxLayout, QWidget

    from ui.widgets import Banner, Panel, Toast, label

    panel = Panel("알림", "무슨 일이 있었는지와 지금 할 수 있는 일 하나를 함께 보여 줍니다.")
    panel.body.addWidget(Banner("info", "신입사원_명단.xlsx에서 128명을 가져왔습니다. 카드의 칸 4개를 파일의 열과 맞췄습니다.",
                                "맞춘 열 확인하기"))
    panel.body.addWidget(Banner("ok", "41장을 모두 인쇄했습니다.", "기록 보기", closable=True))
    panel.body.addWidget(Banner("warn", "선택한 사람 중 1명은 사진이 없습니다. 그대로 뽑으면 사진 칸이 비어 나옵니다.",
                                "사진 넣기"))
    panel.body.addWidget(Banner("error", "리본을 다 썼습니다. 새 리본으로 바꾼 뒤 이어서 인쇄하세요. 17번째 카드부터 다시 시작합니다.",
                                "이어서 인쇄"))
    panel.body.addWidget(label("잠깐 뜨는 알림", "label"))
    stage = QWidget()
    stage.setFixedHeight(56)
    box = QHBoxLayout(stage)
    box.setContentsMargins(0, 0, 0, 0)
    box.setSpacing(12)
    for text, kind, action in (
        ("디자인을 저장했습니다.", "info", None),
        ("3명을 명단에서 지웠습니다.", "info", "되돌리기"),
        ("프린터와 연결이 끊겼습니다.", "error", "다시 연결"),
    ):
        toast = Toast(stage, text, kind, action, timeout_ms=0)
        stage.removeEventFilter(toast)
        toast.adjustSize()
        box.addWidget(toast)
    box.addStretch(1)
    panel.body.addWidget(stage)
    return panel


def _section_empty():
    from ui.widgets import Card, EmptyState

    card = Card(padding=0)
    empty = EmptyState(
        "아직 명단이 없습니다",
        "엑셀이나 CSV 파일을 가져오면 사람마다 이름과 사진이 바뀐 카드를 만듭니다. 원본 파일은 고치지 않습니다.",
        "엑셀·CSV 가져오기", "people", "한 명씩 직접 넣기",
    )
    empty.setFixedHeight(360)
    card.body.addWidget(empty)
    return card


def _section_progress():
    from PySide6.QtWidgets import QGridLayout

    from ui.widgets import Panel, ProgressBar, RibbonProgress, label

    panel = Panel("진행", "카드 한 장은 리본 패널을 차례로 찍어 완성합니다. 지금 찍는 색이 채워집니다.")
    grid = QGridLayout()
    grid.setHorizontalSpacing(32)
    grid.setVerticalSpacing(12)
    waiting = RibbonProgress()
    cyan = RibbonProgress()
    cyan.set_progress(2, 0.55)
    white = RibbonProgress()
    white.set_progress(4, 0.4)
    done = RibbonProgress()
    done.finish()
    samples = (("시작 전", waiting), ("파랑을 찍는 중", cyan), ("화이트를 찍는 중", white), ("한 장 끝", done))
    for i, (name, widget) in enumerate(samples):
        grid.addWidget(label(name, "label"), (i // 2) * 2, i % 2)
        grid.addWidget(widget, (i // 2) * 2 + 1, i % 2)
    panel.body.addLayout(grid)
    panel.body.addWidget(label("전체 진행: 41장 중 17장", "label"))
    bar = ProgressBar()
    bar.set_fraction(17 / 41)
    panel.body.addWidget(bar)
    return panel


def _section_data():
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import (QAbstractItemView, QHBoxLayout, QHeaderView, QListWidget, QTableWidget,
                                   QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

    from ui.widgets import Panel, label

    panel = Panel("표와 목록")
    people = [
        ("강민서", "영업1팀", "대리", "HN260415", "아직"),
        ("고은채", "디자인팀", "사원", "HN260416", "아직"),
        ("김하늘", "경영지원팀", "사원", "HN260417", "아직"),
        ("남지호", "개발팀", "과장", "HN230102", "11월 10일 1장"),
        ("문서윤", "개발팀", "사원", "HN260418", "사진 없음"),
        ("박도윤", "영업2팀", "차장", "HN190233", "11월 10일 2장"),
        ("서지안", "경영지원팀", "사원", "HN260419", "아직"),
        ("양수아", "디자인팀", "대리", "HN260420", "아직"),
        ("이준우", "개발팀", "사원", "HN260421", "아직"),
        ("정예린", "영업1팀", "부장", "HN150027", "11월 9일 1장"),
    ]
    table = QTableWidget(len(people), 6)
    table.setHorizontalHeaderLabels(["", "이름", "부서", "직급", "사번", "인쇄"])
    table.verticalHeader().setVisible(False)
    table.verticalHeader().setDefaultSectionSize(40)
    table.setShowGrid(False)
    table.setSelectionBehavior(QAbstractItemView.SelectRows)
    table.setEditTriggers(QAbstractItemView.NoEditTriggers)
    table.horizontalHeader().setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    table.horizontalHeader().setHighlightSections(False)
    table.setColumnWidth(0, 44)
    table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)
    for column in range(1, 6):
        table.horizontalHeader().setSectionResizeMode(column, QHeaderView.Stretch)
    for row, person in enumerate(people):
        check = QTableWidgetItem()
        check.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        check.setCheckState(Qt.Checked if row in (0, 1, 2, 4) else Qt.Unchecked)
        table.setItem(row, 0, check)
        for column, value in enumerate(person, start=1):
            table.setItem(row, column, QTableWidgetItem(value))
    table.selectRow(2)
    table.setFixedHeight(280)

    side = QWidget()
    side.setFixedWidth(300)
    side_box = QVBoxLayout(side)
    side_box.setContentsMargins(0, 0, 0, 0)
    side_box.setSpacing(12)
    tabs = QTabWidget()
    layers = QListWidget()
    layers.addItems(["이름  글자", "사진  화이트 포함", "사번 바코드", "배경  사진", "회사 로고", "직급  글자",
                     "부서  글자", "QR 코드"])
    layers.setCurrentRow(0)
    tabs.addTab(layers, "레이어")
    tabs.addTab(QWidget(), "명단 항목")
    side_box.addWidget(tabs)

    box = QHBoxLayout()
    box.setSpacing(16)
    box.addWidget(table, 1)
    box.addWidget(side)
    panel.body.addLayout(box)
    panel.body.addWidget(label("표 머리글, 선택한 줄, 체크 칸, 스크롤 막대가 모두 테마 색을 따릅니다.", "hint"))
    return panel


def _section_text():
    from ui.widgets import Panel, label

    panel = Panel("글자")
    panel.body.setSpacing(6)
    panel.body.addWidget(label("이렇게 나옵니다", "display"))
    panel.body.addWidget(label("명단 128명 중 12명 선택", "heading"))
    panel.body.addWidget(label("사람마다 바뀌는 칸", "title"))
    panel.body.addWidget(label("노란 칸은 명단의 내용으로 바뀝니다. 칸을 눌러 글꼴과 크기를 고치세요."))
    panel.body.addWidget(label("카드 위로 끌어다 놓으면 사람마다 바뀌는 칸이 됩니다.", "secondary"))
    panel.body.addWidget(label("리본 남은 양 약 310장", "muted"))
    panel.body.addWidget(label("분홍 점선은 재단선입니다. 중요한 내용은 선 안쪽에 두세요.", "caption"))
    return panel


GALLERY_SECTIONS = (
    ("text", _section_text),
    ("buttons", _section_buttons),
    ("choices", _section_choices),
    ("inputs", _section_inputs),
    ("feedback", _section_feedback),
    ("empty", _section_empty),
    ("progress", _section_progress),
    ("data", _section_data),
)


def build_gallery():
    """(스크롤 창, {이름: 구역 위젯})을 돌려준다."""
    from PySide6.QtWidgets import QScrollArea, QVBoxLayout, QWidget

    from ui.theme import mark_themed

    body = QWidget()
    body.setObjectName("GalleryRoot")
    layout = QVBoxLayout(body)
    layout.setContentsMargins(24, 24, 24, 24)
    layout.setSpacing(20)
    sections = {}
    for name, factory in GALLERY_SECTIONS:
        widget = factory()
        sections[name] = widget
        layout.addWidget(widget)
    layout.addStretch(1)
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(body)
    mark_themed(scroll)
    scroll.setWindowTitle("하나스튜디오 위젯 갤러리")
    scroll.resize(1180, 860)
    return scroll, sections


def build_sample_dialog(parent=None):
    from ui.widgets import Dialog, FormRow, MmField

    dialog = Dialog("명단에서 12명을 지울까요?",
                    "지운 사람은 이 명단에서만 사라집니다. 원본 엑셀 파일은 그대로 있습니다.", parent)
    dialog.body.addWidget(FormRow("지운 뒤 남는 사람", MmField(116, 0, 999, 1, 0), "미리 보기용 예시 칸입니다."))
    dialog.add_button("그만두기", "secondary", "reject")
    dialog.add_button("12명 지우기", "danger", "accept")
    return dialog


def build_sample_menu(parent=None):
    from PySide6.QtWidgets import QMenu

    menu = QMenu(parent)
    menu.addAction("복사")
    menu.addAction("붙여 넣기")
    menu.addAction("맨 앞으로 가져오기")
    menu.addSeparator()
    sub = menu.addMenu("정렬")
    sub.addAction("왼쪽 맞춤")
    off = menu.addAction("묶기 (둘 이상 고르면 쓸 수 있음)")
    off.setEnabled(False)
    menu.addSeparator()
    menu.addAction("지우기")
    return menu


# --------------------------------------------------------------------------- 틀


def _stub_legacy_factory():
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QLabel, QMainWindow

    window = QMainWindow()
    note = QLabel("미리 보기에서는 기존 화면을 열지 않습니다.\n--legacy 옵션을 주면 실제 '빠른 인쇄' 화면이 이 자리에 들어옵니다.")
    note.setAlignment(Qt.AlignCenter)
    window.setCentralWidget(note)
    return window


def build_shell(use_legacy: bool = False):
    from ui.shell import AppShell

    shell = AppShell(legacy_factory=None if use_legacy else _stub_legacy_factory)
    shell.set_document("2026 신입 사원증", True)
    shell.set_printer_status("ready", "RTAI R600 준비됨")
    shell.undoRequested.connect(lambda: shell.show_toast("되돌릴 일이 아직 없습니다."))
    return shell


# --------------------------------------------------------------------------- PNG 저장


def _settle(app, rounds: int = 3) -> None:
    for _ in range(rounds):
        app.processEvents()


def _save(widget, path: Path) -> None:
    pixmap = widget.grab()
    if pixmap.isNull() or not pixmap.save(str(path)):
        raise RuntimeError(f"그림을 저장하지 못했습니다: {path}")


def _focus_strip(app, items, background) -> "QImage":
    """키보드 포커스 링을 위젯마다 하나씩 찍어 한 줄로 이어 붙인다."""
    from PySide6.QtCore import QRect, Qt
    from PySide6.QtGui import QImage, QPainter

    pad = 10
    tiles = []
    items[0].window().activateWindow()
    _settle(app, 5)
    for widget in items:
        window = widget.window()
        window.activateWindow()
        widget.clearFocus()
        widget.setFocus(Qt.TabFocusReason)
        _settle(app)
        rect = QRect(widget.mapTo(window, widget.rect().topLeft()), widget.size()).adjusted(-pad, -pad, pad, pad)
        tiles.append(window.grab(rect).toImage())
        widget.clearFocus()
    width = sum(t.width() / t.devicePixelRatio() for t in tiles) + 16 * (len(tiles) + 1)
    height = max(t.height() / t.devicePixelRatio() for t in tiles) + 32
    dpr = tiles[0].devicePixelRatio()
    image = QImage(round(width * dpr), round(height * dpr), QImage.Format_ARGB32)
    image.setDevicePixelRatio(dpr)
    image.fill(background)
    painter = QPainter(image)
    x = 16
    for tile in tiles:
        painter.drawImage(x, 16, tile)
        x += round(tile.width() / tile.devicePixelRatio()) + 16
    painter.end()
    return image


def _focus_samples():
    from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLineEdit, QRadioButton, QWidget

    from ui.shell.nav_rail import RailButton
    from ui.shell.step_guide import StepPill
    from ui.theme import mark_themed
    from ui.widgets import (ComboBox, DangerButton, FieldChip, FilterChip, MmField, PrimaryButton,
                            SecondaryButton, SegmentedControl, StatusPill)

    host = QWidget()
    host.setObjectName("GalleryRoot")
    mark_themed(host)
    box = QHBoxLayout(host)
    box.setContentsMargins(24, 24, 24, 24)
    box.setSpacing(24)
    segmented = SegmentedControl(["앞면", "뒷면"])
    rail = RailButton("x", "명단", "people")
    rail.setFixedWidth(68)
    step = StepPill("x", 2, "명단 넣기")
    step.set_state("current")
    items = [
        PrimaryButton("인쇄 시작"), SecondaryButton("저장"), DangerButton("지우기"), QLineEdit("김하늘"),
        ComboBox(["맑은 고딕"]), QCheckBox("양면"), QRadioButton("전체"), segmented.button("뒷면"),
        FilterChip("전체", 128, checked=True), FieldChip("이름"), StatusPill("준비됨", "ready"), rail, step,
    ]
    for widget in (*items[:7], segmented, *items[8:]):
        box.addWidget(widget)
    mm = MmField(30.5)
    box.addWidget(mm)
    return host, items


def export_screenshots(app, out_dir: str | os.PathLike) -> list[Path]:
    """틀의 화면들과 갤러리 구역을 두 테마로 PNG로 저장하고 파일 목록을 돌려준다."""
    from PySide6.QtCore import QPoint
    from PySide6.QtWidgets import QToolTip

    from ui.theme import ThemeManager

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manager = ThemeManager.instance()
    manager.apply(app)
    written: list[Path] = []

    def save(widget, name: str) -> None:
        path = out / name
        _save(widget, path)
        written.append(path)

    for theme in ("light", "dark"):
        manager.set_mode(theme, persist=False)

        shell = build_shell()
        shell.resize(1280, 800)
        shell.show()
        _settle(app)
        printer_states = {
            "design": ("ready", "RTAI R600 준비됨"),
            "list": ("ready", "RTAI R600 준비됨"),
            "print": ("busy", "RTAI R600 인쇄 중"),
            "history": ("error", "리본을 다 썼습니다"),
            "quick": ("offline", "프린터 연결 안 됨"),
            "printer": ("offline", "프린터 연결 안 됨"),
            "settings": ("ready", "RTAI R600 준비됨"),
        }
        for index, page_id in enumerate(("design", "list", "print", "history", "quick", "printer", "settings")):
            shell.navigate(page_id)
            shell.set_step_done("design", page_id in ("list", "print"))
            shell.set_step_done("list", page_id == "print")
            shell.set_printer_status(*printer_states[page_id])
            shell.set_document("2026 신입 사원증", page_id != "list")
            _settle(app, 5)
            save(shell, f"shell_{index + 1}_{page_id}_{theme}.png")

        shell.navigate("design")
        shell.set_printer_status("ready", "RTAI R600 준비됨")
        shell.set_guide_visible(False, persist=False)
        shell.show_toast("순서 안내를 숨겼습니다. 설정에서 다시 켤 수 있습니다.", action_text="다시 보기",
                         timeout_ms=0)
        _settle(app)
        save(shell, f"shell_8_toast_{theme}.png")
        shell.set_guide_visible(True, persist=False)

        shell.printer_popover.adjustSize()
        save(shell.printer_popover, f"popover_{theme}.png")

        gallery, sections = build_gallery()
        gallery.resize(1180, 900)
        gallery.show()
        _settle(app, 5)
        for name, widget in sections.items():
            save(widget, f"gallery_{name}_{theme}.png")

        focus_host, focus_items = _focus_samples()
        focus_host.show()
        _settle(app)
        image = _focus_strip(app, focus_items, manager.theme.color("ground"))
        path = out / f"gallery_focus_{theme}.png"
        if not image.save(str(path)):
            raise RuntimeError(f"그림을 저장하지 못했습니다: {path}")
        written.append(path)

        dialog = build_sample_dialog()
        dialog.show()
        _settle(app)
        save(dialog, f"dialog_{theme}.png")

        menu = build_sample_menu()
        menu.adjustSize()
        menu.setActiveAction(menu.actions()[1])
        save(menu, f"menu_{theme}.png")

        QToolTip.showText(QPoint(40, 40), "방금 한 일을 되돌립니다 (Ctrl+Z)", gallery)
        _settle(app)
        tips = [w for w in app.topLevelWidgets() if w.inherits("QTipLabel") and w.isVisible()]
        if tips:
            save(tips[0], f"tooltip_{theme}.png")
        QToolTip.hideText()

        for widget in (dialog, menu, focus_host, gallery, shell):
            widget.close()
            widget.deleteLater()
        _settle(app)

    return written


# --------------------------------------------------------------------------- 실행


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    _prepare_environment(args)

    from PySide6.QtWidgets import QApplication

    from ui.theme import ThemeManager, configure_hidpi

    configure_hidpi()
    app = QApplication.instance() or QApplication(sys.argv[:1])
    manager = ThemeManager.instance()
    manager.apply(app)

    if args.shot:
        files = export_screenshots(app, args.shot)
        print(f"{len(files)}개 저장: {Path(args.shot).resolve()}")
        for path in files:
            print(" ", path.name)
        return 0

    if args.theme:
        manager.set_mode(args.theme, persist=False)

    if args.gallery:
        window, _sections = build_gallery()
    else:
        window = build_shell(args.legacy)
        window.navigate(args.page)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
