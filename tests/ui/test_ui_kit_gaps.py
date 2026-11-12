"""화면 작성자들이 따로 메워 쓰던 키트 빈틈: 대화 상자 제목, 날짜 칸과 달력, 정렬 표시, 아이콘,
세그먼트 너비, 폼 줄 맞춤, 칸 칩의 끌기 형식."""

import pytest
from PySide6.QtCore import QDate, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QIcon, QImage, QPainter
from PySide6.QtWidgets import (
    QAbstractItemView, QCalendarWidget, QDateEdit, QLineEdit, QTableWidget, QToolButton, QVBoxLayout, QWidget,
)

from ui.theme import DARK, LIGHT, THEMES, ThemeManager, build_stylesheet, mark_themed
from ui.widgets import (
    CANVAS_FIELD_MIME, FIELD_MIME, ICON_NAMES, ComboBox, DateField, Dialog, FieldChip, FormRow, SegmentedControl,
    StepperField, align_form_rows, icon_pixmap, style_calendar, themed_icon,
)

KOREAN_SEGMENTS = [("system", "Windows 설정 따르기"), ("light", "밝게"), ("dark", "어둡게")]


def themed_host(*widgets: QWidget, width: int = 520) -> QWidget:
    host = QWidget()
    mark_themed(host)
    layout = QVBoxLayout(host)
    for widget in widgets:
        layout.addWidget(widget)
    host.resize(width, host.sizeHint().height())
    return host


def colours(image: QImage) -> set[str]:
    return {image.pixelColor(x, y).name().upper() for y in range(image.height()) for x in range(image.width())}


def same_pixels(a: QImage, b: QImage) -> bool:
    return a.size() == b.size() and all(
        a.pixel(x, y) == b.pixel(x, y) for y in range(a.height()) for x in range(a.width()))


# --------------------------------------------------------------------------- 1. 대화 상자 제목


def test_dialog_title_can_be_changed_publicly(qapp):
    dialog = Dialog("명단 가져오기", "파일을 고르세요.")
    assert (dialog.title(), dialog.windowTitle(), dialog.description()) == ("명단 가져오기", "명단 가져오기", "파일을 고르세요.")

    dialog.set_title("사원.xlsx에서 128명을 찾았습니다")
    assert dialog.title() == "사원.xlsx에서 128명을 찾았습니다" == dialog.windowTitle()

    dialog.set_title("모두 가져왔습니다", window_title=False)
    assert dialog.title() == "모두 가져왔습니다"
    assert dialog.windowTitle() == "사원.xlsx에서 128명을 찾았습니다"


# --------------------------------------------------------------------------- 2. 날짜 칸, 달력, 정렬 표시


@pytest.mark.parametrize("theme", [LIGHT, DARK], ids=["light", "dark"])
def test_stylesheet_covers_date_calendar_and_sort_arrows(theme, theme_manager):
    from ui.theme.assets import ensure_assets

    assets = ensure_assets(theme)
    for name in ("calendar", "calendar-disabled", "chevron-left", "sort-asc", "sort-desc"):
        assert name in assets
    qss = build_stylesheet(theme, theme_manager.font_spec, assets)
    assert 'QDateTimeEdit[themed="true"][calendarPopup="true"]::down-arrow' in qss
    assert 'QDateTimeEdit[themed="true"]::drop-down' in qss
    assert 'QHeaderView[themed="true"]::up-arrow' in qss and 'QHeaderView[themed="true"]::down-arrow' in qss
    # 달력은 따로 뜨는 창이라 themed 범위 밖에서도 먹어야 한다.
    for line in qss.splitlines():
        if "qt_calendar_" in line or line.startswith("QCalendarWidget"):
            assert "themed" not in line, line
    for selector in ("QCalendarWidget {", "QWidget#qt_calendar_navigationbar", "QToolButton#qt_calendar_prevmonth",
                     "QSpinBox#qt_calendar_yearedit", "QAbstractItemView#qt_calendar_calendarview"):
        assert selector in qss, selector


@pytest.mark.parametrize("theme_name", ["light", "dark"])
def test_date_field_drop_down_shows_the_themed_calendar_icon(qapp, show, theme_manager, theme_name):
    theme_manager.set_mode(theme_name, persist=False)
    theme = THEMES[theme_name]
    field = DateField(QDate(2026, 11, 12))
    plain = QLineEdit("2026-11-12")
    plain.setFixedHeight(40)
    show(themed_host(field, plain))

    assert field.calendarPopup() and field.height() == 40 and field.text() == "2026-11-12"
    button = field.grab().toImage().copy(field.width() - 32, 4, 28, 32)
    blank = plain.grab().toImage().copy(plain.width() - 32, 4, 28, 32)
    seen = colours(button)
    assert theme.hex("text_secondary") in seen, "달력 아이콘이 테마 색으로 그려져야 한다"
    assert not same_pixels(button, blank)
    # 기본 Qt 모양(회색 단추 면, 검은 세모)이 아니다: 단추 자리의 바탕은 입력 칸 바탕 그대로다.
    assert button.pixelColor(1, 1).name().upper() == theme.hex("surface_raised")


def test_date_field_empty_value(qapp):
    field = DateField(empty_text="날짜 없음")
    assert field.is_empty() and field.text() == "날짜 없음"
    field.setDate(QDate(2026, 3, 2))
    assert not field.is_empty() and field.text() == "2026-03-02"
    field.clear_date()
    assert field.is_empty()

    today = DateField()
    assert not today.is_empty() and today.date() == QDate.currentDate()
    today.clear_date()
    assert today.date() == QDate.currentDate(), "빈 값이 없는 칸은 지워지지 않는다"


@pytest.mark.parametrize("theme_name", ["light", "dark"])
def test_calendar_popup_is_themed_even_without_a_themed_parent(qapp, theme_manager, theme_name):
    theme_manager.set_mode(theme_name, persist=False)
    theme = THEMES[theme_name]
    field = DateField(QDate(2026, 11, 12))      # 부모도 없고 mark_themed도 하지 않은 칸
    calendar = field.calendarWidget()
    popup = calendar.window()
    assert popup is not field and bool(popup.windowFlags() & Qt.Popup)
    assert calendar.property("themed") == "true"

    popup.show()
    for _ in range(4):
        qapp.processEvents()
    image = popup.grab().toImage()
    seen = colours(image)

    bar = calendar.findChild(QWidget, "qt_calendar_navigationbar")
    bar_at = bar.mapTo(popup, bar.rect().center())
    assert image.pixelColor(bar_at.x(), bar_at.y() + 14).name().upper() == theme.hex("surface_sunken")
    assert image.pixelColor(0, image.height() // 2).name().upper() == theme.hex("control_line"), "바깥 테두리"
    assert theme.hex("accent") in seen, "고른 날은 강조색"
    assert theme.hex("surface_raised") in seen
    assert theme.hex("text_disabled") in seen, "다른 달 날짜는 흐리게"
    assert "#FF0000" not in seen, "기본 Qt의 빨간 주말 글씨가 남아 있다"
    assert calendar.weekdayTextFormat(Qt.Sunday).foreground().color() == theme.color("text")
    assert not calendar.isGridVisible()
    assert calendar.verticalHeaderFormat() == QCalendarWidget.NoVerticalHeader
    assert popup.width() >= 296 and popup.height() >= 300

    prev = calendar.findChild(QToolButton, "qt_calendar_prevmonth")
    assert prev.toolTip() == "이전 달" and prev.accessibleName() == "이전 달"
    assert not prev.icon().isNull()
    popup.hide()


def test_calendar_follows_a_theme_switch(qapp, theme_manager):
    field = DateField(QDate(2026, 11, 12))
    calendar = field.calendarWidget()
    assert calendar.headerTextFormat().foreground().color() == LIGHT.color("text_muted")
    theme_manager.set_mode("dark", persist=False)
    assert calendar.headerTextFormat().foreground().color() == DARK.color("text_muted")
    assert calendar.weekdayTextFormat(Qt.Saturday).foreground().color() == DARK.color("text")

    popup = calendar.window()
    popup.show()
    for _ in range(4):
        qapp.processEvents()
    seen = colours(popup.grab().toImage())
    assert DARK.hex("accent") in seen and LIGHT.hex("accent") not in seen
    assert LIGHT.hex("surface_raised") not in seen
    popup.hide()


def test_style_calendar_works_on_a_plain_calendar_and_is_repeatable(qapp, show):
    calendar = QCalendarWidget()
    calendar.setSelectedDate(QDate(2026, 11, 12))
    style_calendar(calendar)
    view = calendar.findChild(QAbstractItemView, "qt_calendar_calendarview")
    delegate = view.itemDelegate()
    style_calendar(calendar)
    assert view.itemDelegate() is delegate, "다시 불러도 그리는 객체를 또 만들지 않는다"
    show(calendar)
    seen = colours(calendar.grab().toImage())
    assert LIGHT.hex("accent") in seen and "#FF0000" not in seen


def test_plain_date_edit_without_popup_keeps_spin_arrows(qapp, show):
    spin = QDateEdit(QDate(2026, 11, 12))
    popup = QDateEdit(QDate(2026, 11, 12))
    popup.setCalendarPopup(True)
    show(themed_host(spin, popup))
    a = spin.grab().toImage().copy(spin.width() - 30, 0, 30, spin.height())
    b = popup.grab().toImage().copy(popup.width() - 30, 0, 30, popup.height())
    assert not same_pixels(a, b), "달력 아이콘은 달력을 여는 칸에만 붙는다"


@pytest.mark.parametrize("theme_name", ["light", "dark"])
def test_header_sort_arrows_are_themed(qapp, show, theme_manager, theme_name):
    theme_manager.set_mode(theme_name, persist=False)
    theme = THEMES[theme_name]
    table = QTableWidget(2, 2)
    table.setHorizontalHeaderLabels(["이름", "부서"])
    table.setColumnWidth(0, 160)
    header = table.horizontalHeader()
    show(themed_host(table))

    def arrow_zone() -> QImage:
        qapp.processEvents()
        return header.grab().toImage().copy(160 - 26, 0, 26, header.height())

    header.setSortIndicatorShown(False)
    none = arrow_zone()
    header.setSortIndicatorShown(True)
    header.setSortIndicator(0, Qt.AscendingOrder)
    ascending = arrow_zone()
    header.setSortIndicator(0, Qt.DescendingOrder)
    descending = arrow_zone()

    assert not same_pixels(none, ascending) and not same_pixels(ascending, descending)
    for image in (ascending, descending):
        assert theme.hex("text_secondary") in colours(image), "화살표는 테마 색 꺾쇠여야 한다"

    def spans(image: QImage) -> list[int]:
        """화살표가 그려진 줄마다, 화살표 잉크의 왼쪽 끝에서 오른쪽 끝까지의 폭(위에서 아래로)."""
        result = []
        for y in range(image.height()):
            xs = [x for x in range(image.width()) if image.pixel(x, y) != none.pixel(x, y)]
            if xs:
                result.append(xs[-1] - xs[0] + 1)
        return result

    # 오름차순은 위쪽 꺾쇠(∧: 위가 좁고 아래가 넓다), 내림차순은 아래쪽 꺾쇠(∨).
    up, down = spans(ascending), spans(descending)
    assert len(up) >= 3 and len(down) >= 3
    assert up[0] < up[-1], up
    assert down[0] > down[-1], down


# --------------------------------------------------------------------------- 3. 아이콘


MOVED_ICONS = (
    "shape", "rect", "ellipse", "line", "barcode", "qr", "eye", "eye-off", "lock", "unlock", "grid", "magnet",
    "cut", "fit", "layers", "card-blank", "sparkle", "save", "more", "align-left", "align-center", "align-right",
    "align-top", "align-middle", "align-bottom", "distribute-horizontal", "distribute-vertical", "calendar",
)


def test_kit_icon_set_has_the_design_icons():
    missing = [name for name in MOVED_ICONS if name not in ICON_NAMES]
    assert not missing
    assert len(set(ICON_NAMES)) == len(ICON_NAMES)


def test_design_icons_module_is_a_thin_layer_over_the_kit(qapp):
    from ui.pages.design import icons

    for name in ("eye", "lock", "barcode", "align-middle", "chevron-down", "text"):
        assert same_pixels(icons.pixmap(name, 18, "accent_text").toImage(),
                           icon_pixmap(name, 18, "accent_text").toImage()), name
    assert icons.pixmap("save").size() == icon_pixmap("save", 18).size()
    assert set(icons.KIND_ICONS.values()) <= set(ICON_NAMES)

    canvas = QImage(24, 24, QImage.Format_ARGB32_Premultiplied)
    canvas.fill(Qt.transparent)
    painter = QPainter(canvas)
    icons.paint(painter, "qr", QRectF(0, 0, 24, 24), QColor("black"))
    painter.end()
    assert any(canvas.pixelColor(x, y).alpha() > 0 for x in range(24) for y in range(24))

    with pytest.raises(KeyError):
        icons.pixmap("없는-아이콘")


def test_themed_icon_checked_role(qapp):
    plain = themed_icon("eye", "text", 18)
    toggled = themed_icon("eye", "text", 18, checked_role="accent_text")
    from PySide6.QtCore import QSize

    size = QSize(18, 18)
    off = toggled.pixmap(size, QIcon.Normal, QIcon.Off).toImage()
    on = toggled.pixmap(size, QIcon.Normal, QIcon.On).toImage()
    assert not same_pixels(off, on)
    assert LIGHT.hex("accent_text") in colours(on)
    assert same_pixels(plain.pixmap(size, QIcon.Normal, QIcon.On).toImage(),
                       plain.pixmap(size, QIcon.Normal, QIcon.Off).toImage())

    from ui.pages.design import icons

    design = icons.icon("eye", "text", 18, checked_role="accent_text")
    assert same_pixels(design.pixmap(size, QIcon.Normal, QIcon.On).toImage(), on)


# --------------------------------------------------------------------------- 4. 세그먼트 너비


def strong_advance(button) -> int:
    font = QFont(button.font())
    font.setWeight(QFont.Weight(ThemeManager.instance().font_spec.strong_weight))
    return QFontMetrics(font).horizontalAdvance(button.text())


@pytest.mark.parametrize("options", [
    KOREAN_SEGMENTS,
    [("fixed", "직접 입력"), ("column", "명단 항목")],
    ["앞·뒤 양면", "앞면만"],
], ids=["theme", "source", "sides"])
def test_segments_fit_their_bold_label_in_every_state(qapp, show, options):
    control = SegmentedControl(options)
    show(themed_host(control, width=700))
    padding = 2 * 14        # 스타일시트의 좌우 안쪽 여백

    def widths() -> dict[str, int]:
        qapp.processEvents()
        return {key: control.button(key).width() for key in control.keys()}

    first = widths()
    for key in control.keys():
        control.set_current(key)
        now = widths()
        assert now == first, "고르는 칸이 바뀌어도 너비가 흔들리지 않는다"
        for name in control.keys():
            button = control.button(name)
            assert button.width() - padding >= strong_advance(button), f"'{button.text()}'이 굵은 글씨일 때 잘린다"
            assert button.width() - padding >= button.fontMetrics().horizontalAdvance(button.text())


def test_stretched_segments_still_fit_bold_labels_when_squeezed(qapp, show):
    control = SegmentedControl(KOREAN_SEGMENTS, stretch=True)
    host = themed_host(control, width=700)
    show(host)
    host.resize(host.minimumSizeHint())
    qapp.processEvents()
    for key in control.keys():
        button = control.button(key)
        assert button.width() - 28 >= strong_advance(button)


def test_checked_segment_label_is_fully_painted(qapp, show):
    """굵은 글자를 실제로 그려서, 글자 잉크가 안쪽 여백을 넘어 가장자리까지 닿지 않는지 본다."""
    control = SegmentedControl(KOREAN_SEGMENTS, current="system")
    show(themed_host(control, width=700))
    button = control.button("system")
    image = button.grab().toImage()
    background = image.pixelColor(image.width() // 2, 3)
    assert background.name().upper() == LIGHT.hex("strong_bg")
    ink = [x for x in range(image.width())
           if any(image.pixelColor(x, y).name().upper() == LIGHT.hex("strong_text") for y in range(6, image.height() - 6))]
    assert ink, "글자가 보여야 한다"
    assert ink[0] >= 8 and ink[-1] <= image.width() - 9, "굵은 글자 양옆에 여백이 남아야 한다"


# --------------------------------------------------------------------------- 5. 폼 줄 맞춤


def test_form_row_left_alignment_hugs_the_control(qapp, show):
    segmented = SegmentedControl(KOREAN_SEGMENTS)
    stepper = StepperField(1, 1, 99, 1, 0, "장")
    stepper.setFixedWidth(200)
    combo = ComboBox(["1년", "3년"])
    left_a = FormRow("화면 색", segmented, "도움말", orientation="horizontal", label_width=150, control_align="left")
    left_b = FormRow("한 사람에 몇 장", stepper, orientation="horizontal", label_width=150, control_align="left")
    fill = FormRow("기록을 두는 기간", combo, orientation="horizontal", label_width=150)
    host = themed_host(left_a, left_b, fill, width=900)
    show(host)

    assert left_a.control is segmented and segmented.accessibleName() == "화면 색", "이름은 진짜 입력에 붙는다"
    assert left_a.label_widget().buddy() is segmented
    edge = 150 + 12     # 이름표 폭 + 간격
    assert segmented.x() == stepper.x() == combo.x() == edge
    assert segmented.width() == segmented.sizeHint().width() < left_a.width() - edge
    assert stepper.width() == 200
    assert combo.width() == fill.width() - edge, "fill은 줄 너비를 채운다"
    assert (left_a.control_align(), fill.control_align()) == ("left", "fill")


def test_form_row_left_alignment_in_vertical_rows(qapp, show):
    stepper = StepperField(1, 1, 99, 1, 0, "장")
    stepper.setFixedWidth(200)
    row = FormRow("몇 장", stepper, control_align="left")
    show(themed_host(row, width=600))
    assert stepper.x() == 0 and stepper.width() == 200
    assert row.orientation() == "vertical" and row.label_width() == 0
    row.set_label_width(300)
    assert row.label_widget().maximumWidth() > 300, "세로 줄의 이름표 폭은 건드리지 않는다"


def test_form_row_rejects_unknown_alignment(qapp):
    with pytest.raises(ValueError):
        FormRow("이름", QLineEdit(), control_align="center")


def test_align_form_rows_gives_one_label_column(qapp, show):
    rows = [
        FormRow("화면 색", QLineEdit(), orientation="horizontal", label_width=60),
        FormRow("기록을 두는 기간", QLineEdit(), orientation="horizontal", label_width=200),
        FormRow("한 사람에 몇 장", QLineEdit(), orientation="horizontal"),
    ]
    stacked = FormRow("세로 줄", QLineEdit())
    host = themed_host(*rows, stacked, width=700)
    show(host)

    width = align_form_rows(host)
    qapp.processEvents()
    longest = max(row.label_width_hint() for row in rows)
    assert width == longest
    assert {row.label_width() for row in rows} == {longest}
    assert len({row.control.x() for row in rows}) == 1, "입력 칸의 왼쪽 끝이 한 줄로 선다"
    for row in rows:
        assert row.label_widget().width() >= row.label_widget().sizeHint().width(), "이름표가 잘리지 않는다"
    assert stacked.control.x() == 0

    assert align_form_rows(rows, min_width=longest + 40) == longest + 40
    assert align_form_rows(rows, max_width=80) == 80
    assert align_form_rows([stacked]) == 0
    rows[0].set_label_width(123)
    assert rows[0].label_width() == 123


# --------------------------------------------------------------------------- 6. 칸 칩의 끌기 형식


def test_field_chip_mime_matches_the_editor_canvas():
    from ui.editor import MIME_FIELD
    from ui.editor.controller import FIELD_KINDS
    from ui.widgets.chips import FIELD_CHIP_KINDS

    assert CANVAS_FIELD_MIME == MIME_FIELD
    assert FIELD_CHIP_KINDS == FIELD_KINDS
    assert FIELD_MIME == "application/x-hana-field" != CANVAS_FIELD_MIME


@pytest.mark.parametrize("name, kind", [("이름", "text"), ("사진", "image"), ("사번", "barcode")])
def test_field_chip_drag_is_accepted_by_the_canvas_parser(qapp, name, kind):
    from ui.editor.controller import field_mime, parse_field_mime

    chip = FieldChip(name, kind=kind)
    mime = chip.mime_data()
    assert parse_field_mime(mime) == (name, kind)
    assert bytes(mime.data(CANVAS_FIELD_MIME)) == bytes(field_mime(name, kind).data(CANVAS_FIELD_MIME))
    assert bytes(mime.data(FIELD_MIME)).decode("utf-8") == name, "예전 형식도 그대로 싣는다"
    assert mime.text() == name


def test_field_chip_kind_defaults_and_validation(qapp):
    from ui.editor.controller import parse_field_mime

    chip = FieldChip("부서")
    assert chip.field_kind() == "text"
    chip.set_field_kind("image")
    chip.set_name("사진")
    assert parse_field_mime(chip.mime_data()) == ("사진", "image")
    with pytest.raises(ValueError):
        FieldChip("이름", kind="video")
    with pytest.raises(ValueError):
        chip.set_field_kind("video")
    assert chip.field_kind() == "image"
