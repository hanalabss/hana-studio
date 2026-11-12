import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLineEdit, QVBoxLayout, QWidget

from ui.theme import DARK, LIGHT, mark_themed
from ui.widgets import (
    FIELD_MIME,
    ICON_NAMES,
    Banner,
    Button,
    Card,
    ComboBox,
    DangerButton,
    Dialog,
    Divider,
    EmptyState,
    FieldChip,
    FilterChip,
    FormRow,
    IconButton,
    IconWidget,
    MmField,
    Panel,
    PrimaryButton,
    ProgressBar,
    QuietButton,
    RibbonProgress,
    SearchField,
    SecondaryButton,
    SegmentedControl,
    StatusPill,
    StepperField,
    SwatchPicker,
    Toast,
    icon_pixmap,
    label,
    themed_icon,
)


def count_pixels(image, hex_color: str) -> int:
    target = QColor(hex_color).rgb()
    return sum(
        1
        for y in range(image.height())
        for x in range(image.width())
        if image.pixel(x, y) == target
    )


def host(*widgets) -> QWidget:
    box = QWidget()
    box.setObjectName("GalleryRoot")
    mark_themed(box)
    layout = QVBoxLayout(box)
    for widget in widgets:
        layout.addWidget(widget)
    return box


# ------------------------------------------------------------------ 버튼


@pytest.mark.parametrize("cls,variant", [
    (PrimaryButton, "primary"), (SecondaryButton, "secondary"), (QuietButton, "quiet"), (DangerButton, "danger"),
])
def test_button_variants_click(cls, variant, show, spy):
    button = cls("저장")
    clicked = spy(button.clicked)
    show(host(button))
    assert button.variant() == variant
    assert button.property("variant") == variant
    QTest.mouseClick(button, Qt.LeftButton)
    assert len(clicked) == 1


@pytest.mark.parametrize("size,height", [("sm", 32), ("md", 40), ("lg", 48)])
def test_button_sizes_have_token_heights(size, height, show):
    button = Button("다음: 명단 넣기", "primary", size)
    show(host(button))
    assert button.height() == height
    assert button.width() > button.fontMetrics().horizontalAdvance(button.text()), "글자가 잘리면 안 된다"


def test_primary_button_is_painted_in_accent(show):
    button = PrimaryButton("인쇄 시작")
    show(host(button))
    assert count_pixels(button.grab().toImage(), LIGHT.hex("primary_bg")) > 200


def test_disabled_button_does_not_click(show, spy):
    button = PrimaryButton("인쇄 시작")
    button.setEnabled(False)
    clicked = spy(button.clicked)
    show(host(button))
    QTest.mouseClick(button, Qt.LeftButton)
    assert len(clicked) == 0
    assert count_pixels(button.grab().toImage(), LIGHT.hex("primary_bg")) == 0


def test_button_rejects_unknown_variant_and_size():
    with pytest.raises(ValueError):
        Button("x", "fancy")
    with pytest.raises(ValueError):
        Button("x", "primary", "xl")


def test_button_icon_follows_theme(theme_manager):
    button = SecondaryButton("되돌리기", icon="undo")
    assert button.icon_name() == "undo"
    before = button.icon().pixmap(18, 18).toImage()
    theme_manager.set_mode("dark")
    after = button.icon().pixmap(18, 18).toImage()
    assert count_pixels(before, LIGHT.hex("text")) > 0
    assert count_pixels(after, DARK.hex("text")) > 0
    assert count_pixels(after, LIGHT.hex("text")) == 0


def test_icon_button_has_accessible_name_and_is_square(show):
    button = IconButton("chevron-right", "다음 사람")
    show(host(button))
    assert button.accessibleName() == "다음 사람"
    assert button.toolTip() == "다음 사람"
    assert button.width() == button.height() == 40


def test_keyboard_focus_draws_ring_but_mouse_focus_does_not(show, qapp):
    first = QLineEdit()
    button = SecondaryButton("저장")
    show(host(first, button))
    focus = LIGHT.hex("focus")

    button.setFocus(Qt.MouseFocusReason)
    qapp.processEvents()
    assert button.hasFocus()
    assert count_pixels(button.grab().toImage(), focus) == 0

    first.setFocus()
    QTest.keyClick(first, Qt.Key_Tab)
    qapp.processEvents()
    assert button.hasFocus()
    assert count_pixels(button.grab().toImage(), focus) > 100

    first.setFocus(Qt.MouseFocusReason)
    button.setFocus(Qt.MouseFocusReason)
    assert count_pixels(button.grab().toImage(), focus) == 0
    QTest.keyClick(button, Qt.Key_Shift)
    assert count_pixels(button.grab().toImage(), focus) > 100, "키를 누르기 시작하면 링이 보인다"


def test_filled_buttons_draw_focus_ring_inside(show, qapp):
    first = QLineEdit()
    primary = PrimaryButton("인쇄 시작")
    show(host(first, primary))
    before = count_pixels(primary.grab().toImage(), LIGHT.hex("primary_text"))
    primary.setFocus(Qt.TabFocusReason)
    qapp.processEvents()
    after = count_pixels(primary.grab().toImage(), LIGHT.hex("primary_text"))
    assert after - before > 150, "속이 찬 버튼은 글자색 링을 안쪽에 그린다"


def test_text_input_shows_focus_border(show, qapp):
    edit = QLineEdit("김하늘")
    other = QLineEdit()
    show(host(other, edit))
    other.setFocus()
    qapp.processEvents()
    assert count_pixels(edit.grab().toImage(), LIGHT.hex("focus")) == 0
    edit.setFocus(Qt.TabFocusReason)
    edit.deselect()
    qapp.processEvents()
    assert count_pixels(edit.grab().toImage(), LIGHT.hex("focus")) > 100


# ------------------------------------------------------------------ 세그먼트, 칩


def test_segmented_control_click_and_programmatic(show, spy):
    control = SegmentedControl([("front", "앞면"), ("back", "뒷면")])
    changed = spy(control.currentChanged)
    show(host(control))
    assert control.keys() == ["front", "back"]
    assert control.current() == "front"

    QTest.mouseClick(control.button("back"), Qt.LeftButton)
    assert control.current() == "back"
    assert changed.calls == [("back",)]

    QTest.mouseClick(control.button("back"), Qt.LeftButton)
    assert len(changed) == 1, "이미 고른 것을 다시 눌러도 신호는 한 번"
    assert control.current() == "back"

    control.set_current("front", emit=False)
    assert control.current() == "front"
    assert len(changed) == 1

    control.set_current("back")
    assert changed.last == ("back",)
    with pytest.raises(KeyError):
        control.set_current("side")


def test_segmented_control_plain_strings_and_arrow_keys(show, spy, qapp):
    control = SegmentedControl(["컬러", "화이트", "UV"], current="화이트")
    changed = spy(control.currentChanged)
    show(host(control))
    assert control.current() == "화이트"
    control.button("화이트").setFocus(Qt.TabFocusReason)
    QTest.keyClick(control.button("화이트"), Qt.Key_Right)
    qapp.processEvents()
    assert control.current() == "UV"
    assert changed.calls == [("UV",)]


def test_segmented_selected_segment_uses_strong_colours(show):
    control = SegmentedControl(["앞면", "뒷면"])
    show(host(control))
    image = control.button("앞면").grab().toImage()
    assert count_pixels(image, LIGHT.hex("strong_bg")) > 200


def test_filter_chip_toggle_and_count(show, spy):
    chip = FilterChip("아직 안 뽑음", 41)
    toggled = spy(chip.toggled)
    show(host(chip))
    assert chip.text() == "아직 안 뽑음 41"
    assert not chip.isChecked()
    QTest.mouseClick(chip, Qt.LeftButton)
    assert chip.isChecked() and toggled.calls == [(True,)]
    QTest.mouseClick(chip, Qt.LeftButton)
    assert not chip.isChecked() and toggled.last == (False,)

    chip.set_count(1280)
    assert chip.text() == "아직 안 뽑음 1,280"
    assert chip.count() == 1280 and chip.label() == "아직 안 뽑음"
    chip.set_count(None)
    assert chip.text() == "아직 안 뽑음"


def test_filter_chip_warn_tone_is_painted_in_warn_colours(show):
    chip = FilterChip("사진 없음", 4, tone="warn")
    show(host(chip))
    assert count_pixels(chip.grab().toImage(), LIGHT.hex("warn_bg")) > 200


def test_field_chip_mime_click_and_keyboard(show, spy):
    chip = FieldChip("이름")
    clicked = spy(chip.clicked)
    show(host(chip))
    mime = chip.mime_data()
    assert mime.hasFormat(FIELD_MIME)
    assert bytes(mime.data(FIELD_MIME)).decode("utf-8") == "이름"
    assert mime.text() == "이름"

    QTest.mouseClick(chip, Qt.LeftButton)
    assert clicked.calls == [("이름",)]
    QTest.keyClick(chip, Qt.Key_Return)
    assert len(clicked) == 2

    assert count_pixels(chip.grab().toImage(), LIGHT.hex("field_bg")) > 200
    chip.set_name("부서")
    assert chip.name() == "부서" and chip.text() == "부서"
    assert "부서" in chip.accessibleName()


def test_field_chip_drag_starts_after_threshold(show, spy, monkeypatch):
    chip = FieldChip("사번")
    started = spy(chip.dragStarted)
    show(host(chip))
    from PySide6.QtGui import QDrag

    monkeypatch.setattr(QDrag, "exec", lambda self, *args: Qt.CopyAction)
    QTest.mousePress(chip, Qt.LeftButton, Qt.NoModifier, QPoint(10, 10))
    QTest.mouseMove(chip, QPoint(60, 14))
    assert started.calls == [("사번",)]


@pytest.mark.parametrize("state,bg_role", [
    ("ready", "ok_bg"), ("busy", "info_bg"), ("error", "danger_bg"), ("offline", "idle_bg"),
])
def test_status_pill_states(state, bg_role, show, spy):
    pill = StatusPill("RTAI R600", state)
    clicked = spy(pill.clicked)
    show(host(pill))
    assert pill.state() == state
    assert count_pixels(pill.grab().toImage(), LIGHT.hex(bg_role)) > 200
    QTest.mouseClick(pill, Qt.LeftButton)
    assert len(clicked) == 1


def test_status_pill_updates_and_rejects_unknown_state():
    pill = StatusPill("프린터 연결 안 됨")
    assert pill.state() == "offline"
    pill.set_status("ready", "RTAI R600 준비됨")
    assert (pill.state(), pill.text()) == ("ready", "RTAI R600 준비됨")
    with pytest.raises(ValueError):
        pill.set_status("sleeping")


# ------------------------------------------------------------------ 상자, 폼


def test_card_and_panel(show):
    card = Card()
    card.body.addWidget(label("내용"))
    panel = Panel("인쇄", "선택한 12명을 뽑습니다.")
    extra = QuietButton("인쇄 설정 바꾸기", "sm")
    panel.add_header_widget(extra)
    sunken = Card(sunken=True)
    show(host(card, panel, sunken, Divider()))
    assert card.objectName() == "Card"
    assert panel.title() == "인쇄"
    panel.set_title("이렇게 나옵니다")
    assert panel.title() == "이렇게 나옵니다"
    assert count_pixels(card.grab().toImage(), LIGHT.hex("surface")) > 500
    assert count_pixels(sunken.grab().toImage(), LIGHT.hex("surface_sunken")) > 500


@pytest.mark.parametrize("orientation", ["vertical", "horizontal"])
def test_form_row_label_hint_and_accessibility(orientation, show):
    edit = QLineEdit()
    row = FormRow("사번", edit, "8글자로 씁니다.", orientation=orientation)
    show(host(row))
    assert row.label_widget().text() == "사번"
    assert row.label_widget().buddy() is edit
    assert edit.accessibleName() == "사번"
    assert row.hint() == "8글자로 씁니다."
    row.set_hint("사번은 8글자여야 합니다.", "error")
    assert row.hint() == "사번은 8글자여야 합니다."
    row.set_hint(None)
    assert row.hint() == ""
    assert row.control is edit


def test_combo_box_and_search_field(show):
    combo = ComboBox(["명단의 이름", "명단의 부서"])
    search = SearchField("이름, 부서, 사번으로 찾기")
    show(host(combo, search))
    assert combo.count() == 2 and combo.height() == 40
    combo.setCurrentIndex(1)
    assert combo.currentText() == "명단의 부서"
    assert search.placeholderText() == "이름, 부서, 사번으로 찾기"
    QTest.keyClicks(search, "kim")
    assert search.text() == "kim"


# ------------------------------------------------------------------ 숫자 칸


def test_mm_field_steps_clamps_and_formats(show, spy):
    field = MmField(30.5, minimum=0, maximum=31.0)
    changed = spy(field.valueChanged)
    show(host(field))
    assert field.text() == "30.5 mm"

    QTest.mouseClick(field._plus, Qt.LeftButton)
    assert field.value() == 31.0 and changed.calls == [(31.0,)]
    assert not field._plus.isEnabled(), "끝값에서는 늘리기 버튼을 쓸 수 없다"
    field.step_by(1)
    assert field.value() == 31.0 and len(changed) == 1

    QTest.mouseClick(field._minus, Qt.LeftButton)
    assert field.value() == 30.5
    field.set_value(-5)
    assert field.value() == 0.0 and not field._minus.isEnabled()
    assert field.text() == "0.0 mm"


def test_stepper_typed_value_and_arrow_keys(show, spy):
    field = StepperField(25, 6, 96, 1, 0, "pt")
    changed = spy(field.valueChanged)
    show(host(field, QLineEdit()))
    edit = field.line_edit()
    edit.setFocus()
    edit.selectAll()
    QTest.keyClicks(edit, "40")
    QTest.keyClick(edit, Qt.Key_Return)
    assert field.value() == 40 and field.text() == "40 pt"
    assert changed.last == (40.0,)

    QTest.keyClick(edit, Qt.Key_Up)
    assert field.value() == 41
    QTest.keyClick(edit, Qt.Key_Down)
    QTest.keyClick(edit, Qt.Key_Down)
    assert field.value() == 39

    edit.setText("글자")  # QTest.keyClicks는 한글을 못 친다
    QTest.keyClick(edit, Qt.Key_Return)
    assert field.value() == 39 and field.text() == "39 pt", "숫자가 아니면 이전 값으로 돌아간다"

    edit.selectAll()
    QTest.keyClicks(edit, "500")
    QTest.keyClick(edit, Qt.Key_Return)
    assert field.value() == 96


def test_stepper_silent_set_and_disabled(show, spy):
    field = MmField(10)
    changed = spy(field.valueChanged)
    field.set_value(12.5, emit=False)
    assert field.value() == 12.5 and len(changed) == 0
    field.setEnabled(False)
    show(host(field))
    assert not field._plus.isEnabled() and not field._minus.isEnabled()
    field.setAccessibleName("가로 위치")
    assert field.line_edit().accessibleName() == "가로 위치"


# ------------------------------------------------------------------ 색


def test_swatch_picker_click_and_custom(show, spy):
    picker = SwatchPicker([("검정", "#16181C"), ("흰색", "#FFFFFF"), ("파랑", "#2456E6")], current="#16181C")
    changed = spy(picker.colorChanged)
    show(host(picker))
    assert picker.current() == "#16181C"
    assert picker.colors() == ["#16181C", "#FFFFFF", "#2456E6"]

    QTest.mouseClick(picker._swatches["#2456E6"], Qt.LeftButton)
    assert picker.current() == "#2456E6" and changed.calls == [("#2456E6",)]

    picker.set_current("#0e6b5c")
    assert picker.current() == "#0E6B5C" and changed.last == ("#0E6B5C",)
    assert "#0E6B5C" in picker.colors()

    picker.set_current("#AA0000")
    assert picker.current() == "#AA0000" and changed.last == ("#AA0000",)
    assert "#0E6B5C" not in picker.colors(), "직접 고른 색 칸은 하나만 둔다"
    assert count_pixels(picker.grab().toImage(), "#AA0000") > 100


# ------------------------------------------------------------------ 알림


@pytest.mark.parametrize("kind,bg_role", [
    ("info", "info_bg"), ("ok", "ok_bg"), ("warn", "warn_bg"), ("error", "danger_bg"),
])
def test_banner_kinds_and_action(kind, bg_role, show, spy):
    banner = Banner(kind, "선택한 사람 중 1명은 사진이 없습니다.", "사진 넣기", closable=True)
    action = spy(banner.actionTriggered)
    dismissed = spy(banner.dismissed)
    show(host(banner))
    assert banner.kind() == kind
    assert count_pixels(banner.grab().toImage(), LIGHT.hex(bg_role)) > 500

    QTest.mouseClick(banner.action_button(), Qt.LeftButton)
    assert len(action) == 1 and banner.isVisible()
    QTest.mouseClick(banner._close, Qt.LeftButton)
    assert len(dismissed) == 1 and not banner.isVisible()


def test_banner_can_change_kind_and_text(show):
    banner = Banner("info", "가져오는 중입니다.")
    show(host(banner))
    assert not banner.action_button().isVisible()
    banner.set_kind("error")
    banner.set_text("파일을 읽지 못했습니다. 엑셀에서 닫은 뒤 다시 가져오세요.")
    banner.set_action("다시 가져오기")
    assert banner.kind() == "error" and banner.action_button().isVisible()
    assert count_pixels(banner.grab().toImage(), LIGHT.hex("danger_bg")) > 500
    with pytest.raises(ValueError):
        Banner("loud", "x")


def test_toast_shows_inside_host_and_action_closes_it(show, spy, qapp):
    window = QWidget()
    window.resize(800, 500)
    show(window)
    hits = []
    toast = Toast.show_message(window, "3명을 명단에서 지웠습니다.", action_text="되돌리기",
                               on_action=lambda: hits.append(1), timeout_ms=0)
    closed = spy(toast.closed)
    qapp.processEvents()
    assert toast.isVisible() and toast.text() == "3명을 명단에서 지웠습니다."
    assert window.rect().contains(toast.geometry())
    assert toast.geometry().bottom() <= window.height() - 24
    assert abs(toast.geometry().center().x() - window.width() // 2) <= 1

    window.resize(600, 400)
    qapp.processEvents()
    assert abs(toast.geometry().center().x() - 300) <= 1

    QTest.mouseClick(toast.action_button(), Qt.LeftButton)
    assert hits == [1] and len(closed) == 1 and not toast.isVisible()


def test_toast_replaces_previous_and_times_out(show, qapp, spy):
    window = QWidget()
    window.resize(800, 500)
    show(window)
    first = Toast.show_message(window, "저장했습니다.", timeout_ms=0)
    second = Toast.show_message(window, "다시 저장했습니다.", kind="ok", timeout_ms=30)
    assert not first.isVisible() and second.isVisible()
    assert window._hana_toast is second
    closed = spy(second.closed)
    QTest.qWait(120)
    assert len(closed) == 1, "정한 시간이 지나면 스스로 닫힌다"
    assert window._hana_toast is None
    assert not window.findChildren(Toast), "닫힌 알림은 창에 남지 않는다"


def test_empty_state_texts_and_actions(show, spy):
    empty = EmptyState("아직 명단이 없습니다", "엑셀이나 CSV 파일을 가져오세요.", "엑셀·CSV 가져오기",
                       icon="people", secondary_text="한 명씩 직접 넣기")
    action = spy(empty.actionTriggered)
    secondary = spy(empty.secondaryTriggered)
    show(host(empty))
    assert empty.title() == "아직 명단이 없습니다"
    assert empty.guidance() == "엑셀이나 CSV 파일을 가져오세요."
    QTest.mouseClick(empty.action_button(), Qt.LeftButton)
    QTest.mouseClick(empty.secondary_button(), Qt.LeftButton)
    assert len(action) == 1 and len(secondary) == 1
    empty.set_action(None)
    assert not empty.action_button().isVisible()


def test_empty_state_guidance_wraps_instead_of_clipping(show):
    text = "엑셀이나 CSV 파일을 가져오면 사람마다 이름과 사진이 바뀐 카드를 만듭니다. 원본 파일은 고치지 않습니다."
    empty = EmptyState("아직 명단이 없습니다", text)
    box = host(empty)
    box.resize(900, 600)
    show(box)
    guidance = empty._guidance
    needed = guidance.heightForWidth(guidance.width())
    assert needed > guidance.fontMetrics().height(), "이 글은 한 줄에 들어가지 않는다"
    assert guidance.height() >= needed, "줄이 바뀐 글이 모두 보여야 한다"


# ------------------------------------------------------------------ 진행


def test_progress_bar_fraction(show):
    bar = ProgressBar()
    bar.set_fraction(17 / 41)
    assert bar.value() == 41
    bar.set_fraction(3)
    assert bar.value() == 100
    box = host(bar)
    box.resize(400, 60)
    show(box)
    assert count_pixels(bar.grab().toImage(), LIGHT.hex("accent")) > 100


def test_ribbon_progress_states():
    ribbon = RibbonProgress()
    assert ribbon.panels() == ("y", "m", "c", "k", "w")
    assert ribbon.current_index() == -1 and ribbon.fractions() == [0, 0, 0, 0, 0]
    assert ribbon.accessibleName() == "리본 인쇄 대기"

    ribbon.set_progress(2, 0.55)
    assert ribbon.fractions() == [1.0, 1.0, 0.55, 0.0, 0.0]
    assert ribbon.current_index() == 2
    assert ribbon.accessibleName() == "파랑 인쇄 중 55%"

    ribbon.set_panel("k", 2.0)
    assert ribbon.fractions() == [1.0, 1.0, 1.0, 1.0, 0.0]

    ribbon.finish()
    assert ribbon.fractions() == [1.0] * 5 and ribbon.accessibleName() == "리본 인쇄 끝"
    ribbon.reset()
    assert ribbon.fractions() == [0.0] * 5 and ribbon.current_index() == -1
    with pytest.raises(ValueError):
        RibbonProgress(("y", "x"))


@pytest.mark.parametrize("theme", [LIGHT, DARK], ids=["light", "dark"])
def test_ribbon_progress_paints_current_segment_partly_filled(theme, theme_manager, show):
    theme_manager.set_mode(theme.name)
    ribbon = RibbonProgress()
    ribbon.set_progress(2, 0.5)
    box = QWidget()
    layout = QHBoxLayout(box)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(ribbon)
    box.resize(508, ribbon.sizeHint().height())
    show(box)
    image = ribbon.grab().toImage()
    segment = (508 - 8 * 4) / 5
    y = 5

    def at(index: int, fraction: float) -> str:
        return image.pixelColor(round(index * (segment + 8) + segment * fraction), y).name().upper()

    assert at(0, 0.5) == theme.hex("ribbon_y")
    assert at(1, 0.5) == theme.hex("ribbon_m")
    assert at(2, 0.25) == theme.hex("ribbon_c"), "지금 찍는 패널의 앞쪽은 채워진다"
    assert at(2, 0.75) == theme.hex("ribbon_track"), "지금 찍는 패널의 뒤쪽은 아직 비어 있다"
    assert at(3, 0.5) == theme.hex("ribbon_track")
    assert at(4, 0.5) == theme.hex("ribbon_track")


# ------------------------------------------------------------------ 대화 상자


def test_dialog_scaffold_accept_and_reject(show):
    dialog = Dialog("명단에서 12명을 지울까요?", "원본 엑셀 파일은 그대로 있습니다.")
    body = QLineEdit()
    dialog.set_body(body)
    cancel = dialog.add_button("그만두기", "secondary", "reject")
    confirm = dialog.add_button("12명 지우기", "danger", "accept")
    extra = dialog.add_button("자세히", "quiet", None, left=True)
    show(dialog)
    assert dialog.windowTitle() == dialog.title() == "명단에서 12명을 지울까요?"
    assert confirm.variant() == "danger" and cancel.variant() == "secondary"
    assert extra.geometry().left() < cancel.geometry().left() < confirm.geometry().left()

    QTest.mouseClick(extra, Qt.LeftButton)
    assert dialog.isVisible(), "role이 없는 버튼은 창을 닫지 않는다"
    QTest.mouseClick(confirm, Qt.LeftButton)
    assert dialog.result() == QDialog.Accepted and not dialog.isVisible()

    again = Dialog("저장할까요?")
    no = again.add_button("저장 안 함", "secondary", "reject")
    show(again)
    QTest.mouseClick(no, Qt.LeftButton)
    assert again.result() == QDialog.Rejected


# ------------------------------------------------------------------ 아이콘


@pytest.mark.parametrize("name", ICON_NAMES)
def test_every_icon_draws_in_the_requested_colour(name):
    pixmap = icon_pixmap(name, 24, QColor(LIGHT.hex("accent")), dpr=1.0)
    image = pixmap.toImage()
    assert image.width() == 24 and image.height() == 24
    opaque = [
        image.pixelColor(x, y)
        for y in range(24) for x in range(24)
        if image.pixelColor(x, y).alpha() > 200
    ]
    assert len(opaque) >= 8, f"{name} 아이콘이 거의 비어 있습니다"
    accent = QColor(LIGHT.hex("accent"))
    assert all(abs(c.red() - accent.red()) < 6 and abs(c.blue() - accent.blue()) < 6 for c in opaque)


def test_icon_helpers_follow_theme_and_scale(theme_manager, show):
    assert icon_pixmap("check", 20, "text", dpr=2.0).width() == 40
    widget = IconWidget("printer", 20, "text")
    show(host(widget))
    assert count_pixels(widget.grab().toImage(), LIGHT.hex("text")) > 0
    theme_manager.set_mode("dark")
    assert count_pixels(widget.grab().toImage(), DARK.hex("text")) > 0
    assert not themed_icon("print").isNull()
    with pytest.raises(KeyError):
        icon_pixmap("no-such-icon")


# ------------------------------------------------------------------ 기본 Qt 컨트롤의 테마


@pytest.mark.parametrize("theme", [LIGHT, DARK], ids=["light", "dark"])
def test_native_controls_take_theme_colours(theme, theme_manager, show):
    from PySide6.QtWidgets import QCheckBox, QSpinBox, QTableWidget

    theme_manager.set_mode(theme.name)
    check = QCheckBox("양면")
    check.setChecked(True)
    spin = QSpinBox()
    combo = ComboBox(["맑은 고딕"])
    edit = QLineEdit("김하늘")
    off = QLineEdit("잠김")
    off.setEnabled(False)
    table = QTableWidget(2, 2)
    table.setHorizontalHeaderLabels(["이름", "부서"])
    box = host(check, spin, combo, edit, off, table)
    show(box)
    assert count_pixels(check.grab().toImage(), theme.hex("accent")) > 50
    for widget in (spin, combo, edit):
        assert count_pixels(widget.grab().toImage(), theme.hex("surface_raised")) > 500
    assert count_pixels(off.grab().toImage(), theme.hex("surface_sunken")) > 500
    assert count_pixels(table.horizontalHeader().grab().toImage(), theme.hex("surface_sunken")) > 200
    assert count_pixels(box.grab().toImage(), theme.hex("ground")) > 500
    assert spin.height() == combo.height() == edit.height() == 40, "한 줄 입력 칸은 모두 같은 높이"
