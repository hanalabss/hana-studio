"""사진을 골랐을 때: 그림 바꾸기, 명단 사진 연결, 채우는 방법, 배경 지우기."""

from __future__ import annotations

import os

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication, QButtonGroup, QCheckBox, QFrame, QLabel, QRadioButton, QSlider, QVBoxLayout, QWidget,
)
from shiboken6 import isValid

from design import cutout, renderer, storage
from design.binding import ColumnSource, FixedSource
from printer.base import CAP_WHITE
from ui.widgets import Divider, ProgressBar, SecondaryButton, SegmentedControl, label

from ..background import (
    DEFAULT_THRESHOLD, MAX_THRESHOLD, MAX_TOLERANCE, METHOD_AI, METHOD_COLOR, MIN_THRESHOLD, MIN_TOLERANCE,
    CutoutSettings, CutoutWork, RemovalJob, RembgEngine, has_settings, save_cutout, save_mask_of,
)
from ..common import LinkButton, NoteRow, group, hbox, use_local_icon
from .base import InspectorContext, Section
from .detail import ColumnCombo

FIT_OPTIONS = (("fill", "꽉 채우기"), ("contain", "다 보이게"), ("stretch", "늘려 맞추기"))
FIT_HINTS = {
    "fill": "칸을 가득 채우고 넘치는 부분은 잘라 냅니다. 증명사진에 알맞습니다.",
    "contain": "그림 전체가 보이도록 줄입니다. 칸에 빈 곳이 생길 수 있습니다. 로고에 알맞습니다.",
    "stretch": "칸 모양에 맞춰 그림을 늘립니다. 그림이 찌그러질 수 있습니다.",
}
_PREVIEW_HEIGHT = 150

METHODS = (
    (METHOD_AI, "사람·물건만 남기기", "사람·물건 사진에 씁니다. 주인공을 찾아 나머지를 지웁니다."),
    (METHOD_COLOR, "바탕색만 지우기", "로고·도장·서명처럼 바탕이 흰색이나 한 가지 색인 그림에 씁니다."),
)
WHITE_LABEL = "사람·물건 밑에 흰색 깔기 (투명 카드용)"
WHITE_HINTS = {
    True: "투명 카드에서 사람·물건이 비치지 않게 그 밑에만 흰색을 찍습니다.",
    False: "지금 프린터는 흰색을 찍지 못해 꺼 두었습니다. 흰 카드에는 필요 없습니다.",
    None: "투명 카드에 찍을 때만 켭니다. 흰색을 찍는 프린터가 있어야 합니다.",
}
REVIEW_TEXTS = {
    "partial": "그림의 일부까지 함께 지워졌습니다. 로고·도장·서명처럼 바탕이 한 가지 색인 그림은 "
               "'바탕색만 지우기'가 더 잘 맞습니다.",
    "wipeout": "그림에서 주인공을 찾지 못했습니다. 이대로 쓰면 그림이 거의 다 지워집니다. "
               "로고·도장·서명 같은 그림이면 '바탕색만 지우기'로 해 보세요.",
    "untouched": "지울 배경을 찾지 못했습니다. 이대로 쓰면 그림이 거의 그대로 남습니다. "
                 "바탕이 한 가지 색이면 '바탕색만 지우기'로 해 보세요.",
}
BOUND_HINT = ("사람마다 사진이 달라 명단의 사진은 여기서 배경을 지울 수 없습니다. "
              "배경 지우기는 '그림 파일'로 넣은 그림 한 장에만 쓸 수 있습니다. "
              "사람 사진은 배경을 미리 지운 파일(PNG)을 명단 사진으로 써 주세요.")


class PreviewLabel(QLabel):
    """배경을 지운 모습. 누르면 그 자리(칸 안 좌표)를 알려 준다."""

    pressed = Signal(float, float)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.pressed.emit(event.position().x(), event.position().y())
        super().mousePressEvent(event)


class ColorChip(QFrame):
    """지울 색을 보여 주는 작은 칸."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(28, 20)
        self._color = "#ffffff"
        self.set_color(self._color)

    def color(self) -> str:
        return self._color

    def set_color(self, color: str) -> None:
        self._color = color
        line = "#00000055" if QColor(color).lightnessF() > 0.5 else "#ffffff66"
        self.setStyleSheet(f"background: {color}; border: 1px solid {line}; border-radius: 4px;")
        self.setAccessibleName(f"지울 색 {color}")


class ImageSection(Section):
    def __init__(self, context: InspectorContext, parent: QWidget | None = None) -> None:
        super().__init__(context, parent)
        self.engine = context.background_engine or RembgEngine()
        self._job: RemovalJob | None = None
        self._job_item_id: str | None = None
        self._job_method = METHOD_AI
        self._error: tuple[str, str] | None = None
        self._method = METHOD_AI
        # 항목 id -> CutoutWork. 가려낸 결과를 기억해 두어 정도를 바꿀 때 다시 계산하지 않는다.
        self._results: dict[str, CutoutWork] = context.state.setdefault("cutouts", {})
        # 항목 id -> (CutoutWork, 까닭). 그대로 쓰기엔 이상해서 사용자에게 물어보는 중인 결과.
        self._pending: dict[str, tuple[CutoutWork, str]] = {}
        self._chain_files: set[str] = set()

        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._on_quit)

        self.source_control = SegmentedControl([("file", "그림 파일"), ("column", "명단의 사진")], stretch=True)
        self.source_control.currentChanged.connect(self.set_mode)
        self.replace_button = SecondaryButton("그림 바꾸기", icon="image")
        self.replace_button.clicked.connect(self.choose_image)
        self.file_label = label("", "hint", wrap=True)
        self.file_box = self._box(self.replace_button, self.file_label)
        self.column_combo = ColumnCombo()
        self.column_combo.setAccessibleName("사진이 들어 있는 명단 항목")
        self.column_combo.activated.connect(lambda _i: self.set_column(self.column_combo.column()))
        self.column_hint = label("", "hint", wrap=True)
        self.column_box = self._box(self.column_combo, self.column_hint)
        self.column.addWidget(group("무엇을 보여 줄까요", self.source_control, self.file_box, self.column_box))

        self.fit_control = SegmentedControl(FIT_OPTIONS, stretch=True)
        self.fit_control.currentChanged.connect(self._on_fit)
        self.fit_hint = label("", "hint", wrap=True)
        self.fill_card_button = SecondaryButton("카드 전체로 맞추기", size="sm")
        self.fill_card_button.setToolTip("사진 칸을 카드 크기에 딱 맞게 키우고, 그림이 카드를 가득 채우게 합니다.")
        self.fill_card_button.clicked.connect(self.fill_card)
        self.column.addWidget(group("칸에 맞추는 방법", self.fit_control, self.fit_hint, self.fill_card_button))

        self.column.addWidget(Divider())
        self._build_cutout()

    def _build_cutout(self) -> None:
        self.cutout_title = label("배경 지우기", "strong")
        self.cutout_hint = label("", "hint", wrap=True)

        # 방법 고르기
        self.method_group = QButtonGroup(self)
        self.method_buttons: dict[str, QRadioButton] = {}
        rows: list[QWidget] = []
        for key, title, hint in METHODS:
            button = QRadioButton(title)
            button.setAccessibleDescription(hint)
            button.toggled.connect(lambda checked, key=key: checked and self._on_method(key))
            self.method_group.addButton(button)
            self.method_buttons[key] = button
            note = label(hint, "hint", wrap=True)
            note.setContentsMargins(30, 0, 0, 0)
            row = self._box(button, note)
            row.layout().setSpacing(0)
            rows.append(row)
        self.method_buttons[self._method].setChecked(True)
        self.method_box = self._box(*rows)

        self.remove_button = SecondaryButton("배경 지우기")
        use_local_icon(self.remove_button, "sparkle")
        self.remove_button.clicked.connect(self.start_removal)

        self.progress_bar = ProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_label = label("", "hint", wrap=True)
        self.cancel_button = SecondaryButton("그만두기", size="sm")
        self.cancel_button.clicked.connect(self.cancel_removal)
        self.progress_box = self._box(self.progress_bar, self.progress_label, hbox(self.cancel_button, None))
        self.error_row = NoteRow("problem", "")
        self.error_row.setCursor(Qt.ArrowCursor)

        # 그대로 쓰기 이상한 결과: 보여 주고 물어본다
        self.review_row = NoteRow("problem", "")
        self.review_row.setCursor(Qt.ArrowCursor)
        self.review_row.setFocusPolicy(Qt.NoFocus)
        self.try_color_button = SecondaryButton("바탕색만 지우기로 해 보기", size="sm")
        self.try_color_button.clicked.connect(self.try_color_method)
        self.accept_button = SecondaryButton("이대로 쓰기", size="sm")
        self.accept_button.clicked.connect(self.accept_pending)
        self.discard_button = SecondaryButton("그만두기", size="sm")
        self.discard_button.clicked.connect(self.discard_pending)
        self.review_buttons = self._box(self.try_color_button, hbox(self.accept_button, self.discard_button, None))

        self.done_row = NoteRow("ok", "")
        self.preview = PreviewLabel()
        self.preview.setFixedHeight(_PREVIEW_HEIGHT)
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setAccessibleName("배경을 지운 모습")
        self.preview.pressed.connect(self._on_preview_pressed)

        # 사람·물건: 지우는 정도
        self.threshold_slider = self._slider(MIN_THRESHOLD, MAX_THRESHOLD, "지우는 정도")
        self.slider_ends = hbox(label("더 남기기", "caption"), None, label("더 지우기", "caption"))
        self.threshold_group = group("지우는 정도", self.threshold_slider, self.slider_ends)

        # 바탕색: 지울 색과 비슷한 색까지
        self.color_chip = ColorChip()
        self.corner_button = LinkButton("모서리 색으로")
        self.corner_button.setToolTip("그림 모서리에 있는 색을 바탕색으로 봅니다")
        self.corner_button.clicked.connect(lambda: self.set_key_color(None))
        self.color_hint = label("위 그림에서 지우고 싶은 색을 누르면 그 색을 지웁니다.", "hint", wrap=True)
        self.tolerance_slider = self._slider(MIN_TOLERANCE, MAX_TOLERANCE, "비슷한 색까지 지우기")
        self.tolerance_ends = hbox(label("딱 그 색만", "caption"), None, label("비슷한 색까지", "caption"))
        self.color_group = self._box(
            group("지울 색", hbox(self.color_chip, self.corner_button, None), self.color_hint),
            group("비슷한 색까지 지우기", self.tolerance_slider, self.tolerance_ends))
        self.adjust_box = self._box(self.threshold_group, self.color_group)

        self.white_check = QCheckBox(WHITE_LABEL)
        self.white_check.toggled.connect(self._on_white_toggled)
        self.white_hint = label("", "hint", wrap=True)
        self.white_hint.setContentsMargins(30, 0, 0, 0)
        self.white_box = self._box(self.white_check, self.white_hint)
        self.white_box.layout().setSpacing(0)

        self.restore_button = SecondaryButton("배경 되살리기", size="sm")
        self.restore_button.setToolTip("지운 배경을 원래대로 돌려놓습니다")
        self.restore_button.clicked.connect(self.restore_background)
        self.redo_button = SecondaryButton("다시 다듬기", size="sm")
        self.redo_button.clicked.connect(self.start_removal)

        # 흰색을 깔 자리를 직접 만든 그림으로 정한다(기존 '수동 마스킹').
        self.manual_mask_button = SecondaryButton("흰색 깔 자리를 그림으로 직접 고르기", size="sm")
        self.manual_mask_button.setToolTip(
            "직접 만든 흑백 그림으로 흰색을 깔 자리를 정합니다. 검은 부분에 흰색이 깔립니다(기존 수동 마스킹과 같은 방식).")
        self.manual_mask_button.clicked.connect(self.choose_manual_mask)
        self.manual_mask_invert = QCheckBox("반대로: 흰 부분에 흰색 깔기")
        self.manual_mask_invert.setToolTip("고른 그림이 흰 부분을 칠할 자리로 그린 것이면 켜세요.")
        self.manual_mask_clear = SecondaryButton("직접 고른 자리 빼기", size="sm")
        self.manual_mask_clear.clicked.connect(self.clear_manual_mask)
        self.manual_mask_box = self._box(
            hbox(self.manual_mask_button, self.manual_mask_clear, None), self.manual_mask_invert)

        self.column.addWidget(self._box(
            self.cutout_title, self.cutout_hint, self.method_box, self.remove_button, self.progress_box,
            self.error_row, self.done_row, self.review_row, self.preview, self.review_buttons, self.adjust_box,
            self.white_box, hbox(self.redo_button, self.restore_button, None), self.manual_mask_box))

    def _slider(self, low: int, high: int, name: str) -> QSlider:
        slider = QSlider(Qt.Horizontal)
        slider.setRange(low, high)
        slider.setSingleStep(5)
        slider.setPageStep(25)
        slider.setAccessibleName(name)
        slider.valueChanged.connect(self._on_slider_changed)
        slider.sliderReleased.connect(self.commit_adjustment)
        return slider

    @staticmethod
    def _box(*widgets: QWidget) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        for widget in widgets:
            layout.addWidget(widget)
        return box

    # ---- 값 채우기 ----
    def _photo_columns(self) -> list[str]:
        columns = self.context.columns()
        photos = [name for name, kind in columns if kind == "photo"]
        return photos or [name for name, _ in columns]

    def image_path(self) -> str | None:
        """배경을 지우기 전의 원래 그림 파일."""
        item = self.item
        if item is None or (item.source is not None and item.source.columns()):
            return None
        value = renderer.value_for(self.controller.document, item, None, None, cutout=False)
        return value if isinstance(value, str) and os.path.isfile(value) else None

    def is_busy(self) -> bool:
        return self._job is not None

    def load(self) -> None:
        item = self.item
        if item is None:
            return
        bound = item.source is not None and bool(item.source.columns())
        self.source_control.set_current("column" if bound else "file", emit=False)
        self.file_box.setVisible(not bound)
        self.column_box.setVisible(bound)
        has_file = self.image_path() is not None
        if bound:
            column = next(iter(sorted(item.source.columns())), "")
            self.column_combo.set_columns(self._photo_columns(), column, self.context.has_list())
            if not self.context.has_list():
                self.column_hint.setText("명단을 넣으면 사람마다 그 사람의 사진으로 바뀝니다.")
            elif self.controller.preview_record is None:
                self.column_hint.setText("아래 '명단으로 미리 보기'에서 사람을 넘기면 실제 사진이 보입니다.")
            else:
                self.column_hint.setText("")
            self.column_hint.setVisible(bool(self.column_hint.text()))
        else:
            self.replace_button.setText("그림 바꾸기" if has_file else "그림 고르기")
            self.file_label.setText("" if has_file else "아직 그림이 없습니다. 로고나 사진 파일을 골라 주세요.")
            self.file_label.setVisible(not has_file)
        self.fit_control.set_current(item.fit if item.fit in FIT_HINTS else "fill", emit=False)
        self.fit_hint.setText(self._fit_hint_text(item.fit))
        self._load_cutout(item, bound, has_file)

    def _work(self, item) -> CutoutWork | None:
        work = self._results.get(item.id)
        if work is not None and work.asset != item.asset:
            self._results.pop(item.id, None)
            return None
        return work

    def _pending_for(self, item) -> tuple[CutoutWork, str] | None:
        pending = self._pending.get(item.id)
        if pending is not None and (pending[0].asset != item.asset or item.cutout_asset or item.mask_asset):
            self._pending.pop(item.id, None)
            return None
        return pending

    def _load_cutout(self, item, bound: bool, has_file: bool) -> None:
        busy = self._job is not None and self._job_item_id == item.id
        work = self._work(item)
        has_cut, has_mask = bool(item.cutout_asset), bool(item.mask_asset)
        applied = has_cut or has_mask
        settings = self._settings(item)
        pending = None if busy else self._pending_for(item)
        adjustable = has_cut and work is not None and work.method == settings.method and not busy
        idle = not applied and not busy and pending is None

        if bound:
            self.cutout_hint.setText(BOUND_HINT)
        elif not has_file:
            self.cutout_hint.setText("그림을 먼저 넣으면 배경을 지울 수 있습니다.")
        else:
            self.cutout_hint.setText("그림의 배경을 투명하게 만들어 뒤에 있는 카드 색이나 무늬가 보이게 합니다.")
        self.cutout_hint.setVisible(idle or bound)
        self.manual_mask_box.setVisible(has_file and not bound and not busy)
        self.manual_mask_clear.setVisible(bool(has_mask and item.extra.get(self.MANUAL_MASK_KEY)))
        self.method_box.setVisible(idle and has_file and not bound)
        self.method_buttons[self._method].setChecked(True)
        self.remove_button.setVisible(idle)
        self.remove_button.setEnabled(has_file and not bound and self._job is None)
        self.progress_box.setVisible(busy)
        failed = self._error is not None and self._error[0] == item.id and idle
        self.error_row.setVisible(failed)
        if failed:
            self.error_row.set_text(self._error[1])

        self.review_row.setVisible(pending is not None)
        self.review_buttons.setVisible(pending is not None)
        if pending is not None:
            self.review_row.set_text(REVIEW_TEXTS[pending[1]])

        self.done_row.setVisible(applied and not busy)
        if has_cut and has_mask:
            self.done_row.set_text("배경을 지웠습니다. 남은 부분 밑에는 흰색이 깔립니다.")
        elif has_cut:
            self.done_row.set_text("배경을 지웠습니다. 뒤에 있는 카드 색이나 무늬가 보입니다.")
        else:
            self.done_row.set_text("사람·물건 밑에만 흰색이 깔립니다. 카드에 보이는 그림은 그대로입니다. "
                                   "'다시 다듬기'를 누르면 배경도 투명해집니다.")

        self.preview.setVisible(adjustable or pending is not None)
        self.adjust_box.setVisible(adjustable)
        self.threshold_group.setVisible(adjustable and settings.method == METHOD_AI)
        self.color_group.setVisible(adjustable and settings.method == METHOD_COLOR)
        picking = adjustable and settings.method == METHOD_COLOR
        self.preview.setCursor(Qt.CrossCursor if picking else Qt.ArrowCursor)
        self.preview.setToolTip("지우고 싶은 색을 누르세요" if picking else "")

        self.white_box.setVisible(applied and not busy)
        if applied:
            supported = self.white_supported()
            self.white_check.blockSignals(True)
            self.white_check.setChecked(has_mask)
            self.white_check.blockSignals(False)
            if has_mask and supported is False:
                self.white_hint.setText("지금 프린터는 흰색을 찍지 못합니다. 흰색을 찍는 프린터에서만 깔립니다.")
            else:
                self.white_hint.setText(WHITE_HINTS[supported])

        self.redo_button.setVisible(applied and not adjustable and not busy and has_file)
        self.restore_button.setVisible(applied and not busy)
        if adjustable:
            # 항목에 적어 둔 값이 기준이다(되돌리기를 하면 막대도 그때 값으로 돌아간다).
            for slider, value in ((self.threshold_slider, settings.threshold),
                                  (self.tolerance_slider, settings.tolerance)):
                slider.blockSignals(True)
                slider.setValue(value)
                slider.blockSignals(False)
            self._show_key_color(work, settings)
            self._update_preview(settings)
        elif pending is not None:
            self._update_preview(self._settings(item, pending[0].method), pending[0])

    # ---- 그림 / 연결 ----
    def set_mode(self, mode: str) -> None:
        item = self.item
        if self._loading or item is None:
            return
        bound = item.source is not None and bool(item.source.columns())
        if mode == "column" and not bound:
            columns = self._photo_columns()
            self.set_source(ColumnSource(column=columns[0] if columns else "사진"))
        elif mode == "file" and bound:
            self.set_source(None)

    def set_column(self, column: str) -> None:
        if column:
            self.set_source(ColumnSource(column=column))

    def choose_image(self) -> None:
        path = self.context.pick_image("카드에 넣을 그림 고르기")
        if path:
            self.replace_image(path)

    def replace_image(self, path: str) -> bool:
        item = self.item
        if item is None:
            return False
        if renderer.load_image(path) is None:
            self.controller.message.emit(f"그림을 열 수 없습니다: {os.path.basename(path)}")
            return False
        try:
            asset = storage.import_asset(self._workdir(), path)
        except storage.StorageError as exc:
            self.controller.message.emit(str(exc))
            return False
        values = {"asset": asset, "mask_asset": None, "cutout_asset": None}
        if has_settings(item):
            values["extra"] = cutout.without_settings(item.extra)
        if isinstance(item.source, FixedSource):
            values["source"] = None
        return self.edit(values, text="그림 바꾸기")

    def _workdir(self) -> str:
        return self.controller.ensure_workdir()

    def _fit_hint_text(self, fit: str) -> str:
        """맞추는 방법 설명. 칸과 그림의 모양이 같으면 세 방법이 똑같이 보인다는 점을 먼저 알린다."""
        item = self.item
        path = self.image_path()
        if item is not None and path and item.w > 0 and item.h > 0:
            image = renderer.load_image(path)
            if image is not None and image.height() > 0:
                same_shape = abs(image.width() / image.height() - item.w / item.h) < 0.01 * (item.w / item.h)
                if same_shape:
                    return (FIT_HINTS.get(fit, "") + " 지금은 칸과 그림의 모양이 같아 세 방법이 똑같이 보입니다. "
                            "사진의 옆이나 위아래 가운데 점을 끌어 칸 모양을 바꾸면 차이가 납니다.")
        return FIT_HINTS.get(fit, "")

    def fill_card(self) -> bool:
        """사진 칸을 카드 전체로 키우고 '꽉 채우기'로 바꾼다. 실행취소 한 번으로 되돌아간다."""
        item = self.item
        if item is None or item.locked:
            return False
        width, height = self.controller.side_size_mm()
        stack = self.controller.undo_stack
        stack.beginMacro("카드 전체로 맞추기")
        try:
            self.controller.set_geometry(item.id, {"x": 0.0, "y": 0.0, "w": width, "h": height, "rotation": 0.0})
            if item.fit != "fill":
                self.edit({"fit": "fill"})
        finally:
            stack.endMacro()
        self.fit_control.set_current("fill", emit=False)
        self.fit_hint.setText(self._fit_hint_text("fill"))
        return True

    def _on_fit(self, fit: str) -> None:
        self.edit({"fit": fit})
        self.fit_hint.setText(self._fit_hint_text(fit))

    # ---- 배경 지우기: 설정 ----
    def _default_threshold(self) -> int:
        getter = getattr(self.engine, "default_threshold", None)
        return int(getter()) if callable(getter) else DEFAULT_THRESHOLD

    def _settings(self, item, method: str | None = None) -> CutoutSettings:
        """항목에 적어 둔 설정. 아직 지운 적이 없으면 고른 방법의 기본값."""
        settings = CutoutSettings.from_extra(item.extra, self._default_threshold())
        if method is not None:
            return settings.with_changes(method=method)
        if not (item.cutout_asset or item.mask_asset):
            return settings.with_changes(method=self._method)     # 아직 지우기 전: 화면에서 고른 방법
        return settings

    def method(self) -> str:
        return self._method

    def set_method(self, method: str) -> None:
        self.method_buttons[method].setChecked(True)

    def _on_method(self, method: str) -> None:
        self._method = method

    def white_supported(self) -> bool | None:
        """지금 프린터가 흰색을 찍는지. 알 수 없으면 None."""
        provider = self.context.capabilities
        if provider is None:
            return None
        try:
            capabilities = provider()
        except Exception:
            return None
        return None if capabilities is None else CAP_WHITE in capabilities

    def _white_default(self) -> bool:
        if self.context.capabilities is None:
            return True                 # 프린터를 알 길이 없는 화면(시연 등): 예전처럼 함께 만든다
        return self.white_supported() is True

    # ---- 배경 지우기: 실행 ----
    def start_removal(self) -> bool:
        item = self.item
        path = self.image_path()
        if item is None or path is None or self._job is not None:
            return False
        self._error = None
        self._pending.pop(item.id, None)
        method = self._settings(item).method
        work = self._work(item)
        if work is not None and work.method == method:
            self._on_finished(work, item.id)       # 이미 가려낸 그림: 다시 계산하지 않는다
            return True
        engine = None
        if method == METHOD_AI:
            try:
                ready = self.engine.prepare(self.window())
            except Exception as exc:
                self._job_item_id = item.id
                self._on_failed(str(exc))
                return False
            if not ready:
                return False
            engine = self.engine
        self._job_item_id = item.id
        self._job_method = method
        self._job = RemovalJob(engine, path, self)
        self._job.progress.connect(self.progress_label.setText)
        self._job.finished.connect(self._on_finished)
        self._job.failed.connect(self._on_failed)
        self.progress_label.setText("사진에서 사람과 배경을 가려내고 있습니다." if engine is not None
                                    else "그림에서 바탕색을 찾고 있습니다.")
        self._job.start()
        self.reload()
        return True

    def cancel_removal(self) -> None:
        if self._job is not None:
            self._job.cancel()
            self._job = None
            self._job_item_id = None
            self.reload()

    def _on_quit(self) -> None:
        if isValid(self) and self._job is not None:
            self._job.cancel()
            self._job.wait(20000)

    def wait_for_removal(self, msecs: int = 30000) -> None:
        """진행 중인 계산이 끝나기를 기다린다(테스트와 닫을 때)."""
        if self._job is not None:
            self._job.wait(msecs)

    def _on_finished(self, work: CutoutWork, item_id: str | None = None) -> None:
        if item_id is None:
            item_id, self._job, self._job_item_id = self._job_item_id, None, None
        item = self.controller.item(item_id) if item_id else None
        if item is None:
            self.reload()
            return
        work.asset = item.asset
        settings = self._settings(item, work.method)
        self._results[item_id] = work
        state = "ok"
        if work.method == METHOD_AI:
            preview = work.preview
            alpha = preview.cut(settings)[1]
            state = cutout.verdict(cutout.kept_ratio(alpha), cutout.lost_ratio(preview.rgb, preview.alpha, alpha))
        if state == "ok":
            self._apply(item_id, settings, merge=False)
        else:
            self._pending[item_id] = (work, state)
        self.reload()

    def _on_failed(self, message: str) -> None:
        item_id, self._job, self._job_item_id = self._job_item_id, None, None
        if item_id:
            self._error = (item_id, f"배경을 지우지 못했습니다. {message}")
        self.reload()

    def error_text(self) -> str:
        return self._error[1] if self._error else ""

    def review_text(self) -> str:
        """결과가 이상해서 물어보는 중이면 그 안내문, 아니면 빈 글."""
        pending = self._pending_for(self.item) if self.item is not None else None
        return REVIEW_TEXTS[pending[1]] if pending else ""

    def accept_pending(self) -> None:
        item = self.item
        pending = self._pending.pop(item.id, None) if item is not None else None
        if pending is not None:
            self._apply(item.id, self._settings(item, pending[0].method), merge=False)
            self.reload()

    def discard_pending(self) -> None:
        if self.item is not None:
            self._pending.pop(self.item.id, None)
            self.reload()

    def try_color_method(self) -> None:
        """AI가 주인공을 못 찾았을 때: 바탕색만 지우는 방법으로 바꿔서 해 본다."""
        if self.item is None:
            return
        self._pending.pop(self.item.id, None)
        self._results.pop(self.item.id, None)
        self.set_method(METHOD_COLOR)
        self._method = METHOD_COLOR
        self.start_removal()

    def _apply(self, item_id: str, settings: CutoutSettings, *, merge: bool, white: bool | None = None,
               text: str = "배경 지우기") -> bool:
        work = self._results.get(item_id)
        item = self.controller.item(item_id)
        if work is None or item is None:
            return False
        if white is None:
            applied = bool(item.cutout_asset or item.mask_asset)
            white = bool(item.mask_asset) if applied else self._white_default()
        try:
            workdir = self._workdir()
            assets_dir = os.path.join(workdir, storage.ASSETS_DIR)
            existing = set(os.listdir(assets_dir)) if os.path.isdir(assets_dir) else set()
            rgb, alpha = work.cut(settings)
            cut_asset, mask_asset = save_cutout(rgb, alpha, workdir, white)
        except (OSError, ValueError, storage.StorageError) as exc:
            self.controller.message.emit(f"배경을 지운 결과를 저장하지 못했습니다. {exc}")
            return False
        key = self.context.session.key("mask") if merge else None
        values = {"cutout_asset": cut_asset, "mask_asset": mask_asset, "extra": settings.to_extra(item.extra)}
        stack = self.controller.undo_stack
        before = stack.index()
        changed = self.controller.set_properties(item_id, values, key, text=text)
        self._tidy_chain(workdir, existing, {cut_asset, mask_asset} - {None}, merged=stack.index() == before)
        return changed

    def _tidy_chain(self, workdir: str, existing: set[str], current: set[str], merged: bool) -> None:
        """막대를 끄는 동안 생긴 중간 파일을 지운다. 한 단계로 합쳐진 수정의 중간값은 되돌릴 수도 없다."""
        created = {asset for asset in current if os.path.basename(asset) not in existing}
        if merged:
            for asset in self._chain_files - current:
                try:
                    os.remove(os.path.join(workdir, asset))
                except OSError:
                    pass
            self._chain_files = (self._chain_files & current) | created
        else:
            self._chain_files = created

    # ---- 배경 지우기: 다듬기 ----
    def _live_settings(self) -> CutoutSettings | None:
        """막대의 지금 값을 반영한 설정(아직 적어 넣지 않은 값 포함)."""
        if self.item is None:
            return None
        return self._settings(self.item).with_changes(
            threshold=self.threshold_slider.value(), tolerance=self.tolerance_slider.value())

    def _show_key_color(self, work: CutoutWork, settings: CutoutSettings) -> None:
        color = settings.color if settings.color is not None else work.preview.corner_color()
        self.color_chip.set_color(cutout.format_color(color))
        self.corner_button.setVisible(settings.color is not None)

    def _update_preview(self, settings: CutoutSettings | None, work: CutoutWork | None = None) -> None:
        if work is None and self.item is not None:
            work = self._results.get(self.item.id)
        if work is None or settings is None:
            return
        width = max(120, self.preview.width() or 280)
        self.preview.setPixmap(work.preview.pixmap(settings, width, _PREVIEW_HEIGHT, self.devicePixelRatioF()))

    def _on_slider_changed(self, _value: int) -> None:
        if self._loading:
            return
        self._update_preview(self._live_settings())
        if not self.sender().isSliderDown():
            self.commit_adjustment()

    def set_threshold(self, value: int) -> None:
        self.threshold_slider.setValue(int(value))

    def set_tolerance(self, value: int) -> None:
        self.tolerance_slider.setValue(int(value))

    def commit_adjustment(self) -> None:
        if self._loading or self.item is None or not self.item.cutout_asset:
            return
        self._apply(self.item.id, self._live_settings(), merge=True)

    commit_threshold = commit_adjustment

    def set_key_color(self, color: tuple[int, int, int] | None) -> None:
        """지울 색을 바꾼다. None이면 그림 모서리 색."""
        item = self.item
        if item is None or not item.cutout_asset or self._results.get(item.id) is None:
            return
        self._apply(item.id, self._settings(item).with_changes(color=color), merge=False, text="지울 색 바꾸기")

    def _on_preview_pressed(self, x: float, y: float) -> None:
        item = self.item
        work = self._results.get(item.id) if item is not None else None
        if work is None or not item.cutout_asset or self._settings(item).method != METHOD_COLOR:
            return
        color = work.preview.color_at(x, y, max(120, self.preview.width()), _PREVIEW_HEIGHT)
        if color is not None:
            self.set_key_color(color)

    # ---- 수동 마스킹 ----
    MANUAL_MASK_KEY = "manual_mask"

    def has_manual_mask(self) -> bool:
        item = self.item
        return bool(item is not None and item.mask_asset and item.extra.get(self.MANUAL_MASK_KEY))

    def choose_manual_mask(self) -> None:
        path = self.context.pick_image("흰색을 깔 자리를 그린 그림 고르기")
        if path:
            self.set_manual_mask(path, dark_is_ink=not self.manual_mask_invert.isChecked())

    def set_manual_mask(self, path: str, dark_is_ink: bool = True) -> bool:
        """직접 만든 그림을 화이트 층 마스크로 쓴다. dark_is_ink면 검은 부분에 흰색을 깐다(기존 방식)."""
        import tempfile

        from PIL import Image, ImageOps

        item = self.item
        if item is None:
            return False
        try:
            with Image.open(path) as source:
                mask = ImageOps.exif_transpose(source).convert("L")
            if dark_is_ink:
                mask = ImageOps.invert(mask)
            handle, temporary = tempfile.mkstemp(suffix=".png", prefix="manual_mask_")
            os.close(handle)
            try:
                mask.save(temporary)
                asset = storage.import_asset(self._workdir(), temporary)
            finally:
                try:
                    os.remove(temporary)
                except OSError:
                    pass
        except (OSError, ValueError, storage.StorageError) as exc:
            self.controller.message.emit(f"그림을 열 수 없습니다: {os.path.basename(path)} ({exc})")
            return False
        extra = dict(item.extra)
        extra[self.MANUAL_MASK_KEY] = True
        return self.controller.set_properties(
            item.id, {"mask_asset": asset, "extra": extra}, text="흰색 깔 자리 직접 고르기")

    def clear_manual_mask(self) -> bool:
        item = self.item
        if item is None or not self.has_manual_mask():
            return False
        extra = {key: value for key, value in item.extra.items() if key != self.MANUAL_MASK_KEY}
        return self.controller.set_properties(
            item.id, {"mask_asset": None, "extra": extra}, text="직접 고른 자리 빼기")

    def _on_white_toggled(self, checked: bool) -> None:
        if not self._loading:
            self.set_white(checked)

    def set_white(self, on: bool) -> bool:
        """남은 부분 밑에 흰색을 깔지 정한다(되돌리기 한 단계)."""
        item = self.item
        if item is None or bool(item.mask_asset) == bool(on):
            return False
        mask = None
        if on:
            path = renderer.asset_path(self.controller.document, item.cutout_asset)
            if not path or not os.path.isfile(path):
                return False
            try:
                mask = save_mask_of(path, self._workdir())
            except (OSError, storage.StorageError) as exc:
                self.controller.message.emit(f"흰색을 깔 자리를 만들지 못했습니다. {exc}")
                return False
        return self.controller.set_properties(
            item.id, {"mask_asset": mask}, text="흰색 깔기" if on else "흰색 깔기 끄기")

    def restore_background(self) -> None:
        if self.item is not None:
            self.controller.set_properties(
                self.item.id, {"cutout_asset": None, "mask_asset": None}, text="배경 되살리기")

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.preview.isVisible() and self.item is not None:
            pending = self._pending.get(self.item.id)
            if pending is not None:
                self._update_preview(self._settings(self.item, pending[0].method), pending[0])
            else:
                self._update_preview(self._live_settings())
