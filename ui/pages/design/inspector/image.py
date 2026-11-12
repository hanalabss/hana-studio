"""사진을 골랐을 때: 그림 바꾸기, 명단 사진 연결, 채우는 방법, 배경 지우기."""

from __future__ import annotations

import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QLabel, QSlider, QVBoxLayout, QWidget
from shiboken6 import isValid

from design import renderer, storage
from design.binding import ColumnSource, FixedSource
from ui.widgets import Divider, ProgressBar, SecondaryButton, SegmentedControl, label

from ..background import DEFAULT_THRESHOLD, RemovalJob, RembgEngine, save_mask
from ..common import NoteRow, group, hbox, use_local_icon
from .base import InspectorContext, Section
from .detail import ColumnCombo

FIT_OPTIONS = (("fill", "꽉 채우기"), ("contain", "다 보이게"), ("stretch", "늘려 맞추기"))
FIT_HINTS = {
    "fill": "칸을 가득 채우고 넘치는 부분은 잘라 냅니다. 증명사진에 알맞습니다.",
    "contain": "그림 전체가 보이도록 줄입니다. 칸에 빈 곳이 생길 수 있습니다. 로고에 알맞습니다.",
    "stretch": "칸 모양에 맞춰 그림을 늘립니다. 그림이 찌그러질 수 있습니다.",
}
_PREVIEW_HEIGHT = 150


class ImageSection(Section):
    def __init__(self, context: InspectorContext, parent: QWidget | None = None) -> None:
        super().__init__(context, parent)
        self.engine = context.background_engine or RembgEngine()
        self._job: RemovalJob | None = None
        self._job_item_id: str | None = None
        self._error: tuple[str, str] | None = None
        # 항목 id -> (asset, alpha, CutoutPreview, threshold)
        self._results: dict[str, tuple] = context.state.setdefault("cutouts", {})

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
        self.column.addWidget(group("칸에 맞추는 방법", self.fit_control, self.fit_hint))

        self.column.addWidget(Divider())
        self.cutout_title = label("배경 지우기", "strong")
        self.cutout_hint = label("", "hint", wrap=True)
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

        self.preview = QLabel()
        self.preview.setFixedHeight(_PREVIEW_HEIGHT)
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setAccessibleName("배경을 지운 모습")
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(5, 250)
        self.threshold_slider.setSingleStep(5)
        self.threshold_slider.setPageStep(25)
        self.threshold_slider.setAccessibleName("지우는 정도")
        self.threshold_slider.valueChanged.connect(self._on_threshold_changed)
        self.threshold_slider.sliderReleased.connect(self.commit_threshold)
        self.slider_ends = hbox(label("더 남기기", "caption"), None, label("더 지우기", "caption"))
        self.restore_button = SecondaryButton("배경 되살리기", size="sm")
        self.restore_button.setToolTip("지운 배경을 원래대로 돌려놓습니다")
        self.restore_button.clicked.connect(self.restore_background)
        self.redo_button = SecondaryButton("다시 다듬기", size="sm")
        self.redo_button.clicked.connect(self.start_removal)
        self.done_row = NoteRow("ok", "배경을 지웠습니다. 사람·물건 부분에만 흰색이 깔립니다.")
        self.adjust_box = self._box(
            self.preview, group("지우는 정도", self.threshold_slider, self.slider_ends))
        self.column.addWidget(self._box(
            self.cutout_title, self.cutout_hint, self.remove_button, self.progress_box, self.error_row,
            self.done_row, self.adjust_box, hbox(self.redo_button, self.restore_button, None)))

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
        item = self.item
        if item is None or (item.source is not None and item.source.columns()):
            return None
        value = renderer.value_for(self.controller.document, item, None, None)
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
        self.fit_hint.setText(FIT_HINTS.get(item.fit, ""))
        self._load_cutout(item, bound, has_file)

    def _load_cutout(self, item, bound: bool, has_file: bool) -> None:
        busy = self._job is not None and self._job_item_id == item.id
        result = self._results.get(item.id)
        if result is not None and result[0] != item.asset:
            self._results.pop(item.id, None)
            result = None
        has_mask = bool(item.mask_asset)
        adjustable = has_mask and result is not None
        if bound:
            self.cutout_hint.setText("사람마다 사진이 달라 여기서는 배경을 지울 수 없습니다. 그림 파일을 넣었을 때만 쓸 수 있습니다.")
        elif not has_file:
            self.cutout_hint.setText("그림을 먼저 넣으면 배경을 지울 수 있습니다.")
        else:
            self.cutout_hint.setText(
                "사진에서 사람이나 물건만 찾아 그 밑에만 흰색을 깝니다. 투명 카드에서 배경이 비쳐 보이게 할 때 씁니다.")
        self.cutout_hint.setVisible(not has_mask or bound)
        self.remove_button.setVisible(not has_mask and not busy)
        self.remove_button.setEnabled(has_file and not bound and self._job is None)
        self.progress_box.setVisible(busy)
        failed = self._error is not None and self._error[0] == item.id and not busy and not has_mask
        self.error_row.setVisible(failed)
        if failed:
            self.error_row.set_text(self._error[1])
        self.done_row.setVisible(has_mask and not busy)
        self.adjust_box.setVisible(adjustable and not busy)
        self.redo_button.setVisible(has_mask and not adjustable and not busy and has_file)
        self.restore_button.setVisible(has_mask and not busy)
        if adjustable:
            self.threshold_slider.blockSignals(True)
            self.threshold_slider.setValue(int(result[3]))
            self.threshold_slider.blockSignals(False)
            self._update_preview(int(result[3]))

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
        values = {"asset": asset, "mask_asset": None}
        if isinstance(item.source, FixedSource):
            values["source"] = None
        return self.edit(values, text="그림 바꾸기")

    def _workdir(self) -> str:
        document = self.controller.document
        return document.base_dir or self.controller._ensure_workdir()

    def _on_fit(self, fit: str) -> None:
        self.fit_hint.setText(FIT_HINTS[fit])
        self.edit({"fit": fit})

    # ---- 배경 지우기 ----
    def start_removal(self) -> bool:
        item = self.item
        path = self.image_path()
        if item is None or path is None or self._job is not None:
            return False
        self._error = None
        try:
            ready = self.engine.prepare(self.window())
        except Exception as exc:
            self._job_item_id = item.id
            self._on_failed(str(exc))
            return False
        if not ready:
            return False
        self._job_item_id = item.id
        self._job = RemovalJob(self.engine, path, self)
        self._job.progress.connect(self.progress_label.setText)
        self._job.finished.connect(self._on_finished)
        self._job.failed.connect(self._on_failed)
        self.progress_label.setText("사진에서 사람과 배경을 가려내고 있습니다.")
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

    def _default_threshold(self) -> int:
        getter = getattr(self.engine, "default_threshold", None)
        return int(getter()) if callable(getter) else DEFAULT_THRESHOLD

    def _on_finished(self, alpha, preview) -> None:
        item_id, self._job, self._job_item_id = self._job_item_id, None, None
        item = self.controller.item(item_id) if item_id else None
        if item is None:
            self.reload()
            return
        threshold = self._default_threshold()
        self._results[item_id] = (item.asset, alpha, preview, threshold)
        self._commit_mask(item_id, threshold, merge=False)
        self.reload()

    def _on_failed(self, message: str) -> None:
        item_id, self._job, self._job_item_id = self._job_item_id, None, None
        if item_id:
            self._error = (item_id, f"배경을 지우지 못했습니다. {message}")
        self.reload()

    def error_text(self) -> str:
        return self._error[1] if self._error else ""

    def _commit_mask(self, item_id: str, threshold: int, merge: bool) -> bool:
        result = self._results.get(item_id)
        if result is None:
            return False
        asset, alpha, preview, _old = result
        self._results[item_id] = (asset, alpha, preview, int(threshold))
        try:
            mask = save_mask(alpha, threshold, self._workdir())
        except (OSError, storage.StorageError) as exc:
            self.controller.message.emit(f"배경을 지운 결과를 저장하지 못했습니다. {exc}")
            return False
        key = self.context.session.key("mask") if merge else None
        return self.controller.set_properties(item_id, {"mask_asset": mask}, key, text="배경 지우기")

    def _update_preview(self, threshold: int) -> None:
        result = self._results.get(self.item.id) if self.item is not None else None
        if result is None:
            return
        width = max(120, self.preview.width() or 280)
        self.preview.setPixmap(result[2].pixmap(threshold, width, _PREVIEW_HEIGHT, self.devicePixelRatioF()))

    def _on_threshold_changed(self, value: int) -> None:
        if self._loading:
            return
        self._update_preview(value)
        if not self.threshold_slider.isSliderDown():
            self.commit_threshold()

    def set_threshold(self, value: int) -> None:
        self.threshold_slider.setValue(int(value))

    def commit_threshold(self) -> None:
        if self._loading or self.item is None:
            return
        self._commit_mask(self.item.id, self.threshold_slider.value(), merge=True)

    def restore_background(self) -> None:
        if self.item is not None:
            self.controller.set_properties(self.item.id, {"mask_asset": None}, text="배경 되살리기")

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.adjust_box.isVisible():
            self._update_preview(self.threshold_slider.value())
