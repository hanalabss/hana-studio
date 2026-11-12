"""캔버스 위 도구 줄: 앞면/뒷면, 정렬, 보기(모눈·자석·재단선), 확대."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QFrame, QHBoxLayout, QMenu, QPushButton, QWidget

from ui.editor import EditorController, EditorError, EditorView
from ui.theme import SPACE, ThemeManager
from ui.widgets import Divider, IconButton, QuietButton, SecondaryButton, SegmentedControl

from . import icons
from .common import as_menu_button
from .style import BAR_HEIGHT

ALIGN_ACTIONS = (
    ("left", "왼쪽으로 맞추기", "align-left"),
    ("center", "가로 가운데로 맞추기", "align-center"),
    ("right", "오른쪽으로 맞추기", "align-right"),
    ("top", "위로 맞추기", "align-top"),
    ("middle", "세로 가운데로 맞추기", "align-middle"),
    ("bottom", "아래로 맞추기", "align-bottom"),
)
DISTRIBUTE_ACTIONS = (
    ("horizontal", "가로 간격 같게", "distribute-horizontal"),
    ("vertical", "세로 간격 같게", "distribute-vertical"),
)
_ROOMY_WIDTH = 570
ORDER_ACTIONS = (
    ("bring_to_front", "맨 앞으로 가져오기"),
    ("bring_forward", "한 칸 앞으로"),
    ("send_backward", "한 칸 뒤로"),
    ("send_to_back", "맨 뒤로 보내기"),
)


class CanvasBar(QFrame):
    addBackRequested = Signal()
    compactChanged = Signal(bool)   # 줄이 좁아져 덜 쓰는 버튼을 접었는지

    def __init__(self, controller: EditorController, view: EditorView, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("CanvasBar")
        self.setFixedHeight(BAR_HEIGHT)
        self.controller = controller
        self.view = view
        self._compact = False
        layout = QHBoxLayout(self)
        layout.setContentsMargins(SPACE["md"], 0, SPACE["md"], 0)
        layout.setSpacing(SPACE["xs"])

        self.side_control = SegmentedControl([("front", "앞면"), ("back", "뒷면")])
        self.side_control.currentChanged.connect(controller.set_side)
        layout.addWidget(self.side_control)
        self.add_back_button = QuietButton("뒷면 추가", size="sm", icon="plus", accent=True)
        self.add_back_button.setToolTip("카드 뒷면도 꾸밉니다")
        self.add_back_button.clicked.connect(self.addBackRequested)
        layout.addWidget(self.add_back_button)
        layout.addStretch(1)

        self.align_button = SecondaryButton("정렬", size="sm")
        as_menu_button(self.align_button)
        self.align_button.setToolTip("고른 것을 줄 맞추거나 앞뒤 순서를 바꿉니다")
        self.align_menu = QMenu(self)
        self.align_note = QAction("", self)
        self.align_note.setEnabled(False)
        self.align_menu.addAction(self.align_note)
        self.align_menu.addSeparator()
        self.align_actions: dict[str, QAction] = {}
        for mode, text, _icon in ALIGN_ACTIONS:
            action = QAction(text, self)
            action.triggered.connect(lambda _=False, m=mode: self.align(m))
            self.align_actions[mode] = action
            self.align_menu.addAction(action)
            if mode == "right":
                self.align_menu.addSeparator()
        self.align_menu.addSeparator()
        self.distribute_actions: dict[str, QAction] = {}
        for axis, text, _icon in DISTRIBUTE_ACTIONS:
            action = QAction(text, self)
            action.triggered.connect(lambda _=False, a=axis: self.distribute(a))
            self.distribute_actions[axis] = action
            self.align_menu.addAction(action)
        self.align_menu.addSeparator()
        self.order_actions: dict[str, QAction] = {}
        for name, text in ORDER_ACTIONS:
            action = QAction(text, self)
            action.triggered.connect(lambda _=False, n=name: getattr(self.controller, n)())
            self.order_actions[name] = action
            self.align_menu.addAction(action)
        self.align_menu.aboutToShow.connect(self._sync_align_menu)
        self.align_button.setMenu(self.align_menu)
        layout.addWidget(self.align_button)
        layout.addSpacing(SPACE["xs"])

        self.view_button = SecondaryButton("보기", size="sm")
        as_menu_button(self.view_button)
        self.view_button.setToolTip("모눈, 자석처럼 붙이기, 재단선을 켜고 끕니다")
        self.view_menu = QMenu(self)
        self.grid_toggle = self._check_action("모눈 보기 (1 mm)", controller.set_grid_visible)
        self.snap_toggle = self._check_action("옮길 때 자석처럼 붙이기", controller.set_snap_enabled)
        self.cut_toggle = self._check_action("재단선 보기", controller.set_cut_line_visible)
        self.ruler_toggle = self._check_action("눈금자 보기", view.set_rulers_visible)
        self.ruler_toggle.setChecked(True)
        self.view_button.setMenu(self.view_menu)
        layout.addWidget(self.view_button)

        divider = Divider(Qt.Vertical)
        divider.setFixedHeight(20)
        layout.addSpacing(SPACE["xs"])
        layout.addWidget(divider)
        layout.addSpacing(SPACE["xs"])

        self.zoom_out_button = IconButton("minus", "작게 보기 (Ctrl+-)", size="sm")
        self.zoom_out_button.clicked.connect(view.zoom_out)
        layout.addWidget(self.zoom_out_button)
        self.zoom_button = QPushButton("100%")
        self.zoom_button.setProperty("zoom", "true")
        self.zoom_button.setAutoDefault(False)
        self.zoom_button.setFixedSize(58, 32)
        self.zoom_button.setToolTip("보는 크기 바꾸기")
        self.zoom_button.setAccessibleName("보는 크기")
        zoom_menu = QMenu(self)
        self.fit_action = QAction("카드 전체 보기\tCtrl+0", self)
        self.fit_action.triggered.connect(view.zoom_fit)
        self.actual_action = QAction("실제 크기 (100%)\tCtrl+1", self)
        self.actual_action.triggered.connect(view.zoom_actual)
        self.selection_action = QAction("고른 것 크게 보기\tCtrl+2", self)
        self.selection_action.triggered.connect(view.zoom_to_selection)
        self.zoom_in_action = QAction("크게 보기\tCtrl++", self)
        self.zoom_in_action.triggered.connect(view.zoom_in)
        self.zoom_out_action = QAction("작게 보기\tCtrl+-", self)
        self.zoom_out_action.triggered.connect(view.zoom_out)
        zoom_menu.addActions([self.zoom_in_action, self.zoom_out_action])
        zoom_menu.addSeparator()
        zoom_menu.addActions([self.fit_action, self.actual_action, self.selection_action])
        self.zoom_button.setMenu(zoom_menu)
        layout.addWidget(self.zoom_button)
        self.zoom_in_button = IconButton("plus", "크게 보기 (Ctrl++)", size="sm")
        self.zoom_in_button.clicked.connect(view.zoom_in)
        layout.addWidget(self.zoom_in_button)

        controller.options_changed.connect(self._sync_options)
        controller.card_changed.connect(self._sync_card)
        controller.document_changed.connect(self._sync_card)
        controller.side_changed.connect(self._sync_side)
        controller.selection_changed.connect(self._sync_selection)
        view.zoom_changed.connect(self._sync_zoom)
        ThemeManager.instance().themeChanged.connect(self._refresh_icons)
        self._refresh_icons()
        self._sync_options()
        self._sync_card()
        self._sync_selection()
        self._sync_zoom(view.zoom)

    def _check_action(self, text: str, slot) -> QAction:
        action = QAction(text, self)
        action.setCheckable(True)
        action.toggled.connect(slot)
        self.view_menu.addAction(action)
        return action

    # ---- 동작 ----
    def align(self, mode: str) -> None:
        try:
            self.controller.align(mode)
        except EditorError as exc:
            self.controller.message.emit(str(exc))

    def distribute(self, axis: str) -> None:
        try:
            self.controller.distribute(axis)
        except EditorError as exc:
            self.controller.message.emit(str(exc))

    # ---- 상태 맞추기 ----
    def _refresh_icons(self, *_args) -> None:
        for mode, _text, icon_name in ALIGN_ACTIONS:
            self.align_actions[mode].setIcon(icons.icon(icon_name))
        for axis, _text, icon_name in DISTRIBUTE_ACTIONS:
            self.distribute_actions[axis].setIcon(icons.icon(icon_name))

    def _sync_align_menu(self) -> None:
        count = len(self.controller.selection)
        if count == 0:
            self.align_note.setText("먼저 카드에서 맞출 것을 고르세요")
        elif count == 1:
            self.align_note.setText("카드에 맞춥니다")
        else:
            self.align_note.setText(f"고른 {count}개를 서로 맞춥니다")
        for action in (*self.align_actions.values(), *self.order_actions.values()):
            action.setEnabled(count >= 1)
        for action in self.distribute_actions.values():
            action.setEnabled(count >= 3)
            action.setToolTip("" if count >= 3 else "3개 이상 골랐을 때 쓸 수 있습니다")

    def _sync_options(self) -> None:
        for toggle, value in ((self.grid_toggle, self.controller.grid_visible),
                              (self.snap_toggle, self.controller.snap_enabled),
                              (self.cut_toggle, self.controller.cut_line_visible)):
            if toggle.isChecked() != value:
                toggle.blockSignals(True)
                toggle.setChecked(value)
                toggle.blockSignals(False)

    def _sync_card(self) -> None:
        document = self.controller.document
        dual = bool(document.dual_side)
        self.side_control.setVisible(dual)
        self.add_back_button.setVisible(not dual)
        has_bleed = document.bleed_mm > 0
        self.cut_toggle.setEnabled(has_bleed)
        self.cut_toggle.setText("재단선 보기" if has_bleed else "재단선 보기 (재단 여백이 있는 크기에서만)")
        self._sync_side(self.controller.current_side)

    def _sync_side(self, side_key: str) -> None:
        if side_key in self.side_control.keys():
            self.side_control.set_current(side_key, emit=False)

    def _sync_selection(self, *_args) -> None:
        self.selection_action.setEnabled(bool(self.controller.selection))

    def _sync_zoom(self, zoom: float) -> None:
        self.zoom_button.setText(f"{round(zoom * 100)}%")

    def is_compact(self) -> bool:
        return self._compact

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        compact = self.width() < _ROOMY_WIDTH
        if compact != self._compact:
            self._compact = compact
            self.zoom_out_button.setVisible(not compact)
            self.zoom_in_button.setVisible(not compact)
            self.compactChanged.emit(compact)
