"""디자인 화면: 시작 화면, 넣기·목록(왼쪽), 캔버스(가운데), 속성(오른쪽), 명단 미리 보기(아래)."""

from __future__ import annotations

import os
from typing import Callable

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QDialog, QFileDialog, QFrame, QHBoxLayout, QMenu, QStackedLayout, QVBoxLayout, QWidget,
)

from design import storage
from ui.editor import IMAGE_EXTENSIONS, EditorError, EditorView, Problem
from ui.shell import PAGE_LIST, BasePage, FooterAction
from ui.theme import SPACE, ThemeManager, mark_themed
from ui.widgets import Dialog, Divider, IconButton, SecondaryButton, Toast

from . import icons
from .canvas_bar import CanvasBar
from .common import EditSession, as_menu_button
from .insert_panel import InsertPanel
from .inspector import Inspector, InspectorContext
from .layers_panel import LayersPanel
from .preview_bar import PreviewBar
from services.project import RecentProjects

from .start_panel import StartPanel, pil_to_qimage
from .style import SIDE_WIDTH, build_stylesheet, palette_for

DESIGN_FILTER = f"하나스튜디오 디자인 (*{storage.EXTENSION})"
IMAGE_FILTER = "그림 파일 (" + " ".join(f"*{ext}" for ext in IMAGE_EXTENSIONS) + ")"
STRIP_MIN_WIDTH = 1500
COMPACT_HEIGHT = 580


class FileDialogs:
    """파일 고르기 창. 테스트에서는 이 객체를 바꿔 끼운다."""

    def open_design(self, parent) -> str | None:
        path, _ = QFileDialog.getOpenFileName(parent, "디자인 열기", "", DESIGN_FILTER)
        return path or None

    def save_design(self, parent, suggested: str) -> str | None:
        path, _ = QFileDialog.getSaveFileName(parent, "디자인 저장", suggested, DESIGN_FILTER)
        if path and not path.lower().endswith(storage.EXTENSION):
            path += storage.EXTENSION
        return path or None

    def open_image(self, parent, title: str) -> str | None:
        path, _ = QFileDialog.getOpenFileName(parent, title, "", IMAGE_FILTER)
        return path or None

    def save_image(self, parent, suggested: str) -> str | None:
        path, _ = QFileDialog.getSaveFileName(parent, "미리보기 그림 내보내기", suggested, "PNG 그림 (*.png)")
        if path and not path.lower().endswith(".png"):
            path += ".png"
        return path or None

    def ask_unsaved(self, parent, name: str) -> str:
        """'save' | 'discard' | 'cancel'."""
        dialog = Dialog("바꾼 내용을 저장할까요?",
                        f"'{name}'에 저장하지 않은 변경이 있습니다. 저장하지 않으면 바꾼 내용이 사라집니다.", parent)
        choice = {"value": "cancel"}

        def pick(value: str, accept: bool) -> Callable[[], None]:
            def run() -> None:
                choice["value"] = value
                dialog.accept() if accept else dialog.reject()
            return run

        dialog.add_button("저장 안 함", "secondary", None, left=True).clicked.connect(pick("discard", True))
        dialog.add_button("취소", "secondary", "reject")
        dialog.add_button("저장", "primary", None, default=True).clicked.connect(pick("save", True))
        return choice["value"] if dialog.exec() == QDialog.Accepted else "cancel"


def is_blank(document) -> bool:
    return all(not side.items and not side.background for side in document.sides.values())


class DesignPage(BasePage):
    """`DesignPage(workspace)`를 `shell.register_page(PAGE_DESIGN, page, ...)`로 올리면 틀과 알아서 이어진다."""

    last_message = ""
    navigateRequested = Signal(str)
    documentStateChanged = Signal(str, object)   # 이름, 저장 상태(True/False/None)
    undoStateChanged = Signal(bool, bool)

    def __init__(self, workspace, parent: QWidget | None = None, *, background_engine=None,
                 capabilities: Callable[[], set[str] | None] | None = None) -> None:
        super().__init__("디자인", parent)
        self.workspace = workspace
        self.controller = workspace.editor
        self.dialogs = FileDialogs()
        self._active = False
        self._started = not is_blank(self.controller.document) or self.controller.path is not None
        self.session = EditSession(self)
        mark_themed(self)

        self._stack = QStackedLayout(self)
        self._stack.setContentsMargins(0, 0, 0, 0)

        self.start_panel = StartPanel(self)
        self.start_panel.startRequested.connect(self.start_with_template)
        self.start_panel.openRequested.connect(lambda: self.open_design())
        self.start_panel.imageRequested.connect(lambda: self.start_from_image())
        self.start_panel.openPathRequested.connect(lambda path: self.open_design(path))
        self.refresh_recents()
        self._stack.addWidget(self.start_panel)

        self.editor_host = QWidget(self)
        self._stack.addWidget(self.editor_host)
        row = QHBoxLayout(self.editor_host)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)

        # ---- 왼쪽 ----
        side = QFrame()
        side.setObjectName("DesignSide")
        side.setFixedWidth(SIDE_WIDTH)
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(SPACE["lg"], SPACE["md"] + 2, SPACE["lg"], SPACE["sm"])
        side_layout.setSpacing(SPACE["md"] + 2)
        self.insert_panel = InsertPanel()
        self.insert_panel.addRequested.connect(self.add)
        self.insert_panel.imageFileRequested.connect(lambda: self.add_image_file())
        self.insert_panel.fieldRequested.connect(self.add_field)
        self.insert_panel.listPageRequested.connect(self.go_to_list)
        side_layout.addWidget(self.insert_panel)
        side_layout.addWidget(Divider())
        self.layers_panel = LayersPanel(self.controller)
        side_layout.addWidget(self.layers_panel, 1)
        row.addWidget(side)

        # ---- 가운데 ----
        center = QVBoxLayout()
        center.setContentsMargins(0, 0, 0, 0)
        center.setSpacing(0)
        self.view = EditorView(self.controller, palette_for())
        self.canvas_bar = CanvasBar(self.controller, self.view)
        self.canvas_bar.addBackRequested.connect(self.add_back_side)
        self.canvas_bar.compactChanged.connect(lambda compact: self.save_button.setVisible(not compact))
        self._build_file_menu()
        center.addWidget(self.canvas_bar)
        center.addWidget(self.view, 1)
        self.preview_bar = PreviewBar(workspace)
        center.addWidget(self.preview_bar)
        row.addLayout(center, 1)

        # ---- 오른쪽 ----
        context = InspectorContext(
            controller=self.controller, workspace=workspace, session=self.session,
            go_to_list=self.go_to_list, pick_image=lambda title: self.dialogs.open_image(self, title),
            background_engine=background_engine, capabilities=capabilities,
        )
        self.inspector = Inspector(context)
        self.inspector.problemActivated.connect(self.go_to_problem)
        self._build_item_menu()
        row.addWidget(self.inspector)

        controller = self.controller
        controller.message.connect(self.notify)
        controller.bound_edit_requested.connect(self._on_bound_edit)
        controller.item_activated.connect(self._on_item_activated)
        controller.modified_changed.connect(self._sync_document_state)
        controller.document_changed.connect(self._on_document_changed)
        controller.card_changed.connect(self._sync_document_state)
        controller.problems_changed.connect(self._sync_footer)
        controller.undo_stack.canUndoChanged.connect(self._sync_undo_state)
        controller.undo_stack.canRedoChanged.connect(self._sync_undo_state)
        workspace.store_changed.connect(self._sync_columns)
        workspace.columns_changed.connect(self._sync_columns)
        ThemeManager.instance().themeChanged.connect(self._apply_theme)

        self.set_footer_actions([FooterAction("다음: 명단 넣기", self.go_to_list)])
        self._apply_theme()
        self._sync_columns()
        self._sync_view()

    # ================================================================ 틀과 잇기

    @property
    def shell(self):
        return self.__dict__.get("_shell")

    @shell.setter
    def shell(self, value) -> None:
        self.__dict__["_shell"] = value
        if value is not None:
            self.wire_shell(value)

    @property
    def _wired_shell(self):
        return self.__dict__.get("_wired")

    def wire_shell(self, shell) -> None:
        """틀의 되돌리기 버튼, 문서 이름 표시와 잇는다. register_page가 불러 주므로 보통은 직접 부를 일이 없다."""
        if self._wired_shell is shell:
            return
        self.__dict__["_wired"] = shell
        self.__dict__["_shell"] = shell
        shell.undoRequested.connect(self._on_shell_undo)
        shell.redoRequested.connect(self._on_shell_redo)
        if hasattr(self, "controller"):
            self._push_to_shell()

    def _on_shell_undo(self) -> None:
        if self._active:
            self.undo()

    def _on_shell_redo(self) -> None:
        if self._active:
            self.redo()

    def _push_to_shell(self) -> None:
        shell = self._wired_shell
        if shell is None:
            return
        name, saved = self.document_state()
        shell.set_document(name, saved)
        if self._active:
            can_undo, can_redo = self.undo_state()
            shell.set_undo_enabled(can_undo)
            shell.set_redo_enabled(can_redo)

    def on_enter(self) -> None:
        self._active = True
        self._push_to_shell()
        if self._started:
            self.view.setFocus(Qt.OtherFocusReason)

    def on_leave(self) -> None:
        self._active = False
        self.session.close()
        shell = self._wired_shell
        if shell is not None:
            shell.set_undo_enabled(False)
            shell.set_redo_enabled(False)

    def can_close(self) -> bool:
        """저장하지 않은 변경이 있으면 물어본다. False면 닫지 않는다."""
        if not self.controller.is_modified:
            return True
        choice = self.dialogs.ask_unsaved(self, self.controller.document.name)
        if choice == "save":
            return self.save()
        return choice == "discard"

    def go_to(self, page_id: str) -> None:
        self.navigateRequested.emit(page_id)
        if self._wired_shell is not None:
            self._wired_shell.navigate(page_id)

    def go_to_list(self) -> None:
        self.go_to(PAGE_LIST)

    def notify(self, text: str, kind: str = "info") -> None:
        if self._wired_shell is not None:
            self._wired_shell.show_toast(text, kind)
        elif self.isVisible():
            Toast.show_message(self, text, kind)
        self.last_message = text

    # ================================================================ 상태

    def is_started(self) -> bool:
        return self._started

    def document_state(self) -> tuple[str, bool | None]:
        """(위쪽 줄에 보일 이름, 저장 상태). 저장한 적 없는 새 문서는 상태를 숨긴다(None)."""
        controller = self.controller
        if not self._started:
            return ("새 카드", None)
        name = os.path.basename(controller.path) if controller.path else controller.document.name
        if controller.is_modified:
            return (name, False)
        return (name, True if controller.path else None)

    def undo_state(self) -> tuple[bool, bool]:
        stack = self.controller.undo_stack
        return (self._started and stack.canUndo(), self._started and stack.canRedo())

    def undo(self) -> None:
        self.session.close()
        self.controller.undo_stack.undo()

    def redo(self) -> None:
        self.session.close()
        self.controller.undo_stack.redo()

    def _sync_document_state(self, *_args) -> None:
        name, saved = self.document_state()
        self.documentStateChanged.emit(name, saved)
        self._push_to_shell()

    def _sync_undo_state(self, *_args) -> None:
        self.undoStateChanged.emit(*self.undo_state())
        self._push_to_shell()

    def _sync_footer(self) -> None:
        if not self._started:
            hint = "시작할 모양을 고르면 꾸미는 화면이 열립니다."
        else:
            count = len(self.controller.problems())
            hint = (f"확인할 것이 {count}개 있습니다. 아무것도 고르지 않으면 오른쪽에 목록이 보입니다."
                    if count else "카드를 다 꾸몄으면 명단을 넣습니다.")
        if hint != self.footer_hint():
            self.set_footer_hint(hint)

    def _sync_view(self) -> None:
        self._stack.setCurrentWidget(self.editor_host if self._started else self.start_panel)
        self._sync_footer()
        self._sync_document_state()
        self._sync_undo_state()

    def _on_document_changed(self) -> None:
        if not is_blank(self.controller.document) or self.controller.path:
            self._started = True
        self._sync_view()

    def _sync_columns(self) -> None:
        store = self.workspace.store
        if store is None:
            self.insert_panel.set_columns(None)
        else:
            self.insert_panel.set_columns(
                [(column.name, "image" if column.type == "photo" else "text") for column in store.columns()])

    def _apply_theme(self, *_args) -> None:
        self.setStyleSheet(build_stylesheet())
        self.view.set_palette(palette_for())
        self._refresh_local_icons()
        self._sync_strip()

    def _sync_strip(self) -> None:
        dark = ThemeManager.instance().theme.is_dark
        self.preview_bar.set_strip_allowed(dark or self.width() >= STRIP_MIN_WIDTH)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._sync_strip()
        compact = self.height() < COMPACT_HEIGHT
        self.insert_panel.set_compact(compact)
        self.layers_panel.set_compact(compact)

    # ================================================================ 메뉴

    def _action(self, text: str, slot, shortcut: str | None = None) -> QAction:
        action = QAction(text, self)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
            action.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        action.triggered.connect(lambda _checked=False: slot())
        self.addAction(action)
        return action

    def _build_file_menu(self) -> None:
        self.new_action = self._action("새로 만들기", self.new_design, "Ctrl+N")
        self.open_action = self._action("열기…", self.open_design, "Ctrl+O")
        self.save_action = self._action("저장", self.save, "Ctrl+S")
        self.save_as_action = self._action("다른 이름으로 저장…", self.save_as, "Ctrl+Shift+S")
        self.export_action = self._action("미리보기 그림으로 내보내기…", self.export_preview, "Ctrl+E")
        self.file_menu = QMenu(self)
        self.file_menu.addActions([self.new_action, self.open_action])
        self.file_menu.addSeparator()
        self.file_menu.addActions([self.save_action, self.save_as_action])
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.export_action)
        self.file_button = SecondaryButton("파일", size="sm")
        as_menu_button(self.file_button)
        self.file_button.setToolTip("새로 만들기, 열기, 저장, 그림으로 내보내기")
        self.file_button.setMenu(self.file_menu)
        self.save_button = IconButton("check", "저장 (Ctrl+S)", size="sm")
        self.save_button.set_icon_name(None)
        self.save_button.clicked.connect(lambda: self.save())
        layout = self.canvas_bar.layout()
        layout.insertWidget(0, self.file_button)
        layout.insertWidget(1, self.save_button)
        layout.insertSpacing(2, SPACE["sm"])

    def _build_item_menu(self) -> None:
        controller = self.controller
        menu = QMenu(self)
        for text, slot in (
            ("하나 더 만들기\tCtrl+D", controller.duplicate),
            (None, None),
            ("맨 앞으로 가져오기", controller.bring_to_front),
            ("맨 뒤로 보내기", controller.send_to_back),
            (None, None),
            ("지우기\tDelete", controller.delete),
        ):
            if text is None:
                menu.addSeparator()
                continue
            action = QAction(text, self)
            action.triggered.connect(lambda _checked=False, run=slot: run())
            menu.addAction(action)
        self.item_menu = menu
        button = self.inspector.more_button
        button.setProperty("menu", "true")
        button.setMenu(menu)

    def _refresh_local_icons(self) -> None:
        self.save_button.setIcon(icons.icon("save", "text", 16))
        self.save_button.setIconSize(QSize(16, 16))
        self.inspector.more_button.setIcon(icons.icon("more", "text", 16))
        self.inspector.more_button.setIconSize(QSize(16, 16))

    # ================================================================ 시작 / 파일

    def show_start(self) -> None:
        self._started = False
        self.refresh_recents()
        self._sync_view()

    def _confirm_replace(self) -> bool:
        return not self._started or self.can_close()

    def start_with_template(self, template_key: str) -> None:
        panel = self.start_panel
        template = None if template_key == "blank" else template_key
        self.controller.new_document(
            panel.preset_key(), template, orientation=panel.orientation(), dual_side=panel.dual_side())
        self._begin()

    def _begin(self) -> None:
        self._started = True
        self._sync_view()
        self.view.zoom_fit()
        self.view.setFocus(Qt.OtherFocusReason)

    def new_design(self) -> bool:
        """지금 디자인을 닫고 시작 화면으로 돌아간다."""
        if not self._confirm_replace():
            return False
        self.controller.new_document()
        self.show_start()
        return True

    def open_design(self, path: str | None = None) -> bool:
        if not self._confirm_replace():
            return False
        path = path or self.dialogs.open_design(self)
        if not path:
            return False
        try:
            self.controller.open(path)
        except (storage.StorageError, EditorError, OSError, ValueError) as exc:
            self.notify(f"디자인을 열지 못했습니다. {exc}", "error")
            return False
        self._remember(path)
        self._begin()
        return True

    def start_from_image(self, path: str | None = None) -> bool:
        """완성된 그림 한 장을 카드 전체에 깐 디자인으로 시작한다."""
        if not self._confirm_replace():
            return False
        path = path or self.dialogs.open_image(self, "카드로 쓸 그림 고르기")
        if not path:
            return False
        try:
            document = storage.document_from_single_image(path)
        except storage.StorageError as exc:
            self.notify(str(exc), "error")
            return False
        self.controller.set_document(document, None, owns_workdir=True)
        self._begin()
        return True

    def save(self) -> bool:
        if not self.controller.path:
            return self.save_as()
        return self._save_to(self.controller.path)

    def save_as(self) -> bool:
        suggested = self.controller.path or f"{self.controller.document.name}{storage.EXTENSION}"
        path = self.dialogs.save_design(self, suggested)
        return bool(path) and self._save_to(path)

    def _save_to(self, path: str) -> bool:
        self.view.commit_text_edit()
        try:
            saved = self.controller.save(path)
        except (storage.StorageError, EditorError, OSError) as exc:
            self.notify(f"저장하지 못했습니다. {exc}", "error")
            return False
        self._sync_document_state()
        self._remember(saved)
        self.notify(f"{os.path.basename(saved)}에 저장했습니다.", "ok")
        return True

    # ---- 최근에 작업한 디자인 ----
    def _remember(self, path: str) -> None:
        try:
            RecentProjects().add(str(path), self.controller.document.name)
        except Exception:
            pass
        self.refresh_recents()

    def refresh_recents(self) -> None:
        try:
            self.start_panel.set_recents(RecentProjects().entries(existing_only=True))
        except Exception:
            pass

    def export_preview(self, path: str | None = None) -> bool:
        """지금 보이는 면을 (미리 보는 사람의 내용으로) PNG 그림으로 저장한다."""
        suggested = f"{self.controller.document.name}.png"
        path = path or self.dialogs.save_image(self, suggested)
        if not path:
            return False
        image = self.controller.export_preview_image(dpi=300.0)
        if not image.save(path):
            self.notify("그림을 저장하지 못했습니다. 저장할 위치를 확인하세요.", "error")
            return False
        self.notify(f"{os.path.basename(path)}로 내보냈습니다.", "ok")
        return True

    # ================================================================ 넣기

    def add(self, kind: str):
        try:
            item = self.controller.add_item(kind)
        except EditorError as exc:
            self.notify(str(exc), "error")
            return None
        self.view.setFocus(Qt.OtherFocusReason)
        return item

    def add_image_file(self, path: str | None = None):
        path = path or self.dialogs.open_image(self, "카드에 넣을 그림 고르기")
        if not path:
            return None
        try:
            return self.controller.add_image_from_file(path)
        except EditorError as exc:
            self.notify(str(exc), "error")
            return None

    def add_field(self, column: str, kind: str = "text"):
        try:
            return self.controller.add_bound_field(column, kind)
        except EditorError as exc:
            self.notify(str(exc), "error")
            return None

    def add_back_side(self) -> None:
        try:
            self.controller.set_card_setup(dual_side=True)
        except EditorError as exc:
            self.notify(str(exc), "error")
            return
        self.controller.set_side("back")

    # ================================================================ 캔버스에서 온 일

    def go_to_problem(self, problem: Problem) -> None:
        if problem.side != self.controller.current_side:
            self.controller.set_side(problem.side)
        if problem.item_id:
            self.controller.select(problem.item_id)
        else:
            self.controller.clear_selection()

    def _on_bound_edit(self, item_id: str) -> None:
        self.controller.select(item_id)
        self.inspector.text.focus_content()

    def _on_item_activated(self, item_id: str) -> None:
        item = self.controller.item(item_id)
        if item is None:
            return
        self.controller.select(item_id)
        if item.kind == "image" and not (item.source is not None and item.source.columns()):
            self.inspector.image.choose_image()
        elif item.kind in ("barcode", "qr"):
            self.inspector.code.data_edit.setFocus()
            self.inspector.code.data_edit.selectAll()


__all__ = ["DesignPage", "FileDialogs", "is_blank", "pil_to_qimage"]
