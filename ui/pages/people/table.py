"""명단 표: 머리글, 칸 그리기, 칸 안에서 고치기, 키보드."""

from __future__ import annotations

from PySide6.QtCore import QModelIndex, QPoint, QRect, QRectF, Qt, Signal
from PySide6.QtGui import QFont, QFontMetrics, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDateEdit,
    QHeaderView,
    QLineEdit,
    QStyle,
    QStyledItemDelegate,
    QTableView,
    QWidget,
)

from ui.theme import FONT_SIZES, ThemeManager
from ui.widgets import ComboBox, paint_icon, theme_color

from .model import (
    KIND_CHECK,
    KIND_FIELD,
    KIND_STATUS,
    PeopleTableModel,
    PhotoPathRole,
    SpecRole,
    StatusRole,
)
from .inputs import Rethemed, date_value, make_date_edit, set_date_value
from .thumbs import THUMB_H, THUMB_W, ThumbLoader

ROW_HEIGHT = 48
HEADER_HEIGHT = 40
CHECK_WIDTH = 44
PHOTO_WIDTH = 60
STATUS_WIDTH = 140
FIELD_WIDTH = 150
CELL_PAD = 10


def paint_check(painter: QPainter, rect: QRectF, state: Qt.CheckState, hover: bool = False) -> None:
    painter.save()
    painter.setRenderHint(QPainter.Antialiasing, True)
    box = QRectF(rect.center().x() - 8, rect.center().y() - 8, 16, 16)
    if state == Qt.Unchecked:
        painter.setPen(QPen(theme_color("accent" if hover else "indicator_line"), 1))
        painter.setBrush(theme_color("surface_raised"))
        painter.drawRoundedRect(box.adjusted(0.5, 0.5, -0.5, -0.5), 4, 4)
    else:
        painter.setPen(Qt.NoPen)
        painter.setBrush(theme_color("accent"))
        painter.drawRoundedRect(box, 4, 4)
        name = "check" if state == Qt.Checked else "minus"
        paint_icon(painter, name, box.adjusted(2, 2, -2, -2), "text_on_accent", stroke=2.6)
    painter.restore()


class PeopleHeader(Rethemed, QHeaderView):
    checkToggled = Signal()
    menuRequested = Signal(int, QPoint)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(Qt.Horizontal, parent)
        self.setFixedHeight(HEADER_HEIGHT)
        self.setSectionsClickable(True)
        self.setHighlightSections(False)
        self.setMinimumSectionSize(40)
        self.setMouseTracking(True)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._on_context)
        self.setStyleSheet("QHeaderView { background: transparent; border: 0; }")
        self._hover = -1
        self._check_state = Qt.Unchecked
        self._press_section = -1
        self.follow_theme()

    def set_check_state(self, state: Qt.CheckState) -> None:
        if state != self._check_state:
            self._check_state = state
            self.viewport().update()

    def check_state(self) -> Qt.CheckState:
        return self._check_state

    def _spec(self, section: int):
        model = self.model()
        return model.headerData(section, Qt.Horizontal, SpecRole) if model is not None else None

    def _menu_zone(self, section: int) -> QRect:
        x = self.sectionViewportPosition(section)
        return QRect(x + self.sectionSize(section) - 26, 0, 26, self.height())

    def paintSection(self, painter: QPainter, rect: QRect, logical: int) -> None:
        spec = self._spec(logical)
        if spec is None:
            return
        if spec.kind == KIND_CHECK:
            paint_check(painter, QRectF(rect), self._check_state, hover=self._hover == logical)
            return

        painter.save()
        hovered = self._hover == logical
        font = QFont(self.font())
        font.setPixelSize(FONT_SIZES["small"])
        font.setWeight(QFont.Weight.Medium)
        painter.setFont(font)
        painter.setPen(theme_color("text" if hovered else "text_secondary"))
        left = rect.left() + (4 if spec.is_photo else CELL_PAD)
        right = rect.right() - 6
        model = self.model()
        state = model.sort_state() if isinstance(model, PeopleTableModel) else None
        sorted_here = state is not None and state[0] == logical
        if hovered and spec.kind == KIND_FIELD:
            zone = QRectF(self._menu_zone(logical))
            paint_icon(painter, "chevron-down", QRectF(zone.center().x() - 7, zone.center().y() - 7, 14, 14),
                       "text_muted", stroke=2.0)
            right = int(zone.left())
        metrics = QFontMetrics(font)
        room = right - left - (18 if sorted_here else 0)
        text = metrics.elidedText(str(model.headerData(logical, Qt.Horizontal) or ""), Qt.ElideRight, max(0, room))
        painter.drawText(QRect(left, rect.top(), max(0, room), rect.height()), Qt.AlignLeft | Qt.AlignVCenter, text)
        if sorted_here:
            x = left + metrics.horizontalAdvance(text) + 4
            icon = "chevron-up" if state[1] == "asc" else "chevron-down"
            paint_icon(painter, icon, QRectF(x, rect.center().y() - 6, 13, 13), "accent_text", stroke=2.4)
        painter.restore()

    def mouseMoveEvent(self, event) -> None:
        section = self.logicalIndexAt(event.position().toPoint())
        if section != self._hover:
            self._hover = section
            self.viewport().update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:
        self._hover = -1
        self.viewport().update()
        super().leaveEvent(event)

    def mousePressEvent(self, event) -> None:
        self._press_section = self.logicalIndexAt(event.position().toPoint())
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        pos = event.position().toPoint()
        section = self.logicalIndexAt(pos)
        pressed, self._press_section = self._press_section, -1
        spec = self._spec(section) if section >= 0 else None
        if event.button() == Qt.LeftButton and section == pressed and spec is not None:
            in_menu = (spec.kind == KIND_FIELD and self._menu_zone(section).contains(pos)
                       and self.cursor().shape() != Qt.SplitHCursor)
            if spec.kind == KIND_CHECK or in_menu:
                self.blockSignals(True)
                super().mouseReleaseEvent(event)
                self.blockSignals(False)
                if in_menu:
                    corner = QPoint(self._menu_zone(section).left(), self.height())
                    self.menuRequested.emit(section, self.viewport().mapToGlobal(corner))
                else:
                    self.checkToggled.emit()
                return
        super().mouseReleaseEvent(event)

    def _on_context(self, pos: QPoint) -> None:
        section = self.logicalIndexAt(pos)
        spec = self._spec(section) if section >= 0 else None
        if spec is not None and spec.kind != KIND_CHECK:
            self.menuRequested.emit(section, self.viewport().mapToGlobal(pos))


class CellDelegate(QStyledItemDelegate):
    photoPickRequested = Signal(QModelIndex)

    def __init__(self, table: "PeopleTable"):
        super().__init__(table)
        self._table = table

    # ---- 그리기 ----
    def paint(self, painter: QPainter, option, index: QModelIndex) -> None:
        spec = index.data(SpecRole)
        table = self._table
        rect = QRectF(option.rect)
        selected = bool(option.state & QStyle.State_Selected)
        hovered = table.hover_row() == index.row()
        painter.save()
        if selected:
            painter.fillRect(rect, theme_color("accent_soft"))
        elif hovered:
            painter.fillRect(rect, theme_color("surface_hover"))
        painter.setPen(QPen(theme_color("line_soft"), 1))
        painter.drawLine(option.rect.left(), option.rect.bottom(), option.rect.right(), option.rect.bottom())

        if spec is None:
            painter.restore()
            return
        if spec.kind == KIND_CHECK:
            state = index.data(Qt.CheckStateRole)
            if state is not None:
                paint_check(painter, rect, Qt.CheckState(state), hover=hovered)
        elif spec.is_photo:
            self._paint_photo(painter, rect, index)
        else:
            self._paint_text(painter, option, index, spec, selected)

        if (table.hasFocus() and table.currentIndex() == index and spec.kind == KIND_FIELD
                and table.shows_cell_focus()):
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.setPen(QPen(theme_color("focus"), 1.5))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(rect.adjusted(2, 3, -2, -4), 6, 6)
        painter.restore()

    def _paint_text(self, painter: QPainter, option, index: QModelIndex, spec, selected: bool) -> None:
        font = QFont(option.font)
        role = "text"
        if spec.kind == KIND_STATUS:
            status = index.data(StatusRole)
            text = status[0] if status else ""
            role = status[1] if status else "text_muted"
            font.setPixelSize(FONT_SIZES["small"])
        else:
            text = index.data(Qt.DisplayRole) or ""
            if index.column() == self._table.name_column():
                strong = ThemeManager.instance().font_spec.strong_weight
                font.setWeight(QFont.Weight(strong) if selected else QFont.Weight.Medium)
        painter.setFont(font)
        painter.setPen(theme_color(role))
        box = option.rect.adjusted(CELL_PAD, 0, -CELL_PAD, 0)
        align = Qt.AlignLeft | Qt.AlignVCenter
        if spec.type == "number":
            align = Qt.AlignRight | Qt.AlignVCenter
            box = box.adjusted(0, 0, -6, 0)
        text = QFontMetrics(font).elidedText(str(text), Qt.ElideRight, box.width())
        painter.drawText(box, align, text)

    def _paint_photo(self, painter: QPainter, rect: QRectF, index: QModelIndex) -> None:
        painter.setRenderHint(QPainter.Antialiasing, True)
        box = QRectF(rect.left() + 4, rect.center().y() - THUMB_H / 2, THUMB_W, THUMB_H)
        path = index.data(PhotoPathRole)
        if not path:
            pen = QPen(theme_color("warn_mark"), 1.4, Qt.CustomDashLine)
            pen.setDashPattern([3, 2.5])
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(box.adjusted(0.7, 0.7, -0.7, -0.7), 4, 4)
            return
        clip = QPainterPath()
        clip.addRoundedRect(box, 4, 4)
        image = self._table.thumbs.get(path)
        if image is None or image.isNull():
            painter.fillPath(clip, theme_color("surface_pressed"))
            return
        painter.setClipPath(clip)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        painter.drawImage(box, image)

    # ---- 고치기 ----
    def createEditor(self, parent: QWidget, option, index: QModelIndex):
        spec = index.data(SpecRole)
        if spec is None or spec.kind != KIND_FIELD:
            return None
        if spec.is_photo:
            self.photoPickRequested.emit(index)
            return None
        if spec.type == "date":
            return make_date_edit(parent)
        if spec.type == "list":
            editor = ComboBox(parent=parent)
            editor.addItem("")
            editor.addItems([str(c) for c in spec.column.options.get("choices", [])])
            return editor
        editor = QLineEdit(parent)
        if spec.type == "time":
            editor.setPlaceholderText("14:30")
        return editor

    def setEditorData(self, editor: QWidget, index: QModelIndex) -> None:
        value = index.data(Qt.EditRole)
        text = index.data(Qt.DisplayRole) or ""
        if isinstance(editor, QDateEdit):
            set_date_value(editor, value, fallback_today=True)
        elif isinstance(editor, ComboBox):
            if text and editor.findText(text) < 0:
                editor.addItem(text)
            editor.setCurrentIndex(max(0, editor.findText(text)))
        else:
            editor.setText(text)
            editor.selectAll()

    def setModelData(self, editor: QWidget, model, index: QModelIndex) -> None:
        if isinstance(editor, QDateEdit):
            value = date_value(editor)
        elif isinstance(editor, ComboBox):
            value = editor.currentText() or None
        else:
            value = editor.text()
        model.setData(index, value, Qt.EditRole)

    def updateEditorGeometry(self, editor: QWidget, option, index: QModelIndex) -> None:
        rect = option.rect
        height = 40
        editor.setGeometry(rect.left() + 2, rect.top() + (rect.height() - height) // 2 - 0,
                           max(90, rect.width() - 4), height)


class PeopleTable(Rethemed, QTableView):
    rowActivated = Signal(int)          # 두 번 누른 줄
    deleteRequested = Signal()
    photoPickRequested = Signal(QModelIndex)
    columnMenuRequested = Signal(int, QPoint)
    currentRowChanged = Signal(int)
    allToggled = Signal(bool)

    def __init__(self, model: PeopleTableModel, parent: QWidget | None = None):
        super().__init__(parent)
        self._model = model
        self.thumbs = ThumbLoader(self)
        self.thumbs.loaded.connect(self.viewport().update)
        self._hover_row = -1
        self._last_toggled_row = -1
        self._pressed_photo = None
        self._keyboard = False
        self._fit_columns = True

        header = PeopleHeader(self)
        self.setHorizontalHeader(header)
        self.setModel(model)
        self.verticalHeader().setVisible(False)
        self.verticalHeader().setDefaultSectionSize(ROW_HEIGHT)
        self.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.setShowGrid(False)
        self.setWordWrap(False)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setEditTriggers(QAbstractItemView.EditKeyPressed | QAbstractItemView.AnyKeyPressed)
        self.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.verticalScrollBar().setSingleStep(ROW_HEIGHT // 2)
        self.setMouseTracking(True)
        self.setTabKeyNavigation(False)
        self.setCornerButtonEnabled(False)
        self.setStyleSheet("QTableView { border: 0; background: transparent; }")
        self.verticalScrollBar().setStyleSheet(f"QScrollBar:vertical {{ margin-top: {HEADER_HEIGHT + 2}px; }}")
        self.setAccessibleName("명단")

        self._delegate = CellDelegate(self)
        self._delegate.photoPickRequested.connect(self.photoPickRequested)
        self.setItemDelegate(self._delegate)

        header.checkToggled.connect(self._toggle_all)
        header.menuRequested.connect(self.columnMenuRequested)
        header.sectionClicked.connect(self._on_section_clicked)
        model.modelReset.connect(self._after_reset)
        model.checkedChanged.connect(self._sync_header_check)
        model.orderChanged.connect(header.viewport().update)
        self.doubleClicked.connect(self._on_double_click)
        self.follow_theme()
        self._layout_columns()

    # ---- 열 ----
    def header(self) -> PeopleHeader:
        return self.horizontalHeader()

    def name_column(self) -> int:
        """굵게 보여 줄 열: '이름'이 있으면 그 열, 없으면 첫 글자 열."""
        fields = [(i, s) for i, s in enumerate(self._model.specs) if s.kind == KIND_FIELD and not s.is_photo]
        for index, spec in fields:
            if spec.name in ("이름", "성명", "name", "Name"):
                return index
        return fields[0][0] if fields else -1

    def _layout_columns(self) -> None:
        header = self.header()
        specs = self._model.specs
        fields = [i for i, s in enumerate(specs) if s.kind == KIND_FIELD and not s.is_photo]
        self._fit_columns = len(fields) <= 7
        header.setStretchLastSection(False)
        for index, spec in enumerate(specs):
            if spec.kind == KIND_CHECK:
                header.setSectionResizeMode(index, QHeaderView.Fixed)
                self.setColumnWidth(index, CHECK_WIDTH)
            elif spec.is_photo:
                header.setSectionResizeMode(index, QHeaderView.Fixed)
                self.setColumnWidth(index, PHOTO_WIDTH)
            elif spec.kind == KIND_STATUS:
                header.setSectionResizeMode(index, QHeaderView.Fixed if self._fit_columns else QHeaderView.Interactive)
                self.setColumnWidth(index, STATUS_WIDTH)
            else:
                header.setSectionResizeMode(index, QHeaderView.Stretch if self._fit_columns else QHeaderView.Interactive)
                if not self._fit_columns:
                    self.setColumnWidth(index, FIELD_WIDTH)
        if not fields:
            header.setStretchLastSection(True)

    def _after_reset(self) -> None:
        self._hover_row = -1
        self._last_toggled_row = -1
        self._layout_columns()
        self._sync_header_check()

    def _sync_header_check(self) -> None:
        self.header().set_check_state(self._model.header_check_state())

    def _toggle_all(self) -> None:
        self.allToggled.emit(self._model.toggle_all_in_view())

    def _on_section_clicked(self, section: int) -> None:
        spec = self._model.headerData(section, Qt.Horizontal, SpecRole)
        if spec is None or spec.kind == KIND_CHECK:
            return
        state = self._model.sort_state()
        if state is None or state[0] != section:
            self._model.sort(section, Qt.AscendingOrder)
        elif state[1] == "asc":
            self._model.sort(section, Qt.DescendingOrder)
        else:
            self._model.set_order(None)

    # ---- 상태 ----
    def hover_row(self) -> int:
        return self._hover_row

    def shows_cell_focus(self) -> bool:
        return self._keyboard

    def current_row(self) -> int:
        index = self.currentIndex()
        return index.row() if index.isValid() else -1

    def selected_rows(self) -> list[int]:
        model = self.selectionModel()
        if model is None:
            return []
        return sorted({index.row() for index in model.selectedRows()})

    def selected_ids(self) -> list[int]:
        rows = self.selected_rows()
        if len(rows) > 50:
            # 많이 골랐을 때는 줄을 하나씩 읽지 않고 번호 목록에서 바로 찾는다.
            all_ids = self._model.ids()
            return [all_ids[row] for row in rows if row < len(all_ids)]
        ids = [self._model.id_at(row) for row in rows]
        return [i for i in ids if i is not None]

    def go_to_row(self, row: int, column: int | None = None) -> None:
        if not 0 <= row < self._model.rowCount():
            return
        if column is None:
            column = max(0, self.currentIndex().column()) if self.currentIndex().isValid() else 0
        index = self._model.index(row, column)
        self.setCurrentIndex(index)
        self.selectRow(row)
        self.setCurrentIndex(index)
        self.scrollTo(index, QAbstractItemView.EnsureVisible)

    def currentChanged(self, current: QModelIndex, previous: QModelIndex) -> None:
        super().currentChanged(current, previous)
        if current.row() != previous.row():
            self.currentRowChanged.emit(current.row() if current.isValid() else -1)
        self.viewport().update()

    # ---- 마우스 ----
    def mouseMoveEvent(self, event) -> None:
        row = self.rowAt(event.position().toPoint().y())
        if row != self._hover_row:
            self._hover_row = row
            self.viewport().update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:
        if self._hover_row != -1:
            self._hover_row = -1
            self.viewport().update()
        super().leaveEvent(event)

    def mousePressEvent(self, event) -> None:
        self._keyboard = False
        index = self.indexAt(event.position().toPoint())
        if event.button() == Qt.LeftButton and index.isValid() and index.column() == 0:
            self._click_check(index.row(), bool(event.modifiers() & Qt.ShiftModifier))
            self.setFocus(Qt.MouseFocusReason)
            event.accept()
            return
        spec = index.data(SpecRole) if index.isValid() else None
        empty_photo = (event.button() == Qt.LeftButton and spec is not None and spec.is_photo
                       and not index.data(PhotoPathRole))
        self._pressed_photo = index if empty_photo else None
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        super().mouseReleaseEvent(event)
        # 비어 있는 사진 칸은 한 번만 눌러도 파일을 고르게 한다.
        index = self.indexAt(event.position().toPoint())
        if event.button() != Qt.LeftButton or not index.isValid() or index != self._pressed_photo:
            self._pressed_photo = None
            return
        self._pressed_photo = None
        self.photoPickRequested.emit(index)

    def mouseDoubleClickEvent(self, event) -> None:
        index = self.indexAt(event.position().toPoint())
        if index.isValid() and index.column() == 0:
            self.mousePressEvent(event)
            return
        super().mouseDoubleClickEvent(event)

    def _click_check(self, row: int, extend: bool) -> None:
        record_id = self._model.id_at(row)
        if record_id is None:
            return
        target = not self._model.is_checked(record_id)
        if extend and self._last_toggled_row >= 0:
            low, high = sorted((self._last_toggled_row, row))
            self._model.set_checked(self._model.ids()[low:high + 1], target)
        else:
            self._model.set_checked([record_id], target)
        self._last_toggled_row = row

    def _on_double_click(self, index: QModelIndex) -> None:
        spec = index.data(SpecRole) if index.isValid() else None
        if spec is not None and spec.is_photo and not index.data(PhotoPathRole):
            return      # 빈 사진 칸은 누를 때 이미 파일 고르기를 열었다
        if index.isValid() and index.column() != 0:
            self.rowActivated.emit(index.row())

    # ---- 키보드 ----
    def keyPressEvent(self, event) -> None:
        self._keyboard = True
        key = event.key()
        if self.state() != QAbstractItemView.EditingState:
            if key in (Qt.Key_Return, Qt.Key_Enter):
                self.edit_current()
                event.accept()
                return
            if key == Qt.Key_Space:
                self.toggle_selected_checks()
                event.accept()
                return
            if key == Qt.Key_Delete:
                self.deleteRequested.emit()
                event.accept()
                return
        super().keyPressEvent(event)
        self.viewport().update()

    def edit_current(self) -> None:
        index = self.currentIndex()
        if not index.isValid():
            return
        spec = index.data(SpecRole)
        if spec is None or spec.kind != KIND_FIELD:
            # 체크 칸이나 인쇄 칸에 있으면 그 줄의 첫 항목을 고친다.
            column = self.name_column()
            if column < 0:
                return
            index = self._model.index(index.row(), column)
            self.setCurrentIndex(index)
        self.edit(index)

    def toggle_selected_checks(self) -> None:
        ids = self.selected_ids()
        if not ids and self.current_row() >= 0:
            ids = [self._model.id_at(self.current_row())]
        ids = [i for i in ids if i is not None]
        if not ids:
            return
        target = not all(self._model.is_checked(i) for i in ids)
        self._model.set_checked(ids, target)

    def edit(self, index: QModelIndex, trigger=None, event=None) -> bool:
        if trigger is None:
            return super().edit(index, QAbstractItemView.AllEditTriggers, None)
        if trigger == QAbstractItemView.AnyKeyPressed and event is not None:
            # 글자를 치면 바로 고치기 시작하되, 체크 칸·사진 칸에서는 시작하지 않는다.
            spec = index.data(SpecRole)
            if spec is None or spec.kind != KIND_FIELD or spec.is_photo or spec.type in ("date", "list"):
                return False
        return super().edit(index, trigger, event)
