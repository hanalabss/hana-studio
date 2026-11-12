"""읽기 전용 표: 줄마다 dict 하나, 칸마다 글자와 색 역할."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from PySide6.QtCore import QAbstractTableModel, QModelIndex, QRect, Qt
from PySide6.QtGui import QFont, QPainter
from PySide6.QtWidgets import (
    QAbstractItemView, QHeaderView, QStyle, QStyledItemDelegate, QTableView, QWidget,
)

from ui.theme import ThemeManager
from ui.widgets import theme_color

TONE_TEXT = {
    "ok": "ok_text", "warn": "warn_text", "error": "danger_text", "busy": "accent_text",
    "idle": "text_muted", "muted": "text_muted",
}
ROW_HEIGHT = 42


@dataclass
class ColumnSpec:
    key: str
    title: str
    width: int | None = None          # None이면 남는 폭을 나눠 가진다
    align: Qt.AlignmentFlag = Qt.AlignLeft
    tone_key: str | None = None       # 줄 dict에서 색 역할 이름을 읽을 키
    optional: int = 0                 # 0보다 크면 표가 좁을 때 숨긴다(큰 수부터)


class RowsModel(QAbstractTableModel):
    """rows: dict 목록. '_tone'(줄 전체 글자색), '_current'(강조 줄), '_tip'(풍선 도움말)을 알아듣는다."""

    def __init__(self, columns: Sequence[ColumnSpec], parent=None) -> None:
        super().__init__(parent)
        self._columns = list(columns)
        self._rows: list[dict[str, Any]] = []
        ThemeManager.instance().themeChanged.connect(self._on_theme)

    def _on_theme(self, *_args) -> None:
        if self._rows:
            self.dataChanged.emit(self.index(0, 0), self.index(len(self._rows) - 1, len(self._columns) - 1))

    def columns(self) -> list[ColumnSpec]:
        return list(self._columns)

    def set_columns(self, columns: Sequence[ColumnSpec]) -> None:
        self.beginResetModel()
        self._columns = list(columns)
        self.endResetModel()

    def set_rows(self, rows: Sequence[dict[str, Any]]) -> None:
        self.beginResetModel()
        self._rows = [dict(row) for row in rows]
        self.endResetModel()

    def rows(self) -> list[dict[str, Any]]:
        return self._rows

    def row(self, index: int) -> dict[str, Any]:
        return self._rows[index]

    def update_row(self, index: int, **values: Any) -> None:
        if not 0 <= index < len(self._rows):
            return
        self._rows[index].update(values)
        self.dataChanged.emit(self.index(index, 0), self.index(index, len(self._columns) - 1))

    def find(self, predicate: Callable[[dict[str, Any]], bool]) -> int:
        for index, row in enumerate(self._rows):
            if predicate(row):
                return index
        return -1

    def text(self, row: int, key: str) -> str:
        value = self._rows[row].get(key, "")
        return "" if value is None else str(value)

    # ---- Qt ----
    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._columns)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole):
        if orientation == Qt.Horizontal and 0 <= section < len(self._columns):
            if role == Qt.DisplayRole:
                return self._columns[section].title
            if role == Qt.TextAlignmentRole:
                return int(self._columns[section].align | Qt.AlignVCenter)
        return None

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):
        if not index.isValid():
            return None
        row = self._rows[index.row()]
        column = self._columns[index.column()]
        if role == Qt.DisplayRole:
            value = row.get(column.key, "")
            return "" if value is None else str(value)
        if role == Qt.TextAlignmentRole:
            return int(column.align | Qt.AlignVCenter)
        if role == Qt.ForegroundRole:
            tone = row.get(column.tone_key) if column.tone_key else None
            tone = tone or row.get("_tone")
            if tone:
                return theme_color(TONE_TEXT.get(tone, "text"))
            return None
        if role == Qt.BackgroundRole and row.get("_current"):
            return theme_color("accent_soft")
        if role == Qt.FontRole and (row.get("_current") or (column.tone_key and row.get(column.tone_key) in ("busy", "error"))):
            font = QFont()
            font.setWeight(QFont.Weight(ThemeManager.instance().font_spec.strong_weight))
            return font
        if role == Qt.ToolTipRole:
            return row.get("_tip") or None
        return None

    def flags(self, index: QModelIndex):
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable if index.isValid() else Qt.NoItemFlags


class _RowDelegate(QStyledItemDelegate):
    """글자색과 강조 줄을 모델이 말한 대로 그린다(스타일시트의 ::item 색에 덮이지 않게)."""

    _PAD = 10

    def paint(self, painter: QPainter, option, index: QModelIndex) -> None:
        painter.save()
        rect: QRect = option.rect
        highlighted = index.data(Qt.BackgroundRole) is not None or bool(option.state & QStyle.State_Selected)
        if highlighted:
            painter.fillRect(rect, theme_color("accent_soft"))
        painter.setPen(theme_color("line_soft"))
        painter.drawLine(rect.bottomLeft(), rect.bottomRight())
        font = index.data(Qt.FontRole)
        if font is not None:
            merged = QFont(option.font)
            merged.setWeight(font.weight())
            painter.setFont(merged)
        else:
            painter.setFont(option.font)
        color = index.data(Qt.ForegroundRole)
        painter.setPen(color if color is not None else theme_color("text"))
        inner = rect.adjusted(self._PAD, 0, -self._PAD, 0)
        text = str(index.data(Qt.DisplayRole) or "")
        alignment = index.data(Qt.TextAlignmentRole) or int(Qt.AlignLeft | Qt.AlignVCenter)
        painter.drawText(inner, int(alignment) | int(Qt.TextSingleLine),
                         painter.fontMetrics().elidedText(text, Qt.ElideRight, inner.width()))
        painter.restore()


class RowsTable(QTableView):
    """줄 단위로 고르는 표. 가로 줄 없음, 줄 높이 고정."""

    def __init__(self, model: RowsModel, parent: QWidget | None = None, selectable: bool = True,
                 view_model=None) -> None:
        """view_model: 걸러 보기용 프록시를 끼울 때 준다. 칸 정보는 늘 model에서 읽는다."""
        super().__init__(parent)
        self.setObjectName("Rows")
        self.setModel(view_model if view_model is not None else model)
        self._model = model
        self.setItemDelegate(_RowDelegate(self))
        if selectable:
            self.viewport().setCursor(Qt.PointingHandCursor)
        self.setShowGrid(False)
        self.setAlternatingRowColors(False)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.SingleSelection if selectable else QAbstractItemView.NoSelection)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setWordWrap(False)
        self.setTextElideMode(Qt.ElideRight)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.verticalHeader().setVisible(False)
        self.verticalHeader().setDefaultSectionSize(ROW_HEIGHT)
        self.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        header = self.horizontalHeader()
        header.setHighlightSections(False)
        header.setSectionsClickable(False)
        header.setMinimumSectionSize(40)
        header.setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        model.modelReset.connect(self.apply_widths)
        self.apply_widths()

    _STRETCH_MIN = 130

    def apply_widths(self) -> None:
        header = self.horizontalHeader()
        for index, column in enumerate(self._model.columns()):
            if column.width is None:
                header.setSectionResizeMode(index, QHeaderView.Stretch)
            else:
                header.setSectionResizeMode(index, QHeaderView.Fixed)
                header.resizeSection(index, column.width)
        self._fit_columns()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._fit_columns()

    def _fit_columns(self) -> None:
        columns = self._model.columns()
        available = self.viewport().width()
        hidden: set[int] = set()

        def needed() -> int:
            return sum((c.width if c.width is not None else self._STRETCH_MIN)
                       for i, c in enumerate(columns) if i not in hidden)

        for index in sorted((i for i, c in enumerate(columns) if c.optional), key=lambda i: -columns[i].optional):
            if needed() <= available:
                break
            hidden.add(index)
        for index in range(len(columns)):
            self.setColumnHidden(index, index in hidden)

    def hidden_columns(self) -> list[str]:
        return [c.key for i, c in enumerate(self._model.columns()) if self.isColumnHidden(i)]

    def current_row(self) -> int:
        indexes = self.selectionModel().selectedRows() if self.selectionModel() else []
        return indexes[0].row() if indexes else -1
