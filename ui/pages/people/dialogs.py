"""명단 화면의 작은 창들: 한 사람 넣기·고치기, 조건으로 찾기, 항목 관리, 내보내기."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDateEdit,
    QFileDialog,
    QHBoxLayout,
    QLineEdit,
    QRadioButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from data.errors import DataError
from data.filters import Condition, Group, operators_for
from data.store import Column, DataStore
from data.values import COLUMN_TYPES, IMAGE_EXTENSIONS, is_empty
from ui.theme import SPACE
from ui.widgets import (
    Banner,
    ComboBox,
    Dialog,
    FormRow,
    IconButton,
    QuietButton,
    SecondaryButton,
    SegmentedControl,
    label,
    set_prop,
)

from .inputs import date_value, make_date_edit, set_date_value
from .model import PhotoIndex
from .preview import CardView, render_card_image
from .text import TYPE_HINTS, TYPE_NAMES, display_value, operator_label, people

PHOTO_FILTER = "사진 (" + " ".join(f"*{ext}" for ext in IMAGE_EXTENSIONS) + ")"
_NO_VALUE_OPS = ("empty", "not_empty")


def pick_photo_file(parent: QWidget | None, start: str = "") -> str:
    path, _ = QFileDialog.getOpenFileName(parent, "사진 파일 고르기", start, PHOTO_FILTER)
    return path


# ---------------------------------------------------------------- 한 사람

class _PhotoInput(QWidget):
    changed = Signal()

    def __init__(self, column: Column, photos: PhotoIndex, parent: QWidget | None = None):
        super().__init__(parent)
        self._column = column
        self._photos = photos
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(SPACE["sm"])
        self.edit = QLineEdit()
        self.edit.setPlaceholderText("사진 파일 이름")
        self.edit.textChanged.connect(self.changed)
        row.addWidget(self.edit, 1)
        self.button = SecondaryButton("파일 고르기", icon="folder")
        self.button.clicked.connect(self._pick)
        row.addWidget(self.button)
        self.setFocusProxy(self.edit)

    def _pick(self) -> None:
        folder = self._photos.folder(self._column.name)
        path = pick_photo_file(self, str(folder) if folder else "")
        if path:
            self.set_path(path)

    def set_path(self, path: str) -> None:
        """사진 폴더 안의 파일이면 이름만, 아니면 전체 경로를 적는다."""
        folder = self._photos.folder(self._column.name)
        chosen = Path(path)
        if folder is not None and chosen.parent.resolve() == Path(folder).resolve():
            self.edit.setText(chosen.name)
        else:
            self.edit.setText(str(chosen))

    def value(self) -> str:
        return self.edit.text().strip()

    def set_value(self, value: Any) -> None:
        self.edit.setText(display_value(value))

    def status(self) -> tuple[str, str | None]:
        value = self.value()
        if not value:
            return "비워 두면 카드의 사진 칸이 빈 채로 나옵니다.", None
        if self._photos.path({self._column.name: value}, self._column.name) is None:
            return "이 이름의 사진 파일을 찾지 못했습니다. '파일 고르기'로 직접 골라 주세요.", "warn"
        return "사진을 찾았습니다.", None


class PersonDialog(Dialog):
    """한 명을 새로 넣거나 고친다. 오른쪽에 그 사람 카드를 바로 보여 준다."""

    saved = Signal(int)

    def __init__(self, workspace, photos: PhotoIndex, record: dict | None = None,
                 parent: QWidget | None = None, initial: dict | None = None):
        store: DataStore = workspace.store
        self._editing = record is not None
        name = _person_name(record, store) if record else ""
        title = (f"{name} 고치기" if name else "한 명 고치기") if self._editing else "한 명 추가"
        super().__init__(title, None, parent, width=860)
        self._workspace = workspace
        self._store = store
        self._photos = photos
        self._record = dict(record) if record else None
        self._inputs: dict[str, QWidget] = {}
        self._rows: dict[str, FormRow] = {}
        self.saved_ids: list[int] = []

        outer = QHBoxLayout()
        outer.setSpacing(SPACE["xl"])
        self.body.addLayout(outer, 1)

        left = QVBoxLayout()
        left.setSpacing(SPACE["md"])
        self.banner = Banner("error", "")
        self.banner.setVisible(False)
        left.addWidget(self.banner)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumHeight(360)
        inner = QWidget()
        inner.setObjectName("ScrollBody")
        scroll.setWidget(inner)
        form = QVBoxLayout(inner)
        form.setContentsMargins(0, 0, SPACE["md"], 0)
        form.setSpacing(SPACE["md"])
        for column in store.columns():
            control = self._make_input(column)
            self._inputs[column.name] = control
            row = FormRow(column.name, control, orientation="horizontal", label_width=96)
            self._rows[column.name] = row
            form.addWidget(row)
        form.addStretch(1)
        left.addWidget(scroll, 1)
        outer.addLayout(left, 1)

        right = QVBoxLayout()
        right.setSpacing(SPACE["md"])
        right.addWidget(label("이렇게 나옵니다", "title"))
        self.card = CardView()
        self.card.setFixedWidth(320)
        right.addWidget(self.card)
        self.preview_hint = label("적는 대로 카드에 바로 나타납니다.", "hint", wrap=True)
        self.preview_hint.setFixedWidth(320)
        right.addWidget(self.preview_hint)
        right.addStretch(1)
        outer.addLayout(right)

        self.add_button("취소", "quiet", "reject", left=True)
        self.more_button = None
        if not self._editing:
            self.more_button = self.add_button("넣고 한 명 더", "secondary", None)
            self.more_button.clicked.connect(lambda: self.save(keep_open=True))
        self.save_button = self.add_button("고친 내용 저장" if self._editing else "명단에 넣기", "primary", None,
                                           default=True)
        self.save_button.clicked.connect(lambda: self.save())

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(150)
        self._timer.timeout.connect(self.refresh_preview)
        self.set_values(self._record or initial or {})
        first = next(iter(self._inputs.values()), None)
        if first is not None:
            first.setFocus()

    def _make_input(self, column: Column) -> QWidget:
        if column.type == "photo":
            control = _PhotoInput(column, self._photos)
            control.changed.connect(self._changed)
            return control
        if column.type == "date":
            control = make_date_edit()
            control.dateChanged.connect(self._changed)
            return control
        if column.type == "list":
            control = ComboBox()
            control.addItem("")
            control.addItems([str(c) for c in column.options.get("choices", [])])
            control.currentIndexChanged.connect(self._changed)
            return control
        control = QLineEdit()
        control.setFixedHeight(40)
        if column.type == "time":
            control.setPlaceholderText("14:30")
        control.textChanged.connect(self._changed)
        return control

    def _changed(self, *_args) -> None:
        self.banner.setVisible(False)
        self._timer.start()

    def input(self, name: str) -> QWidget:
        return self._inputs[name]

    def set_values(self, values: dict) -> None:
        for name, control in self._inputs.items():
            value = values.get(name)
            if isinstance(control, _PhotoInput):
                control.set_value(value)
            elif isinstance(control, QDateEdit):
                set_date_value(control, display_value(value))
            elif isinstance(control, ComboBox):
                text = display_value(value)
                if text and control.findText(text) < 0:
                    control.addItem(text)
                control.setCurrentIndex(max(0, control.findText(text)))
            else:
                control.setText(display_value(value))
        self.refresh_preview()

    def values(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, control in self._inputs.items():
            if isinstance(control, _PhotoInput):
                result[name] = control.value() or None
            elif isinstance(control, QDateEdit):
                result[name] = date_value(control)
            elif isinstance(control, ComboBox):
                result[name] = control.currentText() or None
            else:
                result[name] = control.text().strip() or None
        return result

    def refresh_preview(self) -> None:
        values = self.values()
        binding: dict[str, Any] = {}
        for name, value in values.items():
            control = self._inputs[name]
            if isinstance(control, _PhotoInput):
                found = self._photos.path({name: value}, name)
                binding[name] = str(found) if found else ""
                text, tone = control.status()
                self._rows[name].set_hint(text, tone)
            else:
                binding[name] = "" if value is None else value
        document = self._workspace.editor.document
        sides = document.active_side_keys()
        if all(is_empty(v) for v in values.values()):
            binding = None      # 아직 아무것도 안 적었으면 «이름» 같은 자리 표시를 보여 준다.
        try:
            image = render_card_image(
                document, sides[0] if sides else "front", binding, store=self._store,
                record_id=self._record["_id"] if self._record else None,
                width_px=round(640 * max(1.0, self.devicePixelRatioF())))
        except Exception:
            image = None
        self.card.set_image(image)

    def save(self, keep_open: bool = False) -> int | None:
        values = self.values()
        if all(is_empty(v) for v in values.values()):
            self.banner.set_text("아무것도 적지 않았습니다. 이름처럼 한 가지는 적어 주세요.")
            self.banner.setVisible(True)
            return None
        try:
            if self._editing:
                record_id = self._record["_id"]
                self._store.update(record_id, values)
            else:
                record_id = self._store.insert(values)
        except DataError as exc:
            self.banner.set_text(str(exc))
            self.banner.setVisible(True)
            return None
        self.saved_ids.append(record_id)
        self.saved.emit(record_id)
        if keep_open:
            self.set_values({})
            self.preview_hint.setText(f"{people(len(self.saved_ids))}을 넣었습니다. 다음 사람을 적으세요.")
            next(iter(self._inputs.values())).setFocus()
        else:
            self.accept()
        return record_id


def _person_name(record: dict | None, store: DataStore | None) -> str:
    if not record:
        return ""
    for key in ("이름", "성명", "name", "Name"):
        if not is_empty(record.get(key)):
            return display_value(record[key])
    if store is not None:
        for column in store.columns():
            if column.type == "text" and not is_empty(record.get(column.name)):
                return display_value(record[column.name])
    return ""


person_name = _person_name


# ---------------------------------------------------------------- 조건으로 찾기

class _ConditionRow(QWidget):
    changed = Signal()
    removeRequested = Signal(object)

    def __init__(self, columns: list[Column], parent: QWidget | None = None):
        super().__init__(parent)
        self._columns = {c.name: c for c in columns}
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(SPACE["sm"])
        self.column_combo = ComboBox([c.name for c in columns])
        self.column_combo.setMinimumWidth(150)
        self.column_combo.setAccessibleName("찾을 항목")
        self.op_combo = ComboBox()
        self.op_combo.setMinimumWidth(170)
        self.op_combo.setAccessibleName("조건")
        self.value_edit = QLineEdit()
        self.value_edit.setFixedHeight(40)
        self.value_edit.setAccessibleName("값")
        self.value_edit2 = QLineEdit()
        self.value_edit2.setFixedHeight(40)
        self.value_edit2.setAccessibleName("끝 값")
        self.tilde = label("~", "muted")
        self.remove_button = IconButton("close", "이 조건 빼기")
        row.addWidget(self.column_combo)
        row.addWidget(self.value_edit, 1)
        row.addWidget(self.tilde)
        row.addWidget(self.value_edit2, 1)
        row.addWidget(self.op_combo)
        row.addWidget(self.remove_button)
        self.column_combo.currentIndexChanged.connect(self._column_changed)
        self.op_combo.currentIndexChanged.connect(self._op_changed)
        self.value_edit.textChanged.connect(self.changed)
        self.value_edit2.textChanged.connect(self.changed)
        self.remove_button.clicked.connect(lambda: self.removeRequested.emit(self))
        self._column_changed()

    def column(self) -> Column | None:
        return self._columns.get(self.column_combo.currentText())

    def _column_changed(self, *_args) -> None:
        column = self.column()
        current = self.op_combo.currentData()
        self.op_combo.blockSignals(True)
        self.op_combo.clear()
        if column is not None:
            for op in operators_for(column.type):
                self.op_combo.addItem(operator_label(op, column.type), op)
            default = "eq" if column.type in ("number", "date", "time", "list") else "contains"
            index = self.op_combo.findData(current if self.op_combo.findData(current) >= 0 else default)
            self.op_combo.setCurrentIndex(max(0, index))
            placeholder = {"date": "2026-03-02", "time": "14:30", "number": "숫자"}.get(column.type, "찾을 글자")
            self.value_edit.setPlaceholderText(placeholder)
            self.value_edit2.setPlaceholderText(placeholder)
        self.op_combo.blockSignals(False)
        self._op_changed()

    def _op_changed(self, *_args) -> None:
        op = self.op_combo.currentData()
        self.value_edit.setVisible(op not in _NO_VALUE_OPS)
        between = op == "between"
        self.value_edit2.setVisible(between)
        self.tilde.setVisible(between)
        self.changed.emit()

    def set_condition(self, condition: Condition) -> None:
        self.column_combo.setCurrentText(condition.column)
        index = self.op_combo.findData(condition.op)
        if index >= 0:
            self.op_combo.setCurrentIndex(index)
        if isinstance(condition.value, (list, tuple)):
            self.value_edit.setText(display_value(condition.value[0]))
            self.value_edit2.setText(display_value(condition.value[1]))
        else:
            self.value_edit.setText(display_value(condition.value))

    def condition(self) -> Condition | None:
        """다 채워지지 않은 줄은 None."""
        column = self.column()
        op = self.op_combo.currentData()
        if column is None or op is None:
            return None
        if op in _NO_VALUE_OPS:
            return Condition(column.name, op)
        first = self.value_edit.text().strip()
        if op == "between":
            second = self.value_edit2.text().strip()
            return Condition(column.name, op, (first, second)) if first and second else None
        return Condition(column.name, op, first) if first else None


class ConditionDialog(Dialog):
    """항목 + 조건 + 값을 여러 줄로. 결과는 result_filter()."""

    def __init__(self, store: DataStore, current: Group | None = None, parent: QWidget | None = None,
                 base_filter: Any = None):
        super().__init__("조건으로 찾기", "조건에 맞는 사람만 명단에 보여 줍니다.", parent, width=720)
        self._store = store
        self._base = base_filter
        self._columns = [c for c in store.columns() if c.type != "photo"] or store.columns()
        self._rows: list[_ConditionRow] = []

        self.mode_control = SegmentedControl([("and", "모두 맞는 사람"), ("or", "하나라도 맞는 사람")], current="and")
        self.mode_control.currentChanged.connect(lambda *_: self._update_count())
        self.body.addWidget(self.mode_control)
        self.rows_box = QVBoxLayout()
        self.rows_box.setSpacing(SPACE["sm"])
        self.body.addLayout(self.rows_box)
        add_row = QHBoxLayout()
        self.add_row_button = QuietButton("조건 더하기", "sm", icon="plus", accent=True)
        self.add_row_button.clicked.connect(lambda: self.add_row())
        add_row.addWidget(self.add_row_button)
        add_row.addStretch(1)
        self.body.addLayout(add_row)
        self.count_label = label("", "secondary", wrap=True)
        self.count_label.setMinimumHeight(24)
        self.body.addWidget(self.count_label)
        self.body.addStretch(1)

        self.clear_button = self.add_button("조건 모두 지우기", "quiet", None, left=True)
        self.clear_button.clicked.connect(self._clear)
        self.add_button("취소", "secondary", "reject")
        self.apply_button = self.add_button("이 조건으로 보기", "primary", "accept", default=True)

        if current is not None and current.items:
            self.mode_control.set_current(str(current.mode).lower(), emit=False)
            for item in current.items:
                if isinstance(item, Condition):
                    self.add_row(item)
        if not self._rows:
            self.add_row()
        self._update_count()

    def rows(self) -> list[_ConditionRow]:
        return list(self._rows)

    def add_row(self, condition: Condition | None = None) -> _ConditionRow:
        row = _ConditionRow(self._columns, self)
        if condition is not None:
            row.set_condition(condition)
        row.changed.connect(self._update_count)
        row.removeRequested.connect(self._remove_row)
        self._rows.append(row)
        self.rows_box.addWidget(row)
        self._update_count()
        return row

    def _remove_row(self, row: _ConditionRow) -> None:
        if row in self._rows:
            self._rows.remove(row)
            self.rows_box.removeWidget(row)
            row.hide()
            row.deleteLater()
        if not self._rows:
            self.add_row()
        self._update_count()

    def _clear(self) -> None:
        for row in list(self._rows):
            self._rows.remove(row)
            self.rows_box.removeWidget(row)
            row.hide()
            row.deleteLater()
        self.add_row()

    def result_filter(self) -> Group | None:
        conditions = [c for c in (row.condition() for row in self._rows) if c is not None]
        if not conditions:
            return None
        return Group(self.mode_control.current(), conditions)

    def _update_count(self) -> None:
        group = self.result_filter()
        tone = ""
        ok = True
        if group is None:
            text = "조건을 적으면 몇 명이 맞는지 바로 알려 줍니다."
        else:
            try:
                combined = group if self._base is None else Group("and", [self._base, group])
                count = self._store.count(combined)
                text = f"{people(count)}이 맞습니다." if count else "맞는 사람이 없습니다. 조건을 바꿔 보세요."
            except DataError as exc:
                text, tone, ok = str(exc), "error", False
        self.count_label.setText(text)
        set_prop(self.count_label, "role", "hint" if tone else "secondary")
        set_prop(self.count_label, "tone", tone)
        if hasattr(self, "apply_button"):
            self.apply_button.setEnabled(ok)


def describe_filter(group: Group | None) -> str:
    """'부서에 "영업"이 들어 있다' 같은 한 줄 설명."""
    if group is None or not group.items:
        return ""
    parts = []
    for item in group.items:
        if not isinstance(item, Condition):
            continue
        label_text = operator_label(item.op, "text")
        if item.op in _NO_VALUE_OPS:
            parts.append(f"{item.column}: {label_text}")
        elif isinstance(item.value, (list, tuple)):
            parts.append(f"{item.column}: {item.value[0]} ~ {item.value[1]}")
        else:
            parts.append(f"{item.column}: {item.value} ({label_text})")
    joiner = " 그리고 " if str(group.mode).lower() == "and" else " 또는 "
    return joiner.join(parts)


# ---------------------------------------------------------------- 항목

class NameDialog(Dialog):
    def __init__(self, title: str, description: str | None, field_label: str, initial: str = "",
                 confirm_text: str = "확인", parent: QWidget | None = None):
        super().__init__(title, description, parent, width=420)
        self.edit = QLineEdit(initial)
        self.edit.setFixedHeight(40)
        self.edit.selectAll()
        self.row = FormRow(field_label, self.edit)
        self.body.addWidget(self.row)
        self.add_button("취소", "secondary", "reject")
        self.confirm_button = self.add_button(confirm_text, "primary", "accept", default=True)
        self.edit.textChanged.connect(lambda text: self.confirm_button.setEnabled(bool(text.strip())))
        self.confirm_button.setEnabled(bool(initial.strip()))
        self.edit.setFocus()

    def text(self) -> str:
        return self.edit.text().strip()


class AddColumnDialog(Dialog):
    def __init__(self, parent: QWidget | None = None, name: str = ""):
        super().__init__("항목 추가", "명단에 새 항목을 만듭니다. 사람마다 값을 따로 적을 수 있습니다.", parent, width=440)
        self.name_edit = QLineEdit(name)
        self.name_edit.setFixedHeight(40)
        self.name_edit.setPlaceholderText("예: 직급")
        self.body.addWidget(FormRow("항목 이름", self.name_edit))
        self.type_combo = ComboBox()
        for key in COLUMN_TYPES:
            self.type_combo.addItem(TYPE_NAMES[key], key)
        self.type_row = FormRow("무엇을 적나요?", self.type_combo, TYPE_HINTS["text"])
        self.body.addWidget(self.type_row)
        self.choices_edit = QLineEdit()
        self.choices_edit.setFixedHeight(40)
        self.choices_edit.setPlaceholderText("사원, 대리, 과장")
        self.choices_row = FormRow("고를 수 있는 값", self.choices_edit, "쉼표로 나눠 적습니다.")
        self.choices_row.setVisible(False)
        self.body.addWidget(self.choices_row)
        self.add_button("취소", "secondary", "reject")
        self.confirm_button = self.add_button("항목 만들기", "primary", "accept", default=True)
        self.type_combo.currentIndexChanged.connect(self._type_changed)
        self.name_edit.textChanged.connect(lambda text: self.confirm_button.setEnabled(bool(text.strip())))
        self.confirm_button.setEnabled(bool(name.strip()))
        self.name_edit.setFocus()

    def _type_changed(self, *_args) -> None:
        key = self.type_combo.currentData()
        self.type_row.set_hint(TYPE_HINTS[key])
        self.choices_row.setVisible(key == "list")

    def name(self) -> str:
        return self.name_edit.text().strip()

    def column_type(self) -> str:
        return self.type_combo.currentData()

    def options(self) -> dict:
        if self.column_type() != "list":
            return {}
        return {"choices": [c.strip() for c in self.choices_edit.text().split(",") if c.strip()]}


class MatchFieldDialog(Dialog):
    """카드의 칸에 맞는 항목이 명단에 없을 때: 있는 항목을 그 칸으로 쓰거나 새로 만든다."""

    NEW = "\x00new"

    def __init__(self, field_name: str, candidates: list[str], parent: QWidget | None = None):
        super().__init__(
            f"카드의 '{field_name}' 칸 맞추기",
            f"카드에는 '{field_name}' 칸이 있는데 명단에는 그런 항목이 없습니다. 어느 항목을 넣을지 골라 주세요.",
            parent, width=460)
        self.field_name = field_name
        self.combo = ComboBox()
        for name in candidates:
            self.combo.addItem(f"명단의 '{name}' 항목을 쓴다", name)
        self.combo.addItem(f"비어 있는 '{field_name}' 항목을 새로 만든다", self.NEW)
        self.row = FormRow(f"'{field_name}' 칸에 넣을 것", self.combo)
        self.body.addWidget(self.row)
        self.combo.currentIndexChanged.connect(self._sync_hint)
        self.add_button("취소", "secondary", "reject")
        self.add_button("이렇게 맞추기", "primary", "accept", default=True)
        self._sync_hint()

    def _sync_hint(self, *_args) -> None:
        choice = self.combo.currentData()
        if choice == self.NEW:
            self.row.set_hint("새 항목은 비어 있습니다. 명단에서 한 명씩 적어 넣으세요.")
        else:
            self.row.set_hint(f"명단의 '{choice}' 항목 이름이 '{self.field_name}'{'으로' if _final_not_l(self.field_name) else '로'} 바뀝니다. 값은 그대로입니다.")

    def choice(self) -> str | None:
        """고른 항목 이름. 새로 만들기면 None."""
        data = self.combo.currentData()
        return None if data == self.NEW else data


def _final_not_l(word: str) -> bool:
    for ch in reversed(word.strip()):
        if "가" <= ch <= "힣":
            final = (ord(ch) - 0xAC00) % 28
            return final not in (0, 8)
        break
    return False


# ---------------------------------------------------------------- 내보내기

class ExportDialog(Dialog):
    def __init__(self, total: int, visible: int, checked: int, filtered: bool, parent: QWidget | None = None):
        super().__init__("명단 내보내기", "엑셀에서 열 수 있는 파일로 저장합니다.", parent, width=460)
        self.format_control = SegmentedControl([("xlsx", "엑셀 파일"), ("csv", "CSV 파일")], current="xlsx")
        self.body.addWidget(FormRow("파일 종류", self.format_control))
        scope_box = QVBoxLayout()
        scope_box.setSpacing(2)
        self.scope_buttons: dict[str, QRadioButton] = {}
        options = [("all", f"모든 사람 {people(total)}")]
        if filtered:
            options.append(("view", f"지금 보이는 {people(visible)}"))
        if checked:
            options.append(("checked", f"선택한 {people(checked)}"))
        for key, text in options:
            button = QRadioButton(text)
            self.scope_buttons[key] = button
            scope_box.addWidget(button)
        default = "view" if filtered else "all"
        self.scope_buttons[default].setChecked(True)
        holder = QWidget()
        holder.setLayout(scope_box)
        scope_box.setContentsMargins(0, 0, 0, 0)
        self.body.addWidget(FormRow("누구를 내보낼까요?", holder))
        self.status_check = QCheckBox("인쇄 기록도 함께 넣기 (인쇄했는지, 몇 장, 언제)")
        self.body.addWidget(self.status_check)
        self.add_button("취소", "secondary", "reject")
        self.add_button("저장할 곳 고르기", "primary", "accept", default=True)

    def file_format(self) -> str:
        return self.format_control.current()

    def scope(self) -> str:
        return next(key for key, button in self.scope_buttons.items() if button.isChecked())

    def include_status(self) -> bool:
        return self.status_check.isChecked()
