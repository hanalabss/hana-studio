"""다눌처럼 고를 수 있는 길이 단위: 값은 mm 그대로, 보이는 숫자만 바뀐다."""

import pytest

from ui.units import LengthUnits, format_length, mm_to_unit, unit_to_mm
from ui.widgets import MmField


@pytest.fixture
def units(qapp):
    return LengthUnits.instance()


def test_conversions():
    assert mm_to_unit(25.4, "inch") == pytest.approx(1.0)
    assert unit_to_mm(72, "pt") == pytest.approx(25.4)
    assert unit_to_mm(300, "px") == pytest.approx(25.4)
    assert format_length(85.6, "cm") == "8.56 cm"


def test_field_shows_chosen_unit_and_keeps_mm_value(units):
    field = MmField(25.4, 0, 200)
    assert field.text() == "25.4 mm"
    units.set_key("inch")
    assert field.text() == "1.000 in"
    assert field.value() == pytest.approx(25.4)
    units.set_key("cm")
    assert field.text() == "2.54 cm"


def test_typing_in_chosen_unit_emits_mm(units):
    units.set_key("cm")
    field = MmField(0, 0, 200)
    got = []
    field.valueChanged.connect(got.append)
    field.line_edit().setText("3")
    field.line_edit().textEdited.emit("3")
    assert got and got[-1] == pytest.approx(30.0)
    field.line_edit().editingFinished.emit()
    assert field.text() == "3.00 cm"


def test_print_offsets_stay_in_mm(units):
    units.set_key("inch")
    field = MmField(1.5, -10, 10, 0.1, 1, follow_units=False)
    assert field.text() == "1.5 mm"


def test_choice_is_remembered(tmp_path, qapp):
    from PySide6.QtCore import QSettings

    path = str(tmp_path / "remember.ini")
    LengthUnits(QSettings(path, QSettings.IniFormat)).set_key("pt")
    assert LengthUnits(QSettings(path, QSettings.IniFormat)).key == "pt"
