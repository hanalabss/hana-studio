"""화면에 보이는 길이 단위(mm·cm·inch·pt·px). 문서와 계산은 언제나 mm이고, 보여 줄 때만 바꾼다.

다눌처럼 눈금자와 위치·크기 칸의 단위를 고를 수 있게 한다. 고른 단위는 앱 설정에 남는다.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QObject, Signal

KEY_UNIT = "view/length_unit"
PRINT_DPI = 300.0


@dataclass(frozen=True)
class Unit:
    key: str
    label: str          # 메뉴에 보이는 이름
    suffix: str         # 숫자 뒤에 붙는 글자
    mm: float           # 1단위가 몇 mm인가
    decimals: int       # 보여 줄 소수 자리


UNITS: tuple[Unit, ...] = (
    Unit("mm", "밀리미터 (mm)", "mm", 1.0, 1),
    Unit("cm", "센티미터 (cm)", "cm", 10.0, 2),
    Unit("inch", "인치 (inch)", "in", 25.4, 3),
    Unit("pt", "포인트 (pt)", "pt", 25.4 / 72.0, 1),
    Unit("px", "픽셀 (300 dpi 인쇄 기준)", "px", 25.4 / PRINT_DPI, 0),
)
_BY_KEY = {unit.key: unit for unit in UNITS}
DEFAULT_UNIT = "mm"


def unit_of(key: str | None) -> Unit:
    return _BY_KEY.get(key or DEFAULT_UNIT, _BY_KEY[DEFAULT_UNIT])


def mm_to_unit(mm: float, key: str) -> float:
    return float(mm) / unit_of(key).mm


def unit_to_mm(value: float, key: str) -> float:
    return float(value) * unit_of(key).mm


def format_length(mm: float, key: str) -> str:
    unit = unit_of(key)
    return f"{mm_to_unit(mm, key):.{unit.decimals}f} {unit.suffix}"


class LengthUnits(QObject):
    """지금 고른 단위. 바뀌면 changed(단위 키)를 보낸다. 앱 전체에서 하나만 쓴다."""

    changed = Signal(str)
    _instance: "LengthUnits | None" = None

    def __init__(self, settings=None, parent: QObject | None = None) -> None:
        super().__init__(parent)
        if settings is None:
            from ui.theme import app_settings
            settings = app_settings()
        self._settings = settings
        stored = str(self._settings.value(KEY_UNIT, DEFAULT_UNIT) or DEFAULT_UNIT)
        self._key = stored if stored in _BY_KEY else DEFAULT_UNIT

    @classmethod
    def instance(cls) -> "LengthUnits":
        if cls._instance is None:
            cls._instance = LengthUnits()
        return cls._instance

    @classmethod
    def set_instance(cls, units: "LengthUnits | None") -> None:
        cls._instance = units

    @property
    def key(self) -> str:
        return self._key

    @property
    def unit(self) -> Unit:
        return unit_of(self._key)

    def set_key(self, key: str) -> None:
        if key not in _BY_KEY or key == self._key:
            return
        self._key = key
        self._settings.setValue(KEY_UNIT, key)
        self._settings.sync()
        self.changed.emit(key)
