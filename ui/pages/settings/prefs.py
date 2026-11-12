"""인쇄 기본값과 보관 기간. ui.theme.app_settings()에 저장한다."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QSettings, Signal

from design import presets
from services.render_job import PrintOptions
from ui.theme import app_settings, read_bool

log = logging.getLogger(__name__)

KEY_COPIES = "print/copies"
KEY_STOP_ON_ERROR = "print/stop_on_error"
KEY_OFFSET_X = "print/offset_x_mm"
KEY_OFFSET_Y = "print/offset_y_mm"
KEY_CARD_PRESET = "design/default_card"
KEY_RETENTION = "history/retention_days"
KEY_MIGRATED = "print/legacy_migrated"

OFFSET_LIMIT_MM = 10.0
COPIES_MAX = 99
RETENTION_CHOICES: tuple[tuple[int, str], ...] = (
    (30, "30일"), (90, "90일"), (365, "1년"), (0, "지우지 않고 계속 둠"),
)
DEFAULT_RETENTION = 365

# 예전 앱이 config.json의 printer 묶음에 쓸 수 있었던 위치 조정 이름들.
_LEGACY_OFFSET_KEYS = (
    ("offset_x_mm", "offset_y_mm"), ("offset_x", "offset_y"),
    ("adjusted_x", "adjusted_y"), ("position_x", "position_y"),
)


def _number(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def legacy_config_path() -> Path:
    override = os.environ.get("HANA_STUDIO_LEGACY_CONFIG")
    return Path(override) if override else Path.cwd() / "config.json"


def read_legacy_printer(path: Path | None = None) -> dict[str, Any]:
    """예전 config.json의 printer 묶음. 없거나 읽지 못하면 빈 dict."""
    try:
        with open(path or legacy_config_path(), "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return {}
    printer = data.get("printer") if isinstance(data, dict) else None
    return dict(printer) if isinstance(printer, dict) else {}


def legacy_values(printer: dict[str, Any]) -> dict[str, Any]:
    """예전 설정에서 새 설정으로 옮길 값."""
    values: dict[str, Any] = {}
    for key_x, key_y in _LEGACY_OFFSET_KEYS:
        if key_x in printer or key_y in printer:
            values[KEY_OFFSET_X] = _number(printer.get(key_x), 0.0)
            values[KEY_OFFSET_Y] = _number(printer.get(key_y), 0.0)
            break
    width = _number(printer.get("design_width_mm"), 0.0)
    height = _number(printer.get("design_height_mm"), 0.0)
    bleed = _number(printer.get("bleed_mm"), 0.0)
    if width and height:
        long_edge, short_edge = max(width, height), min(width, height)
        for preset in presets.PRESETS:
            if (abs(preset.width_mm - long_edge) < 0.05 and abs(preset.height_mm - short_edge) < 0.05
                    and abs(preset.bleed_mm - (bleed if printer.get("bleed_crop_enabled", True) else 0.0)) < 0.05):
                values[KEY_CARD_PRESET] = preset.key
                break
    return values


class Prefs(QObject):
    """설정 화면과 인쇄 화면이 함께 보는 값. 바뀌면 changed를 보낸다."""

    changed = Signal()

    _instance: "Prefs | None" = None

    def __init__(self, settings: QSettings | None = None, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._settings = settings if settings is not None else app_settings()
        self.migrated: dict[str, Any] = {}
        self._migrate()

    @classmethod
    def instance(cls) -> "Prefs":
        if cls._instance is None:
            cls._instance = Prefs()
        return cls._instance

    @classmethod
    def set_instance(cls, prefs: "Prefs | None") -> None:
        cls._instance = prefs

    @property
    def settings(self) -> QSettings:
        return self._settings

    def _migrate(self) -> None:
        if read_bool(self._settings, KEY_MIGRATED, False):
            return
        values = legacy_values(read_legacy_printer())
        for key, value in values.items():
            if not self._settings.contains(key):
                self._settings.setValue(key, value)
                self.migrated[key] = value
        self._settings.setValue(KEY_MIGRATED, True)
        self._settings.sync()

    def _set(self, key: str, value: Any) -> None:
        if self._settings.value(key) == value:
            return
        self._settings.setValue(key, value)
        self._settings.sync()
        self.changed.emit()

    # ---- 인쇄 기본값 ----
    def copies(self) -> int:
        return max(1, min(COPIES_MAX, int(_number(self._settings.value(KEY_COPIES, 1), 1))))

    def set_copies(self, copies: int) -> None:
        self._set(KEY_COPIES, max(1, min(COPIES_MAX, int(copies))))

    def stop_on_error(self) -> bool:
        return read_bool(self._settings, KEY_STOP_ON_ERROR, True)

    def set_stop_on_error(self, stop: bool) -> None:
        self._set(KEY_STOP_ON_ERROR, bool(stop))

    def offsets(self) -> tuple[float, float]:
        def clamp(key: str) -> float:
            value = _number(self._settings.value(key, 0.0), 0.0)
            return round(max(-OFFSET_LIMIT_MM, min(OFFSET_LIMIT_MM, value)), 2)

        return clamp(KEY_OFFSET_X), clamp(KEY_OFFSET_Y)

    def set_offsets(self, x_mm: float, y_mm: float) -> None:
        before = self.offsets()
        after = (round(float(x_mm), 2), round(float(y_mm), 2))
        if before == after:
            return
        self._settings.setValue(KEY_OFFSET_X, after[0])
        self._settings.setValue(KEY_OFFSET_Y, after[1])
        self._settings.sync()
        self.changed.emit()

    def print_options(self) -> PrintOptions:
        x, y = self.offsets()
        return PrintOptions(copies=self.copies(), stop_on_error=self.stop_on_error(),
                            offset_x_mm=x, offset_y_mm=y)

    # ---- 새 카드 크기 ----
    def card_preset_key(self) -> str:
        key = str(self._settings.value(KEY_CARD_PRESET, presets.CR80.key))
        return key if any(p.key == key for p in presets.PRESETS) else presets.CR80.key

    def card_preset(self) -> presets.CardPreset:
        return presets.get_preset(self.card_preset_key())

    def set_card_preset_key(self, key: str) -> None:
        presets.get_preset(key)
        self._set(KEY_CARD_PRESET, key)

    # ---- 기록 보관 ----
    def retention_days(self) -> int:
        days = int(_number(self._settings.value(KEY_RETENTION, DEFAULT_RETENTION), DEFAULT_RETENTION))
        return days if any(days == choice for choice, _ in RETENTION_CHOICES) else DEFAULT_RETENTION

    def set_retention_days(self, days: int) -> None:
        self._set(KEY_RETENTION, int(days))
