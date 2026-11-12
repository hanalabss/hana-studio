"""앱 설정 저장소. 테스트와 미리 보기는 HANA_STUDIO_SETTINGS_FILE로 다른 파일을 쓴다."""

from __future__ import annotations

import os

from PySide6.QtCore import QSettings

ORGANIZATION = "HanaLabs"
APPLICATION = "HanaStudio"

KEY_THEME_MODE = "ui/theme_mode"
KEY_GUIDE_VISIBLE = "ui/guide_visible"
KEY_GEOMETRY = "ui/shell_geometry"
KEY_MAXIMIZED = "ui/shell_maximized"


def app_settings() -> QSettings:
    path = os.environ.get("HANA_STUDIO_SETTINGS_FILE")
    if path:
        return QSettings(path, QSettings.IniFormat)
    return QSettings(ORGANIZATION, APPLICATION)


def read_bool(settings: QSettings, key: str, default: bool) -> bool:
    value = settings.value(key, default)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)
