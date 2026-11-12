"""설정 화면과 인쇄 기본값."""
from .prefs import Prefs

__all__ = ["Prefs", "SettingsPage"]


def __getattr__(name: str):
    if name == "SettingsPage":
        from .page import SettingsPage

        return SettingsPage
    raise AttributeError(name)
