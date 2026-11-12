"""앱 틀과 화면 약속."""

from .app_shell import AppShell, fit_geometry
from .legacy import LegacyHostPage, default_legacy_factory
from .nav_rail import NavRail, RailButton
from .page import BasePage, FooterAction, Page
from .pages import (
    PAGE_DESIGN,
    PAGE_HISTORY,
    PAGE_LIST,
    PAGE_PRINT,
    PAGE_PRINTER,
    PAGE_QUICK,
    PAGE_SETTINGS,
    PlaceholderPage,
    SettingsPage,
    install_default_pages,
)
from .step_guide import STEP_STATES, GuideStrip, StepPill
from .top_bar import APP_NAME, PrinterPopover, TopBar

__all__ = [
    "APP_NAME", "AppShell", "BasePage", "FooterAction", "GuideStrip", "LegacyHostPage", "NavRail",
    "PAGE_DESIGN", "PAGE_HISTORY", "PAGE_LIST", "PAGE_PRINT", "PAGE_PRINTER", "PAGE_QUICK",
    "PAGE_SETTINGS", "Page", "PlaceholderPage", "PrinterPopover", "RailButton", "STEP_STATES",
    "SettingsPage", "StepPill", "TopBar", "default_legacy_factory", "fit_geometry", "install_default_pages",
]
