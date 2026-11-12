"""테마 적용, 실행 중 전환, 선택 저장."""

from __future__ import annotations

from PySide6.QtCore import QCoreApplication, QObject, QSettings, Qt, Signal
from PySide6.QtGui import QGuiApplication, QPalette
from PySide6.QtWidgets import QApplication

from .assets import ensure_assets
from .fonts import FontSpec, application_font, resolve_font
from .qss import build_stylesheet
from .settings import KEY_THEME_MODE, app_settings
from .tokens import THEMES, Theme

MODES: tuple[str, ...] = ("system", "light", "dark")


class ThemeManager(QObject):
    """themeChanged(str)는 실제로 적용된 테마 이름("light" 또는 "dark")을 보낸다."""

    themeChanged = Signal(str)
    modeChanged = Signal(str)

    _instance: "ThemeManager | None" = None

    def __init__(self, settings: QSettings | None = None, parent: QObject | None = None):
        super().__init__(parent)
        self._settings = settings if settings is not None else app_settings()
        stored = str(self._settings.value(KEY_THEME_MODE, "system"))
        self._mode = stored if stored in MODES else "system"
        self._app: QApplication | None = None
        self._font: FontSpec | None = None
        self._theme: Theme = THEMES[self._resolve(self._mode)]
        self._hints_connected = False

    # ---- 공유 인스턴스 ----
    @classmethod
    def instance(cls) -> "ThemeManager":
        if cls._instance is None:
            cls._instance = ThemeManager()
        return cls._instance

    @classmethod
    def set_instance(cls, manager: "ThemeManager | None") -> None:
        cls._instance = manager

    # ---- 상태 ----
    @property
    def mode(self) -> str:
        return self._mode

    @property
    def theme(self) -> Theme:
        return self._theme

    @property
    def theme_name(self) -> str:
        return self._theme.name

    @property
    def font_spec(self) -> FontSpec:
        if self._font is None:
            self._font = resolve_font()
        return self._font

    @property
    def settings(self) -> QSettings:
        return self._settings

    @staticmethod
    def system_theme_name() -> str:
        app = QGuiApplication.instance()
        if app is None:
            return "light"
        try:
            scheme = app.styleHints().colorScheme()
        except AttributeError:
            return "light"
        return "dark" if scheme == Qt.ColorScheme.Dark else "light"

    def _resolve(self, mode: str) -> str:
        return self.system_theme_name() if mode == "system" else mode

    # ---- 적용 ----
    def apply(self, app: QApplication | None = None) -> None:
        app = app or QApplication.instance()
        if app is None:
            raise RuntimeError("QApplication을 먼저 만들어야 합니다.")
        self._app = app
        if not self._hints_connected:
            try:
                app.styleHints().colorSchemeChanged.connect(self._on_system_scheme_changed)
            except AttributeError:
                pass
            self._hints_connected = True
        # 스타일시트를 쓰는 중에도 부모의 팔레트가 자식에게 내려가게 한다.
        # 기존 화면에 밝은 팔레트를 따로 줄 때 필요하다.
        QCoreApplication.setAttribute(Qt.AA_UseStyleSheetPropagationInWidgetStyles, True)
        app.setStyle("Fusion")
        app.setFont(application_font(self.font_spec))
        self._theme = THEMES[self._resolve(self._mode)]
        self._push()

    def _push(self) -> None:
        if self._app is None:
            return
        self._app.setPalette(self.palette())
        self._app.setStyleSheet(self.stylesheet())

    def stylesheet(self) -> str:
        return build_stylesheet(self._theme, self.font_spec, ensure_assets(self._theme))

    def palette(self, theme: Theme | None = None) -> QPalette:
        t = theme or self._theme
        pal = QPalette()
        roles = {
            QPalette.Window: "ground",
            QPalette.WindowText: "text",
            QPalette.Base: "surface_raised",
            QPalette.AlternateBase: "surface_sunken",
            QPalette.Text: "text",
            QPalette.Button: "surface_raised",
            QPalette.ButtonText: "text",
            QPalette.BrightText: "text_on_accent",
            QPalette.Highlight: "accent",
            QPalette.HighlightedText: "text_on_accent",
            QPalette.ToolTipBase: "tooltip_bg",
            QPalette.ToolTipText: "tooltip_text",
            QPalette.PlaceholderText: "text_muted",
            QPalette.Link: "accent_text",
            QPalette.Mid: "line",
            QPalette.Dark: "control_line",
            QPalette.Light: "surface",
            QPalette.Midlight: "surface_hover",
            QPalette.Shadow: "line",
        }
        for role, token in roles.items():
            pal.setColor(role, t.color(token))
        for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
            pal.setColor(QPalette.Disabled, role, t.color("text_disabled"))
        return pal

    # ---- 전환 ----
    def set_mode(self, mode: str, persist: bool = True) -> None:
        if mode not in MODES:
            raise ValueError(f"알 수 없는 테마: {mode}")
        mode_changed = mode != self._mode
        self._mode = mode
        if persist:
            self._settings.setValue(KEY_THEME_MODE, mode)
            self._settings.sync()
        self._set_theme(self._resolve(mode))
        if mode_changed:
            self.modeChanged.emit(mode)

    def toggle(self) -> None:
        self.set_mode("light" if self._theme.is_dark else "dark")

    def _set_theme(self, name: str) -> None:
        if name == self._theme.name:
            return
        self._theme = THEMES[name]
        self._push()
        self.themeChanged.emit(name)

    def _on_system_scheme_changed(self, *_args) -> None:
        if self._mode == "system":
            self._set_theme(self.system_theme_name())


def current_theme() -> Theme:
    return ThemeManager.instance().theme
