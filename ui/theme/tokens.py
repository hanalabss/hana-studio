"""디자인 토큰. UI 코드에서 색 값을 직접 쓰지 않고 여기 역할 이름으로만 가져다 쓴다."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

LIGHT_COLORS: dict[str, str] = {
    # 바탕
    "ground": "#F6F7F9",
    "surface": "#FFFFFF",
    "surface_raised": "#FFFFFF",
    "surface_sunken": "#F1F3F6",
    "surface_hover": "#EEF1F5",
    "surface_pressed": "#E3E7ED",
    # 선
    "line": "#DDE1E7",
    "line_soft": "#EDF0F3",
    "control_line": "#B9C0CB",
    "indicator_line": "#8792A2",
    # 글자
    "text": "#1B2430",
    "text_secondary": "#4A5568",
    "text_muted": "#5B6676",
    "text_disabled": "#9AA3B0",
    # 강조
    "accent": "#2456E6",
    "accent_hover": "#1D49C9",
    "accent_pressed": "#183DA8",
    "accent_soft": "#E8EEFD",
    "accent_text": "#1D3FA8",
    "text_on_accent": "#FFFFFF",
    "focus": "#2456E6",
    "cutline": "#FF4D8D",
    # 주 버튼
    "primary_bg": "#2456E6",
    "primary_hover": "#1D49C9",
    "primary_pressed": "#183DA8",
    "primary_text": "#FFFFFF",
    "primary_disabled_bg": "#E3E7ED",
    # 선택된 조각(세그먼트, 필터 칩)
    "strong_bg": "#1B2430",
    "strong_text": "#FFFFFF",
    # 왼쪽 메뉴
    "nav_text": "#4A5568",
    "nav_hover_bg": "#F1F3F6",
    "nav_selected_bg": "#E8EEFD",
    "nav_selected_text": "#2456E6",
    "nav_selected_icon": "#2456E6",
    # 3단계 안내
    "step_current_bg": "#2456E6",
    "step_current_text": "#FFFFFF",
    "step_current_mark_bg": "#FFFFFF",
    "step_current_mark_text": "#2456E6",
    "step_done_bg": "#E8EEFD",
    "step_done_text": "#1D3FA8",
    "step_done_mark_bg": "#2456E6",
    "step_done_mark_text": "#FFFFFF",
    # 사람마다 바뀌는 칸
    "field_bg": "#FFE9A8",
    "field_hover": "#FFDF85",
    "field_text": "#4A3800",
    "field_line": "#E0A800",
    # 상태
    "ok_bg": "#EAF6EE",
    "ok_text": "#1D6B3A",
    "ok_mark": "#2E9E57",
    "warn_bg": "#FFF3DC",
    "warn_text": "#8A4B00",
    "warn_line": "#F0C36B",
    "warn_mark": "#D98A00",
    "danger_bg": "#FDECEC",
    "danger_text": "#A12020",
    "danger_line": "#F0B4B4",
    "danger": "#C62828",
    "danger_hover": "#A91F1F",
    "danger_on": "#FFFFFF",
    "info_bg": "#E8EEFD",
    "info_text": "#1D3FA8",
    "info_line": "#BFD0FA",
    "busy_mark": "#2456E6",
    "idle_bg": "#F1F3F6",
    "idle_text": "#4A5568",
    "idle_mark": "#9AA3B0",
    # 리본 패널
    "ribbon_y": "#F5C400",
    "ribbon_m": "#E0338A",
    "ribbon_c": "#1AA7E0",
    "ribbon_k": "#1B2430",
    "ribbon_w": "#FFFFFF",
    "ribbon_track": "#E3E7ED",
    "ribbon_edge": "#B9C0CB",
    # 기타
    "tooltip_bg": "#1B2430",
    "tooltip_text": "#FFFFFF",
    "scrollbar": "#C3CAD4",
    "scrollbar_hover": "#9AA3B0",
    "toast_bg": "#1B2430",
    "toast_text": "#FFFFFF",
    "toast_action": "#A9C0FF",
    "swatch_ring": "#2456E6",
}

DARK_COLORS: dict[str, str] = {
    "ground": "#15171A",
    "surface": "#1E2126",
    "surface_raised": "#272B31",
    "surface_sunken": "#1A1C20",
    "surface_hover": "#30353C",
    "surface_pressed": "#3A4048",
    "line": "#30353C",
    "line_soft": "#272B31",
    "control_line": "#4A515B",
    "indicator_line": "#7A828E",
    "text": "#E8EAED",
    "text_secondary": "#C9CED6",
    "text_muted": "#B4BAC3",
    "text_disabled": "#6B727D",
    "accent": "#FF4D8D",
    "accent_hover": "#FF6B9F",
    "accent_pressed": "#E83E7C",
    "accent_soft": "#35222F",
    "accent_text": "#FF8DB5",
    "text_on_accent": "#15171A",
    "focus": "#FF4D8D",
    "cutline": "#FF4D8D",
    "primary_bg": "#F2F3F5",
    "primary_hover": "#FFFFFF",
    "primary_pressed": "#D5D8DD",
    "primary_text": "#15171A",
    "primary_disabled_bg": "#30353C",
    "strong_bg": "#F2F3F5",
    "strong_text": "#15171A",
    "nav_text": "#B4BAC3",
    "nav_hover_bg": "#272B31",
    "nav_selected_bg": "#30353C",
    "nav_selected_text": "#FFFFFF",
    "nav_selected_icon": "#FF4D8D",
    "step_current_bg": "#F2F3F5",
    "step_current_text": "#15171A",
    "step_current_mark_bg": "#15171A",
    "step_current_mark_text": "#F2F3F5",
    "step_done_bg": "#272B31",
    "step_done_text": "#E8EAED",
    "step_done_mark_bg": "#FF4D8D",
    "step_done_mark_text": "#15171A",
    "field_bg": "#F2D272",
    "field_hover": "#FFE08A",
    "field_text": "#2B2100",
    "field_line": "#B88A00",
    "ok_bg": "#1C3327",
    "ok_text": "#7BE0A8",
    "ok_mark": "#4CD08A",
    "warn_bg": "#3A2E12",
    "warn_text": "#FFC857",
    "warn_line": "#6B5420",
    "warn_mark": "#FFC857",
    "danger_bg": "#3D1F22",
    "danger_text": "#FF9A9A",
    "danger_line": "#6E3136",
    "danger": "#FF6B6B",
    "danger_hover": "#FF8585",
    "danger_on": "#15171A",
    "info_bg": "#1F2A3D",
    "info_text": "#9DBBFF",
    "info_line": "#34466B",
    "busy_mark": "#FF4D8D",
    "idle_bg": "#272B31",
    "idle_text": "#B4BAC3",
    "idle_mark": "#6B727D",
    "ribbon_y": "#F5C400",
    "ribbon_m": "#FF4D8D",
    "ribbon_c": "#35B6F0",
    "ribbon_k": "#0B0C0E",
    "ribbon_w": "#FDFDFB",
    "ribbon_track": "#3A4048",
    "ribbon_edge": "#5A616B",
    "tooltip_bg": "#F2F3F5",
    "tooltip_text": "#15171A",
    "scrollbar": "#3A4048",
    "scrollbar_hover": "#5A616B",
    "toast_bg": "#F2F3F5",
    "toast_text": "#15171A",
    "toast_action": "#B0195A",
    "swatch_ring": "#FF4D8D",
}

COLOR_ROLES: tuple[str, ...] = tuple(LIGHT_COLORS)

# (글자 역할, 바탕 역할). 두 테마 모두 4.5:1 이상이어야 한다.
TEXT_PAIRS: tuple[tuple[str, str], ...] = (
    ("text", "ground"),
    ("text", "surface"),
    ("text", "surface_raised"),
    ("text", "surface_sunken"),
    ("text", "surface_hover"),
    ("text_secondary", "ground"),
    ("text_secondary", "surface"),
    ("text_secondary", "surface_sunken"),
    ("text_muted", "ground"),
    ("text_muted", "surface"),
    ("text_muted", "surface_sunken"),
    ("text_on_accent", "accent"),
    ("accent_text", "accent_soft"),
    ("accent_text", "surface"),
    ("primary_text", "primary_bg"),
    ("primary_text", "primary_hover"),
    ("strong_text", "strong_bg"),
    ("nav_text", "surface"),
    ("nav_selected_text", "nav_selected_bg"),
    ("field_text", "field_bg"),
    ("ok_text", "ok_bg"),
    ("warn_text", "warn_bg"),
    ("danger_text", "danger_bg"),
    ("danger_on", "danger"),
    ("info_text", "info_bg"),
    ("idle_text", "idle_bg"),
    ("tooltip_text", "tooltip_bg"),
    ("toast_text", "toast_bg"),
    ("toast_action", "toast_bg"),
    ("step_current_text", "step_current_bg"),
    ("step_current_mark_text", "step_current_mark_bg"),
    ("step_done_text", "step_done_bg"),
    ("step_done_mark_text", "step_done_mark_bg"),
)

# 포커스 링처럼 글자가 아닌 표시는 3:1 이상.
MARK_PAIRS: tuple[tuple[str, str], ...] = (
    ("focus", "surface"),
    ("focus", "ground"),
    ("indicator_line", "surface"),
    ("nav_selected_icon", "nav_selected_bg"),
)

FONT_SIZES: Mapping[str, int] = MappingProxyType({
    "caption": 12,
    "small": 13,
    "body": 14,
    "title": 16,
    "heading": 20,
    "display": 26,
})

SPACE: Mapping[str, int] = MappingProxyType({
    "xs": 4,
    "sm": 8,
    "md": 12,
    "lg": 16,
    "xl": 24,
    "xxl": 32,
})

RADIUS: Mapping[str, int] = MappingProxyType({
    "sm": 6,
    "md": 8,
    "lg": 12,
})

SIZES: Mapping[str, int] = MappingProxyType({
    "control_sm": 32,
    "control_md": 40,
    "control_lg": 48,
    "topbar": 52,
    "rail": 84,
    "rail_item": 60,
    "guide": 64,
    "footer": 72,
    "chip": 32,
    "pill": 32,
    "icon": 20,
})

# 카드에 자주 쓰는 색. UI 장식이 아니라 사용자가 고르는 내용물이다.
SWATCH_PRESETS: tuple[tuple[str, str], ...] = (
    ("검정", "#16181C"),
    ("흰색", "#FFFFFF"),
    ("남색", "#14315C"),
    ("파랑", "#2456E6"),
    ("초록", "#0E6B5C"),
    ("빨강", "#C62828"),
)


@dataclass(frozen=True)
class Theme:
    name: str
    colors: Mapping[str, str]

    @property
    def is_dark(self) -> bool:
        return self.name == "dark"

    def hex(self, role: str) -> str:
        return self.colors[role]

    def color(self, role: str):
        from PySide6.QtGui import QColor

        return QColor(self.colors[role])


LIGHT = Theme("light", MappingProxyType(LIGHT_COLORS))
DARK = Theme("dark", MappingProxyType(DARK_COLORS))
THEMES: Mapping[str, Theme] = MappingProxyType({"light": LIGHT, "dark": DARK})


def _channel(value: int) -> float:
    c = value / 255.0
    return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4


def relative_luminance(hex_color: str) -> float:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return 0.2126 * _channel(r) + 0.7152 * _channel(g) + 0.0722 * _channel(b)


def contrast_ratio(foreground: str, background: str) -> float:
    a, b = relative_luminance(foreground), relative_luminance(background)
    hi, lo = max(a, b), min(a, b)
    return (hi + 0.05) / (lo + 0.05)
