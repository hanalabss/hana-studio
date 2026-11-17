"""예전 창(라이선스 인증·관리자 패널·모델 받기)에 박혀 있던 밝은 색 스타일시트를 지금 테마 색으로 바꾼다.

같은 색이라도 쓰인 자리(배경·글자·선)에 따라 다른 테마 색이 맞다. 예: #FFFFFF는 배경이면 surface,
글자면 파란 단추 위 글자(primary_text). 그래서 'background-color: #FFFFFF'처럼 속성과 함께 바꾼다.
표에 없는 색은 그대로 둔다.
"""

from __future__ import annotations

import re

_BACKGROUND = {
    "#F9FAFB": "ground", "#FFFFFF": "surface", "#F3F4F6": "surface_hover", "#EFF6FF": "accent_soft",
    "#3B82F6": "primary_bg", "#4A90E2": "primary_bg", "#2563EB": "primary_hover", "#1D4ED8": "primary_pressed",
    "#93C5FD": "primary_disabled_bg", "#FEF2F2": "danger_bg", "#ECFDF5": "ok_bg", "#EF4444": "danger",
    "#10B981": "ok_mark", "#DDE1E7": "line",
}
_TEXT = {
    "#111827": "text", "#1F2937": "text", "#1B2430": "text", "#374151": "text", "#6B7280": "text_muted",
    "#5B6676": "text_muted", "#9CA3AF": "text_disabled", "#FFFFFF": "primary_text", "#1E40AF": "accent_text",
    "#991B1B": "danger_text", "#065F46": "ok_text", "#EF4444": "danger_text", "#10B981": "ok_text",
}
_LINE = {
    "#E5E7EB": "line", "#D1D5DB": "control_line", "#F3F4F6": "line_soft", "#3B82F6": "focus", "#DDE1E7": "line",
    "#EF4444": "danger_line", "#FFFFFF": "surface",
}
# 속성 이름을 정해 둔다. 'QPushButton:hover'처럼 선택자의 콜론을 속성으로 잘못 읽지 않게.
_RULE = re.compile(
    r"(?<![\w-])(?P<prop>selection-background-color|selection-color|background(?:-color)?|color"
    r"|border(?:-(?:top|bottom|left|right))?(?:-color)?|gridline-color|outline(?:-color)?)"
    r"(?P<sep>\s*:\s*)(?P<rest>[^;}\n]*)", re.IGNORECASE)
_HEX = re.compile(r"#[0-9A-Fa-f]{6}\b")


def _table(prop: str) -> dict[str, str]:
    prop = prop.lower()
    if prop in ("selection-background-color",):
        return {key: "accent_soft" for key in _BACKGROUND}
    if prop in ("selection-color",):
        return {key: "accent_text" for key in _TEXT}
    if prop.startswith("background"):
        return _BACKGROUND
    if prop == "color":
        return _TEXT
    if prop.startswith("border") or prop.endswith("-color"):
        return _LINE
    return {}


def themed_qss(qss: str, theme=None) -> str:
    """qss 안의 밝은 색을 지금(또는 넘긴) 테마의 색으로 바꾼 문자열."""
    if theme is None:
        from ui.theme import ThemeManager

        theme = ThemeManager.instance().theme

    def rule(match: re.Match) -> str:
        table = _table(match.group("prop"))
        if not table:
            return match.group(0)
        rest = _HEX.sub(lambda h: theme.hex(table[h.group(0).upper()]) if h.group(0).upper() in table else h.group(0),
                        match.group("rest"))
        return match.group("prop") + match.group("sep") + rest

    return _RULE.sub(rule, qss)
