"""글꼴 선택과 HiDPI 설정."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QFontDatabase, QGuiApplication

from .tokens import FONT_SIZES

# 설치돼 있으면 앞의 것부터 쓴다. 맑은 고딕은 Windows에 항상 있다.
FONT_STACK: tuple[str, ...] = (
    "Pretendard",
    "Pretendard Variable",
    "Noto Sans KR",
    "Malgun Gothic",
)
_SEMIBOLD_FAMILIES = ("Pretendard", "Pretendard Variable")


@dataclass(frozen=True)
class FontSpec:
    family: str
    strong_weight: int

    def css_family(self) -> str:
        names = [self.family] + [n for n in ("Malgun Gothic", "Segoe UI") if n != self.family]
        return ", ".join(f'"{n}"' for n in names)


_FONT_FILE_PATTERNS = ("Pretendard*.ttf", "Pretendard*.otf", "NotoSansKR*.ttf", "NotoSansKR*.otf",
                       "malgun.ttf", "malgunbd.ttf")


def _load_installed_fonts() -> None:
    """화면 없는 환경(offscreen)은 Windows 글꼴을 스스로 찾지 못하므로 파일에서 직접 읽는다."""
    folders = [Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"]
    local = os.environ.get("LOCALAPPDATA")
    if local:
        folders.append(Path(local) / "Microsoft" / "Windows" / "Fonts")
    for folder in folders:
        if not folder.is_dir():
            continue
        for pattern in _FONT_FILE_PATTERNS:
            for path in folder.glob(pattern):
                QFontDatabase.addApplicationFont(str(path))


def resolve_font(preferred: str | None = None) -> FontSpec:
    if not set(FONT_STACK) & set(QFontDatabase.families()):
        _load_installed_fonts()
    available = set(QFontDatabase.families())
    wanted = preferred or os.environ.get("HANA_STUDIO_FONT")
    candidates = ((wanted,) if wanted else ()) + FONT_STACK
    family = next((name for name in candidates if name in available), None)
    if family is None:
        family = QFontDatabase.systemFont(QFontDatabase.GeneralFont).family()
    return FontSpec(family, 600 if family in _SEMIBOLD_FAMILIES else 700)


def application_font(spec: FontSpec) -> QFont:
    font = QFont()
    font.setFamilies([spec.family, "Malgun Gothic", "Segoe UI"])
    font.setPixelSize(FONT_SIZES["body"])
    font.setStyleStrategy(QFont.PreferAntialias)
    return font


def configure_hidpi() -> None:
    """QApplication을 만들기 전에 부른다. 125%, 150% 배율에서도 흐려지지 않게 한다."""
    if QGuiApplication.instance() is not None:
        return
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
