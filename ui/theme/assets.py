"""스타일시트가 쓰는 작은 그림(체크 표시, 화살표)을 테마 색으로 만들어 둔다."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

from .tokens import Theme

# 파일 이름 -> (아이콘 이름, 색 역할, 크기)
_ASSETS: dict[str, tuple[str, str, int]] = {
    "check": ("check", "text_on_accent", 14),
    "check-disabled": ("check", "text_disabled", 14),
    "chevron-down": ("chevron-down", "text_secondary", 14),
    "chevron-down-disabled": ("chevron-down", "text_disabled", 14),
    "chevron-up": ("chevron-up", "text_secondary", 14),
    "chevron-up-disabled": ("chevron-up", "text_disabled", 14),
    "chevron-right": ("chevron-right", "text_secondary", 14),
    "chevron-left": ("chevron-left", "text_secondary", 14),
    "calendar": ("calendar", "text_secondary", 16),
    "calendar-disabled": ("calendar", "text_disabled", 16),
    "sort-asc": ("chevron-up", "text_secondary", 12),
    "sort-desc": ("chevron-down", "text_secondary", 12),
    "spin-up": ("chevron-up", "text_secondary", 10),
    "spin-down": ("chevron-down", "text_secondary", 10),
    "spin-up-disabled": ("chevron-up", "text_disabled", 10),
    "spin-down-disabled": ("chevron-down", "text_disabled", 10),
}


def ensure_assets(theme: Theme) -> dict[str, str]:
    """그림 파일을 만들고 {이름: QSS url에 넣을 경로}를 돌려준다."""
    from ui.widgets.icons import icon_pixmap

    digest = hashlib.sha1(
        (theme.name + "".join(theme.hex(r) for _, r, _ in _ASSETS.values())).encode()
    ).hexdigest()[:10]
    folder = Path(tempfile.gettempdir()) / "hana_studio_ui" / f"{theme.name}-{digest}"
    folder.mkdir(parents=True, exist_ok=True)
    urls: dict[str, str] = {}
    for file_name, (icon, role, size) in _ASSETS.items():
        base = folder / f"{file_name}.png"
        if not base.exists():
            icon_pixmap(icon, size, theme.color(role), dpr=1.0, stroke=2.2).save(str(base))
            icon_pixmap(icon, size, theme.color(role), dpr=2.0, stroke=2.2).save(
                str(folder / f"{file_name}@2x.png")
            )
        urls[file_name] = base.as_posix()
    return urls
