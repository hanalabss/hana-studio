"""앱 아이콘과 로고 그림. 배포판(PyInstaller)과 소스 실행 양쪽에서 같은 경로로 찾는다."""
from __future__ import annotations

import os
import sys

from PySide6.QtGui import QIcon, QPixmap

_SIZES = (32, 48, 64, 128, 256)


def _root() -> str:
    base = getattr(sys, "_MEIPASS", None)
    if base:
        return base
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def brand_path(name: str) -> str:
    return os.path.join(_root(), "assets", "brand", name)


def app_icon() -> QIcon:
    """창·작업 표시줄에 쓰는 아이콘. 그림이 없으면 빈 아이콘."""
    icon = QIcon()
    for size in _SIZES:
        path = brand_path(f"icon_{size}.png")
        if os.path.exists(path):
            icon.addFile(path)
    if icon.isNull():
        fallback = os.path.join(_root(), "hana.ico")
        if os.path.exists(fallback):
            icon = QIcon(fallback)
    return icon


def logo_pixmap(size: int, device_pixel_ratio: float = 1.0) -> QPixmap:
    """화면에 size 크기로 그릴 로고. 배율이 높은 화면에서도 또렷하도록 큰 원본에서 줄인다."""
    from PySide6.QtCore import Qt

    target = max(1, round(size * device_pixel_ratio))
    source = next((s for s in _SIZES if s >= target), _SIZES[-1])
    pixmap = QPixmap(brand_path(f"icon_{source}.png"))
    if pixmap.isNull():
        return pixmap
    pixmap = pixmap.scaled(target, target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    pixmap.setDevicePixelRatio(device_pixel_ratio)
    return pixmap


def set_windows_app_id(app_id: str = "HanaPlatform.HanaStudio") -> None:
    """작업 표시줄에 파이썬 아이콘 대신 앱 아이콘이 보이게 한다(윈도우 전용, 실패해도 무시)."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
    except Exception:
        pass
