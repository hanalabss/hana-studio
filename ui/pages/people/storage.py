"""명단 파일을 어디에 둘지 정하고, 최근에 쓴 명단을 기억한다. 사용자는 위치를 고르지 않는다."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QSettings

from ui.theme import app_settings

LIST_SUFFIX = ".hsdata"
KEY_RECENT = "people/recent_lists"
MAX_RECENT = 8
_BAD_CHARS = re.compile(r'[\/:*?"<>|\x00-\x1f]+')


@dataclass(frozen=True)
class RecentList:
    path: str
    name: str
    modified: datetime
    beside_design: bool = False


def data_folder() -> Path:
    """카드 파일을 아직 저장하지 않았을 때 명단을 두는 사용자별 폴더."""
    override = os.environ.get("HANA_STUDIO_DATA_DIR")
    if override:
        return Path(override)
    base = os.environ.get("APPDATA")
    root = Path(base) / "HanaStudio" if base else Path.home() / ".hana_studio"
    return root / "명단"


def _clean_name(name: str) -> str:
    return _BAD_CHARS.sub(" ", name).strip(" .") or "명단"


def _unique(path: Path) -> Path:
    if not path.exists():
        return path
    for number in range(2, 1000):
        candidate = path.with_name(f"{path.stem} ({number}){path.suffix}")
        if not candidate.exists():
            return candidate
    return path.with_name(f"{path.stem} {datetime.now():%Y%m%d%H%M%S}{path.suffix}")


def beside_design_path(design_path: str | os.PathLike | None) -> Path | None:
    if not design_path:
        return None
    return Path(design_path).with_suffix(LIST_SUFFIX)


def new_list_path(design_path: str | os.PathLike | None, name_hint: str | None = None) -> Path:
    """새 명단 파일 경로. 카드 파일이 저장돼 있으면 그 옆에, 아니면 사용자 폴더에 둔다."""
    beside = beside_design_path(design_path)
    if beside is not None:
        return _unique(beside)
    name = _clean_name(name_hint or f"명단 {datetime.now():%Y-%m-%d}")
    return _unique(data_folder() / f"{name}{LIST_SUFFIX}")


def _read_paths(settings: QSettings) -> list[str]:
    value = settings.value(KEY_RECENT, [])
    if isinstance(value, str):
        value = [value] if value else []
    return [str(item) for item in (value or []) if item]


def remember_list(path: str | os.PathLike, settings: QSettings | None = None) -> None:
    settings = settings if settings is not None else app_settings()
    text = str(Path(path))
    paths = [text] + [p for p in _read_paths(settings) if os.path.normcase(p) != os.path.normcase(text)]
    settings.setValue(KEY_RECENT, paths[:MAX_RECENT])
    settings.sync()


def forget_list(path: str | os.PathLike, settings: QSettings | None = None) -> None:
    settings = settings if settings is not None else app_settings()
    text = os.path.normcase(str(Path(path)))
    settings.setValue(KEY_RECENT, [p for p in _read_paths(settings) if os.path.normcase(p) != text])
    settings.sync()


def recent_lists(design_path: str | os.PathLike | None = None, settings: QSettings | None = None,
                 limit: int = 4) -> list[RecentList]:
    """지금도 있는 파일만. 이 카드 옆에 있는 명단이 맨 앞에 온다."""
    settings = settings if settings is not None else app_settings()
    beside = beside_design_path(design_path)
    candidates: list[tuple[str, bool]] = []
    if beside is not None and beside.exists():
        candidates.append((str(beside), True))
    candidates += [(p, False) for p in _read_paths(settings)]
    seen: set[str] = set()
    result: list[RecentList] = []
    for text, beside_flag in candidates:
        key = os.path.normcase(text)
        if key in seen:
            continue
        seen.add(key)
        path = Path(text)
        try:
            modified = datetime.fromtimestamp(path.stat().st_mtime)
        except OSError:
            continue
        result.append(RecentList(text, path.stem, modified, beside_flag))
        if len(result) >= limit:
            break
    return result
