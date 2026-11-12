"""온라인 업데이트: 새 버전이 있는지 보고, 받아서, 다시 켜면서 바꾼다 (Velopack + GitHub 릴리스).

설치 파일(HanaStudio2-win-Setup.exe)로 설치한 프로그램에서만 동작한다. 소스에서 바로 실행했거나
예전 방식(Inno Setup)으로 설치한 프로그램에서는 `supported`가 False이고 아무 일도 하지 않는다.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Optional

log = logging.getLogger(__name__)

REPO_URL = "https://github.com/hanalabss/hana-studio"
PACK_ID = "HanaStudio2"
# 시험할 때만 쓴다: 릴리스 파일이 든 폴더나 주소를 넣으면 GitHub 대신 거기서 찾는다.
ENV_SOURCE = "HANA_STUDIO_UPDATE_URL"


class UpdateError(Exception):
    """사용자에게 그대로 보여 줄 수 있는 문구를 담는다."""


@dataclass
class AvailableUpdate:
    version: str
    size_bytes: int = 0
    notes: str = ""
    raw: Any = None          # velopack.UpdateInfo

    @property
    def size_text(self) -> str:
        if self.size_bytes <= 0:
            return ""
        megabytes = self.size_bytes / (1024 * 1024)
        return f"{megabytes:.0f} MB" if megabytes >= 10 else f"{megabytes:.1f} MB"


def run_startup_hook() -> None:
    """프로그램이 켜질 때 가장 먼저 부른다. 설치·업데이트·제거 직후의 뒷정리를 Velopack이 여기서 한다.

    그런 일로 켜진 경우에는 이 안에서 프로세스가 끝나거나 다시 켜질 수 있다.
    """
    try:
        import velopack
    except Exception:
        return
    try:
        velopack.App().run()
    except Exception:
        log.exception("업데이트 시작 처리에 실패했습니다")


def _download_size(info: Any) -> int:
    try:
        deltas = list(info.DeltasToTarget or [])
        if deltas and info.BaseRelease is not None:
            return int(sum(asset.Size for asset in deltas))
        return int(info.TargetFullRelease.Size)
    except Exception:
        return 0


class Updater:
    """Velopack UpdateManager를 감싼다. 만들 때 실패하면(설치본이 아님) supported=False로 남는다."""

    def __init__(self, source: str | None = None, manager_factory: Callable[[Any], Any] | None = None) -> None:
        self.source = source or os.environ.get(ENV_SOURCE) or REPO_URL
        self.reason = ""
        self._manager = None
        try:
            self._manager = (manager_factory or self._default_factory)(self.source)
            self._manager.get_current_version()
        except Exception as exc:
            self._manager = None
            self.reason = str(exc)

    @staticmethod
    def _default_factory(source: str) -> Any:
        import velopack

        if source.rstrip("/").lower().startswith("https://github.com/"):
            return velopack.UpdateManager(velopack.GithubSource(source.rstrip("/"), None, False))
        return velopack.UpdateManager(source)

    @property
    def supported(self) -> bool:
        return self._manager is not None

    def current_version(self) -> str:
        return str(self._manager.get_current_version()) if self._manager is not None else ""

    def check(self) -> Optional[AvailableUpdate]:
        """새 버전이 있으면 돌려주고, 없으면 None. 확인하지 못하면 UpdateError."""
        if self._manager is None:
            return None
        try:
            info = self._manager.check_for_updates()
        except Exception as exc:
            print(f"[UPDATE] 확인 중 오류: {type(exc).__name__}: {exc}")
            raise UpdateError("새 버전이 있는지 확인하지 못했습니다. 인터넷 연결을 확인해 주세요.") from exc
        if not info:
            return None
        target = info.TargetFullRelease
        return AvailableUpdate(str(target.Version), _download_size(info), str(target.NotesMarkdown or ""), info)

    def download(self, update: AvailableUpdate, progress: Callable[[int], None] | None = None) -> None:
        """새 버전을 받는다. progress(0~100)."""
        if self._manager is None:
            raise UpdateError("이 프로그램은 온라인 업데이트를 쓸 수 없습니다.")
        try:
            self._manager.download_updates(update.raw, progress)
        except Exception as exc:
            print(f"[UPDATE] 받는 중 오류: {type(exc).__name__}: {exc}")
            raise UpdateError("새 버전을 받지 못했습니다. 인터넷 연결을 확인하고 다시 해 주세요.") from exc

    def apply_and_restart(self, update: AvailableUpdate) -> None:
        """받아 둔 새 버전으로 바꾸고 다시 켠다. 이 프로세스는 곧 끝난다."""
        if self._manager is None:
            raise UpdateError("이 프로그램은 온라인 업데이트를 쓸 수 없습니다.")
        try:
            self._manager.wait_exit_then_apply_updates(update.raw, False, True, None)
        except Exception as exc:
            raise UpdateError("새 버전을 설치하지 못했습니다. 프로그램을 다시 켠 뒤 해 주세요.") from exc
