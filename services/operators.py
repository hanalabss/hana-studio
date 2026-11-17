"""작업자 계정: 누가 인쇄했는지 기록에 남긴다.

- 작업자를 한 명도 만들지 않으면 예전과 똑같이 쓴다(로그인 없음, 기록에 작업자 없음).
- 작업자가 있으면 프로그램을 켤 때 누가 쓰는지 고르고 비밀번호(숫자 4~8자리)를 넣는다.
- 관리자 작업자만 작업자를 더하고 지울 수 있다. 처음 만드는 작업자는 관리자가 된다.
- 비밀번호는 PBKDF2-SHA256(소금 16바이트)으로만 저장한다.
- 이 PC의 사용자 폴더(%LOCALAPPDATA%\\HanaStudio\\operators.json)에 둔다. 비밀번호를 모두 잊으면 이 파일을 지운다.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
from dataclasses import asdict, dataclass
from pathlib import Path

from PySide6.QtCore import QObject, Signal

ITERATIONS = 120_000
PIN_MIN, PIN_MAX = 4, 8
NAME_MAX = 20


class OperatorError(ValueError):
    """사용자에게 그대로 보여 줄 한국어 문구."""


@dataclass
class Operator:
    name: str
    admin: bool
    salt: str
    pin_hash: str


def default_path() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    return Path(base) / "HanaStudio" / "operators.json"


def _hash(pin: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", pin.encode("utf-8"), salt, ITERATIONS).hex()


def check_pin(pin: str) -> str:
    pin = "" if pin is None else str(pin).strip()
    if not pin.isdigit() or not (PIN_MIN <= len(pin) <= PIN_MAX):
        raise OperatorError(f"비밀번호는 숫자 {PIN_MIN}~{PIN_MAX}자리로 정해 주세요.")
    return pin


class OperatorBook(QObject):
    """작업자 목록과 지금 작업자. changed: 목록이 바뀜. currentChanged(str): 지금 작업자가 바뀜("" = 없음)."""

    changed = Signal()
    currentChanged = Signal(str)

    _instance: "OperatorBook | None" = None

    def __init__(self, path: str | os.PathLike | None = None, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.path = Path(path) if path else default_path()
        self._operators: list[Operator] = []
        self._current: str = ""
        self._load()

    @classmethod
    def instance(cls) -> "OperatorBook":
        if cls._instance is None:
            cls._instance = OperatorBook()
        return cls._instance

    @classmethod
    def set_instance(cls, book: "OperatorBook | None") -> None:
        cls._instance = book

    # ---- 파일 ----
    def _load(self) -> None:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self._operators = [Operator(str(o["name"]), bool(o.get("admin")), str(o["salt"]), str(o["pin_hash"]))
                               for o in raw.get("operators", [])]
        except FileNotFoundError:
            self._operators = []
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise OperatorError(f"작업자 목록 파일을 읽지 못했습니다: {self.path} ({exc})") from exc

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(".tmp")
        temp.write_text(json.dumps({"version": 1, "operators": [asdict(o) for o in self._operators]},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp, self.path)
        self.changed.emit()

    # ---- 조회 ----
    @property
    def enabled(self) -> bool:
        """작업자가 한 명이라도 있으면 작업자 확인을 쓴다."""
        return bool(self._operators)

    def names(self) -> list[str]:
        return [o.name for o in self._operators]

    def operators(self) -> list[Operator]:
        return list(self._operators)

    def find(self, name: str) -> Operator | None:
        return next((o for o in self._operators if o.name == name), None)

    @property
    def current(self) -> str:
        return self._current

    def current_operator(self) -> Operator | None:
        return self.find(self._current) if self._current else None

    def can_manage(self) -> bool:
        """작업자를 더하고 지울 수 있는가: 아직 아무도 없거나 지금 작업자가 관리자."""
        if not self._operators:
            return True
        operator = self.current_operator()
        return operator is not None and operator.admin

    # ---- 로그인 ----
    def verify(self, name: str, pin: str) -> bool:
        operator = self.find(name)
        if operator is None:
            return False
        return hmac.compare_digest(operator.pin_hash, _hash(str(pin or "").strip(), bytes.fromhex(operator.salt)))

    def sign_in(self, name: str, pin: str) -> None:
        if not self.verify(name, pin):
            raise OperatorError("비밀번호가 맞지 않습니다.")
        self._set_current(name)

    def sign_out(self) -> None:
        self._set_current("")

    def _set_current(self, name: str) -> None:
        if name != self._current:
            self._current = name
            self.currentChanged.emit(name)

    # ---- 관리 ----
    def _require_manager(self) -> None:
        if not self.can_manage():
            raise OperatorError("작업자는 관리자만 더하거나 지울 수 있습니다.")

    def add(self, name: str, pin: str, admin: bool = False) -> Operator:
        self._require_manager()
        name = " ".join(str(name or "").split())
        if not name:
            raise OperatorError("작업자 이름을 적어 주세요.")
        if len(name) > NAME_MAX:
            raise OperatorError(f"작업자 이름은 {NAME_MAX}자 안으로 적어 주세요.")
        if self.find(name) is not None:
            raise OperatorError(f"'{name}' 작업자가 이미 있습니다.")
        pin = check_pin(pin)
        first = not self._operators
        salt = secrets.token_bytes(16)
        operator = Operator(name, bool(admin) or first, salt.hex(), _hash(pin, salt))
        self._operators.append(operator)
        self._save()
        if first:
            self._set_current(name)             # 처음 만든 사람이 그대로 지금 작업자
        return operator

    def remove(self, name: str) -> None:
        self._require_manager()
        operator = self.find(name)
        if operator is None:
            return
        admins = [o for o in self._operators if o.admin]
        if operator.admin and len(admins) == 1 and len(self._operators) > 1:
            raise OperatorError("관리자가 한 명은 있어야 합니다. 다른 사람을 관리자로 만든 뒤 지워 주세요.")
        self._operators.remove(operator)
        self._save()
        if name == self._current:
            self._set_current("")

    def change_pin(self, name: str, old_pin: str, new_pin: str) -> None:
        operator = self.find(name)
        if operator is None:
            raise OperatorError("그 작업자가 없습니다.")
        manager = self.can_manage() and self._current != name
        if not manager and not self.verify(name, old_pin):
            raise OperatorError("지금 비밀번호가 맞지 않습니다.")
        new_pin = check_pin(new_pin)
        salt = secrets.token_bytes(16)
        operator.salt, operator.pin_hash = salt.hex(), _hash(new_pin, salt)
        self._save()
