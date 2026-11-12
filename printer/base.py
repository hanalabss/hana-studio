"""
printer/base.py
프린터 제품군(RTAI, SMART)이 공통으로 따르는 추상 계층.
Qt에 의존하지 않는다. 진행 상황은 콜백으로 알리고, QThread 래퍼가 시그널로 중계한다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Set, Union

FAMILY_RTAI = "rtai"
FAMILY_SMART = "smart"

ORIENTATION_PORTRAIT = "portrait"
ORIENTATION_LANDSCAPE = "landscape"

CAP_DUAL_SIDE = "dual_side"
CAP_WHITE = "white"
CAP_UV = "uv"
CAP_MAGNETIC = "magnetic"
CAP_STATUS = "status"
CAP_RIBBON_LEVEL = "ribbon_level"
CAP_K_SPLIT = "k_split"
CAP_MIFARE = "mifare"
CAP_CARD_MOVE = "card_move"
CAP_POSITION_OFFSET = "position_offset"

# (진행률 0~100, 사용자에게 보여 줄 한국어 문구)
ProgressCallback = Callable[[int, str], None]


class PrinterError(Exception):
    """인쇄 계층의 모든 오류.

    message는 사용자에게 그대로 보여 줄 한국어 문장이다.
    code는 SDK가 돌려준 원래 값, recoverable은 사용자가 조치(카드 보충, 리본 교체 등) 후
    다시 시도할 수 있는지 여부다.
    """

    def __init__(self, message: str, code: Optional[int] = None,
                 recoverable: bool = False, detail: str = ""):
        super().__init__(message)
        self.message = message
        self.code = code
        self.recoverable = recoverable
        self.detail = detail

    def __str__(self) -> str:
        if self.code is None:
            return self.message
        return f"{self.message} (코드 {format_code(self.code)})"


class NotSupported(PrinterError):
    """해당 프린터가 지원하지 않는 기능을 요청했을 때."""

    def __init__(self, feature: str, message: Optional[str] = None):
        super().__init__(message or f"이 프린터는 '{feature}' 기능을 지원하지 않습니다.")
        self.feature = feature


def format_code(code: int) -> str:
    return f"0x{code & 0xFFFFFFFF:08X}" if code > 0xFFFF or code < 0 else str(code)


@dataclass(frozen=True)
class PrinterDevice:
    """탐지된 프린터 한 대.

    address는 제품군 SDK가 장치를 여는 데 쓰는 값이다
    (RTAI: 열거된 프린터 이름, SMART: Description 문자열).
    """
    family: str
    name: str
    connection: str = ""          # "usb" | "tcp" | "net" | "fake"
    address: str = ""
    extra: Mapping[str, Any] = field(default_factory=dict, compare=False, hash=False)

    @property
    def display_name(self) -> str:
        family = {"rtai": "RTAI", "smart": "SMART"}.get(self.family, self.family)
        connection = {"usb": "USB", "tcp": "네트워크", "net": "네트워크",
                      "fake": "가상"}.get(self.connection, self.connection)
        return f"[{family}] {self.name} ({connection})" if connection else f"[{family}] {self.name}"


@dataclass
class PrinterStatus:
    ready: bool
    message: str
    code: Optional[int] = None
    ribbon_remaining: Optional[int] = None
    film_remaining: Optional[int] = None
    ribbon_type: Optional[int] = None
    feeder_empty: Optional[bool] = None
    card_inside: Optional[bool] = None
    busy: bool = False
    warnings: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SideImages:
    """한 면에 인쇄할 이미지 파일 묶음. 마스크·UV는 없으면 None."""
    color_path: str
    white_mask_path: Optional[str] = None
    uv_path: Optional[str] = None
    orientation: str = ORIENTATION_PORTRAIT


@dataclass
class MagneticTracks:
    """ISO 마그네틱 트랙 데이터. 쓰지 않을 트랙은 None."""
    track1: Optional[str] = None
    track2: Optional[str] = None
    track3: Optional[str] = None

    def is_empty(self) -> bool:
        return not (self.track1 or self.track2 or self.track3)

    def as_tuple(self) -> tuple:
        return (self.track1, self.track2, self.track3)

    @classmethod
    def coerce(cls, value: Union["MagneticTracks", Mapping[str, Any], tuple, list, None]
               ) -> "MagneticTracks":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            def pick(*keys):
                for key in keys:
                    if value.get(key):
                        return str(value[key])
                return None
            return cls(pick("track1", "t1", 1), pick("track2", "t2", 2), pick("track3", "t3", 3))
        items = list(value) + [None, None, None]
        return cls(*(str(v) if v else None for v in items[:3]))


@dataclass
class CardJob:
    """카드 한 장(또는 같은 카드 copies장)의 인쇄 요청.

    options에서 쓰는 키(모두 선택):
      offset_x_mm, offset_y_mm   인쇄 위치 조정(mm)
      magnetic                   MagneticTracks 또는 {"track1":..,"track2":..,"track3":..}
      magnetic_verify            쓴 뒤 다시 읽어 비교할지 (기본 True)
      magnetic_coercivity        "high"(기본) | "low"
      uv_threshold, uv_thickness UV 레이어 임계값(기본 220)·두께(기본 "180") - RTAI
      k_split                    K 패널 분할 인쇄 여부 - RTAI (지정하지 않으면 호출 안 함)
      precheck                   인쇄 전에 상태를 조회해 카드 없음 등을 미리 알릴지 (기본 False)
    record_id는 데이터 레코드와 인쇄 이력을 잇는 값이다(없으면 None).
    """
    front: Optional[SideImages] = None
    back: Optional[SideImages] = None
    copies: int = 1
    options: Dict[str, Any] = field(default_factory=dict)
    record_id: Any = None

    @property
    def is_dual_side(self) -> bool:
        return self.front is not None and self.back is not None

    def validate(self) -> None:
        if self.front is None and self.back is None:
            raise PrinterError("인쇄할 이미지가 없습니다.")
        if self.copies < 1:
            raise PrinterError("인쇄 매수는 1장 이상이어야 합니다.")
        for side in (self.front, self.back):
            if side is not None and side.orientation not in (ORIENTATION_PORTRAIT,
                                                            ORIENTATION_LANDSCAPE):
                raise PrinterError(f"알 수 없는 카드 방향입니다: {side.orientation}")


class CardPrinter(ABC):
    """프린터 제품군 공통 인터페이스."""

    family: str = ""

    def __init__(self) -> None:
        self._progress_callback: Optional[ProgressCallback] = None
        self.device: Optional[PrinterDevice] = None

    # -- 진행 상황 ---------------------------------------------------------
    def set_progress_callback(self, callback: Optional[ProgressCallback]) -> None:
        self._progress_callback = callback

    def _report(self, percent: int, message: str,
                override: Optional[ProgressCallback] = None) -> None:
        callback = override or self._progress_callback
        if callback is not None:
            callback(max(0, min(100, int(percent))), message)

    # -- 필수 구현 ---------------------------------------------------------
    @abstractmethod
    def capabilities(self) -> Set[str]:
        """지원 기능 집합. 연결 전에는 제품군 기준, 연결 후에는 장착 옵션까지 반영한다."""

    @abstractmethod
    def connect(self, device: PrinterDevice) -> None:
        ...

    @abstractmethod
    def close(self) -> None:
        ...

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        ...

    @abstractmethod
    def status(self) -> PrinterStatus:
        ...

    @abstractmethod
    def print_card(self, job: CardJob, progress: Optional[ProgressCallback] = None) -> None:
        """job.copies장을 인쇄한다. 실패하면 카드를 배출한 뒤 PrinterError를 던진다."""

    # -- 선택 기능 (지원하지 않으면 NotSupported) ----------------------------
    def write_magnetic(self, tracks: Union[MagneticTracks, Mapping[str, Any], tuple, list],
                       verify: bool = True) -> None:
        raise NotSupported(CAP_MAGNETIC)

    def read_magnetic(self) -> MagneticTracks:
        raise NotSupported(CAP_MAGNETIC)

    def inject_card(self) -> None:
        raise NotSupported(CAP_CARD_MOVE)

    def eject_card(self) -> None:
        raise NotSupported(CAP_CARD_MOVE)

    def info(self) -> Dict[str, Any]:
        """모델·시리얼·펌웨어 등 표시용 정보. 얻지 못한 항목은 넣지 않는다."""
        return {}

    def supports(self, capability: str) -> bool:
        return capability in self.capabilities()

    def _require_connected(self) -> None:
        if not self.is_connected:
            raise PrinterError("프린터에 연결되어 있지 않습니다.", recoverable=True)

    def __enter__(self) -> "CardPrinter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
