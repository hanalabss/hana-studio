"""
printer/base.py
프린터 제품군(RTAI, SMART)이 공통으로 따르는 추상 계층.
Qt에 의존하지 않는다. 진행 상황은 콜백으로 알리고, QThread 래퍼가 시그널로 중계한다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, List, Mapping, Optional, Protocol, Set, Union, runtime_checkable,
)

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
CAP_COLOR_ADJUST = "color_adjust"      # 밝기·명암·채도를 프린터 쪽에서 조정
CAP_BLACK_RESIN = "black_resin"        # 검정 글자·바코드를 색 패널 대신 검정(K 레진) 패널로 찍기

# 색 조정 값의 범위. RTAI SDK(R600SetImgVisualParam)가 받는 범위를 실제로 넣어 보고 확인한 값이다.
COLOR_ADJUST_MIN, COLOR_ADJUST_MAX = -100, 100
COLOR_ADJUST_KEYS = ("brightness", "contrast", "saturation")


def _adjust_values(options, keys) -> tuple:
    values = []
    for key in keys:
        try:
            value = int(round(float(options.get(key, 0) or 0)))
        except (TypeError, ValueError, AttributeError):
            value = 0
        values.append(max(COLOR_ADJUST_MIN, min(COLOR_ADJUST_MAX, value)))
    return tuple(values)


def color_adjust_from(options) -> tuple:
    """인쇄 옵션에서 (밝기, 명암, 채도)를 꺼낸다. 없거나 잘못된 값은 0, 범위를 넘으면 끝값으로 맞춘다."""
    return _adjust_values(options, COLOR_ADJUST_KEYS)


# 중간 밝기(감마): 흰색과 검정은 그대로 두고 중간 밝기만 밝게(+)·어둡게(-) 한다. RTAI 드라이버의 Gamma에 해당한다.
# RTAI SDK에는 감마 함수가 없어 보내기 전에 그림을 바꾼다. 범위는 색 조정과 같은 -100~100이다.
CAP_GAMMA_ADJUST = "gamma_adjust"


def gamma_adjust_from(options) -> int:
    return _adjust_values(options, ("gamma",))[0]


def gamma_exponent(value: int) -> float:
    """-100~100을 지수로 바꾼다. 0이면 1(그대로), +100이면 0.5(밝게), -100이면 2(어둡게)."""
    return 2.0 ** (-max(COLOR_ADJUST_MIN, min(COLOR_ADJUST_MAX, int(value))) / 100.0)


def apply_gamma(image, value: int):
    """PIL 그림(RGB)에 중간 밝기 조정을 적용한다. 0과 255는 변하지 않는다."""
    if not value:
        return image
    exponent = gamma_exponent(value)
    table = [round(255 * (level / 255) ** exponent) for level in range(256)]
    return image.point(table * len(image.getbands()))


# 진하기 조정: 색 리본을 얼마나 진하게 찍을지(전체·노랑·빨강·파랑). SMART 드라이버의 'Color Correction'과 같은 값이다.
# 범위는 색 조정과 같은 -100~100이고 클수록 진하다. SDK는 범위를 넘는 값도 그대로 받으므로 여기서 막는다.
CAP_DENSITY_ADJUST = "density_adjust"
DENSITY_KEYS = ("density", "density_yellow", "density_magenta", "density_cyan")


def density_adjust_from(options) -> tuple:
    """인쇄 옵션에서 (전체, 노랑, 빨강, 파랑) 진하기를 꺼낸다. 규칙은 color_adjust_from과 같다."""
    return _adjust_values(options, DENSITY_KEYS)

# (진행률 0~100, 사용자에게 보여 줄 한국어 문구)
ProgressCallback = Callable[[int, str], None]

# 카드 한 장을 인쇄하는 동안의 단계. 드라이버가 스스로 경계를 아는 단계만 알린다.
STAGE_UNKNOWN = ""                 # 단계 정보를 주지 않는 드라이버(예전 방식 콜백)
STAGE_PREPARE = "prepare"          # 인쇄할 이미지 준비(PC 쪽 작업)
STAGE_FEED = "feed"                # 카드 삽입
STAGE_MAGNETIC = "magnetic"        # 마그네틱 기록
STAGE_SETUP = "setup"              # 리본 등 인쇄 설정
STAGE_FRONT_IMAGE = "front_image"  # 앞면 이미지를 SDK에 올림
STAGE_BACK_IMAGE = "back_image"    # 뒷면 이미지를 SDK에 올림
STAGE_PRINT = "print"              # 인쇄(SDK 호출 한 번이 양면·뒤집기까지 맡는 장치는 이 단계 하나)
STAGE_PRINT_FRONT = "print_front"  # 앞면 인쇄(면마다 따로 호출하는 장치만)
STAGE_PRINT_BACK = "print_back"    # 뒷면 인쇄(면마다 따로 호출하는 장치만)
STAGE_WAIT = "wait"                # 인쇄 완료 대기
STAGE_EJECT = "eject"              # 카드 배출
STAGE_DONE = "done"                # 이 카드 끝

STAGE_LABELS = {
    STAGE_PREPARE: "이미지 준비", STAGE_FEED: "카드 삽입", STAGE_MAGNETIC: "마그네틱 기록",
    STAGE_SETUP: "인쇄 설정", STAGE_FRONT_IMAGE: "앞면 준비", STAGE_BACK_IMAGE: "뒷면 준비",
    STAGE_PRINT: "인쇄", STAGE_PRINT_FRONT: "앞면 인쇄", STAGE_PRINT_BACK: "뒷면 인쇄",
    STAGE_WAIT: "인쇄 완료 대기", STAGE_EJECT: "카드 배출", STAGE_DONE: "완료",
}


@dataclass(frozen=True)
class ProgressEvent:
    """구조화된 진행 상황 한 건.

    percent  이 print_card 호출(모든 copies) 안에서의 진행률 0~100. 단계 경계에서 정한 값이지
             장치가 잰 값이 아니다.
    text     사용자에게 보여 줄 한국어 문구(예전 콜백의 두 번째 인자와 같다).
    stage    STAGE_* 키. 모르면 STAGE_UNKNOWN("").
    panel    지금 찍는 리본 패널("Y", "M", "C", "K", "O", "W", "UV" ...). 장치가 알려 줄 때만 채운다.
             RTAI(R600PrintDraw)와 SMART(SmartComm_DoPrint)는 인쇄가 SDK 호출 하나로 끝날 때까지
             돌아오지 않고 패널 진행을 알려 주는 함수도 없어서 항상 None이다.
    fraction 현재 단계 안에서의 진행 0.0~1.0. 장치가 알려 줄 때만 채운다(지금은 항상 None).
    side     "front" | "back" - 그 단계가 한 면에만 해당할 때.
    copy / copies  같은 카드를 여러 장 찍을 때 몇 번째 장인지.
    """
    percent: int
    text: str
    stage: str = STAGE_UNKNOWN
    panel: Optional[str] = None
    fraction: Optional[float] = None
    side: Optional[str] = None
    copy: int = 1
    copies: int = 1


class StructuredProgress:
    """ProgressEvent를 받고 싶을 때 print_card(progress=...)에 넘기는 콜백.

        printer.print_card(job, progress=StructuredProgress(lambda event: ...))

    예전 방식 `(percent, text)`로 불러도 되므로(STAGE_UNKNOWN 이벤트가 된다) 단계 정보를 모르는
    드라이버와도 그대로 쓸 수 있다. 일반 함수 `callback(percent, text)`도 계속 받는다.
    """

    def __init__(self, handler: Callable[[ProgressEvent], None]):
        self.handler = handler

    def __call__(self, percent: int, text: str) -> None:
        self.handler(ProgressEvent(max(0, min(100, int(percent))), text))

    def event(self, event: ProgressEvent) -> None:
        self.handler(event)


def emit_progress(callback: Optional[Callable[..., None]], event: ProgressEvent) -> None:
    """콜백 종류에 맞춰 알린다: StructuredProgress면 이벤트째, 아니면 (percent, text)."""
    if callback is None:
        return
    send = getattr(callback, "event", None)
    if callable(send):
        send(event)
    else:
        callback(event.percent, event.text)


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


class PrintCancelled(PrinterError):
    """사용자가 취소해서 이 카드를 인쇄하지 않았다(실패가 아니다).

    카드 공급자(CardProvider)가 아직 프린터로 넘어가지 않은 카드를 버릴 때 던진다.
    인쇄 스레드는 이 카드를 "cancelled"로 기록하고 남은 카드를 인쇄하지 않는다.
    """

    def __init__(self, message: str = "인쇄가 취소되었습니다."):
        super().__init__(message, recoverable=True)


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


# ribbon_remaining / film_remaining 값의 단위
UNIT_UNKNOWN = "unknown"     # SDK가 준 값 그대로. 무엇을 센 값인지 실기로 확인하지 못했다
UNIT_CARDS = "cards"         # 인쇄할 수 있는 카드(면) 수
UNIT_PERCENT = "percent"     # 0~100 %


@dataclass
class PrinterStatus:
    """프린터 상태.

    ribbon_remaining / film_remaining은 SDK가 준 숫자를 그대로 담는다. 그 숫자가 무엇인지는
    ribbon_unit / film_unit이 말한다. 화면은 단위가 UNIT_CARDS일 때만 "약 N장", UNIT_PERCENT일 때만
    "N %"로 쓰고, UNIT_UNKNOWN이면 장수로 바꿔 말하지 않는다(숫자만 보이거나 "확인 불가").
    `ribbon_cards_left`가 장수로 믿을 수 있는 값만 돌려준다.

    어댑터별 실제 값:
      RtaiAdapter   ribbon/film = R600GetRbnAndFilmRemaining의 unsigned short 두 개 그대로.
                    SDK 문서에 단위가 없고 실기로 확인하지 못했다 → 둘 다 UNIT_UNKNOWN.
      SmartAdapter  ribbon = SmartComm_GetRibbonInfo의 remain(남은 인쇄 횟수, 최대값은 ribbon_capacity).
                    SDK가 (최대, 남은) 횟수로 주므로 UNIT_CARDS로 본다(실기 확인 전). 최대값이 0이거나
                    100이라 % 값과 구별할 수 없거나, 잔량이 0~최대를 벗어나면 UNIT_UNKNOWN.
                    film = SMART-81의 GetRibbonInfo2(RETRANSFER).remain, 같은 규칙. 그 밖의 기종은 None.
      가짜·기타     단위를 채우지 않으면 UNIT_UNKNOWN.
    0 이하는 단위와 관계없이 "다 썼다"로 본다.
    """
    ready: bool
    message: str
    code: Optional[int] = None
    ribbon_remaining: Optional[int] = None
    film_remaining: Optional[int] = None
    ribbon_unit: str = UNIT_UNKNOWN
    film_unit: str = UNIT_UNKNOWN
    ribbon_capacity: Optional[int] = None
    ribbon_type: Optional[int] = None
    feeder_empty: Optional[bool] = None
    card_inside: Optional[bool] = None
    busy: bool = False
    warnings: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def ribbon_cards_left(self) -> Optional[int]:
        """리본으로 더 찍을 수 있는 카드(면) 수. 단위가 장수가 아니거나 모르면 None."""
        if self.ribbon_remaining is None or self.ribbon_unit != UNIT_CARDS:
            return None
        return max(0, int(self.ribbon_remaining))

    @property
    def film_cards_left(self) -> Optional[int]:
        """필름으로 더 찍을 수 있는 카드(면) 수. 단위가 장수가 아니거나 모르면 None."""
        if self.film_remaining is None or self.film_unit != UNIT_CARDS:
            return None
        return max(0, int(self.film_remaining))


def remaining_text(value: Optional[int], unit: str = UNIT_UNKNOWN) -> str:
    """잔량을 단위에 맞는 한국어로: "약 310장" / "62 %" / "310 (단위 확인 안 됨)" / "확인 불가"."""
    if value is None:
        return "확인 불가"
    if unit == UNIT_CARDS:
        return f"약 {int(value)}장"
    if unit == UNIT_PERCENT:
        return f"{int(value)} %"
    return f"{int(value)} (단위 확인 안 됨)"


@dataclass
class SideImages:
    """한 면에 인쇄할 이미지 파일 묶음. 마스크·UV는 없으면 None."""
    color_path: str
    white_mask_path: Optional[str] = None
    uv_path: Optional[str] = None
    black_path: Optional[str] = None       # 검정(K 레진) 패널로 찍을 부분. 흰색이 잉크인 마스크.
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
      bleed_handled              이미지가 이미 칼선 크기다. 참이면 RtaiAdapter·SmartAdapter가 비율 기반
                                 도련 제거(utils.bleed_crop)를 하지 않는다 (기본 False = 기존 동작)
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


@runtime_checkable
class CardProvider(Protocol):
    """카드를 필요할 때 만들어 주는 공급자. print_queue.add_card_job(device, provider)에 넘긴다.

    __len__()            인쇄할 카드 수. 큐에 넣을 때 한 번 읽는다(1 이상).
    __getitem__(index)   index번(0부터) 카드의 CardJob. 인쇄 스레드가 그 카드를 찍기 직전에 한 번
                         부른다. 오래 걸려도 된다(렌더링 등). PrinterError를 던지면 그 카드만 실패로
                         처리되고, PrintCancelled를 던지면 그 카드는 "cancelled"로 남고 작업이 멈춘다.
    선택 사항(없어도 된다):
      is_dual_side       양면 작업인지(bool 속성). 작업 정보·이력에만 쓴다. 없으면 False.
      record_id(index)   카드를 만들지 못했을 때도 결과·이력에 레코드를 남기기 위한 조회.
    list·tuple은 그대로 공급자다.
    """

    def __len__(self) -> int:
        ...

    def __getitem__(self, index: int) -> CardJob:
        ...


def as_card_provider(cards: Any) -> Any:
    """add_card_job이 받은 cards를 공급자로 맞춘다. list·tuple은 복사하고, 공급자는 그대로 두며
    (카드를 미리 만들지 않는다), 그 밖의 이터러블(제너레이터 등)은 list로 풀어 쓴다."""
    if isinstance(cards, (list, tuple)):
        return list(cards)
    if hasattr(cards, "__len__") and hasattr(cards, "__getitem__"):
        return cards
    return list(cards)


def provider_is_dual_side(cards: Any) -> bool:
    if isinstance(cards, list):
        return any(card.is_dual_side for card in cards)
    return bool(getattr(cards, "is_dual_side", False))


def provider_record_id(cards: Any, index: int) -> Any:
    """index번 카드의 record_id를 카드를 만들지 않고 알아낸다. 알 수 없으면 None."""
    if isinstance(cards, list):
        return cards[index].record_id
    getter = getattr(cards, "record_id", None)
    if not callable(getter):
        return None
    try:
        return getter(index)
    except Exception:
        return None


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
                override: Optional[ProgressCallback] = None, *,
                stage: str = STAGE_UNKNOWN, panel: Optional[str] = None,
                fraction: Optional[float] = None, side: Optional[str] = None,
                copy: int = 1, copies: int = 1) -> None:
        """진행 상황을 알린다. 콜백이 StructuredProgress면 단계 정보까지, 아니면 (percent, 문구)만."""
        callback = override or self._progress_callback
        if callback is not None:
            emit_progress(callback, ProgressEvent(
                max(0, min(100, int(percent))), message, stage, panel, fraction, side, copy, copies))

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
        """job.copies장을 인쇄한다. 실패하면 카드를 배출한 뒤 PrinterError를 던진다.

        progress: `callback(percent, text)` 또는 StructuredProgress(단계 키·패널까지 받는다).
        """

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
