"""
printer/rtai_adapter.py
RTAI/LUKA R600 재전사 프린터용 CardPrinter 구현.

기존 R600Printer의 인쇄 흐름(양면, R600DrawWaterMark 화이트 마스크, 도련 제거, 위치 조정,
면별 방향)을 같은 호출 순서로 다시 쓰고, UV 레이어와 인쇄 전 마그네틱 기록을 선택적으로 더한다.
"""

from __future__ import annotations

import ctypes
import logging
import os
import re
import time
from typing import Any, Dict, Mapping, Optional, Set, Tuple, Union

from .base import (
    CAP_CARD_MOVE, CAP_DUAL_SIDE, CAP_K_SPLIT, CAP_MAGNETIC, CAP_MIFARE, CAP_POSITION_OFFSET,
    CAP_RIBBON_LEVEL, CAP_STATUS, CAP_UV, CAP_WHITE, FAMILY_RTAI, ORIENTATION_PORTRAIT,
    STAGE_BACK_IMAGE, STAGE_DONE, STAGE_EJECT, STAGE_FEED, STAGE_FRONT_IMAGE, STAGE_MAGNETIC,
    STAGE_PRINT, STAGE_SETUP, STAGE_UNKNOWN, STAGE_WAIT, UNIT_UNKNOWN,
    CardJob, CardPrinter, MagneticTracks, PrinterDevice, PrinterError, PrinterStatus,
    ProgressCallback, SideImages,
)
from .exceptions import R600PrinterError
from .r600_extras import (
    MAG_COERCIVITY_HIGH, MAG_DEFAULT_TIMEOUT, MAG_POSITION_STANDARD, UV_DEFAULT_THICKNESS,
    UV_DEFAULT_THRESHOLD,
)

log = logging.getLogger(__name__)

_CODE_PATTERN = re.compile(r"(?:오류 코드|실패:)\s*(-?\d+)")

# SDK가 주는 영문/중문 문구에서 원인을 짚어 한국어 안내로 바꾼다: (키워드들, 안내, 재시도 가능)
_HINTS: Tuple[Tuple[Tuple[str, ...], str, bool], ...] = (
    # 실기에서 확인한 문구(배출할 카드가 없을 때). '공급함이 비었다'와 다른 상황이므로 먼저 걸러 낸다.
    (("no card in the printer", "no card inside"),
     "프린터 안에 카드가 없습니다. 꺼낼 카드가 없거나, 카드가 센서에 잡히지 않는 자리에 걸려 있을 수 있습니다.", True),
    (("feeder", "hopper", "no card", "card empty", "out of card", "卡盒", "无卡", "缺卡"),
     "카드 공급함이 비어 있습니다. 카드를 넣어 주세요.", True),
    (("ribbon", "色带"), "리본을 확인해 주세요. 다 썼다면 교체가 필요합니다.", True),
    (("film", "膜"), "재전사 필름을 확인해 주세요. 다 썼다면 교체가 필요합니다.", True),
    (("cover", "door", "盖"), "프린터 덮개가 열려 있습니다. 덮개를 닫아 주세요.", True),
    (("jam", "stuck", "堵", "卡住"), "카드가 걸렸습니다. 프린터 안의 카드를 꺼내 주세요.", True),
    (("timeout", "time out", "超时"), "프린터가 응답하지 않습니다. 전원과 케이블을 확인해 주세요.", True),
    (("temperature", "overheat", "heat", "温"), "프린터 온도 문제입니다. 잠시 후 다시 시도해 주세요.", True),
    (("not found", "offline", "disconnect", "connect", "连接", "脱机"),
     "프린터와 연결되어 있지 않습니다. 전원과 케이블을 확인해 주세요.", True),
    (("busy", "忙"), "프린터가 다른 작업을 하고 있습니다. 잠시 후 다시 시도해 주세요.", True),
)


# 실기(LUCA-40KM)에서 확인한 값. 상태 값은 0이 아니라 '정상'을 뜻하는 고유 번호로 온다.
#   주 상태 1004 = Ready, 오류 상태 3001 = No Error, 경고 상태 2002 = Heating
# R600GetErrorInfo 계열은 이런 상태 값에 대해 "그런 오류 코드 없음"(중국어)을 돌려준다.
_NO_SUCH_CODE_TEXTS = frozenset({"没有该错误码"})
_NO_ERROR_TEXTS = frozenset({"no error", "no warning", "none", "normal", "ok", ""})
_NO_ERROR_CODES = frozenset({0, 3001})
_READY_TEXTS = frozenset({"ready", "idle", "standby"})
_STATE_KOREAN = {
    "heating": "예열 중입니다. 잠시 뒤 인쇄할 수 있습니다.",
    "printing": "인쇄 중입니다.",
    "busy": "프린터가 다른 작업을 하고 있습니다.",
    "initializing": "프린터를 준비하는 중입니다.",
    "cleaning": "청소 중입니다.",
    "sleep": "절전 상태입니다.",
}


def _korean_state(text: str) -> str:
    lowered = text.lower()
    for key, sentence in _STATE_KOREAN.items():
        if key in lowered:
            return sentence
    return ""


def readable_sdk_text(sdk_text: Optional[str]) -> str:
    """사용자에게 보여 줘도 되는 SDK 문구만 돌려준다.

    SDK 문구는 영어일 때도 있고 중국어(GBK)일 때도 있다. 중국어나 깨진 글자는 보여 줘도 도움이 되지
    않으므로 영문·숫자·기본 기호로만 된 문구만 보여 주고, 나머지는 빈 문자열로 돌려준다(코드로 대신 안내).
    """
    text = (sdk_text or "").strip()
    if not text or any(ord(ch) < 32 or ord(ch) > 126 for ch in text):
        return ""
    return text


def hint_for(sdk_text: str) -> Optional[Tuple[str, bool]]:
    lowered = sdk_text.lower()
    for keywords, hint, recoverable in _HINTS:
        if any(keyword in lowered for keyword in keywords):
            return hint, recoverable
    return None


def code_from_error(error: Exception) -> Optional[int]:
    code = getattr(error, "code", None)
    if isinstance(code, int):
        return code
    match = _CODE_PATTERN.search(str(error))
    return int(match.group(1)) & 0xFFFFFFFF if match else None


class RtaiAdapter(CardPrinter):
    family = FAMILY_RTAI

    CAPABILITIES = frozenset({
        CAP_DUAL_SIDE, CAP_WHITE, CAP_UV, CAP_MAGNETIC, CAP_STATUS, CAP_RIBBON_LEVEL,
        CAP_K_SPLIT, CAP_MIFARE, CAP_CARD_MOVE, CAP_POSITION_OFFSET,
    })

    def __init__(self, lib: Any = None, dll_path: Optional[str] = None,
                 settle_seconds: float = 1.0, timeout_ms: int = 15000):
        """lib에 가짜(또는 이미 로드한) 라이브러리를 넘기면 DLL을 찾지 않는다."""
        super().__init__()
        self._lib = lib
        self._dll_path = dll_path
        self._printer = None
        self.settle_seconds = settle_seconds
        self.timeout_ms = timeout_ms

    # -- 연결 -------------------------------------------------------------
    def capabilities(self) -> Set[str]:
        return set(self.CAPABILITIES)

    @property
    def is_connected(self) -> bool:
        return self._printer is not None and bool(self._printer.is_initialized)

    @property
    def sdk(self):
        """밑에 있는 R600Printer. Mifare 등 어댑터가 감싸지 않은 기능을 직접 쓸 때."""
        self._require_connected()
        return self._printer

    def connect(self, device: PrinterDevice) -> None:
        from .printer_discovery import PrinterInfo
        from .r600_printer import R600Printer

        if self.is_connected:
            self.close()
        name = device.address or device.name
        info = PrinterInfo(name, "TCP" if device.connection in ("tcp", "net") else "USB")
        try:
            if self._lib is not None:
                printer = R600Printer.from_library(self._lib, selected_printer=info)
            else:
                printer = R600Printer(self._resolve_dll_path(), selected_printer=info)
        except PrinterError:
            raise
        except Exception as exc:
            raise PrinterError("RTAI 프린터를 초기화하지 못했습니다.", code=code_from_error(exc),
                               recoverable=True, detail=str(exc)) from exc
        self._printer = printer
        try:
            if printer.selected_printer_name != info.name:
                printer.select_printer(info.name)
            printer.set_timeout(self.timeout_ms)
        except R600PrinterError as exc:
            error = self._translate(exc)
            self._shutdown()
            raise PrinterError(f"프린터 '{name}'에 연결하지 못했습니다. 전원과 케이블을 확인해 주세요.",
                               code=error.code, recoverable=True, detail=error.message) from exc
        self.device = device

    def _resolve_dll_path(self) -> str:
        if self._dll_path:
            return self._dll_path
        from .printer_utils import find_printer_dll
        path = find_printer_dll()
        if not path:
            raise PrinterError("RTAI 프린터 라이브러리(libDSRetransfer600App.dll)를 찾을 수 없습니다.")
        self._dll_path = path
        return path

    def close(self) -> None:
        self._shutdown()
        self.device = None

    def _shutdown(self) -> None:
        printer, self._printer = self._printer, None
        if printer is None:
            return
        try:
            printer.close_library()
        except Exception as exc:
            log.warning("RTAI 라이브러리 정리 중 오류: %s", exc)

    # -- 오류 변환 ---------------------------------------------------------
    def describe_code(self, code: int) -> str:
        """SDK 문구 함수로 오류 코드 설명을 얻는다. 얻지 못하면 빈 문자열."""
        if self._printer is None:
            return ""
        for getter in (self._printer.get_error_outer_info, self._printer.get_error_info,
                       self._printer.get_check_status_message):
            try:
                text = getter(code)
            except Exception:
                continue
            if text and text.strip() not in _NO_SUCH_CODE_TEXTS:
                return text
        return ""

    def status_text(self, code: int) -> str:
        """상태 값(주 상태, 오류 상태, 경고 상태)의 SDK 영문 문구. 예: 1004 → Ready, 3001 → No Error."""
        if self._printer is None:
            return ""
        try:
            return (self._printer.get_check_status_message(code) or "").strip()
        except Exception:
            return ""

    def _translate(self, error: Exception, operation: Optional[str] = None) -> PrinterError:
        if isinstance(error, PrinterError):
            return error
        text = str(error)
        code = code_from_error(error)
        if operation is None:
            operation = text.split(" 실패", 1)[0].strip() if " 실패" in text else "프린터 작업"
        sdk_text = self.describe_code(code) if code is not None else ""
        hint = hint_for(sdk_text) if sdk_text else None
        parts = [f"{operation}에 실패했습니다."]
        recoverable = False
        if hint:
            parts.append(hint[0])
            recoverable = hint[1]
        shown = readable_sdk_text(sdk_text)
        if shown:
            parts.append(f"[프린터 메시지: {shown}]")
        elif sdk_text and code is not None:
            parts.append(f"(코드 0x{code:X})")
        elif code is None:
            parts.append(f"[{text}]")
        return PrinterError(" ".join(parts), code=code, recoverable=recoverable, detail=text)

    # -- 상태 · 정보 --------------------------------------------------------
    def status(self) -> PrinterStatus:
        self._require_connected()
        printer = self._printer
        try:
            raw = printer.query_status()
        except R600PrinterError as exc:
            error = self._translate(exc)
            return PrinterStatus(ready=False, message=error.message, code=error.code)
        code = raw["error_status"]
        error_text = self.status_text(code)
        if error_text:
            has_error = error_text.lower() not in _NO_ERROR_TEXTS
        else:
            has_error = code not in _NO_ERROR_CODES
        status = PrinterStatus(ready=not has_error, message="인쇄할 수 있습니다.",
                               code=code if has_error else None, details=dict(raw))
        status.details["error_text"] = error_text
        if has_error:
            sdk_text = error_text or self.describe_code(code)
            hint = hint_for(sdk_text) if sdk_text else None
            shown = readable_sdk_text(sdk_text)
            status.message = (hint[0] if hint else "프린터가 오류 상태입니다.") + (
                f" [프린터 메시지: {shown}]" if shown else f" (코드 0x{code:X})")
        # 주 상태(Ready, Heating 등)와 경고는 인쇄를 막지 않고 알려만 준다.
        main_text = self.status_text(raw["main_status"])
        status.details["main_text"] = main_text
        if main_text and main_text.lower() not in _READY_TEXTS and not has_error:
            status.busy = "print" in main_text.lower() or "busy" in main_text.lower()
            note = _korean_state(main_text)
            if note:
                status.warnings.append(note)
        warning_text = self.status_text(raw["warning_status"])
        status.details["warning_text"] = warning_text
        if warning_text and warning_text.lower() not in _NO_ERROR_TEXTS:
            note = _korean_state(warning_text) or readable_sdk_text(warning_text)
            if note and note not in status.warnings:
                status.warnings.append(note)

        def probe(call):
            try:
                return call()
            except R600PrinterError as exc:
                log.debug("RTAI 상태 항목 조회 실패: %s", exc)
                return None

        # 공급함 조회는 기종·환경에 따라 SDK 안에서 죽기도 한다(그때는 None: 모름으로 두고 인쇄를 막지 않는다).
        feeder = probe(printer.is_feeder_not_empty)
        if feeder is not None:
            status.feeder_empty = not feeder
            if status.feeder_empty:
                status.ready = False
                if not has_error:
                    status.message = "카드 공급함이 비어 있습니다. 카드를 넣어 주세요."
        status.card_inside = probe(printer.has_card_inside)
        remaining = probe(printer.get_ribbon_and_film_remaining)
        if remaining is not None:
            # SDK 값 그대로. 장수인지 %인지 실기로 확인하지 못했으므로 단위는 "모름"으로 둔다.
            status.ribbon_remaining, status.film_remaining = remaining
            status.ribbon_unit = status.film_unit = UNIT_UNKNOWN
        settings = probe(printer.read_ribbon_settings)
        if settings is not None:
            status.ribbon_type = settings["ribbon_type"]
            status.details["ribbon_settings"] = settings
        return status

    def info(self) -> Dict[str, Any]:
        self._require_connected()
        printer = self._printer
        result: Dict[str, Any] = {}
        for key, call in (("manufacturer", printer.get_manufacturer_info),
                          ("counters", printer.get_printer_info),
                          ("options", printer.get_option_status),
                          ("serial", printer.get_serial_number)):
            try:
                result[key] = call()
            except R600PrinterError as exc:
                log.debug("RTAI 정보 항목 조회 실패(%s): %s", key, exc)
        return result

    # -- 카드 이동 ---------------------------------------------------------
    def inject_card(self) -> None:
        self._require_connected()
        self._guard(self._printer.inject_card)

    def eject_card(self) -> None:
        self._require_connected()
        self._guard(self._printer.eject_card)

    def _guard(self, call, *args, **kwargs):
        try:
            return call(*args, **kwargs)
        except R600PrinterError as exc:
            raise self._translate(exc) from exc

    def _eject_quietly(self) -> None:
        try:
            self._printer.eject_card()
        except Exception as exc:
            log.warning("오류 후 카드 배출 실패: %s", exc)

    # -- 마그네틱 ---------------------------------------------------------
    def write_magnetic(self, tracks: Union[MagneticTracks, Mapping[str, Any], tuple, list],
                       verify: bool = True, options: Optional[Mapping[str, Any]] = None) -> None:
        """카드 한 장을 넣어 마그네틱을 기록하고 배출한다."""
        self._require_connected()
        tracks = MagneticTracks.coerce(tracks)
        if tracks.is_empty():
            raise PrinterError("기록할 마그네틱 데이터가 없습니다.")
        merged = dict(options or {})
        merged["magnetic_verify"] = verify
        try:
            self._printer.inject_card()
            self._encode_magnetic(tracks, merged)
            self._printer.eject_card()
        except (R600PrinterError, PrinterError) as exc:
            self._eject_quietly()
            raise self._translate(exc) from exc

    def read_magnetic(self, options: Optional[Mapping[str, Any]] = None) -> MagneticTracks:
        """카드 한 장을 넣어 마그네틱을 읽고 배출한다."""
        self._require_connected()
        options = options or {}
        try:
            self._printer.inject_card()
            position = options.get("mag_position", MAG_POSITION_STANDARD)
            if position is not None:
                self._printer.move_card_impr(position)
            raw = self._printer.mag_read(options.get("mag_timeout", MAG_DEFAULT_TIMEOUT))
            self._printer.eject_card()
        except R600PrinterError as exc:
            self._eject_quietly()
            raise self._translate(exc) from exc
        return MagneticTracks(*(data.decode("ascii", "replace") or None for data in raw))

    def _encode_magnetic(self, tracks: MagneticTracks, options: Mapping[str, Any]) -> None:
        printer = self._printer
        position = options.get("mag_position", MAG_POSITION_STANDARD)
        timeout = options.get("mag_timeout", MAG_DEFAULT_TIMEOUT)
        if position is not None:
            printer.move_card_impr(position)
        high = options.get("magnetic_coercivity", "high") != "low"
        try:
            printer.set_mag_option(MAG_COERCIVITY_HIGH if high else 0)
        except R600PrinterError as exc:
            log.warning("마그네틱 보자력 설정 실패(계속 진행): %s", exc)
        data = [value.encode("ascii") if value else None for value in tracks.as_tuple()]
        printer.mag_write(*data, timeout=timeout)
        if options.get("magnetic_verify", True):
            read_back = printer.mag_read(timeout)
            for number, (written, read) in enumerate(zip(data, read_back), start=1):
                if written and written != read:
                    raise PrinterError(
                        f"마그네틱 검증에 실패했습니다. (트랙 {number}에 기록한 내용과 읽은 내용이 다릅니다)",
                        recoverable=True)
        return_position = options.get("mag_return_position")
        if return_position is not None:
            printer.move_card(return_position)

    # -- 인쇄 -------------------------------------------------------------
    def print_card(self, job: CardJob, progress: Optional[ProgressCallback] = None) -> None:
        self._require_connected()
        job.validate()
        self._check_files(job)
        for copy in range(1, job.copies + 1):
            def report(percent: int, message: str, stage: str = STAGE_UNKNOWN,
                       side: Optional[str] = None, _copy=copy) -> None:
                overall = int(((_copy - 1) * 100 + percent) / job.copies)
                prefix = f"[{_copy}/{job.copies}] " if job.copies > 1 else ""
                self._report(overall, prefix + message, progress, stage=stage, side=side,
                             copy=_copy, copies=job.copies)
            self._print_one(job, report)
            if copy < job.copies and self.settle_seconds:
                time.sleep(self.settle_seconds)

    @staticmethod
    def _check_files(job: CardJob) -> None:
        for label, side in (("앞면", job.front), ("뒷면", job.back)):
            if side is None:
                continue
            for kind, path in (("이미지", side.color_path), ("화이트 마스크", side.white_mask_path),
                               ("UV 이미지", side.uv_path)):
                if kind == "이미지" and not path:
                    raise PrinterError(f"{label} 이미지가 지정되지 않았습니다.")
                if path and not os.path.exists(path):
                    raise PrinterError(f"{label} {kind} 파일을 찾을 수 없습니다: {path}")

    def _print_one(self, job: CardJob, report) -> None:
        """카드 한 장. 알릴 수 있는 단계는 우리가 SDK를 부르는 경계뿐이다.

        R600PrintDraw 한 번이 앞면 인쇄·뒤집기·뒷면 인쇄·재전사를 모두 하고 끝나야 돌아온다.
        그 사이의 패널(Y/M/C/K/W)·면 진행을 알려 주는 SDK 함수는 없으므로 STAGE_PRINT 하나로만
        알리고 panel·fraction은 채우지 않는다.
        """
        printer = self._printer
        options = job.options
        dual = job.is_dual_side
        first = job.front or job.back
        tracks = MagneticTracks.coerce(options.get("magnetic"))
        # 이미 칼선 크기로 만든 카드면 드라이버의 비율 기반 도련 제거를 이 카드 동안만 끈다.
        printer.bleed_crop_enabled = not options.get("bleed_handled")
        try:
            printer.set_position_offset(options.get("offset_x_mm", 0.0),
                                        options.get("offset_y_mm", 0.0))
            if options.get("precheck"):
                self._precheck()
            report(10 if dual else 15, "카드 삽입 중...", STAGE_FEED)
            printer.inject_card()
            if not tracks.is_empty():
                report(12 if dual else 18, "마그네틱 기록 중...", STAGE_MAGNETIC)
                self._encode_magnetic(tracks, options)
            report(15 if dual else 25, "리본 설정 중...", STAGE_SETUP)
            if options.get("k_split") is not None:
                printer.set_k_split_print(bool(options["k_split"]) and dual)
            printer.set_ribbon_option(ribbon_type=1, key=0, value="2")

            report(30 if dual else 45, "앞면 준비 중..." if dual else "이미지 준비 중...",
                   STAGE_FRONT_IMAGE, "front")
            front_info = self._prepare_side(first, is_back=False, options=options)
            back_info = None
            if dual:
                report(45, "뒷면 준비 중...", STAGE_BACK_IMAGE, "back")
                back_info = self._prepare_side(job.back, is_back=True, options=options)

            report(70, "인쇄 실행 중...", STAGE_PRINT)
            result = printer.lib.R600PrintDraw(
                front_info.encode("cp949"),
                back_info.encode("cp949") if back_info else ctypes.c_char_p(None))
            printer._check_result(result, "양면 인쇄 실행" if dual else "단면 인쇄 실행")

            report(85, "인쇄 완료 대기...", STAGE_WAIT)
            time.sleep(self.settle_seconds)
            report(95, "카드 배출 중...", STAGE_EJECT)
            printer.eject_card()
            time.sleep(self.settle_seconds)
            report(100, "완료", STAGE_DONE)
        except (R600PrinterError, PrinterError) as exc:
            self._eject_quietly()
            raise self._translate(exc) from exc
        finally:
            printer.bleed_crop_enabled = True
            printer.cleanup_layer_files()

    def _precheck(self) -> None:
        status = self.status()
        if not status.ready:
            raise PrinterError(status.message, code=status.code, recoverable=True)

    def _prepare_side(self, side: SideImages, is_back: bool, options: Mapping[str, Any]) -> str:
        """한 면의 캔버스를 만든다. 호출 순서는 R600Printer.prepare_front/back_canvas와 같다."""
        printer = self._printer
        portrait = side.orientation == ORIENTATION_PORTRAIT
        rotation = 180 if (is_back and portrait) else 0
        width, height = printer.get_card_dimensions(side.orientation)
        x, y = printer.offset_x, printer.offset_y
        if is_back:
            printer.clear_canvas()
            printer.setup_canvas(side.orientation, rotation)
        else:
            printer.setup_canvas(side.orientation)
        if side.white_mask_path:
            if rotation:
                printer.draw_watermark_rotated(x, y, width, height, side.white_mask_path, rotation)
            else:
                printer.draw_watermark(x, y, width, height, side.white_mask_path)
        printer.draw_image(x, y, width, height, side.color_path)
        if side.uv_path:
            printer.draw_uv_layer(x, y, width, height, side.uv_path,
                                  threshold=options.get("uv_threshold", UV_DEFAULT_THRESHOLD),
                                  thickness=options.get("uv_thickness", UV_DEFAULT_THICKNESS),
                                  rotation=rotation)
        info = printer.commit_canvas()
        if is_back:
            printer.back_img_info = info
        else:
            printer.front_img_info = info
        return info
