"""
printer/r600_extras.py
R600 SDK 중 기존 R600Printer가 쓰지 않던 함수의 바인딩.

인자 자료형은 claudedocs/r600-sdk-reference.md 2장(DLL에서 복원한 원형)을 따른다.
값의 의미(위치 번호, 모드 번호 등)는 실기 확인 전이므로 모두 인자로 받고 기본값만 문서의
관찰값으로 둔다. 시그니처는 처음 호출할 때 설정하므로 DLL에 없는 함수가 있어도 기존 동작에는
영향이 없다.
"""

from __future__ import annotations

import ctypes
import io
import os
from typing import Any, Dict, Optional, Tuple

from .exceptions import R600PrinterError

_P = ctypes.POINTER
_u8, _u16, _u32 = ctypes.c_ubyte, ctypes.c_ushort, ctypes.c_uint
_i16, _i32 = ctypes.c_short, ctypes.c_int
_str = ctypes.c_char_p
_dbl = ctypes.c_double

EXTRA_SIGNATURES: Dict[str, Tuple[list, Any]] = {
    "R600UsbSetTimeout": ([_i32, _i32], _u32),
    "R600QueryPrtStatus": ([_P(_i16)] * 3 + [_P(_u32)] * 4 + [_P(_u8)] * 2, _u32),
    "R600GetErrorInfo": ([_u32, _str, _P(_i32)], _u8),
    "R600GetErrorOuterInfo": ([_u32, _str, _P(_i32)], _u8),
    "R600GetCheckStatusMsg": ([_u32, _str, _P(_i32)], _u32),
    "R600IsFeederNoEmpty": ([_P(_i32)], _u32),
    "R600IsPrtHaveCard": ([_P(_u8)], _u32),
    "R600GetCardPos": ([_P(_i32)], _u32),
    "R600GetPrtInfo": ([_P(_u32)] * 5, _u32),
    "R600GetPrtManufInfo": ([_str, _P(_i32)] * 5, _u32),
    "R600GetSN": ([ctypes.c_char, _str, _P(_i32)], _u32),
    "R600GetPrtOptionStatus": ([_i32] + [_P(_u8)] * 7, _u32),
    "R600GetRbnAndFilmRemaining": ([_P(_u16), _P(_u16)], _u32),
    "R600RibbonSettingsRW": ([_i32] + [_P(_u8)] * 4, _u32),
    "R600SetLayerFluorThreshold": ([_i32], _u32),
    "R600DrawLayerFluor": ([_dbl, _dbl, _dbl, _dbl, _str], _u32),
    "R600SetAddImageMode_Rtai": ([_i32, ctypes.c_bool, ctypes.c_bool, _str, _i32], _u32),
    "R600SetImgUtilIsNeedSet_K_ColorRibbonSplitPrint": ([ctypes.c_bool], _u32),
    "R600CardMove": ([_u8], _u32),
    "R600CardMoveImpr": ([_u8], _u32),
    "R600SetMagCardOpt": ([_u8], _u32),
    "R600MAGCardWrite": ([_u8, _P(_u8), _u32, _P(_u8), _u32, _P(_u8), _u32], _u32),
    "R600MAGCardRead": ([_u8, _P(_u8), _P(_u32), _P(_u8), _P(_u32), _P(_u8), _P(_u32)], _u32),
    "R600M1CardGetSN": ([_u8, _P(_u8)], _u32),
    "R600M1CardPwdAuth": ([_i32, _u8, _u8, _str], _u32),
    "R600M1CardRead": ([_i32, _u8, _str], _u32),
    "R600M1CardWrite": ([_i32, _u8, _str], _u32),
    "R600M1CardHalt": ([], _u32),
    "R600PrtReset": ([], _u32),
}

MAG_TRACK_BUFFER = 128          # 참조 제품이 트랙마다 쓰는 버퍼 크기
MAG_DEFAULT_TIMEOUT = 10
MAG_COERCIVITY_HIGH = 1
MAG_POSITION_STANDARD = 6       # 3-in-1 인코더가 아닌 구성
MAG_POSITION_3IN1 = 5
UV_DEFAULT_THRESHOLD = 220
UV_DEFAULT_THICKNESS = "180"
RIBBON_KEY_THICKNESS = 1
SETTINGS_READ = 0


def decode_sdk_text(raw: bytes) -> str:
    raw = raw.split(b"\0", 1)[0]
    for encoding in ("utf-8", "cp949", "gbk"):
        try:
            return raw.decode(encoding).strip()
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1").strip()


class R600ExtrasMixin:
    """R600Printer에 섞어 쓰는 확장 메서드 모음. self.lib(ctypes 라이브러리)만 요구한다."""

    lib: Any

    # -- 내부 도우미 -------------------------------------------------------
    def _ext(self, name: str):
        configured = self.__dict__.setdefault("_ext_configured", set())
        try:
            func = getattr(self.lib, name)
        except AttributeError:
            raise R600PrinterError(f"SDK에 {name} 함수가 없습니다.") from None
        if name not in configured:
            argtypes, restype = EXTRA_SIGNATURES[name]
            func.argtypes = argtypes
            func.restype = restype
            configured.add(name)
        return func

    @staticmethod
    def _ext_check(result: int, operation: str) -> None:
        if result != 0:
            error = R600PrinterError(f"{operation} 실패: 오류 코드 {result}")
            error.code = result
            raise error

    def _ext_text(self, name: str, code: int, size: int = 512) -> str:
        buffer = ctypes.create_string_buffer(size)
        length = ctypes.c_int(size)
        self._ext(name)(code, buffer, ctypes.byref(length))
        return decode_sdk_text(buffer.raw)

    # -- 상태 -------------------------------------------------------------
    def query_status(self) -> Dict[str, int]:
        """R600QueryPrtStatus: 온도 3종과 주/부/오류/경고 상태 값을 그대로 돌려준다."""
        temps = [ctypes.c_short() for _ in range(3)]
        words = [ctypes.c_uint() for _ in range(4)]
        codes = [ctypes.c_ubyte() for _ in range(2)]
        ret = self._ext("R600QueryPrtStatus")(*[ctypes.byref(v) for v in temps + words + codes])
        self._ext_check(ret, "프린터 상태 조회")
        keys = ("chassis_temp", "printhead_temp", "heater_temp", "main_status", "sub_status",
                "error_status", "warning_status", "main_code", "sub_code")
        return {key: var.value for key, var in zip(keys, temps + words + codes)}

    def get_error_info(self, error_code: int) -> str:
        """R600GetErrorInfo: 오류 코드에 대한 SDK 설명 문구. 없으면 빈 문자열."""
        return self._ext_text("R600GetErrorInfo", error_code)

    def get_error_outer_info(self, error_code: int) -> str:
        """R600GetErrorOuterInfo: 사용자용 오류 문구. 없으면 빈 문자열."""
        return self._ext_text("R600GetErrorOuterInfo", error_code)

    def get_check_status_message(self, status_code: int) -> str:
        """R600GetCheckStatusMsg: 상태 코드에 대한 SDK 문구. 없으면 빈 문자열."""
        return self._ext_text("R600GetCheckStatusMsg", status_code)

    def is_feeder_not_empty(self) -> bool:
        """R600IsFeederNoEmpty: 카드 공급함에 카드가 있으면 True."""
        flag = ctypes.c_int()
        self._ext_check(self._ext("R600IsFeederNoEmpty")(ctypes.byref(flag)), "카드 공급함 확인")
        return flag.value != 0

    def has_card_inside(self) -> bool:
        """R600IsPrtHaveCard: 프린터 안에 카드가 있으면 True."""
        flag = ctypes.c_ubyte()
        self._ext_check(self._ext("R600IsPrtHaveCard")(ctypes.byref(flag)), "내부 카드 확인")
        return flag.value != 0

    def get_card_position(self) -> int:
        """R600GetCardPos: 카드 위치 번호(의미는 실기 확인 필요)."""
        pos = ctypes.c_int()
        self._ext_check(self._ext("R600GetCardPos")(ctypes.byref(pos)), "카드 위치 조회")
        return pos.value

    # -- 정보 -------------------------------------------------------------
    def get_printer_info(self) -> Dict[str, int]:
        """R600GetPrtInfo: 헤드·카드 인쇄 횟수와 펌웨어 버전 값."""
        values = [ctypes.c_uint() for _ in range(5)]
        ret = self._ext("R600GetPrtInfo")(*[ctypes.byref(v) for v in values])
        self._ext_check(ret, "프린터 정보 조회")
        keys = ("printhead_print_count", "card_print_count", "main_fw_version",
                "sub_fw_version", "fpga_version")
        return {key: var.value for key, var in zip(keys, values)}

    def get_manufacturer_info(self) -> Dict[str, str]:
        """R600GetPrtManufInfo: 모델명·제조사·시리얼·제조일·펌웨어 문자열."""
        size = 128
        buffers = [ctypes.create_string_buffer(size) for _ in range(5)]
        lengths = [ctypes.c_int(size) for _ in range(5)]
        args = []
        for buffer, length in zip(buffers, lengths):
            args.extend((buffer, ctypes.byref(length)))
        self._ext_check(self._ext("R600GetPrtManufInfo")(*args), "제조 정보 조회")
        keys = ("name", "manufacturer", "serial", "date", "main_fw_version")
        return {key: decode_sdk_text(buffer.raw) for key, buffer in zip(keys, buffers)}

    def get_serial_number(self, device: int = 0) -> str:
        """R600GetSN: 시리얼 번호. device 인자의 의미는 실기 확인 필요(기본 0)."""
        buffer = ctypes.create_string_buffer(128)
        length = ctypes.c_int(128)
        ret = self._ext("R600GetSN")(bytes([device & 0xFF]), buffer, ctypes.byref(length))
        self._ext_check(ret, "시리얼 번호 조회")
        return decode_sdk_text(buffer.raw)

    def get_option_status(self, mode: int = 0) -> Dict[str, int]:
        """R600GetPrtOptionStatus: 인코더·라미네이터 등 옵션 장착 상태 값."""
        values = [ctypes.c_ubyte() for _ in range(7)]
        ret = self._ext("R600GetPrtOptionStatus")(mode, *[ctypes.byref(v) for v in values])
        self._ext_check(ret, "옵션 상태 조회")
        keys = ("reader_3in1", "contact_rw", "contactless_rw", "external_box",
                "magnetic_reader", "laminator_left", "laminator_right")
        return {key: var.value for key, var in zip(keys, values)}

    # -- 소모품 -----------------------------------------------------------
    def get_ribbon_and_film_remaining(self) -> Tuple[int, int]:
        """R600GetRbnAndFilmRemaining: (리본 잔량, 필름 잔량). 단위는 SDK 값 그대로."""
        ribbon, film = ctypes.c_ushort(), ctypes.c_ushort()
        ret = self._ext("R600GetRbnAndFilmRemaining")(ctypes.byref(ribbon), ctypes.byref(film))
        self._ext_check(ret, "리본·필름 잔량 조회")
        return ribbon.value, film.value

    def read_ribbon_settings(self) -> Dict[str, int]:
        """R600RibbonSettingsRW(읽기): 리본·필름 종류 코드와 부족 경고 기준."""
        values = [ctypes.c_ubyte() for _ in range(4)]
        ret = self._ext("R600RibbonSettingsRW")(SETTINGS_READ, *[ctypes.byref(v) for v in values])
        self._ext_check(ret, "리본 설정 조회")
        keys = ("ribbon_type", "film_type", "ribbon_near_end", "film_near_end")
        return {key: var.value for key, var in zip(keys, values)}

    # -- UV 레이어 · K 분할 -------------------------------------------------
    def set_fluor_threshold(self, threshold: int = UV_DEFAULT_THRESHOLD) -> None:
        """R600SetLayerFluorThreshold: UV 레이어 이진화 임계값."""
        self._ext_check(self._ext("R600SetLayerFluorThreshold")(int(threshold)), "UV 임계값 설정")

    def set_ribbon_option_raw(self, is_write: int, key: int, value: str,
                              value_len: Optional[int] = None) -> None:
        """R600SetRibbonOpt를 길이까지 지정해 호출한다. 길이 기본값은 끝의 NUL을 포함한다."""
        data = value.encode("cp949")
        length = len(data) + 1 if value_len is None else value_len
        self._ext_check(self.lib.R600SetRibbonOpt(is_write, key, data, length), "리본 옵션 설정")

    def draw_layer_fluor(self, x: float, y: float, width: float, height: float,
                         image_path: str, rotation: int = 0) -> None:
        """R600DrawLayerFluor: UV(형광) 레이어 이미지를 캔버스에 올린다. 도련은 잘라 낸다
        (bleed_crop_enabled가 False면 자르지 않는다)."""
        prepared = self._prepare_layer_file(image_path, rotation, "uv_layer")
        ret = self._ext("R600DrawLayerFluor")(x, y, width, height, prepared.encode("cp949"))
        self._ext_check(ret, "UV 레이어 그리기")

    def draw_uv_layer(self, x: float, y: float, width: float, height: float, image_path: str,
                      threshold: int = UV_DEFAULT_THRESHOLD,
                      thickness: Optional[str] = UV_DEFAULT_THICKNESS,
                      rotation: int = 0) -> None:
        """UV 레이어 한 장: 임계값 → 두께(리본 옵션 key 1) → 그리기 순서로 호출한다."""
        self.set_fluor_threshold(threshold)
        if thickness:
            self.set_ribbon_option_raw(1, RIBBON_KEY_THICKNESS, str(thickness))
        self.draw_layer_fluor(x, y, width, height, image_path, rotation)

    def set_add_image_mode_rtai(self, mode: int, ymc: bool, k: bool,
                                image_path: Optional[str] = None,
                                threshold: int = UV_DEFAULT_THRESHOLD) -> None:
        """R600SetAddImageMode_Rtai: 다음에 그릴 이미지의 레이어·이진화 방식 지정."""
        path = image_path.encode("cp949") if image_path else None
        ret = self._ext("R600SetAddImageMode_Rtai")(mode, bool(ymc), bool(k), path, threshold)
        self._ext_check(ret, "이미지 추가 모드 설정")

    def set_k_split_print(self, enabled: bool) -> None:
        """R600SetImgUtilIsNeedSet_K_ColorRibbonSplitPrint: K 패널 분할 인쇄 여부."""
        ret = self._ext("R600SetImgUtilIsNeedSet_K_ColorRibbonSplitPrint")(bool(enabled))
        self._ext_check(ret, "K 패널 분할 설정")

    # -- 카드 이동 · 유지보수 ------------------------------------------------
    def move_card(self, position: int) -> None:
        """R600CardMove: 카드를 지정 위치로 이동."""
        self._ext_check(self._ext("R600CardMove")(position), f"카드 이동 ({position})")

    def move_card_impr(self, position: int) -> None:
        """R600CardMoveImpr: 인코딩 위치로 카드 이동(마그네틱 6, 3-in-1 인코더면 5)."""
        self._ext_check(self._ext("R600CardMoveImpr")(position), f"카드 인코딩 위치 이동 ({position})")

    def reset_printer(self) -> None:
        """R600PrtReset: 프린터 초기화."""
        self._ext_check(self._ext("R600PrtReset")(), "프린터 초기화")

    def set_usb_timeout(self, read_ms: int = 10000, write_ms: Optional[int] = None) -> None:
        """R600UsbSetTimeout: USB 읽기/쓰기 제한 시간(ms)."""
        ret = self._ext("R600UsbSetTimeout")(read_ms, read_ms if write_ms is None else write_ms)
        self._ext_check(ret, "USB 타임아웃 설정")

    def close_library(self) -> None:
        """카드를 움직이지 않고 SDK만 정리한다(cleanup_and_close는 먼저 배출을 시도한다)."""
        if not getattr(self, "is_initialized", False):
            return
        self.cleanup_layer_files()
        self.cleanup_session_files()
        try:
            self.lib.R600LibClear()
        finally:
            self.selected_printer_name = None
            self.front_img_info = None
            self.back_img_info = None
            self.is_initialized = False

    # -- 마그네틱 ---------------------------------------------------------
    def set_mag_option(self, mode: int = MAG_COERCIVITY_HIGH) -> None:
        """R600SetMagCardOpt: 보자력 설정(1 = 고보자력)."""
        self._ext_check(self._ext("R600SetMagCardOpt")(mode), "마그네틱 옵션 설정")

    def mag_write(self, track1: Optional[bytes] = None, track2: Optional[bytes] = None,
                  track3: Optional[bytes] = None, timeout: int = MAG_DEFAULT_TIMEOUT) -> None:
        """R600MAGCardWrite: ISO 트랙 쓰기. 비울 트랙은 None."""
        args = [timeout]
        for data in (track1, track2, track3):
            data = data or b""
            args.extend(((ctypes.c_ubyte * max(len(data), 1)).from_buffer_copy(data or b"\0"),
                         len(data)))
        self._ext_check(self._ext("R600MAGCardWrite")(*args), "마그네틱 쓰기")

    def mag_read(self, timeout: int = MAG_DEFAULT_TIMEOUT) -> Tuple[bytes, bytes, bytes]:
        """R600MAGCardRead: 세 트랙을 읽어 (트랙1, 트랙2, 트랙3) 바이트로 돌려준다."""
        buffers = [(ctypes.c_ubyte * MAG_TRACK_BUFFER)() for _ in range(3)]
        lengths = [ctypes.c_uint(MAG_TRACK_BUFFER) for _ in range(3)]
        args = [timeout]
        for buffer, length in zip(buffers, lengths):
            args.extend((buffer, ctypes.byref(length)))
        self._ext_check(self._ext("R600MAGCardRead")(*args), "마그네틱 읽기")
        return tuple(bytes(buffer[:min(length.value, MAG_TRACK_BUFFER)]).rstrip(b"\0")
                     for buffer, length in zip(buffers, lengths))

    # -- Mifare(M1) -------------------------------------------------------
    def m1_get_serial(self, mode: int = 0) -> bytes:
        """R600M1CardGetSN: 카드 UID. mode 값의 의미는 실기 확인 필요."""
        buffer = (ctypes.c_ubyte * 64)()
        self._ext_check(self._ext("R600M1CardGetSN")(mode, buffer), "Mifare 카드 번호 조회")
        return bytes(buffer).rstrip(b"\0")

    def m1_authenticate(self, block: int, password: str, key_type: int = 0,
                        is_hex: bool = True) -> None:
        """R600M1CardPwdAuth: 블록 인증. key_type(A/B 구분) 값은 실기 확인 필요."""
        ret = self._ext("R600M1CardPwdAuth")(1 if is_hex else 0, key_type, block,
                                             password.encode("ascii"))
        self._ext_check(ret, "Mifare 인증")

    def m1_read(self, block: int, is_hex: bool = True) -> str:
        """R600M1CardRead: 블록 읽기(16진 문자열 또는 원문)."""
        buffer = ctypes.create_string_buffer(128)
        self._ext_check(self._ext("R600M1CardRead")(1 if is_hex else 0, block, buffer), "Mifare 읽기")
        return decode_sdk_text(buffer.raw)

    def m1_write(self, block: int, data: str, is_hex: bool = True) -> None:
        """R600M1CardWrite: 블록 쓰기."""
        ret = self._ext("R600M1CardWrite")(1 if is_hex else 0, block, data.encode("ascii"))
        self._ext_check(ret, "Mifare 쓰기")

    def m1_halt(self) -> None:
        """R600M1CardHalt: Mifare 카드 통신 종료."""
        self._ext_check(self._ext("R600M1CardHalt")(), "Mifare 종료")

    # -- 레이어 이미지 준비 -------------------------------------------------
    def _prepare_layer_file(self, image_path: str, rotation: int, prefix: str) -> str:
        """레이어 이미지를 SDK에 넘길 수 있게 정리한다: 도련 제거, RGB 변환, 회전, ASCII 경로."""
        if not image_path or not os.path.exists(image_path):
            raise R600PrinterError(f"이미지 파일 {image_path}이 존재하지 않습니다.")
        from PIL import Image
        from utils.bleed_crop import crop_bleed
        from utils.safe_temp_path import create_safe_temp_file, get_cached_safe_temp_dir

        with open(image_path, "rb") as handle:
            image = Image.open(io.BytesIO(handle.read()))
        image.load()
        if getattr(self, "bleed_crop_enabled", True):
            image = crop_bleed(image)
        image = image.convert("RGB")
        if rotation % 360:
            image = image.rotate(-(rotation % 360), expand=True)
        temp_path = create_safe_temp_file(prefix=prefix, suffix=".jpg",
                                          dir=get_cached_safe_temp_dir())
        image.save(temp_path, "JPEG", quality=95)
        self._remember_temp_file(temp_path)
        return temp_path

    def _remember_temp_file(self, path: str) -> None:
        """카드 한 장이 끝난 뒤 cleanup_layer_files가 지울 임시 파일로 적어 둔다."""
        self.__dict__.setdefault("_ext_temp_files", []).append(path)

    def _remember_session_file(self, path: str) -> None:
        """SDK를 닫을 때(close_library / cleanup_and_close) 지울 임시 파일로 적어 둔다."""
        self.__dict__.setdefault("_session_temp_files", []).append(path)

    def cleanup_session_files(self) -> None:
        """draw_watermark가 만든 임시 파일을 지운다. SDK를 닫을 때 부른다."""
        for path in self.__dict__.pop("_session_temp_files", []):
            try:
                os.remove(path)
            except OSError:
                pass

    def cleanup_layer_files(self) -> None:
        """draw_layer_fluor가 만든 임시 파일을 지운다. 인쇄가 끝난 뒤에 부른다."""
        for path in self.__dict__.pop("_ext_temp_files", []):
            try:
                os.remove(path)
            except OSError:
                pass
