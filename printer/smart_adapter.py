"""
printer/smart_adapter.py
IDP SMART 시리즈용 CardPrinter 구현 (DCL 모드 + SBS 제어).

SMART-30/50/31/51은 카드에 직접 인쇄하는 방식이라 화이트 레이어가 없다. 형광(UV) 패널은
형광 리본(YMCKFO 등)이 장착됐을 때만 지원한다고 보고한다.
"""

from __future__ import annotations

import io
import logging
import os
from typing import Any, Dict, List, Mapping, Optional, Set, Union

from . import smart_printer as sp
from .base import (
    CAP_CARD_MOVE, CAP_DUAL_SIDE, CAP_MAGNETIC, CAP_POSITION_OFFSET, CAP_RIBBON_LEVEL,
    CAP_STATUS, CAP_UV, FAMILY_SMART, ORIENTATION_PORTRAIT, STAGE_DONE, STAGE_EJECT, STAGE_FEED,
    STAGE_MAGNETIC, STAGE_PREPARE, STAGE_PRINT, STAGE_PRINT_BACK, STAGE_PRINT_FRONT, STAGE_SETUP,
    STAGE_UNKNOWN, UNIT_CARDS, UNIT_UNKNOWN, CardJob, CardPrinter,
    MagneticTracks, NotSupported, PrinterDevice, PrinterError, PrinterStatus, ProgressCallback,
    SideImages,
)

log = logging.getLogger(__name__)

PRINT_DPI = 300
MM_PER_INCH = 25.4


def _count_unit(info: "sp.SmartRibbonInfo") -> str:
    """GetRibbonInfo의 remain 단위. SDK는 (최대 횟수, 남은 횟수)를 주므로 보통 카드 장수다.

    최대값이 없거나 정확히 100이거나(% 값과 구별할 수 없다) 잔량이 0~최대값을 벗어나면
    짐작하지 않고 "모름"으로 둔다.
    """
    if info.max > 0 and info.max != 100 and 0 <= info.remain <= info.max:
        return UNIT_CARDS
    return UNIT_UNKNOWN


class SmartAdapter(CardPrinter):
    family = FAMILY_SMART

    def __init__(self, lib: Any = None, dll_path: Optional[str] = None,
                 portrait_rotation: int = 90, back_rotation: int = 0,
                 reopen_per_card: bool = True):
        """lib에 가짜(또는 이미 로드한) 라이브러리를 넘기면 DLL을 찾지 않는다.

        portrait_rotation: 세로형 면을 가로 문서에 올릴 때 돌리는 각도(반시계, 90 또는 270).
        back_rotation: 뒷면에 더 줄 회전(0 또는 180). 둘 다 실기에서 확인해 맞춰야 한다.
        reopen_per_card: 카드마다 장치를 다시 연다(SDK 샘플이 검증한 순서).
        """
        super().__init__()
        self._binding = sp.SmartPrinter(dll_path=dll_path, lib=lib)
        self.portrait_rotation = portrait_rotation
        self.back_rotation = back_rotation
        self.reopen_per_card = reopen_per_card
        self._group = sp.GROUP_SMART51
        self._has_flipper: Optional[bool] = None
        self._has_magnetic: Optional[bool] = None
        self._ribbon_type: Optional[int] = None
        self._cards_since_open = 0

    # -- 연결 -------------------------------------------------------------
    @property
    def sdk(self) -> sp.SmartPrinter:
        return self._binding

    @property
    def is_connected(self) -> bool:
        return self._binding.is_open

    @property
    def group(self) -> int:
        return self._group

    def capabilities(self) -> Set[str]:
        caps = {CAP_STATUS, CAP_RIBBON_LEVEL, CAP_CARD_MOVE, CAP_POSITION_OFFSET}
        if not self.is_connected:
            # 장착 옵션을 아직 모른다. 플리퍼·마그네틱 인코더가 있으면 가능한 기능까지 알린다.
            return caps | {CAP_DUAL_SIDE, CAP_MAGNETIC}
        if self._has_flipper:
            caps.add(CAP_DUAL_SIDE)
        if self._has_magnetic is not False:
            caps.add(CAP_MAGNETIC)
        if self._ribbon_type in sp.FLUORESCENT_RIBBONS:
            caps.add(CAP_UV)
        return caps

    def connect(self, device: PrinterDevice) -> None:
        if self.is_connected:
            self.close()
        address = device.address or device.name
        self._binding.open(address)
        self.device = device
        self._cards_since_open = 0
        extra = device.extra or {}
        self._group = int(extra.get("group") or sp.group_from_pid(int(extra.get("pid") or 0)))
        self._refresh_options()

    def _refresh_options(self) -> None:
        try:
            decoded = sp.decode_status(self._group, self._binding.get_status())
            self._has_flipper = decoded.has_flipper
            self._has_magnetic = decoded.has_magnetic
        except PrinterError as exc:
            log.warning("SMART 상태 조회 실패: %s", exc)
        try:
            self._ribbon_type = self._binding.get_ribbon_info().type
        except PrinterError as exc:
            log.debug("SMART 리본 정보 조회 실패: %s", exc)

    def close(self) -> None:
        try:
            self._binding.close()
        except PrinterError as exc:
            log.warning("SMART 연결 종료 중 오류: %s", exc)
        self.device = None

    def _reopen(self) -> None:
        device = self.device
        try:
            self._binding.close()
        except PrinterError as exc:
            log.warning("SMART 재연결 전 종료 오류: %s", exc)
        self._binding.open(device.address or device.name)
        self._cards_since_open = 0

    # -- 상태 · 정보 --------------------------------------------------------
    def status(self) -> PrinterStatus:
        self._require_connected()
        try:
            decoded = sp.decode_status(self._group, self._binding.get_status())
        except PrinterError as exc:
            return PrinterStatus(ready=False, message=exc.message, code=exc.code)
        self._has_flipper, self._has_magnetic = decoded.has_flipper, decoded.has_magnetic
        status = PrinterStatus(
            ready=not decoded.has_error and not decoded.hopper_empty,
            message="인쇄할 수 있습니다.",
            feeder_empty=decoded.hopper_empty, busy=decoded.busy,
            warnings=list(decoded.warnings),
            details={"status_words": decoded.raw, "group": self._group,
                     "flipper": decoded.has_flipper, "magnetic_encoder": decoded.has_magnetic},
        )
        if decoded.has_error:
            status.message = " ".join(decoded.errors)
            status.code = decoded.raw[1] or (decoded.raw[0] >> 32) or None
        elif decoded.hopper_empty:
            status.message = "카드 공급함(호퍼)이 비어 있습니다. 카드를 넣어 주세요."
        try:
            ribbon = self._binding.get_ribbon_info()
            self._ribbon_type = ribbon.type
            # remain/max는 리본으로 찍을 수 있는 횟수(SDK 샘플 기준, 실기 확인 전).
            status.ribbon_remaining = ribbon.remain
            status.ribbon_unit = _count_unit(ribbon)
            status.ribbon_capacity = ribbon.max or None
            status.ribbon_type = ribbon.type
            status.details["ribbon"] = {"type": ribbon.type, "name": ribbon.type_name,
                                        "max": ribbon.max, "remain": ribbon.remain,
                                        "grade": ribbon.grade}
        except PrinterError as exc:
            log.debug("SMART 리본 정보 조회 실패: %s", exc)
        if self._group == sp.GROUP_SMART81 and self._binding.has_function("SmartComm_GetRibbonInfo2"):
            try:
                film = self._binding.get_ribbon_info2(sp.RI2_RETRANSFER)
                status.film_remaining = film.remain
                status.film_unit = _count_unit(film)
            except PrinterError as exc:
                log.debug("SMART 필름 정보 조회 실패: %s", exc)
        return status

    def info(self) -> Dict[str, Any]:
        self._require_connected()
        result: Dict[str, Any] = {"group": self._group}
        if self.device is not None:
            result["description"] = self.device.address
            result.update({key: self.device.extra[key] for key in ("id", "dev", "pid")
                           if key in (self.device.extra or {})})
        try:
            result.update(self._binding.get_system_info())
        except PrinterError as exc:
            log.debug("SMART 시스템 정보 조회 실패: %s", exc)
        return result

    # -- 카드 이동 ---------------------------------------------------------
    def inject_card(self) -> None:
        self._require_connected()
        self._binding.sbs_start()
        self._binding.card_in()

    def eject_card(self) -> None:
        self._require_connected()
        try:
            self._binding.card_out()
        finally:
            self._end_sbs()

    def _end_sbs(self) -> None:
        try:
            self._binding.sbs_end()
        except PrinterError as exc:
            log.warning("SBS 모드 종료 실패: %s", exc)

    def _eject_quietly(self) -> None:
        try:
            self._binding.card_out()
        except PrinterError as exc:
            log.warning("오류 후 카드 배출 실패: %s", exc)

    # -- 마그네틱 ---------------------------------------------------------
    def write_magnetic(self, tracks: Union[MagneticTracks, Mapping[str, Any], tuple, list],
                       verify: bool = True, options: Optional[Mapping[str, Any]] = None) -> None:
        """카드 한 장을 넣어 마그네틱을 기록하고 배출한다."""
        self._require_connected()
        self._require_magnetic()
        tracks = MagneticTracks.coerce(tracks)
        if tracks.is_empty():
            raise PrinterError("기록할 마그네틱 데이터가 없습니다.")
        merged = dict(options or {})
        merged["magnetic_verify"] = verify
        self._binding.sbs_start()
        try:
            self._binding.card_in()
            self._encode_magnetic(tracks, merged)
            self._binding.card_out()
        except PrinterError:
            self._eject_quietly()
            raise
        finally:
            self._end_sbs()

    def read_magnetic(self) -> MagneticTracks:
        """카드 한 장을 넣어 마그네틱을 읽고 배출한다."""
        self._require_connected()
        self._require_magnetic()
        self._binding.sbs_start()
        try:
            self._binding.card_in()
            self._binding.mag_read()
            raw = self._binding.mag_get_all()
            self._binding.card_out()
        except PrinterError:
            self._eject_quietly()
            raise
        finally:
            self._end_sbs()
        return MagneticTracks(*(data.decode("ascii", "replace") or None for data in raw))

    def _require_magnetic(self) -> None:
        if self._has_magnetic is False:
            raise NotSupported(CAP_MAGNETIC, "이 프린터에는 마그네틱 인코더가 장착되어 있지 않습니다.")

    def _encode_magnetic(self, tracks: MagneticTracks, options: Mapping[str, Any]) -> None:
        binding = self._binding
        data = [value.encode("ascii") if value else None for value in tracks.as_tuple()]
        mask = sum(bit for bit, value in zip((sp.MAG_T1, sp.MAG_T2, sp.MAG_T3), data) if value)
        high = options.get("magnetic_coercivity", "high") != "low"
        if self._group == sp.GROUP_SMART50:
            coercivity = 1 if high else 0          # SMART-50/30은 BOOL bHighCo
        else:
            coercivity = sp.MAG_HICO if high else sp.MAG_LOCO
        binding.mag_set_all(*data)
        binding.mag_write(mask, coercivity)
        if options.get("magnetic_verify", True):
            binding.mag_read(mask)
            read_back = binding.mag_get_all()
            for number, (written, read) in enumerate(zip(data, read_back), start=1):
                if written and written != read:
                    raise PrinterError(
                        f"마그네틱 검증에 실패했습니다. (트랙 {number}에 기록한 내용과 읽은 내용이 다릅니다)",
                        recoverable=True)

    # -- 인쇄 -------------------------------------------------------------
    def print_card(self, job: CardJob, progress: Optional[ProgressCallback] = None) -> None:
        self._require_connected()
        job.validate()
        self._check_job(job)
        for copy in range(1, job.copies + 1):
            def report(percent: int, message: str, stage: str = STAGE_UNKNOWN,
                       side: Optional[str] = None, _copy=copy) -> None:
                overall = int(((_copy - 1) * 100 + percent) / job.copies)
                prefix = f"[{_copy}/{job.copies}] " if job.copies > 1 else ""
                self._report(overall, prefix + message, progress, stage=stage, side=side,
                             copy=_copy, copies=job.copies)
            self._print_one(job, report)

    def _check_job(self, job: CardJob) -> None:
        tracks = MagneticTracks.coerce(job.options.get("magnetic"))
        if not tracks.is_empty():
            self._require_magnetic()
        if job.is_dual_side and self._has_flipper is False:
            raise NotSupported(CAP_DUAL_SIDE, "이 프린터에는 양면 인쇄 장치(플리퍼)가 없습니다.")
        for label, side in (("앞면", job.front), ("뒷면", job.back)):
            if side is None:
                continue
            if not side.color_path or not os.path.exists(side.color_path):
                raise PrinterError(f"{label} 이미지 파일을 찾을 수 없습니다: {side.color_path}")
            if side.uv_path:
                if self._ribbon_type not in sp.FLUORESCENT_RIBBONS:
                    raise NotSupported(CAP_UV, "장착된 리본으로는 UV(형광) 인쇄를 할 수 없습니다.")
                if not os.path.exists(side.uv_path):
                    raise PrinterError(f"{label} UV 이미지 파일을 찾을 수 없습니다: {side.uv_path}")
            if side.white_mask_path:
                log.warning("SMART 프린터에는 화이트 레이어가 없어 %s 화이트 마스크를 쓰지 않습니다.", label)

    def _print_one(self, job: CardJob, report) -> None:
        """카드 한 장. 알릴 수 있는 단계는 우리가 SDK를 부르는 경계뿐이다.

        SmartComm_DoPrint는 인쇄가 끝나야 돌아오고 패널(Y/M/C/K/O) 진행을 알려 주지 않는다.
        SMART-51/81은 DoPrint 한 번이 양면을 모두 찍으므로 STAGE_PRINT 하나, SMART-50/30 양면은
        DoPrint를 면마다 부르므로 STAGE_PRINT_FRONT → STAGE_PRINT_BACK으로 나눠 알린다.
        panel·fraction은 채우지 않는다.
        """
        binding = self._binding
        options = job.options
        dual = job.is_dual_side
        first = job.front or job.back
        tracks = MagneticTracks.coerce(options.get("magnetic"))
        temp_files: List[str] = []
        try:
            report(5, "이미지 준비 중...", STAGE_PREPARE)
            pages = [(sp.PAGE_FRONT, first, False)]
            if dual:
                pages.append((sp.PAGE_BACK, job.back, True))
            prepared = []
            for page, side, is_back in pages:
                color = self._prepare_image(side.color_path, side, is_back, options, temp_files)
                uv = (self._prepare_image(side.uv_path, side, is_back, options, temp_files)
                      if side.uv_path else None)
                prepared.append((page, color, uv))

            if self.reopen_per_card and self._cards_since_open:
                self._reopen()
            self._cards_since_open += 1
            width, height = sp.DOC_SIZE_BY_GROUP[self._group]
            panel = sp.PANELID_BLACK if self._ribbon_type in sp.MONO_RIBBONS else sp.PANELID_COLOR

            binding.sbs_start()
            try:
                report(15, "카드 삽입 중...", STAGE_FEED)
                binding.card_in()
                if not tracks.is_empty():
                    report(25, "마그네틱 기록 중...", STAGE_MAGNETIC)
                    self._encode_magnetic(tracks, options)
                report(40, "인쇄 데이터 준비 중...", STAGE_SETUP)
                for page, color, uv in prepared:
                    binding.draw_image(page, panel, 0, 0, width, height, color)
                    if uv:
                        binding.draw_image(page, sp.PANELID_UV, 0, 0, width, height, uv)
                side_value = sp.SMART_PRINTSIDE_FRONT
                if dual:
                    side_value = int(options.get("smart_print_side_both",
                                                 sp.PRINTSIDE_BOTH_BY_GROUP[self._group]))
                binding.dcl_print(side_value)
                per_side = dual and self._group == sp.GROUP_SMART50
                report(60, "인쇄 실행 중...", STAGE_PRINT_FRONT if per_side else STAGE_PRINT,
                       "front" if per_side else None)
                binding.do_print()
                if per_side:
                    # SMART-50/30은 DoPrint 한 번에 한 면만 인쇄한다.
                    report(78, "뒷면 인쇄 중...", STAGE_PRINT_BACK, "back")
                    binding.do_print()
                report(95, "카드 배출 중...", STAGE_EJECT)
                binding.card_out()
                report(100, "완료", STAGE_DONE)
            except PrinterError:
                self._eject_quietly()
                raise
            finally:
                self._end_sbs()
        finally:
            for path in temp_files:
                try:
                    os.remove(path)
                except OSError:
                    pass

    def _prepare_image(self, path: str, side: SideImages, is_back: bool,
                       options: Mapping[str, Any], temp_files: List[str]) -> str:
        """도련 제거 → 문서 크기로 맞춤 → 위치 조정 → 세로형이면 회전해 BMP로 저장한다.

        options["bleed_handled"]가 참이면 이미 칼선 크기로 만든 이미지이므로 도련 제거를 건너뛴다.
        """
        from PIL import Image, ImageOps
        from utils.bleed_crop import crop_bleed
        from utils.safe_temp_path import create_safe_temp_file, get_cached_safe_temp_dir

        doc_width, doc_height = sp.DOC_SIZE_BY_GROUP[self._group]
        portrait = side.orientation == ORIENTATION_PORTRAIT
        with open(path, "rb") as handle:
            image = Image.open(io.BytesIO(handle.read()))
        image.load()
        image = ImageOps.exif_transpose(image)
        if not options.get("bleed_handled"):
            image = crop_bleed(image)
        image = image.convert("RGB")
        target = (doc_height, doc_width) if portrait else (doc_width, doc_height)
        if image.size != target:
            image = image.resize(target, Image.LANCZOS)
        dx = round(float(options.get("offset_x_mm", 0.0)) * PRINT_DPI / MM_PER_INCH)
        dy = round(float(options.get("offset_y_mm", 0.0)) * PRINT_DPI / MM_PER_INCH)
        if dx or dy:
            canvas = Image.new("RGB", target, "white")
            canvas.paste(image, (dx, dy))
            image = canvas
        rotation = int(options.get("smart_portrait_rotation", self.portrait_rotation)) if portrait else 0
        if is_back:
            rotation += int(options.get("smart_back_rotation", self.back_rotation))
        if rotation % 360:
            image = image.rotate(rotation % 360, expand=True)
        temp_path = create_safe_temp_file(prefix="smart_back" if is_back else "smart_front",
                                          suffix=".bmp", dir=get_cached_safe_temp_dir())
        image.save(temp_path, "BMP")
        temp_files.append(temp_path)
        return temp_path
