"""RtaiAdapter가 SDK를 부르는 순서와 인자를 가짜 라이브러리로 확인한다."""
import pytest
from printer_helpers import IMG, simplify

from printer.base import CardJob, MagneticTracks, PrinterDevice, PrinterError, SideImages
from printer.fakes import (
    OUT, R600_ERR_FEEDER_EMPTY, R600_ERR_PRINT, R600_ERR_RIBBON_OUT, FakeR600Library,
)
from printer.printer_discovery import PrinterInfo
from printer.r600_printer import R600Printer
from printer.rtai_adapter import RtaiAdapter, code_from_error, hint_for

# 실기 SDK는 목록을 조회한 뒤에야 프린터를 선택할 수 있다(조회 없이 선택하면 오류 8404992).
CONNECT_CALLS = ["R600LibInit", "R600EnumTcpPrt", "R600EnumUsbPrt", "R600SelectPrt", "R600TcpSetTimeout"]


@pytest.fixture
def adapter(r600_lib, rtai_device):
    adapter = RtaiAdapter(lib=r600_lib, settle_seconds=0)
    adapter.connect(rtai_device)
    r600_lib.reset_calls()
    return adapter


def front_canvas(short, long_, mask=False, uv=False, x=0.0, y=0.0):
    calls = [("R600SetCanvasPortrait", (1,)), ("R600PrepareCanvas", (0, 0)),
             ("R600SetImagePara", (1, 0, 0.0))]
    if mask:
        calls.append(("R600DrawWaterMark", (x, y, short, long_, IMG)))
    calls.append(("R600DrawImage", (x, y, short, long_, IMG, 1)))
    if uv:
        calls += [("R600SetLayerFluorThreshold", (220,)), ("R600SetRibbonOpt", (1, 1, b"180", 4)),
                  ("R600DrawLayerFluor", (x, y, short, long_, IMG))]
    calls.append(("R600CommitCanvas", (b"", OUT)))
    return calls


def test_connect_selects_the_requested_device(r600_lib, rtai_device):
    adapter = RtaiAdapter(lib=r600_lib)
    adapter.connect(rtai_device)
    assert adapter.is_connected and adapter.device == rtai_device
    assert r600_lib.call_names() == CONNECT_CALLS
    assert r600_lib.calls_named("R600SelectPrt") == [("R600SelectPrt", (b"RTAI R600 FAKE",))]
    assert r600_lib.calls_named("R600TcpSetTimeout") == [("R600TcpSetTimeout", (15000, 15000))]
    assert r600_lib.selected == "RTAI R600 FAKE"


def test_connect_uses_the_second_printer_when_it_is_the_one_selected():
    lib = FakeR600Library(usb_printers=["FIRST", "SECOND"])
    adapter = RtaiAdapter(lib=lib)
    adapter.connect(PrinterDevice("rtai", "SECOND", "usb", "SECOND"))
    assert lib.selected == "SECOND"
    assert ("R600SelectPrt", (b"SECOND",)) in lib.calls
    assert ("R600SelectPrt", (b"FIRST",)) not in lib.calls


def test_connect_to_unknown_device_fails_with_korean_message_and_clears_sdk(r600_lib):
    adapter = RtaiAdapter(lib=r600_lib)
    with pytest.raises(PrinterError) as info:
        adapter.connect(PrinterDevice("rtai", "NOPE", "usb", "NOPE"))
    assert "NOPE" in info.value.message and "연결하지 못했습니다" in info.value.message
    assert info.value.recoverable
    assert not adapter.is_connected
    assert r600_lib.call_names()[-1] == "R600LibClear"


def test_connect_reports_library_init_failure(r600_lib, rtai_device):
    r600_lib.fail("R600LibInit", 9)
    with pytest.raises(PrinterError, match="초기화하지 못했습니다") as info:
        RtaiAdapter(lib=r600_lib).connect(rtai_device)
    assert info.value.code == 9


def test_missing_dll_is_a_printer_error_not_a_crash(monkeypatch, rtai_device):
    import printer.printer_utils as utils
    monkeypatch.setattr(utils, "find_printer_dll", lambda: None)
    with pytest.raises(PrinterError, match="libDSRetransfer600App.dll"):
        RtaiAdapter().connect(rtai_device)


def test_print_requires_connection():
    with pytest.raises(PrinterError, match="연결되어 있지 않습니다"):
        RtaiAdapter(lib=FakeR600Library()).print_card(CardJob(front=SideImages("x.png")))


def test_single_side_sequence(adapter, r600_lib, make_image, card_size):
    short, long_ = card_size
    progress = []
    adapter.print_card(CardJob(front=SideImages(make_image(), orientation="portrait")),
                       progress=lambda percent, message: progress.append((percent, message)))
    assert simplify(r600_lib.calls) == (
        [("R600CardInject", (0,)), ("R600SetRibbonOpt", (1, 0, b"2", 1))]
        + front_canvas(short, long_)
        + [("R600PrintDraw", (b"FAKE_CANVAS_1", None)), ("R600CardEject", (0,))]
    )
    assert [percent for percent, _ in progress] == [15, 25, 45, 70, 85, 95, 100]
    assert progress[-1] == (100, "완료")


def test_single_side_matches_the_legacy_flow_call_for_call(r600_lib, adapter, make_image):
    image = make_image()
    adapter.print_card(CardJob(front=SideImages(image, orientation="landscape")))
    legacy_lib = FakeR600Library()
    legacy = R600Printer.from_library(legacy_lib, PrinterInfo("RTAI R600 FAKE", "USB"))
    legacy_lib.reset_calls()
    legacy.print_single_side_card(image, None, "landscape", "normal")
    assert simplify(r600_lib.calls) == simplify(legacy_lib.calls)


def test_dual_side_with_mask_sequence(adapter, r600_lib, make_image, card_size):
    short, long_ = card_size
    job = CardJob(
        front=SideImages(make_image(), white_mask_path=make_image(), orientation="portrait"),
        back=SideImages(make_image(), white_mask_path=make_image(), orientation="portrait"),
        options={"offset_x_mm": 1.5, "offset_y_mm": -0.5},
    )
    adapter.print_card(job)
    assert simplify(r600_lib.calls) == (
        [("R600CardInject", (0,)), ("R600SetRibbonOpt", (1, 0, b"2", 1))]
        + front_canvas(short, long_, mask=True, x=1.5, y=-0.5)
        + [("R600ClearCanvas", ()),
           ("R600SetCanvasPortrait", (1,)),
           ("R600PrepareCanvas", (0, 0)),
           ("R600SetImagePara", (1, 180, 0.0)),
           ("R600DrawWaterMark", (1.5, -0.5, short, long_, IMG)),
           ("R600DrawImage", (1.5, -0.5, short, long_, IMG, 1)),
           ("R600CommitCanvas", (b"", OUT)),
           ("R600PrintDraw", (b"FAKE_CANVAS_1", b"FAKE_CANVAS_2")),
           ("R600CardEject", (0,))]
    )


@pytest.mark.parametrize("front, back", [("portrait", "portrait"), ("portrait", "landscape"),
                                         ("landscape", "portrait"), ("landscape", "landscape")])
def test_dual_side_matches_the_legacy_flow_for_every_orientation_pair(front, back, make_image):
    images = [make_image() for _ in range(4)]
    lib = FakeR600Library()
    adapter = RtaiAdapter(lib=lib, settle_seconds=0)
    adapter.connect(PrinterDevice("rtai", "RTAI R600 FAKE", "usb", "RTAI R600 FAKE"))
    lib.reset_calls()
    adapter.print_card(CardJob(
        front=SideImages(images[0], white_mask_path=images[2], orientation=front),
        back=SideImages(images[1], white_mask_path=images[3], orientation=back)))

    legacy_lib = FakeR600Library()
    legacy = R600Printer.from_library(legacy_lib, PrinterInfo("RTAI R600 FAKE", "USB"))
    legacy_lib.reset_calls()
    legacy.print_dual_side_card(images[0], images[1], images[2], images[3], front, back, "layered")
    assert simplify(lib.calls) == simplify(legacy_lib.calls)


def test_back_portrait_mask_is_rotated_180_like_the_legacy_flow(adapter, r600_lib, make_image, tmp_path):
    from PIL import Image
    mask = str(tmp_path / "mask.png")
    image = Image.new("RGB", (540, 860), "white")
    image.paste((0, 0, 0), (0, 0, 100, 100))                 # 왼쪽 위에 검은 사각형
    image.save(mask)
    adapter.print_card(CardJob(front=SideImages(make_image()),
                               back=SideImages(make_image(), white_mask_path=mask)))
    sent = r600_lib.calls_named("R600DrawWaterMark")[0].args[4].decode("cp949")
    with Image.open(sent) as rotated:
        assert rotated.getpixel((530, 850))[0] < 60          # 오른쪽 아래로 옮겨졌다
        assert rotated.getpixel((10, 10))[0] > 200


def test_uv_layer_sequence(adapter, r600_lib, make_image, card_size):
    short, long_ = card_size
    adapter.print_card(CardJob(front=SideImages(make_image(), uv_path=make_image())))
    assert simplify(r600_lib.calls) == (
        [("R600CardInject", (0,)), ("R600SetRibbonOpt", (1, 0, b"2", 1))]
        + front_canvas(short, long_, uv=True)
        + [("R600PrintDraw", (b"FAKE_CANVAS_1", None)), ("R600CardEject", (0,))]
    )


def test_uv_options_override_threshold_and_thickness(adapter, r600_lib, make_image):
    adapter.print_card(CardJob(front=SideImages(make_image(), uv_path=make_image()),
                               options={"uv_threshold": 128, "uv_thickness": "255"}))
    assert r600_lib.calls_named("R600SetLayerFluorThreshold")[0].args == (128,)
    assert ("R600SetRibbonOpt", (1, 1, b"255", 4)) in r600_lib.calls


def test_magnetic_write_and_verify_happen_after_inject_and_before_canvas(
        adapter, r600_lib, make_image, card_size):
    short, long_ = card_size
    job = CardJob(front=SideImages(make_image()),
                  options={"magnetic": {"track1": "HONG GILDONG", "track2": "1234567890"}})
    adapter.print_card(job)
    assert simplify(r600_lib.calls) == (
        [("R600CardInject", (0,)),
         ("R600CardMoveImpr", (6,)),
         ("R600SetMagCardOpt", (1,)),
         ("R600MAGCardWrite", (10, b"HONG GILDONG", 12, b"1234567890", 10, b"", 0)),
         ("R600MAGCardRead", (10, b"", OUT, b"", OUT, b"", OUT)),
         ("R600SetRibbonOpt", (1, 0, b"2", 1))]
        + front_canvas(short, long_)
        + [("R600PrintDraw", (b"FAKE_CANVAS_1", None)), ("R600CardEject", (0,))]
    )


def test_magnetic_options_change_position_timeout_coercivity_and_skip_verify(
        adapter, r600_lib, make_image):
    adapter.print_card(CardJob(front=SideImages(make_image()), options={
        "magnetic": MagneticTracks(track2="42"), "mag_position": 5, "mag_timeout": 20,
        "magnetic_coercivity": "low", "magnetic_verify": False}))
    names = r600_lib.call_names()
    assert ("R600CardMoveImpr", (5,)) in r600_lib.calls
    assert ("R600SetMagCardOpt", (0,)) in r600_lib.calls
    assert ("R600MAGCardWrite", (20, b"", 0, b"42", 2, b"", 0)) in r600_lib.calls
    assert "R600MAGCardRead" not in names


def test_magnetic_verify_mismatch_ejects_and_does_not_print(adapter, r600_lib, make_image):
    r600_lib.mag_read_override = (b"WRONG", b"", b"")
    with pytest.raises(PrinterError, match="마그네틱 검증에 실패") as info:
        adapter.print_card(CardJob(front=SideImages(make_image()),
                                   options={"magnetic": {"track1": "RIGHT"}}))
    assert info.value.recoverable
    assert r600_lib.call_names() == ["R600CardInject", "R600CardMoveImpr", "R600SetMagCardOpt",
                                     "R600MAGCardWrite", "R600MAGCardRead", "R600CardEject"]
    assert r600_lib.printed_cards == 0


def test_coercivity_failure_is_ignored_like_the_reference_flow(adapter, r600_lib, make_image):
    r600_lib.fail("R600SetMagCardOpt", 3)
    adapter.print_card(CardJob(front=SideImages(make_image()),
                               options={"magnetic": {"track2": "1"}}))
    assert r600_lib.printed_cards == 1


def test_standalone_magnetic_write_and_read(adapter, r600_lib):
    adapter.write_magnetic({"track1": "ABC"})
    assert r600_lib.call_names() == ["R600CardInject", "R600CardMoveImpr", "R600SetMagCardOpt",
                                     "R600MAGCardWrite", "R600MAGCardRead", "R600CardEject"]
    r600_lib.reset_calls()
    assert adapter.read_magnetic() == MagneticTracks("ABC", None, None)
    assert r600_lib.call_names() == ["R600CardInject", "R600CardMoveImpr", "R600MAGCardRead",
                                     "R600CardEject"]
    with pytest.raises(PrinterError, match="마그네틱 데이터가 없습니다"):
        adapter.write_magnetic({})


def test_k_split_is_only_called_when_requested_and_only_true_for_dual(adapter, r600_lib, make_image):
    adapter.print_card(CardJob(front=SideImages(make_image())))
    assert "R600SetImgUtilIsNeedSet_K_ColorRibbonSplitPrint" not in r600_lib.call_names()
    r600_lib.reset_calls()
    adapter.print_card(CardJob(front=SideImages(make_image()), options={"k_split": True}))
    assert r600_lib.calls[1] == ("R600SetImgUtilIsNeedSet_K_ColorRibbonSplitPrint", (False,))
    r600_lib.reset_calls()
    adapter.print_card(CardJob(front=SideImages(make_image()), back=SideImages(make_image()),
                               options={"k_split": True}))
    assert r600_lib.calls[1] == ("R600SetImgUtilIsNeedSet_K_ColorRibbonSplitPrint", (True,))


def test_print_error_ejects_card_and_raises_korean_error(adapter, r600_lib, make_image):
    r600_lib.ribbon_out = True
    with pytest.raises(PrinterError) as info:
        adapter.print_card(CardJob(front=SideImages(make_image())))
    error = info.value
    assert error.code == R600_ERR_RIBBON_OUT
    assert error.recoverable
    assert "단면 인쇄 실행에 실패했습니다." in error.message
    assert "리본을 확인해 주세요" in error.message
    assert "[프린터 메시지: Ribbon is used up]" in error.message
    names = r600_lib.call_names()
    assert names[names.index("R600PrintDraw") + 1] == "R600CardEject"
    assert names[-1] == "R600GetErrorOuterInfo"
    assert r600_lib.calls[-1].args[0] == R600_ERR_RIBBON_OUT


def test_feeder_empty_error_is_recoverable_and_still_tries_to_eject(adapter, r600_lib, make_image):
    r600_lib.feeder_empty = True
    with pytest.raises(PrinterError) as info:
        adapter.print_card(CardJob(front=SideImages(make_image())))
    assert info.value.code == R600_ERR_FEEDER_EMPTY
    assert info.value.recoverable
    assert "카드 삽입에 실패했습니다." in info.value.message
    assert "카드 공급함이 비어 있습니다" in info.value.message
    assert r600_lib.call_names()[:2] == ["R600CardInject", "R600CardEject"]


def test_error_without_sdk_text_falls_back_to_operation_and_code(adapter, r600_lib, make_image):
    r600_lib.fail("R600PrepareCanvas", 4242)
    with pytest.raises(PrinterError) as info:
        adapter.print_card(CardJob(front=SideImages(make_image())))
    assert info.value.message == "캔버스 준비에 실패했습니다."
    assert info.value.code == 4242
    assert not info.value.recoverable
    assert str(info.value) == "캔버스 준비에 실패했습니다. (코드 4242)"


def test_error_text_falls_back_through_the_three_sdk_message_functions(adapter, r600_lib, make_image):
    r600_lib.missing.update({"R600GetErrorOuterInfo"})
    r600_lib.ribbon_out = True
    with pytest.raises(PrinterError, match="Ribbon is used up"):
        adapter.print_card(CardJob(front=SideImages(make_image())))
    assert r600_lib.call_names()[-1] == "R600GetErrorInfo"


def test_missing_files_are_rejected_before_a_card_is_taken(adapter, r600_lib, make_image):
    with pytest.raises(PrinterError, match="앞면 이미지 파일을 찾을 수 없습니다"):
        adapter.print_card(CardJob(front=SideImages("missing.png")))
    with pytest.raises(PrinterError, match="뒷면 화이트 마스크 파일을 찾을 수 없습니다"):
        adapter.print_card(CardJob(front=SideImages(make_image()),
                                   back=SideImages(make_image(), white_mask_path="nope.png")))
    assert r600_lib.calls == []


def test_copies_print_the_same_card_repeatedly_with_scaled_progress(adapter, r600_lib, make_image):
    progress = []
    adapter.print_card(CardJob(front=SideImages(make_image()), copies=3),
                       progress=lambda percent, message: progress.append((percent, message)))
    assert r600_lib.call_names().count("R600PrintDraw") == 3
    assert r600_lib.call_names().count("R600CardInject") == 3
    assert progress[-1] == (100, "[3/3] 완료")
    assert [p for p, _ in progress] == sorted(p for p, _ in progress)


def test_mid_job_error_stops_remaining_copies(adapter, r600_lib, make_image):
    r600_lib.fail("R600PrintDraw", R600_ERR_PRINT, on_call=2)
    with pytest.raises(PrinterError) as info:
        adapter.print_card(CardJob(front=SideImages(make_image()), copies=3))
    assert info.value.code == R600_ERR_PRINT
    assert r600_lib.printed_cards == 1
    assert r600_lib.call_names().count("R600CardInject") == 2


def test_back_only_job_prints_that_image_as_a_single_side(adapter, r600_lib, make_image):
    adapter.print_card(CardJob(back=SideImages(make_image(), orientation="landscape")))
    assert ("R600SetCanvasPortrait", (0,)) in r600_lib.calls
    assert "R600ClearCanvas" not in r600_lib.call_names()
    assert r600_lib.calls_named("R600PrintDraw")[0].args == (b"FAKE_CANVAS_1", None)


def test_precheck_reports_empty_feeder_without_moving_a_card(adapter, r600_lib, make_image):
    r600_lib.feeder_empty = True
    with pytest.raises(PrinterError, match="카드 공급함이 비어 있습니다"):
        adapter.print_card(CardJob(front=SideImages(make_image()), options={"precheck": True}))
    assert "R600CardInject" not in r600_lib.call_names()


def test_status_ready(adapter, r600_lib):
    status = adapter.status()
    assert status.ready and status.message == "인쇄할 수 있습니다."
    assert status.code is None
    assert (status.ribbon_remaining, status.film_remaining) == (250, 480)
    assert status.feeder_empty is False and status.card_inside is False
    assert status.ribbon_type == 1
    assert r600_lib.call_names() == ["R600QueryPrtStatus", "R600IsFeederNoEmpty",
                                     "R600IsPrtHaveCard", "R600GetRbnAndFilmRemaining",
                                     "R600RibbonSettingsRW"]


def test_status_with_error_uses_sdk_text(adapter, r600_lib):
    r600_lib.error_status = R600_ERR_RIBBON_OUT
    status = adapter.status()
    assert not status.ready
    assert status.code == R600_ERR_RIBBON_OUT
    assert status.message == "리본을 확인해 주세요. 다 썼다면 교체가 필요합니다. [프린터 메시지: Ribbon is used up]"


def test_status_feeder_empty_and_partial_query_failures(adapter, r600_lib):
    r600_lib.feeder_empty = True
    r600_lib.fail("R600GetRbnAndFilmRemaining", 1)
    status = adapter.status()
    assert not status.ready and status.feeder_empty
    assert "카드 공급함이 비어 있습니다" in status.message
    assert status.ribbon_remaining is None


def test_status_query_failure_returns_not_ready_instead_of_raising(adapter, r600_lib):
    r600_lib.fail("R600QueryPrtStatus", 55)
    status = adapter.status()
    assert not status.ready and status.code == 55
    assert "프린터 상태 조회에 실패했습니다." in status.message


def test_info_collects_what_it_can(adapter, r600_lib):
    r600_lib.fail("R600GetPrtInfo", 1)
    info = adapter.info()
    assert info["manufacturer"]["serial"] == "FAKE0001"
    assert info["serial"] == "FAKE0001"
    assert "counters" not in info and "options" in info


def test_capabilities_include_white_uv_magnetic_and_dual():
    caps = RtaiAdapter(lib=FakeR600Library()).capabilities()
    assert {"dual_side", "white", "uv", "magnetic", "status", "ribbon_level"} <= caps


def test_inject_and_eject_translate_errors(adapter, r600_lib):
    adapter.inject_card()
    adapter.eject_card()
    r600_lib.fail("R600CardEject", 3)
    with pytest.raises(PrinterError, match="카드 배출에 실패했습니다"):
        adapter.eject_card()


def test_close_clears_the_library_without_ejecting(adapter, r600_lib):
    adapter.close()
    assert r600_lib.calls == [("R600LibClear", ())]
    assert not adapter.is_connected and adapter.device is None
    adapter.close()


def test_helpers():
    assert hint_for("Feeder is EMPTY")[0].startswith("카드 공급함")
    assert hint_for("something else") is None
    assert code_from_error(Exception("카드 삽입 실패: 오류 코드 8421378")) == 8421378
    assert code_from_error(Exception("라이브러리 초기화 실패: 12")) == 12
    assert code_from_error(Exception("이미지 파일이 없습니다")) is None


# ── 도련 제거 건너뛰기 ───────────────────────────────────────────────────────
class _SizeCapture(FakeR600Library):
    """SDK로 넘어간 이미지 파일의 크기를 호출 시점에 적어 둔다."""

    def __init__(self):
        super().__init__()
        self.sizes = []

    def _capture(self, name, path):
        from PIL import Image
        with Image.open(path.decode("cp949")) as image:
            self.sizes.append((name, image.size))
        return 0

    def _h_R600DrawImage(self, x, y, w, h, path, mode):
        return self._capture("image", path)

    def _h_R600DrawWaterMark(self, x, y, w, h, path):
        return self._capture("mask", path)

    def _h_R600DrawLayerFluor(self, x, y, w, h, path):
        return self._capture("uv", path)


@pytest.fixture
def sized(rtai_device):
    lib = _SizeCapture()
    adapter = RtaiAdapter(lib=lib, settle_seconds=0)
    adapter.connect(rtai_device)
    return lib, adapter


def _bleed_ratio_job(make_image, **options):
    """58x90 도련 규격과 비율이 같은 세 레이어(컬러·화이트·UV)."""
    side = SideImages(make_image(size=(580, 900)), white_mask_path=make_image(size=(580, 900)),
                      uv_path=make_image(size=(580, 900)), orientation="portrait")
    return CardJob(front=side, options=options)


def test_adapter_crops_bleed_by_ratio_when_the_job_is_not_marked(sized, make_image):
    lib, adapter = sized
    adapter.print_card(_bleed_ratio_job(make_image))
    assert lib.sizes == [("mask", (540, 860)), ("image", (540, 860)), ("uv", (540, 860))]


def test_adapter_skips_the_bleed_crop_for_bleed_handled_jobs_and_restores_the_default(sized, make_image):
    lib, adapter = sized
    adapter.print_card(_bleed_ratio_job(make_image, bleed_handled=True))
    assert lib.sizes == [("mask", (580, 900)), ("image", (580, 900)), ("uv", (580, 900))]
    assert adapter.sdk.bleed_crop_enabled is True, "카드가 끝나면 기존 동작으로 돌아온다"

    lib.sizes.clear()
    adapter.print_card(_bleed_ratio_job(make_image))
    assert {size for _, size in lib.sizes} == {(540, 860)}


def test_bleed_crop_default_is_restored_after_a_failed_card(sized, make_image):
    lib, adapter = sized
    lib.fail("R600PrintDraw", R600_ERR_PRINT)
    with pytest.raises(PrinterError):
        adapter.print_card(_bleed_ratio_job(make_image, bleed_handled=True))
    assert adapter.sdk.bleed_crop_enabled is True


def test_legacy_direct_calls_still_crop_bleed_by_default(make_image):
    lib = _SizeCapture()
    printer = R600Printer.from_library(lib)
    assert printer.bleed_crop_enabled is True
    printer.draw_image(0, 0, 55.0, 86.6, make_image(size=(580, 900)))
    printer.draw_watermark(0, 0, 55.0, 86.6, make_image(size=(580, 900)))
    printer.draw_layer_fluor(0, 0, 55.0, 86.6, make_image(size=(580, 900)))
    assert lib.sizes == [("image", (540, 860)), ("mask", (540, 860)), ("uv", (540, 860))]


# ── 구조화된 진행 · 잔량 단위 ────────────────────────────────────────────────
def test_structured_progress_reports_the_stage_boundaries_the_adapter_controls(adapter, make_image):
    from printer.base import ProgressEvent, StructuredProgress

    events = []
    job = CardJob(front=SideImages(make_image(), orientation="portrait"),
                  back=SideImages(make_image(), orientation="landscape"),
                  options={"magnetic": {"track2": "1234"}, "magnetic_verify": False})
    adapter.print_card(job, progress=StructuredProgress(events.append))
    assert [(e.percent, e.stage, e.side) for e in events] == [
        (10, "feed", None), (12, "magnetic", None), (15, "setup", None),
        (30, "front_image", "front"), (45, "back_image", "back"),
        (70, "print", None), (85, "wait", None), (95, "eject", None), (100, "done", None)]
    assert all(isinstance(e, ProgressEvent) and e.text for e in events)
    # R600PrintDraw 한 번이 양면을 모두 찍으므로 패널·단계 안 진행은 알 수 없다.
    assert all(e.panel is None and e.fraction is None for e in events)
    assert all((e.copy, e.copies) == (1, 1) for e in events)


def test_structured_progress_numbers_the_copies(adapter, make_image):
    from printer.base import StructuredProgress

    events = []
    adapter.print_card(CardJob(front=SideImages(make_image()), copies=2),
                       progress=StructuredProgress(events.append))
    assert sorted({(e.copy, e.copies) for e in events}) == [(1, 2), (2, 2)]
    assert events[-1].text == "[2/2] 완료" and events[-1].percent == 100 and events[-1].stage == "done"


def test_remaining_units_are_marked_unknown_for_rtai(adapter, r600_lib):
    status = adapter.status()
    assert (status.ribbon_remaining, status.film_remaining) == (250, 480)
    assert (status.ribbon_unit, status.film_unit) == ("unknown", "unknown")
    assert status.ribbon_cards_left is None and status.film_cards_left is None
