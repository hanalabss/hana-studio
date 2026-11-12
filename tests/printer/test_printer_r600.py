"""기존 R600Printer API가 그대로인지와 새로 붙인 SDK 함수 바인딩을 가짜 라이브러리로 확인한다."""
import inspect

import pytest
from printer_helpers import IMG, simplify

from printer.exceptions import PrinterInitializationError, R600PrinterError
from printer.fakes import OUT, R600_ERR_FEEDER_EMPTY, FakeR600Library
from printer.printer_discovery import PrinterInfo
from printer.r600_extras import EXTRA_SIGNATURES
from printer.r600_printer import R600Printer

LEGACY_PUBLIC_METHODS = {
    "auto_select_printer": ["self"],
    "enum_printers": ["self"],
    "set_timeout": ["self", "timeout_ms"],
    "select_printer": ["self", "printer_name"],
    "inject_card": ["self"],
    "eject_card": ["self"],
    "turnover_card": ["self"],
    "set_ribbon_option": ["self", "ribbon_type", "key", "value"],
    "setup_canvas": ["self", "card_orientation", "rotation"],
    "set_position_offset": ["self", "x", "y"],
    "get_card_dimensions": ["self", "orientation"],
    "clear_canvas": ["self"],
    "draw_watermark": ["self", "x", "y", "width", "height", "image_path"],
    "draw_watermark_rotated": ["self", "x", "y", "width", "height", "image_path", "rotation"],
    "draw_image": ["self", "x", "y", "width", "height", "image_path", "mode"],
    "commit_canvas": ["self"],
    "prepare_front_canvas": ["self", "front_image_path", "watermark_path", "card_width",
                             "card_height", "card_orientation"],
    "prepare_back_canvas": ["self", "back_image_path", "watermark_path", "card_width",
                            "card_height", "card_orientation"],
    "print_dual_side_card": ["self", "front_image_path", "back_image_path",
                             "front_watermark_path", "back_watermark_path", "front_orientation",
                             "back_orientation", "print_mode", "progress_callback"],
    "print_single_side_card": ["self", "image_path", "watermark_path", "card_orientation",
                               "print_mode", "progress_callback"],
    "print_normal_card": ["self", "image_path", "card_orientation"],
    "print_layered_card": ["self", "watermark_path", "image_path", "card_orientation"],
    "cleanup_and_close": ["self"],
    "close": ["self"],
}


@pytest.fixture
def printer(r600_lib):
    return R600Printer.from_library(r600_lib, PrinterInfo("RTAI R600 FAKE", "USB"))


def test_legacy_public_methods_keep_their_names_and_parameters():
    for name, parameters in LEGACY_PUBLIC_METHODS.items():
        method = getattr(R600Printer, name)
        assert list(inspect.signature(method).parameters) == parameters, name
    init = list(inspect.signature(R600Printer.__init__).parameters)
    assert init == ["self", "dll_path", "selected_printer"]


def test_from_library_initialises_and_selects_the_given_printer(r600_lib, printer):
    assert printer.is_initialized
    assert printer.selected_printer_name == "RTAI R600 FAKE"
    # 실기 SDK는 목록 조회 뒤에야 선택할 수 있다.
    assert r600_lib.call_names() == ["R600LibInit", "R600EnumTcpPrt", "R600EnumUsbPrt", "R600SelectPrt"]
    assert r600_lib.calls_named("R600SelectPrt") == [("R600SelectPrt", (b"RTAI R600 FAKE",))]


def test_from_library_reports_init_failure(r600_lib):
    r600_lib.fail("R600LibInit", 5)
    with pytest.raises(PrinterInitializationError):
        R600Printer.from_library(r600_lib)


def test_enum_without_selection_lists_tcp_then_usb():
    lib = FakeR600Library(usb_printers=["USB-A"], tcp_printers=["192.168.0.10"])
    assert R600Printer.from_library(lib).enum_printers() == ["192.168.0.10", "USB-A"]


def test_legacy_single_side_print_sequence_is_unchanged(r600_lib, printer, make_image, card_size):
    short, long_ = card_size
    r600_lib.reset_calls()
    progress = []
    printer.print_single_side_card(make_image(), None, "portrait", "normal",
                                   progress_callback=lambda p, m: progress.append(p))
    assert simplify(r600_lib.calls) == [
        ("R600CardInject", (0,)),
        ("R600SetRibbonOpt", (1, 0, b"2", 1)),
        ("R600SetCanvasPortrait", (1,)),
        ("R600PrepareCanvas", (0, 0)),
        ("R600SetImagePara", (1, 0, 0.0)),
        ("R600DrawImage", (0.0, 0.0, short, long_, IMG, 1)),
        ("R600CommitCanvas", (b"", OUT)),
        ("R600PrintDraw", (b"FAKE_CANVAS_1", None)),
        ("R600CardEject", (0,)),
    ]
    assert progress == [15, 25, 45, 70, 85, 95, 100]


def test_legacy_dual_side_layered_sequence_is_unchanged(r600_lib, printer, make_image, card_size):
    short, long_ = card_size
    r600_lib.reset_calls()
    printer.set_position_offset(1.5, -0.5)
    printer.print_dual_side_card(make_image(), make_image(), make_image(), make_image(),
                                 "portrait", "landscape", "layered")
    assert simplify(r600_lib.calls) == [
        ("R600CardInject", (0,)),
        ("R600SetRibbonOpt", (1, 0, b"2", 1)),
        ("R600SetCanvasPortrait", (1,)),
        ("R600PrepareCanvas", (0, 0)),
        ("R600SetImagePara", (1, 0, 0.0)),
        ("R600DrawWaterMark", (1.5, -0.5, short, long_, IMG)),
        ("R600DrawImage", (1.5, -0.5, short, long_, IMG, 1)),
        ("R600CommitCanvas", (b"", OUT)),
        ("R600ClearCanvas", ()),
        ("R600SetCanvasPortrait", (0,)),
        ("R600PrepareCanvas", (0, 0)),
        ("R600SetImagePara", (1, 0, 0.0)),
        ("R600DrawWaterMark", (1.5, -0.5, long_, short, IMG)),
        ("R600DrawImage", (1.5, -0.5, long_, short, IMG, 1)),
        ("R600CommitCanvas", (b"", OUT)),
        ("R600PrintDraw", (b"FAKE_CANVAS_1", b"FAKE_CANVAS_2")),
        ("R600CardEject", (0,)),
    ]


def test_legacy_error_ejects_and_reraises(r600_lib, printer, make_image):
    r600_lib.feeder_empty = True
    r600_lib.reset_calls()
    with pytest.raises(R600PrinterError, match=str(R600_ERR_FEEDER_EMPTY)):
        printer.print_single_side_card(make_image())
    assert r600_lib.call_names() == ["R600CardInject", "R600CardEject"]


def test_query_status_returns_all_nine_values(r600_lib, printer):
    r600_lib.error_status = 7
    r600_lib.reset_calls()
    status = printer.query_status()
    assert status == {"chassis_temp": 30, "printhead_temp": 35, "heater_temp": 150,
                      "main_status": 0, "sub_status": 0, "error_status": 7,
                      "warning_status": 0, "main_code": 0, "sub_code": 0}
    assert r600_lib.calls == [("R600QueryPrtStatus", (OUT,) * 9)]


def test_error_and_status_messages_use_the_sdk_text_functions(r600_lib, printer):
    r600_lib.reset_calls()
    assert printer.get_error_info(R600_ERR_FEEDER_EMPTY) == "Card feeder is empty"
    assert printer.get_error_outer_info(R600_ERR_FEEDER_EMPTY) == "Card feeder is empty"
    assert printer.get_check_status_message(12345) == ""
    assert r600_lib.calls == [
        ("R600GetErrorInfo", (R600_ERR_FEEDER_EMPTY, b"", OUT)),
        ("R600GetErrorOuterInfo", (R600_ERR_FEEDER_EMPTY, b"", OUT)),
        ("R600GetCheckStatusMsg", (12345, b"", OUT)),
    ]


def test_feeder_card_and_position_queries(r600_lib, printer):
    assert printer.is_feeder_not_empty() is True
    r600_lib.feeder_empty = True
    assert printer.is_feeder_not_empty() is False
    assert printer.has_card_inside() is False
    printer.inject_card.__self__.lib.card_inside = True
    assert printer.has_card_inside() is True
    assert printer.get_card_position() == 1


def test_info_queries(r600_lib, printer):
    assert printer.get_printer_info()["main_fw_version"] == 0x0102
    assert printer.get_manufacturer_info() == {
        "name": "R600 FAKE", "manufacturer": "RTAI", "serial": "FAKE0001",
        "date": "2025-01-01", "main_fw_version": "1.2.3"}
    assert printer.get_serial_number() == "FAKE0001"
    assert printer.get_option_status()["magnetic_reader"] == 1
    assert r600_lib.calls_named("R600GetSN")[0].args == (b"\x00", b"", OUT)
    assert r600_lib.calls_named("R600GetPrtOptionStatus")[0].args == (0,) + (OUT,) * 7


def test_consumable_queries(r600_lib, printer):
    r600_lib.ribbon_remaining, r600_lib.film_remaining = 120, 333
    assert printer.get_ribbon_and_film_remaining() == (120, 333)
    assert printer.read_ribbon_settings() == {"ribbon_type": 1, "film_type": 1,
                                              "ribbon_near_end": 10, "film_near_end": 10}
    assert r600_lib.calls_named("R600RibbonSettingsRW")[0].args == (0, OUT, OUT, OUT, OUT)


def test_uv_layer_calls_threshold_thickness_then_draw(r600_lib, printer, make_image):
    r600_lib.reset_calls()
    printer.draw_uv_layer(1.0, 2.0, 55.0, 86.6, make_image())
    assert simplify(r600_lib.calls) == [
        ("R600SetLayerFluorThreshold", (220,)),
        ("R600SetRibbonOpt", (1, 1, b"180", 4)),
        ("R600DrawLayerFluor", (1.0, 2.0, 55.0, 86.6, IMG)),
    ]


def test_uv_layer_file_has_bleed_cropped_and_rotation_applied(r600_lib, printer, make_image):
    from PIL import Image
    source = make_image(size=(580, 900))          # 58x90 도련 규격
    printer.draw_layer_fluor(0, 0, 55.0, 86.6, source, rotation=90)
    sent = r600_lib.calls_named("R600DrawLayerFluor")[0].args[4].decode("cp949")
    with Image.open(sent) as prepared:
        assert prepared.size == (860, 540)        # 54x86으로 자른 뒤 90도 회전


def test_uv_layer_missing_file_raises(printer):
    with pytest.raises(R600PrinterError, match="존재하지 않습니다"):
        printer.draw_layer_fluor(0, 0, 1, 1, "no_such_file.png")


def test_k_split_card_move_reset_and_usb_timeout(r600_lib, printer):
    r600_lib.reset_calls()
    printer.set_k_split_print(True)
    printer.move_card(2)
    printer.move_card_impr(6)
    printer.reset_printer()
    printer.set_usb_timeout(8000)
    printer.set_add_image_mode_rtai(3, False, True, None, 220)
    assert r600_lib.calls == [
        ("R600SetImgUtilIsNeedSet_K_ColorRibbonSplitPrint", (True,)),
        ("R600CardMove", (2,)),
        ("R600CardMoveImpr", (6,)),
        ("R600PrtReset", ()),
        ("R600UsbSetTimeout", (8000, 8000)),
        ("R600SetAddImageMode_Rtai", (3, False, True, None, 220)),
    ]


def test_magnetic_write_and_read_round_trip(r600_lib, printer):
    r600_lib.reset_calls()
    printer.set_mag_option()
    printer.mag_write(b"TRACK1", b"12345", None)
    assert printer.mag_read() == (b"TRACK1", b"12345", b"")
    assert r600_lib.calls == [
        ("R600SetMagCardOpt", (1,)),
        ("R600MAGCardWrite", (10, b"TRACK1", 6, b"12345", 5, b"", 0)),
        ("R600MAGCardRead", (10, b"", OUT, b"", OUT, b"", OUT)),
    ]


def test_mifare_calls(r600_lib, printer):
    r600_lib.reset_calls()
    assert printer.m1_get_serial() == b"\x04\xA1\xB2\xC3"
    printer.m1_authenticate(4, "FFFFFFFFFFFF")
    assert printer.m1_read(4) == "00112233445566778899AABBCCDDEEFF"
    printer.m1_write(4, "00112233445566778899AABBCCDDEEFF")
    printer.m1_halt()
    assert r600_lib.calls == [
        ("R600M1CardGetSN", (0, b"")),
        ("R600M1CardPwdAuth", (1, 0, 4, b"FFFFFFFFFFFF")),
        ("R600M1CardRead", (1, 4, b"")),
        ("R600M1CardWrite", (1, 4, b"00112233445566778899AABBCCDDEEFF")),
        ("R600M1CardHalt", ()),
    ]


def test_extra_call_failure_raises_with_code(r600_lib, printer):
    r600_lib.fail("R600CardMoveImpr", 77)
    with pytest.raises(R600PrinterError) as info:
        printer.move_card_impr(6)
    assert info.value.code == 77
    assert "오류 코드 77" in str(info.value)


def test_function_missing_from_dll_raises_a_clear_error(r600_lib, printer):
    r600_lib.missing.add("R600QueryPrtStatus")
    with pytest.raises(R600PrinterError, match="R600QueryPrtStatus"):
        printer.query_status()


def test_signatures_are_set_on_first_use(r600_lib, printer):
    printer.get_ribbon_and_film_remaining()
    function = r600_lib.R600GetRbnAndFilmRemaining
    argtypes, restype = EXTRA_SIGNATURES["R600GetRbnAndFilmRemaining"]
    assert function.argtypes == argtypes and function.restype is restype


def test_close_library_clears_sdk_without_moving_the_card(r600_lib, printer):
    r600_lib.reset_calls()
    printer.close_library()
    printer.close_library()
    assert r600_lib.calls == [("R600LibClear", ())]
    assert not printer.is_initialized


def test_extra_signatures_name_functions_the_real_dll_exports():
    """표에 적은 함수 이름이 저장소의 실제 DLL에 모두 들어 있는지 확인한다(파일만 읽는다)."""
    import os
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "dll",
                        "libDSRetransfer600App.dll")
    with open(path, "rb") as handle:
        data = handle.read()
    for name in EXTRA_SIGNATURES:
        assert name.encode("ascii") + b"\0" in data, name


# ── 워터마크 임시 파일 이름이 겹치지 않는다 ─────────────────────────────────────
def test_front_and_back_masks_in_the_same_second_get_their_own_files(monkeypatch, make_image):
    import os
    import time

    from PIL import Image

    from printer.fakes import FakeR600Library

    class Capture(FakeR600Library):
        def __init__(self):
            super().__init__()
            self.seen = []

        def _h_R600DrawWaterMark(self, x, y, w, h, path):
            name = path.decode("cp949")
            with Image.open(name) as image:
                self.seen.append((name, image.convert("RGB").getpixel((5, 5))))
            return 0

    monkeypatch.setattr(time, "time", lambda: 1_700_000_000.4)       # 앞·뒷면이 같은 초에 그려진다
    lib = Capture()
    printer = R600Printer.from_library(lib)
    front = make_image(size=(540, 860), color=(255, 255, 255), suffix=".png")
    back = make_image(size=(540, 860), color=(0, 0, 0), suffix=".png")
    printer.draw_watermark(0, 0, 55.0, 86.6, front)
    printer.draw_watermark_rotated(0, 0, 55.0, 86.6, back, 180)

    (front_path, front_pixel), (back_path, back_pixel) = lib.seen
    assert front_path != back_path, "같은 초라도 서로 다른 파일을 쓴다"
    assert os.path.basename(front_path).startswith("watermark_clean_1700000000_")
    assert front_pixel[0] > 200 and back_pixel[0] < 50
    with Image.open(front_path) as still_front:
        assert still_front.convert("RGB").getpixel((5, 5))[0] > 200, "뒷면이 앞면 마스크를 덮어쓰지 않았다"

    printer.cleanup_layer_files()
    assert os.path.exists(front_path), "카드 사이 정리로는 지우지 않는다(SDK가 아직 쓸 수 있다)"
    printer.close_library()
    assert not os.path.exists(front_path) and not os.path.exists(back_path)


def test_cleanup_and_close_removes_the_watermark_files_it_made(make_image):
    import os

    from printer.fakes import FakeR600Library

    lib = FakeR600Library()
    printer = R600Printer.from_library(lib)
    printer.draw_watermark(0, 0, 55.0, 86.6, make_image(size=(540, 860)))
    sent = lib.calls_named("R600DrawWaterMark")[0].args[4].decode("cp949")
    assert os.path.exists(sent)
    printer.cleanup_and_close()
    assert not os.path.exists(sent)
    assert lib.call_names()[-2:] == ["R600CardEject", "R600LibClear"]


def test_selecting_without_enumeration_fails_like_the_real_sdk(r600_lib):
    """다른 PC 실기에서 확인된 동작: 목록 조회 없이 R600SelectPrt를 부르면 8404992가 돌아온다."""
    from printer.fakes import R600_ERR_SELECT_WITHOUT_ENUM

    r600_lib.R600LibInit()
    assert r600_lib.R600SelectPrt(b"RTAI R600 FAKE") == R600_ERR_SELECT_WITHOUT_ENUM


def test_select_printer_enumerates_first_but_only_once(r600_lib, printer):
    printer.select_printer("RTAI R600 FAKE")
    printer.select_printer("RTAI R600 FAKE")
    names = r600_lib.call_names()
    assert names.count("R600EnumUsbPrt") == 1 and names.count("R600EnumTcpPrt") == 1
    assert names.index("R600EnumUsbPrt") < names.index("R600SelectPrt")
