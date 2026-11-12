"""SmartComm2 바인딩과 SmartAdapter를 가짜 라이브러리로 확인한다."""
import ctypes
import os

import pytest
from PIL import Image
from printer_helpers import IMG, simplify

from printer import smart_printer as sp
from printer.base import (
    CardJob, MagneticTracks, NotSupported, PrinterDevice, PrinterError, SideImages,
)
from printer.fakes import (
    FAKE_SMART_HANDLE, OUT, SMART_PID_50, SMART_PID_51, SMART_PID_81, FakeSmartLibrary,
)
from printer.smart_adapter import SmartAdapter

H = FAKE_SMART_HANDLE
DESC = "SMART-51 FAKE (USB001)"
OPEN_CALL = ("SmartDCL_OpenDevice2", (OUT, DESC, 1, 2))


def device_for(pid, desc="SMART FAKE"):
    info = sp.SmartDeviceInfo("IDP SMART", "SMART", "USB001", desc, pid, "usb")
    return info, PrinterDevice("smart", info.name, "usb", desc, {"pid": pid})


@pytest.fixture
def adapter(smart_lib, smart_device):
    adapter = SmartAdapter(lib=smart_lib)
    adapter.connect(smart_device)
    smart_lib.reset_calls()
    return adapter


# ── 바인딩 ────────────────────────────────────────────────────────────────
def test_struct_layout_matches_the_header():
    assert ctypes.sizeof(sp.SMART_PRINTER_ITEM) == (128 + 64 + 64 + 256) * 2 + 4
    assert ctypes.sizeof(sp.SMART_PRINTER_LIST) == 4 + 32 * 1028
    assert sp.SMART_PRINTER_ITEM.pid.offset == 1024
    assert ctypes.sizeof(sp.SMART_SYSINFO) == 4 + (160 + 24 + 8) + (4 + 112 + 32)
    assert sp.SMART_SYSINFO.laminator.offset == 196


def test_importing_does_not_load_the_dll_and_missing_dll_is_a_printer_error(tmp_path):
    binding = sp.SmartPrinter(dll_path=str(tmp_path / "nope" / "SmartComm2.dll"))
    with pytest.raises(PrinterError, match="SmartComm2.dll"):
        binding.list_devices()


def test_unloadable_dll_is_a_printer_error(tmp_path):
    bogus = tmp_path / "SmartComm2.dll"
    bogus.write_bytes(b"not a dll")
    with pytest.raises(PrinterError, match="불러오지 못했습니다"):
        sp.SmartPrinter(dll_path=str(bogus)).list_devices()


def test_library_without_required_function_is_rejected():
    lib = FakeSmartLibrary()
    lib.missing.add("SmartDCL_Print")
    with pytest.raises(PrinterError, match="버전이 맞지 않습니다") as info:
        sp.SmartPrinter(lib=lib)
    assert "SmartDCL_Print" in info.value.detail


def test_signatures_use_wide_strings_and_unsigned_results(smart_lib):
    sp.SmartPrinter(lib=smart_lib)
    assert smart_lib.SmartDCL_OpenDevice2.argtypes[1] is ctypes.c_wchar_p
    assert smart_lib.SmartComm_DrawImage.argtypes[7] is ctypes.c_wchar_p
    assert smart_lib.SmartComm_DrawImage.argtypes[1] is ctypes.c_ubyte
    assert smart_lib.SmartComm_MagSetAllBuffer.argtypes[2] == ctypes.POINTER(ctypes.c_ubyte)
    assert smart_lib.SmartComm_CardIn.restype is ctypes.c_uint


def test_list_devices_queries_usb_then_network(smart_lib):
    smart_lib.devices.append(sp.SmartDeviceInfo("NET PRINTER", "SMART", "192.168.0.50",
                                                "SMART-51 (192.168.0.50)", SMART_PID_51, "net"))
    found = sp.SmartPrinter(lib=smart_lib).list_devices()
    assert [(d.desc, d.connection, d.group) for d in found] == [
        (DESC, "usb", sp.GROUP_SMART51), ("SMART-51 (192.168.0.50)", "net", sp.GROUP_SMART51)]
    assert smart_lib.calls == [("SmartCommEx_GetDeviceList2", (OUT, 1)),
                               ("SmartCommEx_GetDeviceList2", (OUT, 2))]


def test_list_devices_with_nothing_attached_is_empty_not_an_error():
    binding = sp.SmartPrinter(lib=FakeSmartLibrary(devices=[]))
    assert binding.list_devices() == []
    assert binding.last_list_errors == []


def test_list_devices_falls_back_to_plain_list_on_older_dll(smart_lib):
    smart_lib.missing.add("SmartCommEx_GetDeviceList2")
    found = sp.SmartPrinter(lib=smart_lib).list_devices()
    assert [d.desc for d in found] == [DESC]
    assert smart_lib.call_names() == ["SmartComm_GetDeviceList2"]


def test_list_devices_records_other_failures_as_diagnostics(smart_lib):
    smart_lib.fail("SmartCommEx_GetDeviceList2", 0x80000035, on_call=2)
    binding = sp.SmartPrinter(lib=smart_lib)
    assert len(binding.list_devices()) == 1
    assert "net 목록 조회 실패" in binding.last_list_errors[0]


def test_open_and_close_use_dcl_mode(smart_lib):
    binding = sp.SmartPrinter(lib=smart_lib)
    binding.open(DESC)
    assert binding.is_open
    binding.close()
    binding.close()
    assert smart_lib.calls == [OPEN_CALL, ("SmartDCL_CloseDevice", (H,))]


def test_open_failure_and_use_without_open(smart_lib):
    binding = sp.SmartPrinter(lib=smart_lib)
    with pytest.raises(PrinterError, match="프린터 연결 실패") as info:
        binding.open("NOT THERE")
    assert info.value.code == 0x80000003 and info.value.recoverable
    with pytest.raises(PrinterError, match="열려 있지 않습니다"):
        binding.card_in()


def test_status_ribbon_and_system_info(smart_lib):
    binding = sp.SmartPrinter(lib=smart_lib)
    binding.open(DESC)
    smart_lib.reset_calls()
    ms, ef = binding.get_status()
    decoded = sp.decode_status(sp.GROUP_SMART51, (ms, ef))
    assert decoded.has_flipper and decoded.has_magnetic and not decoded.has_error
    ribbon = binding.get_ribbon_info()
    assert (ribbon.type, ribbon.max, ribbon.remain, ribbon.type_name) == (0, 250, 180, "YMCKO")
    assert binding.get_ribbon_info2(sp.RI2_RETRANSFER).remain == 180
    assert binding.get_system_info()["serial"] == "FAKE51-0001"
    assert smart_lib.calls == [
        ("SmartComm_GetStatus", (H, OUT)),
        ("SmartComm_GetRibbonInfo", (H, OUT, OUT, OUT, OUT)),
        ("SmartComm_GetRibbonInfo2", (H, 1, OUT, OUT, OUT, OUT)),
        ("SmartComm_GetSystemInfo", (H, OUT)),
    ]


@pytest.mark.parametrize("pid, group", [(0x3850, 50), (0x386F, 50), (0x3870, 50), (0x388F, 50),
                                        (0x3510, 51), (0x351F, 51), (0x3310, 51), (0x3810, 81),
                                        (0, 51)])
def test_group_from_pid(pid, group):
    assert sp.group_from_pid(pid) == group


def test_decode_status_smart51():
    status = sp.decode_status(51, (0x0040000000000000 | 0x00100000 | 0x00020000, 0))
    assert "리본이 없습니다. 리본을 장착해 주세요." in status.errors
    assert "프린터 덮개가 열려 있습니다. 덮개를 닫아 주세요." in status.errors
    assert status.hopper_empty and status.cover_open and not status.has_flipper
    assert status.has_magnetic is False
    assert sp.decode_status(51, (0x00000200, 0)).busy


def test_decode_status_smart50_reads_hopper_bit_inverted():
    assert sp.decode_status(50, (0, 0)).hopper_empty
    ready = sp.decode_status(50, (0x00040000 | 0x00800000, 0))
    assert not ready.hopper_empty and ready.has_flipper and ready.has_magnetic is None
    assert sp.decode_status(50, (0x0010000000000000, 0)).errors == ["리본이 없습니다. 리본을 교체해 주세요."]


def test_decode_status_smart81_uses_the_second_word_for_errors():
    status = sp.decode_status(81, (0x0000400000000000 | 0x0000000400000000, 0x0000000800000000))
    assert status.errors == ["필름을 모두 사용했습니다. 필름을 교체해 주세요."]
    assert status.hopper_empty and status.has_magnetic and status.has_flipper
    assert not sp.decode_status(81, (0x10000000, 0)).has_flipper
    negative = sp.decode_status(81, (-0x8000000000000000, 0))
    assert negative.warnings == ["프린터 청소가 필요합니다."]


def test_error_text_known_unknown_and_scard():
    assert sp.error_text(0x8000000B) == ("카드 공급함(호퍼)이 비어 있습니다. 카드를 넣어 주세요.", True)
    assert sp.error_text(0x80100009)[0] == "스마트카드 리더 오류가 발생했습니다."
    assert sp.error_text(0x8FFFFFFF) == ("프린터에서 알 수 없는 오류가 발생했습니다.", False)


def test_default_dll_candidates_point_into_dll_smart():
    assert any(path.replace("\\", "/").endswith("dll/smart/SmartComm2.dll")
               for path in sp.default_dll_candidates())


# ── 어댑터: 연결·상태 ──────────────────────────────────────────────────────
def test_connect_opens_reads_status_and_ribbon(smart_lib, smart_device):
    adapter = SmartAdapter(lib=smart_lib)
    adapter.connect(smart_device)
    assert adapter.is_connected and adapter.group == sp.GROUP_SMART51
    assert smart_lib.calls == [OPEN_CALL, ("SmartComm_GetStatus", (H, OUT)),
                               ("SmartComm_GetRibbonInfo", (H, OUT, OUT, OUT, OUT))]


def test_capabilities_never_claim_white_and_follow_installed_options(smart_lib, smart_device):
    adapter = SmartAdapter(lib=smart_lib)
    assert adapter.capabilities() >= {"dual_side", "magnetic", "status", "ribbon_level"}
    assert "white" not in adapter.capabilities() and "uv" not in adapter.capabilities()
    smart_lib.has_flipper = False
    smart_lib.has_magnetic = False
    adapter.connect(smart_device)
    caps = adapter.capabilities()
    assert "dual_side" not in caps and "magnetic" not in caps and "white" not in caps
    assert {"status", "ribbon_level"} <= caps


def test_uv_capability_only_with_a_fluorescent_ribbon(smart_lib, smart_device):
    smart_lib.ribbon_type = 8          # YMCKFO
    adapter = SmartAdapter(lib=smart_lib)
    adapter.connect(smart_device)
    assert "uv" in adapter.capabilities() and "white" not in adapter.capabilities()


def test_status_ready_with_ribbon_level(adapter, smart_lib):
    status = adapter.status()
    assert status.ready and status.message == "인쇄할 수 있습니다."
    assert status.ribbon_remaining == 180 and status.ribbon_type == 0
    assert status.details["ribbon"]["name"] == "YMCKO"
    assert status.feeder_empty is False


def test_status_hopper_empty_and_ribbon_out(adapter, smart_lib):
    smart_lib.hopper_empty = True
    status = adapter.status()
    assert not status.ready and status.feeder_empty
    assert "카드 공급함(호퍼)이 비어 있습니다" in status.message
    smart_lib.hopper_empty = False
    smart_lib.ribbon_out = True
    status = adapter.status()
    assert not status.ready
    assert status.message == "리본을 모두 사용했습니다. 리본을 교체해 주세요."
    assert status.code


def test_status_failure_is_reported_not_raised(adapter, smart_lib):
    smart_lib.fail("SmartComm_GetStatus", 0x80000006)
    status = adapter.status()
    assert not status.ready and status.code == 0x80000006
    assert "통신하지 못했습니다" in status.message


def test_info_includes_system_info_and_device_fields(adapter):
    info = adapter.info()
    assert info["serial"] == "FAKE51-0001" and info["firmware"] == "1.02.90"
    assert info["pid"] == SMART_PID_51 and info["group"] == 51


# ── 어댑터: 인쇄 순서 ──────────────────────────────────────────────────────
def test_single_side_sequence(adapter, smart_lib, make_image):
    progress = []
    adapter.print_card(CardJob(front=SideImages(make_image(), orientation="landscape")),
                       progress=lambda percent, message: progress.append(percent))
    assert simplify(smart_lib.calls) == [
        ("SmartComm_SBSStart", (H,)),
        ("SmartComm_CardIn", (H,)),
        ("SmartComm_DrawImage", (H, 0, 1, 0, 0, 1012, 636, IMG, None)),
        ("SmartDCL_Print", (H, 0)),
        ("SmartComm_DoPrint", (H,)),
        ("SmartComm_CardOut", (H,)),
        ("SmartComm_SBSEnd", (H,)),
    ]
    assert progress == [5, 15, 40, 60, 95, 100]


def test_dual_side_sequence_on_smart51_sends_both_and_prints_once(adapter, smart_lib, make_image):
    adapter.print_card(CardJob(front=SideImages(make_image()),
                               back=SideImages(make_image(), white_mask_path=make_image())))
    assert simplify(smart_lib.calls) == [
        ("SmartComm_SBSStart", (H,)),
        ("SmartComm_CardIn", (H,)),
        ("SmartComm_DrawImage", (H, 0, 1, 0, 0, 1012, 636, IMG, None)),
        ("SmartComm_DrawImage", (H, 1, 1, 0, 0, 1012, 636, IMG, None)),
        ("SmartDCL_Print", (H, 2)),
        ("SmartComm_DoPrint", (H,)),
        ("SmartComm_CardOut", (H,)),
        ("SmartComm_SBSEnd", (H,)),
    ]


def test_dual_side_on_smart50_uses_side_1_and_prints_each_page(make_image):
    info, device = device_for(SMART_PID_50)
    lib = FakeSmartLibrary(devices=[info])
    adapter = SmartAdapter(lib=lib)
    adapter.connect(device)
    assert adapter.group == sp.GROUP_SMART50
    lib.reset_calls()
    adapter.print_card(CardJob(front=SideImages(make_image()), back=SideImages(make_image())))
    names = lib.call_names()
    assert ("SmartDCL_Print", (H, 1)) in lib.calls
    assert names.count("SmartComm_DoPrint") == 2


def test_smart81_uses_its_own_document_size_and_side_value(make_image):
    info, device = device_for(SMART_PID_81)
    lib = FakeSmartLibrary(devices=[info])
    adapter = SmartAdapter(lib=lib)
    adapter.connect(device)
    lib.reset_calls()
    adapter.print_card(CardJob(front=SideImages(make_image()), back=SideImages(make_image())))
    draws = lib.calls_named("SmartComm_DrawImage")
    assert [call.args[1:7] for call in draws] == [(0, 1, 0, 0, 1040, 664), (1, 1, 0, 0, 1040, 664)]
    assert ("SmartDCL_Print", (H, 2)) in lib.calls
    assert lib.call_names().count("SmartComm_DoPrint") == 1


def test_print_side_value_can_be_overridden(adapter, smart_lib, make_image):
    adapter.print_card(CardJob(front=SideImages(make_image()), back=SideImages(make_image()),
                               options={"smart_print_side_both": 1}))
    assert ("SmartDCL_Print", (H, 1)) in smart_lib.calls


def test_mono_ribbon_draws_on_the_black_panel(smart_lib, smart_device, make_image):
    smart_lib.ribbon_type = 4          # K
    adapter = SmartAdapter(lib=smart_lib)
    adapter.connect(smart_device)
    adapter.print_card(CardJob(front=SideImages(make_image())))
    assert smart_lib.calls_named("SmartComm_DrawImage")[0].args[2] == sp.PANELID_BLACK


def test_magnetic_write_and_verify_sequence(adapter, smart_lib, make_image):
    adapter.print_card(CardJob(front=SideImages(make_image(), orientation="landscape"),
                               options={"magnetic": {"track1": "HONG", "track2": "1234"}}))
    assert simplify(smart_lib.calls) == [
        ("SmartComm_SBSStart", (H,)),
        ("SmartComm_CardIn", (H,)),
        ("SmartComm_MagSetAllBuffer", (H, 1, b"HONG", 4, 1, b"1234", 4, 0, b"", 0)),
        ("SmartComm_MagWriteAction", (H, 3, 0x10)),
        ("SmartComm_MagReadAction", (H, 3)),
        ("SmartComm_MagGetAllBuffer", (H, 1, b"", OUT, 1, b"", OUT, 1, b"", OUT)),
        ("SmartComm_DrawImage", (H, 0, 1, 0, 0, 1012, 636, IMG, None)),
        ("SmartDCL_Print", (H, 0)),
        ("SmartComm_DoPrint", (H,)),
        ("SmartComm_CardOut", (H,)),
        ("SmartComm_SBSEnd", (H,)),
    ]


def test_magnetic_low_coercivity_and_smart50_bool_value(adapter, smart_lib, make_image):
    adapter.print_card(CardJob(front=SideImages(make_image()), options={
        "magnetic": {"track3": "9"}, "magnetic_coercivity": "low", "magnetic_verify": False}))
    assert ("SmartComm_MagWriteAction", (H, 4, 0x00)) in smart_lib.calls
    assert "SmartComm_MagReadAction" not in smart_lib.call_names()

    info, device = device_for(SMART_PID_50)
    lib = FakeSmartLibrary(devices=[info])
    adapter50 = SmartAdapter(lib=lib)
    adapter50.connect(device)
    adapter50.write_magnetic({"track2": "55"})
    assert ("SmartComm_MagWriteAction", (H, 2, 1)) in lib.calls


def test_magnetic_verify_mismatch_ejects_and_leaves_sbs_mode(adapter, smart_lib, make_image):
    smart_lib.mag_read_override = (b"XXXX", b"", b"")
    with pytest.raises(PrinterError, match="마그네틱 검증에 실패"):
        adapter.print_card(CardJob(front=SideImages(make_image()),
                                   options={"magnetic": {"track1": "HONG"}}))
    assert smart_lib.call_names()[-2:] == ["SmartComm_CardOut", "SmartComm_SBSEnd"]
    assert "SmartDCL_Print" not in smart_lib.call_names()


def test_standalone_magnetic_write_and_read(adapter, smart_lib):
    adapter.write_magnetic(("A1", None, "C3"))
    assert smart_lib.call_names() == [
        "SmartComm_SBSStart", "SmartComm_CardIn", "SmartComm_MagSetAllBuffer",
        "SmartComm_MagWriteAction", "SmartComm_MagReadAction", "SmartComm_MagGetAllBuffer",
        "SmartComm_CardOut", "SmartComm_SBSEnd"]
    smart_lib.reset_calls()
    assert adapter.read_magnetic() == MagneticTracks("A1", None, "C3")
    assert smart_lib.call_names() == [
        "SmartComm_SBSStart", "SmartComm_CardIn", "SmartComm_MagReadAction",
        "SmartComm_MagGetAllBuffer", "SmartComm_CardOut", "SmartComm_SBSEnd"]


def test_magnetic_without_encoder_is_not_supported(smart_lib, smart_device, make_image):
    smart_lib.has_magnetic = False
    adapter = SmartAdapter(lib=smart_lib)
    adapter.connect(smart_device)
    smart_lib.reset_calls()
    with pytest.raises(NotSupported):
        adapter.write_magnetic({"track1": "A"})
    with pytest.raises(NotSupported):
        adapter.print_card(CardJob(front=SideImages(make_image()),
                                   options={"magnetic": {"track1": "A"}}))
    assert smart_lib.calls == []


def test_hopper_empty_error_path_tries_eject_and_ends_sbs(adapter, smart_lib, make_image):
    smart_lib.hopper_empty = True
    with pytest.raises(PrinterError) as info:
        adapter.print_card(CardJob(front=SideImages(make_image())))
    assert info.value.code == 0x8000000B and info.value.recoverable
    assert info.value.message == "카드 투입 실패: 카드 공급함(호퍼)이 비어 있습니다. 카드를 넣어 주세요."
    assert smart_lib.calls == [("SmartComm_SBSStart", (H,)), ("SmartComm_CardIn", (H,)),
                               ("SmartComm_CardOut", (H,)), ("SmartComm_SBSEnd", (H,))]


def test_ribbon_out_during_print_ejects_card(adapter, smart_lib, make_image):
    smart_lib.ribbon_out = True
    with pytest.raises(PrinterError, match="리본을 모두 사용했습니다") as info:
        adapter.print_card(CardJob(front=SideImages(make_image())))
    assert info.value.code == 0x8000001D
    assert smart_lib.call_names() == [
        "SmartComm_SBSStart", "SmartComm_CardIn", "SmartComm_DrawImage", "SmartDCL_Print",
        "SmartComm_DoPrint", "SmartComm_CardOut", "SmartComm_SBSEnd"]
    assert not smart_lib.card_inside and not smart_lib.sbs_mode


def test_mid_job_error_on_second_copy_stops_the_rest(adapter, smart_lib, make_image):
    smart_lib.fail("SmartComm_DoPrint", 0x8000001A, on_call=2)
    with pytest.raises(PrinterError, match="인쇄에 실패했습니다"):
        adapter.print_card(CardJob(front=SideImages(make_image()), copies=3))
    assert smart_lib.printed_pages == 1
    assert smart_lib.call_names().count("SmartComm_CardIn") == 2


def test_device_is_reopened_between_cards(adapter, smart_lib, make_image):
    adapter.print_card(CardJob(front=SideImages(make_image()), copies=2))
    names = smart_lib.call_names()
    first_end = names.index("SmartComm_SBSEnd")
    assert names[first_end + 1:first_end + 4] == ["SmartDCL_CloseDevice", "SmartDCL_OpenDevice2",
                                                  "SmartComm_SBSStart"]
    assert names.count("SmartComm_DoPrint") == 2


def test_reopen_can_be_turned_off(smart_lib, smart_device, make_image):
    adapter = SmartAdapter(lib=smart_lib, reopen_per_card=False)
    adapter.connect(smart_device)
    smart_lib.reset_calls()
    adapter.print_card(CardJob(front=SideImages(make_image()), copies=2))
    assert "SmartDCL_OpenDevice2" not in smart_lib.call_names()


def test_dual_side_without_flipper_is_not_supported(smart_lib, smart_device, make_image):
    smart_lib.has_flipper = False
    adapter = SmartAdapter(lib=smart_lib)
    adapter.connect(smart_device)
    smart_lib.reset_calls()
    with pytest.raises(NotSupported, match="플리퍼"):
        adapter.print_card(CardJob(front=SideImages(make_image()), back=SideImages(make_image())))
    assert smart_lib.calls == []


def test_uv_is_refused_without_fluorescent_ribbon_and_printed_with_one(
        adapter, smart_lib, smart_device, make_image):
    job = CardJob(front=SideImages(make_image(), uv_path=make_image()))
    with pytest.raises(NotSupported, match="UV"):
        adapter.print_card(job)
    assert smart_lib.calls == []

    smart_lib.ribbon_type = 8
    adapter.connect(smart_device)
    smart_lib.reset_calls()
    adapter.print_card(job)
    panels = [call.args[2] for call in smart_lib.calls_named("SmartComm_DrawImage")]
    assert panels == [sp.PANELID_COLOR, sp.PANELID_UV]


def test_missing_image_is_rejected_before_a_card_is_taken(adapter, smart_lib):
    with pytest.raises(PrinterError, match="앞면 이미지 파일을 찾을 수 없습니다"):
        adapter.print_card(CardJob(front=SideImages("missing.png")))
    assert smart_lib.calls == []


# ── 어댑터: 이미지 준비 ────────────────────────────────────────────────────
class _Capture(FakeSmartLibrary):
    """DrawImage가 받은 파일을 열어 둔다(어댑터가 인쇄 뒤 임시 파일을 지우기 때문)."""

    def __init__(self):
        super().__init__()
        self.images = []
        self.paths = []

    def _h_SmartComm_DrawImage(self, handle, page, panel, x, y, cx, cy, path, area):
        with Image.open(path) as image:
            self.images.append(image.copy())
        self.paths.append(path)
        return 0


@pytest.fixture
def capture(smart_device):
    lib = _Capture()
    adapter = SmartAdapter(lib=lib)
    adapter.connect(smart_device)
    return lib, adapter


def test_landscape_image_is_scaled_to_the_document_and_saved_as_bmp(capture, make_image):
    lib, adapter = capture
    adapter.print_card(CardJob(front=SideImages(make_image(size=(860, 540)), orientation="landscape")))
    assert lib.images[0].size == (1012, 636)
    assert lib.paths[0].endswith(".bmp") and os.path.isabs(lib.paths[0])
    assert not os.path.exists(lib.paths[0])


def test_portrait_image_is_rotated_into_the_landscape_document(capture, tmp_path):
    lib, adapter = capture
    source = str(tmp_path / "portrait.png")
    image = Image.new("RGB", (636, 1012), "white")
    image.paste((0, 0, 0), (0, 0, 100, 100))                 # 세로형 기준 왼쪽 위
    image.save(source)
    adapter.print_card(CardJob(front=SideImages(source, orientation="portrait")))
    sent = lib.images[0]
    assert sent.size == (1012, 636)
    assert sent.getpixel((50, 600))[0] < 60                  # 반시계 90도: 왼쪽 아래로 간다
    assert sent.getpixel((50, 50))[0] > 200

    adapter.portrait_rotation = 270
    adapter.print_card(CardJob(front=SideImages(source, orientation="portrait")))
    assert lib.images[1].getpixel((960, 50))[0] < 60         # 시계 90도: 오른쪽 위로 간다


def test_bleed_is_cropped_and_offset_shifts_the_image(capture, tmp_path):
    lib, adapter = capture
    source = str(tmp_path / "bleed.png")
    image = Image.new("RGB", (900, 580), (255, 0, 0))        # 90x58 도련 규격, 가장자리 빨강
    image.paste((0, 0, 255), (20, 20, 880, 560))             # 칼선 안쪽은 파랑
    image.save(source)
    adapter.print_card(CardJob(front=SideImages(source, orientation="landscape")))
    assert lib.images[0].getpixel((2, 2)) == (0, 0, 255)

    adapter.print_card(CardJob(front=SideImages(source, orientation="landscape"),
                               options={"offset_x_mm": 2.54, "offset_y_mm": 0}))
    shifted = lib.images[1]
    assert shifted.getpixel((10, 300)) == (255, 255, 255)    # 30px 밀린 자리는 흰색(잉크 없음)
    assert shifted.getpixel((40, 300)) == (0, 0, 255)


def test_back_rotation_option_turns_the_back_image(capture, tmp_path):
    lib, adapter = capture
    source = str(tmp_path / "back.png")
    image = Image.new("RGB", (1012, 636), "white")
    image.paste((0, 0, 0), (0, 0, 100, 100))
    image.save(source)
    adapter.print_card(CardJob(front=SideImages(source, orientation="landscape"),
                               back=SideImages(source, orientation="landscape"),
                               options={"smart_back_rotation": 180}))
    assert lib.images[0].getpixel((50, 50))[0] < 60
    assert lib.images[1].getpixel((960, 590))[0] < 60


def test_inject_and_eject_card(adapter, smart_lib):
    adapter.inject_card()
    adapter.eject_card()
    assert smart_lib.call_names() == ["SmartComm_SBSStart", "SmartComm_CardIn",
                                      "SmartComm_CardOut", "SmartComm_SBSEnd"]


def test_close_is_safe_to_repeat(adapter, smart_lib):
    adapter.close()
    adapter.close()
    assert smart_lib.calls == [("SmartDCL_CloseDevice", (H,))]
    assert not adapter.is_connected
