"""SMART 진하기 조정(전체·노랑·빨강·파랑): SDK 설정에 들어가는 값과 순서, 그리고 실제 프린터에서 인쇄 데이터가 바뀌는지."""
import ctypes
import os
import sys

import pytest
from PIL import Image

from printer import smart_printer as sp
from printer.base import CAP_DENSITY_ADJUST, CardJob, PrinterDevice, PrinterError, SideImages, density_adjust_from
from printer.fakes import SMART_PID_50, SMART_PID_81, FakeSmartLibrary
from printer.rtai_adapter import RtaiAdapter
from printer.smart_adapter import SmartAdapter

SURFACE = "SmartDCL_GetSurface2"
REAL_DLL = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "dll", "smart", "SmartComm2.dll")


@pytest.fixture
def adapter(smart_lib, smart_device):
    adapter = SmartAdapter(lib=smart_lib)
    adapter.connect(smart_device)
    smart_lib.reset_calls()
    return adapter


def other_group_adapter(pid):
    lib = FakeSmartLibrary([sp.SmartDeviceInfo(name="IDP SMART", id="SMART", dev="USB001",
                                               desc="SMART FAKE (USB001)", pid=pid, connection="usb")])
    adapter = SmartAdapter(lib=lib)
    adapter.connect(PrinterDevice("smart", "IDP SMART", "usb", "SMART FAKE (USB001)", {"pid": pid}))
    lib.reset_calls()
    return adapter, lib


def test_smart_51_says_it_can_adjust_density_and_rtai_does_not(adapter, r600_lib):
    assert CAP_DENSITY_ADJUST in adapter.capabilities()
    assert CAP_DENSITY_ADJUST not in RtaiAdapter(lib=r600_lib).capabilities()


@pytest.mark.parametrize("pid", [SMART_PID_50, SMART_PID_81])
def test_models_whose_settings_layout_was_not_verified_do_not_offer_it(pid, make_image):
    adapter, lib = other_group_adapter(pid)
    assert CAP_DENSITY_ADJUST not in adapter.capabilities()
    adapter.print_card(CardJob(front=SideImages(make_image()), options={"density": 40}))
    assert lib.calls_named(SURFACE) == [], "구조를 모르는 기종의 설정에는 손대지 않는다"
    assert lib.printed_pages == 1


def test_an_old_library_without_the_function_does_not_offer_it(smart_device):
    lib = FakeSmartLibrary()
    lib.missing.add(SURFACE)
    adapter = SmartAdapter(lib=lib)
    adapter.connect(smart_device)
    assert CAP_DENSITY_ADJUST not in adapter.capabilities()


def test_values_are_read_from_options_and_kept_inside_the_range():
    assert density_adjust_from({}) == (0, 0, 0, 0)
    assert density_adjust_from({"density": 30, "density_yellow": -10, "density_magenta": 5,
                                "density_cyan": 15}) == (30, -10, 5, 15)
    assert density_adjust_from({"density": 150, "density_yellow": -150, "density_cyan": "x"}) == (100, -100, 0, 0), \
        "SDK는 범위를 넘는 값도 그대로 받으므로 우리가 막는다"


def test_a_plain_print_does_not_touch_the_settings(adapter, smart_lib, make_image):
    adapter.print_card(CardJob(front=SideImages(make_image())))
    assert smart_lib.calls_named(SURFACE) == []
    assert smart_lib.printed_density == [(0, 0, 0, 0)]


def test_the_values_are_in_place_when_the_print_data_is_made_and_put_back_after(adapter, smart_lib, make_image):
    adapter.print_card(CardJob(front=SideImages(make_image()),
                               options={"density": 30, "density_yellow": -20, "density_cyan": 10}))
    assert smart_lib.printed_density == [(30, -20, 0, 10)]
    assert smart_lib.density() == (0, 0, 0, 0), "끝나면 넣기 전 값으로 되돌린다"
    names = smart_lib.call_names()
    assert names.index(SURFACE) < names.index("SmartComm_CardIn"), "카드를 넣기 전에 설정한다"
    assert names.index("SmartComm_CardIn") < names.index("SmartDCL_Print")


def test_both_sides_get_the_same_values(adapter, smart_lib, make_image):
    seen = {}
    original = smart_lib._h_SmartDCL_Print

    def spy(handle, side):
        seen["front"], seen["back"] = smart_lib.density(sp.PAGE_FRONT), smart_lib.density(sp.PAGE_BACK)
        return original(handle, side)

    smart_lib._h_SmartDCL_Print = spy
    adapter.print_card(CardJob(front=SideImages(make_image()), back=SideImages(make_image()),
                               options={"density": 25}))
    assert seen == {"front": (25, 0, 0, 0), "back": (25, 0, 0, 0)}


def test_values_are_put_back_even_when_the_print_fails(smart_device, make_image):
    lib = FakeSmartLibrary()
    adapter = SmartAdapter(lib=lib, reopen_per_card=False)
    adapter.connect(smart_device)
    lib.fail("SmartComm_DoPrint", sp.SM_F_EMPTYRIBBON)
    with pytest.raises(PrinterError):
        adapter.print_card(CardJob(front=SideImages(make_image()), options={"density": 50}))
    assert lib.printed_density == [(50, 0, 0, 0)]
    assert lib.density() == (0, 0, 0, 0)


def test_an_unexpected_settings_layout_stops_before_a_card_is_used(adapter, smart_lib, make_image):
    head = sp._Surface51Head.from_buffer(smart_lib._surface(sp.PAGE_FRONT))
    head.prop.width, head.prop.height = 123, 456
    with pytest.raises(PrinterError):
        adapter.sdk.set_color_density(10)
    assert smart_lib.density() == (0, 0, 0, 0)
    assert "SmartComm_CardIn" not in smart_lib.call_names()


# ---------------------------------------------------------------- 실제 프린터
@pytest.fixture(scope="module")
def real_printer():
    """실제 DLL과 연결된 SMART-51/31. 인쇄 데이터를 파일로만 만들고 프린터로는 보내지 않는다. 카드는 쓰지 않는다."""
    if sys.platform != "win32" or not os.path.isfile(REAL_DLL):
        pytest.skip("실제 SMART DLL이 있어야 한다")
    printer = sp.SmartPrinter(dll_path=REAL_DLL)
    try:
        devices = [d for d in printer.list_devices() if d.group == sp.GROUP_SMART51]
    except Exception as exc:
        pytest.skip(f"SMART 프린터 목록을 얻지 못했습니다: {exc}")
    if not devices:
        pytest.skip("연결된 SMART-51/31 프린터가 없다")
    printer.open(devices[0].desc)
    lib, handle = printer.lib, printer._h()
    set_config, get_config = lib.SmartCommEx_SetConfig, lib.SmartCommEx_GetConfig
    set_config.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int64]
    get_config.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int64)]
    set_config.restype = get_config.restype = ctypes.c_int
    method = ctypes.c_int64(-1)
    file_only = (set_config(handle, 0x0B, 2) == 0 and get_config(handle, 0x0B, ctypes.byref(method)) == 0
                 and method.value == 2)
    if not file_only:
        printer.close()
        pytest.skip("'파일로만 출력'을 켜지 못해 프린터로 나갈 수 있으므로 건너뛴다")
    printer._test_set_config = set_config
    yield printer
    try:
        printer.set_color_density(0, 0, 0, 0)
        set_config(handle, 0x0B, 0)
    finally:
        printer.close()


@pytest.fixture(scope="module")
def chart(tmp_path_factory):
    from PIL import ImageDraw

    image = Image.new("RGB", (1012, 636), "white")
    draw = ImageDraw.Draw(image)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255),
              (128, 128, 128), (200, 120, 60)]
    for index, color in enumerate(colors):
        left, top = 40 + (index % 4) * 240, 40 + (index // 4) * 200
        draw.rectangle((left, top, left + 210, top + 170), fill=color)
    for x in range(40, 972):
        value = round((x - 40) * 255 / 931)
        draw.line((x, 460, x, 590), fill=(value, value, value))
    path = tmp_path_factory.mktemp("chart") / "chart.bmp"
    image.save(path, "BMP")
    return str(path)


def print_data(printer, chart, folder, name, density):
    """우리 인쇄 경로와 같은 호출(설정 → DrawImage → SmartDCL_Print)로 만든 인쇄 데이터. 머리말의 일련값은 뺀다."""
    printer.set_color_density(*density)
    path = os.path.join(str(folder), name + ".prn")
    name_buffer = ctypes.create_unicode_buffer(path)
    assert printer._test_set_config(printer._h(), 0x0C, ctypes.cast(name_buffer, ctypes.c_void_p).value) == 0
    printer.draw_image(sp.PAGE_FRONT, sp.PANELID_COLOR, 0, 0, 1012, 636, chart)
    printer.dcl_print(sp.SMART_PRINTSIDE_FRONT)
    with open(path, "rb") as handle:
        return handle.read()[20:]


def test_real_printer_settings_start_at_zero_and_keep_what_we_set(real_printer):
    assert real_printer.set_color_density(0, 0, 0, 0) == (0, 0, 0, 0)
    real_printer.set_color_density(30, -20, 10, 5)
    assert real_printer.get_color_density(sp.PAGE_FRONT) == (30, -20, 10, 5)
    assert real_printer.get_color_density(sp.PAGE_BACK) == (30, -20, 10, 5)
    assert real_printer.set_color_density(0, 0, 0, 0) == (30, -20, 10, 5)


def test_real_printer_each_value_changes_the_print_data_and_zero_restores_it(real_printer, chart, tmp_path):
    base = print_data(real_printer, chart, tmp_path, "base", (0, 0, 0, 0))
    darker = print_data(real_printer, chart, tmp_path, "darker", (40, 0, 0, 0))
    lighter = print_data(real_printer, chart, tmp_path, "lighter", (-40, 0, 0, 0))
    assert darker != base and lighter != base and darker != lighter
    seen = {base, darker, lighter}
    for name, density in (("yellow", (0, 40, 0, 0)), ("magenta", (0, 0, 40, 0)), ("cyan", (0, 0, 0, 40))):
        data = print_data(real_printer, chart, tmp_path, name, density)
        assert data not in seen, f"{name}만 바꿔도 인쇄 데이터가 달라져야 한다"
        seen.add(data)
    assert print_data(real_printer, chart, tmp_path, "again", (0, 0, 0, 0)) == base, "0으로 되돌리면 처음과 같다"
