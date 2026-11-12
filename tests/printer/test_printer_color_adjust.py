"""RTAI 색 조정(밝기·명암·채도): SDK에 넘기는 값과 순서, 그리고 실제 DLL에서 그림이 바뀌는지."""
import os
import sys

import pytest
from PIL import Image, ImageStat

from printer.base import CAP_COLOR_ADJUST, CardJob, PrinterError, SideImages, color_adjust_from
from printer.exceptions import R600PrinterError
from printer.fakes import FakeR600Library
from printer.r600_printer import R600Printer
from printer.rtai_adapter import RtaiAdapter
from printer.smart_adapter import SmartAdapter

VISUAL = "R600SetImgVisualParam"
REAL_DLL = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "dll", "libDSRetransfer600App.dll")


@pytest.fixture
def adapter(r600_lib, rtai_device):
    adapter = RtaiAdapter(lib=r600_lib, settle_seconds=0)
    adapter.connect(rtai_device)
    r600_lib.reset_calls()
    return adapter


def visual_calls(lib):
    return [call.args for call in lib.calls_named(VISUAL)]


def test_only_rtai_says_it_can_adjust_color(adapter, smart_lib):
    assert CAP_COLOR_ADJUST in adapter.capabilities()
    assert CAP_COLOR_ADJUST not in SmartAdapter(lib=smart_lib).capabilities()


def test_values_are_read_from_options_and_kept_inside_the_sdk_range():
    assert color_adjust_from({}) == (0, 0, 0)
    assert color_adjust_from({"brightness": 10, "contrast": -20, "saturation": 30}) == (10, -20, 30)
    assert color_adjust_from({"brightness": 500, "contrast": -500, "saturation": "12"}) == (100, -100, 12)
    assert color_adjust_from({"brightness": None, "contrast": "abc", "saturation": 7.6}) == (0, 0, 8)


def test_a_plain_print_sends_zero_because_the_sdk_remembers_the_last_value(adapter, r600_lib, make_image):
    adapter.print_card(CardJob(front=SideImages(make_image())))
    assert visual_calls(r600_lib) == [(0, 0, 0), (0, 0, 0)], "그리기 전에 한 번, 끝난 뒤 되돌리기 한 번"


def test_the_chosen_values_go_in_after_the_canvas_is_ready_and_before_the_picture(adapter, r600_lib, make_image):
    adapter.print_card(CardJob(front=SideImages(make_image()),
                               options={"brightness": 15, "contrast": -10, "saturation": 25}))
    names = r600_lib.call_names()
    first = names.index(VISUAL)
    assert names.index("R600PrepareCanvas") < first < names.index("R600DrawImage")
    assert visual_calls(r600_lib)[0] == (15, -10, 25)
    assert names[-1] == VISUAL and visual_calls(r600_lib)[-1] == (0, 0, 0), "끝나면 원래대로 돌려 둔다"


def test_both_sides_get_the_same_values(adapter, r600_lib, make_image):
    adapter.print_card(CardJob(front=SideImages(make_image()), back=SideImages(make_image()),
                               options={"saturation": 40}))
    assert visual_calls(r600_lib) == [(0, 0, 40), (0, 0, 40), (0, 0, 0)]
    names = r600_lib.call_names()
    draws = [i for i, name in enumerate(names) if name == "R600DrawImage"]
    visuals = [i for i, name in enumerate(names) if name == VISUAL]
    assert visuals[0] < draws[0] < visuals[1] < draws[1]


def test_values_are_reset_even_when_the_print_fails(adapter, r600_lib, make_image):
    r600_lib.fail("R600PrintDraw", 8388611)
    with pytest.raises(PrinterError):
        adapter.print_card(CardJob(front=SideImages(make_image()), options={"brightness": 30}))
    assert visual_calls(r600_lib)[-1] == (0, 0, 0)


def test_the_legacy_quick_print_flow_does_not_touch_color_adjust(make_image):
    from printer.printer_discovery import PrinterInfo

    lib = FakeR600Library()
    legacy = R600Printer.from_library(lib, PrinterInfo("RTAI R600 FAKE", "USB"))
    legacy.print_single_side_card(make_image(), None, "portrait", "normal")
    assert lib.calls_named(VISUAL) == []


def test_sdk_refusal_is_reported(r600_lib):
    from printer.printer_discovery import PrinterInfo

    printer = R600Printer.from_library(r600_lib, PrinterInfo("RTAI R600 FAKE", "USB"))
    r600_lib.fail(VISUAL, 8388609)
    with pytest.raises(R600PrinterError):
        printer.set_image_visual_params(200, 0, 0)


# ---------------------------------------------------------------- 실제 DLL
needs_real_dll = pytest.mark.skipif(sys.platform != "win32" or not os.path.isfile(REAL_DLL),
                                    reason="실제 RTAI DLL이 있어야 한다")


@pytest.fixture(scope="module")
def real_printer():
    """실제 DLL. 프린터가 연결돼 있지 않아도 캔버스와 미리 보기 그림은 만들어진다. 카드는 쓰지 않는다."""
    try:
        printer = R600Printer(dll_path=REAL_DLL)
    except Exception as exc:
        pytest.skip(f"실제 DLL을 불러오지 못했습니다: {exc}")
    yield printer
    try:
        printer.set_image_visual_params(0, 0, 0)
    finally:
        printer.close()


def render(printer, source, params):
    """우리 인쇄 경로와 같은 순서로 캔버스를 만들고, SDK가 만든 미리 보기 그림을 돌려준다."""
    printer.setup_canvas("portrait", 0)
    if params is not None:
        printer.set_image_visual_params(*params)
    printer.draw_image(0.0, 0.0, 53.98, 85.6, source)
    path = printer.preview_image_path(printer.commit_canvas())
    with Image.open(path) as image:
        result = image.convert("RGB")
    printer.clear_canvas()
    return result


def brightness_of(image):
    return sum(ImageStat.Stat(image).mean) / 3


def contrast_of(image):
    return sum(ImageStat.Stat(image).stddev) / 3


def saturation_of(image):
    return ImageStat.Stat(image.convert("HSV")).mean[1]


@pytest.fixture(scope="module")
def chart(tmp_path_factory):
    from PIL import ImageDraw

    image = Image.new("RGB", (638, 1011), "white")
    draw = ImageDraw.Draw(image)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255),
              (128, 128, 128), (200, 120, 60), (60, 90, 160)]
    for index, color in enumerate(colors):
        left, top = 40 + (index % 3) * 190, 40 + (index // 3) * 150
        draw.rectangle((left, top, left + 170, top + 130), fill=color)
    for x in range(40, 600):
        value = round((x - 40) * 255 / 560)
        draw.line((x, 520, x, 640), fill=(value, value, value))
        draw.line((x, 660, x, 780), fill=(value, 60, 255 - value))
    path = tmp_path_factory.mktemp("chart") / "chart.png"
    image.save(path)
    return str(path)


@needs_real_dll
def test_real_dll_zero_means_unchanged(real_printer, chart):
    plain = render(real_printer, chart, None)
    zero = render(real_printer, chart, (0, 0, 0))
    assert abs(brightness_of(plain) - brightness_of(zero)) < 1.0
    assert abs(saturation_of(plain) - saturation_of(zero)) < 1.0


@needs_real_dll
def test_real_dll_each_value_moves_the_picture_the_expected_way(real_printer, chart):
    base = render(real_printer, chart, (0, 0, 0))
    assert brightness_of(render(real_printer, chart, (50, 0, 0))) > brightness_of(base) + 5
    assert brightness_of(render(real_printer, chart, (-50, 0, 0))) < brightness_of(base) - 5
    assert contrast_of(render(real_printer, chart, (0, 50, 0))) > contrast_of(base) + 2
    assert contrast_of(render(real_printer, chart, (0, -50, 0))) < contrast_of(base) - 2
    assert saturation_of(render(real_printer, chart, (0, 0, 50))) > saturation_of(base) + 5
    assert saturation_of(render(real_printer, chart, (0, 0, -50))) < saturation_of(base) - 5


@needs_real_dll
def test_real_dll_accepts_exactly_minus_100_to_100(real_printer):
    for value in (-100, 100):
        real_printer.set_image_visual_params(value, value, value)
    for value in (-101, 101):
        with pytest.raises(R600PrinterError):
            real_printer.set_image_visual_params(value, 0, 0)
    real_printer.set_image_visual_params(0, 0, 0)


@needs_real_dll
def test_real_dll_keeps_the_value_until_it_is_set_again(real_printer, chart):
    """우리 코드가 조정하지 않는 인쇄에도 0을 넣고, 인쇄 뒤 0으로 되돌리는 이유."""
    base = render(real_printer, chart, (0, 0, 0))
    render(real_printer, chart, (60, 0, 0))
    carried = render(real_printer, chart, None)
    assert brightness_of(carried) > brightness_of(base) + 5, "다시 넣지 않으면 앞의 값이 남는다"
    restored = render(real_printer, chart, (0, 0, 0))
    assert abs(brightness_of(restored) - brightness_of(base)) < 1.0
