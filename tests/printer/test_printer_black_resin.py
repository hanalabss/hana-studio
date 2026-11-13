"""SMART: 검정 글자를 색 패널 대신 검정(K 레진) 패널로 찍기."""
import numpy as np
import pytest
from PIL import Image

from printer import smart_printer as sp
from printer.base import CAP_BLACK_RESIN, CardJob, NotSupported, PrinterError, SideImages
from printer.fakes import FakeSmartLibrary
from printer.smart_adapter import SmartAdapter


class _Capture(FakeSmartLibrary):
    def __init__(self):
        super().__init__()
        self.drawn = []

    def _h_SmartComm_DrawImage(self, handle, page, panel, x, y, cx, cy, path, area):
        with Image.open(path) as image:
            self.drawn.append((page, panel, image.copy()))
        return 0


@pytest.fixture
def capture(smart_device):
    lib = _Capture()
    adapter = SmartAdapter(lib=lib)
    adapter.connect(smart_device)
    return lib, adapter


def _mask(tmp_path, size=(1012, 636), box=(100, 100, 300, 160)):
    """흰색이 잉크인 마스크(렌더러가 주는 형식)."""
    image = Image.new("L", size, 0)
    image.paste(255, box)
    path = tmp_path / "black.png"
    image.save(path)
    return str(path)


def test_capability_follows_the_ribbon(smart_lib, smart_device):
    adapter = SmartAdapter(lib=smart_lib)
    assert CAP_BLACK_RESIN in adapter.capabilities(), "연결 전에는 가능한 기능으로 알린다"
    adapter.connect(smart_device)                      # YMCKO
    assert CAP_BLACK_RESIN in adapter.capabilities()
    smart_lib.ribbon_type = 4                          # K 단색 리본
    adapter.connect(smart_device)
    assert CAP_BLACK_RESIN not in adapter.capabilities()


def test_black_content_is_drawn_on_the_black_panel_as_black_on_white(capture, make_image, tmp_path):
    lib, adapter = capture
    job = CardJob(front=SideImages(make_image(size=(1012, 636)), black_path=_mask(tmp_path),
                                   orientation="landscape"))
    adapter.print_card(job)
    assert [(page, panel) for page, panel, _ in lib.drawn] == [
        (sp.PAGE_FRONT, sp.PANELID_COLOR), (sp.PAGE_FRONT, sp.PANELID_BLACK)]
    black = np.asarray(lib.drawn[1][2].convert("L"))
    assert black.shape == (636, 1012)
    assert set(np.unique(black)) == {0, 255}, "검정·흰색 둘뿐이어야 한다"
    assert black[130, 200] == 0 and black[10, 10] == 255, "잉크 자리는 검정, 나머지는 흰색"


def test_black_content_is_scaled_and_shifted_like_the_color_image(capture, make_image, tmp_path):
    lib, adapter = capture
    job = CardJob(front=SideImages(make_image(size=(506, 318)),
                                   black_path=_mask(tmp_path, (506, 318), (50, 50, 150, 80)),
                                   orientation="landscape"),
                  options={"offset_x_mm": 2.0})
    adapter.print_card(job)
    black = np.asarray(lib.drawn[1][2].convert("L"))
    shift = round(2.0 * 300 / 25.4)
    columns = np.where((black == 0).any(axis=0))[0]
    assert abs(columns[0] - (100 + shift)) <= 1 and abs(columns[-1] - (299 + shift)) <= 1
    assert set(np.unique(black)) == {0, 255}


def test_refused_before_taking_a_card_when_the_ribbon_has_no_black_panel_with_color(
        smart_lib, smart_device, make_image, tmp_path):
    smart_lib.ribbon_type = 4
    adapter = SmartAdapter(lib=smart_lib)
    adapter.connect(smart_device)
    smart_lib.reset_calls()
    job = CardJob(front=SideImages(make_image(), black_path=_mask(tmp_path)))
    with pytest.raises(NotSupported, match="검정 패널"):
        adapter.print_card(job)
    assert smart_lib.calls == []


def test_missing_black_file_is_rejected(smart_lib, smart_device, make_image):
    adapter = SmartAdapter(lib=smart_lib)
    adapter.connect(smart_device)
    smart_lib.reset_calls()
    with pytest.raises(PrinterError, match="검정 패널 이미지 파일을 찾을 수 없습니다"):
        adapter.print_card(CardJob(front=SideImages(make_image(), black_path="missing_black.png")))
    assert smart_lib.calls == []


# ── RTAI: 카드프레소와 같은 호출 순서(색 레이어 → 검정 레이어) ──────────────────
def _rtai(r600_lib, rtai_device):
    from printer.rtai_adapter import RtaiAdapter
    adapter = RtaiAdapter(lib=r600_lib, settle_seconds=0)
    adapter.connect(rtai_device)
    r600_lib.reset_calls()
    return adapter


def _layer_calls(lib):
    return [(call.name, call.args[-1] if call.name == "R600DrawImage" else tuple(call.args[:3]))
            for call in lib.calls if call.name in ("R600SetAddImageMode_Rtai", "R600DrawImage")]


def test_rtai_draws_a_separate_black_layer_after_the_color_layer(r600_lib, rtai_device, make_image, tmp_path):
    adapter = _rtai(r600_lib, rtai_device)
    assert CAP_BLACK_RESIN in adapter.capabilities()
    adapter.print_card(CardJob(front=SideImages(make_image(), black_path=_mask(tmp_path, (540, 860)))))
    assert _layer_calls(r600_lib) == [
        ("R600SetAddImageMode_Rtai", (0, True, False)), ("R600DrawImage", 1),     # 색(YMC) 레이어
        ("R600SetAddImageMode_Rtai", (3, False, True)), ("R600DrawImage", 0),     # 검정(K) 레이어
        ("R600SetAddImageMode_Rtai", (0, True, False)),                           # 뒤 레이어를 위해 되돌림
    ]
    names = r600_lib.call_names()
    assert names.index("R600CommitCanvas") > max(i for i, n in enumerate(names) if n == "R600DrawImage")


def test_rtai_black_layer_file_is_black_on_white(tmp_path):
    from printer.rtai_adapter import sdk_black_layer
    with Image.open(sdk_black_layer(_mask(tmp_path))) as image:
        pixels = np.asarray(image.convert("L"))
    assert pixels[130, 200] == 0 and pixels[10, 10] == 255


def test_rtai_without_black_layer_keeps_the_old_calls_until_the_mode_was_touched(
        r600_lib, rtai_device, make_image, tmp_path):
    adapter = _rtai(r600_lib, rtai_device)
    adapter.print_card(CardJob(front=SideImages(make_image())))
    assert _layer_calls(r600_lib) == [("R600DrawImage", 1)], "옵션을 안 쓰면 예전 호출 그대로다"

    adapter.print_card(CardJob(front=SideImages(make_image(), black_path=_mask(tmp_path, (540, 860)))))
    r600_lib.reset_calls()
    adapter.print_card(CardJob(front=SideImages(make_image())))
    assert _layer_calls(r600_lib) == [("R600SetAddImageMode_Rtai", (0, True, False)), ("R600DrawImage", 1)], \
        "검정 레이어를 쓴 뒤에는 색 레이어 모드로 되돌리고 그린다"
