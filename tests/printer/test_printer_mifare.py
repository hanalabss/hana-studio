"""Mifare(비접촉 IC) 쓰기: 다눌의 RF 항목. 블록·키 검사, 인코더 위치 찾기, 쓰고 다시 읽어 확인."""
import pytest

from printer.base import CardJob, PrinterError, SideImages
from printer.mifare import MifareError, block_data, check_block, check_key
from printer.rtai_adapter import RtaiAdapter


@pytest.fixture(autouse=True)
def _forget_positions():
    from printer.mifare import KNOWN_POSITIONS
    KNOWN_POSITIONS.clear()
    yield
    KNOWN_POSITIONS.clear()


@pytest.fixture
def adapter(r600_lib, rtai_device):
    adapter = RtaiAdapter(lib=r600_lib, settle_seconds=0)
    adapter.connect(rtai_device)
    r600_lib.reset_calls()
    return adapter


def _job(make_image, **mifare):
    write = {"block": 4, "data": block_data("EMP-0042"), "key": "FFFFFFFFFFFF", "key_type": "A"}
    write.update(mifare)
    return CardJob(front=SideImages(make_image()), options={"mifare": write})


def test_block_data_text_and_hex_are_padded_like_danul():
    assert block_data("AB") == "4142" + "0" * 28
    assert block_data("0a1b", "hex") == "0A1B" + "0" * 28
    assert len(block_data("홍길동")) == 32
    with pytest.raises(MifareError, match="16바이트"):
        block_data("A" * 17)
    with pytest.raises(MifareError, match="16진수"):
        block_data("XYZ", "hex")


@pytest.mark.parametrize("block, message", [(0, "1~63"), (3, "키가 들어 있는"), (7, "키가 들어 있는"), (64, "1~63")])
def test_unsafe_blocks_are_refused(block, message):
    with pytest.raises(MifareError, match=message):
        check_block(block)
    assert check_block(4) == 4


def test_key_must_be_12_hex():
    assert check_key("ff ff ff ff ff ff") == "FFFFFFFFFFFF"
    with pytest.raises(MifareError):
        check_key("1234")


def test_write_happens_after_inject_and_before_printing(adapter, r600_lib, make_image):
    adapter.print_card(_job(make_image))
    names = r600_lib.call_names()
    order = [names.index(n) for n in ("R600CardInject", "R600M1CardGetSN", "R600M1CardPwdAuth",
                                      "R600M1CardWrite", "R600M1CardRead", "R600M1CardHalt", "R600PrintDraw")]
    assert order == sorted(order)
    assert r600_lib.mifare_blocks[4] == block_data("EMP-0042")
    assert r600_lib.printed_cards == 1


def test_encoder_position_is_found_then_remembered(adapter, r600_lib, make_image):
    r600_lib.mifare_position = 2
    adapter.print_card(_job(make_image))
    moves = [c.args[0] for c in r600_lib.calls_named("R600CardMoveImpr")]
    assert moves == [1, 2] and adapter.mifare_position == 2
    r600_lib.reset_calls()
    adapter.print_card(_job(make_image))
    assert [c.args[0] for c in r600_lib.calls_named("R600CardMoveImpr")] == [2]


def test_wrong_key_ejects_without_printing(adapter, r600_lib, make_image):
    with pytest.raises(PrinterError, match="인증에 실패") as info:
        adapter.print_card(_job(make_image, key="A0A1A2A3A4A5"))
    assert info.value.recoverable
    assert r600_lib.printed_cards == 0
    names = [n for n in r600_lib.call_names() if n != "R600SetImgVisualParam"]
    assert names[-1] == "R600CardEject" and "R600M1CardHalt" in names


def test_no_ic_card_is_reported(adapter, r600_lib, make_image):
    r600_lib.mifare_position = 9           # 어느 위치에서도 카드 번호가 안 읽힌다
    with pytest.raises(PrinterError, match="IC 카드를 찾지 못했습니다"):
        adapter.print_card(_job(make_image))
    assert r600_lib.printed_cards == 0


def test_position_found_by_test_is_used_by_another_connection(adapter, r600_lib, rtai_device, make_image):
    r600_lib.mifare_position = 3
    adapter.test_mifare()
    other = RtaiAdapter(lib=r600_lib, settle_seconds=0)
    other.connect(rtai_device)
    r600_lib.reset_calls()
    other.print_card(_job(make_image))
    assert [c.args[0] for c in r600_lib.calls_named("R600CardMoveImpr")] == [3]


def test_test_mifare_reads_only_and_ejects(adapter, r600_lib):
    result = adapter.test_mifare(block=4)
    assert result["serial"] == "04A1B2C3" and result["position"] == 1
    names = r600_lib.call_names()
    assert "R600M1CardWrite" not in names and "R600PrintDraw" not in names
    assert names[-1] == "R600CardEject"
