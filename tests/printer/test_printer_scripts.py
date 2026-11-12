"""scripts/hw_*.py를 가상 프린터로 돌려 본다. 실제 DLL은 로드하지 않는다."""
import os
import sys

import pytest

from printer import registry

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPTS = os.path.join(ROOT, "scripts")

MOVING_CALLS = {"R600CardInject", "R600CardEject", "R600CardMove", "R600CardMoveImpr",
                "R600CardTurnover", "R600PrintDraw", "R600PrtReset", "SmartComm_CardIn",
                "SmartComm_CardOut", "SmartComm_Move", "SmartComm_DoPrint", "SmartDCL_Print",
                "SmartComm_SBSStart"}


@pytest.fixture
def scripts(monkeypatch):
    monkeypatch.chdir(ROOT)
    monkeypatch.syspath_prepend(SCRIPTS)
    registry.use_fakes(None)
    yield
    registry.use_fakes(None)
    for name in ("hw_probe", "hw_print_test", "hw_common"):
        sys.modules.pop(name, None)


def fake_calls():
    return [name for library in registry._fake_libraries.values() for name in library.call_names()]


def test_probe_lists_and_reads_both_families_without_moving_a_card(scripts, capfd):
    import hw_probe
    assert hw_probe.main(["--fake"]) == 0
    output = capfd.readouterr().out
    assert "카드·리본을 소모하지 않습니다" in output
    assert "찾은 프린터: 2대" in output
    assert "리본 잔량: 250" in output and "리본 잔량: 180" in output
    assert "결과: 프린터 2대 중 2대 점검 성공" in output
    calls = fake_calls()
    assert "R600QueryPrtStatus" in calls and "SmartComm_GetStatus" in calls
    assert not MOVING_CALLS & set(calls)
    assert calls.count("R600LibClear") == 1 and calls.count("SmartDCL_CloseDevice") == 1


def test_probe_reports_no_device_found(scripts, capfd, monkeypatch):
    import hw_probe
    monkeypatch.setattr(registry, "discover",
                        lambda families: registry.DiscoveryResult([], ["RTAI: 연결된 프린터가 없습니다."]))
    assert hw_probe.main(["--fake"]) == 0
    output = capfd.readouterr().out
    assert "RTAI: 연결된 프린터가 없습니다." in output
    assert "찾은 프린터: 0대" in output and "점검할 프린터가 없습니다" in output


def test_print_test_refuses_without_yes(scripts, capfd):
    import hw_print_test
    assert hw_print_test.main(["--fake", "--index", "2"]) == 2
    output = capfd.readouterr().out
    assert "카드 1장과 리본" in output and "--yes" in output
    assert not MOVING_CALLS & set(fake_calls())


def test_print_test_list_only(scripts, capfd):
    import hw_print_test
    assert hw_print_test.main(["--fake", "--list"]) == 0
    assert "2. IDP SMART / IDP SMART-51 Card Printer (가상)" in capfd.readouterr().out
    assert not MOVING_CALLS & set(fake_calls())


def test_print_test_with_yes_prints_one_card_on_the_chosen_fake_device(scripts, capfd):
    import hw_print_test
    assert hw_print_test.main(["--fake", "--family", "smart", "--dual", "--yes"]) == 0
    output = capfd.readouterr().out
    assert "[100%] 완료" in output and "인쇄가 끝났습니다" in output
    smart = registry._fake_libraries["smart"]
    assert smart.printed_pages == 1
    assert len(smart.calls_named("SmartComm_DrawImage")) == 2
    assert "rtai" not in registry._fake_libraries


def test_print_test_rejects_bad_index(scripts, capfd):
    import hw_print_test
    assert hw_print_test.main(["--fake", "--index", "9", "--yes"]) == 1
    assert "범위를 벗어났습니다" in capfd.readouterr().out
