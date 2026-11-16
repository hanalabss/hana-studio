"""폴더 자동 인쇄(가짜 프린터)."""

from pathlib import Path

import pytest
from PIL import Image

from design import storage
from services.auto_print import DONE_DIR, FAILED_DIR, AutoPrintError, FolderAutoPrinter, design_columns
from services_helpers import wait_until

CSV = "이름,사진\n김하나,kim\n이둘,lee\n"


@pytest.fixture
def watch(tmp_path):
    folder = tmp_path / "자동 인쇄"
    folder.mkdir()
    for name in ("kim", "lee"):
        Image.new("RGB", (60, 80), (90, 120, 200)).save(folder / f"{name}.png")
    return folder


@pytest.fixture
def design_file(tmp_path, simple_document):
    return storage.save(simple_document, tmp_path / "사원증.hsd")


@pytest.fixture
def auto(app, service, stub_device, tmp_path):
    device = {"value": stub_device}
    printer = FolderAutoPrinter(service, lambda: device["value"], work_dir=tmp_path / "work", interval_ms=10_000)
    printer.device_box = device
    yield printer
    printer.stop()


def poll_twice(auto):
    auto.poll()          # 처음 본 크기를 적는다
    auto.poll()          # 크기가 그대로면 처리한다


def test_design_columns(simple_document):
    assert design_columns(simple_document) == {"이름": False, "사진": True}


def test_new_file_is_printed_and_moved(app, auto, watch, design_file, make_stub):
    stub = make_stub()
    (watch / "old.csv").write_text(CSV, encoding="utf-8-sig")       # 켜기 전에 있던 파일
    auto.start(watch, design_file)
    events = []
    auto.fileFinished.connect(events.append)
    (watch / "오늘.csv").write_text(CSV, encoding="utf-8-sig")
    poll_twice(auto)
    assert auto.busy
    wait_until(app, lambda: events)
    assert len(stub.printed) == 2
    event = events[0]
    assert event.ok and event.printed == 2 and event.name == "오늘.csv"
    assert (watch / DONE_DIR / "오늘.csv").exists() and not (watch / "오늘.csv").exists()
    assert (watch / "old.csv").exists()                              # 켜기 전 파일은 그대로
    assert not list((auto._work_dir).glob("*.hsdata"))               # 임시 명단은 지운다
    assert "다음 파일" in auto.status


def test_file_still_being_copied_waits(app, auto, watch, design_file, make_stub):
    make_stub()
    auto.start(watch, design_file)
    path = watch / "a.csv"
    path.write_text("이름,사진\n", encoding="utf-8-sig")
    auto.poll()
    path.write_text(CSV, encoding="utf-8-sig")                       # 크기가 바뀌었다
    auto.poll()
    assert not auto.busy
    auto.poll()
    assert auto.busy


def test_failed_card_moves_file_to_failed(app, auto, watch, design_file, make_stub):
    make_stub(fail_on={2})
    auto.start(watch, design_file)
    events = []
    auto.fileFinished.connect(events.append)
    (watch / "b.csv").write_text(CSV, encoding="utf-8-sig")
    poll_twice(auto)
    wait_until(app, lambda: events)
    assert not events[0].ok and events[0].failed >= 1
    assert (watch / FAILED_DIR / "b.csv").exists()


def test_columns_that_do_not_match_go_to_failed(app, auto, watch, design_file, make_stub):
    stub = make_stub()
    auto.start(watch, design_file)
    events = []
    auto.fileFinished.connect(events.append)
    (watch / "c.csv").write_text("성명,번호\n김하나,1\n", encoding="utf-8-sig")
    poll_twice(auto)
    assert events and not events[0].ok and "맞지 않습니다" in events[0].message
    assert (watch / FAILED_DIR / "c.csv").exists() and stub.printed == []


def test_waits_for_a_printer(app, auto, watch, design_file, make_stub):
    make_stub()
    auto.start(watch, design_file)
    auto.device_box["value"] = None
    (watch / "d.csv").write_text(CSV, encoding="utf-8-sig")
    poll_twice(auto)
    assert not auto.busy and "프린터가 연결되면" in auto.status and (watch / "d.csv").exists()


def test_start_checks_folder_and_design(auto, tmp_path, watch, design_file):
    from design import presets

    with pytest.raises(AutoPrintError, match="폴더"):
        auto.start(tmp_path / "없음", design_file)
    with pytest.raises(AutoPrintError, match="디자인 파일이 없습니다"):
        auto.start(watch, tmp_path / "없음.hsd")
    plain = storage.save(presets.new_document(presets.CR80), tmp_path / "빈 카드.hsd")
    with pytest.raises(AutoPrintError, match="명단과 이어진 칸"):
        auto.start(watch, plain)
    assert not auto.running


def test_stop(auto, watch, design_file):
    states = []
    auto.runningChanged.connect(states.append)
    auto.start(watch, design_file)
    auto.stop()
    assert states == [True, False] and not auto.running and auto.status == "꺼져 있습니다."
