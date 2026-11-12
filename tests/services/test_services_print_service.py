"""PrintService: 명단 → 디자인 → 프린터까지 한 번에. 가짜 프린터로 끝까지 돌린다."""
import os

import pytest
from PIL import Image

from data.selection import PrintSelection
from design import presets
from design.binding import CounterSource
from design.model import RectItem, TextItem
from printer.fakes import R600_ERR_PRINT
from printer.r600_printer import R600Printer
from services import preflight
from services.print_service import (
    CARD_CANCELLED, CARD_FAILED, CARD_SUCCESS, STATE_CANCELLED, STATE_COMPLETED, STATE_FAILED,
    STATE_PARTIAL, PrintService, PrintServiceError,
)
from services.render_job import PrintOptions
from services_helpers import Recorder, pump, wait_until


def printed_flags(store):
    return [record["_printed"] for record in store.rows()]


# ── 끝에서 끝까지 ────────────────────────────────────────────────────────────
def test_csv_to_five_printed_cards_on_fake_rtai(app, store, document, rtai_device, rtai_lib,
                                                 service, recorder, history):
    assert preflight.check(document, store, None, rtai_device).ok
    handle = service.start(document, store, PrintSelection.all(), rtai_device)
    assert handle.total == 5 and handle.is_active
    wait_until(app, lambda: recorder.finished)

    summary = recorder.finished[0]
    assert (summary.state, summary.total, summary.success, summary.failed, summary.cancelled) == (
        STATE_COMPLETED, 5, 5, 0, 0)
    assert summary.ok and summary.failed_record_ids == []
    assert rtai_lib.printed_cards == 5
    assert recorder.errors == []

    assert printed_flags(store) == [True] * 5
    assert [record["_print_count"] for record in store.rows()] == [1] * 5

    jobs = history.list_jobs()
    assert len(jobs) == 1 and jobs[0].status == "completed" and jobs[0].total_cards == 5
    cards = history.list_cards(jobs[0].id)
    assert [card.record_id for card in cards] == [str(i) for i in store.ids()]
    assert all(card.status == "success" for card in cards)

    assert [(c.index, c.record_name, c.status) for c in recorder.cards] == [
        (1, "김철수", CARD_SUCCESS), (2, "이영희", CARD_SUCCESS), (3, "박민수", CARD_SUCCESS),
        (4, "최지우", CARD_SUCCESS), (5, "정하나", CARD_SUCCESS)]
    assert recorder.queued == [handle] and recorder.started == [handle]
    assert handle.summary is summary and not handle.is_active


def test_progress_reports_done_total_record_and_stage(app, store, document, rtai_device, service, recorder):
    service.start(document, store, store.ids()[:2], rtai_device)
    wait_until(app, lambda: recorder.finished)
    assert all(p.total == 2 and 0 <= p.done <= 2 for p in recorder.progress)
    assert {p.record_name for p in recorder.progress} == {"김철수", "이영희"}
    stages = [p.stage for p in recorder.progress if p.index == 1]
    assert any("카드 삽입" in stage for stage in stages) and any("인쇄 실행" in stage for stage in stages)
    assert all(" - " not in stage and "장" not in stage.split()[0] for stage in stages)
    second = [p for p in recorder.progress if p.index == 2]
    assert second and all(p.done >= 1 and p.record_id == store.ids()[1] for p in second)
    assert all(p.panel == "" for p in recorder.progress)     # 지금 드라이버는 리본 패널을 알려 주지 않는다


def test_current_record_only(app, store, document, rtai_device, rtai_lib, service, recorder):
    target = store.ids()[2]
    service.start(document, store, PrintSelection.current(target), rtai_device)
    wait_until(app, lambda: recorder.finished)
    assert rtai_lib.printed_cards == 1
    assert printed_flags(store) == [False, False, True, False, False]
    assert recorder.cards[0].record_id == target and recorder.cards[0].record_name == "박민수"


def test_selection_none_prints_every_record_of_a_bound_design(app, store, simple_document, stub_device,
                                                             make_stub, service, recorder):
    stub = make_stub()
    service.start(simple_document, store, None, stub_device)
    wait_until(app, lambda: recorder.finished)
    assert stub.printed == store.ids()


def test_unprinted_selection_and_grid_order(app, store, simple_document, stub_device, make_stub,
                                            service, recorder):
    stub = make_stub()
    ids = store.ids()
    store.mark_printed(ids[:2])
    service.start(simple_document, store, PrintSelection.unprinted(), stub_device, order=("이름", "desc"))
    wait_until(app, lambda: recorder.finished)
    assert stub.printed == [ids[3], ids[4], ids[2]]       # 최지우, 정하나, 박민수


def test_design_without_binding_prints_n_copies(app, rtai_device, rtai_lib, service, recorder, history):
    document = presets.new_document(presets.CR80)
    document.side("front").add_item(TextItem(x=5, y=5, w=40, h=10, text="방문증"))
    handle = service.start(document, None, None, rtai_device, PrintOptions(copies=3))
    wait_until(app, lambda: recorder.finished)
    assert handle.total == 3 and rtai_lib.printed_cards == 3
    assert recorder.finished[0].success == 3
    assert [c.record_id for c in recorder.cards] == [None] * 3
    assert handle._render_job.render_count == 1          # 같은 그림은 한 번만 만든다
    assert [card.record_id for card in history.list_cards(history.list_jobs()[0].id)] == [None] * 3


def test_start_refuses_impossible_requests(store, document, rtai_device, service):
    with pytest.raises(PrintServiceError, match="프린터"):
        service.start(document, store, None, None)
    with pytest.raises(PrintServiceError, match="명단"):
        service.start(document, None, [1, 2], rtai_device)
    with pytest.raises(PrintServiceError, match="뽑을 사람이 없습니다"):
        service.start(document, store, [], rtai_device)
    with pytest.raises(PrintServiceError):
        service.start(document, store, PrintSelection.ranges("9-12"), rtai_device)
    assert service.active_jobs() == []


# ── 실패 ────────────────────────────────────────────────────────────────────
def test_mid_job_failure_marks_only_successes_and_reports_failed_ids(
        app, store, document, rtai_device, rtai_lib, service, recorder, history):
    rtai_lib.fail("R600PrintDraw", R600_ERR_PRINT, on_call=3)
    ids = store.ids()
    handle = service.start(document, store, ids, rtai_device)
    wait_until(app, lambda: recorder.finished)

    summary = recorder.finished[0]
    assert (summary.state, summary.success, summary.failed, summary.cancelled) == (STATE_PARTIAL, 2, 1, 2)
    assert summary.failed_record_ids == [ids[2]]
    assert summary.unprinted_record_ids == ids[3:]          # 시도하지 못한 카드(실패한 장은 따로 센다)
    assert summary.message
    assert printed_flags(store) == [True, True, False, False, False]
    assert [c.status for c in recorder.cards] == [CARD_SUCCESS, CARD_SUCCESS, CARD_FAILED]
    assert recorder.cards[2].record_id == ids[2] and recorder.cards[2].message
    assert handle.failed_indices == [2]

    cards = history.list_cards(history.list_jobs()[0].id)
    assert [(card.record_id, card.status) for card in cards] == [
        (str(ids[0]), "success"), (str(ids[1]), "success"), (str(ids[2]), "failed")]


def test_continue_on_error_prints_the_rest(app, store, document, rtai_device, rtai_lib, service, recorder):
    rtai_lib.fail("R600PrintDraw", R600_ERR_PRINT, on_call=2)
    ids = store.ids()
    service.start(document, store, ids, rtai_device, PrintOptions(stop_on_error=False))
    wait_until(app, lambda: recorder.finished)
    summary = recorder.finished[0]
    assert (summary.success, summary.failed, summary.cancelled) == (4, 1, 0)
    assert summary.failed_record_ids == [ids[1]] and summary.state == STATE_PARTIAL
    assert printed_flags(store) == [True, False, True, True, True]


def test_retry_failed_prints_only_the_failed_records(app, store, document, rtai_device, rtai_lib,
                                                     service, recorder, history):
    rtai_lib.fail("R600PrintDraw", R600_ERR_PRINT, on_call=2)
    ids = store.ids()
    first = service.start(document, store, ids, rtai_device, PrintOptions(stop_on_error=False))
    with pytest.raises(PrintServiceError, match="끝난 뒤"):
        service.retry_failed(first)
    wait_until(app, lambda: recorder.finished)

    retry = service.retry_failed(first)
    assert retry.total == 1 and retry.retry_of == first.id and retry.record_ids == [ids[1]]
    assert retry._render_job.specs[0].counter_index == 1       # 일련번호는 처음과 같다
    wait_until(app, lambda: len(recorder.finished) == 2)
    assert recorder.finished[1].state == STATE_COMPLETED
    assert printed_flags(store) == [True] * 5
    assert rtai_lib.printed_cards == 5
    assert len(history.cards_for_record(ids[1])) == 2
    with pytest.raises(PrintServiceError, match="다시 인쇄할 카드가 없습니다"):
        service.retry_failed(retry)


def test_retry_can_include_cards_left_unprinted(app, store, simple_document, stub_device, make_stub,
                                                service, recorder):
    make_stub(fail_on={2})
    ids = store.ids()
    first = service.start(simple_document, store, ids, stub_device)
    wait_until(app, lambda: recorder.finished)
    assert recorder.finished[0].cancelled == 3

    stub = make_stub()
    retry = service.retry_failed(first, include_unprinted=True)
    wait_until(app, lambda: len(recorder.finished) == 2)
    assert stub.printed == ids[1:] and retry.summary.success == 4
    assert printed_flags(store) == [True] * 5


def test_connection_failure_is_reported_as_an_error(app, store, simple_document, stub_device, queue,
                                                    service, recorder):
    from printer.base import PrinterError

    def broken(device):
        raise PrinterError("프린터 전원이 꺼져 있습니다.")

    queue.set_printer_factory(broken)
    handle = service.start(simple_document, store, store.ids(), stub_device)
    wait_until(app, lambda: recorder.finished)
    summary = recorder.finished[0]
    assert (summary.state, summary.success, summary.failed, summary.cancelled) == (STATE_FAILED, 0, 0, 5)
    assert recorder.errors and recorder.errors[0][0] is handle
    assert "프린터 전원이 꺼져 있습니다." in recorder.errors[0][1]
    assert printed_flags(store) == [False] * 5
    assert handle._render_job.render_count == 0


def test_render_failure_fails_that_card_only(app, store, simple_document, stub_device, make_stub,
                                             service, recorder):
    stub = make_stub()
    ids = store.ids()
    handle = service.start(simple_document, store, ids[:3], stub_device, PrintOptions(stop_on_error=False))
    store.delete([ids[1]])
    wait_until(app, lambda: recorder.finished)
    assert [c.status for c in recorder.cards] == [CARD_SUCCESS, CARD_FAILED, CARD_SUCCESS]
    assert "지워진 사람" in recorder.cards[1].message
    assert stub.printed == [ids[0], ids[2]]
    assert handle.summary.failed_record_ids == [ids[1]]


def test_mark_printed_failure_is_surfaced_but_does_not_stop_the_job(app, store, simple_document,
                                                                    stub_device, make_stub, service,
                                                                    recorder, monkeypatch):
    stub = make_stub()
    original = store.mark_printed
    calls = []

    def flaky(ids, printed=True):
        calls.append(list(ids))
        if len(calls) == 1:
            raise RuntimeError("disk full")
        return original(ids, printed)

    monkeypatch.setattr(store, "mark_printed", flaky)
    service.start(simple_document, store, store.ids()[:2], stub_device)
    wait_until(app, lambda: recorder.finished)
    assert len(stub.printed) == 2 and recorder.finished[0].success == 2
    assert len(recorder.errors) == 1 and "김철수" in recorder.errors[0][1]
    assert printed_flags(store)[:2] == [False, True]


# ── 취소 ────────────────────────────────────────────────────────────────────
def test_cancel_after_current_finishes_the_card_in_progress(app, store, simple_document, stub_device,
                                                            make_stub, service, recorder, history):
    stub = make_stub(gate_on=1)
    ids = store.ids()
    handle = service.start(simple_document, store, ids, stub_device)
    wait_until(app, lambda: stub.entered.is_set())
    assert service.cancel(handle) is True
    assert handle.is_active                      # 진행 중인 카드가 끝날 때까지는 살아 있다
    stub.gate.set()
    wait_until(app, lambda: recorder.finished)

    summary = recorder.finished[0]
    assert (summary.state, summary.success, summary.failed, summary.cancelled) == (STATE_CANCELLED, 1, 0, 4)
    assert summary.unprinted_record_ids == ids[1:] and summary.failed_record_ids == []
    assert stub.printed == [ids[0]]
    assert printed_flags(store) == [True, False, False, False, False]
    assert history.list_jobs()[0].status == "cancelled"
    assert service.cancel(handle) is False       # 이미 끝난 작업


def test_cancel_now_does_not_start_another_card(app, store, simple_document, stub_device, make_stub,
                                                service, recorder):
    stub = make_stub(gate_on=1)
    handle = service.start(simple_document, store, store.ids(), stub_device)
    wait_until(app, lambda: stub.entered.is_set())
    assert service.cancel(handle, "now") is True
    stub.gate.set()
    wait_until(app, lambda: recorder.finished)
    summary = recorder.finished[0]
    assert summary.state == STATE_CANCELLED and summary.success == 1 and summary.failed == 0
    assert len(stub.printed) == 1
    assert all(c.status != CARD_FAILED for c in recorder.cards)
    with pytest.raises(ValueError):
        service.cancel(handle, "someday")


def test_cancel_now_drops_a_card_that_is_still_being_prepared(app, store, simple_document, stub_device,
                                                             make_stub, service, recorder):
    """인쇄 스레드가 이미지를 기다리는 동안 취소하면 그 카드는 프린터로 가지 않는다."""
    stub = make_stub()
    handle = service.start(simple_document, store, store.ids(), stub_device)
    # 이벤트 루프를 돌리지 않았으므로 인쇄 스레드는 첫 카드의 렌더링을 기다리고 있다.
    assert service.cancel(handle, "now") is True
    wait_until(app, lambda: recorder.finished)
    summary = recorder.finished[0]
    assert (summary.state, summary.success, summary.failed, summary.cancelled) == (STATE_CANCELLED, 0, 0, 5)
    assert stub.printed == []
    assert [c.status for c in recorder.cards] in ([], [CARD_CANCELLED])
    assert printed_flags(store) == [False] * 5


def test_cancel_a_job_that_is_still_waiting_in_the_queue(app, store, simple_document, stub_device,
                                                         make_stub, service, recorder):
    stub = make_stub(gate_on=1)
    ids = store.ids()
    first = service.start(simple_document, store, ids[:2], stub_device)
    second = service.start(simple_document, store, ids[2:], stub_device)
    wait_until(app, lambda: stub.entered.is_set())
    assert second.state == "queued"
    assert service.cancel(second) is True
    assert [s.job_id for s in recorder.finished] == [second.id]
    assert recorder.finished[0].state == STATE_CANCELLED and recorder.finished[0].cancelled == 3
    assert second._render_job.render_count == 0
    stub.gate.set()
    wait_until(app, lambda: len(recorder.finished) == 2)
    assert first.summary.state == STATE_COMPLETED and stub.printed == ids[:2]


# ── 지연 렌더링 · 임시 파일 ───────────────────────────────────────────────────
def test_rendering_is_lazy_and_bounded(app, store, simple_document, stub_device, make_stub, service,
                                       recorder):
    stub = make_stub(gate_on=1)
    options = PrintOptions(copies=1)
    ids = store.ids() * 1
    for _ in range(3):                                   # 20명으로 늘린다
        for record in store.rows()[:5]:
            store.insert({"이름": record["이름"], "사번": record["사번"], "사진": record["사진"]})
    ids = store.ids()
    assert len(ids) == 20

    handle = service.start(simple_document, store, ids, stub_device, options)
    job = handle._render_job
    assert job.render_count == 0 and job.temp_dir is None      # 시작만으로는 아무것도 만들지 않는다
    wait_until(app, lambda: stub.entered.is_set())
    pump(app, 150)
    # 첫 장이 프린터 안에 있는 동안: 그 장 + 미리 만든 한 장뿐이다.
    assert job.render_count <= 2
    assert len(os.listdir(job.temp_dir)) <= 2
    stub.gate.set()
    wait_until(app, lambda: recorder.finished)
    assert recorder.finished[0].success == 20
    assert job.render_count == 20
    assert job.max_live <= 2


def test_card_files_are_removed_after_each_result_and_folder_at_the_end(
        app, store, simple_document, stub_device, make_stub, service, recorder):
    stub = make_stub()
    leftovers = []

    def on_card(number, job):
        # 앞서 끝난 카드의 파일은 (결과 처리까지 한 박자 여유를 두고) 이미 지워져 있어야 한다.
        leftovers.append([path for path in stub.files[:-2] if os.path.exists(path)])

    stub.on_card = on_card
    handle = service.start(simple_document, store, store.ids(), stub_device)
    job = handle._render_job
    wait_until(app, lambda: stub.printed)
    folder = job.temp_dir
    assert folder and os.path.isdir(folder)
    wait_until(app, lambda: recorder.finished)
    assert leftovers and all(not left for left in leftovers)
    assert len(set(stub.files)) == 5
    assert not os.path.exists(folder)
    assert not any(os.path.exists(path) for path in stub.files)


def test_temp_folder_is_removed_on_cancel(app, store, simple_document, stub_device, make_stub, service,
                                          recorder):
    stub = make_stub(gate_on=1)
    handle = service.start(simple_document, store, store.ids(), stub_device)
    wait_until(app, lambda: stub.entered.is_set())
    pump(app, 100)
    folder = handle._render_job.temp_dir
    assert os.path.isdir(folder)
    service.cancel(handle)
    stub.gate.set()
    wait_until(app, lambda: recorder.finished)
    assert not os.path.exists(folder)


def test_printer_receives_cut_size_images(app, stub_device, make_stub, service, recorder):
    document = presets.new_document(presets.CUT_54X86_BLEED)
    document.side("front").add_item(RectItem(x=0, y=0, w=90, h=58, fill="#224466", stroke_color=None))
    stub = make_stub()
    sizes = []

    def on_card(number, job):
        with Image.open(job.front.color_path) as image:
            sizes.append(image.size)

    stub.on_card = on_card
    service.start(document, None, None, stub_device)
    wait_until(app, lambda: recorder.finished)
    assert sizes == [(1016, 638)]
    assert stub.options[0]["bleed_handled"] is True


def test_counter_runs_across_cards(app, store, stub_device, make_stub, service, recorder, monkeypatch):
    from design import renderer

    document = presets.new_document(presets.CR80)
    document.side("front").add_item(TextItem(x=5, y=5, w=40, h=10, source=CounterSource(start=1)))
    seen = []
    original = renderer.render_side
    monkeypatch.setattr(
        renderer, "render_side",
        lambda d, s, r=None, dpi=300, context=None, **k: seen.append(context.counter_index)
        or original(d, s, r, dpi, context, **k))
    make_stub()
    service.start(document, None, None, stub_device, PrintOptions(copies=3, counter_offset=7))
    wait_until(app, lambda: recorder.finished)
    assert seen == [7, 8, 9]


# ── 큐에 기존 방식 작업이 섞여 있을 때 ─────────────────────────────────────────
def test_works_behind_a_legacy_job_in_the_same_queue(app, tmp_path, store, simple_document, rtai_device,
                                                     rtai_lib, queue, service, recorder):
    queue.set_legacy_printer_factory(lambda path, selected: R600Printer.from_library(rtai_lib, selected))
    legacy_image = tmp_path / "legacy.jpg"
    Image.new("RGB", (540, 860), (10, 120, 60)).save(legacy_image)
    queue_cards = []
    queue.card_result.connect(lambda *args: queue_cards.append(args))
    all_done = []
    queue.all_jobs_completed.connect(lambda *args: all_done.append(args))

    legacy_id = queue.add_job(dll_path="x.dll", front_image_path=str(legacy_image), is_dual_side=False,
                              quantity=2)
    handle = service.start(simple_document, store, store.ids()[:3], rtai_device)
    assert handle.state == "queued" and handle.queue_job_id != legacy_id
    assert recorder.started == []
    wait_until(app, lambda: all_done)

    assert recorder.finished[0].state == STATE_COMPLETED and recorder.finished[0].success == 3
    assert len(recorder.cards) == 3                         # 기존 작업의 카드 결과는 섞이지 않는다
    assert [args[0] for args in queue_cards] == [legacy_id, legacy_id] + [handle.queue_job_id] * 3
    assert all(p.total == 3 for p in recorder.progress)
    assert rtai_lib.printed_cards == 5
    assert printed_flags(store) == [True, True, True, False, False]
    assert all_done[0][:2] == (5, 0)


def test_two_services_on_one_queue_do_not_see_each_others_jobs(app, store, simple_document, stub_device,
                                                               make_stub, queue, service, recorder):
    other = PrintService(queue)
    other_recorder = Recorder(other)
    make_stub()
    service.start(simple_document, store, store.ids()[:2], stub_device)
    other.start(simple_document, store, store.ids()[2:], stub_device)
    wait_until(app, lambda: recorder.finished and other_recorder.finished)
    assert recorder.finished[0].success == 2 and other_recorder.finished[0].success == 3
    assert len(recorder.cards) == 2 and len(other_recorder.cards) == 3
    assert printed_flags(store) == [True] * 5


# ── SMART: 화이트 레이어가 있는 디자인 ─────────────────────────────────────────
def test_smart_printer_warns_about_white_layer_and_still_prints_colour(
        app, store, simple_document, smart_device, smart_lib, service, recorder, history):
    simple_document.side("front").add_item(RectItem(name="화이트 바탕", x=5, y=30, w=30, h=15, layer="white"))

    report = preflight.check(simple_document, store, None, smart_device)
    white = report.by_code("no_white")
    assert len(white) == 1 and white[0].severity == "warning"
    assert "컬러만 인쇄" in white[0].message
    assert report.ok

    captured = []
    service.card_result.connect(lambda result: captured.append(result))
    handle = service.start(simple_document, store, store.ids()[:2], smart_device)
    files = []
    handle._feeder.card_rendered.connect(
        lambda index: files.append(sorted(os.listdir(handle._render_job.temp_dir))))
    wait_until(app, lambda: recorder.finished)

    assert recorder.finished[0].state == STATE_COMPLETED and recorder.finished[0].success == 2
    assert smart_lib.printed_pages == 2
    assert len(smart_lib.drawn) == 2                        # 컬러 패널만 그렸다
    assert all(not any("white" in name for name in names) for names in files)
    assert printed_flags(store)[:2] == [True, True]
    assert len(history.list_cards(history.list_jobs()[0].id)) == 2


def test_rtai_gets_the_white_mask(app, store, simple_document, rtai_device, rtai_lib, service, recorder):
    simple_document.side("front").add_item(RectItem(x=5, y=30, w=30, h=15, layer="white"))
    service.start(simple_document, store, store.ids()[:1], rtai_device)
    wait_until(app, lambda: recorder.finished)
    assert recorder.finished[0].success == 1
    assert rtai_lib.calls_named("R600DrawWaterMark")


def test_forget_drops_a_finished_job(app, store, simple_document, stub_device, make_stub, service, recorder):
    make_stub()
    handle = service.start(simple_document, store, store.ids()[:1], stub_device)
    service.forget(handle)
    assert service.job(handle.id) is handle                 # 진행 중에는 버리지 않는다
    wait_until(app, lambda: recorder.finished)
    service.forget(handle.id)
    assert service.job(handle.id) is None and not service.is_printing


def test_shutdown_stops_every_job_of_this_service(app, store, simple_document, stub_device, make_stub,
                                                  service, recorder):
    stub = make_stub(gate_on=1)
    ids = store.ids()
    service.start(simple_document, store, ids[:3], stub_device)
    service.start(simple_document, store, ids[3:], stub_device)
    wait_until(app, lambda: stub.entered.is_set())
    service.shutdown()
    stub.gate.set()
    wait_until(app, lambda: len(recorder.finished) == 2)
    assert {s.state for s in recorder.finished} == {STATE_CANCELLED}
    assert len(stub.printed) == 1 and not service.is_printing


# ── 큐의 공개 API만 쓴다: job_error · history_id · 취소 상태 · 구조화된 진행 ─────────
def test_connection_failure_message_comes_from_the_public_job_error_signal(app, store, simple_document,
                                                                           stub_device, queue, service,
                                                                           recorder):
    from printer.base import PrinterError

    def broken(device):
        raise PrinterError("프린터 전원이 꺼져 있습니다.")

    queue.set_printer_factory(broken)
    queue._last_error_message = "다른 작업이 남긴 오류"       # 비공개 속성은 더 읽지 않는다
    handle = service.start(simple_document, store, store.ids()[:2], stub_device)
    wait_until(app, lambda: recorder.finished)
    assert recorder.errors == [(handle, "프린터 연결 실패: 프린터 전원이 꺼져 있습니다.")]
    assert recorder.finished[0].message == "프린터 연결 실패: 프린터 전원이 꺼져 있습니다."
    assert not hasattr(service, "_on_queue_progress")


def test_handle_exposes_the_history_id_of_its_job(app, store, simple_document, stub_device, make_stub,
                                                  service, recorder, history):
    make_stub()
    seen = []
    service.job_started.connect(lambda started: seen.append(started.history_id))
    handle = service.start(simple_document, store, store.ids()[:1], stub_device)
    wait_until(app, lambda: recorder.finished)
    assert handle.history_id is not None and seen == [handle.history_id]
    stored = history.get_job(handle.history_id)
    assert stored is not None and stored.status == "completed" and stored.total_cards == 1


def test_cancel_now_records_the_dropped_card_as_cancelled_in_history(app, store, simple_document,
                                                                     stub_device, make_stub, service,
                                                                     recorder, history, queue):
    stub = make_stub()
    finished = []
    queue.all_jobs_completed.connect(lambda *args: finished.append(args))
    ids = store.ids()
    handle = service.start(simple_document, store, ids, stub_device)
    # 이벤트 루프를 돌리지 않고, 인쇄 스레드가 첫 카드의 이미지를 요청해 기다릴 때까지 둔다.
    from PySide6.QtCore import QThread
    for _ in range(5000):
        if 0 in handle._feeder._slots:
            break
        QThread.msleep(2)
    assert 0 in handle._feeder._slots and handle._render_job.render_count == 0
    assert service.cancel(handle, "now") is True
    wait_until(app, lambda: recorder.finished and finished)

    assert stub.printed == []
    assert [(c.index, c.status, c.ok) for c in recorder.cards] == [(1, CARD_CANCELLED, False)]
    summary = recorder.finished[0]
    assert (summary.state, summary.success, summary.failed, summary.cancelled) == (STATE_CANCELLED, 0, 0, 5)
    assert summary.failed_record_ids == [] and summary.unprinted_record_ids == ids
    job = history.get_job(handle.history_id)
    assert (job.status, job.success_cards, job.failed_cards) == ("cancelled", 0, 0)
    assert [(c.card_index, c.status, c.record_id) for c in history.list_cards(job.id)] == [
        (1, "cancelled", str(ids[0]))]
    assert finished[0][:2] == (0, 0), "취소한 카드는 실패 장수에 들어가지 않는다"


def test_queue_cancel_all_ends_service_jobs_as_cancelled_without_blocking(app, store, simple_document,
                                                                          stub_device, make_stub, service,
                                                                          recorder, queue):
    stub = make_stub(gate_on=1)
    ids = store.ids()
    first = service.start(simple_document, store, ids[:3], stub_device)
    second = service.start(simple_document, store, ids[3:], stub_device)
    wait_until(app, lambda: stub.entered.is_set())

    queue.cancel_all()                                   # 첫 카드가 프린터 안에 있어도 바로 돌아온다
    assert [s.job_id for s in recorder.finished] == [second.id] and second.state == STATE_CANCELLED
    assert first.is_active
    stub.gate.set()
    wait_until(app, lambda: len(recorder.finished) == 2)
    assert (first.summary.state, first.summary.success, first.summary.cancelled) == (STATE_CANCELLED, 1, 2)
    assert stub.printed == [ids[0]] and printed_flags(store) == [True, False, False, False, False]


def test_progress_carries_stage_keys_from_the_driver(app, store, simple_document, rtai_device, service,
                                                     recorder):
    service.start(simple_document, store, store.ids()[:2], rtai_device)
    wait_until(app, lambda: recorder.finished)
    first = [p for p in recorder.progress if p.index == 1 and p.done == 0]
    assert [p.stage_key for p in first] == [
        "connect", "feed", "setup", "front_image", "print", "wait", "eject", "done"]
    by_key = {p.stage_key: p for p in first}
    assert by_key["feed"].stage == "카드 삽입 중..." and by_key["feed"].card_percent == 15
    assert by_key["front_image"].side == "front" and by_key["print"].side is None
    assert by_key["connect"].stage == "프린터 연결 중..."
    between = [p for p in recorder.progress if p.done == 1 and p.index == 2]
    assert between[0].stage_key == "next_card" and between[0].stage == "다음 카드 준비 중..."
    assert [p.stage_key for p in between[1:]] == ["feed", "setup", "front_image", "print", "wait", "eject", "done"]
    # 장치가 알려 주지 않는 값은 꾸며 내지 않는다.
    assert all(p.panel == "" and p.fraction is None for p in recorder.progress)
    assert all(0 <= p.card_percent <= 100 and p.percent in (0, 50) for p in recorder.progress)


def test_progress_from_a_driver_without_stage_keys_keeps_text_and_reads_a_named_panel(
        app, store, simple_document, stub_device, make_stub, service, recorder):
    from printer.base import ProgressEvent, emit_progress

    stub = make_stub()
    original = stub.print_card

    def print_card(job, progress=None):
        emit_progress(progress, ProgressEvent(40, "컬러 인쇄 중", "print", panel="Y", fraction=0.25))
        stub._report(45, "K 패널 인쇄 중", progress)
        original(job, progress)

    stub.print_card = print_card
    service.start(simple_document, store, store.ids()[:1], stub_device)
    wait_until(app, lambda: recorder.finished)
    shown = [(p.stage, p.stage_key, p.panel, p.fraction) for p in recorder.progress if p.stage_key != "connect"]
    assert shown == [
        ("컬러 인쇄 중", "print", "Y", 0.25),          # 드라이버가 알려 준 패널
        ("K 패널 인쇄 중", "", "K", None),             # 단계 키 없는 드라이버: 문구에 적힌 패널만 읽는다
        ("인쇄 실행 중...", "", "", None),
        ("완료", "", "", None),
    ]


def test_card_feeder_is_a_lazy_card_provider(app, store, simple_document):
    from printer.base import CardProvider, PrintCancelled
    from services.render_job import CardFeeder, RenderJob

    ids = store.ids()[:3]
    job = RenderJob.for_records(simple_document, store, ids)
    feeder = CardFeeder(job)
    try:
        cards = feeder.cards()
        assert isinstance(cards, CardProvider) and len(cards) == 3
        assert cards.is_dual_side is False and [cards.record_id(i) for i in range(3)] == ids
        assert job.render_count == 0, "길이·레코드 조회로는 렌더링하지 않는다"

        card = cards[1]                                    # 같은 스레드에서는 바로 렌더링한다
        assert card.record_id == ids[1] and os.path.exists(card.front.color_path)
        assert card.options["bleed_handled"] is True and job.render_count == 1
        with pytest.raises(IndexError):
            cards[3]

        feeder.abort()
        with pytest.raises(PrintCancelled):
            cards[2]
        assert job.render_count == 1
    finally:
        feeder.close()
        feeder.deleteLater()
