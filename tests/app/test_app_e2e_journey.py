"""처음 쓰는 사람이 가는 길 그대로: 모양 고르기 → 명단 가져오기 → 인쇄 → 기록. 조립된 창 전체로 확인한다."""
from app_helpers import *  # noqa: F401,F403
from app_helpers import (
    MISSING_PHOTO, PEOPLE, click, fake_library, id_of, journey_to_print, names_in, print_page_ready,
    pump, wait_done, wait_until,
)

from printer.fakes import R600_ERR_RIBBON_OUT
from ui.pages.printing import STATE_DONE, STATE_READY, STATE_RUNNING
from ui.pages.printing.run_view import ROW_CANCELLED, ROW_FAILED, ROW_SUCCESS

PRINTABLE = [name for name, *_ in PEOPLE if name not in MISSING_PHOTO]


def _exclude_missing_and_start(app, page):
    """"사진 없는 2명 빼고 인쇄" → 알아 둘 것이 있으면 읽었다고 체크 → 인쇄 시작."""
    finding = page.ready.finding("missing_photo")
    assert finding is not None and finding.problem.blocking
    click(finding.button("exclude"))
    pump(app, 30)
    assert page.ready.finding("missing_photo") is None
    if page.ready.acknowledge.isVisibleTo(page.ready):
        assert not page.ready.start_button.isEnabled()
        click(page.ready.acknowledge)
        if not page.ready.acknowledge.isChecked():      # 글자 밖을 눌렀으면 직접 체크한다
            page.ready.acknowledge.setChecked(True)
    assert page.ready.start_button.isEnabled(), page.blocking_reason()
    click(page.ready.start_button)
    assert page.state() == STATE_RUNNING


def test_happy_path_from_template_to_history(app_env, app_window, people_csv, photo_folder, tmp_path):
    window, app = app_window, app_env
    original = people_csv.read_bytes()
    page = journey_to_print(app, window, people_csv, photo_folder)

    # 가져오기: 머리글이 달라도 카드의 칸에 맞춰졌고, 모두 체크되어 넘어왔다
    workspace = window.workspace
    store = workspace.store
    assert store.column_names()[:4] == ["이름", "부서", "직급", "사번"]
    assert store.column("사진").type == "photo"
    assert [r["이름"] for r in store.rows()] == [name for name, *_ in PEOPLE]      # 한글이 깨지지 않았다
    assert len(workspace.checked_ids) == 8
    assert people_csv.read_bytes() == original

    # 인쇄 화면: 8명, 사진 없는 2명이 막고 있다
    assert page.mode() == "checked"
    assert page.ready.mode_buttons["checked"].text() == "선택한 8명"
    assert page.ready.start_button.text() == "8장 인쇄 시작" and not page.ready.start_button.isEnabled()
    finding = page.ready.finding("missing_photo")
    assert "사진이 없는 사람이 2명" in finding.text() and "문서윤" in finding.text() and "서지안" in finding.text()
    assert finding.button("exclude").text() == "사진 없는 2명 빼고 인쇄"
    for word in ("레코드", "열을", "' 열"):
        assert all(word not in row.text() for row in page.ready.finding_rows), word

    _exclude_missing_and_start(app, page)
    assert page.run.title.text() == "이번에 뽑는 6명"
    wait_done(app, page)

    assert [page.run.row_state(i) for i in range(6)] == [ROW_SUCCESS] * 6
    assert page.run.summary_headline.text() == "6장을 모두 뽑았습니다"
    assert fake_library("rtai").printed_cards == 6

    # 명단: 그 6명만 뽑은 것으로 보인다
    printed = {r["이름"] for r in store.rows() if r["_printed"]}
    assert printed == set(PRINTABLE)
    people = window.page("list")
    window.navigate("list")
    pump(app)
    assert people.counts()["printed"] == 6 and people.counts()["unprinted"] == 2
    people.set_chip("printed")
    assert names_in(people) == PRINTABLE
    people.set_chip("all")

    # 기록: 그 인쇄가 사람별 줄과 함께 남았다. 인쇄 화면의 단추는 바로 그 기록으로 간다.
    window.navigate("print")
    click(page.run.history_button)
    history = window.page("history")
    assert window.current_page_id() == "history"
    jobs = window.services.history.list_jobs()
    assert len(jobs) == 1 and jobs[0].status == "completed" and jobs[0].success_cards == 6
    assert history.selected_job_id() == jobs[0].id == page.job().history_id
    assert [row["person"] for row in history.card_rows()] == PRINTABLE
    assert {row["result"] for row in history.card_rows()} == {"끝남"}

    # 다시 인쇄 화면: "아직 안 뽑은" 쪽에는 뽑은 6명이 없다
    window.navigate("print")
    click(page.run.back_button)
    assert page.state() == STATE_READY
    page.set_mode("unprinted")
    pump(app, 30)
    assert page.ready.mode_buttons["unprinted"].text() == "아직 안 뽑은 2명 모두"
    left = {store.get(i)["이름"] for i in page.target_ids()}
    assert left == set(MISSING_PHOTO)


def test_failure_on_third_card_then_retry_only_that_one(app_env, app_window, people_csv, photo_folder):
    window, app = app_window, app_env
    page = journey_to_print(app, window, people_csv, photo_folder)
    store = window.workspace.store
    choices = page.choices().clone()
    choices.options.stop_on_error = False          # 실패해도 나머지는 계속 뽑는다
    page.apply_choices(choices)
    library = fake_library("rtai")
    library.fail("R600PrintDraw", R600_ERR_RIBBON_OUT, on_call=3)

    _exclude_missing_and_start(app, page)
    wait_done(app, page)

    states = [page.run.row_state(i) for i in range(6)]
    assert states == [ROW_SUCCESS, ROW_SUCCESS, ROW_FAILED, ROW_SUCCESS, ROW_SUCCESS, ROW_SUCCESS]
    assert page.run.summary_headline.text() == "6장 중 5장을 뽑았습니다"
    failures = page.run.failure_texts()
    assert len(failures) == 1 and failures[0].startswith(PRINTABLE[2] + ":")
    assert "리본" in failures[0], failures[0]                     # 이유를 우리말로
    assert "Ribbon" not in failures[0] and "0x" not in failures[0].split(":")[0]
    failed_id = id_of(store, PRINTABLE[2])
    assert not store.get(failed_id)["_printed"]
    assert sum(1 for r in store.rows() if r["_printed"]) == 5
    assert page.run.retry_button.text() == "실패한 1장 다시 인쇄"

    printed_before = library.printed_cards
    click(page.run.retry_button)
    assert page.state() == STATE_RUNNING
    assert page.run.title.text() == "다시 뽑는 1장"
    assert page.run.model.rowCount() == 1 and page.run.model.row(0)["name"] == PRINTABLE[2]
    wait_done(app, page)
    assert library.printed_cards == printed_before + 1            # 그 한 장만 다시 뽑았다
    assert page.run.summary_headline.text() == "1장을 모두 뽑았습니다"
    assert {r["이름"] for r in store.rows() if r["_printed"]} == set(PRINTABLE)
    assert store.get(failed_id)["_print_count"] == 1

    window.navigate("list")
    pump(app)
    assert window.page("list").counts()["printed"] == 6

    # 기록에는 두 번의 인쇄가 따로, 실패한 사람의 이름과 이유와 함께 남는다
    history_page = window.page("history")
    window.navigate("history")
    statuses = [job.status for job in window.services.history.list_jobs()]
    assert statuses == ["completed", "partial"]
    first = window.services.history.list_jobs()[1]
    assert history_page.show_job(first.id)
    failed_rows = [row for row in history_page.card_rows() if row["result"] == "실패"]
    assert [row["person"] for row in failed_rows] == [PRINTABLE[2]] and "리본" in failed_rows[0]["reason"]


def test_stop_after_this_card_then_continue_keeps_serial_numbers(app_env, app_window, people_csv, photo_folder,
                                                                 monkeypatch):
    import threading

    from design.binding import CounterSource
    from printer import registry
    from printer.rtai_adapter import RtaiAdapter
    from services import render_job as render_job_module

    window, app = app_window, app_env
    page = journey_to_print(app, window, people_csv, photo_folder)
    store = window.workspace.store

    # 디자인에 일련번호를 넣는다(1번부터).
    window.navigate("design")
    editor = window.workspace.editor
    serial = editor.add_item("text", pos=(20.0, 6.0))
    editor.set_source([serial.id], CounterSource(start=1))
    print_page_ready(app, window)

    # 카드마다 어떤 일련번호로 그렸는지 적어 둔다.
    rendered: list[tuple[int | None, int]] = []
    real_render = render_job_module.RenderJob._render

    def spy(self, spec):
        rendered.append((spec.record_id, spec.counter_index))
        return real_render(self, spec)

    monkeypatch.setattr(render_job_module.RenderJob, "_render", spy)

    # 세 번째 카드에서 프린터를 붙잡아 두고 "이 카드까지만"을 누른다.
    entered, release = threading.Event(), threading.Event()
    count = {"n": 0}

    class Held(RtaiAdapter):
        def print_card(self, job, progress=None):
            count["n"] += 1
            if count["n"] == 3:
                job.front
                entered.set()
                release.wait(20)
            return super().print_card(job, progress)

    registry.register_adapter("rtai", lambda device: Held(lib=registry._fake_library("rtai"), settle_seconds=0.0))

    _exclude_missing_and_start(app, page)
    wait_until(app, entered.is_set)
    click(page.run.stop_after_button)
    assert page.run.stop_note.isVisibleTo(page.run)
    release.set()
    wait_done(app, page)

    states = [page.run.row_state(i) for i in range(6)]
    assert states == [ROW_SUCCESS] * 3 + [ROW_CANCELLED] * 3
    assert page.run.summary_headline.text() == "인쇄를 멈췄습니다"
    assert page.run.continue_button.text() == "남은 3장 이어서 인쇄"
    assert sum(1 for r in store.rows() if r["_printed"]) == 3

    click(page.run.continue_button)
    assert page.run.title.text() == "이어서 뽑는 3장"
    assert [page.run.model.row(i)["name"] for i in range(3)] == PRINTABLE[3:]
    wait_done(app, page)
    assert page.run.summary_headline.text() == "3장을 모두 뽑았습니다"
    assert {r["이름"] for r in store.rows() if r["_printed"]} == set(PRINTABLE)

    # 이어서 뽑은 카드는 처음 계획했던 번호 그대로다(4, 5, 6번째 사람은 3, 4, 5번).
    by_person = {}
    for record_id, counter in rendered:
        by_person.setdefault(store.get(record_id)["이름"], set()).add(counter)
    assert [by_person[name] for name in PRINTABLE] == [{0}, {1}, {2}, {3}, {4}, {5}]
    assert all(r["_print_count"] == 1 for r in store.rows() if r["_printed"])
