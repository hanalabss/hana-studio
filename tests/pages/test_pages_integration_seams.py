"""화면 사이 이음매: 작업 상태 신호, 설정 기본값, 단계 문구, 남은 양, 디자인 화면의 미리 보기 줄."""
from print_helpers import *  # noqa: F401,F403
from print_helpers import finish_job, make_page, pump

from data.filters import Condition
from data.store import DataStore
from printer.base import UNIT_CARDS, UNIT_PERCENT, UNIT_UNKNOWN
from ui.pages.printing.text import plain, stage_sentence, supply_text


# ---------------------------------------------------------------- Workspace

def test_set_store_marks_follow_up_signals_and_resets_the_view(env, workspace, tmp_path):
    seen: list[tuple[str, bool]] = []
    workspace.store_changed.connect(lambda: seen.append(("store", workspace.store_opening)))
    workspace.columns_changed.connect(lambda: seen.append(("columns", workspace.store_opening)))
    workspace.records_changed.connect(lambda: seen.append(("records", workspace.store_opening)))
    workspace.view_changed.connect(lambda: seen.append(("view", workspace.store_opening)))

    workspace.set_view(Condition("부서", "eq", "개발부"), [("이름", "desc")])
    assert seen == [("view", False)]
    workspace.set_view(Condition("부서", "eq", "개발부"), [("이름", "desc")])      # 같은 값이면 조용하다
    assert seen == [("view", False)]
    workspace.view_order = None                                                  # 예전처럼 속성에 넣어도 알린다
    assert seen[-1] == ("view", False) and len(seen) == 2

    seen.clear()
    other = DataStore(tmp_path / "other.hsdata")
    other.add_column("성함")
    workspace.set_store(other)
    assert seen == [("store", False), ("columns", True), ("records", True), ("view", False)]
    assert workspace.view_filter is None and workspace.view_order is None       # 앞 명단의 조건을 끌고 가지 않는다
    assert not workspace.store_opening

    seen.clear()
    workspace.notify_columns_changed()
    workspace.notify_records_changed()
    assert seen == [("columns", False), ("records", False)]


def test_design_preview_bar_follows_list_filter_and_sort(env, workspace):
    from ui.pages.design.preview_bar import PreviewBar

    bar = PreviewBar(workspace)
    store = workspace.store
    assert bar.record_ids() == store.ids()

    developers = Condition("부서", "eq", "개발부")
    workspace.set_view(developers, None)
    assert bar.record_ids() == store.ids(developers) and len(bar.record_ids()) == 2
    workspace.set_current_record(bar.record_ids()[1])                 # 박민수
    assert bar.position_label.text().startswith("2 / 2")

    workspace.set_view(developers, [("이름", "asc")])                   # 가나다순이면 박민수가 먼저다
    assert bar.record_ids() == store.ids(developers, [("이름", "asc")])
    assert bar.position_label.text().startswith("1 / 2")

    workspace.set_view(None, None)
    assert len(bar.record_ids()) == 5


# ---------------------------------------------------------------- 설정 기본값

def test_changing_settings_does_not_reset_choices_made_for_this_print(env, workspace, services):
    page = make_page(env, workspace, services)
    prefs = page.prefs
    choices = page.choices().clone()
    choices.options.copies = 3
    choices.options.offset_x_mm = 1.5
    page.apply_choices(choices)
    assert page.card_count() == 15

    prefs.set_retention_days(30)                    # 인쇄와 상관없는 설정
    prefs.set_stop_on_error(False)                  # 이번 인쇄에서 건드리지 않은 값은 따라간다
    assert page.choices().options.copies == 3 and page.choices().options.offset_x_mm == 1.5
    assert page.choices().options.stop_on_error is False

    prefs.set_copies(2)                             # 이번 인쇄에서 따로 정한 장수는 그대로다
    assert page.choices().options.copies == 3 and page.card_count() == 15

    fresh = make_page(env, workspace, services, find_printers=False)      # 새로 시작하는 인쇄는 기본값을 쓴다
    assert fresh.choices().options.copies == 2 and fresh.choices().options.stop_on_error is False
    prefs.set_copies(4)
    assert fresh.choices().options.copies == 4      # 손대지 않았으니 바뀐 기본값을 따른다
    assert page.choices().options.copies == 3


def test_settings_change_during_a_job_does_not_touch_it(env, workspace, services):
    page = make_page(env, workspace, services)
    handle = page.start_printing()
    assert handle is not None
    page.prefs.set_copies(5)
    finish_job(env, page)
    assert handle.total == 5 and handle.summary.success == 5


# ---------------------------------------------------------------- 말

def test_stage_sentences_come_from_stage_keys():
    assert stage_sentence("feed", "카드 삽입 중...") == "카드를 프린터 안으로 넣고 있습니다"
    assert stage_sentence("print_back", "뒷면 인쇄 중...", "back") == "뒷면을 찍고 있습니다"
    assert stage_sentence("next_card", "다음 카드 준비 중...") == "다음 카드를 준비하고 있습니다"
    assert stage_sentence("next_card", "카드 인쇄 실패") == "이 카드를 찍지 못했습니다"
    assert stage_sentence("", "K 패널 인쇄 중...") == "K 패널 인쇄 중"            # 단계 키가 없으면 받은 말 그대로
    assert stage_sentence("made-up-key", "무언가 하는 중") == "무언가 하는 중"


def test_run_view_shows_stage_sentence_for_the_current_card(env, workspace, services):
    page = make_page(env, workspace, services)
    shown: list[str] = []
    services.printing.progress.connect(lambda _progress: shown.append(page.run.stage_text.text()))
    page.start_printing()
    finish_job(env, page)
    assert "카드를 찍고 있습니다" in shown and "카드를 내보내고 있습니다" in shown
    assert not any(text.endswith("...") or "SDK" in text for text in shown)


def test_remaining_supply_is_only_called_cards_when_the_printer_counts_cards():
    assert supply_text(120, UNIT_CARDS) == "약 120장 남음"
    assert supply_text(120, UNIT_CARDS, spent=20) == "약 100장 남음"
    assert supply_text(62, UNIT_PERCENT) == "62% 남음"
    unknown = supply_text(250, UNIT_UNKNOWN)
    assert "장" not in unknown and "250" in unknown
    assert supply_text(None, UNIT_CARDS) == "" and supply_text(0, UNIT_UNKNOWN) == "다 썼습니다"


def test_rtai_counter_is_never_shown_as_cards_on_print_or_printer_page(env, workspace, services, rtai_lib):
    from ui.pages.printer.probe import CheckLine  # noqa: F401  (같은 말을 쓰는지 아래에서 본다)

    page = make_page(env, workspace, services)
    status = services.printers.status()
    assert status.ribbon_unit == UNIT_UNKNOWN and status.ribbon_cards_left is None
    supplies = page.ready.printer_supplies.text() if hasattr(page.ready, "printer_supplies") else ""
    text = supply_text(status.ribbon_remaining, status.ribbon_unit)
    assert "장" not in text
    assert "약" not in supplies or "장 남음" not in supplies


def test_service_messages_already_use_house_words(env, workspace, services):
    from data.selection import parse_ranges
    from data.errors import RangeError
    from services import preflight

    store = workspace.store
    store.rename_column("부서", "소속")
    report = preflight.check(workspace.editor.document, store, store.ids()[:2] + [999])
    messages = [finding.message for finding in report.findings] if hasattr(report, "findings") else \
        [finding.message for finding in report.errors + report.warnings]
    try:
        parse_ranges("9", 5)
    except RangeError as exc:
        messages.append(str(exc))
    assert messages
    for message in messages:
        assert plain(message) == message.strip()
        for word in ("레코드", "' 열", "열을", "열이", "인쇄 옵션", "인쇄 대상", "화이트 레이어"):
            assert word not in message, message
    assert any("지워진 사람이 1명" in message for message in messages)
    assert any("'부서' 항목이 명단에 없습니다" in message for message in messages)


# ---------------------------------------------------------------- 막다른 길

def test_blank_design_on_print_page_points_back_to_design(env, workspace, services):
    from design.model import Document

    class Shell:
        def __init__(self):
            self.went = []

        def navigate(self, page_id):
            self.went.append(page_id)

        def show_toast(self, *args, **kwargs):
            pass

        def set_printer_status(self, *args, **kwargs):
            pass

        def set_step_done(self, *args, **kwargs):
            pass

    page = make_page(env, workspace, services)
    assert page.ready.finding("blank_design") is None
    workspace.editor.set_document(Document(name="새 디자인"))
    page.refresh_now()
    row = page.ready.finding("blank_design")
    assert row is not None and row.problem.blocking
    assert "카드에 아직 아무것도 없습니다" in row.text()
    assert not page.ready.start_button.isEnabled()
    shell = Shell()
    page._shell = shell
    row.button("open_design").click()
    assert shell.went == ["design"]


def test_field_tint_steps_aside_while_previewing_a_real_person(env, workspace):
    from ui.editor import EditorView

    editor = workspace.editor
    view = EditorView(editor)
    view.resize(700, 500)
    view.show()
    env.processEvents()
    name = next(item for _side, item in editor.document.iter_items() if item.name == "이름")

    def corner_colour():
        env.processEvents()
        image = view.viewport().grab().toImage()
        outline = view._outline(name).boundingRect()
        return image.pixelColor(int(outline.right() - 6), int(outline.top() + 6))

    workspace.set_current_record(None)
    tinted = corner_colour()
    assert tinted.blue() < tinted.red() - 15, "칸 이름만 보일 때는 옅은 노랑으로 알려 준다"
    workspace.set_current_record(workspace.store.ids()[0])
    plain_colour = corner_colour()
    assert abs(plain_colour.blue() - plain_colour.red()) < 6, "실제 사람을 볼 때는 나올 모습 그대로다"
    view.close()
