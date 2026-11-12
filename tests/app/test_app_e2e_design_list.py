"""디자인 저장·다시 열기, 사람별 사진 맞추기가 인쇄에 그대로 쓰이는지, 다른 화면에서 명단으로 건너가기."""
from app_helpers import *  # noqa: F401,F403
from app_helpers import (
    MISSING_PHOTO, PEOPLE, build_window, click, click_footer, fake_library, find_printers, id_of, import_people,
    journey_to_print, names_in, pick_template, pump, wait_done, wait_until,
)

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage

from data.store import PhotoAdjustment
from design.binding import columns_used


PHOTO = (0.07, 0.31, 0.30, 0.90)        # 사원증(가로) 앞면에서 사진 칸 안쪽 (왼쪽, 위, 오른쪽, 아래 비율)


def _difference(a: QImage, b: QImage, region=PHOTO) -> float:
    """같은 크기로 맞춘 두 그림의 사진 칸 안쪽 평균 색 차이(0~255).

    바코드와 글자는 해상도가 다르면 가장자리가 달라지므로, 큰 색 면인 사진 칸만 견준다.
    """
    width, height = 300, max(1, round(300 * a.height() / a.width()))
    a = a.convertToFormat(QImage.Format_RGB888).scaled(width, height, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
    b = b.convertToFormat(QImage.Format_RGB888).scaled(width, height, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
    left, top, right, bottom = (round(region[0] * width), round(region[1] * height),
                                round(region[2] * width), round(region[3] * height))
    total = count = 0
    for y in range(top, bottom):
        for x in range(left, right):
            pa, pb = a.pixelColor(x, y), b.pixelColor(x, y)
            total += abs(pa.red() - pb.red()) + abs(pa.green() - pb.green()) + abs(pa.blue() - pb.blue())
            count += 3
    return total / max(1, count)


def test_saved_design_reopens_with_list_and_photo_adjustment_reaches_print(
        app_env, app_window, people_csv, photo_folder, tmp_path):
    from ui.pages.people.preview import render_card_image as list_preview
    from ui.pages.printing.preview import render_card_image as print_render

    window, app = app_window, app_env
    pick_template(window, "employee")
    click_footer(window, "다음: 명단 넣기")
    import_people(app, window, people_csv, photo_folder)
    people, workspace = window.page("list"), window.workspace
    store = workspace.store
    list_path = store.path
    person = id_of(store, "강민서")

    # 명단 화면에서 이 사람의 사진만 키우고 아래로 내린다
    people.reveal(person)
    pump(app)
    assert workspace.current_record_id == person
    before = print_render(workspace.editor.document, store, person, dpi=150)
    people.preview.set_adjustment(PhotoAdjustment(zoom=1.8, offset_x=0.0, offset_y=0.3))
    saved = store.get_photo_adjustment(person, "사진")
    assert saved.zoom == 1.8 and saved.offset_y > 0
    other = id_of(store, "고은채")
    assert store.get_photo_adjustment(other, "사진", default=None) is None        # 다른 사람은 그대로

    # 디자인을 저장한다
    design = window.page("design")
    target = tmp_path / "우리회사 사원증.hsd"
    design.dialogs.save_design = lambda parent, suggested: str(target)
    window.navigate("design")
    assert design.save() and target.exists()
    bindings = {item.name: (item.kind, sorted(item.source.columns()))
                for _side, item in workspace.editor.document.iter_items() if item.source is not None}
    used = columns_used(workspace.editor.document)
    assert {"이름", "사번", "사진"} <= used

    # 닫고, 새로 켜서 디자인과 명단을 다시 연다
    window.shutdown()
    window.close()
    pump(app, 30)
    reopened, services = build_window(tmp_path, "history2.sqlite")
    try:
        reopened.resize(1280, 800)
        reopened.show()
        design2, people2, ws2 = reopened.page("design"), reopened.page("list"), reopened.workspace
        assert design2.open_design(str(target)) and design2.is_started()
        reopened.navigate("list")
        assert people2.open_list(list_path)
        pump(app)
        document, store2 = ws2.editor.document, ws2.store
        assert columns_used(document) == used
        assert {item.name: (item.kind, sorted(item.source.columns()))
                for _side, item in document.iter_items() if item.source is not None} == bindings
        assert names_in(people2) == [name for name, *_ in PEOPLE]
        assert not people2.issues() or all(issue.kind != "unmatched_field" for issue in people2.issues())
        kept = store2.get_photo_adjustment(person, "사진")
        assert (kept.zoom, kept.offset_x, kept.offset_y) == (saved.zoom, saved.offset_x, saved.offset_y)

        # 명단 화면에 보이는 카드와 인쇄에 쓰는 그림이 같다(사진을 옮긴 그대로)
        record = store2.get(person)
        shown = list_preview(document, "front", ws2.binding_record(record), store=store2, record_id=person,
                             width_px=600)
        printed = print_render(document, store2, person, dpi=150)
        assert printed is not None and not shown.isNull()
        assert _difference(shown, printed) < 2.0, "명단에서 본 카드와 인쇄할 카드가 다릅니다"
        assert _difference(before, printed) > 8.0, \
            "사진을 옮긴 것이 인쇄 그림에 반영되지 않았습니다"
        unadjusted_shown = list_preview(document, "front", ws2.binding_record(store2.get(other)), store=store2,
                                        record_id=other, width_px=600)
        assert _difference(unadjusted_shown, print_render(document, store2, other, dpi=150)) < 2.0
    finally:
        reopened.shutdown()
        wait_until(app, lambda: not services.printers.is_busy)
        reopened.close()
        reopened.deleteLater()


def test_print_page_fix_in_list_shows_only_those_people(app_env, app_window, people_csv, photo_folder):
    window, app = app_window, app_env
    page = journey_to_print(app, window, people_csv, photo_folder)
    click(page.ready.finding("missing_photo").button("show_people"))
    pump(app)

    people = window.page("list")
    assert window.current_page_id() == "list"
    assert names_in(people) == list(MISSING_PHOTO)
    assert people.people_banner.isVisibleTo(people) and "2명만 보고 있습니다" in people.people_banner.text()
    assert "8명 중 2명 보는 중" in people.stats_label.text()
    assert len(window.workspace.checked_ids) == 8              # 체크는 그대로다

    click(people.people_banner.action_button())
    assert len(names_in(people)) == 8 and not people.people_banner.isVisibleTo(people)
    assert people.shown_people() is None and window.workspace.view_filter is None


def test_history_failed_people_jump_to_list_and_back_to_everyone(app_env, app_window, people_csv, photo_folder):
    from printer.fakes import R600_ERR_PRINT

    window, app = app_window, app_env
    page = journey_to_print(app, window, people_csv, photo_folder)
    choices = page.choices().clone()
    choices.options.stop_on_error = False
    page.apply_choices(choices)
    fake_library("rtai").fail("R600PrintDraw", R600_ERR_PRINT, on_call=2)
    click(page.ready.finding("missing_photo").button("exclude"))
    pump(app, 30)
    if page.ready.acknowledge.isVisibleTo(page.ready):
        page.acknowledge_warnings(True)
    click(page.ready.start_button)
    wait_done(app, page)

    click(page.run.history_button)
    history = window.page("history")
    assert window.current_page_id() == "history" and history.selected_job_id() == page.job().history_id
    assert history.show_failed_button.text() == "실패한 1명만 명단에서 보기"
    click(history.show_failed_button)
    pump(app)

    people = window.page("list")
    failed_name = [name for name, *_ in PEOPLE if name not in MISSING_PHOTO][1]
    assert window.current_page_id() == "list"
    assert names_in(people) == [failed_name]
    assert people.people_banner.isVisibleTo(people)

    # 찾기를 지우는 "전체 보기"도 좁혀 보기를 함께 푼다
    people.clear_filters()
    assert len(names_in(people)) == 8 and not people.people_banner.isVisibleTo(people)

    # 명단을 닫으면 기록의 이름은 그대로 남고, 다른 명단을 열어도 엉뚱한 이름이 붙지 않는다
    window.navigate("history")
    people.close_list()
    pump(app)
    rows = history.card_rows()
    assert rows[1]["person"] == failed_name and rows[1]["result"] == "실패"
    assert not history.show_failed_button.isVisibleTo(history)


def test_history_names_do_not_follow_another_list(app_env, app_window, people_csv, photo_folder, tmp_path):
    from data.store import DataStore

    window, app = app_window, app_env
    page = journey_to_print(app, window, people_csv, photo_folder)
    click(page.ready.finding("missing_photo").button("exclude"))
    pump(app, 30)
    if page.ready.acknowledge.isVisibleTo(page.ready):
        page.acknowledge_warnings(True)
    click(page.ready.start_button)
    wait_done(app, page)
    printed_names = [name for name, *_ in PEOPLE if name not in MISSING_PHOTO]

    # 같은 번호(_id)를 쓰는 전혀 다른 명단을 연다
    other = DataStore(tmp_path / "다른 명단.hsdata")
    other.add_column("이름")
    other.insert_many([{"이름": f"다른사람{i}"} for i in range(1, 9)])
    other.close()
    window.navigate("list")
    assert window.page("list").open_list(tmp_path / "다른 명단.hsdata")
    pump(app)

    history = window.page("history")
    window.navigate("history")
    assert history.show_job(page.job().history_id)
    assert [row["person"] for row in history.card_rows()] == printed_names
    assert "신입사원_명단" in history.names_note.text() and "명단으로 뽑은 기록" in history.names_note.text()
    assert not history.reprint_button.isVisibleTo(history)          # 다른 명단의 사람을 다시 뽑게 하지 않는다
    history.set_search("강민서")
    assert [row["id"] for row in history.job_rows()] == [page.job().history_id]     # 이름으로도 찾는다
    history.set_search("다른사람1")
    assert history.job_rows() == []
