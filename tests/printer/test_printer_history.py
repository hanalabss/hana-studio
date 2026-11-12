import sqlite3

import pytest

from printer.base import PrinterDevice
from printer.history import JobHistory, default_history_path


@pytest.fixture
def history(tmp_path):
    return JobHistory(str(tmp_path / "sub" / "history.sqlite"))


@pytest.fixture
def device():
    return PrinterDevice("smart", "SMART-51", "usb", "SMART-51 (USB001)")


def test_creates_the_file_and_parent_directory(history):
    import os
    assert os.path.isfile(history.path)


def test_job_lifecycle_all_cards_succeed(history, device):
    job_id = history.start_job("사원증 3월", device, total_cards=2, options={"dual": True})
    job = history.get_job(job_id)
    assert (job.status, job.name, job.family, job.device_name, job.total_cards) == (
        "pending", "사원증 3월", "smart", "SMART-51", 2)
    assert job.options == {"dual": True} and job.started_at is None

    history.mark_job_started(job_id)
    assert history.get_job(job_id).status == "printing"
    history.record_card(job_id, 1, "success", record_id=101)
    history.record_card(job_id, 2, "success", record_id=102)
    assert history.finish_job(job_id) == "completed"

    job = history.get_job(job_id)
    assert (job.status, job.success_cards, job.failed_cards) == ("completed", 2, 0)
    assert job.started_at and job.finished_at and job.error_text is None


def test_card_results_keep_error_text_code_record_id_and_timestamps(history, device):
    job_id = history.start_job("테스트", device, total_cards=3)
    history.record_card(job_id, 1, "success", record_id="A-1", started_at="2026-01-01T10:00:00",
                        finished_at="2026-01-01T10:00:40")
    history.record_card(job_id, 2, "failed", error_text="리본을 모두 사용했습니다.",
                        error_code=0x8000001D, record_id="A-2")
    assert history.finish_job(job_id, error_text="리본을 모두 사용했습니다.") == "partial"

    cards = history.list_cards(job_id)
    assert [(c.card_index, c.status, c.record_id) for c in cards] == [
        (1, "success", "A-1"), (2, "failed", "A-2")]
    assert cards[0].started_at == "2026-01-01T10:00:00"
    assert cards[0].finished_at == "2026-01-01T10:00:40"
    assert cards[1].error_text == "리본을 모두 사용했습니다." and cards[1].error_code == 0x8000001D
    assert cards[1].finished_at

    job = history.get_job(job_id)
    assert (job.status, job.success_cards, job.failed_cards, job.error_text) == (
        "partial", 1, 1, "리본을 모두 사용했습니다.")


def test_finish_status_failed_and_explicit_cancelled(history, device):
    failed = history.start_job("실패", device, total_cards=1)
    history.record_card(failed, 1, "failed", error_text="카드 없음")
    assert history.finish_job(failed) == "failed"
    cancelled = history.start_job("취소", device, total_cards=5)
    history.record_card(cancelled, 1, "success")
    assert history.finish_job(cancelled, "cancelled") == "cancelled"
    with pytest.raises(KeyError):
        history.finish_job(9999)


def test_history_survives_reopening_the_file(history, device):
    job_id = history.start_job("재시작", device, total_cards=1)
    history.record_card(job_id, 1, "success", record_id=7)
    history.finish_job(job_id)
    reopened = JobHistory(history.path)
    assert reopened.get_job(job_id).status == "completed"
    assert reopened.list_cards(job_id)[0].record_id == "7"


def test_list_jobs_newest_first_with_paging_and_filters(history, device):
    other = PrinterDevice("rtai", "R600", "usb", "R600")
    first = history.start_job("명찰", device, total_cards=1)
    second = history.start_job("출입증", other, total_cards=1)
    third = history.start_job("명찰 재인쇄", device, total_cards=1)
    history.record_card(second, 1, "failed", error_text="덮개 열림")
    history.finish_job(second, error_text="덮개 열림")

    assert [j.id for j in history.list_jobs()] == [third, second, first]
    assert [j.id for j in history.list_jobs(limit=1, offset=1)] == [second]
    assert [j.id for j in history.list_jobs(status="failed")] == [second]
    assert [j.id for j in history.list_jobs(family="smart")] == [third, first]
    assert [j.id for j in history.list_jobs(search="명찰")] == [third, first]
    assert [j.id for j in history.list_jobs(search="덮개")] == [second]
    assert history.count_jobs() == 3 and history.count_jobs(search="명찰") == 2
    assert history.list_jobs(since="2999-01-01") == []
    assert len(history.list_jobs(until="2999-01-01")) == 3


def test_search_text_is_bound_as_a_parameter(history, device):
    history.start_job("a", device, total_cards=1)
    assert history.list_jobs(search="'; DROP TABLE jobs; --") == []
    assert history.count_jobs() == 1


def test_cards_for_record_finds_every_print_of_one_record(history, device):
    first = history.start_job("1차", device, total_cards=1)
    history.record_card(first, 1, "failed", record_id=55, error_text="걸림")
    second = history.start_job("2차", device, total_cards=1)
    history.record_card(second, 1, "success", record_id=55)
    assert [(c.job_id, c.status) for c in history.cards_for_record(55)] == [
        (second, "success"), (first, "failed")]


def test_delete_job_removes_its_cards(history, device):
    job_id = history.start_job("삭제", device, total_cards=1)
    history.record_card(job_id, 1, "success")
    history.delete_job(job_id)
    assert history.get_job(job_id) is None and history.list_cards(job_id) == []


def test_purge_older_than(history, device):
    old = history.start_job("오래된 작업", device, total_cards=1)
    recent = history.start_job("최근 작업", device, total_cards=1)
    with sqlite3.connect(history.path) as connection:
        connection.execute("UPDATE jobs SET created_at = '2001-01-01T00:00:00' WHERE id = ?", (old,))
    assert history.purge_older_than(30) == 1
    assert [j.id for j in history.list_jobs()] == [recent]


def test_close_interrupted_jobs_marks_unfinished_jobs_failed(history, device):
    pending = history.start_job("대기", device, total_cards=1)
    printing = history.start_job("인쇄 중", device, total_cards=1)
    history.mark_job_started(printing)
    done = history.start_job("완료", device, total_cards=1)
    history.record_card(done, 1, "success")
    history.finish_job(done)
    assert history.close_interrupted_jobs() == 2
    assert history.get_job(pending).status == "failed"
    assert history.get_job(printing).error_text == "프로그램이 인쇄 도중 종료되었습니다."
    assert history.get_job(done).status == "completed"


def test_job_without_device_and_default_path(history):
    job_id = history.start_job()
    assert history.get_job(job_id).family == ""
    assert default_history_path().endswith("print_history.sqlite")
