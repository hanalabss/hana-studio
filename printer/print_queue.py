"""
printer/print_queue.py
인쇄 큐 시스템 - 여러 인쇄 요청을 순차적으로 처리하고 모든 작업 완료 시 알림
"""

from dataclasses import dataclass, field
from typing import Optional, Callable, List
from PySide6.QtCore import QObject, Signal
from .printer_thread import PrinterThread


@dataclass
class PrintJob:
    """단일 인쇄 작업 정보"""
    job_id: int
    dll_path: str
    front_image_path: str
    back_image_path: Optional[str] = None
    front_mask_path: Optional[str] = None
    back_mask_path: Optional[str] = None
    print_mode: str = "normal"
    is_dual_side: bool = True
    quantity: int = 1
    front_orientation: str = "portrait"
    back_orientation: str = "portrait"
    adjusted_x: float = 0.0
    adjusted_y: float = 0.0

    # 작업 상태
    status: str = "pending"  # pending, printing, completed, failed
    completed_cards: int = 0

    def get_total_cards(self) -> int:
        """이 작업의 총 카드 수"""
        return self.quantity


class PrintQueueManager(QObject):
    """
    인쇄 큐 관리자
    - 여러 인쇄 요청을 큐에 추가
    - 순차적으로 처리
    - 모든 작업 완료 시 all_jobs_completed 시그널 발송
    """

    # 시그널 정의
    job_added = Signal(int, int)  # job_id, queue_size
    job_started = Signal(int, int, int)  # job_id, job_index, total_jobs
    job_progress = Signal(int, str)  # overall_progress (%), status_message
    job_completed = Signal(int, bool)  # job_id, success
    all_jobs_completed = Signal(int, int, int, int, str)  # success_cards, failed_cards, success_jobs, failed_jobs, last_error
    queue_updated = Signal(int)  # remaining_jobs
    log_message = Signal(str)  # 로그 메시지

    def __init__(self):
        super().__init__()
        self._queue: List[PrintJob] = []
        self._current_job: Optional[PrintJob] = None
        self._current_thread: Optional[PrinterThread] = None
        self._job_counter = 0
        self._is_processing = False

        # 완료된 작업 히스토리
        self._job_history: List[PrintJob] = []

        # 통계
        self._completed_jobs = 0
        self._failed_jobs = 0
        self._total_cards_printed = 0
        self._total_cards_failed = 0
        self._last_error_message = ""

    def add_job(self, **kwargs) -> int:
        """
        새 인쇄 작업을 큐에 추가

        Returns:
            job_id: 추가된 작업의 ID
        """
        self._job_counter += 1
        job_id = self._job_counter

        # PrintJob 생성
        job = PrintJob(
            job_id=job_id,
            dll_path=kwargs.get('dll_path', ''),
            front_image_path=kwargs.get('front_image_path', ''),
            back_image_path=kwargs.get('back_image_path'),
            front_mask_path=kwargs.get('front_mask_path'),
            back_mask_path=kwargs.get('back_mask_path'),
            print_mode=kwargs.get('print_mode', 'normal'),
            is_dual_side=kwargs.get('is_dual_side', True),
            quantity=kwargs.get('quantity', 1),
            front_orientation=kwargs.get('front_orientation', 'portrait'),
            back_orientation=kwargs.get('back_orientation', 'portrait'),
            adjusted_x=kwargs.get('adjusted_x', 0.0),
            adjusted_y=kwargs.get('adjusted_y', 0.0)
        )

        self._queue.append(job)
        self.job_added.emit(job_id, len(self._queue))
        self.log_message.emit(f"[QUEUE] 인쇄 작업 #{job_id} 추가됨 (대기열: {len(self._queue)}개)")

        # 처리 중이 아니면 시작
        if not self._is_processing:
            self._process_next()

        return job_id

    def get_queue_status(self) -> dict:
        """현재 큐 상태 반환"""
        return {
            'is_processing': self._is_processing,
            'queue_size': len(self._queue),
            'current_job_id': self._current_job.job_id if self._current_job else None,
            'completed_jobs': self._completed_jobs,
            'failed_jobs': self._failed_jobs,
            'total_cards_printed': self._total_cards_printed,
            'total_cards_in_queue': self.get_total_cards_in_queue()
        }

    def get_total_cards_in_queue(self) -> int:
        """전체 큐의 총 카드 수 (대기 + 현재 작업)"""
        total = sum(job.quantity for job in self._queue)
        if self._current_job:
            total += self._current_job.quantity
        return total

    def get_completed_cards_count(self) -> int:
        """완료된 카드 수 (성공 + 실패 포함)"""
        completed = self._total_cards_printed + self._total_cards_failed
        if self._current_job:
            completed += self._current_job.completed_cards
        return completed

    def get_overall_progress(self) -> tuple:
        """
        전체 진행률 계산 (카드 기준)

        Returns:
            (progress_percent, status_message)
        """
        if not self._is_processing and len(self._queue) == 0:
            return (0, "대기 중")

        # 전체 카드 수 계산 (이미 완료된 것 + 대기 중 + 현재 작업)
        total_cards = self._total_cards_printed + self._total_cards_failed + self.get_total_cards_in_queue()

        if total_cards == 0:
            return (0, "대기 중")

        # 완료된 카드 수
        completed_cards = self.get_completed_cards_count()

        overall = int((completed_cards / total_cards) * 100)

        # 직관적인 상태 메시지: "N/M장 - 대기 중" 또는 "N/M장 - 인쇄 준비"
        status = f"{completed_cards}/{total_cards}장 - 인쇄 준비"

        return (overall, status)

    def cancel_all(self):
        """모든 대기 작업 취소 및 현재 작업 중단"""
        # 대기 중인 작업 모두 제거
        cancelled_count = len(self._queue)
        self._queue.clear()

        # 현재 진행 중인 작업 중단
        if self._current_thread and self._is_processing:
            self._current_thread.stop_printing()
            try:
                self._current_thread.wait(5000)  # 최대 5초 대기
            except Exception:
                pass

        self.log_message.emit(f"[QUEUE] {cancelled_count}개 대기 작업 취소됨")
        self.queue_updated.emit(0)

    def _process_next(self):
        """다음 작업 처리"""
        if len(self._queue) == 0:
            # 모든 작업 완료
            self._finish_all_jobs()
            return

        self._is_processing = True
        self._current_job = self._queue.pop(0)
        self._current_job.status = "printing"

        # 현재 작업 인덱스 계산
        current_index = self._completed_jobs + self._failed_jobs + 1
        total_jobs = current_index + len(self._queue)

        self.job_started.emit(self._current_job.job_id, current_index, total_jobs)
        self.log_message.emit(f"[QUEUE] 작업 #{self._current_job.job_id} 시작 ({current_index}/{total_jobs})")

        # PrinterThread 생성 및 시작
        self._current_thread = PrinterThread(
            dll_path=self._current_job.dll_path,
            front_image_path=self._current_job.front_image_path,
            back_image_path=self._current_job.back_image_path,
            front_mask_path=self._current_job.front_mask_path,
            back_mask_path=self._current_job.back_mask_path,
            print_mode=self._current_job.print_mode,
            is_dual_side=self._current_job.is_dual_side,
            quantity=self._current_job.quantity,
            front_orientation=self._current_job.front_orientation,
            back_orientation=self._current_job.back_orientation,
            adjusted_x=self._current_job.adjusted_x,
            adjusted_y=self._current_job.adjusted_y
        )

        # 시그널 연결
        self._current_thread.progress.connect(self._on_thread_progress)
        self._current_thread.finished.connect(self._on_thread_finished)
        self._current_thread.error.connect(self._on_thread_error)
        self._current_thread.card_completed.connect(self._on_card_completed)
        self._current_thread.step_progress.connect(self._on_step_progress)

        self._current_thread.start()

    def _on_thread_progress(self, message: str):
        """스레드 진행 메시지 - 오류 메시지도 캡처"""
        self.log_message.emit(message)

        # 오류 메시지 캡처 (실패, 오류 키워드 포함 시)
        if "실패" in message or "오류" in message:
            # 구체적인 오류 내용 추출 (예: "❌ 1번째 카드 인쇄 실패: 카드 삽입 실패: 오류 코드 8421378")
            if ":" in message:
                # 첫 번째 콜론 이후의 내용이 실제 오류 원인
                parts = message.split(":", 1)
                if len(parts) > 1:
                    error_detail = parts[1].strip()
                    # "오류 코드" 부분 제거 - 핵심 원인만 추출
                    if ":" in error_detail:
                        error_detail = error_detail.split(":")[0].strip()

                    # 유효한 오류 메시지인지 확인
                    is_valid = (
                        error_detail and
                        len(error_detail) > 3 and
                        not error_detail.isdigit() and  # 숫자만 있는 경우 제외
                        "/" not in error_detail and     # 파일 경로 제외
                        "\\" not in error_detail
                    )

                    if is_valid:
                        self._last_error_message = error_detail
                    elif parts[0].strip():
                        # 유효하지 않으면 콜론 앞부분에서 핵심 추출
                        first_part = parts[0].strip()
                        # 이모지 제거 후 "실패" 또는 "오류" 포함된 부분 추출
                        if "실패" in first_part or "오류" in first_part:
                            # "❌ 1번째 카드 인쇄 실패" → "카드 인쇄 실패" 또는 전체
                            self._last_error_message = first_part.lstrip("❌⚠️ ").strip()
            # 콜론 없는 일반 메시지는 이미 구체적 오류가 없을 때만 저장
            elif not self._last_error_message:
                self._last_error_message = message

    def _on_card_completed(self, card_num: int):
        """개별 카드 완료 - completed_cards만 업데이트 (total은 작업 완료 시 반영)"""
        if self._current_job:
            self._current_job.completed_cards = card_num

        # 전체 진행률 업데이트
        progress, status = self.get_overall_progress()
        self.job_progress.emit(progress, status)

    def _on_step_progress(self, progress: int, message: str):
        """단계별 진행률 - 전체 큐 기준으로 재계산"""
        # SDK progress는 현재 작업 내 진행률 (0-100%)
        # 전체 큐 기준으로 재계산

        # 전체 카드 수 (완료 + 현재 작업 + 대기)
        total_cards = self._total_cards_printed + self._total_cards_failed + self.get_total_cards_in_queue()

        if total_cards == 0:
            self.job_progress.emit(progress, message)
            return

        # 현재 작업의 진행된 카드 수 (SDK progress 기준)
        current_job_cards = 0
        if self._current_job:
            # SDK progress는 현재 작업 전체의 진행률
            current_job_cards = (progress / 100) * self._current_job.quantity

        # 전체 완료된 카드 수
        completed_cards = self._total_cards_printed + self._total_cards_failed + current_job_cards

        # 전체 진행률
        overall_progress = int((completed_cards / total_cards) * 100)

        # 직관적인 메시지 생성 - SDK 메시지에서 [x/y] 부분 제거
        clean_message = message
        if message.startswith("[") and "]" in message:
            clean_message = message.split("]", 1)[1].strip()

        # "N/M장 - 상태" 형식으로 표시
        completed_int = int(completed_cards)
        full_message = f"{completed_int}/{total_cards}장 - {clean_message}"

        self.job_progress.emit(overall_progress, full_message)

    def _on_thread_finished(self, success: bool):
        """작업 완료"""
        job_id = None
        if self._current_job:
            job_id = self._current_job.job_id
            total_cards = self._current_job.quantity
            completed_cards = self._current_job.completed_cards

            # 완료된 카드 수 반영 (작업 완료 시점에 한 번에 추가)
            self._total_cards_printed += completed_cards

            if success:
                self._current_job.status = "completed"
                self._completed_jobs += 1
            else:
                self._current_job.status = "failed"
                self._failed_jobs += 1
                # 실패한 카드 수 계산 (전체 - 완료된 카드)
                failed_cards = total_cards - completed_cards
                self._total_cards_failed += failed_cards

            # 히스토리에 추가
            self._job_history.append(self._current_job)
            self.job_completed.emit(self._current_job.job_id, success)

        self._current_job = None

        # 스레드 정리 - wait()로 완전히 종료될 때까지 대기
        if self._current_thread:
            try:
                self._current_thread.wait(5000)  # 최대 5초 대기
            except Exception:
                pass
            self._current_thread = None

        # 남은 작업 수 알림
        self.queue_updated.emit(len(self._queue))

        # 다음 작업 처리
        self._process_next()

    def _on_thread_error(self, error_message: str):
        """작업 오류 - 마지막 오류 메시지 저장"""
        self._last_error_message = error_message
        # finished 시그널에서 처리됨

    def _finish_all_jobs(self):
        """모든 작업 완료 처리"""
        self._is_processing = False

        total_cards = self._total_cards_printed + self._total_cards_failed
        if self._total_cards_failed == 0:
            self.log_message.emit(f"[QUEUE] 모든 작업 완료! ({self._total_cards_printed}장 인쇄됨)")
        else:
            self.log_message.emit(f"[QUEUE] 작업 완료 ({total_cards}장 중 {self._total_cards_printed}장 인쇄됨)")

        # 최종 완료 시그널 발송 (카드 단위 + 작업 단위 + 오류 메시지)
        self.all_jobs_completed.emit(
            self._total_cards_printed,  # 성공한 카드 수
            self._total_cards_failed,   # 실패한 카드 수
            self._completed_jobs,       # 성공한 작업 수
            self._failed_jobs,          # 실패한 작업 수
            self._last_error_message    # 마지막 오류 메시지
        )

        # 통계 초기화
        self._reset_stats()

    def _reset_stats(self):
        """통계 초기화"""
        self._completed_jobs = 0
        self._failed_jobs = 0
        self._total_cards_printed = 0
        self._total_cards_failed = 0
        self._last_error_message = ""
        self._job_history.clear()

    def get_all_jobs_for_display(self) -> List[dict]:
        """
        대기열 표시용 전체 작업 목록 반환

        Returns:
            List[dict]: [{'job_id': int, 'quantity': int, 'status': str, 'display_status': str}, ...]
            display_status: '완료', '실패', '작업중', '대기'
        """
        jobs = []

        # 완료된 작업 (히스토리에서)
        for job in self._job_history:
            display_status = "완료" if job.status == "completed" else "실패"
            jobs.append({
                'job_id': job.job_id,
                'quantity': job.quantity,
                'status': job.status,
                'display_status': display_status,
                'completed_cards': job.completed_cards
            })

        # 현재 작업 중인 것
        if self._current_job:
            jobs.append({
                'job_id': self._current_job.job_id,
                'quantity': self._current_job.quantity,
                'status': 'printing',
                'display_status': '작업중',
                'completed_cards': self._current_job.completed_cards
            })

        # 대기 중인 작업
        for job in self._queue:
            jobs.append({
                'job_id': job.job_id,
                'quantity': job.quantity,
                'status': 'pending',
                'display_status': '대기',
                'completed_cards': 0
            })

        return jobs


# 전역 인스턴스
print_queue = PrintQueueManager()
