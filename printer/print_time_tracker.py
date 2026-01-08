"""
인쇄 시간 추적기 - 실제 인쇄 시간을 기록하고 예상 시간을 계산
"""
import json
import time
import os
from typing import Optional


class PrintTimeTracker:
    """인쇄 시간 추적 및 예상 시간 계산"""

    # 기본 예상 시간 (초) - 데이터 없을 때 사용
    DEFAULT_SINGLE_TIME = 40
    DEFAULT_DUPLEX_TIME = 75  # 1분 15초

    # 최대 기록 수
    MAX_RECORDS = 50

    def __init__(self):
        self._data_file = self._get_data_file_path()
        self._records = []
        self._current_print_start: Optional[float] = None
        self._current_print_info: dict = {}
        self._load_records()

    def _get_data_file_path(self) -> str:
        """데이터 파일 경로 반환 - ProgramData\HanaStudio\temp 사용"""
        program_data = os.environ.get('ProgramData', 'C:\\ProgramData')
        data_dir = os.path.join(program_data, 'HanaStudio', 'temp')
        os.makedirs(data_dir, exist_ok=True)
        return os.path.join(data_dir, 'print_times.json')

    def _load_records(self):
        """JSON 파일에서 기록 로드"""
        try:
            if os.path.exists(self._data_file):
                with open(self._data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._records = data.get('records', [])
        except Exception as e:
            print(f"[PrintTimeTracker] 기록 로드 실패: {e}")
            self._records = []

    def _save_records(self):
        """JSON 파일에 기록 저장"""
        try:
            # 최대 기록 수 제한
            if len(self._records) > self.MAX_RECORDS:
                self._records = self._records[-self.MAX_RECORDS:]

            with open(self._data_file, 'w', encoding='utf-8') as f:
                json.dump({'records': self._records}, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[PrintTimeTracker] 기록 저장 실패: {e}")

    def start_print(self, is_duplex: bool = False, print_mode: str = "normal"):
        """인쇄 시작 시간 기록"""
        self._current_print_start = time.time()
        self._current_print_info = {
            'is_duplex': is_duplex,
            'print_mode': print_mode
        }

    def end_print(self, success: bool = True):
        """인쇄 완료 - 실제 소요 시간 기록"""
        if self._current_print_start is None:
            return

        if success:
            duration = time.time() - self._current_print_start
            record = {
                'timestamp': int(time.time()),
                'duration': round(duration, 1),
                'is_duplex': self._current_print_info.get('is_duplex', False),
                'print_mode': self._current_print_info.get('print_mode', 'normal')
            }
            self._records.append(record)
            self._save_records()

        # 초기화
        self._current_print_start = None
        self._current_print_info = {}

    def get_estimated_time(self, is_duplex: bool = False, print_mode: str = "normal") -> float:
        """
        예상 시간 반환 (초 단위)
        - 최근 10개 기록의 평균값 사용
        - 기록 없으면 기본값 사용
        """
        # 같은 조건의 기록 필터링
        matching_records = [
            r for r in self._records
            if r.get('is_duplex') == is_duplex and r.get('print_mode') == print_mode
        ]

        if not matching_records:
            # 기록 없으면 기본값
            return self.DEFAULT_DUPLEX_TIME if is_duplex else self.DEFAULT_SINGLE_TIME

        # 최근 10개 평균
        recent = matching_records[-10:]
        avg_time = sum(r['duration'] for r in recent) / len(recent)
        return round(avg_time, 1)

    def get_estimated_time_formatted(self, quantity: int = 1, is_duplex: bool = False,
                                      print_mode: str = "normal") -> str:
        """
        포맷된 예상 시간 문자열 반환
        예: "약 26초", "약 1분 18초"
        """
        per_card = self.get_estimated_time(is_duplex, print_mode)
        total_seconds = int(per_card * quantity)

        if total_seconds < 60:
            return f"약 {total_seconds}초"
        else:
            minutes = total_seconds // 60
            seconds = total_seconds % 60
            if seconds == 0:
                return f"약 {minutes}분"
            else:
                return f"약 {minutes}분 {seconds}초"

    def get_statistics(self) -> dict:
        """통계 정보 반환"""
        if not self._records:
            return {
                'total_prints': 0,
                'single_prints': 0,
                'duplex_prints': 0,
                'avg_single': self.DEFAULT_SINGLE_TIME,
                'avg_duplex': self.DEFAULT_DUPLEX_TIME,
            }

        single_records = [r for r in self._records if not r.get('is_duplex')]
        duplex_records = [r for r in self._records if r.get('is_duplex')]

        return {
            'total_prints': len(self._records),
            'single_prints': len(single_records),
            'duplex_prints': len(duplex_records),
            'avg_single': round(sum(r['duration'] for r in single_records) / len(single_records), 1) if single_records else self.DEFAULT_SINGLE_TIME,
            'avg_duplex': round(sum(r['duration'] for r in duplex_records) / len(duplex_records), 1) if duplex_records else self.DEFAULT_DUPLEX_TIME,
            'min_time': min(r['duration'] for r in self._records),
            'max_time': max(r['duration'] for r in self._records),
        }

    def clear_records(self):
        """기록 초기화"""
        self._records = []
        self._save_records()


# 싱글톤 인스턴스
print_time_tracker = PrintTimeTracker()
