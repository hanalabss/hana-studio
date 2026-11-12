"""인쇄·기록·프린터 화면이 쓰는 말."""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Iterable

from printer.base import (
    CAP_DUAL_SIDE, CAP_MAGNETIC, CAP_UV, CAP_WHITE, FAMILY_RTAI, FAMILY_SMART, UNIT_CARDS, UNIT_PERCENT,
    UNIT_UNKNOWN, PrinterDevice, PrinterStatus,
)

CAPABILITY_WORDS: tuple[tuple[str, str], ...] = (
    (CAP_DUAL_SIDE, "양면"), (CAP_WHITE, "화이트"), (CAP_UV, "UV"), (CAP_MAGNETIC, "마그네틱"),
)
FAMILY_NAMES = {FAMILY_RTAI: "RTAI", FAMILY_SMART: "SMART"}
CONNECTION_SENTENCES = {
    "usb": "USB 선으로 연결됨", "tcp": "네트워크로 연결됨", "net": "네트워크로 연결됨",
    "fake": "연습용 가짜 프린터",
}

JOB_STATE_LABELS = {
    "completed": "완료", "partial": "일부 실패", "failed": "실패", "cancelled": "취소",
    "printing": "인쇄 중", "pending": "기다리는 중", "queued": "기다리는 중",
}
JOB_STATE_TONES = {
    "completed": "ok", "partial": "warn", "failed": "error", "cancelled": "idle",
    "printing": "busy", "pending": "idle", "queued": "idle",
}
CARD_STATE_LABELS = {"success": "끝남", "failed": "실패", "cancelled": "취소됨"}
CARD_STATE_TONES = {"success": "ok", "failed": "error", "cancelled": "idle"}


def plain(message: str | None) -> str:
    """서비스가 준 문구를 화면에 넣을 글자로. 낱말은 서비스가 이미 화면과 같은 말(사람, 항목, 장)로 쓴다."""
    return str(message or "").strip()


_PRINTER_NOTE = re.compile(r"\s*\[프린터 메시지:[^\]]*\]")


def brief(message: str | None) -> str:
    """실패 이유를 화면에 보일 만큼만: SDK가 붙인 영어 원문은 빼고 코드는 남긴다."""
    return _PRINTER_NOTE.sub("", plain(message)).strip()


# 단계 키(PrintProgress.stage_key) → 화면에 보일 말. 드라이버마다 다른 문구 대신 같은 말을 쓴다.
STAGE_SENTENCES = {
    "connect": "프린터에 연결하고 있습니다",
    "prepare": "카드 그림을 준비하고 있습니다",
    "feed": "카드를 프린터 안으로 넣고 있습니다",
    "magnetic": "마그네틱에 정보를 적고 있습니다",
    "setup": "인쇄를 준비하고 있습니다",
    "front_image": "앞면 그림을 프린터로 보내고 있습니다",
    "back_image": "뒷면 그림을 프린터로 보내고 있습니다",
    "print": "카드를 찍고 있습니다",
    "print_front": "앞면을 찍고 있습니다",
    "print_back": "뒷면을 찍고 있습니다",
    "wait": "다 찍히기를 기다리고 있습니다",
    "eject": "카드를 내보내고 있습니다",
    "done": "이 카드를 다 찍었습니다",
    "next_card": "다음 카드를 준비하고 있습니다",
}


def stage_sentence(stage_key: str | None, text: str | None = "", side: str | None = None) -> str:
    """지금 하는 일을 한 문장으로. 단계 키를 모르는 드라이버는 보내 준 문구를 그대로 쓴다."""
    text = plain(text).rstrip(". ").strip()
    if stage_key == "next_card" and "실패" in text:
        return "이 카드를 찍지 못했습니다"
    return STAGE_SENTENCES.get(stage_key or "") or text


def capability_words(capabilities: Iterable[str]) -> list[str]:
    caps = set(capabilities)
    return [word for key, word in CAPABILITY_WORDS if key in caps]


def printer_title(device: PrinterDevice | None) -> str:
    return device.name if device is not None else "프린터 없음"


def connection_sentence(device: PrinterDevice) -> str:
    return CONNECTION_SENTENCES.get(device.connection, device.connection or "연결 방식 모름")


def status_sentence(status: PrinterStatus | None) -> str:
    if status is None:
        return "상태를 확인하고 있습니다."
    if status.ready:
        return "바로 인쇄할 수 있습니다."
    if status.feeder_empty:
        return "카드가 없습니다. 카드 넣는 곳에 카드를 채워 주세요."
    if status.ribbon_remaining is not None and status.ribbon_remaining <= 0:
        return "리본을 다 썼습니다. 새 리본으로 바꿔 주세요."
    return plain(status.message) or "인쇄할 수 없는 상태입니다. 프린터 화면의 안내를 확인해 주세요."


def pill_state(device: PrinterDevice | None, status: PrinterStatus | None, printing: bool,
               searching: bool = False) -> tuple[str, str, str]:
    """위쪽 줄 프린터 알약에 넣을 (state, 글자, 자세한 설명)."""
    if device is None:
        if searching:
            return "offline", "프린터 찾는 중", "연결된 프린터를 찾고 있습니다."
        return ("offline", "프린터 연결 안 됨",
                "프린터 전원과 USB 선을 확인한 뒤 프린터 화면에서 다시 찾아 주세요.")
    if printing:
        return ("busy", f"{device.name} 인쇄 중",
                "지금 카드를 찍고 있습니다. 끝날 때까지 프린터 덮개를 열지 마세요.")
    if status is None:
        return "offline", f"{device.name} 확인 중", "프린터 상태를 확인하고 있습니다."
    if status.ready:
        return "ready", f"{device.name} 준비됨", "바로 인쇄할 수 있습니다."
    return "error", f"{device.name} 확인 필요", status_sentence(status)


def supply_text(value: int | None, unit: str = UNIT_UNKNOWN, spent: int = 0) -> str:
    """리본·필름 남은 양. 프린터가 장수로 알려 줄 때만 "약 N장"이라고 말한다."""
    if value is None:
        return ""
    if value <= 0:
        return "다 썼습니다"
    if unit == UNIT_CARDS:
        return f"약 {max(0, int(value) - spent):,}장 남음"
    if unit == UNIT_PERCENT:
        return f"{int(value)}% 남음"
    return f"남아 있음 (프린터 표시값 {int(value):,})"


def supplies_lines(status: PrinterStatus | None) -> list[str]:
    if status is None:
        return []
    lines = []
    ribbon = supply_text(status.ribbon_remaining, status.ribbon_unit)
    film = supply_text(status.film_remaining, status.film_unit)
    if ribbon:
        lines.append(f"리본 {ribbon}")
    if film:
        lines.append(f"필름 {film}")
    return lines


def duration_text(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 50:
        return "1분 안에 끝납니다"
    minutes = int(round(seconds / 60.0))
    if minutes < 60:
        return f"약 {minutes}분 남음"
    hours, rest = divmod(minutes, 60)
    return f"약 {hours}시간 {rest}분 남음" if rest else f"약 {hours}시간 남음"


def parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def date_text(value: Any, now: datetime | None = None) -> str:
    """'오늘 14:05', '어제 09:10', '11월 10일 14:05', '2025년 3월 2일'."""
    moment = parse_time(value)
    if moment is None:
        return ""
    now = now or datetime.now()
    clock = f"{moment.hour:02d}:{moment.minute:02d}"
    days = (now.date() - moment.date()).days
    if days == 0:
        return f"오늘 {clock}"
    if days == 1:
        return f"어제 {clock}"
    if moment.year == now.year:
        return f"{moment.month}월 {moment.day}일 {clock}"
    return f"{moment.year}년 {moment.month}월 {moment.day}일"


def clock_text(value: Any) -> str:
    moment = parse_time(value)
    return "" if moment is None else f"{moment.hour:02d}:{moment.minute:02d}:{moment.second:02d}"


def job_counts_short(total: int, success: int) -> str:
    return f"{total:,}장 모두 완료" if success >= total else f"{total:,}장 중 {success:,}장 완료"


def short_clock(value: Any) -> str:
    moment = parse_time(value)
    return "" if moment is None else f"{moment.hour:02d}:{moment.minute:02d}"


def job_counts_text(total: int, success: int, failed: int) -> str:
    if failed:
        return f"{total:,}장 중 {success:,}장 완료, {failed:,}장 실패"
    if success >= total:
        return f"{total:,}장 모두 완료"
    return f"{total:,}장 중 {success:,}장 완료"
