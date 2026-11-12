import pytest

from printer.base import (
    CardJob, CardPrinter, MagneticTracks, NotSupported, PrinterDevice, PrinterError,
    PrinterStatus, SideImages,
)


def test_printer_error_carries_korean_message_code_and_recoverable_flag():
    error = PrinterError("카드 공급함이 비어 있습니다.", code=0x8000000B, recoverable=True)
    assert error.message == "카드 공급함이 비어 있습니다."
    assert error.code == 0x8000000B
    assert error.recoverable is True
    assert str(error) == "카드 공급함이 비어 있습니다. (코드 0x8000000B)"


def test_printer_error_without_code_shows_only_message():
    assert str(PrinterError("인쇄할 이미지가 없습니다.")) == "인쇄할 이미지가 없습니다."


def test_not_supported_is_a_printer_error_naming_the_feature():
    error = NotSupported("uv")
    assert isinstance(error, PrinterError)
    assert error.feature == "uv"
    assert "uv" in error.message


def test_printer_device_is_hashable_and_ignores_extra_in_equality():
    first = PrinterDevice("smart", "SMART-51", "usb", "desc", {"pid": 0x3510})
    second = PrinterDevice("smart", "SMART-51", "usb", "desc", {"pid": 1})
    assert first == second
    assert len({first, second}) == 1
    assert first.display_name == "[SMART] SMART-51 (USB)"


@pytest.mark.parametrize("value, expected", [
    (None, (None, None, None)),
    ({"track1": "A", "track2": "12"}, ("A", "12", None)),
    ({"t3": "9"}, (None, None, "9")),
    (("A", "", "C"), ("A", None, "C")),
    (["A"], ("A", None, None)),
    (MagneticTracks("X", None, None), ("X", None, None)),
])
def test_magnetic_tracks_coerce(value, expected):
    assert MagneticTracks.coerce(value).as_tuple() == expected


def test_magnetic_tracks_is_empty():
    assert MagneticTracks().is_empty()
    assert not MagneticTracks(track2="1").is_empty()


def test_card_job_validation_rejects_empty_job_bad_copies_and_bad_orientation():
    with pytest.raises(PrinterError, match="인쇄할 이미지가 없습니다"):
        CardJob().validate()
    with pytest.raises(PrinterError, match="1장 이상"):
        CardJob(front=SideImages("a.png"), copies=0).validate()
    with pytest.raises(PrinterError, match="카드 방향"):
        CardJob(front=SideImages("a.png", orientation="diagonal")).validate()
    CardJob(front=SideImages("a.png", orientation="landscape")).validate()


def test_card_job_is_dual_side_only_with_both_sides():
    assert CardJob(front=SideImages("a"), back=SideImages("b")).is_dual_side
    assert not CardJob(front=SideImages("a")).is_dual_side
    assert not CardJob(back=SideImages("b")).is_dual_side


class _Minimal(CardPrinter):
    family = "test"

    def capabilities(self):
        return {"status"}

    def connect(self, device):
        self.device = device

    def close(self):
        self.device = None

    @property
    def is_connected(self):
        return self.device is not None

    def status(self):
        return PrinterStatus(True, "ok")

    def print_card(self, job, progress=None):
        self._report(150, "끝", progress)


def test_optional_features_raise_not_supported_by_default():
    printer = _Minimal()
    for call in (lambda: printer.write_magnetic({"track1": "A"}), printer.read_magnetic,
                 printer.inject_card, printer.eject_card):
        with pytest.raises(NotSupported):
            call()
    assert printer.supports("status") and not printer.supports("uv")


def test_progress_goes_to_the_callback_and_is_clamped():
    printer = _Minimal()
    seen = []
    printer.set_progress_callback(lambda percent, message: seen.append((percent, message)))
    printer.print_card(CardJob(front=SideImages("a")))
    override = []
    printer.print_card(CardJob(front=SideImages("a")), progress=lambda p, m: override.append(p))
    assert seen == [(100, "끝")]
    assert override == [100]


def test_require_connected_raises_recoverable_error():
    with pytest.raises(PrinterError) as info:
        _Minimal()._require_connected()
    assert info.value.recoverable


def test_context_manager_closes_the_printer():
    printer = _Minimal()
    with printer:
        printer.connect(PrinterDevice("test", "x"))
        assert printer.is_connected
    assert not printer.is_connected


# ── 구조화된 진행 · 잔량 단위 · 카드 공급자 ────────────────────────────────────
def test_structured_progress_receives_events_and_plain_callbacks_keep_working():
    from printer.base import STAGE_FEED, ProgressEvent, StructuredProgress, emit_progress

    class Staged(_Minimal):
        def print_card(self, job, progress=None):
            self._report(15, "카드 삽입 중...", progress, stage=STAGE_FEED)
            self._report(250, "끝", progress)

    events, plain = [], []
    Staged().print_card(CardJob(front=SideImages("a")), progress=StructuredProgress(events.append))
    Staged().print_card(CardJob(front=SideImages("a")), progress=lambda p, m: plain.append((p, m)))
    assert events == [ProgressEvent(15, "카드 삽입 중...", STAGE_FEED), ProgressEvent(100, "끝")]
    assert (events[0].panel, events[0].fraction, events[0].side) == (None, None, None)
    assert plain == [(15, "카드 삽입 중..."), (100, "끝")]

    # 예전 방식으로 불러도 이벤트가 된다(단계는 모름)
    collected = []
    callback = StructuredProgress(collected.append)
    callback(140, "예전 방식")
    emit_progress(callback, ProgressEvent(5, "새 방식", "prepare", panel="K", fraction=0.5))
    emit_progress(None, ProgressEvent(5, "버려짐"))
    assert collected == [ProgressEvent(100, "예전 방식"),
                         ProgressEvent(5, "새 방식", "prepare", panel="K", fraction=0.5)]


def test_status_remaining_is_only_cards_when_the_unit_says_so():
    from printer.base import UNIT_CARDS, UNIT_PERCENT, UNIT_UNKNOWN, remaining_text

    unknown = PrinterStatus(True, "ok", ribbon_remaining=310, film_remaining=40)
    assert (unknown.ribbon_unit, unknown.film_unit) == (UNIT_UNKNOWN, UNIT_UNKNOWN)
    assert unknown.ribbon_cards_left is None and unknown.film_cards_left is None

    cards = PrinterStatus(True, "ok", ribbon_remaining=310, ribbon_unit=UNIT_CARDS,
                          film_remaining=-3, film_unit=UNIT_CARDS)
    assert cards.ribbon_cards_left == 310 and cards.film_cards_left == 0
    assert PrinterStatus(True, "ok", ribbon_remaining=62, ribbon_unit=UNIT_PERCENT).ribbon_cards_left is None
    assert PrinterStatus(True, "ok", ribbon_unit=UNIT_CARDS).ribbon_cards_left is None

    assert remaining_text(310, UNIT_CARDS) == "약 310장"
    assert remaining_text(62, UNIT_PERCENT) == "62 %"
    assert remaining_text(310) == "310 (단위 확인 안 됨)" and "장" not in remaining_text(310)
    assert remaining_text(None, UNIT_CARDS) == "확인 불가"


def test_card_provider_helpers():
    from printer.base import (
        CardProvider, PrintCancelled, as_card_provider, provider_is_dual_side, provider_record_id,
    )

    jobs = (CardJob(front=SideImages("a"), record_id=7),
            CardJob(front=SideImages("a"), back=SideImages("b"), record_id=8))
    as_list = as_card_provider(jobs)
    assert as_list == list(jobs) and isinstance(as_list, list) and isinstance(as_list, CardProvider)
    assert provider_is_dual_side(as_list) and provider_record_id(as_list, 0) == 7
    assert as_card_provider(job for job in jobs) == list(jobs)

    class Provider:
        touched = []

        def __len__(self):
            return 2

        def __getitem__(self, index):
            self.touched.append(index)
            return jobs[index]

    provider = Provider()
    assert as_card_provider(provider) is provider
    assert provider_is_dual_side(provider) is False and provider_record_id(provider, 1) is None
    provider.is_dual_side = True
    provider.record_id = lambda index: 1 // 0
    assert provider_is_dual_side(provider) is True and provider_record_id(provider, 1) is None
    assert provider.touched == [], "길이·양면 여부·레코드 조회로는 카드를 만들지 않는다"

    cancelled = PrintCancelled()
    assert isinstance(cancelled, PrinterError) and cancelled.recoverable and "취소" in str(cancelled)
