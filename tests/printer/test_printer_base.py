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
