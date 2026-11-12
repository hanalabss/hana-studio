"""온라인 업데이트 감싸개: 설치본이 아닐 때는 조용히 아무 일도 하지 않는다."""
import pytest

from services.updates import AvailableUpdate, UpdateError, Updater


class Asset:
    def __init__(self, version, size, notes=""):
        self.Version, self.Size, self.NotesMarkdown = version, size, notes


class Info:
    def __init__(self, full, deltas=(), base=None):
        self.TargetFullRelease, self.DeltasToTarget, self.BaseRelease = full, list(deltas), base


class FakeManager:
    def __init__(self, info=None, fail=None):
        self.info, self.fail, self.calls = info, fail or set(), []

    def get_current_version(self):
        return "2.0.0"

    def check_for_updates(self):
        if "check" in self.fail:
            raise RuntimeError("network")
        return self.info

    def download_updates(self, info, progress=None):
        if "download" in self.fail:
            raise RuntimeError("network")
        for value in (10, 100):
            progress(value)
        self.calls.append(("download", info))

    def wait_exit_then_apply_updates(self, info, silent, restart, args):
        self.calls.append(("apply", info, silent, restart))


def make(manager):
    return Updater("https://example.invalid/feed", manager_factory=lambda source: manager)


def test_not_an_installed_copy_means_unsupported_and_silent():
    def factory(source):
        raise RuntimeError("not installed")
    updater = Updater("x", manager_factory=factory)
    assert not updater.supported and updater.check() is None and updater.current_version() == ""
    with pytest.raises(UpdateError):
        updater.download(AvailableUpdate("2.0.1"))


def test_no_update_and_update_found_with_delta_size():
    assert make(FakeManager(None)).check() is None
    info = Info(Asset("2.0.1", 250_000_000, "notes"), deltas=[Asset("2.0.1", 6_000_000)], base=Asset("2.0.0", 1))
    found = make(FakeManager(info)).check()
    assert (found.version, found.size_bytes, found.notes, found.size_text) == ("2.0.1", 6_000_000, "notes", "5.7 MB")
    full_only = make(FakeManager(Info(Asset("2.0.1", 250_000_000)))).check()
    assert full_only.size_text == "238 MB"


def test_download_reports_progress_then_apply_restarts():
    info = Info(Asset("2.0.1", 1))
    manager = FakeManager(info)
    updater = make(manager)
    found = updater.check()
    seen = []
    updater.download(found, seen.append)
    updater.apply_and_restart(found)
    assert seen == [10, 100]
    assert manager.calls == [("download", info), ("apply", info, False, True)]


def test_network_failures_become_plain_messages():
    with pytest.raises(UpdateError, match="인터넷 연결"):
        make(FakeManager(fail={"check"})).check()
    updater = make(FakeManager(Info(Asset("2.0.1", 1)), fail={"download"}))
    with pytest.raises(UpdateError, match="받지 못했습니다"):
        updater.download(updater.check())


def test_default_source_is_the_public_repository(monkeypatch):
    seen = []
    monkeypatch.delenv("HANA_STUDIO_UPDATE_URL", raising=False)
    Updater(manager_factory=lambda source: seen.append(source) or FakeManager())
    monkeypatch.setenv("HANA_STUDIO_UPDATE_URL", r"C:\feed")
    Updater(manager_factory=lambda source: seen.append(source) or FakeManager())
    assert seen == ["https://github.com/hanalabss/hana-studio", r"C:\feed"]
