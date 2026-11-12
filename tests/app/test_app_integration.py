"""메인 창이 모든 화면과 함께 뜨고, 화면 사이를 오갈 수 있는지 확인한다."""
import pytest

from printer import registry
from printer.history import JobHistory
from printer.print_queue import PrintQueueManager


@pytest.fixture
def shell(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setenv("APPDATA", str(tmp_path / "roaming"))
    monkeypatch.setenv("HANA_STUDIO_DATA_DIR", str(tmp_path / "data"))
    registry.use_fakes(True)
    from services.print_service import PrintService
    from services.printers import PrinterService
    from ui.app import build_main_window
    from ui.pages.services import AppServices

    history = JobHistory(tmp_path / "history.sqlite")
    queue = PrintQueueManager(history=history)
    services = AppServices(printers=PrinterService(queue=queue, settings_path=tmp_path / "printer.json"),
                           printing=PrintService(queue=queue, history=history),
                           history=history, queue=queue)
    window = build_main_window(services=services, start_services=False)
    yield window
    services.shutdown()
    window.workspace.close()
    window.deleteLater()
    registry.use_fakes(False)


def test_all_pages_registered_and_navigable(shell):
    from ui.pages.design import DesignPage
    from ui.pages.history import HistoryPage
    from ui.pages.people import PeoplePage
    from ui.pages.printer import PrinterPage
    from ui.pages.printing import PrintPage
    from ui.pages.settings import SettingsPage

    expected = {"design": DesignPage, "list": PeoplePage, "print": PrintPage,
                "history": HistoryPage, "printer": PrinterPage, "settings": SettingsPage}
    for page_id, cls in expected.items():
        assert isinstance(shell.page(page_id), cls)
        assert shell.navigate(page_id)
        assert shell.current_page_id() == page_id
    assert "quick" in shell.page_ids()


def test_pages_share_one_workspace(shell):
    workspace = shell.workspace
    assert shell.page("design").workspace is workspace or getattr(shell.page("design"), "_workspace", None) is workspace
    assert workspace.editor.document is not None


def test_window_renders(shell):
    shell.resize(1280, 800)
    for page_id in ("design", "list", "print", "history", "printer", "settings"):
        shell.navigate(page_id)
        image = shell.grab().toImage()
        assert not image.isNull() and image.width() >= 1000
