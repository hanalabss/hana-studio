import sys
import time
from pathlib import Path

import pytest
from PIL import Image

ROOT = str(Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.importers import import_into, preview_csv  # noqa: E402
from data.store import DataStore  # noqa: E402
from design import presets  # noqa: E402
from design.binding import ColumnSource  # noqa: E402
from design.model import ImageItem, TextItem  # noqa: E402
from printer import registry  # noqa: E402
from printer.history import JobHistory  # noqa: E402
from printer.print_queue import PrintQueueManager  # noqa: E402
from services_helpers import STUB_DEVICE, Recorder, StubPrinter  # noqa: E402


PEOPLE_CSV = (
    "이름,사번,부서,직급,사진\n"
    "김철수,A0001,영업부,과장,kim\n"
    "이영희,A0002,개발부,대리,lee\n"
    "박민수,A0003,개발부,사원,park\n"
    "최지우,A0004,인사부,부장,choi\n"
    "정하나,A0005,영업부,사원,jung\n"
)


@pytest.fixture(autouse=True)
def _environment(qapp, tmp_path, monkeypatch):
    """가짜 프린터, 사용자 폴더 격리, 안정화 대기 제거."""
    from design.renderer import ensure_gui

    ensure_gui()
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "appdata"))
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    registry.use_fakes(True)
    yield
    registry.use_fakes(False)
    registry._use_fakes = None


@pytest.fixture
def app(qapp):
    return qapp


@pytest.fixture
def history(tmp_path):
    return JobHistory(str(tmp_path / "history.sqlite"))


@pytest.fixture
def queue(qapp, history):
    manager = PrintQueueManager(history=history)
    yield manager
    manager.cancel_all()
    manager.wait_idle()


@pytest.fixture
def photo_dir(tmp_path):
    folder = tmp_path / "photos"
    folder.mkdir()
    for index, name in enumerate(("kim", "lee", "park", "choi", "jung")):
        Image.new("RGB", (120, 160), (40 * index + 30, 80, 200 - 30 * index)).save(folder / f"{name}.png")
    return folder


@pytest.fixture
def store(tmp_path, photo_dir):
    """CSV를 가져온 명단 5명. 사진 열은 photo_dir를 가리킨다."""
    csv_path = tmp_path / "people.csv"
    csv_path.write_text(PEOPLE_CSV, encoding="utf-8-sig")
    with DataStore(tmp_path / "people.hsdata") as opened:
        result = import_into(opened, preview_csv(csv_path))
        assert result.inserted == 5
        opened.set_column_type("사진", "photo", {"folder": str(photo_dir)})
        yield opened


@pytest.fixture
def document():
    """사원증 템플릿(CR80, 가로): 이름·부서 글자, 사진, 사번 바코드."""
    return presets.build_template("employee")


@pytest.fixture
def simple_document():
    """이름 글자와 사진만 있는 작은 문서."""
    doc = presets.new_document(presets.CR80)
    side = doc.side("front")
    side.add_item(TextItem(name="이름", x=5, y=5, w=40, h=10, source=ColumnSource(column="이름")))
    side.add_item(ImageItem(name="사진", x=50, y=5, w=20, h=20, source=ColumnSource(column="사진")))
    return doc


@pytest.fixture
def rtai_device():
    return registry.discover(["rtai"])[0]


@pytest.fixture
def smart_device():
    return registry.discover(["smart"])[0]


@pytest.fixture
def rtai_lib():
    registry.discover(["rtai"])
    return registry._fake_library("rtai")


@pytest.fixture
def smart_lib():
    registry.discover(["smart"])
    return registry._fake_library("smart")


@pytest.fixture
def stub_device():
    return STUB_DEVICE


@pytest.fixture
def make_stub(queue):
    """StubPrinter를 큐에 물린다."""

    def factory(**kwargs):
        stub = StubPrinter(**kwargs)
        queue.set_printer_factory(lambda device: stub)
        return stub

    return factory


@pytest.fixture
def service(queue):
    from services.print_service import PrintService

    return PrintService(queue)


@pytest.fixture
def recorder(service):
    return Recorder(service)
