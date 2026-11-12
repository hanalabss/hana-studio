"""명단 화면 테스트 공용 도우미. 테스트 파일에서 `from people_helpers import *`로 가져다 쓴다."""

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

ROOT = str(Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

__all__ = ["people_env", "workspace", "make_store", "people_store", "page", "photo_folder", "pump", "PEOPLE",
           "write_png", "spy_rows"]

PEOPLE = [
    {"이름": "강민서", "부서": "영업1팀", "직급": "대리", "사번": "HN260415", "입사일": "2026-03-02", "사진": "HN260415"},
    {"이름": "고은채", "부서": "디자인팀", "직급": "사원", "사번": "HN260416", "입사일": "2026-03-02", "사진": "HN260416"},
    {"이름": "김하늘", "부서": "경영지원팀", "직급": "사원", "사번": "HN260417", "입사일": "2026-03-02", "사진": "HN260417"},
    {"이름": "남지호", "부서": "개발팀", "직급": "과장", "사번": "HN230102", "입사일": "2023-01-02", "사진": "HN230102"},
    {"이름": "문서윤", "부서": "개발팀", "직급": "사원", "사번": "HN260418", "입사일": "2026-03-09", "사진": "HN260418"},
    {"이름": "박도윤", "부서": "영업2팀", "직급": "차장", "사번": "HN190233", "입사일": "2019-07-01", "사진": "HN190233"},
    {"이름": "서지안", "부서": "경영지원팀", "직급": "사원", "사번": "HN260419", "입사일": "2026-03-09", "사진": ""},
    {"이름": "양수아", "부서": "디자인팀", "직급": "대리", "사번": "HN260420", "입사일": "2026-03-16", "사진": "HN260420"},
]
MISSING_PHOTO = ("문서윤", "서지안")       # 파일이 없는 사람, 값이 빈 사람
PRINTED = ("남지호", "박도윤")


def write_png(path, size=(60, 76), color=(200, 210, 230)):
    from PIL import Image

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return str(path)


def pump(qapp, rounds: int = 4) -> None:
    for _ in range(rounds):
        qapp.processEvents()


@pytest.fixture(autouse=True)
def people_env(qapp, tmp_path, monkeypatch):
    """테스트마다 설정 파일, 명단 폴더, 테마를 새로 쓴다."""
    from PySide6.QtCore import QCoreApplication, QEvent, QSettings

    from ui.theme import ThemeManager

    monkeypatch.setenv("HANA_STUDIO_SETTINGS_FILE", str(tmp_path / "settings.ini"))
    monkeypatch.setenv("HANA_STUDIO_DATA_DIR", str(tmp_path / "lists"))
    manager = ThemeManager(QSettings(str(tmp_path / "theme.ini"), QSettings.IniFormat))
    ThemeManager.set_instance(manager)
    manager.set_mode("light", persist=False)
    manager.apply(qapp)
    yield manager
    for widget in qapp.topLevelWidgets():
        widget.close()
        widget.deleteLater()
    qapp.processEvents()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    ThemeManager.set_instance(None)
    qapp.setStyleSheet("")


@pytest.fixture
def workspace(qapp):
    from design import presets
    from ui.pages.workspace import Workspace

    ws = Workspace()
    ws.editor.new_document(presets.CUT_54X86_BLEED, "employee", name="사원증")
    yield ws
    ws.close()


@pytest.fixture
def photo_folder(tmp_path):
    folder = tmp_path / "사진"
    for index, person in enumerate(PEOPLE):
        if person["이름"] in MISSING_PHOTO:
            continue
        write_png(folder / f"{person['사번']}.png", color=(120 + index * 10, 150, 200))
    return folder


@pytest.fixture
def make_store(tmp_path, photo_folder):
    """사원증 카드에 맞는 8명짜리 명단을 만든다. 두 명은 인쇄했고, 두 명은 사진이 없다."""
    from data.photos import set_photo_folder
    from data.store import DataStore

    counter = {"n": 0}

    def factory(people=None, with_photos: bool = True):
        counter["n"] += 1
        store = DataStore(tmp_path / f"people{counter['n']}.hsdata")
        store.add_column("이름")
        store.add_column("부서")
        store.add_column("직급")
        store.add_column("사번")
        store.add_column("입사일", "date")
        if with_photos:
            store.add_column("사진", "photo")
            set_photo_folder(store, "사진", photo_folder)
        rows = PEOPLE if people is None else people
        if not with_photos:
            rows = [{k: v for k, v in row.items() if k != "사진"} for row in rows]
        store.insert_many(rows)
        if people is None:
            printed = [r["_id"] for r in store.rows() if r["이름"] in PRINTED]
            store.mark_printed(printed)
        return store

    return factory


@pytest.fixture
def people_store(make_store):
    return make_store()


@pytest.fixture
def page(qapp, workspace, people_store):
    """명단이 들어 있는 명단 화면."""
    from ui.pages.people import PeoplePage

    widget = PeoplePage(workspace)
    widget.resize(1196, 620)
    widget.show()
    workspace.set_store(people_store)
    pump(qapp)
    yield widget
    widget.commit_pending_delete()
    widget.close()


def spy_rows(store):
    """store.rows 호출을 기록한다: [(offset, limit, 받은 줄 수), ...]."""
    calls = []
    original = store.rows

    def rows(filter=None, order=None, offset=0, limit=None):
        result = original(filter, order, offset, limit)
        calls.append((offset, limit, len(result)))
        return result

    store.rows = rows
    return calls
