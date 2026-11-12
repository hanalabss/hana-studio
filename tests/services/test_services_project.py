"""project: .hsproj 저장·열기·폴더 이동, 최근 목록, 자동 저장과 복구."""
import json
import os
import shutil
import time

import pytest

from design import presets, storage
from design.model import ImageItem, TextItem
from printer.base import PrinterDevice
from services import project as project_module
from services.project import Project, ProjectError, RecentProjects, unsaved_recoveries
from services.render_job import PrintOptions


@pytest.fixture
def workspace(tmp_path, store, document):
    """프로젝트 폴더: 디자인(.hsd)과 명단(.hsdata)이 같이 들어 있다."""
    folder = tmp_path / "사원증 작업"
    folder.mkdir()
    design_path = folder / "design.hsd"
    storage.save(document, design_path)
    data_path = folder / "data" / "people.hsdata"
    data_path.parent.mkdir()
    store.close()
    shutil.copy(store.path, data_path)
    return folder, design_path, data_path


def test_new_project_is_a_small_versioned_json_with_relative_paths(workspace):
    folder, design_path, data_path = workspace
    project = Project.new(folder / "사원증", design_path=design_path, data_path=data_path,
                          options=PrintOptions(copies=2, magnetic={"track2": "사번"}))
    assert project.path == str(folder / "사원증.hsproj")
    raw = json.loads((folder / "사원증.hsproj").read_text(encoding="utf-8"))
    assert raw["format"] == "hana-studio-project" and raw["version"] == 1
    assert raw["name"] == "사원증"
    assert raw["design"] == "design.hsd" and raw["data"] == "data/people.hsdata"
    assert raw["print_options"]["copies"] == 2 and raw["print_options"]["magnetic"] == {"track2": "사번"}
    assert os.path.getsize(project.path) < 4096


def test_open_restores_everything(workspace):
    folder, design_path, data_path = workspace
    created = Project.new(folder / "a.hsproj", design_path=design_path, data_path=data_path,
                          options=PrintOptions(stop_on_error=False, name_column="이름"))
    created.printer = PrinterDevice("smart", "SMART-51", "usb", "SMART-51 (USB001)", {"pid": 13584})
    created.save()

    opened = Project.open(created.path)
    assert opened.name == "a"
    assert opened.design_path == str(design_path) and opened.data_path == str(data_path)
    assert opened.options.stop_on_error is False and opened.options.name_column == "이름"
    assert opened.printer == created.printer and opened.printer.extra["pid"] == 13584
    assert opened.missing_files() == []

    document = opened.load_document()
    try:
        assert document.name == "사원증" and len(document.side("front").items) > 3
    finally:
        storage.cleanup_workdir(document)
    with opened.open_store() as reopened:
        assert reopened.count() == 5


def test_project_folder_can_be_moved(tmp_path, workspace):
    folder, design_path, data_path = workspace
    Project.new(folder / "a.hsproj", design_path=design_path, data_path=data_path)
    moved = tmp_path / "옮긴 곳" / "새 폴더"
    moved.parent.mkdir()
    shutil.move(str(folder), str(moved))

    opened = Project.open(moved / "a.hsproj")
    assert opened.design_path == str(moved / "design.hsd")
    assert opened.data_path == str(moved / "data" / "people.hsdata")
    assert opened.missing_files() == []
    document = opened.load_document()
    storage.cleanup_workdir(document)
    with opened.open_store() as reopened:
        assert reopened.count() == 5


def test_absolute_path_that_moved_with_the_project_is_found_beside_it(tmp_path, workspace):
    folder, design_path, _ = workspace
    project_file = folder / "a.hsproj"
    project_file.write_text(json.dumps({
        "format": "hana-studio-project", "version": 1,
        "design": str(tmp_path / "예전 위치" / "design.hsd"), "data": None,
    }), encoding="utf-8")
    opened = Project.open(project_file)
    assert opened.design_path == str(design_path)
    assert opened.data_path is None
    with pytest.raises(ProjectError, match="명단 파일"):
        opened.open_store()


def test_missing_files_are_listed(workspace):
    folder, design_path, data_path = workspace
    project = Project.new(folder / "a.hsproj", design_path=design_path, data_path=data_path)
    os.remove(design_path)
    opened = Project.open(project.path)
    assert opened.missing_files() == [str(design_path)]
    with pytest.raises(ProjectError, match="찾을 수 없습니다"):
        opened.load_document()


def test_unknown_fields_and_newer_versions_survive_a_save(workspace):
    folder, design_path, _ = workspace
    project_file = folder / "future.hsproj"
    project_file.write_text(json.dumps({
        "format": "hana-studio-project", "version": 7, "name": "미래", "design": "design.hsd",
        "cloud_sync": {"enabled": True},
    }), encoding="utf-8")
    opened = Project.open(project_file)
    assert opened.is_newer_format and opened.extra == {"cloud_sync": {"enabled": True}}
    opened.save()
    raw = json.loads(project_file.read_text(encoding="utf-8"))
    assert raw["version"] == 7 and raw["cloud_sync"] == {"enabled": True}


@pytest.mark.parametrize("content", ["not json", "[]", '{"format": "other"}',
                                     '{"format": "hana-studio-project", "version": "x"}'])
def test_broken_project_files_give_korean_errors(tmp_path, content):
    path = tmp_path / "bad.hsproj"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ProjectError):
        Project.open(path)
    with pytest.raises(ProjectError, match="찾을 수 없습니다"):
        Project.open(tmp_path / "none.hsproj")


def test_save_without_a_location_is_refused():
    with pytest.raises(ProjectError, match="위치"):
        Project.new().save()


# ── 최근 프로젝트 ───────────────────────────────────────────────────────────
def test_recent_projects_live_under_the_user_profile(tmp_path, workspace):
    folder, design_path, _ = workspace
    first = Project.new(folder / "first.hsproj", design_path=design_path)
    second = Project.new(folder / "second.hsproj", design_path=design_path)
    Project.open(first.path)

    recent = RecentProjects()
    assert recent.path == str(tmp_path / "appdata" / "HanaStudio" / "recent_projects.json")
    assert os.path.isfile(recent.path)
    assert [(e.name, e.exists) for e in recent.entries()] == [("first", True), ("second", True)]

    os.remove(second.path)
    assert [e.name for e in recent.entries(existing_only=True)] == ["first"]
    recent.remove(second.path)
    assert [e.name for e in recent.entries()] == ["first"]
    recent.clear()
    assert recent.entries() == []


def test_recent_list_is_bounded_and_deduplicated(tmp_path):
    recent = RecentProjects(str(tmp_path / "recent.json"), limit=3)
    for name in ("a", "b", "c", "d", "b"):
        recent.add(str(tmp_path / f"{name}.hsproj"))
    assert [e.name for e in recent.entries()] == ["b", "d", "c"]
    (tmp_path / "recent.json").write_text("garbage", encoding="utf-8")
    assert recent.entries() == []


def test_remember_false_leaves_the_recent_list_alone(workspace):
    folder, design_path, _ = workspace
    project = Project.new(folder / "quiet.hsproj", design_path=design_path, remember=False)
    Project.open(project.path, remember=False)
    assert RecentProjects().entries() == []


# ── 자동 저장 · 복구 ─────────────────────────────────────────────────────────
def test_autosave_sidecar_recovers_unsaved_design_work(workspace):
    folder, design_path, data_path = workspace
    project = Project.new(folder / "a.hsproj", design_path=design_path, data_path=data_path)
    assert not project.has_recovery()

    document = project.load_document()
    try:
        document.side("front").add_item(TextItem(name="새 항목", x=1, y=1, w=10, h=5, text="저장 전"))
        sidecar = project.autosave(document)
    finally:
        storage.cleanup_workdir(document)
    assert sidecar == project.path + ".autosave" and os.path.isfile(sidecar)

    # 프로그램이 죽었다가 다시 열렸다.
    reopened = Project.open(project.path)
    assert reopened.has_recovery()
    original = reopened.load_document()
    recovered = reopened.recover()
    try:
        assert "새 항목" in [item.name for item in recovered.side("front").items]
        assert "새 항목" not in [item.name for item in original.side("front").items]
        reopened.save_document(recovered)
    finally:
        storage.cleanup_workdir(original)
        storage.cleanup_workdir(recovered)
    assert not reopened.has_recovery() and not os.path.exists(sidecar)
    saved = reopened.load_document()
    try:
        assert "새 항목" in [item.name for item in saved.side("front").items]
    finally:
        storage.cleanup_workdir(saved)


def test_stale_autosave_older_than_the_saved_design_is_ignored(workspace):
    folder, design_path, _ = workspace
    project = Project.new(folder / "a.hsproj", design_path=design_path)
    document = project.load_document()
    try:
        project.autosave(document)
        old = time.time() - 3600
        os.utime(project.autosave_path, (old, old))
        assert not project.has_recovery()
        project.discard_recovery()
        assert not os.path.exists(project.autosave_path)
        project.discard_recovery()                       # 없어도 조용히 지나간다
        with pytest.raises(ProjectError, match="자동 저장본"):
            project.recover()
    finally:
        storage.cleanup_workdir(document)


def test_unsaved_project_autosaves_under_the_user_profile_and_follows_the_first_save(tmp_path):
    document = presets.build_template("visitor")
    draft = Project.new()
    path = draft.autosave(document)
    assert path.startswith(str(tmp_path / "appdata" / "HanaStudio" / "autosave"))
    assert unsaved_recoveries() == [path] and draft.has_recovery()

    target = draft.save(tmp_path / "방문증")
    assert target.endswith("방문증.hsproj")
    assert unsaved_recoveries() == []
    assert os.path.isfile(target + ".autosave") and draft.has_recovery()

    design = draft.save_document(document, tmp_path / "방문증 디자인")
    assert design.endswith(".hsd") and draft.design_path == design
    assert not draft.has_recovery()
    draft.save()
    assert Project.open(target).design_path == design


def test_save_document_reports_storage_errors(tmp_path):
    document = presets.new_document(presets.CR80)
    document.side("front").add_item(ImageItem(asset="assets/missing.png"))
    project = Project.new(tmp_path / "a.hsproj")
    with pytest.raises(ProjectError, match="저장할 수 없습니다"):
        project.save_document(document, tmp_path / "d.hsd")
    with pytest.raises(ProjectError, match="위치"):
        Project.new().save_document(presets.new_document(presets.CR80))
    assert project_module.EXTENSION == ".hsproj"
