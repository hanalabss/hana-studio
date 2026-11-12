"""관리자 권한 없이 Program Files에 설치돼 설치 폴더에 쓸 수 없어도 프로그램이 뜨는지 확인한다."""
import json
import os
import sys
from pathlib import Path


def _deny_writes_under(monkeypatch, folder: Path):
    """folder 아래에는 폴더도 파일도 만들 수 없게 한다(권한 없음 흉내)."""
    real_mkdir = Path.mkdir
    real_write = Path.write_text
    real_open = open

    def inside(path) -> bool:
        try:
            return Path(os.path.abspath(path)).is_relative_to(folder)
        except Exception:
            return False

    def mkdir(self, *args, **kwargs):
        if inside(self):
            raise PermissionError(13, "Access is denied", str(self))
        return real_mkdir(self, *args, **kwargs)

    def write_text(self, *args, **kwargs):
        if inside(self):
            raise PermissionError(13, "Access is denied", str(self))
        return real_write(self, *args, **kwargs)

    def guarded_open(path, mode="r", *args, **kwargs):
        if any(flag in mode for flag in "wax+") and inside(path):
            raise PermissionError(13, "Access is denied", str(path))
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", mkdir)
    monkeypatch.setattr(Path, "write_text", write_text)
    monkeypatch.setattr("builtins.open", guarded_open)


def test_config_falls_back_to_user_folder_when_install_folder_is_read_only(tmp_path, monkeypatch):
    install = tmp_path / "Program Files" / "Hana Studio 2"
    install.mkdir(parents=True)
    local = tmp_path / "local"
    monkeypatch.setenv("LOCALAPPDATA", str(local))
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(install / "HanaStudio.exe"))
    monkeypatch.chdir(install)
    _deny_writes_under(monkeypatch, install)

    import config as config_module

    cfg = config_module.Config()          # 예전에는 여기서 PermissionError로 프로그램이 뜨지 못했다
    for key, path in cfg.settings["directories"].items():
        assert Path(path).is_absolute() and Path(path).is_relative_to(local), (key, path)
        assert Path(path).is_dir()


def test_license_key_is_saved_to_user_folder_when_install_folder_is_read_only(tmp_path, monkeypatch):
    install = tmp_path / "Program Files" / "Hana Studio 2"
    install.mkdir(parents=True)
    local = tmp_path / "local"
    from licensing import dialog

    monkeypatch.setattr(dialog, "LICENSE_FILE", str(install / "license.json"))
    monkeypatch.setattr(dialog, "LEGACY_LICENSE_FILE", str(install / "_internal" / "license.json"))
    monkeypatch.setattr(dialog, "USER_LICENSE_FILE", str(local / "HanaStudio" / "license.json"))
    _deny_writes_under(monkeypatch, install)

    assert dialog.save_license_key("HANA-TEST-TEST-TEST") is True
    assert json.loads((local / "HanaStudio" / "license.json").read_text(encoding="utf-8"))["license_key"] == "HANA-TEST-TEST-TEST"
    assert dialog.load_license_key() == "HANA-TEST-TEST-TEST"
