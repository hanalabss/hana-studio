"""화면 언어(다눌의 Language 메뉴). 한국어면 아무것도 바꾸지 않고, 다른 언어면 번역표에 있는 글자만 바꾼다."""

import json

import pytest
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QLabel, QMenu, QPushButton

from ui import i18n
from ui.widgets import FieldChip


@pytest.fixture
def english(qapp, tmp_path, monkeypatch):
    table = {"저장": "Save", "복사": "Copy", "이름": None, "인쇄 설정": "Print settings"}
    (tmp_path / "en.json").write_text(json.dumps(table, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(i18n, "_DIR", tmp_path)
    assert i18n.install(qapp, "en") == "en"
    yield
    i18n.uninstall(qapp)


def test_korean_installs_nothing(qapp):
    assert i18n.install(qapp, "ko") == "ko"
    assert QLabel.setText is i18n.QLabel.setText and not i18n._ORIGINAL
    label = QLabel("저장")
    assert label.text() == "저장"


def test_setters_and_constructors_are_translated(english, qapp):
    button = QPushButton("저장")
    button.show()
    qapp.processEvents()
    assert button.text() == "Save"
    label = QLabel()
    label.setText("인쇄 설정")
    assert label.text() == "Print settings"
    label.setText("번역표에 없는 안내 문장")
    assert label.text() == "번역표에 없는 안내 문장"
    button.close()


def test_menu_shortcut_after_tab_is_kept(english, qapp):
    menu = QMenu()
    action = QAction("복사\tCtrl+C", menu)
    menu.addAction(action)
    i18n._translate_menu(menu)
    assert action.text() == "Copy\tCtrl+C"


def test_data_values_are_left_alone(english, qapp):
    chip = FieldChip("저장")              # 명단 항목 이름이 우연히 번역표에 있어도 그대로
    chip.show()
    qapp.processEvents()
    assert chip.text() == "저장"
    assert i18n.translate("이름") == "이름"   # null로 둔 항목
    chip.close()


def test_language_choice_is_stored(tmp_path, qapp):
    from PySide6.QtCore import QSettings

    settings = QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat)
    assert i18n.stored_language(settings) == "ko"
    i18n.save_language("ja", settings)
    assert i18n.stored_language(settings) == "ja"
