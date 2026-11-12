"""시작 화면: 최근에 작업한 디자인."""
from services.project import RecentEntry
from ui.pages.design.start_panel import StartPanel


def test_recent_row_is_hidden_until_there_is_something(qapp):
    panel = StartPanel()
    assert panel.recent_box.isHidden() and panel.templates_title.text() == "시작할 모양"


def test_recent_designs_are_listed_and_open_on_click(qapp, tmp_path):
    panel = StartPanel()
    entries = [RecentEntry(str(tmp_path / f"{n}.hsd"), n, "2026-11-12T10:00:00") for n in ("사원증", "방문증")]
    opened = []
    panel.openPathRequested.connect(opened.append)
    panel.set_recents(entries)
    assert not panel.recent_box.isHidden()
    assert panel.templates_title.text() == "새로 만들기"
    assert [card.accessibleName() for card in panel.recent_cards] == ["사원증 다시 열기", "방문증 다시 열기"]
    panel.recent_cards[1].clicked.emit()
    assert opened == [entries[1].path]
    panel.set_recents([])
    assert panel.recent_box.isHidden() and panel.recent_cards == []


def test_only_the_newest_five_are_shown(qapp, tmp_path):
    panel = StartPanel()
    panel.set_recents([RecentEntry(str(tmp_path / f"{i}.hsd"), str(i)) for i in range(9)])
    assert len(panel.recent_cards) == 5
