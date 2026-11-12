import importlib.util
from pathlib import Path

import pytest
from PySide6.QtGui import QImage

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def preview():
    spec = importlib.util.spec_from_file_location("ui_preview", ROOT / "scripts" / "ui_preview.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def distinct_colours(image: QImage, step: int = 7) -> int:
    seen = set()
    for y in range(0, image.height(), step):
        for x in range(0, image.width(), step):
            seen.add(image.pixel(x, y))
    return len(seen)


def test_screenshot_export_writes_real_images_for_both_themes(preview, qapp, tmp_path):
    from ui.theme import DARK, LIGHT

    files = preview.export_screenshots(qapp, tmp_path / "shots")
    names = {path.name for path in files}
    assert len(names) == len(files) >= 40

    for theme in ("light", "dark"):
        for page in ("1_design", "2_list", "3_print", "4_history", "5_quick", "6_printer", "7_settings", "8_toast"):
            assert f"shell_{page}_{theme}.png" in names
        for section, _factory in preview.GALLERY_SECTIONS:
            assert f"gallery_{section}_{theme}.png" in names
        for extra in ("gallery_focus", "dialog", "menu", "popover", "calendar"):
            assert f"{extra}_{theme}.png" in names

    for path in files:
        assert path.is_file() and path.stat().st_size > 300, path.name
        image = QImage(str(path))
        assert not image.isNull(), path.name
        assert image.width() >= 100 and image.height() >= 20, path.name
        assert distinct_colours(image) >= 3, f"{path.name}이 한 가지 색뿐입니다"

    shots = tmp_path / "shots"
    light = QImage(str(shots / "shell_1_design_light.png"))
    dark = QImage(str(shots / "shell_1_design_dark.png"))
    assert (light.width(), light.height()) == (1280, 800) == (dark.width(), dark.height())
    assert light.pixelColor(700, 600).name().upper() == LIGHT.hex("ground")
    assert dark.pixelColor(700, 600).name().upper() == DARK.hex("ground")
    assert light.pixelColor(40, 500).name().upper() == LIGHT.hex("surface")
    assert dark.pixelColor(40, 500).name().upper() == DARK.hex("surface")


def test_gallery_shows_every_widget_family(preview, qapp, show):
    from ui import widgets

    window, sections = preview.build_gallery()
    show(window)
    assert list(sections) == [name for name, _ in preview.GALLERY_SECTIONS]
    for cls in (widgets.PrimaryButton, widgets.SecondaryButton, widgets.QuietButton, widgets.DangerButton,
                widgets.IconButton, widgets.SegmentedControl, widgets.FilterChip, widgets.FieldChip,
                widgets.StatusPill, widgets.Panel, widgets.FormRow, widgets.MmField, widgets.StepperField,
                widgets.SwatchPicker, widgets.Banner, widgets.Toast, widgets.EmptyState,
                widgets.ProgressBar, widgets.RibbonProgress, widgets.ComboBox, widgets.SearchField):
        assert window.findChildren(cls), f"갤러리에 {cls.__name__}이 없습니다"
    disabled = [b for b in window.findChildren(widgets.Button) if not b.isEnabled()]
    assert len(disabled) >= 5, "쓸 수 없는 상태도 보여야 한다"


def test_preview_arguments(preview):
    args = preview._parse(["--theme", "dark", "--gallery"])
    assert args.theme == "dark" and args.gallery and args.shot is None
    args = preview._parse(["--shot", "out"])
    assert args.shot == "out"
    with pytest.raises(SystemExit):
        preview._parse(["--theme", "sepia"])


def test_preview_shell_runs_without_licence_or_printer(preview, qapp, show):
    shell = show(preview.build_shell())
    assert shell.current_page_id() == "design"
    shell.navigate("quick")
    assert shell.page("quick").load_now() is True
    assert not shell.page("quick").legacy_window().isWindow()
