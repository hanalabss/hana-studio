"""배경 지우기를 진짜 모델로 돌려 보고 화면을 저장한다. 라이선스 확인도 프린터도 없이 돈다.

    python scripts/cutout_demo.py --shot <폴더>                    두 테마 모두, 지우기 전·후 화면과 카드 그림
    python scripts/cutout_demo.py --shot <폴더> --image 사진.png    다른 그림으로
    python scripts/cutout_demo.py --shot <폴더> --theme dark --white   흰색을 찍는 프린터가 연결된 것처럼

실제 Windows 화면에 창을 띄워 그 창만 찍는다(QWidget.grab). 모델은 이미 내려받아 둔 것을 쓴다.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_IMAGE = os.path.join(os.path.expanduser("~"), "Desktop", "테스트이미지.png")
CARD_COLOR = "#F08C1E"


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="하나스튜디오 배경 지우기 확인")
    parser.add_argument("--shot", metavar="DIR", required=True, help="PNG를 저장할 폴더")
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="배경을 지울 그림 (기본: 바탕 화면의 테스트이미지.png)")
    parser.add_argument("--theme", choices=("light", "dark"), default=None)
    parser.add_argument("--white", action="store_true", help="흰색을 찍는 프린터가 연결된 것으로 본다")
    parser.add_argument("--size", default="1280x800")
    return parser.parse_args(argv)


def make_stamp(path: str) -> str:
    """흰 바탕의 도장 그림. AI가 주인공을 못 찾기 쉬운 종류다."""
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (960, 960), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    red = (196, 30, 38)
    draw.ellipse((60, 60, 900, 900), outline=red, width=44)
    draw.ellipse((150, 150, 810, 810), outline=red, width=14)
    draw.rectangle((250, 400, 710, 560), fill=red)
    for index in range(5):
        x = 300 + index * 90
        draw.polygon([(x, 300), (x + 30, 360), (x - 30, 360)], fill=red)
        draw.rectangle((x - 22, 620, x + 22, 700), fill=red)
    image = image.resize((480, 480), Image.Resampling.LANCZOS)
    image.save(path)
    return path


def build(capabilities: set[str]):
    from PySide6.QtWidgets import QWidget

    from ui.pages.design import DesignPage
    from ui.pages.workspace import Workspace
    from ui.shell import PAGE_DESIGN, AppShell

    shell = AppShell(legacy_factory=lambda: QWidget())
    shell.set_printer_status("ready", "프린터 준비됨")
    workspace = Workspace(shell)
    page = DesignPage(workspace, capabilities=lambda: set(capabilities))     # 진짜 엔진(rembg)을 쓴다
    shell.register_page(PAGE_DESIGN, page, "디자인", "design", step_label="카드 꾸미기")
    shell.workspace, shell.design_page = workspace, page
    return shell


def _settle(app, rounds: int = 8) -> None:
    for _ in range(rounds):
        app.processEvents()


def _wait(app, section, seconds: float = 120.0) -> None:
    deadline = time.time() + seconds
    while section.is_busy() and time.time() < deadline:
        section.wait_for_removal(200)
        _settle(app)
    if section.is_busy():
        raise RuntimeError("배경 지우기가 제때 끝나지 않았습니다.")
    _settle(app)


def _save_card(controller, path: Path, crop: tuple[float, float, float, float] | None = None) -> None:
    """카드 한 면을 인쇄와 같은 방법으로 그려 저장한다. crop: 확대해 볼 부분(0..1 비율)."""
    from PIL import Image

    from design import renderer

    card = renderer.render_side(controller.document, "front", None, dpi=300).color
    if crop is not None:
        left, top, right, bottom = crop
        box = (round(card.width * left), round(card.height * top), round(card.width * right), round(card.height * bottom))
        card = card.crop(box)
        card = card.resize((card.width * 3, card.height * 3), Image.Resampling.NEAREST)
    card.save(str(path))


def export(app, args: argparse.Namespace) -> list[Path]:
    from ui.theme import ThemeManager

    if not os.path.isfile(args.image):
        raise SystemExit(f"그림을 찾을 수 없습니다: {args.image}")
    out = Path(args.shot)
    out.mkdir(parents=True, exist_ok=True)
    width, height = (int(v) for v in args.size.lower().split("x"))
    manager = ThemeManager.instance()
    manager.apply(app)
    stamp = make_stamp(os.path.join(tempfile.mkdtemp(prefix="hscut_"), "stamp.png"))
    capabilities = {"dual_side", "white"} if args.white else {"dual_side"}
    written: list[Path] = []

    for theme in ((args.theme,) if args.theme else ("light", "dark")):
        manager.set_mode(theme, persist=False)
        shell = build(capabilities)
        shell.resize(width, height)
        shell.show()
        _settle(app)
        page, controller = shell.design_page, shell.workspace.editor
        page.dialogs.ask_unsaved = lambda *_args: "discard"
        section = page.inspector.image
        bar = page.inspector.scroll.verticalScrollBar()

        def save(name: str) -> None:
            _settle(app)
            path = out / f"{theme}_{name}.png"
            if not page.grab().save(str(path)):
                raise RuntimeError(f"그림을 저장하지 못했습니다: {path}")
            written.append(path)

        def card(name: str, crop=None) -> None:
            path = out / f"{name}.png"
            _save_card(controller, path, crop)
            if path not in written:
                written.append(path)

        # ---- 사람 그림: AI ----
        page.start_panel.choose("blank")
        _settle(app)
        controller.set_card_setup(name="배경 지우기 확인")
        controller.document.side("front").background = CARD_COLOR
        controller.scene.sync_items() if hasattr(controller.scene, "sync_items") else None
        stripe = controller.add_item("rect")
        controller.set_properties(stripe.id, {"x": 0.0, "y": 34.0, "w": 85.6, "h": 12.0, "fill": "#1F4E79",
                                              "stroke_color": None, "name": "띠"})
        photo = controller.add_image_from_file(args.image)
        controller.set_properties(photo.id, {"x": 22.0, "y": 3.0, "w": 45.0, "h": 48.0, "fit": "contain",
                                             "name": "사람 그림"})
        controller.select(photo.id)
        _settle(app)
        bar.setValue(bar.maximum())
        save("1_before")
        card("card_1_before")

        started = time.time()
        section.start_removal()
        _wait(app, section)
        print(f"[{theme}] AI {time.time() - started:.1f}초, 남은 넓이 "
              f"{section._results[photo.id].preview.kept_ratio(section._settings(photo)):.0%}, "
              f"흰색 깔기 {'켬' if photo.mask_asset else '끔'}")
        bar.setValue(bar.maximum())
        save("2_after")
        card("card_2_after")
        card("card_2_after_zoom_head", (0.30, 0.05, 0.62, 0.50))
        card("card_2_after_zoom_body", (0.26, 0.45, 0.60, 0.98))

        # ---- 도장: AI가 못 찾으면 물어보고, 바탕색만 지우기로 ----
        controller.delete([photo.id])
        logo = controller.add_image_from_file(stamp)
        controller.set_properties(logo.id, {"x": 26.0, "y": 8.0, "w": 34.0, "h": 34.0, "fit": "contain", "name": "도장"})
        controller.select(logo.id)
        _settle(app)
        section.set_method("ai")
        section.start_removal()
        _wait(app, section)
        bar.setValue(bar.maximum())
        asked = bool(section.review_text())
        print(f"[{theme}] 도장을 AI로: {'결과가 이상해 물어봄' if asked else '그대로 적용됨'}")
        save("3_stamp_ai")
        if asked:
            section.try_color_button.click()
        else:
            section.restore_background()
            _settle(app)
            section.set_method("color")
            section.start_removal()
        _wait(app, section)
        bar.setValue(bar.maximum())
        save("4_stamp_color")
        card("card_4_stamp_color")
        card("card_4_stamp_color_zoom", (0.30, 0.12, 0.52, 0.50))

        shell.close()
        shell.workspace.close()
        shell.deleteLater()
        _settle(app)
    return written


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    if "HANA_STUDIO_SETTINGS_FILE" not in os.environ:
        folder = Path(tempfile.gettempdir()) / "hana_studio_ui"
        folder.mkdir(parents=True, exist_ok=True)
        os.environ["HANA_STUDIO_SETTINGS_FILE"] = str(folder / "cutout_demo.ini")

    from PySide6.QtWidgets import QApplication

    from ui.theme import configure_hidpi

    configure_hidpi()
    app = QApplication.instance() or QApplication(sys.argv[:1])
    for path in export(app, args):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
