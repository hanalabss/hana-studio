"""편집기 캔버스 데모.

    python scripts/editor_demo.py                 # 창 띄우기
    python scripts/editor_demo.py --shot <폴더>   # 화면 없이 PNG 스크린샷 저장
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

SAMPLE_RECORD = {"이름": "김하늘", "부서": "경영지원팀", "직급": "사원", "사번": "HN260417"}


def make_sample_photo(directory: str) -> str:
    from PySide6.QtCore import QPointF, QRectF, Qt
    from PySide6.QtGui import QColor, QImage, QPainter

    image = QImage(360, 450, QImage.Format.Format_RGB32)
    image.fill(QColor("#C9D6E8"))
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor("#F1D3BC"))
    painter.drawEllipse(QPointF(180, 170), 78, 90)
    painter.setBrush(QColor("#2B3A55"))
    painter.drawChord(QRectF(96, 70, 168, 150), 0, 180 * 16)
    painter.setBrush(QColor("#3F5A8A"))
    painter.drawRoundedRect(QRectF(50, 290, 260, 220), 90, 90)
    painter.end()
    path = os.path.join(directory, "sample_photo.png")
    image.save(path)
    return path


def build_sample(controller, *, bleed: bool = True, orientation: str = "landscape") -> None:
    from design import presets

    preset = presets.CUT_54X86_BLEED if bleed else presets.CR80
    controller.new_document(preset, "employee", orientation=orientation, name="2026 신입 사원증")


def find_item(controller, name: str):
    return next(item for item in controller.side().items if item.name == name)


def build_window(palette_name: str = "light"):
    from PySide6.QtGui import QAction, QKeySequence
    from PySide6.QtWidgets import QFileDialog, QLabel, QMainWindow, QMessageBox, QToolBar

    from ui.editor import EditorController, EditorError, EditorPalette, EditorView

    window = QMainWindow()
    window.setWindowTitle("하나스튜디오 편집기 데모")
    controller = EditorController(window)
    palettes = {"light": EditorPalette.light(), "dark": EditorPalette.dark()}
    state = {"palette": palette_name, "photo_dir": tempfile.mkdtemp(prefix="hsdemo_")}
    view = EditorView(controller, palettes[palette_name])
    window.setCentralWidget(view)
    window.controller, window.view = controller, view
    build_sample(controller)

    status = QLabel()
    window.statusBar().addPermanentWidget(status)
    controller.message.connect(lambda text: window.statusBar().showMessage(text, 4000))

    def update_status(*_args) -> None:
        width, height = controller.side_size_mm()
        problems = len(controller.problems())
        status.setText(
            f"{width:g} × {height:g} mm · {round(view.zoom * 100)}% · 선택 {len(controller.selection)}개"
            f" · 확인할 점 {problems}개{' · 수정됨' if controller.is_modified else ''}"
        )

    for signal in (view.zoom_changed, controller.selection_changed, controller.problems_changed,
                   controller.modified_changed, controller.side_changed, controller.card_changed):
        signal.connect(update_status)

    def guarded(func):
        def run(*_args):
            try:
                func()
            except EditorError as exc:
                QMessageBox.warning(window, "알림", str(exc))
        return run

    def add_bar(title: str) -> QToolBar:
        bar = QToolBar(title)
        window.addToolBar(bar)
        return bar

    def action(bar: QToolBar, text: str, func, shortcut=None, checkable=False, checked=False) -> QAction:
        act = QAction(text, window)
        if checkable:
            act.setCheckable(True)
            act.setChecked(checked)
            act.toggled.connect(func)
        else:
            act.triggered.connect(guarded(func))
        if shortcut:
            act.setShortcut(QKeySequence(shortcut))
        bar.addAction(act)
        return act

    insert = add_bar("넣기")
    for kind, label in (("text", "글자"), ("image", "사진 칸"), ("rect", "사각형"), ("ellipse", "원"),
                        ("line", "선"), ("barcode", "바코드"), ("qr", "QR")):
        action(insert, label, lambda kind=kind: controller.add_item(kind))

    def add_image_file() -> None:
        path, _ = QFileDialog.getOpenFileName(window, "이미지 고르기", "", "이미지 (*.png *.jpg *.jpeg *.bmp *.gif *.webp)")
        if path:
            controller.add_image_from_file(path)

    action(insert, "이미지 파일", add_image_file)
    action(insert, "명단: 이름", lambda: controller.add_bound_field("이름", "text"))
    action(insert, "명단: 사번 QR", lambda: controller.add_bound_field("사번", "qr"))

    edit = add_bar("편집")
    undo = controller.undo_stack.createUndoAction(window, "되돌리기")
    undo.setShortcut(QKeySequence.StandardKey.Undo)
    redo = controller.undo_stack.createRedoAction(window, "다시 하기")
    redo.setShortcut(QKeySequence.StandardKey.Redo)
    edit.addAction(undo)
    edit.addAction(redo)
    action(edit, "복제", controller.duplicate)
    action(edit, "삭제", controller.delete)
    action(edit, "맨 앞", controller.bring_to_front)
    action(edit, "맨 뒤", controller.send_to_back)
    action(edit, "잠금", lambda: controller.set_locked(None, True))
    action(edit, "모두 잠금 해제", lambda: controller.set_locked([it.id for it in controller.side().items], False))

    arrange = add_bar("정렬")
    for mode, label in (("left", "왼쪽"), ("center", "가운데"), ("right", "오른쪽"),
                        ("top", "위"), ("middle", "중간"), ("bottom", "아래")):
        action(arrange, label, lambda mode=mode: controller.align(mode))
    action(arrange, "가로 간격", lambda: controller.distribute("horizontal"))
    action(arrange, "세로 간격", lambda: controller.distribute("vertical"))

    window.addToolBarBreak()
    look = add_bar("보기")
    action(look, "확대", view.zoom_in)
    action(look, "축소", view.zoom_out)
    action(look, "화면에 맞춤", view.zoom_fit)
    action(look, "100%", view.zoom_actual)
    action(look, "선택 확대", view.zoom_to_selection)
    action(look, "모눈", controller.set_grid_visible, checkable=True)
    action(look, "달라붙기", controller.set_snap_enabled, checkable=True, checked=True)
    action(look, "재단선", controller.set_cut_line_visible, checkable=True, checked=True)
    action(look, "눈금자", view.set_rulers_visible, checkable=True, checked=True)

    def toggle_palette() -> None:
        state["palette"] = "dark" if state["palette"] == "light" else "light"
        view.set_palette(palettes[state["palette"]])

    action(look, "밝게/어둡게", toggle_palette)

    card = add_bar("카드")
    action(card, "앞면", lambda: controller.set_side("front"))
    action(card, "뒷면", lambda: controller.set_side("back"))
    action(card, "가로", lambda: controller.set_orientation("landscape"))
    action(card, "세로", lambda: controller.set_orientation("portrait"))
    action(card, "도련 있음", lambda: controller.set_card_setup(preset="cut_54x86_bleed"))
    action(card, "CR80", lambda: controller.set_card_setup(preset="cr80"))

    def toggle_record(on: bool) -> None:
        if on:
            record = dict(SAMPLE_RECORD, 사진=make_sample_photo(state["photo_dir"]))
            controller.set_preview_record(record)
        else:
            controller.set_preview_record(None)

    action(card, "예시 사람 보기", toggle_record, checkable=True)

    def save_as() -> None:
        path, _ = QFileDialog.getSaveFileName(window, "디자인 저장", "", "하나스튜디오 디자인 (*.hsd)")
        if path:
            controller.save(path)

    def open_file() -> None:
        path, _ = QFileDialog.getOpenFileName(window, "디자인 열기", "", "하나스튜디오 디자인 (*.hsd)")
        if path:
            controller.open(path)

    action(card, "저장", save_as)
    action(card, "열기", open_file)
    update_status()
    window.resize(1180, 780)
    return window


def _drag(view, start, end, *, release: bool, steps: int = 6) -> None:
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtWidgets import QApplication

    viewport = view.viewport()

    def send(kind, pos, button, buttons):
        point = QPointF(pos)
        event = QMouseEvent(kind, point, QPointF(viewport.mapToGlobal(point.toPoint())), button, buttons,
                            Qt.KeyboardModifier.NoModifier)
        QApplication.sendEvent(viewport, event)

    left = Qt.MouseButton.LeftButton
    send(QEvent.Type.MouseButtonPress, start, left, left)
    for index in range(1, steps + 1):
        t = index / steps
        send(QEvent.Type.MouseMove, start + (end - start) * t, Qt.MouseButton.NoButton, left)
    if release:
        send(QEvent.Type.MouseButtonRelease, end, left, Qt.MouseButton.NoButton)


def take_shots(directory: str) -> list[str]:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    from ui.editor import EditorController, EditorPalette, EditorView

    app = QApplication.instance()
    os.makedirs(directory, exist_ok=True)
    photo = make_sample_photo(tempfile.mkdtemp(prefix="hsdemo_"))
    record = dict(SAMPLE_RECORD, 사진=photo)
    saved: list[str] = []

    def shoot(view, name: str) -> None:
        app.processEvents()
        path = os.path.join(directory, name)
        view.grab().save(path)
        saved.append(path)

    def make(palette, **sample):
        controller = EditorController()
        build_sample(controller, **sample)
        view = EditorView(controller, palette)
        view.resize(1040, 720)
        view.show()
        app.processEvents()
        return controller, view

    for palette in (EditorPalette.light(), EditorPalette.dark()):
        controller, view = make(palette)
        controller.set_preview_record(record)
        controller.select([find_item(controller, "이름").id])
        shoot(view, f"employee_{palette.name}.png")

        controller.set_grid_visible(True)
        name = find_item(controller, "이름")
        photo_item = find_item(controller, "사진")
        start = view.mm_to_viewport(*name.center())
        target_x = photo_item.bounds()[2] + 3.0 + name.w / 2.0
        end = view.mm_to_viewport(target_x, photo_item.center()[1] + 0.3)
        _drag(view, start, end, release=False)
        shoot(view, f"drag_guides_{palette.name}.png")
        _drag(view, end, end, release=True, steps=1)
        view.close()
        controller.close()

    controller, view = make(EditorPalette.dark(), bleed=False)
    stamp = controller.add_item("rect", fill="#FFD54F", w=22.0, h=12.0, pos=(70.0, 44.0))
    controller.set_rotation([stamp.id], 20.0)
    controller.move_items_to({stamp.id: (72.0, 44.0)})
    line = controller.add_item("line", pos=(60.0, 20.0), color="#C62828", width_mm=0.6)
    controller.set_locked([find_item(controller, "상단 띠").id], True)
    controller.select([stamp.id])
    shoot(view, "rotated_warning_dark.png")
    controller.select([line.id])
    shoot(view, "line_selected_dark.png")
    view.close()
    controller.close()

    controller, view = make(EditorPalette.light(), orientation="portrait")
    controller.set_preview_record(record)
    controller.select([find_item(controller, "사번 바코드").id, find_item(controller, "이름").id])
    shoot(view, "portrait_multi_light.png")
    view.close()
    controller.close()
    return saved


def main() -> int:
    parser = argparse.ArgumentParser(description="하나스튜디오 편집기 캔버스 데모")
    parser.add_argument("--shot", metavar="DIR", help="화면 없이 스크린샷을 이 폴더에 저장")
    parser.add_argument("--dark", action="store_true", help="어두운 색으로 시작")
    args = parser.parse_args()
    if args.shot:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"

    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv[:1])
    if args.shot:
        for path in take_shots(args.shot):
            print(path)
        return 0
    window = build_window("dark" if args.dark else "light")
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
