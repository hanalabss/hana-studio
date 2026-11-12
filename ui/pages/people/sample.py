"""예시 파일 받기: 저장할 폴더를 묻고, 예시 명단과 사진을 만들고, 바로 가져와 볼지 묻는다."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QStandardPaths, Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QDialog, QFileDialog, QHBoxLayout, QWidget

from data.errors import DataError
from data.samples import SampleFiles, write_sample
from ui.theme import SPACE
from ui.widgets import Card, Dialog, IconWidget, label

from .importing import design_fields
from .inputs import ElidedLabel
from .text import has_final_consonant, people

TRY_BUTTON_TEXT = "이 파일로 바로 해 보기"


def default_sample_folder() -> str:
    """처음 보여 줄 폴더: 내 문서."""
    folder = QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation)
    return folder or str(Path.home())


def choose_sample_folder(parent: QWidget | None = None) -> str | None:
    folder = QFileDialog.getExistingDirectory(parent, "예시 파일을 저장할 폴더 고르기", default_sample_folder())
    return folder or None


def sample_headers_for(document) -> tuple[list[str] | None, list[str]]:
    """(카드가 쓰는 칸 이름들, 그 가운데 사진 칸). 카드에 사람마다 바뀌는 칸이 없으면 (None, [])."""
    fields = design_fields(document)
    if not fields:
        return None, []
    return [f.name for f in fields], [f.name for f in fields if f.is_photo]


def save_sample(document, folder: str | os.PathLike) -> SampleFiles:
    """카드에 맞는 예시 파일을 folder에 만든다. 못 만들면 DataError."""
    headers, photo_headers = sample_headers_for(document)
    return write_sample(folder, headers, photo_headers)


class SampleSavedDialog(Dialog):
    """어디에 무엇을 저장했는지 알리고, 바로 가져와 볼지 묻는다. exec()가 Accepted면 '바로 해 보기'."""

    def __init__(self, sample: SampleFiles, parent: QWidget | None = None):
        super().__init__("예시 파일을 저장했습니다",
                         "엑셀로 열어 보고, 같은 모양으로 우리 명단을 만들면 됩니다.", parent, width=600)
        self.setModal(True)
        self.sample = sample
        box = Card(padding=SPACE["md"], sunken=True, spacing=SPACE["sm"])
        self.folder_label = ElidedLabel(str(sample.directory), "strong")
        self.folder_label.setAccessibleName("저장한 폴더")
        box.body.addWidget(self.folder_label)
        found = sample.people - 1
        self.lines = [
            ("people", f"{sample.workbook_path.name}: 지어낸 {people(sample.people)}의 명단"),
            ("image", f"{sample.photo_dir.name} 폴더: 사진 {found}장. 파일 이름이 {sample.id_header}"
                      f"{'과' if has_final_consonant(sample.id_header) else '와'} 같습니다."),
            ("info", f"{sample.missing_photo_name} 님 사진은 일부러 뺐습니다. 사진 없는 사람의 예시입니다."),
        ]
        for icon, text in self.lines:
            row = QHBoxLayout()
            row.setSpacing(SPACE["sm"])
            row.addWidget(IconWidget(icon, 18, "text_muted"), 0, Qt.AlignTop)
            row.addWidget(label(text, "secondary", wrap=True), 1)
            box.body.addLayout(row)
        self.body.addWidget(box)

        self.open_button = self.add_button("폴더 열기", "quiet", None, left=True)
        self.open_button.setAccessibleName("예시 파일을 저장한 폴더 열기")
        self.open_button.clicked.connect(self.open_folder)
        self.close_button = self.add_button("닫기", "secondary", "reject")
        self.try_button = self.add_button(TRY_BUTTON_TEXT, "primary", "accept", default=True)
        self.try_button.setToolTip("방금 저장한 예시 명단과 사진으로 가져오기를 해 봅니다.")

    def open_folder(self) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.sample.directory)))


def offer_sample(parent: QWidget, document, on_error: Callable[[str], None],
                 folder: str | os.PathLike | None = None, ask: bool = True) -> tuple[SampleFiles | None, bool]:
    """예시 파일 받기 전체 흐름. (만든 파일, 바로 해 보기를 골랐는지)를 돌려준다.

    folder가 없으면 폴더를 묻는다(취소하면 (None, False)). ask=False면 저장만 하고 묻지 않는다.
    """
    if folder is None:
        folder = choose_sample_folder(parent)
        if not folder:
            return None, False
    try:
        sample = save_sample(document, folder)
    except DataError as exc:
        on_error(str(exc))
        return None, False
    if not ask:
        return sample, False
    dialog = SampleSavedDialog(sample, parent)
    wants_try = dialog.exec() == QDialog.Accepted
    dialog.deleteLater()
    return sample, wants_try
