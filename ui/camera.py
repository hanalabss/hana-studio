"""웹캠으로 사진 찍기(다눌의 '사진 캡쳐'). 카드 사진 칸이나 명단의 사람 사진에 바로 넣는다.

카메라가 없으면 그렇다고 알려 주고 '다시 찾기'로 다시 볼 수 있다. 찍은 사진은 넣을 칸 모양(가로:세로)에
맞춰 가운데를 잘라 준다. 카메라를 켜는 것은 창을 연 뒤이고, 창을 닫으면 바로 끈다.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QImage, QKeySequence, QShortcut
from PySide6.QtWidgets import QLabel, QSizePolicy, QStackedLayout, QWidget

from ui.widgets import ComboBox, Dialog, label

PREVIEW_SIZE = (480, 360)


def crop_to_aspect(image: QImage, aspect: float | None) -> QImage:
    """가운데를 기준으로 가로:세로 = aspect가 되게 자른다. aspect가 없으면 그대로."""
    if image.isNull() or not aspect or aspect <= 0:
        return image
    width, height = image.width(), image.height()
    if width / height > aspect:
        new_width = max(1, round(height * aspect))
        return image.copy(QRect((width - new_width) // 2, 0, new_width, height))
    new_height = max(1, round(width / aspect))
    return image.copy(QRect(0, (height - new_height) // 2, width, new_height))


def unique_path(folder: str | os.PathLike, stem: str, suffix: str = ".jpg") -> Path:
    """folder/stem.jpg. 이미 있으면 stem-2.jpg, stem-3.jpg …"""
    folder = Path(folder)
    safe = "".join(ch for ch in stem if ch not in '\\/:*?"<>|').strip() or "사진"
    candidate = folder / f"{safe}{suffix}"
    number = 2
    while candidate.exists():
        candidate = folder / f"{safe}-{number}{suffix}"
        number += 1
    return candidate


def save_photo(image: QImage, path: str | os.PathLike) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not image.save(str(path), quality=95):
        raise OSError(f"사진을 저장하지 못했습니다: {path}")
    return str(path)


def temp_photo_path() -> Path:
    handle, name = tempfile.mkstemp(prefix="hana_camera_", suffix=".jpg")
    os.close(handle)
    return Path(name)


class _Still(QLabel):
    """찍은 사진(넣을 칸 모양으로 자른 것), 또는 '카메라 없음' 안내를 보여 주는 칸."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setWordWrap(True)
        self.setMinimumSize(*PREVIEW_SIZE)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)


class CameraDialog(Dialog):
    """카메라 미리보기 → [찍기] → [다시 찍기 | 이 사진 쓰기]. 받은 사진은 captured_image()."""

    def __init__(self, parent: QWidget | None = None, aspect: float | None = None,
                 title: str = "카메라로 사진 찍기") -> None:
        super().__init__(title, "얼굴이 가운데 오게 맞추고 [찍기]를 누르세요. 스페이스 키로도 찍힙니다. "
                         "사진은 넣을 칸 모양에 맞춰 가운데가 잘립니다.",
                         parent, width=560)
        self.aspect = aspect
        self._image: QImage | None = None
        self._camera = None
        self._session = None
        self._capture = None
        self._video = None

        self.device_combo = ComboBox()
        self.device_combo.setAccessibleName("카메라")
        self.device_combo.activated.connect(lambda _i: self._start(self.device_combo.currentData()))
        self.body.addWidget(self.device_combo)

        self.stage = QWidget()
        self.stack = QStackedLayout(self.stage)
        self.still = _Still()
        self.stack.addWidget(self.still)
        self.body.addWidget(self.stage, 1)
        self.status = label("", "hint", wrap=True)
        self.body.addWidget(self.status)

        self.retry_button = self.add_button("다시 찾기", "secondary", None, left=True)
        self.retry_button.clicked.connect(self.refresh_devices)
        self.add_button("취소", "secondary", "reject")
        self.again_button = self.add_button("다시 찍기", "secondary", None)
        self.again_button.clicked.connect(self.retake)
        self.shoot_button = self.add_button("찍기", "primary", None, default=True)
        self.shoot_button.clicked.connect(self.shoot)
        self.use_button = self.add_button("이 사진 쓰기", "primary", None)
        self.use_button.clicked.connect(self.accept)
        QShortcut(QKeySequence(Qt.Key_Space), self, activated=self._space)

        self.refresh_devices()

    # ---- 장치 ----
    @staticmethod
    def devices() -> list:
        try:
            from PySide6.QtMultimedia import QMediaDevices
        except ImportError:
            return []
        return list(QMediaDevices.videoInputs())

    def refresh_devices(self) -> None:
        self._stop()
        self.device_combo.clear()
        devices = self.devices()
        for device in devices:
            self.device_combo.addItem(device.description(), device)
        self.device_combo.setVisible(len(devices) > 1)
        if not devices:
            self._show_message("연결된 카메라가 없습니다.\nUSB 카메라를 꽂은 뒤 [다시 찾기]를 누르세요.")
            self._sync_buttons()
            return
        self._start(devices[0])

    def _start(self, device) -> None:
        self._stop()
        if device is None:
            return
        try:
            from PySide6.QtMultimedia import QCamera, QImageCapture, QMediaCaptureSession
            from PySide6.QtMultimediaWidgets import QVideoWidget
        except ImportError:
            self._show_message("이 컴퓨터에서는 카메라 기능을 쓸 수 없습니다.")
            return
        self._image = None
        self._camera = QCamera(device)
        self._session = QMediaCaptureSession()
        self._capture = QImageCapture()
        if self._video is None:
            self._video = QVideoWidget()
            self._video.setMinimumSize(*PREVIEW_SIZE)
            self.stack.addWidget(self._video)
        self._session.setCamera(self._camera)
        self._session.setImageCapture(self._capture)
        self._session.setVideoOutput(self._video)
        self._capture.imageCaptured.connect(lambda _id, image: self.set_captured(image))
        self._capture.errorOccurred.connect(lambda _id, _err, text: self._show_message(f"찍지 못했습니다. {text}"))
        self._camera.errorOccurred.connect(lambda _err, text: self._show_message(f"카메라를 켜지 못했습니다. {text}"))
        self._camera.start()
        self.stack.setCurrentWidget(self._video)
        self.status.setText(f"{device.description()} 사용 중")
        self._sync_buttons()

    def _stop(self) -> None:
        if self._camera is not None:
            self._camera.stop()
        self._camera = self._session = self._capture = None

    # ---- 찍기 ----
    def _space(self) -> None:
        if self._image is None:
            self.shoot()
        else:
            self.accept()

    def shoot(self) -> None:
        if self._capture is None or not self._capture.isReadyForCapture():
            self.status.setText("카메라가 아직 준비되지 않았습니다. 잠시 뒤에 다시 눌러 주세요.")
            return
        self._capture.capture()

    def set_captured(self, image: QImage) -> None:
        """찍은 사진을 받는다(칸 모양으로 잘라 둔다). 테스트에서는 이것으로 사진을 넣는다."""
        self._image = crop_to_aspect(image, self.aspect)
        self.still.setPixmap(_fit_pixmap(self._image, self.still.size()))
        self.stack.setCurrentWidget(self.still)
        self.status.setText("이 사진을 쓰려면 [이 사진 쓰기], 다시 찍으려면 [다시 찍기]를 누르세요.")
        self._sync_buttons()

    def retake(self) -> None:
        self._image = None
        if self._video is not None and self._camera is not None:
            self.stack.setCurrentWidget(self._video)
        self._sync_buttons()

    def captured_image(self) -> QImage | None:
        return self._image

    # ---- 화면 ----
    def _show_message(self, text: str) -> None:
        self.still.clear()
        self.still.setText(text)
        self.stack.setCurrentWidget(self.still)
        self.status.setText("")

    def _sync_buttons(self) -> None:
        has_camera = self._camera is not None
        taken = self._image is not None
        self.shoot_button.setVisible(not taken)
        self.shoot_button.setEnabled(has_camera)
        self.again_button.setVisible(taken and has_camera)
        self.use_button.setVisible(taken)
        self.retry_button.setVisible(not has_camera)

    def done(self, result: int) -> None:
        self._stop()
        super().done(result)


def _fit_pixmap(image: QImage, size):
    from PySide6.QtGui import QPixmap

    return QPixmap.fromImage(image).scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation)


def take_photo(parent: QWidget | None, aspect: float | None = None) -> QImage | None:
    """카메라 창을 열어 찍은 사진을 돌려준다. 그만두면 None."""
    dialog = CameraDialog(parent, aspect)
    if dialog.exec() != Dialog.Accepted:
        return None
    return dialog.captured_image()

