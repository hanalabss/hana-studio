"""사진에서 얼굴을 찾아 사진 칸 안의 크기·위치(PhotoAdjustment)를 사람마다 맞춘다.

얼굴 찾기는 OpenCV에 들어 있는 정면 얼굴 검출기(Haar cascade)를 쓴다. 새 모델 파일을 받지 않아도
되고, 증명사진처럼 정면을 보는 사진에서는 충분히 찾는다. 옆모습·가린 얼굴·그림은 못 찾을 수 있고,
그때는 그 사람의 사진을 건드리지 않는다.

맞추는 기준(사진 칸 크기에 대한 비율):
  FACE_HEIGHT  검출된 얼굴 상자(이마~턱)의 높이. 머리카락까지 치면 칸 높이의 약 60%가 되고 위에 조금 여백이 남는다.
  FACE_CENTER  얼굴 상자 가운데가 놓일 위치. 가로는 가운데, 세로는 조금 위(어깨가 보이게).
사진이 작아 그 크기까지 못 키우거나, 옮기면 칸에 빈 곳이 생기는 경우에는 빈 곳이 생기지 않는
범위까지만 맞춘다(services.render_job.clamp_adjustment).
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

from PySide6.QtGui import QImage

from data.photos import PhotoResolver
from data.store import DataStore, PhotoAdjustment
from design import renderer
from services.render_job import PHOTO_ZOOM_MAX, PHOTO_ZOOM_MIN, _drawn_size, clamp_adjustment

log = logging.getLogger(__name__)

FACE_HEIGHT = 0.42
FACE_CENTER = (0.5, 0.48)
CASCADE_FILE = "haarcascade_frontalface_default.xml"
_DETECT_LONG_SIDE = 1024      # 큰 사진은 이 크기로 줄여서 찾는다(속도)
_MIN_FACE_RATIO = 0.08        # 짧은 변의 8%보다 작은 얼굴은 무시(배경의 잡음·작은 사람)


@dataclass(frozen=True)
class FaceBox:
    """사진(EXIF 회전 반영 뒤) 픽셀 좌표의 얼굴 상자."""
    x: float
    y: float
    w: float
    h: float

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.w / 2.0, self.y + self.h / 2.0


class FaceDetectorUnavailable(RuntimeError):
    """OpenCV나 검출기 파일이 없다."""


def _cascade_path() -> Path:
    candidates = []
    try:
        import cv2  # noqa: F401
        candidates.append(Path(cv2.data.haarcascades) / CASCADE_FILE)
    except Exception:  # pragma: no cover - cv2가 없거나 data가 빠진 배포판
        pass
    base = getattr(sys, "_MEIPASS", None)
    if base:
        candidates.append(Path(base) / "cv2" / "data" / CASCADE_FILE)
    for path in candidates:
        if path.is_file():
            return path
    raise FaceDetectorUnavailable("얼굴 찾기 파일이 없습니다. 프로그램을 다시 설치해 주세요.")


_classifier = None


def _detector():
    global _classifier
    if _classifier is None:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover
            raise FaceDetectorUnavailable("얼굴 찾기 기능을 쓸 수 없습니다(OpenCV 없음).") from exc
        path = _cascade_path()
        # 한글 경로를 OpenCV가 못 여는 경우가 있어 파일 내용을 직접 읽어 넘긴다.
        classifier = cv2.CascadeClassifier()
        storage = cv2.FileStorage(path.read_text(encoding="utf-8"), cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY)
        if not classifier.read(storage.getFirstTopLevelNode()):
            raise FaceDetectorUnavailable("얼굴 찾기 파일을 읽지 못했습니다.")
        _classifier = classifier
    return _classifier


def _gray_array(image: QImage):
    import numpy as np

    gray = image.convertToFormat(QImage.Format.Format_Grayscale8)
    width, height, stride = gray.width(), gray.height(), gray.bytesPerLine()
    buffer = np.frombuffer(gray.constBits(), dtype=np.uint8, count=stride * height)
    return buffer.reshape(height, stride)[:, :width].copy()


def detect_face(image: QImage | str | os.PathLike | None) -> FaceBox | None:
    """가장 큰 정면 얼굴. 못 찾으면 None. 좌표는 renderer.load_image와 같은(EXIF 회전 반영) 기준."""
    import cv2

    if image is None:
        return None
    if not isinstance(image, QImage):
        image = renderer.load_image(image)
        if image is None:
            return None
    if image.isNull() or image.width() < 16 or image.height() < 16:
        return None
    scale = min(1.0, _DETECT_LONG_SIDE / max(image.width(), image.height()))
    small = image if scale >= 1.0 else image.scaled(
        round(image.width() * scale), round(image.height() * scale))
    gray = cv2.equalizeHist(_gray_array(small))
    side = max(16, int(min(gray.shape) * _MIN_FACE_RATIO))
    faces = _detector().detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(side, side))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: int(f[2]) * int(f[3]))
    real = 1.0 / scale
    return FaceBox(float(x) * real, float(y) * real, float(w) * real, float(h) * real)


def fit_face(
    face: FaceBox, image_size: Sequence[float], frame_size: Sequence[float], fit: str = "fill",
    *, face_height: float = FACE_HEIGHT, center: tuple[float, float] = FACE_CENTER,
) -> PhotoAdjustment:
    """얼굴 상자가 칸 높이의 face_height만큼, center 위치에 오도록 하는 조정값(빈 곳 없게 잘라 낸 값)."""
    image_w, image_h = float(image_size[0]), float(image_size[1])
    frame_w, frame_h = float(frame_size[0]), float(frame_size[1])
    if min(image_w, image_h, frame_w, frame_h, face.h) <= 0:
        return PhotoAdjustment()
    base_w, _ = _drawn_size(image_size, frame_size, fit, 1.0)
    base_scale = base_w / image_w                       # zoom 1일 때 사진 1픽셀 = 칸 단위 몇
    zoom = (face_height * frame_h / face.h) / base_scale
    zoom = max(PHOTO_ZOOM_MIN, min(PHOTO_ZOOM_MAX, zoom))
    scale = base_scale * zoom
    face_x, face_y = face.center
    # 그려진 사진의 가운데 = 칸 가운데 + offset × 칸 크기 (PhotoAdjustment 정의)
    offset_x = center[0] - 0.5 - (face_x - image_w / 2.0) * scale / frame_w
    offset_y = center[1] - 0.5 - (face_y - image_h / 2.0) * scale / frame_h
    return clamp_adjustment(PhotoAdjustment(zoom, offset_x, offset_y), image_size, frame_size, fit)


@dataclass
class FaceFitResult:
    fitted: list[int] = field(default_factory=list)      # 맞춘 사람 id
    no_face: list[int] = field(default_factory=list)     # 사진은 있는데 얼굴을 못 찾음
    no_photo: list[int] = field(default_factory=list)    # 사진 파일이 없음

    @property
    def total(self) -> int:
        return len(self.fitted) + len(self.no_face) + len(self.no_photo)


def fit_people(
    store: DataStore, column: str, frame_size: Sequence[float], fit: str = "fill",
    ids: Iterable[int] | None = None, *, progress: Callable[[int, int], bool] | None = None,
) -> FaceFitResult:
    """명단의 사진 열 column에서 사람마다 얼굴을 찾아 사진 조정값을 저장한다.

    얼굴을 못 찾은 사람의 기존 조정값은 그대로 둔다. progress(done, total)가 False를 돌려주면 멈춘다.
    """
    resolver = PhotoResolver(store)
    targets = list(store.ids() if ids is None else ids)
    result = FaceFitResult()
    for done, record_id in enumerate(targets):
        if progress is not None and progress(done, len(targets)) is False:
            break
        record = store.get(record_id)
        path = resolver.path(record, column) if record is not None else None
        image = renderer.load_image(path) if path else None
        if image is None:
            result.no_photo.append(record_id)
            continue
        face = detect_face(image)
        if face is None:
            result.no_face.append(record_id)
            continue
        adjustment = fit_face(face, (image.width(), image.height()), frame_size, fit)
        store.set_photo_adjustment(record_id, column, adjustment)
        result.fitted.append(record_id)
    if progress is not None:
        progress(len(targets), len(targets))
    return result
