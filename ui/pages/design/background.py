"""사진 배경 지우기: 기존 AI(rembg) 결과를 화이트 층 마스크로 바꾼다. 무거운 일은 별도 스레드에서 한다."""

from __future__ import annotations

import os
import tempfile
from typing import Callable

import numpy as np
from PIL import Image, ImageOps
from PySide6.QtCore import QObject, QRectF, QThread, Qt, Signal
from PySide6.QtGui import QImage, QPainter, QPixmap

from design import storage
from ui.widgets import theme_color

DEFAULT_THRESHOLD = 200
PREVIEW_MAX_PX = 520


class BackgroundEngine:
    """배경 분리 엔진. prepare는 화면 스레드에서, compute_alpha는 작업 스레드에서 불린다."""

    def prepare(self, parent=None) -> bool:
        """필요한 준비(모델 내려받기 등)를 한다. 사용자가 그만두면 False."""
        return True

    def compute_alpha(self, image_path: str, progress: Callable[[str], None]) -> np.ndarray:
        """이미지와 같은 크기의 uint8 배열(255 = 사람·물건, 0 = 배경)을 돌려준다."""
        raise NotImplementedError


class RembgEngine(BackgroundEngine):
    """'빠른 인쇄' 화면이 쓰는 것과 같은 모델·세션(core.model_loader)을 쓴다."""

    def default_threshold(self) -> int:
        try:
            from config import config
            return int(config.get("alpha_threshold", DEFAULT_THRESHOLD))
        except Exception:
            return DEFAULT_THRESHOLD

    def prepare(self, parent=None) -> bool:
        from core.model_downloader import ensure_model_downloaded, is_model_downloaded
        from core.model_loader import get_model_loader

        loader = get_model_loader()
        if loader.is_ready() or is_model_downloaded(loader.model_name):
            return True
        return bool(ensure_model_downloaded(loader.model_name, parent))

    def compute_alpha(self, image_path: str, progress: Callable[[str], None]) -> np.ndarray:
        import io

        from core.image_processor import ImageProcessor
        from core.model_loader import get_model_loader

        loader = get_model_loader()
        if not loader.is_ready():
            progress("배경을 알아보는 기능을 준비하고 있습니다. 처음에는 조금 걸립니다.")
            if not loader.wait_for_loading():
                raise RuntimeError("배경을 알아보는 기능을 준비하지 못했습니다. 잠시 뒤에 다시 해 보세요.")
        session = loader.get_loaded_session()
        processor = ImageProcessor()
        extract = getattr(processor, "extract_alpha", None)
        progress("사진에서 사람과 배경을 가려내고 있습니다.")
        if callable(extract):
            return np.asarray(extract(image_path, session=session), dtype=np.uint8)
        # ImageProcessor.remove_background는 임계값을 적용한 결과만 주므로, 같은 방식으로 알파만 직접 얻는다.
        from rembg import remove

        with open(image_path, "rb") as handle:
            result = remove(handle.read(), session=session)
        return np.array(Image.open(io.BytesIO(result)).convert("RGBA").split()[-1], dtype=np.uint8)


def load_oriented(image_path: str) -> Image.Image:
    with Image.open(image_path) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


def fit_alpha(alpha: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """알파를 이미지 크기(가로, 세로)에 맞춘다. 회전 정보 때문에 가로세로가 뒤바뀐 경우도 바로잡는다."""
    alpha = np.asarray(alpha, dtype=np.uint8)
    if alpha.ndim == 3:
        alpha = alpha[..., -1]
    width, height = size
    if alpha.shape == (height, width):
        return alpha
    image = Image.fromarray(alpha, "L")
    if alpha.shape == (width, height) and width != height:
        image = image.transpose(Image.Transpose.ROTATE_270)
    return np.array(image.resize((width, height), Image.Resampling.BILINEAR), dtype=np.uint8)


def mask_from_alpha(alpha: np.ndarray, threshold: int) -> np.ndarray:
    """화이트 층 마스크: 흰색(255) = 흰 잉크를 까는 곳 = 사람·물건."""
    return np.where(np.asarray(alpha) > int(threshold), 255, 0).astype(np.uint8)


def save_mask(alpha: np.ndarray, threshold: int, workdir: str) -> str:
    """마스크를 문서 작업 폴더의 assets/에 넣고 상대 경로를 돌려준다."""
    handle, temporary = tempfile.mkstemp(suffix=".png", prefix="mask_")
    os.close(handle)
    try:
        Image.fromarray(mask_from_alpha(alpha, threshold), "L").save(temporary)
        return storage.import_asset(workdir, temporary)
    finally:
        try:
            os.remove(temporary)
        except OSError:
            pass


class CutoutPreview:
    """작게 줄인 사진과 알파. 임계값이 바뀔 때마다 바로 다시 그릴 수 있다."""

    def __init__(self, rgb: np.ndarray, alpha: np.ndarray) -> None:
        self.rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        self.alpha = np.ascontiguousarray(alpha, dtype=np.uint8)

    @classmethod
    def build(cls, image: Image.Image, alpha: np.ndarray) -> "CutoutPreview":
        scale = min(1.0, PREVIEW_MAX_PX / max(image.size))
        size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        small = image.resize(size, Image.Resampling.BILINEAR)
        small_alpha = Image.fromarray(alpha, "L").resize(size, Image.Resampling.BILINEAR)
        return cls(np.array(small), np.array(small_alpha))

    def kept_ratio(self, threshold: int) -> float:
        return float(np.count_nonzero(self.alpha > int(threshold))) / max(1, self.alpha.size)

    def pixmap(self, threshold: int, width: int, height: int, dpr: float = 1.0) -> QPixmap:
        keep = np.where(self.alpha > int(threshold), 255, 0).astype(np.uint8)
        rgba = np.ascontiguousarray(np.dstack([self.rgb, keep]))
        h, w = keep.shape
        image = QImage(rgba.data, w, h, w * 4, QImage.Format.Format_RGBA8888).copy()
        pixmap = QPixmap(round(width * dpr), round(height * dpr))
        pixmap.setDevicePixelRatio(dpr)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        scale = min(width / w, height / h)
        target = QRectF((width - w * scale) / 2, (height - h * scale) / 2, w * scale, h * scale)
        light, dark = theme_color("surface_sunken"), theme_color("surface_pressed")
        painter.setClipRect(target)
        cell = 8
        for row in range(int(height // cell) + 1):
            for col in range(int(width // cell) + 1):
                painter.fillRect(col * cell, row * cell, cell, cell, dark if (row + col) % 2 else light)
        painter.drawImage(target, image)
        painter.end()
        return pixmap


class _Worker(QThread):
    done = Signal(object, object)   # alpha(원본 크기), CutoutPreview
    failed = Signal(str)
    progress = Signal(str)

    def __init__(self, engine: BackgroundEngine, image_path: str, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._engine = engine
        self._path = image_path

    def run(self) -> None:
        try:
            image = load_oriented(self._path)
            alpha = fit_alpha(self._engine.compute_alpha(self._path, self.progress.emit), image.size)
            self.done.emit(alpha, CutoutPreview.build(image, alpha))
        except Exception as exc:
            self.failed.emit(str(exc) or "배경을 지우지 못했습니다.")


class RemovalJob(QObject):
    """배경 분리 한 번. 끝나면 finished(alpha, preview) 또는 failed(message)를 화면 스레드로 보낸다."""

    finished = Signal(object, object)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(self, engine: BackgroundEngine, image_path: str, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._cancelled = False
        self._worker = _Worker(engine, image_path, self)
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.progress.connect(self._on_progress)

    def start(self) -> None:
        self._worker.start()

    def is_running(self) -> bool:
        return self._worker.isRunning()

    def cancel(self) -> None:
        """결과를 버린다. 이미 시작한 계산은 끝까지 돌지만 화면에는 반영하지 않는다."""
        self._cancelled = True

    def wait(self, msecs: int = 30000) -> bool:
        return self._worker.wait(msecs)

    def _on_done(self, alpha, preview) -> None:
        if not self._cancelled:
            self.finished.emit(alpha, preview)

    def _on_failed(self, message: str) -> None:
        if not self._cancelled:
            self.failed.emit(message)

    def _on_progress(self, message: str) -> None:
        if not self._cancelled:
            self.progress.emit(message)
