"""그림 배경 지우기: AI(rembg) 결과나 바탕색으로 배경이 투명한 그림을 만든다. 무거운 일은 별도 스레드에서 한다."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Callable

import numpy as np
from PIL import Image, ImageOps
from PySide6.QtCore import QObject, QRectF, QThread, Qt, Signal
from PySide6.QtGui import QImage, QPainter, QPixmap

from design import cutout, storage
from design.cutout import (  # noqa: F401  (이 화면의 다른 파일과 테스트가 여기서 가져다 쓴다)
    DEFAULT_THRESHOLD, DEFAULT_TOLERANCE, MAX_THRESHOLD, MAX_TOLERANCE, METHOD_AI, METHOD_COLOR,
    MIN_THRESHOLD, MIN_TOLERANCE, THRESHOLD_KEY, CutoutSettings,
)
from ui.widgets import theme_color

PREVIEW_MAX_PX = 520


def stored_threshold(item) -> int | None:
    """사진 항목에 적어 둔 '지우는 정도'. 없거나 쓸 수 없는 값이면 None."""
    value = getattr(item, "extra", {}).get(THRESHOLD_KEY)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(min(MAX_THRESHOLD, max(MIN_THRESHOLD, round(value))))


def with_threshold(item, threshold: int | None) -> dict:
    """'지우는 정도'를 적어 넣은(None이면 지운) 새 extra. 다른 값은 그대로 둔다."""
    extra = {key: value for key, value in item.extra.items() if key != THRESHOLD_KEY}
    if threshold is not None:
        extra[THRESHOLD_KEY] = int(threshold)
    return extra


def has_settings(item) -> bool:
    """배경 지우기 설정(방법·정도·색)이 항목에 적혀 있는지."""
    return len(cutout.without_settings(item.extra)) != len(item.extra)


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
        from core.image_processor import ImageProcessor
        from core.model_loader import get_model_loader

        loader = get_model_loader()
        if not loader.is_ready():
            progress("배경을 알아보는 기능을 준비하고 있습니다. 처음에는 조금 걸립니다.")
            if not loader.wait_for_loading():
                raise RuntimeError("배경을 알아보는 기능을 준비하지 못했습니다. 잠시 뒤에 다시 해 보세요.")
        session = loader.get_loaded_session()
        progress("사진에서 사람과 배경을 가려내고 있습니다.")
        try:
            alpha = ImageProcessor().extract_alpha(image_path, session=session)
        except (RuntimeError, ValueError) as exc:
            # 처리기의 오류 문구는 개발자용이라 그대로 보여 주지 않는다.
            raise RuntimeError("사진을 읽거나 가려내는 중에 문제가 생겼습니다. 다른 사진으로 해 보세요.") from exc
        return np.asarray(alpha, dtype=np.uint8)


def load_oriented(image_path: str) -> Image.Image:
    """회전 정보를 반영해 연다. 원래 투명한 곳이 있는 그림은 RGBA, 아니면 RGB."""
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source)
        has_alpha = image.mode in ("RGBA", "LA", "PA") or "transparency" in image.info
        return image.convert("RGBA" if has_alpha else "RGB")


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


def _import_png(image: Image.Image, workdir: str, prefix: str) -> str:
    handle, temporary = tempfile.mkstemp(suffix=".png", prefix=prefix)
    os.close(handle)
    try:
        image.save(temporary, compress_level=3)
        return storage.import_asset(workdir, temporary)
    finally:
        try:
            os.remove(temporary)
        except OSError:
            pass


def save_cutout(rgb: np.ndarray, alpha: np.ndarray, workdir: str, white: bool) -> tuple[str, str | None]:
    """배경을 지운 그림(과 white면 흰 잉크 마스크)을 assets/에 넣고 상대 경로들을 돌려준다."""
    cut_asset = _import_png(cutout.compose(rgb, alpha), workdir, "cut_")
    mask_asset = _import_png(Image.fromarray(cutout.white_mask(alpha), "L"), workdir, "mask_") if white else None
    return cut_asset, mask_asset


def save_mask_of(cutout_path: str, workdir: str) -> str:
    """이미 만들어 둔 배경 지운 그림에서 흰 잉크 마스크를 만든다."""
    with Image.open(cutout_path) as image:
        alpha = np.array(image.convert("RGBA").getchannel("A"))
    return _import_png(Image.fromarray(cutout.white_mask(alpha), "L"), workdir, "mask_")


def _split(image: Image.Image) -> tuple[np.ndarray, np.ndarray | None]:
    if image.mode == "RGBA":
        data = np.array(image)
        return np.ascontiguousarray(data[..., :3]), np.ascontiguousarray(data[..., 3])
    return np.array(image.convert("RGB")), None


def _cut(rgb, base_alpha, own_alpha, settings: CutoutSettings) -> tuple[np.ndarray, np.ndarray]:
    out, alpha = cutout.cut(rgb, base_alpha, settings)
    if own_alpha is not None:
        alpha = np.minimum(alpha, own_alpha)
    return out, alpha


class CutoutPreview:
    """작게 줄인 그림과 알파. 정도를 바꿀 때마다 바로 다시 그릴 수 있다."""

    def __init__(self, rgb: np.ndarray, alpha: np.ndarray | None, own_alpha: np.ndarray | None = None) -> None:
        self.rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        self.alpha = None if alpha is None else np.ascontiguousarray(alpha, dtype=np.uint8)
        self.own_alpha = own_alpha

    @classmethod
    def build(cls, image: Image.Image, alpha: np.ndarray | None) -> "CutoutPreview":
        scale = min(1.0, PREVIEW_MAX_PX / max(image.size))
        size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        rgb, own = _split(image.resize(size, Image.Resampling.BILINEAR))
        small_alpha = None
        if alpha is not None:
            small_alpha = np.array(Image.fromarray(alpha, "L").resize(size, Image.Resampling.BILINEAR))
        return cls(rgb, small_alpha, own)

    def cut(self, settings: CutoutSettings) -> tuple[np.ndarray, np.ndarray]:
        return _cut(self.rgb, self.alpha, self.own_alpha, settings)

    def kept_ratio(self, settings: CutoutSettings) -> float:
        return cutout.kept_ratio(self.cut(settings)[1])

    def corner_color(self) -> tuple[int, int, int]:
        return cutout.corner_color(self.rgb)

    def target_rect(self, width: float, height: float) -> QRectF:
        h, w = self.rgb.shape[:2]
        scale = min(width / w, height / h)
        return QRectF((width - w * scale) / 2, (height - h * scale) / 2, w * scale, h * scale)

    def color_at(self, x: float, y: float, width: float, height: float) -> tuple[int, int, int] | None:
        """미리 보기 칸(width × height) 안의 한 점에 있는 그림 색. 그림 밖이면 None."""
        target = self.target_rect(width, height)
        if not target.contains(x, y) or target.width() <= 0:
            return None
        h, w = self.rgb.shape[:2]
        col = min(w - 1, max(0, int((x - target.left()) / target.width() * w)))
        row = min(h - 1, max(0, int((y - target.top()) / target.height() * h)))
        patch = self.rgb[max(0, row - 1): row + 2, max(0, col - 1): col + 2].reshape(-1, 3)
        color = np.median(patch, axis=0)
        return (int(color[0]), int(color[1]), int(color[2]))

    def pixmap(self, settings: CutoutSettings, width: int, height: int, dpr: float = 1.0) -> QPixmap:
        rgb, keep = self.cut(settings)
        rgba = np.ascontiguousarray(np.dstack([rgb, keep]))
        h, w = keep.shape
        image = QImage(rgba.data, w, h, w * 4, QImage.Format.Format_RGBA8888).copy()
        pixmap = QPixmap(round(width * dpr), round(height * dpr))
        pixmap.setDevicePixelRatio(dpr)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        target = self.target_rect(width, height)
        light, dark = theme_color("surface_sunken"), theme_color("surface_pressed")
        painter.setClipRect(target)
        cell = 8
        for row in range(int(height // cell) + 1):
            for col in range(int(width // cell) + 1):
                painter.fillRect(col * cell, row * cell, cell, cell, dark if (row + col) % 2 else light)
        painter.drawImage(target, image)
        painter.end()
        return pixmap


@dataclass
class CutoutWork:
    """그림 한 장을 가려낸 결과. 기억해 두면 정도를 바꿀 때 다시 계산하지 않는다."""

    asset: str | None
    method: str
    rgb: np.ndarray                        # 작업 크기(긴 변 최대 cutout.MAX_CUTOUT_PX)
    alpha: np.ndarray | None               # "ai": 모델이 준 알파. "color": 없음
    preview: CutoutPreview
    own_alpha: np.ndarray | None = None    # 그림이 원래 가지고 있던 투명도

    def cut(self, settings: CutoutSettings) -> tuple[np.ndarray, np.ndarray]:
        return _cut(self.rgb, self.alpha, self.own_alpha, settings)


def build_work(image_path: str, method: str, alpha: np.ndarray | None, asset: str | None = None) -> CutoutWork:
    image = load_oriented(image_path)
    if alpha is not None:
        alpha = fit_alpha(alpha, image.size)
    size = cutout.work_size(image.size)
    if size != image.size:
        image = image.resize(size, Image.Resampling.LANCZOS)
        if alpha is not None:
            alpha = np.array(Image.fromarray(alpha, "L").resize(size, Image.Resampling.LANCZOS))
    rgb, own = _split(image)
    return CutoutWork(asset, method, rgb, alpha, CutoutPreview.build(image, alpha), own)


class _Worker(QThread):
    done = Signal(object)           # CutoutWork
    failed = Signal(str)
    progress = Signal(str)

    def __init__(self, engine: BackgroundEngine | None, image_path: str, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._engine = engine
        self._path = image_path

    def run(self) -> None:
        try:
            if self._engine is None:
                self.done.emit(build_work(self._path, METHOD_COLOR, None))
            else:
                alpha = self._engine.compute_alpha(self._path, self.progress.emit)
                self.done.emit(build_work(self._path, METHOD_AI, alpha))
        except Exception as exc:
            self.failed.emit(str(exc) or "배경을 지우지 못했습니다.")


class RemovalJob(QObject):
    """배경 가려내기 한 번. 끝나면 finished(CutoutWork) 또는 failed(message)를 화면 스레드로 보낸다.

    engine이 None이면 AI 없이 그림만 읽는다(바탕색 지우기용).
    """

    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(self, engine: BackgroundEngine | None, image_path: str, parent: QObject | None = None) -> None:
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

    def _on_done(self, work) -> None:
        if not self._cancelled:
            self.finished.emit(work)

    def _on_failed(self, message: str) -> None:
        if not self._cancelled:
            self.failed.emit(message)

    def _on_progress(self, message: str) -> None:
        if not self._cancelled:
            self.progress.emit(message)
