"""사진 작은 그림. 그릴 때는 기다리지 않고, 한 번에 하나씩 읽어 채운다."""

from __future__ import annotations

from collections import OrderedDict, deque

from PySide6.QtCore import QObject, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QImage, QImageIOHandler, QImageReader

THUMB_W, THUMB_H = 30, 38
_SCALE = 3          # 150% 배율 화면에서도 또렷하게
_CAPACITY = 600


def read_cover(path: str, width: int, height: int) -> QImage:
    """틀을 꽉 채우도록 줄여 읽는다. 얼굴이 잘리지 않게 위쪽을 조금 더 남긴다."""
    reader = QImageReader(path)
    reader.setAutoTransform(True)
    size = reader.size()
    if size.isValid() and size.width() > 0 and size.height() > 0:
        turned = bool(reader.transformation() & QImageIOHandler.Transformation.TransformationRotate90)
        if turned:
            size = QSize(size.height(), size.width())
        ratio = max(width / size.width(), height / size.height())
        scaled = QSize(max(1, round(size.width() * ratio)), max(1, round(size.height() * ratio)))
        if turned:
            scaled = QSize(scaled.height(), scaled.width())
        if ratio < 1.0:
            reader.setScaledSize(scaled)
    image = reader.read()
    if image.isNull():
        return image
    if image.width() != width or image.height() != height:
        image = image.scaled(width, height, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
    x = max(0, (image.width() - width) // 2)
    y = max(0, round((image.height() - height) * 0.3))
    return image.copy(x, y, width, height)


class ThumbLoader(QObject):
    loaded = Signal()

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._cache: OrderedDict[str, QImage] = OrderedDict()
        self._queue: deque[str] = deque()
        self._queued: set[str] = set()
        self._timer = QTimer(self)
        self._timer.setInterval(0)
        self._timer.timeout.connect(self._step)

    def get(self, path: str) -> QImage | None:
        """있으면 바로 주고, 없으면 읽을 줄에 세운 뒤 None."""
        image = self._cache.get(path)
        if image is not None:
            self._cache.move_to_end(path)
            return image
        if path not in self._queued:
            self._queued.add(path)
            self._queue.append(path)
            if len(self._queue) > 80:
                self._queued.discard(self._queue.popleft())
            self._timer.start()
        return None

    def _load(self, path: str) -> None:
        image = read_cover(path, THUMB_W * _SCALE, THUMB_H * _SCALE)
        self._cache[path] = image
        while len(self._cache) > _CAPACITY:
            self._cache.popitem(last=False)

    def _step(self) -> None:
        if not self._queue:
            self._timer.stop()
            return
        # 방금 화면에 들어온 줄부터 읽는다.
        path = self._queue.pop()
        self._queued.discard(path)
        self._load(path)
        if not self._queue:
            self._timer.stop()
        self.loaded.emit()

    def flush(self) -> None:
        while self._queue:
            path = self._queue.pop()
            self._queued.discard(path)
            self._load(path)
        self._timer.stop()
        self.loaded.emit()

    def forget(self, path: str | None = None) -> None:
        if path is None:
            self._cache.clear()
        else:
            self._cache.pop(path, None)
