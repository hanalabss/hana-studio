"""
CardCanvas - RTAI 카드 디자인을 위한 QGraphicsView
줌, 팬, 드래그 앤 드롭 지원
"""

from PySide6.QtWidgets import QGraphicsView
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPainter, QWheelEvent, QMouseEvent

from .card_scene import CardScene


class CardCanvas(QGraphicsView):
    """RTAI 카드 디자인 캔버스 뷰"""

    zoom_changed = Signal(float)  # 줌 레벨 변경 시그널

    def __init__(self):
        super().__init__()

        # 씬 생성 및 설정
        self.scene = CardScene()
        self.setScene(self.scene)

        # 줌 설정
        self.zoom_level = 1.0
        self.zoom_min = 0.5
        self.zoom_max = 2.0

        # 팬 모드 플래그
        self.is_panning = False
        self.pan_start_pos = None

        # 뷰 설정
        self._setup_view()

    def _setup_view(self):
        """뷰 기본 설정"""
        # 렌더링 힌트
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        # 뷰 설정
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # 초기 줌 레벨로 씬 중앙 표시
        self.centerOn(self.scene.get_card_rect().center())

    def wheelEvent(self, event: QWheelEvent):
        """마우스 휠로 줌 인/아웃"""
        # Ctrl 키 없이도 줌 가능
        delta = event.angleDelta().y()

        if delta > 0:
            # 줌 인
            self.zoom_in()
        else:
            # 줌 아웃
            self.zoom_out()

        event.accept()

    def mousePressEvent(self, event: QMouseEvent):
        """마우스 버튼 눌림 - 중간 버튼 드래그로 팬"""
        if event.button() == Qt.MouseButton.MiddleButton:
            # 팬 모드 시작
            self.is_panning = True
            self.pan_start_pos = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        """마우스 이동 - 팬 처리"""
        if self.is_panning and self.pan_start_pos:
            # 팬 이동
            delta = event.pos() - self.pan_start_pos
            self.pan_start_pos = event.pos()

            # 스크롤바 조정
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        """마우스 버튼 릴리즈 - 팬 종료"""
        if self.is_panning:
            self.is_panning = False
            self.pan_start_pos = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def zoom_in(self):
        """줌 인 (25% 증가)"""
        self.set_zoom(self.zoom_level * 1.25)

    def zoom_out(self):
        """줌 아웃 (25% 감소)"""
        self.set_zoom(self.zoom_level / 1.25)

    def set_zoom(self, zoom_level: float):
        """줌 레벨 설정"""
        # 최소/최대 제한
        zoom_level = max(self.zoom_min, min(self.zoom_max, zoom_level))

        if zoom_level != self.zoom_level:
            # 현재 줌에서 새 줌으로 변환
            scale_factor = zoom_level / self.zoom_level
            self.scale(scale_factor, scale_factor)

            self.zoom_level = zoom_level
            self.zoom_changed.emit(self.zoom_level)

    def reset_zoom(self):
        """줌 레벨을 100%로 리셋"""
        self.set_zoom(1.0)
        self.centerOn(self.scene.get_card_rect().center())

    def fit_in_view(self):
        """카드가 뷰에 딱 맞도록 조정"""
        self.fitInView(self.scene.get_card_rect(), Qt.AspectRatioMode.KeepAspectRatio)
        # 실제 줌 레벨 계산
        transform = self.transform()
        self.zoom_level = transform.m11()  # x축 스케일
        self.zoom_changed.emit(self.zoom_level)

    def get_zoom_percentage(self):
        """줌 레벨을 퍼센트로 반환"""
        return int(self.zoom_level * 100)
