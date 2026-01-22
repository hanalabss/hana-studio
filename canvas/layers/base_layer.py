"""
BaseLayer - 모든 레이어의 기본 클래스
드래그, 크기 조절, 회전 핸들 제공
"""

from PySide6.QtWidgets import QGraphicsItem, QGraphicsEllipseItem
from PySide6.QtCore import Qt, QRectF, QPointF, Signal, QObject
from PySide6.QtGui import QPen, QBrush, QColor, QCursor
import math


class LayerSignals(QObject):
    """레이어 시그널"""
    selection_changed = Signal(bool)  # 선택 상태 변경
    transform_changed = Signal()  # 변형 변경 (위치, 크기, 회전)


class ResizeHandle(QGraphicsEllipseItem):
    """크기 조절 핸들"""

    def __init__(self, position, parent=None):
        super().__init__(-4, -4, 8, 8, parent)  # 시각적 크기: 8x8px
        self.position = position  # 'tl', 'tr', 'bl', 'br' 등
        self._is_dragging = False
        self.setBrush(QBrush(QColor("#4A90E2")))
        self.setPen(QPen(QColor("#FFFFFF"), 2))
        self.setZValue(1000)
        self.setCursor(self._get_cursor())
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)

    def shape(self):
        """클릭 판정 영역 확장"""
        from PySide6.QtGui import QPainterPath
        path = QPainterPath()
        path.addEllipse(-10, -10, 20, 20)  # 클릭 판정: 20x20px
        return path

    def mousePressEvent(self, event):
        """핸들 클릭 시 부모에게 리사이즈 시작 알림"""
        self._is_dragging = True
        self.grabMouse()  # 마우스 캡처 - 다른 아이템이 간섭 못함
        parent = self.parentItem()
        if parent and hasattr(parent, '_start_resize_from_handle'):
            parent._start_resize_from_handle(self.position, event)
        event.accept()

    def mouseMoveEvent(self, event):
        """드래그 시 부모에게 전달"""
        if not self._is_dragging:
            return
        parent = self.parentItem()
        if parent and hasattr(parent, '_handle_resize_from_handle'):
            parent._handle_resize_from_handle(event)
        event.accept()

    def mouseReleaseEvent(self, event):
        """릴리즈 시 부모에게 전달"""
        self._is_dragging = False
        self.ungrabMouse()  # 마우스 릴리즈
        parent = self.parentItem()
        if parent and hasattr(parent, '_end_resize_from_handle'):
            parent._end_resize_from_handle(event)
        event.accept()

    def _get_cursor(self):
        """위치에 따른 커서 설정"""
        cursor_map = {
            'tl': Qt.CursorShape.SizeFDiagCursor,  # 좌상단
            'tr': Qt.CursorShape.SizeBDiagCursor,  # 우상단
            'bl': Qt.CursorShape.SizeBDiagCursor,  # 좌하단
            'br': Qt.CursorShape.SizeFDiagCursor,  # 우하단
            't': Qt.CursorShape.SizeVerCursor,     # 상단
            'b': Qt.CursorShape.SizeVerCursor,     # 하단
            'l': Qt.CursorShape.SizeHorCursor,     # 좌측
            'r': Qt.CursorShape.SizeHorCursor,     # 우측
        }
        return cursor_map.get(self.position, Qt.CursorShape.SizeAllCursor)

    def paint(self, painter, option, widget):
        """핸들 그리기 - 인쇄 모드에서는 렌더링하지 않음"""
        # 인쇄 모드 확인
        scene = self.scene()
        if scene and hasattr(scene, 'is_printing') and scene.is_printing:
            return  # 인쇄 모드에서는 그리지 않음

        # 일반 모드에서만 렌더링
        super().paint(painter, option, widget)


class RotateHandle(QGraphicsEllipseItem):
    """회전 핸들"""

    def __init__(self, parent=None):
        super().__init__(-5, -5, 10, 10, parent)  # 시각적 크기: 10x10px
        self._is_dragging = False
        self.setBrush(QBrush(QColor("#28A745")))
        self.setPen(QPen(QColor("#FFFFFF"), 2))
        self.setZValue(1000)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)

    def shape(self):
        """클릭 판정 영역 확장"""
        from PySide6.QtGui import QPainterPath
        path = QPainterPath()
        path.addEllipse(-10, -10, 20, 20)  # 클릭 판정: 20x20px
        return path

    def mousePressEvent(self, event):
        """핸들 클릭 시 부모에게 회전 시작 알림"""
        self._is_dragging = True
        self.grabMouse()  # 마우스 캡처
        parent = self.parentItem()
        if parent and hasattr(parent, '_start_rotate_from_handle'):
            parent._start_rotate_from_handle(event)
        event.accept()

    def mouseMoveEvent(self, event):
        """드래그 시 부모에게 전달"""
        if not self._is_dragging:
            return
        parent = self.parentItem()
        if parent and hasattr(parent, '_handle_rotate_from_handle'):
            parent._handle_rotate_from_handle(event)
        event.accept()

    def mouseReleaseEvent(self, event):
        """릴리즈 시 부모에게 전달"""
        self._is_dragging = False
        self.ungrabMouse()  # 마우스 릴리즈
        parent = self.parentItem()
        if parent and hasattr(parent, '_end_rotate_from_handle'):
            parent._end_rotate_from_handle(event)
        event.accept()

    def paint(self, painter, option, widget):
        """핸들 그리기 - 인쇄 모드에서는 렌더링하지 않음"""
        # 인쇄 모드 확인
        scene = self.scene()
        if scene and hasattr(scene, 'is_printing') and scene.is_printing:
            return  # 인쇄 모드에서는 그리지 않음

        # 일반 모드에서만 렌더링
        super().paint(painter, option, widget)


class BaseLayer(QGraphicsItem):
    """기본 레이어 클래스"""

    def __init__(self):
        super().__init__()

        self.signals = LayerSignals()

        # 레이어 속성
        self.layer_name = "Layer"
        self.layer_locked = False
        self.layer_visible = True

        # 플래그 설정
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsFocusable, True)

        # 크기 조절 핸들
        self.resize_handles = {}
        self.rotate_handle = None
        self.is_resizing = False
        self.resize_start_rect = None
        self.resize_start_pos = None
        self.resize_handle_active = None

        # 회전
        self.is_rotating = False
        self.rotate_start_angle = 0

        # 핸들 생성
        self._create_handles()

    def _create_handles(self):
        """크기 조절 및 회전 핸들 생성"""
        # 4개 모서리 핸들
        positions = ['tl', 'tr', 'bl', 'br']
        for pos in positions:
            handle = ResizeHandle(pos, self)
            handle.setVisible(False)
            self.resize_handles[pos] = handle

        # 회전 핸들
        self.rotate_handle = RotateHandle(self)
        self.rotate_handle.setVisible(False)

    def boundingRect(self):
        """바운딩 박스 - 서브클래스에서 구현"""
        return QRectF(0, 0, 100, 100)

    def paint(self, painter, option, widget):
        """페인팅 - 서브클래스에서 구현"""
        pass

    def itemChange(self, change, value):
        """아이템 변경 감지"""
        if change == QGraphicsItem.GraphicsItemChange.ItemSelectedChange:
            # 선택 상태 변경
            is_selected = value
            self._update_handles_visibility(is_selected)
            self.signals.selection_changed.emit(is_selected)

        elif change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            # 위치 변경
            if not self.layer_locked:
                self.signals.transform_changed.emit()

        return super().itemChange(change, value)

    def _update_handles_visibility(self, visible):
        """핸들 가시성 업데이트"""
        if not self.layer_locked:
            for handle in self.resize_handles.values():
                handle.setVisible(visible)
            if self.rotate_handle:
                self.rotate_handle.setVisible(visible)
        else:
            for handle in self.resize_handles.values():
                handle.setVisible(False)
            if self.rotate_handle:
                self.rotate_handle.setVisible(False)

    def _update_handle_positions(self):
        """핸들 위치 업데이트 - 서브클래스에서 구현"""
        rect = self.boundingRect()

        # 모서리 핸들
        self.resize_handles['tl'].setPos(rect.topLeft())
        self.resize_handles['tr'].setPos(rect.topRight())
        self.resize_handles['bl'].setPos(rect.bottomLeft())
        self.resize_handles['br'].setPos(rect.bottomRight())

        # 회전 핸들 (상단 중앙 위 20px)
        if self.rotate_handle:
            top_center = QPointF(rect.center().x(), rect.top())
            self.rotate_handle.setPos(top_center.x(), top_center.y() - 20)

    def mousePressEvent(self, event):
        """마우스 클릭 - 크기 조절/회전 시작"""
        if self.layer_locked:
            event.ignore()
            return

        # 핸들 클릭 체크 - 씬에서 클릭된 모든 아이템 확인
        clicked_item = self.scene().itemAt(event.scenePos(), self.scene().views()[0].transform())

        # 회전 핸들 클릭
        if clicked_item == self.rotate_handle:
            self.is_rotating = True
            center = self.boundingRect().center()
            self.rotate_start_angle = self._calculate_angle(center, event.pos())
            # 회전 중에는 드래그 비활성화
            self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
            event.accept()
            return

        # 크기 조절 핸들 클릭
        for pos, handle in self.resize_handles.items():
            if clicked_item == handle:
                self.is_resizing = True
                self.resize_handle_active = pos
                self.resize_start_rect = self.boundingRect()
                self._resize_start_item_pos = self.pos()  # 초기 위치 저장
                self.resize_start_pos = event.pos()
                # 리사이징 중에는 드래그 비활성화
                self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
                event.accept()
                return

        # 핸들이 아닌 경우 기본 드래그 동작 활성화
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """마우스 이동 - 크기 조절/회전 처리"""
        if self.is_rotating:
            self._handle_rotation(event.pos())
            event.accept()
            return

        if self.is_resizing:
            self._handle_resize(event.pos())
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """마우스 릴리즈 - 크기 조절/회전 종료"""
        if self.is_rotating or self.is_resizing:
            self.is_rotating = False
            self.is_resizing = False
            self.resize_handle_active = None

            # 크기 조절 정리 (서브클래스에서 사용)
            self._cleanup_resize()

            # 드래그 다시 활성화 (잠금 상태가 아닌 경우)
            if not self.layer_locked:
                self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)

            self.signals.transform_changed.emit()
            event.accept()
            return

        super().mouseReleaseEvent(event)

    def _cleanup_resize(self):
        """크기 조절 후 정리 - 서브클래스에서 오버라이드 가능"""
        # 텍스트 레이어에서 사용하는 임시 속성 정리
        if hasattr(self, 'resize_start_font_size'):
            delattr(self, 'resize_start_font_size')
        # 리사이즈 시작 위치 정리
        if hasattr(self, '_resize_start_item_pos'):
            delattr(self, '_resize_start_item_pos')
        # 화면 좌표 정리
        if hasattr(self, '_resize_start_screen_pos'):
            delattr(self, '_resize_start_screen_pos')

    def _handle_rotation(self, pos):
        """회전 처리 - 서브클래스에서 구현 가능"""
        center = self.boundingRect().center()
        current_angle = self._calculate_angle(center, pos)
        delta_angle = current_angle - self.rotate_start_angle

        self.setRotation(self.rotation() + delta_angle)
        self.rotate_start_angle = current_angle

    # ============================================================
    # 핸들에서 직접 호출하는 메서드들 (이미지 밖에서도 동작)
    # ============================================================

    def _start_resize_from_handle(self, position, event):
        """핸들에서 리사이즈 시작"""
        if self.layer_locked:
            return
        self.is_resizing = True
        self.resize_handle_active = position
        self.resize_start_rect = self.boundingRect()
        self._resize_start_item_pos = self.pos()  # 초기 위치 저장
        # 화면 절대 좌표 저장 (어떤 변환도 거치지 않음)
        self._resize_start_screen_pos = event.screenPos()
        # 리사이징 중에는 드래그 비활성화
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)

    def _handle_resize_from_handle(self, event):
        """핸들에서 리사이즈 처리"""
        if not self.is_resizing:
            return
        # 화면 절대 좌표로 delta 계산 (어떤 좌표 변환도 없음)
        current_screen = event.screenPos()
        dx = current_screen.x() - self._resize_start_screen_pos.x()
        dy = current_screen.y() - self._resize_start_screen_pos.y()
        from PySide6.QtCore import QPointF
        delta = QPointF(dx, dy)
        self._handle_resize_with_delta(delta)

    def _end_resize_from_handle(self, event):
        """핸들에서 리사이즈 종료"""
        if self.is_resizing:
            self.is_resizing = False
            self.resize_handle_active = None
            self._cleanup_resize()
            # 드래그 다시 활성화 (잠금 상태가 아닌 경우)
            if not self.layer_locked:
                self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
            self.signals.transform_changed.emit()

    def _start_rotate_from_handle(self, event):
        """핸들에서 회전 시작"""
        if self.layer_locked:
            return
        self.is_rotating = True
        center = self.boundingRect().center()
        # scene 좌표를 레이어 로컬 좌표로 변환
        pos = self.mapFromScene(event.scenePos())
        self.rotate_start_angle = self._calculate_angle(center, pos)
        # 회전 중에는 드래그 비활성화
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)

    def _handle_rotate_from_handle(self, event):
        """핸들에서 회전 처리"""
        if not self.is_rotating:
            return
        # scene 좌표를 레이어 로컬 좌표로 변환
        pos = self.mapFromScene(event.scenePos())
        self._handle_rotation(pos)

    def _end_rotate_from_handle(self, event):
        """핸들에서 회전 종료"""
        if self.is_rotating:
            self.is_rotating = False
            # 드래그 다시 활성화 (잠금 상태가 아닌 경우)
            if not self.layer_locked:
                self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
            self.signals.transform_changed.emit()

    def _handle_resize(self, pos):
        """크기 조절 처리 - 서브클래스에서 구현"""
        pass

    def _handle_resize_with_delta(self, delta):
        """delta 기반 크기 조절 처리 - 서브클래스에서 구현"""
        pass

    def _calculate_angle(self, center, point):
        """중심점으로부터 각도 계산 (도)"""
        dx = point.x() - center.x()
        dy = point.y() - center.y()
        return math.degrees(math.atan2(dy, dx))

    def set_locked(self, locked: bool):
        """레이어 잠금 설정"""
        self.layer_locked = locked
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, not locked)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, not locked)
        self._update_handles_visibility(self.isSelected() and not locked)

    def set_visible(self, visible: bool):
        """레이어 가시성 설정"""
        self.layer_visible = visible
        self.setVisible(visible)

    def is_printing_mode(self) -> bool:
        """
        인쇄 모드 확인

        Returns:
            bool: 인쇄 모드이면 True
        """
        scene = self.scene()
        if scene and hasattr(scene, 'is_printing'):
            return scene.is_printing
        return False

    def get_layer_data(self):
        """레이어 데이터 딕셔너리 반환 - 저장/불러오기용"""
        return {
            'type': self.__class__.__name__,
            'name': self.layer_name,
            'x': self.pos().x(),
            'y': self.pos().y(),
            'rotation': self.rotation(),
            'locked': self.layer_locked,
            'visible': self.layer_visible,
        }
