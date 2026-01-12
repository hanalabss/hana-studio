"""
ImageLayer - 이미지 레이어
드래그, 크기 조절, 회전 지원
"""

from PySide6.QtWidgets import QGraphicsItem
from PySide6.QtCore import QRectF, Qt, QPointF
from PySide6.QtGui import QPixmap, QPainter, QPen, QColor, QTransform

from .base_layer import BaseLayer


class ImageLayer(BaseLayer):
    """이미지 레이어"""

    def __init__(self, image_path: str = None, pixmap: QPixmap = None):
        super().__init__()

        self.layer_name = "Image"
        self.image_path = image_path

        # 이미지 로드
        if pixmap:
            self.original_pixmap = pixmap
        elif image_path:
            self.original_pixmap = QPixmap(image_path)
        else:
            # 기본 이미지 (빈 회색 박스)
            self.original_pixmap = QPixmap(200, 200)
            self.original_pixmap.fill(QColor("#CCCCCC"))

        # 변환된 픽스맵 (반전 적용)
        self.pixmap = self.original_pixmap.copy()

        # 마스킹 관련 픽스맵 (배경제거 시 생성됨)
        self.mask_pixmap = None           # 마스크 이미지
        self.final_result_pixmap = None   # 최종 합성 결과 (비교용)

        # 이미지 크기
        self.image_width = self.pixmap.width()
        self.image_height = self.pixmap.height()

        # 비율 유지 플래그
        self.keep_aspect_ratio = True

        # 반전 플래그
        self.flip_horizontal_enabled = False
        self.flip_vertical_enabled = False

        # 투명도 (0.0 ~ 1.0)
        self.opacity_value = 1.0

        # 핸들 위치 업데이트
        self._update_handle_positions()

    def boundingRect(self):
        """바운딩 박스"""
        return QRectF(0, 0, self.image_width, self.image_height)

    def paint(self, painter, option, widget):
        """이미지 그리기"""
        # 투명도 설정
        painter.setOpacity(self.opacity_value)

        # 이미지 그리기
        painter.drawPixmap(0, 0, self.image_width, self.image_height, self.pixmap)

        # 투명도 복원
        painter.setOpacity(1.0)

        # 선택 시 테두리 (인쇄 모드가 아닐 때만)
        if self.isSelected() and not self.is_printing_mode():
            painter.setPen(QPen(QColor("#4A90E2"), 2, Qt.PenStyle.DashLine))
            painter.drawRect(self.boundingRect())

    def _handle_resize(self, pos):
        """크기 조절 처리 - 부드러운 업데이트"""
        if not self.resize_handle_active or not self.resize_start_rect:
            return

        # 핸들에서 시작한 경우 _handle_resize_with_delta 사용
        if hasattr(self, '_resize_start_screen_pos'):
            return

        delta = pos - self.resize_start_pos
        new_width = self.resize_start_rect.width()
        new_height = self.resize_start_rect.height()
        dx = 0  # x 위치 변화량
        dy = 0  # y 위치 변화량

        handle = self.resize_handle_active

        # 각 핸들에 따른 크기 조정
        if 'r' in handle:  # 오른쪽 (tr, br, r)
            new_width = max(50, self.resize_start_rect.width() + delta.x())
        elif 'l' in handle:  # 왼쪽 (tl, bl, l)
            new_width = max(50, self.resize_start_rect.width() - delta.x())
            dx = delta.x()  # 왼쪽으로 드래그하면 위치도 이동

        if 'b' in handle:  # 하단 (bl, br, b)
            new_height = max(50, self.resize_start_rect.height() + delta.y())
        elif 't' in handle:  # 상단 (tl, tr, t)
            new_height = max(50, self.resize_start_rect.height() - delta.y())
            dy = delta.y()  # 위로 드래그하면 위치도 이동

        # 비율 유지
        if self.keep_aspect_ratio:
            aspect_ratio = self.pixmap.width() / self.pixmap.height()
            if abs(new_width - self.resize_start_rect.width()) > abs(new_height - self.resize_start_rect.height()):
                new_height = new_width / aspect_ratio
                # 상단 핸들이면 높이 변화에 따라 dy 재계산
                if 't' in handle:
                    dy = self.resize_start_rect.height() - new_height
            else:
                new_width = new_height * aspect_ratio
                # 왼쪽 핸들이면 너비 변화에 따라 dx 재계산
                if 'l' in handle:
                    dx = self.resize_start_rect.width() - new_width

        # 크기가 실제로 변경된 경우에만 업데이트
        if abs(new_width - self.image_width) > 0.5 or abs(new_height - self.image_height) > 0.5:
            self.prepareGeometryChange()
            self.image_width = new_width
            self.image_height = new_height

            # 좌측/상단 핸들인 경우 위치 이동
            if dx != 0 or dy != 0:
                if not hasattr(self, '_resize_start_item_pos'):
                    self._resize_start_item_pos = self.pos()
                self.setPos(self._resize_start_item_pos.x() + dx,
                           self._resize_start_item_pos.y() + dy)

            self._update_handle_positions()
            self.update()

    def _handle_resize_with_delta(self, delta):
        """delta 기반 크기 조절 처리 (핸들에서 호출)"""
        if not self.resize_handle_active or not self.resize_start_rect:
            return

        start_width = self.resize_start_rect.width()
        start_height = self.resize_start_rect.height()
        new_width = start_width
        new_height = start_height

        handle = self.resize_handle_active

        # 각 핸들에 따른 크기 조정
        if 'r' in handle:  # 오른쪽 (tr, br, r)
            new_width = max(50, start_width + delta.x())
        elif 'l' in handle:  # 왼쪽 (tl, bl, l)
            new_width = max(50, start_width - delta.x())

        if 'b' in handle:  # 하단 (bl, br, b)
            new_height = max(50, start_height + delta.y())
        elif 't' in handle:  # 상단 (tl, tr, t)
            new_height = max(50, start_height - delta.y())

        # 비율 유지 - 핸들 위치에 따라 기준 축 고정 (튀는 현상 방지)
        if self.keep_aspect_ratio:
            aspect_ratio = self.pixmap.width() / self.pixmap.height()

            if handle in ['t', 'b']:
                # 상/하 핸들: height 기준으로 width 계산
                new_width = new_height * aspect_ratio
            else:
                # 그 외 (l, r, tl, tr, bl, br): width 기준으로 height 계산
                new_height = new_width / aspect_ratio

        # 실제 크기 변화량으로 위치 이동량 계산 (delta가 아닌 크기 변화 기준)
        dx = 0
        dy = 0
        if 'l' in handle:
            dx = start_width - new_width  # 크기가 줄면 dx > 0 (오른쪽으로 이동)
        if 't' in handle:
            dy = start_height - new_height  # 크기가 줄면 dy > 0 (아래로 이동)

        # 크기가 실제로 변경된 경우에만 업데이트
        if abs(new_width - self.image_width) > 0.5 or abs(new_height - self.image_height) > 0.5:
            self.prepareGeometryChange()
            self.image_width = new_width
            self.image_height = new_height

            # 좌측/상단 핸들인 경우 위치 이동
            if dx != 0 or dy != 0:
                self.setPos(self._resize_start_item_pos.x() + dx,
                           self._resize_start_item_pos.y() + dy)

            self._update_handle_positions()
            self.update()

    def set_size(self, width: float, height: float):
        """크기 설정"""
        self.prepareGeometryChange()
        self.image_width = max(10, width)
        self.image_height = max(10, height)
        self._update_handle_positions()
        self.update()

    def set_image(self, image_path: str = None, pixmap: QPixmap = None):
        """이미지 변경"""
        if pixmap:
            self.pixmap = pixmap
            self.image_path = None
        elif image_path:
            self.pixmap = QPixmap(image_path)
            self.image_path = image_path

        self.prepareGeometryChange()
        # 이미지 크기에 맞게 조정
        self.image_width = self.pixmap.width()
        self.image_height = self.pixmap.height()
        self._update_handle_positions()
        self.update()

    def flip_horizontal(self):
        """좌우 반전"""
        self.flip_horizontal_enabled = not self.flip_horizontal_enabled
        self._apply_transformations()
        self.signals.transform_changed.emit()

    def flip_vertical(self):
        """상하 반전"""
        self.flip_vertical_enabled = not self.flip_vertical_enabled
        self._apply_transformations()
        self.signals.transform_changed.emit()

    def _apply_transformations(self):
        """반전 변환 적용"""
        transform = QTransform()

        # 좌우 반전
        if self.flip_horizontal_enabled:
            transform.scale(-1, 1)
            transform.translate(-self.original_pixmap.width(), 0)

        # 상하 반전
        if self.flip_vertical_enabled:
            transform.scale(1, -1)
            transform.translate(0, -self.original_pixmap.height())

        # 변환 적용
        self.pixmap = self.original_pixmap.transformed(transform, Qt.TransformationMode.SmoothTransformation)
        self.update()

    def set_opacity(self, opacity: float):
        """투명도 설정 (0.0 ~ 1.0)"""
        self.opacity_value = max(0.0, min(1.0, opacity))
        self.update()

    def get_opacity(self):
        """투명도 반환"""
        return self.opacity_value

    def get_layer_data(self):
        """레이어 데이터"""
        data = super().get_layer_data()
        data.update({
            'image_path': self.image_path,
            'width': self.image_width,
            'height': self.image_height,
            'keep_aspect_ratio': self.keep_aspect_ratio,
            'flip_horizontal': self.flip_horizontal_enabled,
            'flip_vertical': self.flip_vertical_enabled,
            'opacity': self.opacity_value,
        })
        return data

    @classmethod
    def from_data(cls, data: dict):
        """데이터로부터 레이어 복원"""
        # 이미지 로드
        layer = cls(image_path=data.get('image_path'))

        # 위치 및 회전
        layer.setPos(data.get('x', 0), data.get('y', 0))
        layer.setRotation(data.get('rotation', 0))

        # 크기
        layer.set_size(data.get('width', layer.image_width), data.get('height', layer.image_height))

        # 속성
        layer.keep_aspect_ratio = data.get('keep_aspect_ratio', True)
        layer.opacity_value = data.get('opacity', 1.0)

        # 반전 (플래그만 설정하고 _apply_transformations 호출)
        layer.flip_horizontal_enabled = data.get('flip_horizontal', False)
        layer.flip_vertical_enabled = data.get('flip_vertical', False)
        if layer.flip_horizontal_enabled or layer.flip_vertical_enabled:
            layer._apply_transformations()

        return layer
