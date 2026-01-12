"""
TextLayer - 텍스트 레이어
드래그, 크기 조절, 회전, 인라인 편집 지원
"""

from PySide6.QtWidgets import QGraphicsItem, QInputDialog
from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QFont, QColor, QPen, QPainter, QFontMetrics

from .base_layer import BaseLayer


class TextLayer(BaseLayer):
    """텍스트 레이어"""

    def __init__(self, text: str = "텍스트"):
        super().__init__()

        self.layer_name = "Text"

        # 텍스트 속성
        self.text_content = text
        self.text_font = QFont("맑은 고딕", 24)
        self.text_color = QColor("#000000")
        self.text_alignment = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop

        # 바운딩 박스 계산
        self._update_bounding_rect()

        # 핸들 위치 업데이트
        self._update_handle_positions()

    def _update_bounding_rect(self):
        """바운딩 박스 계산"""
        metrics = QFontMetrics(self.text_font)
        self.text_rect = metrics.boundingRect(self.text_content)
        # 여백 추가
        self.text_rect.adjust(-5, -5, 5, 5)

    def boundingRect(self):
        """바운딩 박스"""
        return QRectF(self.text_rect)

    def paint(self, painter, option, widget):
        """텍스트 그리기"""
        # 배경 및 테두리 (선택 시, 인쇄 모드가 아닐 때만)
        if self.isSelected() and not self.is_printing_mode():
            painter.fillRect(self.boundingRect(), QColor(255, 255, 255, 50))
            painter.setPen(QPen(QColor("#4A90E2"), 2, Qt.PenStyle.DashLine))
            painter.drawRect(self.boundingRect())

        # 텍스트 그리기
        painter.setFont(self.text_font)
        painter.setPen(QPen(self.text_color))
        painter.drawText(self.text_rect, self.text_alignment, self.text_content)

    def mouseDoubleClickEvent(self, event):
        """더블클릭 - 텍스트 편집 다이얼로그"""
        if not self.layer_locked:
            text, ok = QInputDialog.getText(
                None,
                "텍스트 편집",
                "텍스트를 입력하세요:",
                text=self.text_content
            )

            if ok and text:
                self.set_text(text)

            event.accept()
        else:
            super().mouseDoubleClickEvent(event)

    def set_text(self, text: str):
        """텍스트 설정"""
        self.prepareGeometryChange()
        self.text_content = text
        self._update_bounding_rect()
        self._update_handle_positions()
        self.update()
        self.signals.transform_changed.emit()

    def get_text(self):
        """텍스트 반환"""
        return self.text_content

    def set_font(self, font: QFont):
        """폰트 설정"""
        self.prepareGeometryChange()
        self.text_font = font
        self._update_bounding_rect()
        self._update_handle_positions()
        self.update()

    def get_font(self):
        """폰트 반환"""
        return self.text_font

    def set_color(self, color: QColor):
        """텍스트 색상 설정"""
        self.text_color = color
        self.update()

    def get_color(self):
        """텍스트 색상 반환"""
        return self.text_color

    def set_font_size(self, size: int):
        """폰트 크기 설정"""
        self.prepareGeometryChange()
        self.text_font.setPointSize(size)
        self._update_bounding_rect()
        self._update_handle_positions()
        self.update()
        self.signals.transform_changed.emit()

    def get_font_size(self):
        """폰트 크기 반환"""
        return self.text_font.pointSize()

    def set_bold(self, bold: bool):
        """굵기 설정"""
        self.text_font.setBold(bold)
        self._update_bounding_rect()
        self._update_handle_positions()
        self.update()

    def get_bold(self):
        """굵기 반환"""
        return self.text_font.bold()

    def set_italic(self, italic: bool):
        """이탤릭 설정"""
        self.text_font.setItalic(italic)
        self._update_bounding_rect()
        self._update_handle_positions()
        self.update()

    def get_italic(self):
        """이탤릭 반환"""
        return self.text_font.italic()

    def set_underline(self, underline: bool):
        """밑줄 설정"""
        self.text_font.setUnderline(underline)
        self.update()

    def get_underline(self):
        """밑줄 반환"""
        return self.text_font.underline()

    def set_alignment(self, alignment: Qt.AlignmentFlag):
        """정렬 설정"""
        self.text_alignment = alignment
        self.update()

    def get_alignment(self):
        """정렬 반환"""
        return self.text_alignment

    def _handle_resize(self, pos):
        """크기 조절 - 폰트 크기 조정 (부드러운 업데이트)"""
        if not self.resize_handle_active or not self.resize_start_rect:
            return

        # 핸들에서 시작한 경우 _handle_resize_with_delta 사용
        if hasattr(self, '_resize_start_screen_pos'):
            return

        delta = pos - self.resize_start_pos
        self._apply_font_resize(delta)

    def _handle_resize_with_delta(self, delta):
        """delta 기반 크기 조절 처리 (핸들에서 호출)"""
        if not self.resize_handle_active or not self.resize_start_rect:
            return

        self._apply_font_resize(delta)

    def _apply_font_resize(self, delta):
        """실제 폰트 크기 조절 로직"""
        handle = self.resize_handle_active

        # 초기 폰트 크기 저장 (처음 한 번만)
        if not hasattr(self, 'resize_start_font_size'):
            self.resize_start_font_size = self.text_font.pointSize()

        # 각 핸들의 "바깥쪽" 방향으로 드래그하면 커지도록 계산
        # br: +x, +y → 커짐 / bl: -x, +y → 커짐 / tr: +x, -y → 커짐 / tl: -x, -y → 커짐
        outward_x = 0
        outward_y = 0
        if 'r' in handle:
            outward_x = delta.x()
        elif 'l' in handle:
            outward_x = -delta.x()
        if 'b' in handle:
            outward_y = delta.y()
        elif 't' in handle:
            outward_y = -delta.y()

        scale_factor = 1.0 + (outward_x + outward_y) / 200.0

        # 폰트 크기 변경 (시작 크기 기준으로 계산하여 부드럽게)
        new_size = max(8, min(200, int(self.resize_start_font_size * scale_factor)))
        current_size = self.text_font.pointSize()

        if new_size != current_size:
            # 변경 전 바운딩 박스와 앵커 포인트 계산
            old_rect = self.boundingRect()

            # 각 핸들의 반대쪽(고정될 곳) 좌표 저장
            if 'l' in handle and 't' in handle:  # tl → 우하단 고정
                anchor = old_rect.bottomRight()
            elif 'r' in handle and 't' in handle:  # tr → 좌하단 고정
                anchor = old_rect.bottomLeft()
            elif 'l' in handle and 'b' in handle:  # bl → 우상단 고정
                anchor = old_rect.topRight()
            else:  # br → 좌상단 고정
                anchor = old_rect.topLeft()

            # 폰트 크기 변경
            self.set_font_size(new_size)

            # 변경 후 바운딩 박스
            new_rect = self.boundingRect()

            # 새로운 앵커 위치 계산
            if 'l' in handle and 't' in handle:
                new_anchor = new_rect.bottomRight()
            elif 'r' in handle and 't' in handle:
                new_anchor = new_rect.bottomLeft()
            elif 'l' in handle and 'b' in handle:
                new_anchor = new_rect.topRight()
            else:
                new_anchor = new_rect.topLeft()

            # 앵커가 제자리에 있도록 위치 보정
            dx = anchor.x() - new_anchor.x()
            dy = anchor.y() - new_anchor.y()
            if dx != 0 or dy != 0:
                self.setPos(self.pos().x() + dx, self.pos().y() + dy)

    def get_layer_data(self):
        """레이어 데이터"""
        data = super().get_layer_data()
        data.update({
            'text': self.text_content,
            'font_family': self.text_font.family(),
            'font_size': self.text_font.pointSize(),
            'bold': self.text_font.bold(),
            'italic': self.text_font.italic(),
            'underline': self.text_font.underline(),
            'color': self.text_color.name(),
            'alignment': int(self.text_alignment),
        })
        return data

    @classmethod
    def from_data(cls, data: dict):
        """데이터로부터 레이어 복원"""
        # 텍스트 레이어 생성
        layer = cls(data.get('text', '텍스트'))

        # 위치 및 회전
        layer.setPos(data.get('x', 0), data.get('y', 0))
        layer.setRotation(data.get('rotation', 0))

        # 폰트 설정
        from PySide6.QtGui import QFont, QColor
        font = QFont()
        font.setFamily(data.get('font_family', '맑은 고딕'))
        font.setPointSize(data.get('font_size', 24))
        font.setBold(data.get('bold', False))
        font.setItalic(data.get('italic', False))
        font.setUnderline(data.get('underline', False))
        layer.text_font = font

        # 색상
        layer.text_color = QColor(data.get('color', '#000000'))

        # 정렬
        from PySide6.QtCore import Qt
        layer.text_alignment = Qt.AlignmentFlag(data.get('alignment', Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop))

        # 경계 업데이트
        layer._update_bounding_rect()
        layer._update_handle_positions()

        return layer
