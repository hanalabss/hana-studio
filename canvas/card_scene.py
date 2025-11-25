"""
CardScene - RTAI 카드 디자인을 위한 QGraphicsScene
"""

from PySide6.QtWidgets import QGraphicsScene, QGraphicsRectItem, QGraphicsLineItem
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QPen, QBrush, QColor

# RTAI 카드 규격 (CR80 표준)
CARD_WIDTH_MM = 85.6
CARD_HEIGHT_MM = 53.98

# DPI 설정
DISPLAY_DPI = 72  # 화면 표시용
PRINT_DPI = 600   # 인쇄용

# mm를 픽셀로 변환 (화면 표시용)
def mm_to_px(mm, dpi=DISPLAY_DPI):
    """밀리미터를 픽셀로 변환"""
    return (mm / 25.4) * dpi

# 카드 크기 (픽셀)
CARD_WIDTH_PX = mm_to_px(CARD_WIDTH_MM)
CARD_HEIGHT_PX = mm_to_px(CARD_HEIGHT_MM)

# 안전 여백 (각 면에서 2mm)
SAFE_MARGIN_MM = 2.0
SAFE_MARGIN_PX = mm_to_px(SAFE_MARGIN_MM)


class CardScene(QGraphicsScene):
    """RTAI 카드 디자인 씬"""

    def __init__(self):
        super().__init__()

        # 방향 설정 (기본: 세로)
        self.orientation = "portrait"
        # 세로: height가 큰 상태 (너비와 높이 교환)
        self.current_width = CARD_HEIGHT_PX  # 작은 값 (53.98mm)
        self.current_height = CARD_WIDTH_PX  # 큰 값 (85.6mm)

        # 인쇄 모드 플래그
        self.is_printing = False

        # 씬 크기 설정 (카드 크기 + 여유 공간)
        padding = 50
        self.setSceneRect(
            -padding,
            -padding,
            self.current_width + padding * 2,
            self.current_height + padding * 2
        )

        # 배경색 설정
        self.setBackgroundBrush(QBrush(QColor("#F5F5F5")))

        # 카드 요소 그룹 (나중에 재생성용)
        self.card_background = None  # 카드 배경 (인쇄 필요)
        self.guide_elements = []  # 가이드라인들 (인쇄 시 숨김)

        # 카드 요소들 추가
        self._setup_card_elements()

    def _setup_card_elements(self):
        """카드 배경, 안전 여백, 재단선 설정"""
        # 기존 카드 요소 제거
        if self.card_background:
            self.removeItem(self.card_background)
        for item in self.guide_elements:
            self.removeItem(item)
        self.guide_elements.clear()

        # 1. 카드 배경 (흰색) - 인쇄 시에도 필요
        self.card_background = QGraphicsRectItem(0, 0, self.current_width, self.current_height)
        self.card_background.setBrush(QBrush(QColor("#FFFFFF")))
        self.card_background.setPen(QPen(QColor("#CCCCCC"), 1))
        self.card_background.setZValue(-100)  # 가장 뒤에 배치
        self.addItem(self.card_background)

        # 2. 안전 여백 (점선) - 가이드라인
        safe_zone_rect = QGraphicsRectItem(
            SAFE_MARGIN_PX,
            SAFE_MARGIN_PX,
            self.current_width - SAFE_MARGIN_PX * 2,
            self.current_height - SAFE_MARGIN_PX * 2
        )
        pen = QPen(QColor("#4A90E2"), 1, Qt.PenStyle.DashLine)
        safe_zone_rect.setPen(pen)
        safe_zone_rect.setBrush(Qt.BrushStyle.NoBrush)
        safe_zone_rect.setZValue(-50)
        self.addItem(safe_zone_rect)
        self.guide_elements.append(safe_zone_rect)

        # 3. 재단선 (실선, 회색) - 가이드라인
        cut_line_rect = QGraphicsRectItem(0, 0, self.current_width, self.current_height)
        cut_line_rect.setPen(QPen(QColor("#999999"), 2, Qt.PenStyle.SolidLine))
        cut_line_rect.setBrush(Qt.BrushStyle.NoBrush)
        cut_line_rect.setZValue(-40)
        self.addItem(cut_line_rect)
        self.guide_elements.append(cut_line_rect)

        # 4. 중앙 십자선 (가이드) - 가이드라인
        center_x = self.current_width / 2
        center_y = self.current_height / 2

        # 수평선
        h_line = QGraphicsLineItem(0, center_y, self.current_width, center_y)
        h_line.setPen(QPen(QColor("#CCCCCC"), 1, Qt.PenStyle.DotLine))
        h_line.setZValue(-30)
        self.addItem(h_line)
        self.guide_elements.append(h_line)

        # 수직선
        v_line = QGraphicsLineItem(center_x, 0, center_x, self.current_height)
        v_line.setPen(QPen(QColor("#CCCCCC"), 1, Qt.PenStyle.DotLine))
        v_line.setZValue(-30)
        self.addItem(v_line)
        self.guide_elements.append(v_line)

    def get_card_rect(self):
        """카드 영역 반환"""
        return QRectF(0, 0, self.current_width, self.current_height)

    def get_safe_zone_rect(self):
        """안전 여백 영역 반환"""
        return QRectF(
            SAFE_MARGIN_PX,
            SAFE_MARGIN_PX,
            self.current_width - SAFE_MARGIN_PX * 2,
            self.current_height - SAFE_MARGIN_PX * 2
        )

    def set_orientation(self, orientation: str):
        """카드 방향 설정 (portrait/landscape)"""
        if self.orientation == orientation:
            return

        self.orientation = orientation

        # 방향에 따라 크기 설정
        if orientation == "landscape":
            # 가로: width가 큰 상태
            self.current_width = CARD_WIDTH_PX   # 큰 값 (85.6mm)
            self.current_height = CARD_HEIGHT_PX  # 작은 값 (53.98mm)
        else:
            # 세로: height가 큰 상태
            self.current_width = CARD_HEIGHT_PX  # 작은 값 (53.98mm)
            self.current_height = CARD_WIDTH_PX  # 큰 값 (85.6mm)

        # 씬 크기 재설정
        padding = 50
        self.setSceneRect(
            -padding,
            -padding,
            self.current_width + padding * 2,
            self.current_height + padding * 2
        )

        # 카드 요소 재생성
        self._setup_card_elements()

        print(f"[INFO] 캔버스 방향 변경: {orientation} ({self.current_width}x{self.current_height})")

    def get_orientation(self):
        """현재 카드 방향 반환"""
        return self.orientation

    def set_printing_mode(self, enabled: bool):
        """
        인쇄 모드 설정

        Args:
            enabled: True일 때 가이드라인 숨김, False일 때 가이드라인 표시
        """
        self.is_printing = enabled

        # 가이드라인 표시/숨김
        for guide in self.guide_elements:
            guide.setVisible(not enabled)
