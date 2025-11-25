"""
PrintRenderer - 600DPI 고해상도 렌더링
캔버스를 인쇄용 고해상도 이미지로 변환
"""

from PySide6.QtCore import QRectF, QSizeF, Qt
from PySide6.QtGui import QImage, QPainter, QTransform
from PySide6.QtWidgets import QGraphicsScene

from canvas.card_scene import CARD_WIDTH_MM, CARD_HEIGHT_MM, DISPLAY_DPI, PRINT_DPI


class PrintRenderer:
    """600DPI 인쇄 렌더러"""

    def __init__(self, scene: QGraphicsScene):
        self.scene = scene

    def render_to_image(self, dpi: int = PRINT_DPI) -> QImage:
        """
        씬을 고해상도 이미지로 렌더링

        Args:
            dpi: 인쇄 해상도 (기본 600DPI)

        Returns:
            QImage: 렌더링된 고해상도 이미지
        """
        # DPI에 따른 스케일 계산
        scale_factor = dpi / DISPLAY_DPI

        # 카드 영역 가져오기
        card_rect = self.scene.get_card_rect()

        # 씬의 orientation 확인
        scene_orientation = self.scene.get_orientation()

        # 현재 캔버스 크기에 맞춰 렌더링 (고해상도)
        render_width_px = int(card_rect.width() * scale_factor)
        render_height_px = int(card_rect.height() * scale_factor)

        # 이미지 생성 (캔버스 크기대로)
        image = QImage(
            render_width_px,
            render_height_px,
            QImage.Format.Format_ARGB32
        )
        image.fill(Qt.GlobalColor.white)

        # 페인터 생성
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        # 스케일 적용
        painter.scale(scale_factor, scale_factor)

        # 인쇄 모드 활성화 (가이드라인 및 선택 테두리 숨김)
        self.scene.set_printing_mode(True)

        try:
            # 씬 렌더링
            self.scene.render(
                painter,
                target=QRectF(0, 0, card_rect.width(), card_rect.height()),
                source=card_rect
            )
        finally:
            # 인쇄 모드 비활성화 (예외 발생 시에도 복원)
            self.scene.set_printing_mode(False)

        painter.end()

        print(f"[RENDER] 렌더링 완료: {render_width_px}x{render_height_px} ({dpi} DPI, orientation: {scene_orientation})")
        return image

    def save_to_file(self, file_path: str, dpi: int = PRINT_DPI) -> bool:
        """
        씬을 파일로 저장

        Args:
            file_path: 저장 경로
            dpi: 인쇄 해상도

        Returns:
            bool: 성공 여부
        """
        image = self.render_to_image(dpi)

        # DPI 메타데이터 설정
        dpm = int(dpi / 25.4 * 1000)  # dots per meter
        image.setDotsPerMeterX(dpm)
        image.setDotsPerMeterY(dpm)

        success = image.save(file_path, "PNG")

        if success:
            print(f"[OK] 이미지 저장: {file_path}")
        else:
            print(f"[ERROR] 이미지 저장 실패: {file_path}")

        return success


def render_scene_to_image(scene: QGraphicsScene, dpi: int = PRINT_DPI) -> QImage:
    """
    씬을 이미지로 렌더링 (간편 함수)

    Args:
        scene: QGraphicsScene
        dpi: 인쇄 해상도

    Returns:
        QImage: 렌더링된 이미지
    """
    renderer = PrintRenderer(scene)
    return renderer.render_to_image(dpi)
