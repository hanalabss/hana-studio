"""
ui/splash_screen.py
Hana Studio 스플래시 스크린 - 초간단 버전 (아이콘 + 상태 텍스트 + 프로그레스바)
"""

from PySide6.QtWidgets import QSplashScreen, QApplication
from PySide6.QtCore import Qt, QTimer, QRect
from PySide6.QtGui import QPixmap, QPainter, QColor, QFont, QBrush, QPen
import sys
from pathlib import Path


class HanaStudioSplash(QSplashScreen):
    """초간단 스플래시 스크린 - 아이콘과 프로그레스바"""
    
    def __init__(self):
        # 스플래시 이미지 생성
        pixmap = self.create_splash_image()
        super().__init__(pixmap, Qt.WindowStaysOnTopHint)
        
        self.progress = 0
        self.message = ""
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        
        # 투명 배경 지원 활성화
        self.setAttribute(Qt.WA_TranslucentBackground)
        
    def create_splash_image(self):
        """초간단 스플래시 이미지 생성 (투명 배경)"""
        # 300x200 크기의 이미지
        pixmap = QPixmap(300, 200)
        pixmap.fill(Qt.transparent)  # 투명 배경
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 아이콘 파일 체크 및 로드
        icon_path = Path("hana.ico")
        if icon_path.exists():
            # 아이콘 파일이 있으면 사용
            icon_pixmap = QPixmap(str(icon_path))
            # 아이콘 크기 조정 (80x80)
            icon_pixmap = icon_pixmap.scaled(80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            # 중앙에 아이콘 그리기
            icon_x = (300 - 80) // 2
            icon_y = 40
            painter.drawPixmap(icon_x, icon_y, icon_pixmap)
        else:
            # 아이콘 파일이 없으면 간단한 H 로고
            painter.setPen(QPen(QColor("#4A90E2"), 2))
            painter.setBrush(QBrush(QColor("#4A90E2")))
            painter.drawEllipse(110, 40, 80, 80)
            
            painter.setPen(QPen(QColor("#FFFFFF"), 5))
            painter.setFont(QFont("Arial", 36, QFont.Bold))
            painter.drawText(110, 40, 80, 80, Qt.AlignCenter, "H")
        
        painter.end()
        return pixmap
        
    def update_status(self, message: str, progress: int):
        """로딩 상태 업데이트"""
        self.progress = progress
        self.message = message
        self.repaint()
        QApplication.processEvents()
        
    def drawContents(self, painter):
        """상태 텍스트와 프로그레스바 그리기"""
        super().drawContents(painter)
        
        # 상태 메시지 (프로그레스바 위에)
        if self.message:
            painter.setPen(QColor("#495057"))
            painter.setFont(QFont("Segoe UI", 9))
            message_rect = QRect(0, 135, 300, 20)
            painter.drawText(message_rect, Qt.AlignCenter, self.message)
        
        # 프로그레스바 배경
        progress_bg_rect = QRect(50, 160, 200, 6)
        painter.fillRect(progress_bg_rect, QColor("#E9ECEF"))
        
        # 프로그레스바
        if self.progress > 0:
            progress_rect = QRect(50, 160, int(200 * self.progress / 100), 6)
            painter.fillRect(progress_rect, QColor("#4A90E2"))


class SimpleSplash(QSplashScreen):
    """간단한 대체 스플래시 스크린"""
    
    def __init__(self):
        pixmap = QPixmap(250, 150)
        pixmap.fill(Qt.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # H 로고만
        painter.setPen(QPen(QColor("#4A90E2"), 2))
        painter.setFont(QFont("Arial", 48, QFont.Bold))
        painter.drawText(0, 0, 250, 150, Qt.AlignCenter, "H")
        
        painter.end()
        
        super().__init__(pixmap, Qt.WindowStaysOnTopHint)
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
    def update_status(self, message: str, progress: int = 0):
        """상태 업데이트 (아무것도 안 함)"""
        QApplication.processEvents()