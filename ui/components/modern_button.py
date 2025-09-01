"""
[EMOJI] [EMOJI] [EMOJI] [EMOJI]
"""

from PySide6.QtWidgets import QPushButton
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont


class ModernButton(QPushButton):
    """[EMOJI] [EMOJI] [EMOJI] [EMOJI]"""
    
    def __init__(self, text, icon_path=None, primary=False):
        super().__init__(text)
        self.primary = primary
        self.setFixedHeight(45)
        self.setFont(QFont("Segoe UI", 10, QFont.Weight.Medium))
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._apply_style()
    
    def _apply_style(self):
        """[EMOJI] [EMOJI] [EMOJI]"""
        if self.primary:
            style = """
                QPushButton {
                    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                               stop: 0 #4A90E2, stop: 1 #357ABD);
                    color: white;
                    border: none;
                    border-radius: 8px;
                    padding: 0 20px;
                    font-weight: 600;
                }
                QPushButton:hover {
                    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                               stop: 0 #5BA0F2, stop: 1 #4A90E2);
                }
                QPushButton:pressed {
                    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                               stop: 0 #357ABD, stop: 1 #2E6B9E);
                }
                QPushButton:disabled {
                    background: #CCCCCC;
                    color: #888888;
                }
            """
        else:
            style = """
                QPushButton {
                    background: #F8F9FA;
                    color: #495057;
                    border: 2px solid #E9ECEF;
                    border-radius: 8px;
                    padding: 0 20px;
                    font-weight: 500;
                }
                QPushButton:hover {
                    background: #E9ECEF;
                    border-color: #DEE2E6;
                }
                QPushButton:pressed {
                    background: #DEE2E6;
                    border-color: #CED4DA;
                }
                QPushButton:disabled {
                    background: #F8F9FA;
                    color: #ADB5BD;
                    border-color: #E9ECEF;
                }
            """
        
        self.setStyleSheet(style)
    
    def set_primary(self, primary: bool):
        """primary [EMOJI] [EMOJI]"""
        self.primary = primary
        self._apply_style()