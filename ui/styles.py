"""
UI 스타일 정의
"""

from PySide6.QtGui import QPalette, QColor


def get_app_style() -> str:
    """애플리케이션 전체 스타일 반환"""
    return """
        QMainWindow {
            background-color: #F8F9FA;
        }
        QWidget {
            background-color: #F8F9FA;
            color: #212529;
            font-family: 'Segoe UI', Arial, sans-serif;
        }
        QSplitter {
            background-color: #F8F9FA;
        }
        QSplitter::handle {
            background-color: #DEE2E6;
        }
        QGroupBox {
            font-weight: 600;
            font-size: 12px;
            color: #495057;
            border: 2px solid #DEE2E6;
            border-radius: 10px;
            margin-top: 10px;
            padding-top: 10px;
            background-color: #FFFFFF;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 15px;
            padding: 0 10px;
            background-color: #FFFFFF;
            color: #495057;
        }
        QProgressBar {
            border: none;
            border-radius: 6px;
            background-color: #E9ECEF;
            height: 12px;
        }
        QProgressBar::chunk {
            background-color: #4A90E2;
            border-radius: 6px;
        }
        QTextEdit {
            background-color: #FFFFFF;
            border: 2px solid #E9ECEF;
            border-radius: 8px;
            padding: 10px;
            font-family: 'Consolas', 'Courier New', monospace;
            font-size: 11px;
            color: #212529;
        }
        QLabel {
            background-color: transparent;
            color: #212529;
        }
    """


def get_light_palette() -> QPalette:
    """라이트 테마 팔레트 반환"""
    light_palette = QPalette()
    light_palette.setColor(QPalette.ColorRole.Window, QColor("#F8F9FA"))
    light_palette.setColor(QPalette.ColorRole.WindowText, QColor("#212529"))
    light_palette.setColor(QPalette.ColorRole.Base, QColor("#FFFFFF"))
    light_palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#E9ECEF"))
    light_palette.setColor(QPalette.ColorRole.ToolTipBase, QColor("#FFFFFF"))
    light_palette.setColor(QPalette.ColorRole.ToolTipText, QColor("#212529"))
    light_palette.setColor(QPalette.ColorRole.Text, QColor("#212529"))
    light_palette.setColor(QPalette.ColorRole.Button, QColor("#F8F9FA"))
    light_palette.setColor(QPalette.ColorRole.ButtonText, QColor("#212529"))
    light_palette.setColor(QPalette.ColorRole.BrightText, QColor("#DC3545"))
    light_palette.setColor(QPalette.ColorRole.Link, QColor("#4A90E2"))
    light_palette.setColor(QPalette.ColorRole.Highlight, QColor("#4A90E2"))
    light_palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#FFFFFF"))
    return light_palette


def get_header_style() -> str:
    """헤더 프레임 스타일"""
    return """
        QFrame {
            background-color: #FFFFFF;
            border: none;
            border-bottom: 3px solid #4A90E2;
            border-radius: 0;
        }
    """


def get_status_bar_style() -> str:
    """상태바 스타일"""
    return """
        QFrame {
            background-color: #FFFFFF;
            border: none;
            border-top: 1px solid #DEE2E6;
        }
    """


def get_radio_button_style() -> str:
    """라디오 버튼 스타일"""
    return """
        QRadioButton {
            font-size: 11px;
            font-weight: 500;
            color: #495057;
            padding: 5px;
            spacing: 8px;
        }
        QRadioButton::indicator {
            width: 14px;
            height: 14px;
        }
        QRadioButton::indicator:unchecked {
            border: 2px solid #6C757D;
            border-radius: 7px;
            background-color: white;
        }
        QRadioButton::indicator:checked {
            border: 2px solid #4A90E2;
            border-radius: 7px;
            background-color: #4A90E2;
        }
        QRadioButton::indicator:checked:hover {
            background-color: #357ABD;
            border-color: #357ABD;
        }
        QRadioButton:hover {
            color: #4A90E2;
        }
    """