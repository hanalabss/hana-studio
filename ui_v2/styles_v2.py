"""
Canva 스타일 CSS 정의
"""

CANVA_STYLE = """
/* 메인 윈도우 */
QMainWindow {
    background-color: #FFFFFF;
}

/* 패널 스타일 */
QWidget#left_panel, QWidget#right_panel {
    background-color: #F8F9FA;
    border: 1px solid #E9ECEF;
}

/* 그룹박스 */
QGroupBox {
    font-size: 13px;
    font-weight: bold;
    color: #495057;
    border: 1px solid #DEE2E6;
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 12px;
    background-color: #FFFFFF;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 5px;
    background-color: #FFFFFF;
}

/* 버튼 - Canva 스타일 */
QPushButton {
    background-color: #FFFFFF;
    color: #495057;
    border: 1px solid #DEE2E6;
    border-radius: 6px;
    padding: 10px 16px;
    font-size: 13px;
    font-weight: 500;
}

QPushButton:hover {
    background-color: #F8F9FA;
    border-color: #ADB5BD;
}

QPushButton:pressed {
    background-color: #E9ECEF;
}

/* Primary 버튼 */
QPushButton#primary_btn {
    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                               stop: 0 #4A90E2, stop: 1 #357ABD);
    color: white;
    border: none;
    font-weight: 600;
}

QPushButton#primary_btn:hover {
    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                               stop: 0 #5BA0F2, stop: 1 #4A90E2);
}

QPushButton#primary_btn:pressed {
    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                               stop: 0 #357ABD, stop: 1 #2E6B9E);
}

/* 라벨 */
QLabel {
    color: #495057;
    font-size: 12px;
}

QLabel#title_label {
    font-size: 14px;
    font-weight: bold;
    color: #212529;
}

/* 스핀박스 */
QSpinBox, QDoubleSpinBox {
    background-color: #FFFFFF;
    border: 1px solid #DEE2E6;
    border-radius: 4px;
    padding: 6px;
    font-size: 12px;
}

QSpinBox:focus, QDoubleSpinBox:focus {
    border-color: #4A90E2;
    border-width: 2px;
}

/* 콤보박스 */
QComboBox {
    background-color: #FFFFFF;
    border: 1px solid #DEE2E6;
    border-radius: 4px;
    padding: 6px;
    font-size: 12px;
}

QComboBox:hover {
    border-color: #ADB5BD;
}

QComboBox:focus {
    border-color: #4A90E2;
}

QComboBox::drop-down {
    border: none;
    width: 20px;
}

QComboBox::down-arrow {
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 6px solid #495057;
    margin-right: 8px;
}

/* 슬라이더 */
QSlider::groove:horizontal {
    background: #E9ECEF;
    height: 4px;
    border-radius: 2px;
}

QSlider::handle:horizontal {
    background: #4A90E2;
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}

QSlider::handle:horizontal:hover {
    background: #5BA0F2;
}

/* 스크롤바 */
QScrollBar:vertical {
    background: #F8F9FA;
    width: 12px;
    border-radius: 6px;
}

QScrollBar::handle:vertical {
    background: #CED4DA;
    border-radius: 6px;
    min-height: 30px;
}

QScrollBar::handle:vertical:hover {
    background: #ADB5BD;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

/* 툴바 */
QToolBar {
    background-color: #FFFFFF;
    border-bottom: 1px solid #E9ECEF;
    spacing: 8px;
    padding: 4px;
}

QToolButton {
    background-color: transparent;
    border: 1px solid transparent;
    border-radius: 4px;
    padding: 6px;
}

QToolButton:hover {
    background-color: #F8F9FA;
    border-color: #DEE2E6;
}

QToolButton:pressed {
    background-color: #E9ECEF;
}
"""


def get_canva_style():
    """Canva 스타일 CSS 반환"""
    return CANVA_STYLE
