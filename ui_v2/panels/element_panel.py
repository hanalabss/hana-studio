"""
ElementPanel - 왼쪽 요소 추가 패널
이미지, 텍스트 추가 버튼
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QGroupBox,
    QFileDialog, QLabel, QSpacerItem, QSizePolicy
)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QFont


class ElementPanel(QWidget):
    """요소 추가 패널"""

    # 시그널
    image_add_requested = Signal(str)  # 이미지 경로
    text_add_requested = Signal()      # 텍스트 추가
    bg_remove_requested = Signal()     # 배경제거 요청

    def __init__(self):
        super().__init__()
        self.setObjectName("left_panel")
        self.setFixedWidth(220)
        self._setup_ui()

    def _setup_ui(self):
        """UI 설정"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # 타이틀
        title = QLabel("요소 추가")
        title.setObjectName("title_label")
        title.setFont(QFont("맑은 고딕", 14, QFont.Weight.Bold))
        layout.addWidget(title)

        # 이미지 그룹
        image_group = QGroupBox("🖼️ 이미지")
        image_layout = QVBoxLayout(image_group)
        image_layout.setSpacing(8)

        self.add_image_btn = QPushButton("이미지 업로드")
        self.add_image_btn.setMinimumHeight(40)
        self.add_image_btn.clicked.connect(self._on_add_image_clicked)
        image_layout.addWidget(self.add_image_btn)

        layout.addWidget(image_group)

        # 텍스트 그룹
        text_group = QGroupBox("✏️ 텍스트")
        text_layout = QVBoxLayout(text_group)
        text_layout.setSpacing(8)

        self.add_text_btn = QPushButton("텍스트 추가")
        self.add_text_btn.setMinimumHeight(40)
        self.add_text_btn.clicked.connect(self._on_add_text_clicked)
        text_layout.addWidget(self.add_text_btn)

        layout.addWidget(text_group)

        # AI 기능 그룹
        ai_group = QGroupBox("🤖 AI 기능")
        ai_layout = QVBoxLayout(ai_group)
        ai_layout.setSpacing(8)

        self.bg_remove_btn = QPushButton("🎨 배경 제거")
        self.bg_remove_btn.setMinimumHeight(40)
        self.bg_remove_btn.clicked.connect(self._on_bg_remove_clicked)
        ai_layout.addWidget(self.bg_remove_btn)

        layout.addWidget(ai_group)

        # 스페이서
        layout.addItem(QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))

    def _on_add_image_clicked(self):
        """이미지 추가 버튼 클릭"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "이미지 선택",
            "",
            "이미지 파일 (*.png *.jpg *.jpeg *.bmp)"
        )

        if file_path:
            self.image_add_requested.emit(file_path)

    def _on_add_text_clicked(self):
        """텍스트 추가 버튼 클릭"""
        self.text_add_requested.emit()

    def _on_bg_remove_clicked(self):
        """배경제거 버튼 클릭"""
        self.bg_remove_requested.emit()
