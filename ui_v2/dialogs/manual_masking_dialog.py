"""
ManualMaskingDialog - 수동 마스킹 이미지 업로드 및 위치 조정 다이얼로그
외부 프로그램에서 만든 마스킹 이미지와 원본 이미지를 각각 업로드하고
위치를 미세조정하여 레이아웃 인쇄
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QMessageBox, QGroupBox, QSpinBox,
    QFileDialog
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap, QImage
import numpy as np
import cv2
import os


class ManualMaskingDialog(QDialog):
    """수동 마스킹 다이얼로그"""

    # threshold, mask_path, original_path, offset_x_orig, offset_y_orig, offset_x_mask, offset_y_mask
    masking_applied = Signal(int, str, str, int, int, int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("수동 마스킹")
        self.resize(1000, 700)

        # 이미지 경로
        self.original_image_path = None
        self.mask_image_path = None

        # 이미지 데이터
        self.original_image = None
        self.mask_image = None

        # 오프셋 (픽셀 단위)
        self.offset_x_original = 0
        self.offset_y_original = 0
        self.offset_x_mask = 0
        self.offset_y_mask = 0

        self._setup_ui()

    def _setup_ui(self):
        """UI 설정"""
        layout = QVBoxLayout(self)

        # === 설명 ===
        info_label = QLabel(
            "외부 프로그램에서 만든 원본 이미지와 마스킹 이미지를 업로드하세요.\n"
            "위치가 약간 틀어진 경우 미세조정 기능을 사용하세요."
        )
        info_label.setStyleSheet("color: #555; font-size: 11px; padding: 10px;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # === 이미지 업로드 버튼 ===
        upload_layout = QHBoxLayout()

        # 원본 이미지 업로드
        original_group = QGroupBox("1. 원본 이미지")
        original_layout = QVBoxLayout(original_group)

        self.original_upload_btn = QPushButton("원본 이미지 선택...")
        self.original_upload_btn.clicked.connect(self._upload_original)
        original_layout.addWidget(self.original_upload_btn)

        self.original_path_label = QLabel("선택 안됨")
        self.original_path_label.setStyleSheet("color: #888; font-size: 10px;")
        self.original_path_label.setWordWrap(True)
        original_layout.addWidget(self.original_path_label)

        upload_layout.addWidget(original_group)

        # 마스킹 이미지 업로드
        mask_group = QGroupBox("2. 마스킹 이미지")
        mask_layout = QVBoxLayout(mask_group)

        self.mask_upload_btn = QPushButton("마스킹 이미지 선택...")
        self.mask_upload_btn.clicked.connect(self._upload_mask)
        mask_layout.addWidget(self.mask_upload_btn)

        self.mask_path_label = QLabel("선택 안됨")
        self.mask_path_label.setStyleSheet("color: #888; font-size: 10px;")
        self.mask_path_label.setWordWrap(True)
        mask_layout.addWidget(self.mask_path_label)

        upload_layout.addWidget(mask_group)

        layout.addLayout(upload_layout)

        # === 미리보기 ===
        preview_group = QGroupBox("3. 최종 인쇄 미리보기")
        preview_layout = QVBoxLayout(preview_group)

        self.preview_label = QLabel()
        self.preview_label.setMinimumSize(900, 400)
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet("border: 1px solid #ccc; background: white;")
        self.preview_label.setText("원본과 마스킹 이미지를 업로드하면 미리보기가 표시됩니다.")
        preview_layout.addWidget(self.preview_label)

        layout.addWidget(preview_group)

        # === 위치 미세조정 ===
        adjust_group = QGroupBox("4. 위치 미세조정 (픽셀 단위)")
        adjust_layout = QHBoxLayout(adjust_group)

        # 원본 이미지 조정
        original_adjust = QVBoxLayout()
        original_adjust.addWidget(QLabel("원본 이미지 위치:"))

        orig_x_layout = QHBoxLayout()
        orig_x_layout.addWidget(QLabel("X:"))
        self.offset_x_original_spin = QSpinBox()
        self.offset_x_original_spin.setRange(-100, 100)
        self.offset_x_original_spin.setValue(0)
        self.offset_x_original_spin.valueChanged.connect(self._on_offset_changed)
        orig_x_layout.addWidget(self.offset_x_original_spin)
        original_adjust.addLayout(orig_x_layout)

        orig_y_layout = QHBoxLayout()
        orig_y_layout.addWidget(QLabel("Y:"))
        self.offset_y_original_spin = QSpinBox()
        self.offset_y_original_spin.setRange(-100, 100)
        self.offset_y_original_spin.setValue(0)
        self.offset_y_original_spin.valueChanged.connect(self._on_offset_changed)
        orig_y_layout.addWidget(self.offset_y_original_spin)
        original_adjust.addLayout(orig_y_layout)

        adjust_layout.addLayout(original_adjust)

        # 마스킹 이미지 조정
        mask_adjust = QVBoxLayout()
        mask_adjust.addWidget(QLabel("마스킹 이미지 위치:"))

        mask_x_layout = QHBoxLayout()
        mask_x_layout.addWidget(QLabel("X:"))
        self.offset_x_mask_spin = QSpinBox()
        self.offset_x_mask_spin.setRange(-100, 100)
        self.offset_x_mask_spin.setValue(0)
        self.offset_x_mask_spin.valueChanged.connect(self._on_offset_changed)
        mask_x_layout.addWidget(self.offset_x_mask_spin)
        mask_adjust.addLayout(mask_x_layout)

        mask_y_layout = QHBoxLayout()
        mask_y_layout.addWidget(QLabel("Y:"))
        self.offset_y_mask_spin = QSpinBox()
        self.offset_y_mask_spin.setRange(-100, 100)
        self.offset_y_mask_spin.setValue(0)
        self.offset_y_mask_spin.valueChanged.connect(self._on_offset_changed)
        mask_y_layout.addWidget(self.offset_y_mask_spin)
        mask_adjust.addLayout(mask_y_layout)

        adjust_layout.addLayout(mask_adjust)

        # 리셋 버튼
        reset_btn = QPushButton("위치 리셋")
        reset_btn.clicked.connect(self._reset_offsets)
        adjust_layout.addWidget(reset_btn)

        layout.addWidget(adjust_group)

        # === 하단 버튼 ===
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.apply_btn = QPushButton("적용")
        self.apply_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 8px 16px;")
        self.apply_btn.clicked.connect(self._on_apply)
        self.apply_btn.setEnabled(False)
        button_layout.addWidget(self.apply_btn)

        cancel_btn = QPushButton("취소")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        layout.addLayout(button_layout)

    def _upload_original(self):
        """원본 이미지 업로드"""
        from config import config
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "원본 이미지 선택",
            "",
            config.get_image_filter()
        )

        if file_path:
            self.original_image_path = file_path
            self.original_path_label.setText(os.path.basename(file_path))

            # 이미지 로드
            from core.file_manager import FileManager
            file_mgr = FileManager()
            self.original_image = file_mgr._safe_imread(file_path)

            if self.original_image is None:
                QMessageBox.critical(self, "오류", "원본 이미지를 로드할 수 없습니다.")
                return

            print(f"[INFO] 원본 이미지 로드: {file_path}")
            self._update_preview()

    def _upload_mask(self):
        """마스킹 이미지 업로드"""
        from config import config
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "마스킹 이미지 선택",
            "",
            config.get_image_filter()
        )

        if file_path:
            self.mask_image_path = file_path
            self.mask_path_label.setText(os.path.basename(file_path))

            # 이미지 로드
            from core.file_manager import FileManager
            file_mgr = FileManager()
            self.mask_image = file_mgr._safe_imread(file_path)

            if self.mask_image is None:
                QMessageBox.critical(self, "오류", "마스킹 이미지를 로드할 수 없습니다.")
                return

            print(f"[INFO] 마스킹 이미지 로드: {file_path}")
            self._update_preview()

    def _on_offset_changed(self):
        """오프셋 변경 시"""
        self.offset_x_original = self.offset_x_original_spin.value()
        self.offset_y_original = self.offset_y_original_spin.value()
        self.offset_x_mask = self.offset_x_mask_spin.value()
        self.offset_y_mask = self.offset_y_mask_spin.value()

        print(f"[INFO] 오프셋 변경: 원본({self.offset_x_original}, {self.offset_y_original}), "
              f"마스크({self.offset_x_mask}, {self.offset_y_mask})")

        self._update_preview()

    def _reset_offsets(self):
        """오프셋 리셋"""
        self.offset_x_original_spin.setValue(0)
        self.offset_y_original_spin.setValue(0)
        self.offset_x_mask_spin.setValue(0)
        self.offset_y_mask_spin.setValue(0)

    def _update_preview(self):
        """미리보기 업데이트"""
        if self.original_image is None or self.mask_image is None:
            return

        try:
            # 크기 확인
            if self.original_image.shape[:2] != self.mask_image.shape[:2]:
                # 크기가 다르면 마스크를 원본 크기로 리사이즈
                h, w = self.original_image.shape[:2]
                mask_resized = cv2.resize(self.mask_image, (w, h), interpolation=cv2.INTER_LANCZOS4)
                print(f"[INFO] 마스크 크기 조정: {self.mask_image.shape[:2]} → {(w, h)}")
            else:
                mask_resized = self.mask_image.copy()

            # 오프셋 적용된 최종 합성 생성
            result = self._create_composite_with_offset(
                self.original_image,
                mask_resized,
                self.offset_x_original,
                self.offset_y_original,
                self.offset_x_mask,
                self.offset_y_mask
            )

            # BGR to RGB 변환
            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

            # QPixmap으로 변환하여 표시
            height, width, channel = result_rgb.shape
            bytes_per_line = 3 * width
            q_image = QImage(result_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)

            self.preview_label.setPixmap(pixmap.scaled(
                900, 400,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))

            # 적용 버튼 활성화
            self.apply_btn.setEnabled(True)

        except Exception as e:
            print(f"[ERROR] 미리보기 업데이트 실패: {e}")
            import traceback
            traceback.print_exc()

    def _create_composite_with_offset(self, original, mask, offset_x_orig, offset_y_orig, offset_x_mask, offset_y_mask):
        """오프셋이 적용된 최종 합성 이미지 생성"""
        h, w = original.shape[:2]

        # 캔버스 생성 (약간 여유 공간)
        canvas_h = h + 200
        canvas_w = w + 200
        canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

        # 중앙 기준점
        center_x = 100
        center_y = 100

        # 마스크 배치 (오프셋 적용)
        mask_x = center_x + offset_x_mask
        mask_y = center_y + offset_y_mask

        # 원본 배치 (오프셋 적용)
        orig_x = center_x + offset_x_orig
        orig_y = center_y + offset_y_orig

        # 마스크의 검은색 영역 추출
        gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        object_mask = (gray_mask < 128)

        # W 레이어 시뮬레이션: 마스크 검은색 부분에 흰색 베이스 표시
        # (실제로는 W 레이어가 먼저 인쇄되지만, 여기서는 시각화를 위해 생략)

        # YMC 레이어: 원본 이미지 배치
        result = canvas.copy()
        result[orig_y:orig_y+h, orig_x:orig_x+w] = original

        # W 레이어 없는 부분(배경)을 반투명 효과
        # 마스크 오프셋 적용된 위치에서 배경 영역 계산
        background_mask_canvas = np.zeros((canvas_h, canvas_w), dtype=bool)
        background_mask = (gray_mask >= 128)
        background_mask_canvas[mask_y:mask_y+h, mask_x:mask_x+w] = background_mask

        # 원본 영역 내에서만 반투명 효과 적용
        orig_region_mask = np.zeros((canvas_h, canvas_w), dtype=bool)
        orig_region_mask[orig_y:orig_y+h, orig_x:orig_x+w] = True

        # 배경이면서 원본 영역 내인 곳만 반투명 효과
        apply_transparency = background_mask_canvas & orig_region_mask

        result = result.astype(np.float32)
        white_bg = np.ones_like(result) * 255
        result[apply_transparency] = cv2.addWeighted(
            result[apply_transparency], 0.3,
            white_bg[apply_transparency], 0.7,
            0
        )
        result = result.astype(np.uint8)

        # 실제 사용될 영역만 잘라내기
        return result[center_y:center_y+h, center_x:center_x+w]

    def _on_apply(self):
        """적용 버튼 클릭"""
        try:
            if self.original_image is None or self.mask_image is None:
                QMessageBox.warning(self, "경고", "원본과 마스킹 이미지를 모두 업로드하세요.")
                return

            # 시그널 발생 (threshold=0은 수동 마스킹 표시)
            self.masking_applied.emit(
                0,  # threshold (수동 마스킹은 0)
                self.mask_image_path,
                self.original_image_path,
                self.offset_x_original,
                self.offset_y_original,
                self.offset_x_mask,
                self.offset_y_mask
            )

            print(f"[OK] 수동 마스킹 적용: 원본={self.original_image_path}, 마스크={self.mask_image_path}")
            print(f"[OK] 오프셋: 원본({self.offset_x_original}, {self.offset_y_original}), "
                  f"마스크({self.offset_x_mask}, {self.offset_y_mask})")

            self.accept()

        except Exception as e:
            QMessageBox.critical(self, "오류", f"적용 실패:\n{e}")
            print(f"[ERROR] 수동 마스킹 적용 오류: {e}")
            import traceback
            traceback.print_exc()
