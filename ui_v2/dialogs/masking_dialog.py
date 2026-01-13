"""
MaskingDialog - 마스킹 처리 미리보기 및 임계값 조정 다이얼로그
원본/마스크 탭과 레이아웃 인쇄 시뮬레이션 탭으로 마스킹 결과 확인 가능
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QSlider, QPushButton, QMessageBox, QTabWidget,
    QWidget
)
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtGui import QPixmap, QImage
import numpy as np
import cv2


class BackgroundRemovalWorker(QThread):
    """배경 제거 비동기 처리 워커"""

    finished = Signal(np.ndarray)  # 완료 시 마스크 이미지 전달
    error = Signal(str)  # 에러 시 메시지 전달

    def __init__(self, image_path: str, ai_session, threshold: int):
        super().__init__()
        self.image_path = image_path
        self.ai_session = ai_session
        self.threshold = threshold

    def run(self):
        """백그라운드에서 배경 제거 실행"""
        try:
            from core.image_processor import ImageProcessor
            from config import config as cfg_instance

            processor = ImageProcessor()

            # 임계값을 config에 임시 설정
            original_threshold = cfg_instance.get('alpha_threshold', 45)
            cfg_instance.set('alpha_threshold', self.threshold)

            # 마스크 생성
            mask_rgb = processor.remove_background(
                self.image_path,
                self.ai_session
            )

            # 원래 임계값으로 복원
            cfg_instance.set('alpha_threshold', original_threshold)

            # 완료 시그널 발생
            self.finished.emit(mask_rgb)

        except Exception as e:
            self.error.emit(str(e))


class MaskingDialog(QDialog):
    """마스킹 처리 미리보기 다이얼로그"""

    threshold_applied = Signal(int, str)  # threshold, mask_path

    def __init__(self, original_image_path: str, ai_session, parent=None):
        super().__init__(parent)

        self.original_image_path = original_image_path
        self.ai_session = ai_session
        self.current_mask_rgb = None
        self.original_image = None  # 원본 이미지 저장
        self.current_threshold = 45  # 슬라이더 현재값
        self.applied_threshold = 45  # 마지막으로 적용된 임계값
        self.is_processing = False  # 처리 중 플래그
        self.worker = None  # Worker 스레드 참조

        self.setWindowTitle("AI 배경 제거 - 마스킹 미리보기")
        self.setModal(True)
        self.resize(1200, 700)  # 탭을 위해 크기 증가

        self._setup_ui()
        self._generate_initial_mask()

    def _setup_ui(self):
        """UI 설정 - 탭 구조로 개선"""
        layout = QVBoxLayout(self)

        # === 탭 위젯 추가 ===
        self.tab_widget = QTabWidget()

        # 탭 1: 원본 vs 마스크
        self.separate_tab = self._create_separate_view()
        self.tab_widget.addTab(self.separate_tab, "원본 / 마스크")

        # 탭 2: 레이아웃 인쇄 시뮬레이션
        self.print_preview_tab = self._create_print_preview_view()
        self.tab_widget.addTab(self.print_preview_tab, "레이아웃 인쇄 시뮬레이션")

        layout.addWidget(self.tab_widget)

        # 임계값 조정 슬라이더
        threshold_layout = QVBoxLayout()

        threshold_label_layout = QHBoxLayout()
        threshold_label_layout.addWidget(QLabel("임계값 (Alpha Threshold):"))
        self.threshold_value_label = QLabel(f"{self.current_threshold}")
        threshold_label_layout.addWidget(self.threshold_value_label)
        threshold_label_layout.addStretch()
        threshold_layout.addLayout(threshold_label_layout)

        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setMinimum(1)
        self.threshold_slider.setMaximum(255)
        self.threshold_slider.setValue(self.current_threshold)
        self.threshold_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.threshold_slider.setTickInterval(10)
        self.threshold_slider.valueChanged.connect(self._on_threshold_changed)
        threshold_layout.addWidget(self.threshold_slider)

        info_label = QLabel("💡 임계값을 조정한 후 '배경 재처리' 버튼을 눌러 새로운 마스크를 생성할 수 있습니다.")
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #666; font-size: 11px;")
        threshold_layout.addWidget(info_label)

        layout.addLayout(threshold_layout)

        # 하단: 버튼
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.cancel_btn = QPushButton("취소")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)

        self.reprocess_btn = QPushButton("배경 재처리")
        self.reprocess_btn.setEnabled(False)  # 초기에는 비활성화
        self.reprocess_btn.setStyleSheet("background-color: #e0e0e0; padding: 8px 16px;")
        self.reprocess_btn.clicked.connect(self._on_reprocess)
        button_layout.addWidget(self.reprocess_btn)

        self.apply_btn = QPushButton("적용")
        self.apply_btn.setObjectName("primary_btn")
        self.apply_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 8px 16px;")
        self.apply_btn.clicked.connect(self._on_apply)
        button_layout.addWidget(self.apply_btn)

        layout.addLayout(button_layout)

    def _create_separate_view(self):
        """원본/마스크 분리 뷰 생성"""
        widget = QWidget()
        layout = QHBoxLayout(widget)

        # 원본 이미지
        original_group = QVBoxLayout()
        original_label = QLabel("원본 이미지")
        original_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        original_group.addWidget(original_label)

        self.original_preview = QLabel()
        self.original_preview.setMinimumSize(500, 400)
        self.original_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.original_preview.setStyleSheet("border: 1px solid #ccc; background: white;")
        original_group.addWidget(self.original_preview)

        layout.addLayout(original_group)

        # 마스크 이미지
        mask_group = QVBoxLayout()
        mask_label = QLabel("마스크 이미지 (흰색=배경, 검은색=객체)")
        mask_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mask_group.addWidget(mask_label)

        self.mask_preview = QLabel()
        self.mask_preview.setMinimumSize(500, 400)
        self.mask_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.mask_preview.setStyleSheet("border: 1px solid #ccc; background: white;")
        mask_group.addWidget(self.mask_preview)

        layout.addLayout(mask_group)

        return widget

    def _create_print_preview_view(self):
        """레이아웃 인쇄 시뮬레이션 뷰 생성 - 3개 캔버스"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # 설명 레이블
        info_label = QLabel(
            "프린터는 W 레이어(마스킹)를 먼저 인쇄한 뒤, 그 위에 YMC 레이어(원본)를 겹쳐서 인쇄합니다."
        )
        info_label.setStyleSheet("color: #555; font-size: 12px; padding: 10px;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # 3개 캔버스를 가로로 배치
        canvas_layout = QHBoxLayout()

        # 1. 원본 이미지 캔버스
        original_group = QVBoxLayout()
        original_title = QLabel("1. 원본 이미지")
        original_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        original_title.setStyleSheet("font-weight: bold; color: #4A90E2; font-size: 11px;")
        original_group.addWidget(original_title)

        self.print_canvas_original = QLabel()
        self.print_canvas_original.setMinimumSize(330, 400)
        self.print_canvas_original.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.print_canvas_original.setStyleSheet("border: 2px solid #4A90E2; background: white; border-radius: 8px;")
        self.print_canvas_original.setText("YMC 레이어")
        original_group.addWidget(self.print_canvas_original)

        original_desc = QLabel("YMC 레이어로 인쇄")
        original_desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        original_desc.setStyleSheet("color: #666; font-size: 10px;")
        original_group.addWidget(original_desc)

        canvas_layout.addLayout(original_group)

        # 2. 마스킹 이미지 캔버스
        mask_group = QVBoxLayout()
        mask_title = QLabel("2. 마스킹 이미지")
        mask_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mask_title.setStyleSheet("font-weight: bold; color: #6C757D; font-size: 11px;")
        mask_group.addWidget(mask_title)

        self.print_canvas_mask = QLabel()
        self.print_canvas_mask.setMinimumSize(330, 400)
        self.print_canvas_mask.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.print_canvas_mask.setStyleSheet("border: 2px solid #6C757D; background: white; border-radius: 8px;")
        self.print_canvas_mask.setText("W 레이어")
        mask_group.addWidget(self.print_canvas_mask)

        mask_desc = QLabel("W 레이어로 인쇄")
        mask_desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mask_desc.setStyleSheet("color: #666; font-size: 10px;")
        mask_group.addWidget(mask_desc)

        canvas_layout.addLayout(mask_group)

        # 3. 합성 결과 캔버스
        composite_group = QVBoxLayout()
        composite_title = QLabel("3. 합성 결과")
        composite_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        composite_title.setStyleSheet("font-weight: bold; color: #28A745; font-size: 11px;")
        composite_group.addWidget(composite_title)

        self.print_canvas_composite = QLabel()
        self.print_canvas_composite.setMinimumSize(330, 400)
        self.print_canvas_composite.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.print_canvas_composite.setStyleSheet("border: 2px solid #28A745; background: white; border-radius: 8px;")
        self.print_canvas_composite.setText("최종 인쇄물")
        composite_group.addWidget(self.print_canvas_composite)

        composite_desc = QLabel("W + YMC 합성")
        composite_desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        composite_desc.setStyleSheet("color: #666; font-size: 10px;")
        composite_group.addWidget(composite_desc)

        canvas_layout.addLayout(composite_group)

        layout.addLayout(canvas_layout)

        return widget

    def _update_print_preview(self):
        """인쇄 미리보기 업데이트 - 3개 캔버스에 표시"""
        if self.current_mask_rgb is None or self.original_image is None:
            return

        try:
            # === 캔버스 1: 원본 이미지 (YMC 레이어) ===
            original_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            height, width, channel = original_rgb.shape
            bytes_per_line = 3 * width
            q_image = QImage(original_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.print_canvas_original.setPixmap(pixmap.scaled(
                330, 400,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))

            # === 캔버스 2: 마스킹 이미지 (W 레이어) ===
            # BGR to RGB 변환
            mask_display = self.current_mask_rgb.copy()
            if len(mask_display.shape) == 3:
                mask_display_rgb = cv2.cvtColor(mask_display, cv2.COLOR_BGR2RGB)
            else:
                mask_display_rgb = cv2.cvtColor(mask_display, cv2.COLOR_GRAY2RGB)

            height, width, channel = mask_display_rgb.shape
            bytes_per_line = 3 * width
            q_image = QImage(mask_display_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.print_canvas_mask.setPixmap(pixmap.scaled(
                330, 400,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))

            # === 캔버스 3: 합성 결과 (W + YMC) ===
            # 원본 이미지 + 녹색 테두리 (마스킹 영역 표시)
            composite = self.original_image.copy()

            # 마스크에서 객체 영역 찾기
            if len(self.current_mask_rgb.shape) == 3:
                gray_mask = cv2.cvtColor(self.current_mask_rgb, cv2.COLOR_BGR2GRAY)
            else:
                gray_mask = self.current_mask_rgb

            # 검은색 = 객체 영역
            object_mask = (gray_mask < 128).astype(np.uint8)

            # 윤곽선 찾기
            contours, _ = cv2.findContours(
                object_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            # 녹색 테두리 그리기 (레이어 분리 표시)
            cv2.drawContours(composite, contours, -1, (0, 255, 0), 3)

            # BGR to RGB 변환
            composite_rgb = cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)
            height, width, channel = composite_rgb.shape
            bytes_per_line = 3 * width
            q_image = QImage(composite_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.print_canvas_composite.setPixmap(pixmap.scaled(
                330, 400,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))

            print("[PRINT_PREVIEW] 3개 캔버스 업데이트 완료")

        except Exception as e:
            print(f"[ERROR] 인쇄 미리보기 업데이트 실패: {e}")
            import traceback
            traceback.print_exc()
            self.print_canvas_original.setText(f"오류: {str(e)}")
            self.print_canvas_mask.setText(f"오류: {str(e)}")
            self.print_canvas_composite.setText(f"오류: {str(e)}")

    def _generate_initial_mask(self):
        """초기 마스크 생성"""
        try:
            # 원본 이미지 로드 (QPixmap)
            original_pixmap = QPixmap(self.original_image_path)
            self.original_preview.setPixmap(original_pixmap.scaled(
                500, 400,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))

            # 원본 이미지 로드 (numpy array - 오버레이용)
            self.original_image = cv2.imread(self.original_image_path)
            if self.original_image is None:
                # 한글 경로 대응
                with open(self.original_image_path, 'rb') as f:
                    image_data = f.read()
                nparr = np.frombuffer(image_data, np.uint8)
                self.original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # 초기 마스크 생성 - 비동기로 실행
            self._start_background_removal(self.current_threshold)

        except Exception as e:
            QMessageBox.critical(self, "오류", f"이미지 로드 실패:\n{e}")
            self.reject()

    def _start_background_removal(self, threshold: int):
        """배경 제거 비동기 시작"""
        if self.is_processing:
            return

        self.is_processing = True

        # UI 피드백 - 재처리 버튼
        self.reprocess_btn.setText("⏳ 처리 중...")
        self.reprocess_btn.setEnabled(False)
        self.reprocess_btn.setStyleSheet("background-color: #ff9800; color: white; padding: 8px 16px;")

        # 다른 버튼들도 비활성화
        self.apply_btn.setEnabled(False)
        self.cancel_btn.setEnabled(False)
        self.threshold_slider.setEnabled(False)

        # 커서 변경
        self.setCursor(Qt.CursorShape.WaitCursor)

        # Worker 생성 및 시작
        self.worker = BackgroundRemovalWorker(
            self.original_image_path,
            self.ai_session,
            threshold
        )
        self.worker.finished.connect(self._on_background_removal_finished)
        self.worker.error.connect(self._on_background_removal_error)
        self.worker.start()

    def _on_background_removal_finished(self, mask_rgb: np.ndarray):
        """배경 제거 완료 시"""
        try:
            self.current_mask_rgb = mask_rgb
            self.applied_threshold = self.current_threshold

            # 마스크를 QPixmap으로 변환하여 표시
            height, width = mask_rgb.shape[:2]
            bytes_per_line = 3 * width
            q_image = QImage(
                mask_rgb.data,
                width, height,
                bytes_per_line,
                QImage.Format.Format_RGB888
            )
            mask_pixmap = QPixmap.fromImage(q_image)

            self.mask_preview.setPixmap(mask_pixmap.scaled(
                500, 400,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))

            # 인쇄 미리보기 업데이트
            self._update_print_preview()

            # UI 복원
            self._restore_ui()

        except Exception as e:
            self._on_background_removal_error(f"마스크 표시 실패: {e}")

    def _on_background_removal_error(self, error_msg: str):
        """배경 제거 오류 시"""
        QMessageBox.critical(self, "오류", f"마스크 생성 실패:\n{error_msg}")
        self._restore_ui()

    def _restore_ui(self):
        """UI 상태 복원"""
        self.is_processing = False

        # 버튼 상태 복원
        self.reprocess_btn.setText("배경 재처리")
        self.reprocess_btn.setEnabled(False)  # 임계값이 변경되기 전까지 비활성화
        self.reprocess_btn.setStyleSheet("background-color: #e0e0e0; padding: 8px 16px;")

        self.apply_btn.setEnabled(True)
        self.cancel_btn.setEnabled(True)
        self.threshold_slider.setEnabled(True)

        # 커서 복원
        self.setCursor(Qt.CursorShape.ArrowCursor)

    def _on_threshold_changed(self, value: int):
        """임계값 변경 시 - 숫자만 업데이트, 재생성은 안 함"""
        self.current_threshold = value
        self.threshold_value_label.setText(f"{value}")

        # 재처리 버튼 활성화 (임계값이 변경된 경우)
        if value != self.applied_threshold and not self.is_processing:
            self.reprocess_btn.setEnabled(True)
            self.reprocess_btn.setStyleSheet("background-color: #2196F3; color: white; padding: 8px 16px;")
        elif value == self.applied_threshold:
            self.reprocess_btn.setEnabled(False)
            self.reprocess_btn.setStyleSheet("background-color: #e0e0e0; padding: 8px 16px;")

    def _on_reprocess(self):
        """재처리 버튼 클릭 - 새로운 임계값으로 배경 제거"""
        if self.is_processing:
            return

        # 배경 제거 시작
        self._start_background_removal(self.current_threshold)

    def _on_apply(self):
        """적용 버튼 클릭"""
        try:
            if self.current_mask_rgb is None:
                QMessageBox.warning(self, "경고", "마스크가 생성되지 않았습니다.")
                return

            # 마스크를 임시 파일로 저장 (한글 경로 대응)
            import cv2
            import tempfile
            import os

            temp_dir = tempfile.gettempdir()
            mask_filename = f"mask_{os.path.basename(self.original_image_path)}"
            mask_path = os.path.join(temp_dir, mask_filename)

            print(f"[DEBUG] 마스크 저장 시도: {mask_path}")

            # 한글 경로 대응: imencode를 사용하여 바이트로 변환 후 저장
            try:
                # 방법 1: imencode 사용 (한글 경로 대응)
                success, encoded_image = cv2.imencode('.jpg', self.current_mask_rgb)
                if success:
                    with open(mask_path, 'wb') as f:
                        f.write(encoded_image.tobytes())
                    print(f"[DEBUG] 마스크 저장 성공: {mask_path}")
                else:
                    raise ValueError("cv2.imencode 실패")
            except Exception as e:
                print(f"[ERROR] 마스크 저장 실패: {e}")
                raise

            # 저장된 파일 존재 확인
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"마스크 파일 저장에 실패했습니다: {mask_path}")

            # 시그널 발생
            self.threshold_applied.emit(self.current_threshold, mask_path)
            self.accept()

        except Exception as e:
            QMessageBox.critical(self, "오류", f"마스크 저장 실패:\n{e}")

    def closeEvent(self, event):
        """다이얼로그 닫기 시 Worker 스레드 정리"""
        if self.worker and self.worker.isRunning():
            self.worker.finished.disconnect()
            self.worker.error.disconnect()
            self.worker.quit()
            self.worker.wait(1000)  # 최대 1초 대기

        super().closeEvent(event)
