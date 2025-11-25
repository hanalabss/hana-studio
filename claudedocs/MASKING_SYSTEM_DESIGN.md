# Masking System Design & User Experience Improvement

## 문제 분석 (Problem Analysis)

### 현재 문제점

1. **마스킹 크기 불일치 이슈**
   - AI 배경제거 후 마스킹 이미지 크기가 비정상적으로 커짐
   - `save_mask_for_printing()` 저장 시 원본 이미지 크기와 다른 크기로 저장됨
   - 인쇄 시 마스킹과 원본 이미지 크기 불일치로 레이아웃이 깨짐

2. **사용자 검증 어려움**
   - 마스킹 처리가 제대로 되었는지 육안으로 확인하기 어려움
   - 원본 이미지와 마스킹 이미지의 겹침 미리보기 부재
   - 레이어 인쇄 시 최종 결과 예측이 어려움

3. **수동 마스킹 워크플로우 부재**
   - AI 마스킹이 만족스럽지 않을 때 수정 방법 없음
   - 수동 마스킹 업로드 기능은 있으나 검증 방법 부족

## 핵심 요구사항 (Core Requirements)

### 1. 마스킹 크기 일관성 보장
```
입력: 원본 이미지 (W x H)
AI 처리: 배경제거 마스킹 → (W x H) 유지
저장: mask_print.jpg → (W x H) 보장
인쇄: 원본 + 마스크 → 정확한 레이어 정렬
```

### 2. 시각적 검증 시스템
- 원본 이미지와 마스킹 이미지의 오버레이 미리보기
- 토글 기능으로 원본 ↔ 마스킹 ↔ 합성 전환
- 레이어 인쇄 최종 결과 시뮬레이션

### 3. 수동 마스킹 워크플로우
- 사용자가 외부 툴에서 제작한 마스킹 이미지 업로드
- 자동 크기 검증 및 원본 이미지 크기와 비교
- 크기 불일치 시 자동 리사이즈 또는 경고

## 설계 솔루션 (Design Solutions)

### Solution 1: 마스킹 크기 보장 메커니즘

#### 1.1 AI 배경제거 처리 파이프라인 개선
```python
# core/image_processor.py 개선안

def process_background_removal(self, image_path: str, threshold: int = 200):
    """배경제거 처리 - 원본 크기 보장"""
    # 1. 원본 이미지 로드 및 크기 확인
    original_image = self._safe_imread(image_path)
    original_height, original_width = original_image.shape[:2]

    print(f"[MASK] 원본 크기: {original_width} x {original_height}")

    # 2. AI 배경제거 수행
    mask_result = rembg.remove(original_image)

    # 3. 결과 크기 검증 및 강제 일치
    result_height, result_width = mask_result.shape[:2]

    if result_height != original_height or result_width != original_width:
        print(f"[WARNING] 마스크 크기 불일치 감지: {result_width}x{result_height}")
        print(f"[FIX] 원본 크기로 리사이즈: {original_width}x{original_height}")
        mask_result = cv2.resize(
            mask_result,
            (original_width, original_height),
            interpolation=cv2.INTER_LANCZOS4  # 고품질 리사이즈
        )

    # 4. 최종 검증
    assert mask_result.shape[:2] == (original_height, original_width), \
        f"마스크 크기 보장 실패: {mask_result.shape} != {original_image.shape}"

    print(f"[OK] 마스크 크기 검증 완료: {mask_result.shape}")

    return mask_result
```

#### 1.2 저장 시 크기 재검증
```python
# core/file_manager.py 개선안

def save_mask_for_printing(self, mask_image: np.ndarray,
                          original_image_path: str,
                          side: str = "front") -> Optional[str]:
    """프린터용 마스크 저장 - 원본 크기 강제 일치"""

    # 1. 원본 이미지 로드하여 크기 확인
    original_image = self._safe_imread(original_image_path)
    if original_image is None:
        print(f"[ERROR] 원본 이미지 로드 실패: {original_image_path}")
        return None

    orig_h, orig_w = original_image.shape[:2]
    mask_h, mask_w = mask_image.shape[:2]

    print(f"[MASK SAVE] 원본 크기: {orig_w} x {orig_h}")
    print(f"[MASK SAVE] 마스크 크기: {mask_w} x {mask_h}")

    # 2. 크기 불일치 시 강제 리사이즈
    if mask_h != orig_h or mask_w != orig_w:
        print(f"[FIX] 마스크를 원본 크기로 리사이즈")
        mask_image = cv2.resize(
            mask_image,
            (orig_w, orig_h),
            interpolation=cv2.INTER_LANCZOS4
        )

    # 3. 저장
    mask_filename = self._generate_safe_filename(original_image_path, side, "mask_print")
    mask_path = os.path.join(self.temp_dir, mask_filename)

    success = self._safe_imwrite(mask_path, mask_image)

    if success:
        # 4. 저장 후 재검증 (파일 로드하여 크기 확인)
        saved_mask = self._safe_imread(mask_path)
        if saved_mask is not None:
            saved_h, saved_w = saved_mask.shape[:2]
            if saved_h == orig_h and saved_w == orig_w:
                print(f"[OK] 마스크 저장 및 크기 검증 완료: {mask_path}")
                return mask_path
            else:
                print(f"[ERROR] 저장된 마스크 크기 불일치: {saved_w}x{saved_h}")
                return None

    return None
```

### Solution 2: 오버레이 미리보기 시스템

#### 2.1 통합 뷰어 개선 (UnifiedMaskViewer)
```python
# ui/components/image_viewer.py 개선안

class UnifiedMaskViewer(QWidget):
    """통합 마스킹 뷰어 - 오버레이 기능 포함"""

    # 새로운 시그널
    preview_mode_changed = Signal(str)  # "original", "mask", "overlay"

    def __init__(self, title=""):
        super().__init__()
        self.title = title

        # 데이터 저장
        self.original_image = None  # 원본 이미지 저장 추가
        self.auto_mask_array = None
        self.manual_mask_array = None
        self.current_mask_type = None

        # 미리보기 모드
        self.preview_mode = "overlay"  # "original", "mask", "overlay"
        self.overlay_opacity = 0.7  # 오버레이 투명도

        self._setup_ui()

    def _setup_ui(self):
        """UI 구성"""
        layout = QVBoxLayout(self)

        # 상단 여백
        spacer_top = QWidget()
        spacer_top.setFixedHeight(UNIFIED_TOP_MARGIN)
        layout.addWidget(spacer_top)

        # === 새로운 미리보기 모드 토글 버튼 ===
        preview_control = QHBoxLayout()
        preview_control.setSpacing(6)

        self.btn_original = QRadioButton("원본")
        self.btn_mask = QRadioButton("마스크")
        self.btn_overlay = QRadioButton("겹침")
        self.btn_overlay.setChecked(True)

        preview_group = QButtonGroup()
        preview_group.addButton(self.btn_original, 0)
        preview_group.addButton(self.btn_mask, 1)
        preview_group.addButton(self.btn_overlay, 2)

        # 스타일 적용
        for btn in [self.btn_original, self.btn_mask, self.btn_overlay]:
            btn.setFixedSize(75, 30)
            btn.setFont(QFont("Segoe UI", 9))
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setStyleSheet("""
                QRadioButton {
                    background-color: #F8F9FA;
                    border: 1px solid #DEE2E6;
                    border-radius: 4px;
                    padding: 5px 8px;
                }
                QRadioButton:checked {
                    background-color: #28A745;
                    color: white;
                    font-weight: 600;
                }
                QRadioButton::indicator { width: 0; height: 0; }
            """)

        preview_control.addStretch()
        preview_control.addWidget(self.btn_original)
        preview_control.addWidget(self.btn_mask)
        preview_control.addWidget(self.btn_overlay)
        preview_control.addStretch()

        preview_container = QWidget()
        preview_container.setFixedHeight(40)
        preview_container.setLayout(preview_control)
        layout.addWidget(preview_container)

        # 시그널 연결
        self.btn_original.toggled.connect(lambda: self._on_preview_mode_changed("original"))
        self.btn_mask.toggled.connect(lambda: self._on_preview_mode_changed("mask"))
        self.btn_overlay.toggled.connect(lambda: self._on_preview_mode_changed("overlay"))

        # === 이미지 표시 라벨 ===
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setFixedHeight(UNIFIED_IMAGE_HEIGHT - 40)  # 버튼 공간 확보
        self.image_label.setStyleSheet("""
            QLabel {
                background: #FFFFFF;
                border: 2px solid #28A745;
                border-radius: 12px;
            }
        """)
        layout.addWidget(self.image_label)

        # === 하단 상태 라벨 ===
        self.type_label = QLabel()
        self.type_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.type_label.setFixedHeight(32)
        layout.addWidget(self.type_label)

        layout.addStretch()

    def _on_preview_mode_changed(self, mode: str):
        """미리보기 모드 변경"""
        if self.btn_original.isChecked():
            self.preview_mode = "original"
        elif self.btn_mask.isChecked():
            self.preview_mode = "mask"
        elif self.btn_overlay.isChecked():
            self.preview_mode = "overlay"

        print(f"[PREVIEW] 모드 변경: {self.preview_mode}")
        self.preview_mode_changed.emit(self.preview_mode)
        self._update_composite_display()

    def set_original_image(self, image_array: np.ndarray):
        """원본 이미지 설정 (오버레이용)"""
        self.original_image = image_array.copy() if image_array is not None else None
        self._update_composite_display()

    def _update_composite_display(self):
        """미리보기 모드에 따라 합성 이미지 생성 및 표시"""

        # 현재 활성 마스크 가져오기
        current_mask = self.get_current_mask()

        if self.preview_mode == "original":
            # 원본 이미지만 표시
            if self.original_image is not None:
                display_image = self.original_image
            else:
                self._set_placeholder_text()
                return

        elif self.preview_mode == "mask":
            # 마스크만 표시
            if current_mask is not None:
                display_image = current_mask
            else:
                self._set_placeholder_text()
                return

        elif self.preview_mode == "overlay":
            # 오버레이 합성 표시
            if self.original_image is not None and current_mask is not None:
                # 크기 일치 확인
                if self.original_image.shape[:2] != current_mask.shape[:2]:
                    print(f"[WARNING] 크기 불일치 - 원본: {self.original_image.shape}, 마스크: {current_mask.shape}")
                    # 마스크를 원본 크기로 리사이즈
                    h, w = self.original_image.shape[:2]
                    current_mask = cv2.resize(current_mask, (w, h), interpolation=cv2.INTER_LANCZOS4)

                # 오버레이 합성
                display_image = self._create_overlay(
                    self.original_image,
                    current_mask,
                    self.overlay_opacity
                )
            else:
                self._set_placeholder_text()
                return

        else:
            self._set_placeholder_text()
            return

        # QPixmap 변환 및 표시
        pixmap = self._numpy_to_pixmap(display_image)
        if not pixmap.isNull():
            self.original_pixmap = pixmap
            self.update_display()

    def _create_overlay(self, original: np.ndarray, mask: np.ndarray, opacity: float = 0.7) -> np.ndarray:
        """원본과 마스크를 오버레이 합성"""
        try:
            # 마스크를 반투명하게 합성
            # 마스크의 불투명한 부분은 녹색 오버레이로 표시
            overlay = original.copy()

            # 마스크에서 알파 채널 추출 (BGRA 형식인 경우)
            if mask.shape[2] == 4:
                alpha = mask[:, :, 3]
                # 알파가 0이 아닌 부분에 녹색 틴트 적용
                green_overlay = np.zeros_like(original)
                green_overlay[:, :, 1] = 255  # 녹색 채널

                # 알파 값을 0-1 범위로 정규화
                alpha_normalized = alpha.astype(float) / 255.0

                # 녹색 오버레이 적용
                for c in range(3):
                    overlay[:, :, c] = (
                        original[:, :, c] * (1 - alpha_normalized * opacity) +
                        green_overlay[:, :, c] * alpha_normalized * opacity
                    ).astype(np.uint8)

            else:
                # BGR 마스크인 경우 단순 블렌딩
                overlay = cv2.addWeighted(original, 1 - opacity, mask, opacity, 0)

            return overlay

        except Exception as e:
            print(f"[ERROR] 오버레이 생성 실패: {e}")
            return original
```

#### 2.2 메인 창 통합
```python
# hana_studio.py 개선안

def on_front_processing_finished(self, mask_array, used_threshold, original_threshold):
    """앞면 자동 배경제거 완료"""
    # 임계값 복원
    from config import config
    config.set('alpha_threshold', original_threshold)

    self.front_auto_mask_image = mask_array

    # === 개선: 원본 이미지도 함께 설정 ===
    self.ui.components['front_unified_mask_viewer'].set_original_image(
        self.front_original_image
    )
    self.ui.components['front_unified_mask_viewer'].set_auto_mask(mask_array)

    self.log(f"[OK] 앞면 자동 배경 제거 완료!")
    self.log("   💡 겹침 미리보기로 마스킹 결과를 확인하세요")

    # UI 정리
    self.ui.components['progress_panel'].hide_progress()
    self.ui.components['front_original_viewer'].set_process_enabled(True)

    self._update_ui_state()
    self._update_print_button_state()
```

### Solution 3: 수동 마스킹 워크플로우

#### 3.1 크기 검증 및 자동 조정
```python
# hana_studio.py 개선안

def on_manual_mask_uploaded(self, file_path: str, is_front: bool):
    """수동 마스킹 이미지 업로드 처리 - 크기 검증 포함"""
    try:
        import cv2
        import numpy as np

        image_processor = self.get_image_processor()

        # 1. 이미지 유효성 검사
        is_valid, error_msg = image_processor.validate_image(file_path)
        if not is_valid:
            QMessageBox.warning(self, "경고", f"마스킹 이미지 오류: {error_msg}")
            return

        # 2. 마스킹 이미지 로드
        with open(file_path, 'rb') as f:
            image_data = f.read()
        nparr = np.frombuffer(image_data, np.uint8)
        mask_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if mask_image is None:
            QMessageBox.warning(self, "경고", "마스킹 이미지를 읽을 수 없습니다.")
            return

        side_text = "앞면" if is_front else "뒷면"

        # === 3. 원본 이미지와 크기 비교 및 검증 ===
        original_image = self.front_original_image if is_front else self.back_original_image

        if original_image is None:
            QMessageBox.warning(
                self,
                "경고",
                f"{side_text} 원본 이미지를 먼저 선택해주세요."
            )
            return

        orig_h, orig_w = original_image.shape[:2]
        mask_h, mask_w = mask_image.shape[:2]

        self.log(f"[MANUAL] {side_text} 마스킹 업로드 - 크기 검증 중...")
        self.log(f"   원본: {orig_w} x {orig_h}")
        self.log(f"   마스크: {mask_w} x {mask_h}")

        # 4. 크기 불일치 시 처리
        if mask_h != orig_h or mask_w != orig_w:
            # 사용자에게 자동 리사이즈 확인
            reply = QMessageBox.question(
                self,
                "크기 불일치",
                f"{side_text} 마스킹 이미지 크기가 원본과 다릅니다.\n\n"
                f"원본 크기: {orig_w} x {orig_h}\n"
                f"마스크 크기: {mask_w} x {mask_h}\n\n"
                f"자동으로 원본 크기에 맞춰 조정하시겠습니까?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes
            )

            if reply == QMessageBox.StandardButton.Yes:
                self.log(f"[RESIZE] 마스크를 원본 크기로 리사이즈...")
                mask_image = cv2.resize(
                    mask_image,
                    (orig_w, orig_h),
                    interpolation=cv2.INTER_LANCZOS4  # 고품질 리사이즈
                )
                self.log(f"[OK] 리사이즈 완료: {mask_image.shape}")
            else:
                self.log(f"[CANCEL] 크기가 맞지 않아 업로드 취소됨")
                return

        # 5. 마스킹 설정
        file_name = os.path.basename(file_path)

        if is_front:
            self.front_manual_mask_path = file_path
            self.front_manual_mask_image = mask_image

            # 통합 뷰어에 원본 + 수동 마스킹 설정
            self.ui.components['front_unified_mask_viewer'].set_original_image(
                self.front_original_image
            )
            self.ui.components['front_unified_mask_viewer'].set_manual_mask(mask_image)

            self.log(f"[OK] {side_text} 수동 마스킹 업로드 완료!")
            self.log(f"   파일: {file_name}")
            self.log(f"   💡 겹침 미리보기로 마스킹을 확인하세요")
        else:
            self.back_manual_mask_path = file_path
            self.back_manual_mask_image = mask_image

            self.ui.components['back_unified_mask_viewer'].set_original_image(
                self.back_original_image
            )
            self.ui.components['back_unified_mask_viewer'].set_manual_mask(mask_image)

            self.log(f"[OK] {side_text} 수동 마스킹 업로드 완료!")
            self.log(f"   파일: {file_name}")
            self.log(f"   💡 겹침 미리보기로 마스킹을 확인하세요")

        # UI 상태 업데이트
        self._update_ui_state()
        self._update_print_button_state()

    except Exception as e:
        side_text = "앞면" if is_front else "뒷면"
        error_msg = f"{side_text} 수동 마스킹 업로드 실패: {e}"
        self.log(f"[ERROR] {error_msg}")
        QMessageBox.critical(self, "업로드 오류", error_msg)
```

### Solution 4: 메뉴 통합 및 사용성 개선

#### 4.1 "요소 추가" 메뉴에 수동 마스킹 메뉴 추가
```python
# ui/main_window.py 개선안 (메뉴 추가)

def _create_menu_bar(self):
    """메뉴바 생성"""
    menubar = self.menuBar()

    # 파일 메뉴
    file_menu = menubar.addMenu("파일")
    # ... 기존 메뉴들 ...

    # === 새로운 "요소 추가" 메뉴 ===
    add_menu = menubar.addMenu("요소 추가")

    # 수동 마스킹 업로드 서브메뉴
    add_front_mask_action = add_menu.addAction("앞면 마스킹 이미지 업로드...")
    add_front_mask_action.triggered.connect(
        lambda: self._upload_manual_mask_from_menu(is_front=True)
    )

    add_back_mask_action = add_menu.addAction("뒷면 마스킹 이미지 업로드...")
    add_back_mask_action.triggered.connect(
        lambda: self._upload_manual_mask_from_menu(is_front=False)
    )

    add_menu.addSeparator()

    # 추후 확장 가능한 메뉴들
    # add_menu.addAction("텍스트 추가...")  # v2 기능
    # add_menu.addAction("로고 추가...")    # v2 기능

def _upload_manual_mask_from_menu(self, is_front: bool):
    """메뉴에서 수동 마스킹 업로드"""
    side_text = "앞면" if is_front else "뒷면"

    # 원본 이미지 확인
    original_path = self.front_image_path if is_front else self.back_image_path
    if not original_path:
        QMessageBox.warning(
            self,
            "경고",
            f"{side_text} 원본 이미지를 먼저 선택해주세요."
        )
        return

    # 파일 선택 다이얼로그
    from config import config
    import sys

    initial_dir = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.getcwd()

    file_path, _ = QFileDialog.getOpenFileName(
        self,
        f"{side_text} 마스킹 이미지 선택",
        initial_dir,
        config.get_image_filter()
    )

    if file_path:
        # 기존 on_manual_mask_uploaded 로직 호출
        self.on_manual_mask_uploaded(file_path, is_front)
```

#### 4.2 단축키 추가
```python
# ui/main_window.py 개선안 (단축키)

def _setup_shortcuts(self):
    """단축키 설정"""
    from PySide6.QtGui import QShortcut, QKeySequence

    # 앞면 마스킹 업로드: Ctrl+Shift+M
    shortcut_front_mask = QShortcut(QKeySequence("Ctrl+Shift+M"), self)
    shortcut_front_mask.activated.connect(
        lambda: self._upload_manual_mask_from_menu(is_front=True)
    )

    # 뒷면 마스킹 업로드: Ctrl+Alt+M
    shortcut_back_mask = QShortcut(QKeySequence("Ctrl+Alt+M"), self)
    shortcut_back_mask.activated.connect(
        lambda: self._upload_manual_mask_from_menu(is_front=False)
    )

    # 미리보기 모드 토글: V 키
    shortcut_toggle_preview = QShortcut(QKeySequence("V"), self)
    shortcut_toggle_preview.activated.connect(self._toggle_preview_mode)

def _toggle_preview_mode(self):
    """미리보기 모드 순환 토글 (원본 → 마스크 → 겹침)"""
    # 현재 활성 탭의 통합 뷰어 가져오기
    current_tab = self.ui.get_current_tab_index()

    if current_tab == 0:  # 앞면
        viewer = self.ui.components['front_unified_mask_viewer']
    elif current_tab == 1:  # 뒷면
        viewer = self.ui.components['back_unified_mask_viewer']
    else:
        return

    # 모드 순환
    if viewer.btn_original.isChecked():
        viewer.btn_mask.setChecked(True)
    elif viewer.btn_mask.isChecked():
        viewer.btn_overlay.setChecked(True)
    else:
        viewer.btn_original.setChecked(True)
```

## UI/UX 개선 사항 (User Experience Enhancements)

### 1. 시각적 피드백 개선

#### 크기 불일치 경고 배지
```python
# UnifiedMaskViewer에 경고 표시 추가

def _show_size_mismatch_warning(self):
    """크기 불일치 경고 표시"""
    self.warning_label = QLabel("⚠️ 크기 불일치")
    self.warning_label.setStyleSheet("""
        QLabel {
            background-color: #FFC107;
            color: #000;
            font-weight: 700;
            padding: 6px 12px;
            border-radius: 4px;
            font-size: 11px;
        }
    """)
    # 상단 오른쪽에 배치
```

#### 마스킹 품질 인디케이터
```python
def _show_quality_indicator(self, coverage_percent: float):
    """마스킹 적용 범위 표시"""
    # 마스크가 이미지의 몇 %를 차지하는지 계산하여 표시
    self.quality_label.setText(f"적용 범위: {coverage_percent:.1f}%")
```

### 2. 사용자 가이드 개선

#### 툴팁 추가
- 각 뷰어에 상세한 툴팁 제공
- 버튼에 단축키 정보 표시
- 첫 사용 시 간단한 가이드 표시

#### 컨텍스트 메뉴
```python
def _setup_context_menu(self):
    """마우스 우클릭 메뉴"""
    self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
    self.customContextMenuRequested.connect(self._show_context_menu)

def _show_context_menu(self, position):
    """컨텍스트 메뉴 표시"""
    menu = QMenu()

    # 미리보기 모드 변경
    menu.addAction("원본 보기 (V)", lambda: self.btn_original.setChecked(True))
    menu.addAction("마스크 보기 (V)", lambda: self.btn_mask.setChecked(True))
    menu.addAction("겹침 보기 (V)", lambda: self.btn_overlay.setChecked(True))

    menu.addSeparator()

    # 저장 옵션
    menu.addAction("마스크 이미지로 저장...", self._save_mask_as_file)
    menu.addAction("미리보기 이미지로 저장...", self._save_preview_as_file)

    menu.exec(self.mapToGlobal(position))
```

## 구현 우선순위 (Implementation Priority)

### Phase 1: 긴급 수정 (Critical Fixes)
1. ✅ 마스킹 크기 보장 메커니즘 구현
   - `process_background_removal()` 크기 검증 추가
   - `save_mask_for_printing()` 강제 리사이즈 추가
2. ✅ 수동 마스킹 크기 검증 로직
   - `on_manual_mask_uploaded()` 크기 비교 및 조정

### Phase 2: 핵심 UX 개선 (Core UX)
3. 🔄 오버레이 미리보기 구현
   - `UnifiedMaskViewer` 개선 (원본/마스크/겹침 모드)
   - 원본 이미지 저장 및 합성 로직
4. 🔄 메뉴 및 단축키 추가
   - "요소 추가" 메뉴 구현
   - 단축키 바인딩

### Phase 3: 사용성 폴리싱 (Usability Polish)
5. ⬜ 시각적 피드백 추가
   - 크기 불일치 경고 배지
   - 품질 인디케이터
6. ⬜ 컨텍스트 메뉴 및 툴팁
   - 우클릭 메뉴
   - 상세 툴팁

## 테스트 시나리오 (Test Scenarios)

### Test 1: AI 마스킹 크기 검증
```
1. 앞면 이미지 선택 (예: 1024x768)
2. AI 배경제거 실행
3. 검증:
   - 마스크 배열 크기 = 1024x768 ✓
   - 저장된 파일 크기 = 1024x768 ✓
   - 오버레이 미리보기 정렬 정확 ✓
```

### Test 2: 수동 마스킹 크기 조정
```
1. 앞면 이미지 선택 (800x600)
2. 다른 크기 마스킹 업로드 (1000x750)
3. 크기 불일치 경고 표시 ✓
4. 자동 리사이즈 확인 선택
5. 검증:
   - 마스크 크기 = 800x600으로 조정 ✓
   - 오버레이 정렬 정확 ✓
```

### Test 3: 미리보기 모드 전환
```
1. 마스킹 처리 완료된 이미지
2. "원본" 버튼 클릭 → 원본 이미지만 표시 ✓
3. "마스크" 버튼 클릭 → 마스크만 표시 ✓
4. "겹침" 버튼 클릭 → 녹색 오버레이 표시 ✓
5. V 키로 순환 토글 동작 ✓
```

### Test 4: 레이어 인쇄 검증
```
1. 레이어 인쇄 모드 선택
2. 마스킹 처리된 이미지로 인쇄
3. 검증:
   - 인쇄 시 마스크와 원본 정렬 정확 ✓
   - 크기 불일치 없음 ✓
```

## 성공 지표 (Success Metrics)

1. **기술적 정확성**
   - 마스킹 크기 불일치 발생률: 0%
   - 인쇄 정렬 오류율: 0%

2. **사용자 만족도**
   - 마스킹 결과 확인 용이성: ⭐⭐⭐⭐⭐
   - 수동 마스킹 워크플로우 직관성: ⭐⭐⭐⭐⭐
   - 전체 작업 완료 시간: 30% 단축

3. **안정성**
   - 크기 관련 에러 발생률: 0%
   - 인쇄 실패율: < 1%

## 결론 (Conclusion)

이 설계는 다음 세 가지 핵심 문제를 해결합니다:

1. **마스킹 크기 일관성**: AI 처리 및 저장 시 강제 검증으로 100% 보장
2. **시각적 검증**: 오버레이 미리보기로 사용자가 즉시 확인 가능
3. **수동 조정 워크플로우**: 크기 자동 조정 및 간편한 업로드 메뉴

**사용자 관점에서의 개선**:
- 마스킹 처리 후 즉시 결과 확인 (원본/마스크/겹침 토글)
- 크기 불일치 시 자동 감지 및 조정 제안
- 메뉴와 단축키로 빠른 수동 마스킹 업로드
- 레이어 인쇄 최종 결과 예측 가능

**가장 중요한 원칙**:
> "사용자가 쉽게 써야 한다" - 복잡한 기술을 간단한 UI/UX로 숨기고,
> 시각적 피드백으로 명확한 확인을 제공합니다.
