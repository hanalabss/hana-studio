# AI 배경제거 및 마스킹 시스템 개선 설계

## 📋 목차
1. [현재 상황 분석](#1-현재-상황-분석)
2. [문제점 정의](#2-문제점-정의)
3. [핵심 요구사항](#3-핵심-요구사항)
4. [시스템 아키텍처](#4-시스템-아키텍처)
5. [상세 설계](#5-상세-설계)
6. [UI/UX 설계](#6-uiux-설계)
7. [구현 계획](#7-구현-계획)

---

## 1. 현재 상황 분석

### 1.1 현재 구현된 기능

**core/image_processor.py**
- `remove_background()`: AI 배경 제거 및 마스크 생성
  - ✅ EXIF 정보 보존
  - ✅ 원본 크기 유지 로직 존재
  - ✅ 크기 불일치 시 자동 리사이즈 (LANCZOS)
  - ✅ 알파 임계값 기반 마스크 생성

**core/file_manager.py**
- `save_mask_for_printing()`: 프린터용 마스크 저장
  - ✅ 원본 이미지 크기 검증
  - ✅ 크기 불일치 시 강제 리사이즈
  - ✅ 저장 후 재검증
  - ✅ 한글 경로 지원

**ui_v2/dialogs/masking_dialog.py**
- `MaskingDialog`: 마스킹 미리보기 및 임계값 조정
  - ✅ 원본/마스크 병렬 표시
  - ✅ 임계값 슬라이더 조정
  - ✅ 재처리 기능
  - ✅ 비동기 처리 (Worker 스레드)

**hana_studio.py**
- 양면 이미지 지원
  - ✅ front/back 이미지 분리
  - ✅ auto_mask_image (자동 마스킹)
  - ✅ manual_mask (수동 마스킹) 구조 준비됨
  - ✅ saved_mask_path (프린터용 저장 경로)

### 1.2 코드 품질 평가

**강점:**
- 크기 검증 로직이 여러 단계에서 존재
- EXIF 정보 처리 명확
- 비동기 처리로 UI 블로킹 방지
- 한글 경로 지원

**약점:**
- 마스킹 크기가 커지는 문제의 원인이 명확하지 않음
- 사용자가 마스킹 결과를 확인하기 어려움
- 수동 마스킹 업로드 기능 미구현
- 원본과 마스크 오버레이 표시 기능 없음

---

## 2. 문제점 정의

### 2.1 마스킹 크기 문제

**증상:**
> "AI 배경제거를 하고 적용하면 마스킹 크기가 엄청 커진다."

**잠재적 원인 분석:**

1. **AI 모델(rembg) 출력 크기 불일치**
   ```python
   # image_processor.py:66-68
   img_rgba = Image.open(io.BytesIO(result)).convert("RGBA")
   result_width, result_height = img_rgba.size
   ```
   - rembg가 원본과 다른 크기로 출력할 가능성
   - 현재 코드는 이를 감지하고 리사이즈하지만, 어딘가에서 검증 누락

2. **프린터 적용 시점의 크기 불일치**
   ```python
   # file_manager.py:171-209
   def save_mask_for_printing(self, mask_image: np.ndarray, ...)
   ```
   - 프린터로 전달될 때 다시 크기가 변경될 가능성
   - 저장된 이미지 경로 vs 실제 적용 시점의 차이

3. **UI 표시 vs 실제 데이터 크기 불일치**
   - `MaskingDialog`에서 미리보기는 스케일링됨 (400x300)
   - 실제 저장되는 데이터는 원본 크기
   - 사용자가 보는 것과 저장되는 것의 불일치

### 2.2 사용자 경험 문제

**문제:**
1. **마스킹 확인 어려움**
   - 원본과 마스크를 별도로 봐야 함
   - 겹쳐서 보는 기능 없음
   - 마스킹이 정확한지 판단 어려움

2. **AI 마스킹 불만족 시 대안 없음**
   - AI가 실패하면 다시 시도만 가능
   - 수동으로 포토샵 등에서 만든 마스크 업로드 불가

3. **레이아웃 인쇄 동작 이해 어려움**
   - 마스킹 처리 인쇄 + 원본 덮기 구조가 명확하지 않음

---

## 3. 핵심 요구사항

### 3.1 기능 요구사항

| 우선순위 | 요구사항 | 설명 |
|---------|---------|------|
| 🔴 필수 | 마스킹 크기 정확성 보장 | 저장했던 이미지 크기대로 마스킹 적용 |
| 🔴 필수 | 원본-마스크 오버레이 표시 | 겹쳐서 보여주기 (투명도 조절) |
| 🔴 필수 | 수동 마스킹 업로드 | AI 불만족 시 직접 만든 마스크 사용 |
| 🟡 중요 | 레이아웃 인쇄 시각화 | 마스킹 + 원본 오버레이 구조 명확히 표시 |
| 🟡 중요 | 실시간 미리보기 | 마스킹 결과를 즉시 확인 가능 |
| 🟢 선택 | 마스킹 편집 도구 | 간단한 브러시로 마스크 수정 |

### 3.2 사용자 경험 원칙

**"가장 중요한 건 사용자가 쉽게 써야 한다"**

1. **직관성**: 마스킹 결과를 한눈에 확인
2. **신뢰성**: 보이는 대로 저장되고 인쇄됨
3. **유연성**: AI + 수동 마스킹 모두 지원
4. **피드백**: 각 단계마다 명확한 시각적 피드백

---

## 4. 시스템 아키텍처

### 4.1 마스킹 데이터 플로우

```
[원본 이미지]
    ↓
[AI 배경제거] ← rembg
    ↓
[크기 검증 & 리사이즈] ← image_processor.py
    ↓
[마스크 생성] (np.ndarray)
    ↓
    ├─→ [자동 마스크 저장] → auto_mask_image
    └─→ [수동 마스크 업로드] → manual_mask_image
            ↓
    [마스크 선택 로직] (auto vs manual)
            ↓
    [크기 재검증] ← 원본 크기와 비교
            ↓
    [프린터용 저장] → saved_mask_path
            ↓
    [레이아웃 인쇄]
        ├─ 1단계: 마스킹 이미지 인쇄 (검은색 부분만)
        └─ 2단계: 원본 이미지 오버레이
```

### 4.2 컴포넌트 구조

```
ImageProcessor (core/image_processor.py)
├─ remove_background() : AI 마스킹
├─ validate_mask_size() : NEW - 크기 검증
└─ create_overlay_preview() : NEW - 오버레이 미리보기

FileManager (core/file_manager.py)
├─ save_mask_for_printing() : 프린터용 저장 (개선)
├─ load_manual_mask() : NEW - 수동 마스크 로드
└─ validate_manual_mask() : NEW - 수동 마스크 검증

MaskingDialog (ui_v2/dialogs/masking_dialog.py)
├─ _setup_ui() : UI 구성 (개선)
│   ├─ 원본 미리보기
│   ├─ 마스크 미리보기
│   └─ 오버레이 미리보기 (NEW)
├─ _toggle_overlay_mode() : NEW - 오버레이 토글
└─ _apply_mask() : 마스크 적용 (개선)

ImageViewer (ui/components/image_viewer.py)
├─ show_overlay_preview() : NEW - 오버레이 표시
├─ upload_manual_mask() : NEW - 수동 마스크 업로드
└─ switch_mask_mode() : NEW - auto/manual 전환
```

---

## 5. 상세 설계

### 5.1 마스킹 크기 정확성 보장

#### 5.1.1 크기 검증 강화

**새로운 검증 함수 추가 (ImageProcessor)**
```python
def validate_mask_size(self, mask: np.ndarray, original_path: str) -> Tuple[bool, str]:
    """
    마스크 크기를 원본과 비교하여 검증

    Returns:
        (is_valid, message)
    """
    # 원본 이미지 크기 확인
    with open(original_path, 'rb') as f:
        original_data = f.read()
    original_pil = Image.open(io.BytesIO(original_data))
    orig_w, orig_h = original_pil.size

    # 마스크 크기 확인
    mask_h, mask_w = mask.shape[:2]

    # 크기 일치 검증
    if mask_w != orig_w or mask_h != orig_h:
        return False, f"크기 불일치: 원본({orig_w}x{orig_h}) vs 마스크({mask_w}x{mask_h})"

    return True, "크기 일치 확인됨"
```

#### 5.1.2 모든 저장 시점에 검증 추가

**save_mask_for_printing() 개선**
```python
def save_mask_for_printing(self, mask_image: np.ndarray, original_image_path: str, ...) -> Optional[str]:
    # === 1. 저장 전 크기 검증 (필수) ===
    is_valid, msg = self.validate_mask_size(mask_image, original_image_path)
    if not is_valid:
        print(f"[ERROR] {msg}")
        # 강제 리사이즈 시도
        mask_image = self._force_resize_to_original(mask_image, original_image_path)

    # === 2. 저장 ===
    # ... (기존 로직)

    # === 3. 저장 후 재검증 (필수) ===
    saved_mask = self._safe_imread(mask_path)
    is_valid, msg = self.validate_mask_size(saved_mask, original_image_path)
    if not is_valid:
        print(f"[CRITICAL] 저장 후 크기 불일치: {msg}")
        return None

    return mask_path
```

### 5.2 원본-마스크 오버레이 표시

#### 5.2.1 오버레이 생성 함수

**ImageProcessor에 추가**
```python
def create_overlay_preview(self, original_image: np.ndarray, mask_image: np.ndarray,
                          opacity: float = 0.5, mode: str = "color") -> np.ndarray:
    """
    원본과 마스크를 겹쳐서 표시

    Args:
        original_image: 원본 이미지 (BGR)
        mask_image: 마스크 이미지 (흰색=배경, 검은색=객체)
        opacity: 마스크 투명도 (0.0~1.0)
        mode: "color" = 컬러 오버레이, "red" = 빨간색 하이라이트

    Returns:
        오버레이 이미지
    """
    # 크기 일치 확인
    if original_image.shape[:2] != mask_image.shape[:2]:
        mask_image = cv2.resize(mask_image,
                               (original_image.shape[1], original_image.shape[0]),
                               interpolation=cv2.INTER_LANCZOS4)

    overlay = original_image.copy()

    if mode == "color":
        # 컬러맵 적용 (JET)
        mask_colored = cv2.applyColorMap(mask_image, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(original_image, 1.0 - opacity, mask_colored, opacity, 0)

    elif mode == "red":
        # 빨간색 하이라이트 (객체 영역만)
        red_overlay = original_image.copy()
        red_overlay[:, :, 2] = np.maximum(red_overlay[:, :, 2], 200)  # R 채널 강조

        # 마스크에서 객체 영역만 선택 (검은색 = 객체)
        object_mask = (mask_image[:, :, 0] < 128).astype(np.uint8)
        object_mask_3ch = np.stack([object_mask, object_mask, object_mask], axis=2)

        overlay = np.where(object_mask_3ch, red_overlay, original_image)
        overlay = cv2.addWeighted(overlay, 1.0, original_image, 0.0, 0)

    elif mode == "outline":
        # 경계선만 표시
        gray_mask = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_mask, 50, 150)
        overlay = original_image.copy()
        overlay[edges > 0] = [0, 0, 255]  # 빨간색 경계선

    return overlay
```

#### 5.2.2 MaskingDialog에 오버레이 탭 추가

```python
class MaskingDialog(QDialog):
    def _setup_ui(self):
        # === 탭 위젯 추가 ===
        self.tab_widget = QTabWidget()

        # 탭 1: 원본 vs 마스크 (기존)
        self.separate_tab = self._create_separate_view()
        self.tab_widget.addTab(self.separate_tab, "원본 / 마스크")

        # 탭 2: 오버레이 뷰 (NEW)
        self.overlay_tab = self._create_overlay_view()
        self.tab_widget.addTab(self.overlay_tab, "오버레이 미리보기")

        layout.addWidget(self.tab_widget)

    def _create_overlay_view(self):
        """오버레이 뷰 생성"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # 오버레이 이미지 표시
        self.overlay_preview = QLabel()
        self.overlay_preview.setMinimumSize(800, 600)
        self.overlay_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.overlay_preview.setStyleSheet("border: 1px solid #ccc; background: white;")
        layout.addWidget(self.overlay_preview)

        # 오버레이 모드 선택
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("표시 모드:"))

        self.overlay_mode_combo = QComboBox()
        self.overlay_mode_combo.addItems(["컬러 오버레이", "빨간색 하이라이트", "경계선만"])
        self.overlay_mode_combo.currentTextChanged.connect(self._update_overlay_preview)
        mode_layout.addWidget(self.overlay_mode_combo)

        # 투명도 슬라이더
        mode_layout.addWidget(QLabel("투명도:"))
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(50)
        self.opacity_slider.valueChanged.connect(self._update_overlay_preview)
        mode_layout.addWidget(self.opacity_slider)

        layout.addLayout(mode_layout)

        return widget

    def _update_overlay_preview(self):
        """오버레이 미리보기 업데이트"""
        if self.current_mask_rgb is None or self.original_image is None:
            return

        # 모드 맵핑
        mode_map = {
            "컬러 오버레이": "color",
            "빨간색 하이라이트": "red",
            "경계선만": "outline"
        }
        mode = mode_map[self.overlay_mode_combo.currentText()]
        opacity = self.opacity_slider.value() / 100.0

        # 오버레이 생성
        processor = ImageProcessor()
        overlay = processor.create_overlay_preview(
            self.original_image,
            self.current_mask_rgb,
            opacity=opacity,
            mode=mode
        )

        # QPixmap으로 변환하여 표시
        height, width = overlay.shape[:2]
        q_image = QImage(overlay.data, width, height, 3 * width, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)

        self.overlay_preview.setPixmap(pixmap.scaled(
            800, 600,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))
```

### 5.3 수동 마스킹 업로드 기능

#### 5.3.1 수동 마스크 로드 및 검증

**FileManager에 추가**
```python
def load_manual_mask(self, mask_path: str, original_path: str) -> Tuple[Optional[np.ndarray], str]:
    """
    수동 마스크 이미지 로드 및 검증

    Args:
        mask_path: 수동 마스크 이미지 경로
        original_path: 원본 이미지 경로 (크기 비교용)

    Returns:
        (mask_image or None, message)
    """
    try:
        # 1. 마스크 이미지 로드
        mask_image = self._safe_imread(mask_path)
        if mask_image is None:
            return None, "마스크 이미지를 읽을 수 없습니다."

        # 2. 원본 이미지 크기 확인
        original_image = self._safe_imread(original_path)
        if original_image is None:
            return None, "원본 이미지를 읽을 수 없습니다."

        orig_h, orig_w = original_image.shape[:2]
        mask_h, mask_w = mask_image.shape[:2]

        # 3. 크기 검증
        if mask_w != orig_w or mask_h != orig_h:
            print(f"[MANUAL MASK] 크기 불일치 감지: 원본({orig_w}x{orig_h}) vs 마스크({mask_w}x{mask_h})")
            print(f"[MANUAL MASK] 자동 리사이즈 수행...")

            mask_image = cv2.resize(
                mask_image,
                (orig_w, orig_h),
                interpolation=cv2.INTER_LANCZOS4
            )

            return mask_image, f"크기가 자동 조정되었습니다 ({mask_w}x{mask_h} → {orig_w}x{orig_h})"

        # 4. 흑백 검증 (선택적)
        is_grayscale = self._check_if_grayscale(mask_image)
        if not is_grayscale:
            print("[MANUAL MASK] 컬러 이미지 감지 - 흑백 변환 필요할 수 있음")
            return mask_image, "컬러 마스크가 업로드되었습니다. 흰색=배경, 검은색=객체로 처리됩니다."

        return mask_image, "수동 마스크가 성공적으로 로드되었습니다."

    except Exception as e:
        return None, f"마스크 로드 실패: {e}"

def _check_if_grayscale(self, image: np.ndarray) -> bool:
    """이미지가 흑백인지 확인"""
    if len(image.shape) == 2:
        return True

    # BGR 채널이 모두 동일한지 확인
    b, g, r = cv2.split(image)
    return np.array_equal(b, g) and np.array_equal(g, r)
```

#### 5.3.2 ImageViewer에 수동 업로드 버튼 추가

```python
class ImageViewer(QWidget):
    manual_mask_uploaded = Signal(str)  # NEW 시그널

    def _setup_controls(self):
        # ... (기존 컨트롤들)

        # === 수동 마스크 업로드 버튼 추가 ===
        self.manual_mask_btn = QPushButton("수동 마스크 업로드")
        self.manual_mask_btn.setFixedSize(140, 36)
        self.manual_mask_btn.setStyleSheet("""
            QPushButton {
                background-color: #9C27B0;
                color: white;
                border-radius: 5px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #AB47BC;
            }
        """)
        self.manual_mask_btn.clicked.connect(self._upload_manual_mask)
        self.manual_mask_btn.setEnabled(False)

        controls_layout.addWidget(self.manual_mask_btn)

    def _upload_manual_mask(self):
        """수동 마스크 업로드"""
        if not self.current_image_path:
            QMessageBox.warning(self, "경고", "먼저 원본 이미지를 업로드하세요.")
            return

        # 파일 선택 다이얼로그
        mask_path, _ = QFileDialog.getOpenFileName(
            self,
            "수동 마스크 이미지 선택",
            "",
            "이미지 파일 (*.jpg *.jpeg *.png *.bmp)"
        )

        if not mask_path:
            return

        # FileManager로 검증
        from core.file_manager import FileManager
        file_mgr = FileManager()

        mask_image, message = file_mgr.load_manual_mask(mask_path, self.current_image_path)

        if mask_image is None:
            QMessageBox.critical(self, "오류", message)
            return

        # 성공 시 미리보기 표시
        QMessageBox.information(self, "성공", message)

        # 마스크 이미지 표시
        self._display_mask(mask_image)

        # 시그널 발생
        self.manual_mask_uploaded.emit(mask_path)
```

### 5.4 레이아웃 인쇄 시스템

#### 5.4.1 인쇄 프로세스 시각화

**PrinterManager 개선 (새로운 메서드 추가)**
```python
def generate_print_preview(self,
                          original_path: str,
                          mask_path: str,
                          show_layers: bool = True) -> np.ndarray:
    """
    레이아웃 인쇄 미리보기 생성

    Args:
        original_path: 원본 이미지 경로
        mask_path: 마스크 이미지 경로
        show_layers: 레이어 구조 표시 여부

    Returns:
        미리보기 이미지 (2단계 합성)
    """
    from core.file_manager import FileManager
    file_mgr = FileManager()

    # 이미지 로드
    original = file_mgr._safe_imread(original_path)
    mask = file_mgr._safe_imread(mask_path)

    if original is None or mask is None:
        raise ValueError("이미지 로드 실패")

    # 크기 검증
    if original.shape[:2] != mask.shape[:2]:
        raise ValueError("원본과 마스크 크기 불일치")

    if not show_layers:
        # 단순 합성
        return self._create_simple_composite(original, mask)

    # 레이어 구조 표시 (3단계)
    h, w = original.shape[:2]

    # 캔버스 생성 (가로로 3개 배치 + 설명 공간)
    canvas_h = h + 100  # 상단 설명 공간
    canvas_w = w * 3 + 100  # 3개 이미지 + 간격
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

    # 1단계: 마스크 인쇄 (검은색 부분만)
    step1 = self._create_mask_print_preview(mask)
    canvas[100:100+h, 20:20+w] = step1
    cv2.putText(canvas, "Step 1: Mask Print", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(canvas, "(Black area only)", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)

    # 2단계: 원본 오버레이
    step2 = original.copy()
    canvas[100:100+h, 40+w:40+w+w] = step2
    cv2.putText(canvas, "Step 2: Original Overlay", (40+w, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # 3단계: 최종 결과
    step3 = self._create_final_composite(original, mask)
    canvas[100:100+h, 60+w*2:60+w*2+w] = step3
    cv2.putText(canvas, "Step 3: Final Result", (60+w*2, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(canvas, "(Mask + Original)", (60+w*2, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)

    return canvas

def _create_mask_print_preview(self, mask: np.ndarray) -> np.ndarray:
    """마스크 인쇄 시뮬레이션 (검은색 부분만)"""
    # 흰색 배경 생성
    white_bg = np.ones_like(mask) * 255

    # 마스크의 검은색 부분만 표시
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    object_mask = (gray_mask < 128)

    result = white_bg.copy()
    result[object_mask] = [0, 0, 0]  # 검은색

    return result

def _create_final_composite(self, original: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """최종 합성 이미지 (마스크 위에 원본 오버레이)"""
    # 마스크의 객체 영역 추출
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    object_mask = (gray_mask < 128)

    # 흰색 배경
    result = np.ones_like(original) * 255

    # 객체 영역에만 원본 이미지 합성
    result[object_mask] = original[object_mask]

    return result
```

---

## 6. UI/UX 설계

### 6.1 ImageViewer 개선안

#### 6.1.1 레이아웃 구조

```
┌────────────────────────────────────────────────┐
│ [이미지 미리보기 영역]                            │
│                                                │
│  ┌──────────────────────────────────────┐      │
│  │                                      │      │
│  │        [원본 또는 오버레이 표시]         │      │
│  │                                      │      │
│  └──────────────────────────────────────┘      │
│                                                │
│ [표시 모드 선택]                                 │
│  ( ) 원본만  ( ) 마스크만  (●) 오버레이          │
│                                                │
│ [마스킹 방법 선택]                               │
│  (●) AI 자동 마스킹                              │
│      [배경 제거 실행] [임계값 조정]                │
│  ( ) 수동 마스크 업로드                          │
│      [파일 선택...]                              │
│                                                │
│ [방향 설정]                                     │
│  (●) 세로  ( ) 가로                             │
└────────────────────────────────────────────────┘
```

#### 6.1.2 워크플로우

**시나리오 1: AI 마스킹 사용**
1. 사용자가 이미지 업로드
2. "AI 자동 마스킹" 선택 (기본값)
3. "배경 제거 실행" 클릭
4. MaskingDialog 표시
   - 원본 / 마스크 탭에서 결과 확인
   - 오버레이 탭에서 정확도 확인
   - 임계값 조정 가능
5. "적용" 클릭
6. ImageViewer로 돌아와서 오버레이 표시
7. 만족하면 "인쇄" 진행

**시나리오 2: 수동 마스킹 사용**
1. 사용자가 이미지 업로드
2. AI 마스킹 시도 후 불만족
3. "수동 마스크 업로드" 라디오 버튼 선택
4. "파일 선택..." 버튼 클릭
5. 포토샵 등에서 만든 마스크 이미지 선택
6. 자동 크기 검증 및 조정
7. 오버레이로 결과 확인
8. 만족하면 "인쇄" 진행

### 6.2 MaskingDialog 개선안

#### 6.2.1 탭 구조

```
┌────────────────────────────────────────────────────────┐
│ [원본 / 마스크] [오버레이 미리보기] [레이아웃 인쇄 시뮬레이션] │
├────────────────────────────────────────────────────────┤
│                                                        │
│  [탭 내용 영역]                                         │
│                                                        │
│  • 원본 / 마스크: 좌우 비교                              │
│  • 오버레이: 겹친 상태 + 투명도 조절                      │
│  • 레이아웃 인쇄: 3단계 프로세스 시각화                   │
│                                                        │
├────────────────────────────────────────────────────────┤
│ 임계값: [슬라이더] 45  [배경 재처리]                      │
├────────────────────────────────────────────────────────┤
│                          [취소] [적용]                   │
└────────────────────────────────────────────────────────┘
```

#### 6.2.2 시각적 피드백

**크기 일치 상태 표시**
```
✅ 크기 일치 (1200 x 1800)
⚠️ 크기 자동 조정됨 (1000x1500 → 1200x1800)
❌ 크기 불일치 - 적용 불가
```

**마스킹 품질 지표**
```
객체 영역: 18.5% (적정)
배경 영역: 81.5%
추천 임계값: 45 (현재: 50)
```

### 6.3 접근성 개선

**색맹 사용자 고려**
- 오버레이 모드에 "경계선만" 옵션 제공
- 흑백 대비만으로도 마스킹 확인 가능

**키보드 단축키**
- `Tab`: 탭 전환
- `Space`: 오버레이 토글
- `←/→`: 임계값 조절
- `Enter`: 적용
- `Esc`: 취소

---

## 7. 구현 계획

### 7.1 우선순위별 단계

#### Phase 1: 크기 정확성 보장 (필수)
**목표**: 마스킹 크기 문제 완전 해결

- [ ] `validate_mask_size()` 함수 구현
- [ ] `save_mask_for_printing()` 검증 강화
- [ ] 모든 저장/로드 시점에 크기 검증 추가
- [ ] 단위 테스트 작성
- [ ] 실제 데이터로 검증

**예상 시간**: 2-3시간

#### Phase 2: 오버레이 표시 (필수)
**목표**: 사용자가 마스킹 결과를 명확히 확인

- [ ] `create_overlay_preview()` 함수 구현
- [ ] MaskingDialog에 오버레이 탭 추가
- [ ] ImageViewer에 오버레이 표시 모드 추가
- [ ] 투명도/모드 조절 UI 구현

**예상 시간**: 3-4시간

#### Phase 3: 수동 마스킹 업로드 (필수)
**목표**: AI 실패 시 대안 제공

- [ ] `load_manual_mask()` 함수 구현
- [ ] ImageViewer에 업로드 버튼 추가
- [ ] 마스킹 방법 선택 라디오 버튼 (AI/수동)
- [ ] 수동 마스크 검증 로직
- [ ] 통합 테스트

**예상 시간**: 2-3시간

#### Phase 4: 레이아웃 인쇄 시각화 (중요)
**목표**: 인쇄 프로세스 이해도 향상

- [ ] `generate_print_preview()` 함수 구현
- [ ] MaskingDialog에 "레이아웃 인쇄 시뮬레이션" 탭 추가
- [ ] 3단계 프로세스 시각화
- [ ] 사용자 가이드 추가

**예상 시간**: 2-3시간

### 7.2 테스트 계획

#### 7.2.1 단위 테스트

```python
# tests/test_masking_system.py

class TestMaskingSizeValidation:
    def test_validate_mask_size_match(self):
        """크기 일치 시 검증 통과"""

    def test_validate_mask_size_mismatch(self):
        """크기 불일치 시 검증 실패"""

    def test_auto_resize_on_save(self):
        """저장 시 자동 리사이즈"""

class TestOverlayGeneration:
    def test_create_color_overlay(self):
        """컬러 오버레이 생성"""

    def test_create_red_highlight(self):
        """빨간색 하이라이트 생성"""

    def test_create_outline_mode(self):
        """경계선 모드 생성"""

class TestManualMaskUpload:
    def test_load_valid_mask(self):
        """유효한 수동 마스크 로드"""

    def test_load_different_size_mask(self):
        """크기 다른 마스크 자동 조정"""

    def test_load_color_mask(self):
        """컬러 마스크 처리"""
```

#### 7.2.2 통합 테스트

**시나리오 테스트**
1. AI 마스킹 → 오버레이 확인 → 적용 → 인쇄
2. AI 마스킹 → 불만족 → 수동 업로드 → 적용 → 인쇄
3. 양면 카드 (앞면 AI, 뒷면 수동) → 인쇄

**엣지 케이스**
- 매우 큰 이미지 (5000x5000)
- 매우 작은 이미지 (100x100)
- 한글 경로 이미지
- 회전된 EXIF 이미지
- 다양한 포맷 (JPG, PNG, BMP)

### 7.3 배포 전 체크리스트

- [ ] 모든 크기 검증 로직 작동 확인
- [ ] 오버레이 표시 정상 작동
- [ ] 수동 마스킹 업로드 정상 작동
- [ ] 양면 카드 시나리오 테스트
- [ ] 한글 경로 지원 확인
- [ ] 성능 테스트 (1000장 배치 처리)
- [ ] 사용자 가이드 문서 작성
- [ ] 릴리즈 노트 작성

---

## 8. 리스크 관리

### 8.1 기술적 리스크

| 리스크 | 확률 | 영향 | 완화 전략 |
|-------|-----|-----|---------|
| rembg 출력 크기 예측 불가 | 중 | 높음 | 강력한 검증 + 강제 리사이즈 |
| 오버레이 렌더링 성능 저하 | 낮 | 중 | 비동기 처리 + 캐싱 |
| 수동 마스크 형식 호환성 | 중 | 중 | 다양한 포맷 지원 + 검증 |

### 8.2 사용자 경험 리스크

| 리스크 | 확률 | 영향 | 완화 전략 |
|-------|-----|-----|---------|
| 오버레이 이해 어려움 | 낮 | 중 | 명확한 레이블 + 툴팁 |
| 수동 마스크 생성 방법 모름 | 높음 | 중 | 가이드 문서 + 샘플 제공 |
| 인쇄 결과 예상과 다름 | 중 | 높음 | 레이아웃 시뮬레이션 탭 |

---

## 9. 성공 지표

### 9.1 기술 지표

- ✅ 마스킹 크기 일치율: **100%**
- ✅ 크기 검증 누락 케이스: **0건**
- ✅ 수동 마스크 호환률: **>95%**

### 9.2 사용자 경험 지표

- ✅ 오버레이 기능 사용률: **>80%**
- ✅ 수동 마스킹 사용률: **>20%**
- ✅ 마스킹 관련 오류 신고: **<5%**

---

## 10. 결론

### 10.1 핵심 개선 사항

1. **크기 정확성 보장**: 다단계 검증으로 100% 일치 보장
2. **시각적 피드백 강화**: 오버레이 표시로 마스킹 품질 즉시 확인
3. **유연성 증대**: AI + 수동 마스킹 하이브리드 지원
4. **사용자 이해도 향상**: 레이아웃 인쇄 프로세스 시각화

### 10.2 사용자 가치

> "이제 사용자는 마스킹 결과를 눈으로 확인하고, AI가 실패하면 직접 만든 마스크를 사용할 수 있으며, 인쇄될 모습을 미리 볼 수 있습니다. 모든 것이 투명하고, 예측 가능하며, 쉽게 사용할 수 있습니다."

### 10.3 다음 단계

이 설계 문서를 바탕으로:
1. Phase 1부터 순차적으로 구현
2. 각 Phase마다 테스트 및 검증
3. 사용자 피드백 수집 및 개선
4. 최종 통합 테스트 및 배포

---

**문서 버전**: 2.0
**작성일**: 2025-01-24
**작성자**: Claude Code
**상태**: 설계 완료, 구현 대기
