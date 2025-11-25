# 캔버스 UX 개선 설계 - 인쇄 모드별 미리보기 시스템

## 📋 설계 목표

사용자 요구사항:
1. **배경제거 적용 시**: 캔버스에 레이아웃 인쇄 시뮬레이션 표시
2. **일반인쇄 모드**: ORIGINAL 이미지가 인쇄됨
3. **레이아웃인쇄 모드**: MASK 선인쇄 → ORIGINAL 인쇄
4. **FINAL_RESULT**: 사용자 비교용 (인쇄 결과 미리보기)

**핵심 UX 원칙**: **캔버스에 표시되는 것 = 실제 인쇄되는 것** (WYSIWYG)

---

## 🎯 현재 시스템 분석

### 현재 구조 (ui_v2/)

#### 1. 캔버스 표시 시스템
- **ImageLayer** (`canvas/layers/image_layer.py`)
  - `self.original_pixmap`: 원본 이미지
  - `self.pixmap`: 캔버스에 실제 표시되는 이미지
  - 현재는 배경제거 시 FINAL_RESULT(합성 이미지)를 `pixmap`에 설정

#### 2. 인쇄 시스템
- **_start_printing()** (`ui_v2/main_window.py:620`)
  - 일반인쇄: ORIGINAL만 인쇄
  - 레이아웃인쇄: MASK 선인쇄 → ORIGINAL 인쇄

#### 3. 배경제거 적용 흐름
```
배경제거 완료
  ↓
_on_masking_applied() / _on_manual_masking_applied()
  ↓
자동으로 레이아웃 인쇄 모드 전환
  ↓
FINAL_RESULT 생성 (배경 반투명 + 마스킹 불투명)
  ↓
캔버스에 FINAL_RESULT 표시 (self.selected_layer.pixmap)
```

### 현재 문제점

#### ❌ 문제 1: 캔버스 표시와 인쇄 결과 불일치
- **캔버스**: FINAL_RESULT (배경 반투명 합성 이미지) 표시
- **실제 인쇄**: MASK 선인쇄 → ORIGINAL 인쇄 (합성 이미지 아님)
- **결과**: 사용자가 보는 것과 인쇄되는 것이 다름

#### ❌ 문제 2: 모드 전환 시 캔버스 업데이트 없음
- 레이아웃 인쇄 → 일반 인쇄로 전환해도 캔버스는 FINAL_RESULT 유지
- 일반 인쇄는 ORIGINAL을 인쇄하므로 불일치

#### ❌ 문제 3: FINAL_RESULT의 역할 모호
- FINAL_RESULT는 비교용이어야 하는데 캔버스에 표시됨
- 사용자가 실제 인쇄 결과를 예측하기 어려움

---

## 🎨 새로운 UX 설계

### 설계 원칙

**원칙 1: 캔버스 = 실제 인쇄 미리보기**
- 캔버스에 표시되는 것이 정확히 인쇄되는 것이어야 함
- 인쇄 모드에 따라 캔버스 표시 동적 변경

**원칙 2: 모드별 명확한 시각적 피드백**
- 일반인쇄: ORIGINAL 그대로 표시
- 레이아웃인쇄: 레이아웃 인쇄 시뮬레이션 표시

**원칙 3: FINAL_RESULT는 비교 전용**
- FINAL_RESULT는 별도 미리보기 패널에 표시
- 사용자가 원본/마스크/결과를 비교할 수 있게 함

---

## 🏗️ 새로운 시스템 아키텍처

### 1. 이미지 상태 관리 구조

```python
# ImageLayer 확장 속성
class ImageLayer:
    # 기본 이미지
    self.original_pixmap        # 원본 이미지 (불변)
    self.mask_pixmap           # 마스크 이미지 (배경제거 시 생성)

    # 캔버스 표시용
    self.pixmap                # 캔버스에 실제 표시되는 이미지
                               # 인쇄 모드에 따라 동적 변경

    # 비교용 미리보기
    self.final_result_pixmap   # 최종 합성 결과 (비교 전용)
```

### 2. 캔버스 표시 로직

#### A. 일반 인쇄 모드
```python
def update_canvas_for_normal_print():
    """일반 인쇄 모드: ORIGINAL 표시"""
    self.selected_layer.pixmap = self.selected_layer.original_pixmap.copy()
    self.selected_layer.update()
```

**표시**: ORIGINAL 이미지 그대로
**인쇄**: ORIGINAL 이미지 인쇄
**일치**: ✅ 100% 일치

#### B. 레이아웃 인쇄 모드 (배경제거 적용 전)
```python
def update_canvas_for_layered_print_no_mask():
    """레이아웃 인쇄 모드 (마스크 없음): ORIGINAL 표시"""
    self.selected_layer.pixmap = self.selected_layer.original_pixmap.copy()
    self.selected_layer.update()
```

**표시**: ORIGINAL 이미지 그대로
**인쇄**: ORIGINAL 이미지만 인쇄 (마스크 없으므로)
**일치**: ✅ 100% 일치

#### C. 레이아웃 인쇄 모드 (배경제거 적용 후)
```python
def update_canvas_for_layered_print_with_mask():
    """레이아웃 인쇄 모드 (마스크 있음): 시뮬레이션 표시"""
    # 레이아웃 인쇄 시뮬레이션 이미지 생성
    simulation = create_layered_print_simulation(
        original=self.selected_layer.original_pixmap,
        mask=self.selected_layer.mask_pixmap
    )
    self.selected_layer.pixmap = simulation
    self.selected_layer.update()

def create_layered_print_simulation(original, mask):
    """실제 레이아웃 인쇄 결과 시뮬레이션"""
    # 1. MASK의 검은색 영역 = W 리본 인쇄 영역
    # 2. ORIGINAL 전체에 YMC 리본 인쇄
    # 3. 결과: W 리본 영역은 불투명, 나머지는 반투명

    result = original.copy()

    # 마스크 검은색 영역 찾기
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    w_layer_mask = (gray_mask < 128)  # 검은색 = W 리본 영역

    # 배경 영역(W 리본 없는 곳)은 반투명 효과 적용
    # (실제로는 YMC만 인쇄되어 반투명하게 보임)
    background_mask = ~w_layer_mask
    result[background_mask] = (result[background_mask] * 0.3 + 255 * 0.7)

    return result
```

**표시**: 레이아웃 인쇄 시뮬레이션 (배경 반투명)
**인쇄**: MASK 선인쇄 → ORIGINAL 인쇄 (결과적으로 배경 반투명)
**일치**: ✅ 100% 일치

### 3. 모드 전환 이벤트 핸들링

```python
def on_print_mode_changed(mode: str):
    """인쇄 모드 변경 시 캔버스 즉시 업데이트"""
    if mode == "normal":
        # 일반 인쇄: ORIGINAL 표시
        update_canvas_for_normal_print()
        print("[CANVAS] 일반 인쇄 모드 → 원본 이미지 표시")

    elif mode == "layered":
        # 레이아웃 인쇄: 마스크 여부에 따라 다르게 표시
        if self.mask_image_path:
            # 마스크 있음: 시뮬레이션 표시
            update_canvas_for_layered_print_with_mask()
            print("[CANVAS] 레이아웃 인쇄 모드 (마스크 있음) → 시뮬레이션 표시")
        else:
            # 마스크 없음: ORIGINAL 표시
            update_canvas_for_layered_print_no_mask()
            print("[CANVAS] 레이아웃 인쇄 모드 (마스크 없음) → 원본 이미지 표시")
```

### 4. 배경제거 적용 흐름 (수정)

```python
def _on_masking_applied(threshold: int, mask_path: str):
    """배경제거 적용"""
    try:
        # 1. 마스크 경로 저장
        self.mask_image_path = mask_path

        # 2. 원본/마스크 이미지 로드
        original = load_image(self.selected_layer.image_path)
        mask = load_image(mask_path)

        # 3. ImageLayer에 마스크 저장
        self.selected_layer.mask_pixmap = QPixmap(mask_path)

        # 4. FINAL_RESULT 생성 (비교용)
        final_result = create_final_result(original, mask)
        self.selected_layer.final_result_pixmap = final_result

        # 5. 자동으로 레이아웃 인쇄 모드로 전환
        current_mode = self.property_panel.get_print_mode()
        if current_mode != "layered":
            print("[AUTO] 배경제거 완료 → 레이아웃 인쇄 모드로 자동 전환")
            self.property_panel.set_print_mode("layered")

        # 6. 캔버스 업데이트 (레이아웃 인쇄 시뮬레이션)
        update_canvas_for_layered_print_with_mask()

        # 7. 비교 패널 업데이트 (FINAL_RESULT 표시)
        self.update_comparison_panel()

        print("[OK] 배경제거 적용 완료")
        print("[CANVAS] 레이아웃 인쇄 시뮬레이션 표시 (실제 인쇄 결과 예상)")

    except Exception as e:
        print(f"[ERROR] 배경제거 적용 실패: {e}")

def create_final_result(original, mask):
    """최종 합성 결과 (비교용 - 인쇄와는 무관)"""
    # 마스킹 영역을 녹색으로 강조하여 어디가 선명한지 표시
    result = original.copy()
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    masked_area = (gray_mask < 128)

    # 마스킹 영역 녹색 틴트
    result[masked_area, 1] = np.clip(result[masked_area, 1] + 50, 0, 255)

    return result
```

---

## 🖼️ UI 구성 변경

### 현재 UI (문제)
```
┌─────────────────────────────────────┐
│           Main Window               │
├─────────────────────────────────────┤
│  [Canvas]                           │
│  - FINAL_RESULT 표시 (합성 이미지)  │
│  - 인쇄 결과와 불일치 ❌            │
└─────────────────────────────────────┘
```

### 개선된 UI (목표)

```
┌─────────────────────────────────────────────────────────┐
│                   Main Window                           │
├─────────────────────────────────┬───────────────────────┤
│  [Canvas - 실제 인쇄 미리보기]  │  [비교 패널]          │
│                                 │                       │
│  일반인쇄 모드:                 │  • 원본 이미지        │
│  → ORIGINAL 표시                │  • 마스크 이미지      │
│                                 │  • 최종 결과 (참고용) │
│  레이아웃인쇄 모드 (마스크 없음):│                      │
│  → ORIGINAL 표시                │  마스크 없을 때:      │
│                                 │  → 비교 패널 비활성   │
│  레이아웃인쇄 모드 (마스크 있음):│                      │
│  → 시뮬레이션 표시              │  마스크 있을 때:      │
│     (배경 반투명)               │  → 3개 이미지 표시    │
│                                 │                       │
│  ✅ 캔버스 = 실제 인쇄 결과     │  ✅ 비교 가능         │
└─────────────────────────────────┴───────────────────────┘
```

---

## 🔄 전체 워크플로우

### 시나리오 1: 배경제거 없이 일반 인쇄

```
1. 이미지 업로드
   → 캔버스: ORIGINAL 표시
   → 비교 패널: 비활성

2. 인쇄 모드: 일반 인쇄 (기본값)
   → 캔버스: ORIGINAL 유지

3. 인쇄 실행
   → 실제 인쇄: ORIGINAL
   → ✅ 일치
```

### 시나리오 2: 배경제거 적용 후 레이아웃 인쇄

```
1. 이미지 업로드
   → 캔버스: ORIGINAL 표시
   → 비교 패널: 비활성

2. 배경제거 버튼 클릭
   → 자동으로 레이아웃 인쇄 모드로 전환
   → 캔버스: 레이아웃 인쇄 시뮬레이션 표시 (배경 반투명)
   → 비교 패널: 활성화 (원본/마스크/결과 표시)

3. 인쇄 실행
   → 실제 인쇄: MASK 선인쇄 → ORIGINAL 인쇄
   → 결과: 배경 반투명 (시뮬레이션과 동일)
   → ✅ 일치
```

### 시나리오 3: 배경제거 후 일반 인쇄로 전환

```
1. 배경제거 적용됨
   → 캔버스: 레이아웃 인쇄 시뮬레이션 표시
   → 비교 패널: 활성

2. 일반 인쇄 모드로 전환
   → 캔버스: ORIGINAL로 즉시 변경
   → 비교 패널: 유지 (참고용)

3. 인쇄 실행
   → 실제 인쇄: ORIGINAL
   → ✅ 일치
```

### 시나리오 4: 레이아웃 인쇄 모드에서 배경제거

```
1. 레이아웃 인쇄 모드 선택
   → 캔버스: ORIGINAL 표시 (마스크 없음)

2. 배경제거 적용
   → 캔버스: 레이아웃 인쇄 시뮬레이션으로 변경
   → 비교 패널: 활성화

3. 인쇄 실행
   → 실제 인쇄: MASK 선인쇄 → ORIGINAL 인쇄
   → ✅ 일치
```

---

## 📊 캔버스 표시 상태 매트릭스

| 인쇄 모드 | 마스크 여부 | 캔버스 표시 | 실제 인쇄 | 일치 여부 |
|-----------|-------------|-------------|-----------|-----------|
| 일반 인쇄 | 없음 | ORIGINAL | ORIGINAL | ✅ 일치 |
| 일반 인쇄 | 있음 | ORIGINAL | ORIGINAL | ✅ 일치 |
| 레이아웃 인쇄 | 없음 | ORIGINAL | ORIGINAL | ✅ 일치 |
| 레이아웃 인쇄 | 있음 | 시뮬레이션 (배경 반투명) | MASK 선인쇄 + ORIGINAL | ✅ 일치 |

---

## 🛠️ 구현 계획

### Phase 1: 핵심 로직 구현
1. **ImageLayer 확장**
   - `mask_pixmap` 속성 추가
   - `final_result_pixmap` 속성 추가

2. **캔버스 업데이트 함수**
   - `update_canvas_for_normal_print()`
   - `update_canvas_for_layered_print_no_mask()`
   - `update_canvas_for_layered_print_with_mask()`
   - `create_layered_print_simulation()`

3. **모드 전환 이벤트 핸들러 수정**
   - PropertyPanel의 `print_mode_combo` 변경 이벤트 연결
   - `on_print_mode_changed()` 수정하여 캔버스 즉시 업데이트

4. **배경제거 적용 로직 수정**
   - `_on_masking_applied()` 수정
   - `_on_manual_masking_applied()` 수정
   - FINAL_RESULT는 별도 속성으로 저장, 캔버스에는 시뮬레이션 표시

### Phase 2: 비교 패널 추가 (선택적)
1. **ComparisonPanel 위젯 생성**
   - 원본/마스크/결과 미리보기
   - 탭 또는 버튼으로 전환 가능

2. **MainWindow에 비교 패널 통합**
   - 오른쪽 또는 하단에 도킹 가능한 패널
   - 마스크 없을 때는 자동 숨김

### Phase 3: 테스트 및 검증
1. **각 시나리오 테스트**
   - 모든 모드 조합 테스트
   - 한글 파일명 테스트
   - 양면 인쇄 테스트

2. **사용자 피드백 수집**
   - 실제 인쇄 테스트
   - UI/UX 개선 사항 수집

---

## 💡 기술적 고려사항

### 1. 성능 최적화
- **이미지 캐싱**: 시뮬레이션 이미지를 캐싱하여 모드 전환 시 빠르게 표시
- **레이지 로딩**: 비교 패널은 필요할 때만 이미지 로드

### 2. 메모리 관리
- 큰 이미지의 경우 메모리 사용량 고려
- 불필요한 중복 이미지 제거

### 3. 색상 정확도
- 시뮬레이션이 실제 인쇄 결과와 최대한 일치하도록 색상 조정
- 프린터 특성 반영 (선택적)

### 4. 한글 경로 지원
- 모든 이미지 저장/로드 시 `cv2.imencode()` + Python file handling 사용
- QPixmap 생성 시 한글 경로 문제 없음 확인

---

## 🎉 기대 효과

### 사용자 경험 개선
1. **직관성**: 캔버스에 보이는 것이 정확히 인쇄됨
2. **예측 가능성**: 인쇄 전에 정확한 결과 확인 가능
3. **실수 방지**: 잘못된 모드로 인쇄하는 실수 감소

### 시스템 일관성
1. **WYSIWYG 원칙**: 모든 UI가 일관된 원칙 따름
2. **명확한 역할 분리**: 캔버스 = 인쇄 미리보기, 비교 패널 = 비교/검토
3. **유지보수성**: 명확한 로직으로 디버깅 용이

### 확장성
1. **추가 인쇄 모드**: 새로운 인쇄 모드 추가 시 동일한 패턴 적용 가능
2. **고급 기능**: 프린터 색상 프로파일, 재질별 미리보기 등 확장 가능

---

## 📝 다음 단계

1. ✅ **설계 문서 리뷰** (현재)
2. **Phase 1 구현**: 핵심 캔버스 로직 구현
3. **테스트**: 각 시나리오별 동작 검증
4. **Phase 2 구현** (선택적): 비교 패널 추가
5. **최종 검증**: 실제 인쇄 테스트로 시뮬레이션 정확도 확인

---

## 🔗 관련 파일

### 수정이 필요한 파일
1. `canvas/layers/image_layer.py`: ImageLayer 클래스 확장
2. `ui_v2/main_window.py`:
   - `_on_masking_applied()` 수정
   - `_on_manual_masking_applied()` 수정
   - 캔버스 업데이트 함수 추가
3. `ui_v2/panels/property_panel.py`: 모드 전환 이벤트 시그널 연결

### 새로 생성할 파일 (Phase 2)
1. `ui_v2/panels/comparison_panel.py`: 비교 패널 위젯

---

**설계자**: Claude
**설계 날짜**: 2025-11-24
**버전**: 1.0
