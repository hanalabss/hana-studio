# 배경제거 캔버스 표시 문제 수정 설계서

## 📋 문제 정의

### 현재 문제
일반인쇄 모드에서 배경제거를 적용한 후, 캔버스의 미리보기에서 **마스킹 제외 부분(배경)이 연하게 표시**되는 문제가 발생합니다. 이는 사용자에게 혼란을 주며, 실제 인쇄 결과와 다르게 보입니다.

### 근본 원인
현재 `UnifiedMaskViewer`의 오버레이 모드(`_create_overlay()`)에서:
- **배경 영역(흰색 마스크)**을 반투명하게 표시하도록 구현됨
- 이는 **레이아웃 인쇄 모드**의 실제 인쇄 결과를 시뮬레이션하기 위한 것
- 하지만 **일반 인쇄 모드**에서는 원본 이미지 전체가 정상적으로 인쇄되므로, 이 표시 방식이 부적절함

### 인쇄 모드별 동작 차이

#### 1. 일반 인쇄 (Normal Print)
```
인쇄 순서: YMC 리본만 사용
- 원본 이미지 전체를 컬러(YMC)로 인쇄
- 마스크는 사용되지 않음
- 결과: 원본 이미지가 그대로 인쇄됨
```

#### 2. 레이아웃 인쇄 (Layered Print)
```
인쇄 순서: W 리본 → YMC 리본
1단계: W (White) 리본으로 마스킹된 영역(검은색)만 선인쇄
2단계: YMC (Color) 리본으로 원본 이미지 전체 인쇄

결과:
- 마스킹 영역(검은색): W + YMC = 불투명하게 원본 색상
- 배경 영역(흰색): YMC만 = 반투명하게 원본 색상
```

## 🎯 수정 사항

### 1. 핵심 변경점
캔버스의 미리보기 표시 방식은 **현재 인쇄 모드**에 따라 달라져야 합니다:

- **일반 인쇄 모드**: 원본 이미지를 그대로 표시 (마스킹 효과 없음)
- **레이아웃 인쇄 모드**: 현재처럼 오버레이 효과 표시 (배경 반투명)

### 2. 구현 위치

#### A. `UnifiedMaskViewer` 클래스 수정
**파일**: `ui/components/image_viewer.py`

**현재 상태**:
```python
class UnifiedMaskViewer(QWidget):
    def _update_composite_display(self):
        # 항상 오버레이 모드로 표시
        if self.preview_mode == "overlay":
            display_image = self._create_overlay(
                self.original_image,
                current_mask,
                self.overlay_opacity
            )
```

**수정 필요 사항**:
1. 인쇄 모드를 추적하는 인스턴스 변수 추가: `self.print_mode`
2. `set_print_mode(mode: str)` 메서드 추가
3. `_create_overlay()` 메서드가 인쇄 모드를 고려하도록 수정

#### B. `MainWindow` 연결
**파일**: `hana_studio.py`

인쇄 모드가 변경될 때 `UnifiedMaskViewer`에 모드를 전달:
```python
def on_print_mode_changed(self, mode: str):
    # 기존 로직...

    # 추가: UnifiedMaskViewer에 인쇄 모드 전달
    self.ui.components['front_unified_mask_viewer'].set_print_mode(mode)
    if self.is_dual_side:
        self.ui.components['back_unified_mask_viewer'].set_print_mode(mode)
```

## 🔧 상세 구현 계획

### Phase 1: UnifiedMaskViewer 수정

#### 1.1 인스턴스 변수 추가
```python
class UnifiedMaskViewer(QWidget):
    def __init__(self, title=""):
        super().__init__()
        # 기존 초기화...
        self.print_mode = "normal"  # 기본값: 일반 인쇄
```

#### 1.2 set_print_mode 메서드 추가
```python
def set_print_mode(self, mode: str):
    """
    인쇄 모드 설정 및 미리보기 업데이트

    Args:
        mode: "normal" (일반 인쇄) 또는 "layered" (레이아웃 인쇄)
    """
    if mode not in ["normal", "layered"]:
        print(f"[WARNING] 잘못된 인쇄 모드: {mode}, 기본값 'normal' 사용")
        mode = "normal"

    old_mode = self.print_mode
    self.print_mode = mode

    if old_mode != mode:
        print(f"[CANVAS] 인쇄 모드 변경: {old_mode} → {mode}")
        # 미리보기 즉시 업데이트
        self._update_composite_display()
```

#### 1.3 _create_overlay 메서드 수정
```python
def _create_overlay(self, original: np.ndarray, mask: np.ndarray,
                   opacity: float = 0.6) -> np.ndarray:
    """
    원본과 마스크를 오버레이 합성

    인쇄 모드에 따라 다른 미리보기 생성:
    - normal: 원본 이미지 + 녹색 틴트로 마스킹 영역 강조만
    - layered: 배경 영역 반투명 처리 (실제 인쇄 결과 시뮬레이션)
    """
    try:
        overlay = original.copy()

        # 마스크가 BGR 3채널인 경우
        if len(mask.shape) == 3 and mask.shape[2] == 3:
            # 검은색 영역(객체)을 찾기: 모든 채널이 100 이하
            black_mask = np.all(mask <= 100, axis=2)

            if self.print_mode == "normal":
                # === 일반 인쇄 모드: 원본 그대로 + 마스킹 영역만 녹색 강조 ===
                green_tint = np.zeros_like(original)
                green_tint[:, :, 1] = 80  # 약한 녹색

                # 마스킹 영역에만 녹색 틴트 적용 (투명도 낮춤)
                for c in range(3):
                    overlay[:, :, c] = np.where(
                        black_mask,
                        (original[:, :, c] * 0.7 + green_tint[:, :, c] * 0.3).astype(np.uint8),
                        original[:, :, c]  # 배경은 원본 그대로
                    )

            elif self.print_mode == "layered":
                # === 레이아웃 인쇄 모드: 실제 인쇄 결과 시뮬레이션 ===
                # 마스킹 영역: 녹색 강조
                green_tint = np.zeros_like(original)
                green_tint[:, :, 1] = 100

                # 흰색 영역(배경) 찾기
                white_mask = np.all(mask >= 200, axis=2)

                # 배경 영역: 반투명 효과 (YMC만 인쇄)
                # 마스킹 영역: 녹색 틴트 (W + YMC 인쇄)
                for c in range(3):
                    overlay[:, :, c] = np.where(
                        black_mask,
                        # 마스킹 영역: 녹색 틴트
                        (original[:, :, c] * (1 - opacity) + green_tint[:, :, c] * opacity).astype(np.uint8),
                        np.where(
                            white_mask,
                            # 배경 영역: 반투명 (원본 30% + 흰색 70%)
                            (original[:, :, c] * 0.3 + 255 * 0.7).astype(np.uint8),
                            original[:, :, c]
                        )
                    )

        return overlay

    except Exception as e:
        print(f"[ERROR] 오버레이 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        return original
```

### Phase 2: MainWindow 연결

#### 2.1 on_print_mode_changed 수정
```python
def on_print_mode_changed(self, mode: str):
    """인쇄 모드 변경 시 호출"""
    self.print_mode = mode

    # 기존: 설정 저장 및 UI 업데이트
    config.set('print_mode', mode)
    self.ui.components['control_panel'].update_print_mode_display(
        mode, self.is_dual_side
    )
    self._update_print_button_state()

    mode_text = '일반 인쇄' if mode == 'normal' else '레이어 인쇄(YMCW)'
    self.log(f"인쇄 모드 변경: {mode_text}")

    # === 추가: UnifiedMaskViewer에 모드 전달 ===
    self.ui.components['front_unified_mask_viewer'].set_print_mode(mode)
    if self.is_dual_side:
        self.ui.components['back_unified_mask_viewer'].set_print_mode(mode)

    # 로그 추가
    canvas_effect = "원본 그대로" if mode == "normal" else "배경 반투명"
    self.log(f"   캔버스 미리보기: {canvas_effect}")
```

#### 2.2 초기화 시 모드 설정
```python
def __init__(self):
    # 기존 초기화...

    # === 추가: 초기 인쇄 모드를 UnifiedMaskViewer에 전달 ===
    initial_mode = config.get('print_mode', 'normal')
    self.ui.components['front_unified_mask_viewer'].set_print_mode(initial_mode)
    if self.is_dual_side:
        self.ui.components['back_unified_mask_viewer'].set_print_mode(initial_mode)
```

## 📊 시각적 비교

### 일반 인쇄 모드 (수정 후)
```
캔버스 미리보기:
┌─────────────────────┐
│                     │
│   [원본 이미지]     │  ← 배경도 정상 표시
│   🟢 [마스킹 영역]  │  ← 약한 녹색 틴트만
│                     │
└─────────────────────┘
```

### 레이아웃 인쇄 모드 (기존 유지)
```
캔버스 미리보기:
┌─────────────────────┐
│                     │
│   [반투명 배경]     │  ← 배경 반투명
│   🟢 [마스킹 영역]  │  ← 불투명 + 녹색 틴트
│                     │
└─────────────────────┘
```

## ✅ 검증 계획

### 1. 단위 테스트
- `set_print_mode()` 메서드가 올바르게 모드를 설정하는지 확인
- `_create_overlay()` 메서드가 모드에 따라 다른 결과를 반환하는지 확인

### 2. 통합 테스트
1. 일반 인쇄 모드로 설정
2. 이미지 업로드 및 배경제거 실행
3. 캔버스에서 배경이 정상적으로 표시되는지 확인
4. 레이아웃 인쇄 모드로 전환
5. 캔버스에서 배경이 반투명하게 표시되는지 확인

### 3. 사용자 시나리오 테스트
- **시나리오 1**: 일반 인쇄 모드에서 배경제거 → 적용 → 인쇄 미리보기 확인
- **시나리오 2**: 레이아웃 인쇄 모드로 전환 → 캔버스 변화 확인
- **시나리오 3**: 모드 전환 후 새 이미지 업로드 → 올바른 미리보기 표시 확인

## 🚀 구현 우선순위

1. **High Priority**: `UnifiedMaskViewer._create_overlay()` 수정
   - 사용자가 직접 보는 부분이므로 최우선

2. **High Priority**: `UnifiedMaskViewer.set_print_mode()` 추가
   - 모드 전환 기능의 핵심

3. **Medium Priority**: `MainWindow` 연결 코드
   - 기능 완성을 위한 통합

## 📝 추가 고려사항

### 1. 미리보기 모드별 동작
- **"원본" 모드**: 인쇄 모드와 관계없이 항상 원본만 표시
- **"마스크" 모드**: 인쇄 모드와 관계없이 항상 마스크만 표시
- **"겹침" 모드**: 인쇄 모드에 따라 다르게 표시 ← 이번 수정의 핵심

### 2. 사용자 가이드
인쇄 모드 변경 시 사용자에게 알림:
```
일반 인쇄 모드: 원본 이미지가 전체 인쇄됩니다
레이아웃 인쇄 모드: 마스킹 영역이 선명하게, 배경이 반투명하게 인쇄됩니다
```

### 3. 로깅 강화
모드 전환 시 상세 로그:
```python
self.log(f"[인쇄 모드] {mode_text}")
self.log(f"   실제 인쇄: {print_description}")
self.log(f"   캔버스 미리보기: {canvas_description}")
```

## 🔄 롤백 계획

만약 문제가 발생하면:
1. `_create_overlay()` 메서드만 원래대로 복원
2. `set_print_mode()` 호출 제거
3. Git을 통해 이전 커밋으로 롤백

## 📌 결론

이 설계는 **인쇄 모드별로 올바른 캔버스 미리보기**를 제공하여 사용자 경험을 개선합니다. 일반 인쇄 모드에서는 원본이 그대로 인쇄되므로 캔버스도 원본을 정상적으로 표시하고, 레이아웃 인쇄 모드에서는 실제 인쇄 결과를 정확히 시뮬레이션합니다.

**핵심 원칙**: "사용자가 보는 것이 인쇄되는 것이다 (WYSIWYG)"
