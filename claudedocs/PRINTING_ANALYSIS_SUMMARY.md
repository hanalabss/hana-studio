# 인쇄 로직 분석 요약 및 즉시 실행 가능한 해결 방안

## 🔴 핵심 문제 (Root Cause)

### 현재 코드의 잘못된 부분

**파일: `printer/r600_printer.py`**
**라인: 609-627 (prepare_front_canvas)**

```python
# ❌ 문제 코드
def prepare_front_canvas(self, front_image_path: str, watermark_path: Optional[str] = None,
                       card_width: float = 55, card_height: float = 86.6,
                       card_orientation: str = "portrait") -> str:
    print(f"=== 앞면 캔버스 준비 ({card_orientation}) ===")

    self.setup_canvas(card_orientation)

    # ❌ 문제: 워터마크(마스크)를 먼저 그림
    if watermark_path:
        self.draw_watermark(0.0, 0.0, card_width, card_height, watermark_path)

    # ❌ 문제: 그 위에 원본 이미지를 그림 → 색상 겹침 발생!
    self.draw_image(0.0, 0.0, card_width, card_height, front_image_path)

    self.front_img_info = self.commit_canvas()
    return self.front_img_info
```

**왜 문제인가?**
- 같은 캔버스에 워터마크(마스크)와 원본 이미지를 겹쳐 그리고 있음
- 마스크의 검은색 부분 위에 원본 색상이 합성되면서 색상이 이상하게 변함
- 예: 빨간색 + 검은색 = 어두운 빨간색 (의도하지 않은 결과)

---

## ✅ 해결 방안 (Solution)

### 핵심 아이디어

> **R600DrawWaterMark()와 R600DrawImage()는 이미 레이어별로 분리되어 동작한다!**

프린터 DLL 분석 결과:
- `R600DrawWaterMark()` → **W(White) 레이어**에 그려짐
- `R600DrawImage()` → **YMC(컬러) 레이어**에 그려짐
- 두 함수를 같은 캔버스에서 호출하면, 프린터가 알아서 레이어를 분리해서 인쇄함

**따라서 현재 코드가 이론적으로는 맞지만, 실제로는 색상이 겹치는 이유:**
→ **마스크 이미지가 잘못 준비되었거나, 프린터가 마스크를 제대로 해석하지 못하고 있음**

### 실제 문제 지점

1. **마스크 이미지 형식 문제**
   - 현재: 검은색(객체) / 흰색(배경)
   - 프린터 기대: 흰색 영역만 W 레이어로 인쇄해야 함
   - **반전 필요 여부 확인 필요!**

2. **워터마크 함수 사용 방법 문제**
   - `draw_watermark()`가 마스크의 **어떤 부분**을 W 레이어로 변환하는지 불명확
   - 예전 프로그램에서는 마스크를 **반전시켜서** 전송했을 가능성

---

## 🎯 즉시 실행 가능한 해결책

### 방안 1: 마스크 이미지 반전 (가장 가능성 높음)

```python
# core/image_processor.py 또는 printer/r600_printer.py

def prepare_mask_for_watermark(mask_image_path: str) -> str:
    """
    마스크 이미지를 워터마크 인쇄용으로 변환

    현재 마스크: 검은색 = 객체, 흰색 = 배경
    워터마크용:  흰색 = W 레이어 인쇄 영역, 검은색 = 인쇄 안 함

    → 반전 필요!
    """
    import cv2
    import numpy as np
    from utils.safe_temp_path import create_safe_temp_file

    # 마스크 읽기
    mask = cv2.imread(mask_image_path)

    # ✅ 반전: 검은색 ↔ 흰색
    inverted_mask = cv2.bitwise_not(mask)

    # 임시 파일로 저장
    inverted_path = create_safe_temp_file(
        prefix="watermark_inverted",
        suffix=".jpg"
    )
    cv2.imwrite(inverted_path, inverted_mask, [cv2.IMWRITE_JPEG_QUALITY, 95])

    print(f"[WATERMARK] 마스크 반전 완료: {inverted_path}")
    return inverted_path
```

**적용 위치: `printer/r600_printer.py` - `prepare_front_canvas()`**

```python
def prepare_front_canvas(self, front_image_path: str, watermark_path: Optional[str] = None,
                       card_width: float = 55, card_height: float = 86.6,
                       card_orientation: str = "portrait") -> str:
    print(f"=== 앞면 캔버스 준비 ({card_orientation}) ===")

    self.setup_canvas(card_orientation)

    # ✅ 수정: 마스크를 반전시켜서 워터마크로 사용
    if watermark_path:
        inverted_watermark_path = self.prepare_mask_for_watermark(watermark_path)
        self.draw_watermark(0.0, 0.0, card_width, card_height, inverted_watermark_path)

    # 원본 이미지 그리기
    self.draw_image(0.0, 0.0, card_width, card_height, front_image_path)

    self.front_img_info = self.commit_canvas()
    return self.front_img_info
```

---

### 방안 2: 일반 인쇄와 레이어 인쇄 완전 분리 (근본적 해결)

```python
def prepare_front_canvas(self, front_image_path: str, watermark_path: Optional[str] = None,
                       card_width: float = 55, card_height: float = 86.6,
                       card_orientation: str = "portrait",
                       print_mode: str = "normal") -> str:
    """앞면 캔버스 준비 - 인쇄 모드에 따라 분리"""

    if print_mode == "layered":
        return self._prepare_layered_canvas(
            front_image_path, watermark_path, card_width, card_height, card_orientation
        )
    else:
        return self._prepare_normal_canvas(
            front_image_path, card_width, card_height, card_orientation
        )

def _prepare_normal_canvas(self, image_path, card_width, card_height, orientation):
    """일반 인쇄: 원본만"""
    self.setup_canvas(orientation)
    self.draw_image(0.0, 0.0, card_width, card_height, image_path)
    return self.commit_canvas()

def _prepare_layered_canvas(self, image_path, watermark_path, card_width, card_height, orientation):
    """레이어 인쇄: 마스크(반전) + 원본"""
    self.setup_canvas(orientation)

    # ✅ 마스크 반전 후 워터마크로
    inverted_watermark = self.prepare_mask_for_watermark(watermark_path)
    self.draw_watermark(0.0, 0.0, card_width, card_height, inverted_watermark)

    # 원본 이미지
    self.draw_image(0.0, 0.0, card_width, card_height, image_path)

    return self.commit_canvas()
```

---

## 🧪 테스트 방법

### 1단계: 마스크 반전 확인
```python
# 마스크 이미지를 반전시켜서 저장 후 육안 확인
mask_path = "front_mask_print.jpg"
inverted_path = prepare_mask_for_watermark(mask_path)

# 반전된 이미지 확인:
# - 원래 검은색(객체) → 흰색
# - 원래 흰색(배경) → 검은색
```

### 2단계: 레이어 인쇄 테스트
```
1. 배경제거 실행
2. 레이어 인쇄 모드 선택
3. 인쇄 실행
4. 결과 확인:
   ✓ 객체 영역: 불투명 + 정확한 원본 색상
   ✓ 배경 영역: 반투명 + 정확한 원본 색상
   ✗ 색상 겹침 없음
```

---

## 📋 구현 우선순위

### 🔴 긴급 (Immediate)

1. **마스크 반전 함수 추가** (`prepare_mask_for_watermark`)
   - 위치: `printer/r600_printer.py`
   - 라인: 새 메서드로 추가

2. **prepare_front_canvas 수정**
   - 워터마크 사용 시 반전 적용
   - 동일하게 `prepare_back_canvas`도 수정

3. **테스트**
   - 실제 프린터로 레이어 인쇄 테스트
   - 색상 겹침 문제 해결 확인

### 🟡 중요 (Important)

4. **일반 인쇄 vs 레이어 인쇄 로직 분리**
   - `_prepare_normal_canvas()` 함수 추가
   - `_prepare_layered_canvas()` 함수 추가

5. **뒷면 캔버스도 동일하게 수정**
   - `prepare_back_canvas()` 리팩토링

---

## 💡 추가 고려 사항

### 마스크 반전이 필요한 이유

```
YMCW 리본 프린터 동작:
├─ W 레이어: 워터마크 함수로 전달된 이미지의 "밝은 부분"을 흰색으로 인쇄
├─ YMC 레이어: 원본 이미지 전체를 컬러로 인쇄
└─ 최종 결과: W 레이어 위에 YMC가 겹쳐서 불투명 효과

현재 마스크:
├─ 검은색 = 객체 (불투명하게 하고 싶은 부분)
└─ 흰색 = 배경 (반투명하게 하고 싶은 부분)

프린터 워터마크 해석:
├─ 밝은 부분(흰색) = W 레이어 인쇄 → 불투명
└─ 어두운 부분(검은색) = W 레이어 없음 → 반투명

⚠️ 문제: 현재 마스크는 반대!
✅ 해결: 마스크를 반전시켜야 함!
```

---

## 🎯 결론

**즉시 실행 가능한 해결책:**
1. 마스크 반전 함수 구현 (`prepare_mask_for_watermark`)
2. `prepare_front_canvas`와 `prepare_back_canvas`에서 워터마크 사용 시 반전 적용
3. 테스트 및 검증

**예상 결과:**
- 레이어 인쇄 시 색상 겹침 문제 해결
- 객체 영역 불투명, 배경 영역 반투명
- 정확한 원본 색상 재현

**다음 단계:**
- 위 코드 적용 후 실제 프린터로 테스트
- 문제 지속 시 예전 프로그램 코드 비교 분석 필요
