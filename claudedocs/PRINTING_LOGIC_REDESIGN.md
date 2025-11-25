# 인쇄 로직 재설계 - 배경제거 후 레이어 인쇄 문제 해결

## 🔴 현재 문제점 분석 (Current Problem Analysis)

### 문제 상황
배경제거 후 레이어 인쇄 시 검은색 마스크 부분과 원본 이미지가 겹치면서 **색상이 이상하게 나타나는 문제** 발생

### 현재 잘못된 구조

```python
# ❌ 현재 printer/r600_printer.py의 잘못된 로직

def prepare_front_canvas(self, front_image_path: str, watermark_path: Optional[str] = None, ...):
    """앞면 캔버스 준비"""
    self.setup_canvas(card_orientation)

    # ❌ 문제 1: 워터마크(마스크)를 먼저 그림
    if watermark_path:
        self.draw_watermark(0.0, 0.0, card_width, card_height, watermark_path)

    # ❌ 문제 2: 그 위에 원본 이미지를 그림 → 색상이 겹쳐서 이상해짐
    self.draw_image(0.0, 0.0, card_width, card_height, front_image_path)
```

**왜 문제인가?**
- 마스크의 검은색 부분 위에 원본 이미지의 색상이 겹쳐지면서 **색상이 혼합**됨
- 예: 빨간색 위에 검은색이 겹치면 어두운 빨간색으로 변함
- 실제 프린터는 **레이어별로 분리해서 인쇄**해야 하는데, 현재는 **겹쳐서 합성**하고 있음

---

## ✅ 올바른 인쇄 방식 (Correct Printing Method)

### YMCW 리본 프린터의 동작 원리

```
YMCW 리본 프린터 (R600):
├─ Y (Yellow)   레이어
├─ M (Magenta)  레이어
├─ C (Cyan)     레이어
└─ W (White)    레이어 ← 워터마크(마스크)로 사용
```

### 레이어 인쇄의 올바른 순서

```
1단계: W(White) 레이어 먼저 인쇄
   ├─ 마스크의 검은색 부분만 흰색으로 인쇄
   └─ 투명한 카드에 불투명한 베이스를 깔아줌

2단계: YMC(컬러) 레이어 나중 인쇄
   ├─ 원본 이미지 전체를 인쇄
   └─ W 레이어 위에 색상이 올라가면 불투명하게 보임

최종 결과:
   ├─ W 레이어 있는 부분 (마스크 검은색) = 불투명 + 원본 색상
   └─ W 레이어 없는 부분 (마스크 흰색) = 반투명 + 원본 색상
```

### 예전 프로그램의 올바른 구조 (참고)

```python
# ✅ 올바른 방식 (예전 프로그램)

# 1단계: W 레이어 준비 (마스크만)
draw_watermark(mask_path)  # 검은색 부분만 흰색으로

# 2단계: YMC 레이어 준비 (원본 전체)
draw_image(original_path)  # 원본 이미지 전체

# 3단계: 프린터로 전송 (레이어 분리되어 전송)
R600PrintDraw(front_canvas, back_canvas)
```

---

## 🎯 해결 방안 (Solution Design)

### 핵심 원칙

> **마스크와 원본 이미지를 겹치지 말고, 레이어별로 분리해서 전송**

### 새로운 설계 구조

```python
# ✅ 새로운 올바른 구조

class R600Printer:

    def prepare_layered_canvas_separated(
        self,
        original_image_path: str,
        mask_image_path: str,
        card_width: float,
        card_height: float,
        card_orientation: str = "portrait"
    ):
        """
        레이어 인쇄를 위한 분리된 캔버스 준비

        핵심: 마스크와 원본을 겹치지 않고, 레이어별로 분리
        """

        print(f"=== 레이어 인쇄 캔버스 준비 (분리 모드) ===")

        # === 1단계: W 레이어만 그리기 (마스크) ===
        self.setup_canvas(card_orientation)

        # 마스크만 워터마크로 그림 (검은색 → 흰색 변환은 프린터가 자동 처리)
        self.draw_watermark(0.0, 0.0, card_width, card_height, mask_image_path)

        # W 레이어 커밋
        w_layer_info = self.commit_canvas()

        # === 2단계: YMC 레이어만 그리기 (원본) ===
        self.clear_canvas()
        self.setup_canvas(card_orientation)

        # 원본 이미지만 그림 (마스크와 겹치지 않음)
        self.draw_image(0.0, 0.0, card_width, card_height, original_image_path)

        # YMC 레이어 커밋
        ymc_layer_info = self.commit_canvas()

        print(f"[OK] W 레이어: {w_layer_info}")
        print(f"[OK] YMC 레이어: {ymc_layer_info}")

        return w_layer_info, ymc_layer_info
```

---

## 📋 구현 계획 (Implementation Plan)

### Phase 1: 분석 단계 (Analysis)

#### 1.1 예전 프로그램 코드 확인
```bash
# 예전 프로그램에서 워터마크 인쇄 코드 찾기
- 워터마크 먼저 인쇄하는 로직 확인
- YMC 레이어 나중 인쇄하는 로직 확인
- 두 레이어가 어떻게 분리되어 전송되는지 확인
```

#### 1.2 현재 문제 코드 위치 파악
```python
# 수정 필요한 파일 목록
1. printer/r600_printer.py
   - prepare_front_canvas()  ← 여기가 핵심!
   - prepare_back_canvas()   ← 여기도!

2. printer/printer_thread.py
   - print_dual_side_card()  ← 레이어 모드 처리
   - print_single_side_card() ← 레이어 모드 처리
```

### Phase 2: 코드 수정 (Implementation)

#### 2.1 일반 인쇄 vs 레이어 인쇄 분리

```python
# printer/r600_printer.py 수정안

def prepare_front_canvas(
    self,
    front_image_path: str,
    watermark_path: Optional[str] = None,
    card_width: float = 55,
    card_height: float = 86.6,
    card_orientation: str = "portrait",
    print_mode: str = "normal"  # ← 새로운 파라미터
) -> str:
    """앞면 캔버스 준비 - 인쇄 모드에 따라 다르게 처리"""

    if print_mode == "layered" and watermark_path:
        # ✅ 레이어 인쇄 모드: 마스크만 그리기
        return self._prepare_layered_front_canvas_w_layer(
            watermark_path, card_width, card_height, card_orientation
        )
    else:
        # ✅ 일반 인쇄 모드: 원본만 그리기
        return self._prepare_normal_front_canvas(
            front_image_path, card_width, card_height, card_orientation
        )

def _prepare_layered_front_canvas_w_layer(
    self,
    watermark_path: str,
    card_width: float,
    card_height: float,
    card_orientation: str
) -> str:
    """레이어 인쇄용 W 레이어 준비 (마스크만)"""
    print(f"=== 앞면 W 레이어 준비 (마스크만) ===")

    self.setup_canvas(card_orientation)

    # 마스크만 워터마크로 그리기
    self.draw_watermark(0.0, 0.0, card_width, card_height, watermark_path)

    # 커밋
    front_img_info = self.commit_canvas()
    return front_img_info

def _prepare_layered_front_canvas_ymc_layer(
    self,
    front_image_path: str,
    card_width: float,
    card_height: float,
    card_orientation: str
) -> str:
    """레이어 인쇄용 YMC 레이어 준비 (원본만)"""
    print(f"=== 앞면 YMC 레이어 준비 (원본만) ===")

    self.clear_canvas()
    self.setup_canvas(card_orientation)

    # 원본 이미지만 그리기
    self.draw_image(0.0, 0.0, card_width, card_height, front_image_path)

    # 커밋
    front_img_info = self.commit_canvas()
    return front_img_info

def _prepare_normal_front_canvas(
    self,
    front_image_path: str,
    card_width: float,
    card_height: float,
    card_orientation: str
) -> str:
    """일반 인쇄용 캔버스 준비 (원본만)"""
    print(f"=== 앞면 일반 인쇄 준비 (원본만) ===")

    self.setup_canvas(card_orientation)

    # 원본 이미지만 그리기
    self.draw_image(0.0, 0.0, card_width, card_height, front_image_path)

    # 커밋
    front_img_info = self.commit_canvas()
    return front_img_info
```

#### 2.2 양면 인쇄 로직 수정

```python
# printer/r600_printer.py - print_dual_side_card() 수정

def print_dual_side_card(
    self,
    front_image_path: str,
    back_image_path: Optional[str] = None,
    front_watermark_path: Optional[str] = None,
    back_watermark_path: Optional[str] = None,
    front_orientation: str = "portrait",
    back_orientation: str = "portrait",
    print_mode: str = "normal"
):
    """양면 카드 인쇄 - 레이어 모드 완전 재설계"""

    try:
        print(f"=== 양면 카드 인쇄 시작: {print_mode} 모드 ===")

        # 1. 카드 삽입
        self.inject_card()

        # 2. 리본 옵션 설정
        self.set_ribbon_option(ribbon_type=1, key=0, value="2")

        # 3. 앞면 캔버스 준비
        front_width, front_height = self.get_card_dimensions(front_orientation)

        if print_mode == "layered":
            # ✅ 레이어 인쇄: W 레이어 + YMC 레이어 분리
            print("[LAYERED] W 레이어 먼저 준비")
            front_w_info = self._prepare_layered_front_canvas_w_layer(
                front_watermark_path, front_width, front_height, front_orientation
            )

            print("[LAYERED] YMC 레이어 나중 준비")
            front_ymc_info = self._prepare_layered_front_canvas_ymc_layer(
                front_image_path, front_width, front_height, front_orientation
            )

            # 두 레이어를 어떻게 합칠지는 프린터 DLL 사양 확인 필요
            # 방법 1: 두 번 R600PrintDraw 호출?
            # 방법 2: 특별한 레이어 합성 함수?
            # ⚠️ 예전 프로그램 코드 확인 필요!

        else:
            # ✅ 일반 인쇄: 원본만
            front_img_info = self._prepare_normal_front_canvas(
                front_image_path, front_width, front_height, front_orientation
            )

        # 4. 뒷면 캔버스 준비 (동일한 로직)
        # ...

        # 5. 인쇄 실행
        # ⚠️ 레이어 모드일 때 어떻게 전송할지 확인 필요

    except Exception as e:
        print(f"인쇄 중 오류: {e}")
        raise
```

---

## 🔍 추가 조사 필요 사항 (Further Investigation Required)

### 1. 예전 프로그램 코드 분석
```
❓ 질문 1: 예전 프로그램에서 워터마크를 어떻게 전송했나?
   - draw_watermark() 함수 사용?
   - 별도의 레이어 함수?

❓ 질문 2: W 레이어와 YMC 레이어를 어떻게 분리 전송했나?
   - R600PrintDraw를 두 번 호출?
   - 특별한 레이어 설정 함수?

❓ 질문 3: 마스크 이미지 형식은?
   - 검은색/흰색만 있는 이진 이미지?
   - 프린터가 검은색 → 흰색으로 자동 변환?
```

### 2. 프린터 DLL 함수 확인
```python
# libDSRetransfer600App.dll 함수 목록 확인 필요

R600DrawWaterMark()  # ← 이게 W 레이어 전용인가?
R600DrawImage()      # ← 이게 YMC 레이어 전용인가?
R600PrintDraw()      # ← 이게 두 레이어를 어떻게 처리하나?

# 혹시 별도의 레이어 함수가 있나?
R600SetLayer()?
R600PrintLayer()?
```

---

## 🎬 실행 계획 (Action Plan)

### Step 1: 예전 코드 확인 (우선순위 🔴 높음)
```bash
1. 예전 프로그램 실행 파일에서 DLL 호출 로그 확인
2. 워터마크 인쇄 시 어떤 함수가 호출되는지 확인
3. 레이어 분리 로직 파악
```

### Step 2: 현재 코드 수정
```python
1. r600_printer.py 리팩토링
   - 일반 인쇄 / 레이어 인쇄 완전 분리
   - _prepare_layered_*_w_layer() 함수 추가
   - _prepare_layered_*_ymc_layer() 함수 추가

2. printer_thread.py 수정
   - print_mode에 따라 다른 로직 호출

3. 테스트
   - 일반 인쇄: 원본 그대로 출력 ✓
   - 레이어 인쇄: W + YMC 분리 출력 ✓
```

### Step 3: 검증
```bash
1. 배경제거 후 레이어 인쇄 테스트
   - 마스크 영역: 불투명 + 원본 색상
   - 배경 영역: 반투명 + 원본 색상

2. 색상 겹침 문제 해결 확인
   - ❌ 이전: 검은색 + 원본 색상 = 어두운 색
   - ✅ 이후: W 레이어 + 원본 색상 = 정확한 색
```

---

## 📊 비교표 (Comparison)

| 항목 | ❌ 현재 (잘못됨) | ✅ 올바른 방식 |
|------|----------------|--------------|
| 일반 인쇄 | 원본 이미지만 그림 | 원본 이미지만 그림 |
| 레이어 인쇄 | 마스크 + 원본 **겹쳐서** 그림 | **W 레이어 먼저**, **YMC 레이어 나중** |
| 결과 | 색상이 겹쳐서 이상함 | 색상이 정확함 |
| 투명도 | 모두 불투명 | W 영역 불투명, 나머지 반투명 |

---

## 🎯 최종 목표 (Final Goal)

```
✅ 일반 인쇄
   → 원본 이미지 그대로 인쇄
   → 마스킹 없음

✅ 레이어 인쇄
   → 1단계: 마스크(검은색 부분)를 W 레이어로 먼저 인쇄
   → 2단계: 원본 이미지 전체를 YMC 레이어로 나중 인쇄
   → 결과: 마스크 영역은 불투명, 배경은 반투명으로 입체감

❌ 절대 안 됨
   → 마스크와 원본을 겹쳐서 합성하면 색상이 이상해짐!
```

---

## 다음 단계 (Next Steps)

1. **예전 프로그램 코드 분석** (가장 중요!)
   - 워터마크 인쇄 코드 찾기
   - DLL 함수 호출 순서 확인

2. **현재 코드 리팩토링**
   - 레이어 분리 로직 구현
   - 테스트 및 검증

3. **문서화**
   - 레이어 인쇄 원리 정리
   - 코드 주석 추가
