# 레이어 분리 인쇄 구현 옵션

## 현재 문제

```python
# ❌ 현재 잘못된 방식
def prepare_front_canvas(...):
    self.setup_canvas()

    if watermark_path:
        self.draw_watermark(...)  # W 레이어

    self.draw_image(...)  # YMC 레이어

    img_info = self.commit_canvas()  # 한 번에 커밋
    return img_info
```

**문제**: 워터마크와 원본이 같은 캔버스에 그려져서 색상이 겹침

---

## 가능한 해결 방법

### 옵션 1: 완전 분리 - 두 번 인쇄

```python
def print_dual_side_card_layered(...):
    """레이어 인쇄 - 두 번 인쇄"""

    # === 1단계: W 레이어만 인쇄 (마스킹) ===
    self.inject_card()  # 카드 삽입

    # 워터마크만 있는 캔버스 준비
    self.setup_canvas(front_orientation)
    self.draw_watermark(0, 0, width, height, front_mask_path)
    front_w_info = self.commit_canvas()

    # W 레이어 인쇄
    ret = self.lib.R600PrintDraw(front_w_info.encode(), None)
    self._check_result(ret, "W 레이어 인쇄")

    # === 2단계: YMC 레이어만 인쇄 (원본) ===
    # 캔버스 클리어
    self.clear_canvas()

    # 원본만 있는 캔버스 준비
    self.setup_canvas(front_orientation)
    self.draw_image(0, 0, width, height, front_image_path)
    front_ymc_info = self.commit_canvas()

    # YMC 레이어 인쇄
    ret = self.lib.R600PrintDraw(front_ymc_info.encode(), None)
    self._check_result(ret, "YMC 레이어 인쇄")

    # 카드 배출
    self.eject_card()
```

**장점**: 완전히 분리되어서 겹칠 가능성 없음
**단점**: 두 번 인쇄해야 하고, 프린터가 이를 지원하는지 확인 필요

---

### 옵션 2: 리본 옵션 제어

```python
def print_dual_side_card_layered(...):
    """레이어 인쇄 - 리본 옵션으로 제어"""

    self.inject_card()

    # === 1단계: W 레이어만 활성화 ===
    self.set_ribbon_option(ribbon_type=1, key=0, value="W")  # W만 활성화

    # 워터마크 그리기
    self.setup_canvas(front_orientation)
    self.draw_watermark(0, 0, width, height, front_mask_path)
    front_w_info = self.commit_canvas()

    # W 레이어 인쇄
    ret = self.lib.R600PrintDraw(front_w_info.encode(), None)

    # === 2단계: YMC 레이어만 활성화 ===
    self.set_ribbon_option(ribbon_type=1, key=0, value="YMC")  # YMC만 활성화

    # 원본 그리기
    self.clear_canvas()
    self.setup_canvas(front_orientation)
    self.draw_image(0, 0, width, height, front_image_path)
    front_ymc_info = self.commit_canvas()

    # YMC 레이어 인쇄
    ret = self.lib.R600PrintDraw(front_ymc_info.encode(), None)

    self.eject_card()
```

**장점**: 리본 옵션으로 레이어 제어
**단점**: 리본 옵션 값이 정확한지 확인 필요

---

### 옵션 3: 프린터 모드 설정

```python
def print_dual_side_card_layered(...):
    """레이어 인쇄 - 프린터 모드 제어"""

    self.inject_card()

    # === 레이어 모드 활성화 ===
    # 혹시 이런 함수가 있다면?
    # self.lib.R600SetLayerMode(1)  # 레이어 모드 ON

    # 워터마크와 원본을 같은 캔버스에 그리지만,
    # 레이어 모드가 활성화되어 있어서 자동으로 분리됨
    self.setup_canvas(front_orientation)
    self.draw_watermark(0, 0, width, height, front_mask_path)  # W 레이어
    self.draw_image(0, 0, width, height, front_image_path)      # YMC 레이어
    front_info = self.commit_canvas()

    # 인쇄
    ret = self.lib.R600PrintDraw(front_info.encode(), None)

    self.eject_card()
```

**장점**: 간단함
**단점**: 이런 함수가 존재하는지 확인 필요

---

### 옵션 4: 별도 함수 존재 가능성

```python
def print_dual_side_card_layered(...):
    """레이어 인쇄 - 별도 함수 사용"""

    self.inject_card()

    # === W 레이어 캔버스 준비 ===
    self.setup_canvas(front_orientation)
    self.draw_watermark(0, 0, width, height, front_mask_path)
    front_w_info = self.commit_canvas()

    # === YMC 레이어 캔버스 준비 ===
    self.clear_canvas()
    self.setup_canvas(front_orientation)
    self.draw_image(0, 0, width, height, front_image_path)
    front_ymc_info = self.commit_canvas()

    # === 레이어 인쇄 전용 함수 (혹시?) ===
    # self.lib.R600PrintDrawLayered(
    #     front_w_info.encode(),    # W 레이어
    #     front_ymc_info.encode(),  # YMC 레이어
    #     None, None                # 뒷면
    # )

    # 또는
    # self.lib.R600PrintLayer(front_w_info.encode(), "W")
    # self.lib.R600PrintLayer(front_ymc_info.encode(), "YMC")

    self.eject_card()
```

**장점**: 레이어 전용 함수가 있다면 가장 정확
**단점**: 함수 존재 여부 확인 필요

---

## 질문사항

### V1 구현 방식을 알기 위한 질문

1. **인쇄 횟수**: V1에서 레이어 인쇄 시 프린터가 몇 번 동작했나요?
   - [ ] 한 번에 인쇄됨
   - [ ] 두 번 인쇄됨 (W → YMC 순서로)

2. **함수 호출**: V1에서 어떤 함수를 사용했나요?
   - [ ] `R600PrintDraw()` 한 번만
   - [ ] `R600PrintDraw()` 두 번
   - [ ] 다른 함수 (함수명: _____________)

3. **캔버스 준비**: V1에서 캔버스를 어떻게 준비했나요?
   - [ ] 한 캔버스에 워터마크 + 원본 함께
   - [ ] 두 캔버스 따로 (워터마크 캔버스 + 원본 캔버스)
   - [ ] 기타 방법

4. **리본 옵션**: V1에서 리본 옵션을 변경했나요?
   - [ ] 변경 안 함 (항상 `value="2"`)
   - [ ] W와 YMC를 따로 설정
   - [ ] 기타 설정

5. **특별한 설정**: V1에서 레이어 모드 관련 특별한 설정이 있었나요?
   - [ ] 없음
   - [ ] 있음 (설명: _____________)

---

## 추천 방법

**사용자님의 설명을 바탕으로 추천:**

> "W 또는 S 인쇄로 먼저 인쇄를 하는거야. 검은색 마스킹 부분을. 그래서 함수로 그 이미지를 워터마크로 전달하는 함수가 v1에 구현되어있고."

이 설명으로 보면, **옵션 1 (완전 분리 - 두 번 인쇄)** 또는 **옵션 2 (리본 옵션 제어)**가 가장 유사해 보입니다.

### 옵션 1 구현 예시

```python
def print_layered_mode(self, front_image_path, front_mask_path, ...):
    """레이어 인쇄 - W 레이어 먼저, YMC 레이어 나중"""

    # 카드 삽입
    self.inject_card()

    # === W 레이어 인쇄 (마스킹) ===
    print("[LAYER 1] W 레이어 인쇄 중 (마스킹)...")
    self.setup_canvas(front_orientation)
    self.draw_watermark(0.0, 0.0, card_width, card_height, front_mask_path)
    w_layer_info = self.commit_canvas()

    # W 레이어 인쇄 실행
    ret = self.lib.R600PrintDraw(w_layer_info.encode('cp949'), None)
    self._check_result(ret, "W 레이어 인쇄")
    time.sleep(1)  # 프린터 대기

    # === YMC 레이어 인쇄 (원본) ===
    print("[LAYER 2] YMC 레이어 인쇄 중 (원본)...")
    self.clear_canvas()
    self.setup_canvas(front_orientation)
    self.draw_image(0.0, 0.0, card_width, card_height, front_image_path)
    ymc_layer_info = self.commit_canvas()

    # YMC 레이어 인쇄 실행
    ret = self.lib.R600PrintDraw(ymc_layer_info.encode('cp949'), None)
    self._check_result(ret, "YMC 레이어 인쇄")
    time.sleep(1)  # 프린터 대기

    # 카드 배출
    self.eject_card()
```

이 방식이 맞나요? 아니면 다른 방식인가요?
