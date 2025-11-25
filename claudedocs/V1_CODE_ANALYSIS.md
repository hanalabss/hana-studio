# V1 코드 분석 결과

## 🔍 V1 코드 확인 (main 브랜치, 커밋 5d316eb)

### V1의 레이어 인쇄 로직

```python
# V1: printer/r600_printer.py

def prepare_front_canvas(self, front_image_path, watermark_path, ...):
    """앞면 캔버스 준비"""
    self.setup_canvas(card_orientation)

    # 워터마크 그리기 (레이어 모드인 경우)
    if watermark_path:
        self.draw_watermark(0.0, 0.0, card_width, card_height, watermark_path)

    # 앞면 이미지 그리기
    self.draw_image(0.0, 0.0, card_width, card_height, front_image_path)

    self.front_img_info = self.commit_canvas()
    return self.front_img_info

def print_dual_side_card(self, ..., print_mode="normal"):
    """양면 카드 인쇄"""
    self.inject_card()
    self.set_ribbon_option(ribbon_type=1, key=0, value="2")

    # 앞면 캔버스 준비
    if print_mode == "layered":
        front_img_info = self.prepare_front_canvas(
            front_image_path, front_watermark_path, ...
        )
    else:
        front_img_info = self.prepare_front_canvas(
            front_image_path, None, ...
        )

    # 뒷면 캔버스 준비 (동일)
    # ...

    # 양면 인쇄 실행 (한 번만)
    ret = self.lib.R600PrintDraw(
        front_img_info.encode('cp949'),
        back_img_info.encode('cp949')
    )

    self.eject_card()
```

---

## 🎯 핵심 발견

### V1도 동일한 구조!

**V1에서도**:
1. ✅ 같은 캔버스에 `draw_watermark()` + `draw_image()` 함께 그림
2. ✅ `R600PrintDraw()` 한 번만 호출
3. ✅ 레이어 분리 없이 한 번에 인쇄

### 프린터 DLL의 역할

```
draw_watermark()  →  R600DrawWaterMark()  →  W (White) 레이어에 그림
draw_image()      →  R600DrawImage()      →  YMC (컬러) 레이어에 그림

R600PrintDraw()   →  프린터가 알아서 W와 YMC 레이어를 분리해서 인쇄
```

**즉, 코드에서 레이어를 분리하는 것이 아니라, 프린터 DLL이 자동으로 처리!**

---

## ❓ 그렇다면 현재 문제는?

### 가능한 원인들

#### 1. 마스크 이미지 문제
```
❌ 잘못된 마스크: 검은색이 아닌 회색, 또는 부분적으로 투명
✅ 올바른 마스크: 순수 검은색(#000000) / 순수 흰색(#FFFFFF)
```

#### 2. 프린터 DLL 함수 호출 순서
```
현재: draw_watermark() → draw_image() → commit_canvas()
문제: 혹시 이 순서가 중요할까? (V1과 동일하지만...)
```

#### 3. 리본 옵션 값
```
현재: self.set_ribbon_option(ribbon_type=1, key=0, value="2")
확인 필요: "2"가 YMCW 모드를 의미하는지?
```

#### 4. 미리보기 표시 방법
```
사용자 요구사항:
- 레이어 모드일 때: "원본 + 마스킹 테두리"로 표시
- 일반 모드일 때: "원본만" 표시

현재 미리보기 코드를 확인 필요!
```

---

## 🤔 사용자님께 질문

### 1. 실제로 문제가 발생하는지 확인

**질문**: 현재 V2 코드로 레이어 인쇄를 실제로 해보셨나요?
- [ ] 네, 해봤고 색상이 이상했습니다
- [ ] 아니오, 아직 테스트 안 했습니다

**만약 색상이 이상했다면**:
- 어떤 색상이었나요? (예: 빨간색 → 어두운 빨간색)
- 전체가 이상했나요, 아니면 일부분만?

### 2. V1에서는 정상이었는지 확인

**질문**: V1 (main 브랜치)에서는 레이어 인쇄가 정상적으로 되었나요?
- [ ] 네, V1에서는 정상이었습니다
- [ ] V1도 테스트 안 해봤습니다
- [ ] V1도 문제가 있었습니다

### 3. 미리보기 표시 요구사항 확인

**질문**: 미리보기에서 보고 싶은 것은?

**레이어 인쇄 모드**:
```
┌─────────────────────┐
│  [원본 이미지]       │
│  ┌───────────────┐  │
│  │ 객체 영역     │  │ ← 이 부분 테두리 표시?
│  └───────────────┘  │
└─────────────────────┘
```

**일반 인쇄 모드**:
```
┌─────────────────────┐
│  [원본 이미지]       │
│  (테두리 없음)       │
└─────────────────────┘
```

이게 맞나요?

---

## 📋 다음 단계

### 시나리오 A: 실제로 문제가 있다면
1. 마스크 이미지 검증 (순수 검은색/흰색인지)
2. 프린터 로그 확인
3. 리본 옵션 값 확인

### 시나리오 B: 미리보기만 수정하면 된다면
1. 미리보기 로직 수정
2. 레이어 모드 시 테두리 표시
3. 일반 모드 시 테두리 없음

### 시나리오 C: 코드 구조 변경이 필요하다면
1. V1과 다른 방식으로 구현
2. 레이어 완전 분리
3. 두 번 인쇄

---

## 💡 결론

**V1 코드와 현재 코드는 동일한 구조입니다!**

따라서:
1. V1에서 정상 작동했다면 → 현재 코드도 정상이어야 함
2. 문제가 있다면 → V1에도 동일한 문제가 있었을 것
3. 미리보기만 다르게 보이길 원하는 것일 수도

**명확히 해주세요:**
- 실제 인쇄 결과가 문제인가요?
- 아니면 미리보기 표시 방법이 문제인가요?
- 둘 다 문제인가요?
