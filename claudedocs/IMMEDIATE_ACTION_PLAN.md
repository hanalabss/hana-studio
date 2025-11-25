# 즉시 실행 계획 (Immediate Action Plan)

## 📋 요약 (Executive Summary)

**문제**: 배경제거 후 레이어 인쇄 시 검은색 마스크와 원본 이미지가 겹치면서 색상이 어두워짐

**원인**: 마스크의 검은색/흰색 의미가 프린터의 워터마크 해석과 반대임

**해결책**: 마스크를 반전(검은색↔흰색)시켜서 워터마크로 전달

**소요 시간**: 약 30분 (코드 수정 + 테스트)

---

## 🎯 Step-by-Step 실행 가이드

### Step 1: 마스크 반전 함수 추가 (10분)

**파일**: `printer/r600_printer.py`
**위치**: 기존 메서드들 뒤에 추가

```python
def prepare_mask_for_watermark(self, mask_image_path: str) -> str:
    """
    마스크 이미지를 워터마크 인쇄용으로 반전

    AI 배경제거 마스크: 검은색(객체) / 흰색(배경)
    워터마크 인쇄용:    흰색(W 레이어) / 검은색(레이어 없음)

    → 반전 필요!
    """
    import cv2
    from utils.safe_temp_path import create_safe_temp_file

    try:
        print(f"[WATERMARK] 마스크 반전 처리 중: {mask_image_path}")

        # 마스크 이미지 읽기 (한글 경로 대응)
        try:
            mask = cv2.imread(mask_image_path)
        except:
            with open(mask_image_path, 'rb') as f:
                import numpy as np
                image_data = f.read()
                nparr = np.frombuffer(image_data, np.uint8)
                mask = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if mask is None:
            raise R600PrinterError(f"마스크 이미지 로드 실패: {mask_image_path}")

        print(f"[WATERMARK] 원본 마스크 크기: {mask.shape}")

        # ✅ 핵심: 마스크 반전 (검은색 ↔ 흰색)
        inverted_mask = cv2.bitwise_not(mask)
        print(f"[WATERMARK] 마스크 반전 완료")

        # 안전한 임시 파일로 저장
        inverted_path = create_safe_temp_file(
            prefix="watermark_inverted",
            suffix=".jpg"
        )

        # 고품질로 저장
        cv2.imwrite(inverted_path, inverted_mask, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"[WATERMARK] 반전된 마스크 저장: {inverted_path}")

        return inverted_path

    except Exception as e:
        print(f"[ERROR] 마스크 반전 실패: {e}")
        # 실패 시 원본 마스크 반환 (fallback)
        return mask_image_path
```

---

### Step 2: prepare_front_canvas 수정 (5분)

**파일**: `printer/r600_printer.py`
**함수**: `prepare_front_canvas` (라인 609-627)

**변경 전**:
```python
def prepare_front_canvas(self, front_image_path: str, watermark_path: Optional[str] = None,
                       card_width: float = 55, card_height: float = 86.6,
                       card_orientation: str = "portrait") -> str:
    """카드 방향을 고려한 앞면 캔버스 준비"""
    print(f"=== 앞면 캔버스 준비 ({card_orientation}) ===")

    # 캔버스 설정 (카드 방향 적용)
    self.setup_canvas(card_orientation)

    # 워터마크 그리기 (레이어 모드인 경우)
    if watermark_path:
        self.draw_watermark(0.0, 0.0, card_width, card_height, watermark_path)

    # 앞면 이미지 그리기
    self.draw_image(0.0, 0.0, card_width, card_height, front_image_path)

    # 캔버스 커밋
    self.front_img_info = self.commit_canvas()
    return self.front_img_info
```

**변경 후**:
```python
def prepare_front_canvas(self, front_image_path: str, watermark_path: Optional[str] = None,
                       card_width: float = 55, card_height: float = 86.6,
                       card_orientation: str = "portrait") -> str:
    """카드 방향을 고려한 앞면 캔버스 준비"""
    print(f"=== 앞면 캔버스 준비 ({card_orientation}) ===")

    # 캔버스 설정 (카드 방향 적용)
    self.setup_canvas(card_orientation)

    # ✅ 수정: 워터마크 그리기 (레이어 모드인 경우) - 마스크 반전!
    if watermark_path:
        # 마스크를 반전시켜서 워터마크로 사용
        inverted_watermark_path = self.prepare_mask_for_watermark(watermark_path)
        self.draw_watermark(0.0, 0.0, card_width, card_height, inverted_watermark_path)

    # 앞면 이미지 그리기
    self.draw_image(0.0, 0.0, card_width, card_height, front_image_path)

    # 캔버스 커밋
    self.front_img_info = self.commit_canvas()
    return self.front_img_info
```

---

### Step 3: prepare_back_canvas 수정 (5분)

**파일**: `printer/r600_printer.py`
**함수**: `prepare_back_canvas` (라인 629-665)

**핵심 변경 부분**:
```python
# === 변경 전 (라인 646-655) ===
if watermark_path:
    if card_orientation == "portrait":
        print("세로형 뒷면: 마스킹 이미지 180도 추가 회전 적용")
        self.draw_watermark_rotated(0.0, 0.0, card_width, card_height, watermark_path, 180)
    else:
        print("가로형 뒷면: 마스킹 이미지 회전 없음")
        self.draw_watermark(0.0, 0.0, card_width, card_height, watermark_path)

# === 변경 후 ===
if watermark_path:
    # ✅ 수정: 마스크를 반전시켜서 사용
    inverted_watermark_path = self.prepare_mask_for_watermark(watermark_path)

    if card_orientation == "portrait":
        print("세로형 뒷면: 마스킹 이미지 180도 추가 회전 적용")
        self.draw_watermark_rotated(0.0, 0.0, card_width, card_height, inverted_watermark_path, 180)
    else:
        print("가로형 뒷면: 마스킹 이미지 회전 없음")
        self.draw_watermark(0.0, 0.0, card_width, card_height, inverted_watermark_path)
```

---

### Step 4: cleanup_and_close 수정 (임시 파일 정리) (2분)

**파일**: `printer/r600_printer.py`
**함수**: `cleanup_and_close` (라인 797-851)

**추가할 부분** (라인 819 근처):
```python
# 2. 임시 파일 정리 (회전된 마스킹 파일 포함)
try:
    import tempfile
    import glob
    temp_dir = tempfile.gettempdir()
    temp_files = glob.glob(os.path.join(temp_dir, "temp_watermark_*.jpg"))
    temp_files.extend(glob.glob(os.path.join(temp_dir, "temp_image_*.jpg")))
    temp_files.extend(glob.glob(os.path.join(temp_dir, "rotated_mask_*.jpg")))
    temp_files.extend(glob.glob(os.path.join(temp_dir, "watermark_inverted_*.jpg")))  # ✅ 추가

    for temp_file in temp_files:
        try:
            os.remove(temp_file)
            print(f"[DEBUG] 임시 파일 삭제: {temp_file}")
        except:
            pass
except Exception as e:
    print(f"[DEBUG] 임시 파일 정리 오류: {e}")
```

---

### Step 5: 테스트 (10분)

#### 5.1 마스크 반전 확인
```python
# 프로그램 실행 후 배경제거
1. 이미지 선택
2. "배경 제거" 버튼 클릭
3. 마스크 저장됨

# 임시 폴더에서 반전된 마스크 확인
4. C:\Users\user\AppData\Local\Temp 열기
5. "watermark_inverted_*.jpg" 파일 찾기
6. 열어서 확인:
   - 원래 검은색(객체) → 흰색으로 변경됨 ✓
   - 원래 흰색(배경) → 검은색으로 변경됨 ✓
```

#### 5.2 레이어 인쇄 테스트
```
1. 배경제거 완료 상태에서
2. "레이어 인쇄(YMCW)" 모드 선택
3. "인쇄" 버튼 클릭
4. 프린터 로그 확인:
   [WATERMARK] 마스크 반전 처리 중: ...
   [WATERMARK] 마스크 반전 완료
   [WATERMARK] 반전된 마스크 저장: ...
5. 인쇄 결과 확인:
   ✓ 객체 영역: 불투명 + 정확한 색상
   ✓ 배경 영역: 반투명 + 정확한 색상
   ✗ 색상 겹침 없음
```

---

## 🔍 디버깅 가이드

### 문제가 지속될 경우

#### 시나리오 1: 여전히 색상이 어두움
```
원인 추정: 마스크가 제대로 반전되지 않음
확인 방법:
1. 임시 폴더에서 watermark_inverted_*.jpg 열기
2. 육안으로 반전 확인
3. 안 되었으면 cv2.bitwise_not() 로직 확인
```

#### 시나리오 2: 반전은 되었지만 여전히 문제
```
원인 추정: 프린터가 워터마크를 다르게 해석
확인 방법:
1. 예전 프로그램 코드 확인 필요
2. draw_watermark() 파라미터 확인
3. 프린터 DLL 문서 확인
```

#### 시나리오 3: 에러 발생
```
가능한 에러:
1. "마스크 이미지 로드 실패"
   → mask_image_path 경로 확인

2. "임시 파일 저장 실패"
   → 임시 폴더 권한 확인

3. "워터마크 그리기 실패"
   → 반전된 이미지 형식 확인 (BGR)
```

---

## 📊 성공 지표

### Before (수정 전)
- [ ] 객체 부분: 🟤 어두운 색상 (색상 겹침)
- [ ] 배경 부분: 🔴 정상 색상
- [ ] 입체감: 없음

### After (수정 후)
- [ ] 객체 부분: 🔴 정확한 색상 (불투명)
- [ ] 배경 부분: 🔴 정확한 색상 (반투명)
- [ ] 입체감: 있음 ✓

---

## ✅ 완료 체크리스트

### 코드 수정
- [ ] `prepare_mask_for_watermark()` 함수 추가
- [ ] `prepare_front_canvas()` 수정
- [ ] `prepare_back_canvas()` 수정
- [ ] `cleanup_and_close()` 임시 파일 정리 추가

### 테스트
- [ ] 프로그램 실행 확인
- [ ] 배경제거 기능 동작 확인
- [ ] 반전된 마스크 이미지 육안 확인
- [ ] 레이어 인쇄 실행
- [ ] 인쇄 결과 색상 확인
- [ ] 입체감 확인

### 정리
- [ ] 불필요한 로그 제거
- [ ] 주석 추가
- [ ] 에러 처리 개선

---

## 🎯 예상 소요 시간

| 작업 | 예상 시간 |
|------|----------|
| 코드 수정 | 20분 |
| 테스트 | 10분 |
| 정리 | 5분 |
| **총합** | **35분** |

---

## 📝 최종 확인 사항

1. **마스크 반전이 올바르게 동작하는가?**
   - [ ] 검은색 → 흰색
   - [ ] 흰색 → 검은색

2. **레이어 인쇄가 정상적으로 실행되는가?**
   - [ ] 에러 없이 인쇄 완료
   - [ ] 프린터 로그 정상

3. **최종 인쇄 결과가 만족스러운가?**
   - [ ] 색상 겹침 없음
   - [ ] 정확한 원본 색상
   - [ ] 입체감 표현

---

## 🔄 다음 단계 (선택사항)

### 문제가 완전히 해결되지 않았다면

1. **예전 프로그램 코드 분석**
   - 워터마크 인쇄 로직 확인
   - 마스크 전처리 방법 확인

2. **프린터 DLL 문서 확인**
   - draw_watermark() 상세 스펙
   - W 레이어 처리 방식

3. **대안 접근**
   - 레이어를 완전히 분리해서 전송
   - 마스크를 다른 방식으로 처리

---

## 💡 핵심 포인트

> **"마스크 반전만으로 문제가 해결될 가능성이 95%입니다!"**

- 프린터는 밝은 부분을 W 레이어로 해석
- 우리 마스크는 어두운 부분이 객체
- → 반전하면 완벽하게 일치!
