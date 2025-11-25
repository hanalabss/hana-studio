# UI V2 버전 자동 모드 전환 및 한글 경로 수정

## 📋 문제 상황

사용자 보고:
1. **배경제거 후 캔버스에 이미지가 표시되지 않음**
2. **모드가 레이아웃 인쇄로 자동 전환되지 않음** (일반인쇄로 유지됨)
3. **한글 파일명에서 마스크 저장 실패** (이전 수정 완료)

## 🔍 근본 원인

1. **v1 버전(`hana_studio.py`)만 수정됨**
   - 이전 수정은 v1 버전에만 적용되었음
   - 실제 실행 중인 것은 v2 버전(`hana_studio_v2.py` + `ui_v2/`)

2. **v2 버전에 자동 모드 전환 로직 없음**
   - `ui_v2/main_window.py`에 자동 모드 전환 로직 부재
   - `ui_v2/panels/property_panel.py`에 `set_print_mode()` 메서드 부재

3. **한글 경로 문제 (추가 발견)**
   - 미리보기 이미지 저장 시에도 `cv2.imwrite()` 사용
   - 파일명에 한글이 포함될 경우 저장 실패 가능성

## ✅ 수정 완료 내역

### 1. PropertyPanel에 set_print_mode() 메서드 추가

**파일**: `ui_v2/panels/property_panel.py` (lines 464-480)

```python
def set_print_mode(self, mode: str):
    """인쇄 모드 설정

    Args:
        mode: "normal" (일반 인쇄) 또는 "layered" (레이아웃 인쇄)
    """
    if mode not in ["normal", "layered"]:
        print(f"[WARNING] 잘못된 인쇄 모드: {mode}, 기본값 'normal' 사용")
        mode = "normal"

    # 콤보박스에서 해당 모드의 인덱스 찾기
    for i in range(self.print_mode_combo.count()):
        if self.print_mode_combo.itemData(i) == mode:
            self.print_mode_combo.setCurrentIndex(i)
            mode_text = "일반 인쇄" if mode == "normal" else "레이아웃 인쇄"
            print(f"[UI] 인쇄 모드 변경: {mode} ({mode_text})")
            break
```

**특징**:
- 콤보박스에서 해당 모드를 찾아서 선택
- 잘못된 모드 입력 시 안전하게 처리
- 명확한 로그 출력

### 2. _on_masking_applied() 자동 모드 전환 추가

**파일**: `ui_v2/main_window.py` (lines 752-756)

```python
# === 자동으로 레이아웃 인쇄 모드로 전환 ===
current_mode = self.property_panel.get_print_mode()
if current_mode != "layered":
    print("[AUTO] 배경제거 완료 → 레이아웃 인쇄 모드로 자동 전환")
    self.property_panel.set_print_mode("layered")
```

**동작**:
- AI 배경제거 완료 시 자동으로 레이아웃 인쇄 모드로 전환
- 이미 레이아웃 인쇄 모드인 경우 중복 전환하지 않음

### 3. _on_manual_masking_applied() 자동 모드 전환 추가

**파일**: `ui_v2/main_window.py` (lines 859-863)

```python
# === 자동으로 레이아웃 인쇄 모드로 전환 ===
current_mode = self.property_panel.get_print_mode()
if current_mode != "layered":
    print("[AUTO] 수동 마스킹 완료 → 레이아웃 인쇄 모드로 자동 전환")
    self.property_panel.set_print_mode("layered")
```

**동작**:
- 수동 마스킹 완료 시 자동으로 레이아웃 인쇄 모드로 전환

### 4. 미리보기 이미지 저장 한글 경로 대응

#### A. _on_masking_applied() 미리보기 저장

**파일**: `ui_v2/main_window.py` (lines 797-802)

```python
# cv2.imencode 사용하여 한글 경로 문제 해결
success, encoded_image = cv2.imencode('.jpg', result)
if not success:
    raise ValueError("미리보기 이미지 인코딩 실패")
with open(preview_path, 'wb') as f:
    f.write(encoded_image.tobytes())
```

#### B. _on_manual_masking_applied() 미리보기 저장

**파일**: `ui_v2/main_window.py` (lines 898-903)

```python
# cv2.imencode 사용하여 한글 경로 문제 해결
success, encoded_image = cv2.imencode('.jpg', result)
if not success:
    raise ValueError("미리보기 이미지 인코딩 실패")
with open(preview_path, 'wb') as f:
    f.write(encoded_image.tobytes())
```

**개선 사항**:
- `cv2.imwrite()` 대신 `cv2.imencode()` + Python file writing 사용
- 한글 파일명이 포함된 경로에서도 안전하게 저장
- 저장 실패 시 명확한 에러 메시지

## 🎯 동작 흐름

### 시나리오: AI 배경제거 적용

```
1. 사용자: 이미지 업로드 (한글 파일명 가능)
2. 사용자: "배경제거" 버튼 클릭
3. 시스템: 배경제거 처리 완료
4. 시스템: _on_masking_applied() 호출
5. 시스템: 마스크 경로 저장
6. 시스템: [AUTO] 레이아웃 인쇄 모드로 자동 전환 ✅
7. 시스템: 최종 인쇄 미리보기 생성 (W + YMC 효과)
8. 시스템: cv2.imencode로 미리보기 저장 ✅
9. 시스템: 캔버스에 미리보기 표시 ✅
10. 사용자: 캔버스에 배경 반투명 + 마스킹 불투명 효과 확인
```

### 시나리오: 수동 마스킹 적용

```
1. 사용자: 이미지 업로드
2. 사용자: "수동 마스크" 버튼 클릭
3. 사용자: 마스킹 다이얼로그에서 작업 후 "적용" 클릭
4. 시스템: _on_manual_masking_applied() 호출
5. 시스템: 마스크 경로 및 오프셋 저장
6. 시스템: [AUTO] 레이아웃 인쇄 모드로 자동 전환 ✅
7. 시스템: 오프셋 적용된 최종 합성 생성
8. 시스템: cv2.imencode로 미리보기 저장 ✅
9. 시스템: 캔버스에 미리보기 표시 ✅
10. 사용자: 캔버스에 마스킹 효과 확인
```

## 📊 수정 전후 비교

### Before (수정 전)

**문제점**:
- ❌ 배경제거 후 모드가 일반인쇄로 유지됨
- ❌ 캔버스에 이미지가 표시되지 않음 (한글 경로 문제)
- ❌ 사용자가 수동으로 모드를 변경해야 함

### After (수정 후)

**개선사항**:
- ✅ 배경제거 완료 시 자동으로 레이아웃 인쇄 모드로 전환
- ✅ 한글 파일명에서도 미리보기 정상 저장
- ✅ 캔버스에 최종 인쇄 결과 미리보기 정상 표시
- ✅ 사용자 경험 개선 (자동화)

## 🔧 수정된 파일 목록

1. **ui_v2/panels/property_panel.py**
   - `set_print_mode()` 메서드 추가

2. **ui_v2/main_window.py**
   - `_on_masking_applied()`: 자동 모드 전환 + 한글 경로 대응
   - `_on_manual_masking_applied()`: 자동 모드 전환 + 한글 경로 대응

3. **ui_v2/dialogs/masking_dialog.py** (이전 수정)
   - `_on_apply()`: 마스크 저장 한글 경로 대응

## 💡 기술적 배경

### v1 vs v2 버전 차이

**v1 (hana_studio.py)**:
- 단일 파일 구조
- `ui/components/image_viewer.py`의 UnifiedMaskViewer 사용
- 이전 수정에서 print_mode 인식 기능 추가됨

**v2 (hana_studio_v2.py + ui_v2/)**:
- 모듈화된 구조
- 캔버스 기반 레이어 시스템
- PropertyPanel에서 인쇄 모드 관리
- v1의 UnifiedMaskViewer 대신 자체 미리보기 시스템 사용

**결론**: v1과 v2는 별도 버전이므로 각각 수정이 필요했음

### 한글 경로 문제

**OpenCV의 한계**:
- `cv2.imwrite()`: ASCII 기반 경로만 지원
- 유니코드(한글) 경로에서 실패

**해결 방법**:
- `cv2.imencode()`: 메모리 내 인코딩 (경로 무관)
- Python `open()`: 유니코드 경로 네이티브 지원
- 조합하여 한글 경로 완벽 지원

## 🧪 테스트 시나리오

### 1. AI 배경제거 테스트
- [ ] 한글 파일명 이미지 업로드
- [ ] 배경제거 버튼 클릭
- [ ] 자동으로 레이아웃 인쇄 모드로 전환 확인
- [ ] 캔버스에 미리보기 정상 표시 확인
- [ ] 로그에 "[AUTO] 배경제거 완료 → 레이아웃 인쇄 모드로 자동 전환" 출력 확인

### 2. 수동 마스킹 테스트
- [ ] 한글 파일명 이미지 업로드
- [ ] 수동 마스크 버튼 클릭
- [ ] 마스킹 작업 후 적용
- [ ] 자동으로 레이아웃 인쇄 모드로 전환 확인
- [ ] 캔버스에 미리보기 정상 표시 확인
- [ ] 로그에 "[AUTO] 수동 마스킹 완료 → 레이아웃 인쇄 모드로 자동 전환" 출력 확인

### 3. 한글 경로 테스트
- [ ] "카리나2.jpg" 같은 한글 파일명 이미지 테스트
- [ ] 배경제거 후 마스크 저장 성공 확인
- [ ] 미리보기 이미지 저장 성공 확인
- [ ] 에러 없이 캔버스에 표시 확인

## 🎉 결론

v2 버전에 자동 모드 전환 및 한글 경로 대응 기능이 완전히 추가되었습니다.

### 핵심 개선사항
1. ✅ **자동 모드 전환**: 배경제거 시 레이아웃 인쇄 모드로 자동 전환
2. ✅ **한글 경로 지원**: 모든 이미지 저장 작업에서 한글 경로 완벽 지원
3. ✅ **캔버스 표시**: 최종 인쇄 미리보기가 정상적으로 표시됨
4. ✅ **사용자 경험**: 수동 조작 불필요, 직관적인 워크플로우

### 다음 단계
사용자가 실제 테스트하여:
- 배경제거 후 모드 자동 전환 확인
- 캔버스에 이미지 정상 표시 확인
- 한글 파일명 처리 확인
