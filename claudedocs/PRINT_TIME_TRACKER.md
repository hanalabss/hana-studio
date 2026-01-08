# 인쇄 예상 시간 시스템

## 개요

실제 인쇄 시간을 기록하고, 이를 바탕으로 예상 시간을 계산하는 시스템.

## 핵심 동작

### 1. 데이터 저장
- **위치**: `C:\ProgramData\HanaStudio\temp\print_times.json`
- **저장 시점**: 인쇄 성공 시 자동 저장
- **최대 기록**: 50개 (오래된 것부터 삭제)

### 2. 예상 시간 계산
```
최근 10개 기록의 평균값 사용
(단면/양면, 일반/레이어 모드별 분리 계산)
```

### 3. 기본값 (기록 없을 때)
| 모드 | 기본 예상 시간 |
|------|---------------|
| 단면 | 40초 |
| 양면 | 75초 (1분 15초) |

## 데이터 흐름

```
┌─────────────────────────────────────────────────────────────────┐
│                        앱 시작                                   │
├─────────────────────────────────────────────────────────────────┤
│  print_times.json 로드 → PrintQuantityPanel 초기 예상 시간 표시  │
│  기록 없으면 기본값 사용 (단면: 40초, 양면: 75초)                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   설정 변경 시                                   │
├─────────────────────────────────────────────────────────────────┤
│  양면/단면 변경 → set_print_settings() → 예상 시간 재계산       │
│  인쇄 모드 변경 → set_print_settings() → 예상 시간 재계산       │
│  매수 변경 → _update_time_estimate() → 예상 시간 재계산         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      인쇄 실행                                   │
├─────────────────────────────────────────────────────────────────┤
│  1. _start_multi_print() → start_print() 호출 (시간 측정 시작)  │
│  2. 프린터 작업 실행                                            │
│  3. on_printer_finished() → end_print() 호출 (시간 기록)        │
│     - 성공 시: 실제 소요 시간 JSON에 저장 + UI 예상 시간 갱신   │
│     - 실패 시: 기록하지 않음                                    │
└─────────────────────────────────────────────────────────────────┘
```

## JSON 구조

```json
{
  "records": [
    {
      "timestamp": 1767842030,
      "duration": 43.8,
      "is_duplex": false,
      "print_mode": "normal"
    },
    {
      "timestamp": 1767842137,
      "duration": 72.2,
      "is_duplex": true,
      "print_mode": "normal"
    }
  ]
}
```

| 필드 | 설명 |
|------|------|
| timestamp | Unix timestamp (초) |
| duration | 실제 소요 시간 (초) |
| is_duplex | 양면 인쇄 여부 |
| print_mode | "normal" 또는 "layered" |

## UI 표시 위치

### 1. PrintQuantityPanel (메인 화면)
```
┌─────────────────────────────┐
│ [DATA] 인쇄 매수            │
├─────────────────────────────┤
│ 📄 매수: [-] [1] [+] 장     │
│ [TIME] 예상 시간: 약 40초   │
└─────────────────────────────┘
```

### 2. 인쇄 확인 모달
```
┌─────────────────────────────┐
│ 카드 인쇄                   │
├─────────────────────────────┤
│ 인쇄 방식: 단면 일반 인쇄   │
│ 인쇄 매수: 3장              │
│ 예상 시간: 약 2분           │  ← 동적 계산
│                             │
│ [예] [아니오]               │
└─────────────────────────────┘
```

## API

```python
from printer.print_time_tracker import print_time_tracker

# 예상 시간 조회 (초)
seconds = print_time_tracker.get_estimated_time(
    is_duplex=True,      # 양면 여부
    print_mode="normal"  # "normal" or "layered"
)

# 포맷된 예상 시간 문자열
time_str = print_time_tracker.get_estimated_time_formatted(
    quantity=3,          # 매수
    is_duplex=True,
    print_mode="normal"
)
# 결과: "약 3분 36초"

# 인쇄 시작 (시간 측정 시작)
print_time_tracker.start_print(is_duplex=False, print_mode="normal")

# 인쇄 완료 (시간 기록)
print_time_tracker.end_print(success=True)

# 통계 조회
stats = print_time_tracker.get_statistics()
# {
#     'total_prints': 15,
#     'single_prints': 14,
#     'duplex_prints': 1,
#     'avg_single': 39.6,
#     'avg_duplex': 72.2,
#     'min_time': 37.5,
#     'max_time': 72.2,
# }

# 기록 초기화
print_time_tracker.clear_records()
```

## 관련 파일

| 파일 | 역할 |
|------|------|
| `printer/print_time_tracker.py` | 시간 측정, 기록 저장/로드, 예상 시간 계산 |
| `ui/components/control_panels.py` | PrintQuantityPanel - 예상 시간 표시 |
| `hana_studio.py` | 시간 기록 연동 (start/end), 모달 예상 시간 |

## 예시

```
저장된 기록 (단면, normal 모드):
[38.9초, 39.0초, 43.8초, 43.5초, 40.2초]

평균: 41.1초

UI 표시:
- 1장 선택 → "예상 시간: 약 41초"
- 3장 선택 → "예상 시간: 약 2분 3초" (41×3=123초)
- 5장 선택 → "예상 시간: 약 3분 25초" (41×5=205초)
```
