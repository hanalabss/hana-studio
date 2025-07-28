# 🎨 Hana Studio 프로젝트 설정 가이드

## 📁 프로젝트 구조

```
hana-studio/
├── 📄 hana_studio.py              # 메인 애플리케이션
├── 📄 printer_integration.py      # 프린터 연동 모듈 (향후 확장용)
├── 📄 setup.py                   # 자동 설치 스크립트
├── 📄 requirements.txt            # Python 패키지 의존성
├── 📄 run_hana_studio.bat         # Windows 실행 스크립트
├── 📄 README.md                  # 사용자 가이드
├── 📄 PROJECT_SETUP.md           # 이 파일 (개발자 가이드)
├── 📁 output/                    # 처리된 이미지 저장
├── 📁 temp/                      # 임시 파일
├── 📁 models/                    # AI 모델 캐시
├── 📁 logs/                      # 로그 파일
└── 📁 dll/                       # 프린터 DLL 파일 (선택사항)
    └── 📄 libDSRetransfer600App.dll
```

## 🚀 빠른 시작

### Windows 사용자 (권장)

1. **모든 파일을 하나의 폴더에 다운로드**
2. **`run_hana_studio.bat` 더블클릭**
3. **완료!** 🎉

### 수동 설정

1. **Python 3.8+ 설치 확인**
   ```bash
   python --version
   ```

2. **가상환경 생성 (권장)**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux  
   source venv/bin/activate
   ```

3. **패키지 설치**
   ```bash
   pip install -r requirements.txt
   ```

4. **애플리케이션 실행**
   ```bash
   python hana_studio.py
   ```

## 🔧 개발 환경 설정

### 필수 패키지
- **PySide6**: Modern GUI framework
- **rembg**: AI background removal
- **opencv-python**: Image processing
- **Pillow**: Image manipulation
- **numpy**: Numerical operations
- **onnxruntime**: AI model runtime

### 개발 도구 (선택사항)
```bash
pip install black flake8 pytest
```

## 🖥️ 애플리케이션 구조

### 메인 클래스들

#### `HanaStudio` (메인 윈도우)
- 전체 UI 관리
- 이벤트 처리
- 상태 관리

#### `ModernButton` (커스텀 버튼)
- 현대적인 스타일링
- Primary/Secondary 테마

#### `ImageViewer` (이미지 뷰어)
- 이미지 표시 및 크기 조정
- 드래그 앤 드롭 지원

#### `ProcessingThread` (백그라운드 처리)
- AI 모델 실행
- 진행상황 리포팅
- 에러 처리

### 주요 기능 흐름

1. **이미지 선택** → `select_image()`
2. **AI 처리** → `process_image()` → `ProcessingThread`
3. **결과 표시** → `on_processing_finished()`
4. **저장** → `export_results()`

## 🎨 UI 디자인 가이드

### 색상 팔레트
- **Primary**: `#4A90E2` (블루)
- **Secondary**: `#F8F9FA` (라이트 그레이)
- **Success**: `#28A745` (그린)
- **Warning**: `#FFC107` (옐로우)
- **Error**: `#DC3545` (레드)

### 폰트
- **Primary**: Segoe UI
- **Monospace**: Consolas

### 스타일링 원칙
- **Modern Flat Design**
- **Consistent Spacing** (15px, 20px, 30px)
- **Rounded Corners** (8px, 10px, 12px)
- **Smooth Gradients**
- **Subtle Shadows**

## 🔌 프린터 연동 (향후 확장)

### DLL 파일 설정
1. `libDSRetransfer600App.dll`을 다음 위치 중 하나에 배치:
   - 메인 폴더 (`./`)
   - `dll/` 폴더
   - `lib/` 폴더

2. 프린터 연동 활성화:
   ```python
   # hana_studio.py에서 import 추가
   from printer_integration import PrinterThread, find_printer_dll
   ```

### 프린터 기능 추가 방법
1. **UI에 프린터 버튼 추가**
2. **프린터 상태 확인 기능**
3. **인쇄 설정 다이얼로그**
4. **프린터 큐 관리**

## 🧪 테스트

### 단위 테스트 실행
```bash
pytest tests/
```

### 수동 테스트 체크리스트
- [ ] 이미지 선택 기능
- [ ] AI 처리 기능
- [ ] 결과 저장 기능
- [ ] 오류 처리
- [ ] UI 반응성

## 📦 배포

### Windows Executable 생성
```bash
pip install pyinstaller
pyinstaller --onefile --windowed --name "HanaStudio" hana_studio.py
```

### macOS App Bundle 생성
```bash
pip install py2app
python setup_py2app.py py2app
```

## 🐛 디버깅

### 로그 확인
- 애플리케이션 내 "처리 로그" 창
- `logs/` 폴더의 로그 파일
- 콘솔 출력 (`python hana_studio.py`)

### 일반적인 문제 해결

#### 1. QImage 관련 오류
```python
# hana_studio.py의 ImageViewer.set_image() 메서드에서
from PySide6.QtGui import QImage  # import 추가 필요
```

#### 2. AI 모델 다운로드 실패
- 인터넷 연결 확인
- 방화벽/프록시 설정 확인
- `~/.cache/rembg/` 폴더 삭제 후 재시도

#### 3. 메모리 부족
- 이미지 크기 축소
- 다른 프로그램 종료
- 가상 메모리 늘리기

#### 4. DLL 로드 실패 (프린터 연동 시)
- DLL 파일 경로 확인
- Visual C++ Redistributable 설치
- 32bit/64bit 호환성 확인

### 개발 모드 실행
```bash
# 상세 로깅과 함께 실행
python -v hana_studio.py

# 프로파일링
python -m cProfile -o profile.stats hana_studio.py
```

## 🔄 업데이트 가이드

### 코드 수정 시 주의사항
1. **스타일 일관성 유지**
2. **에러 처리 추가**
3. **로그 메시지 포함**
4. **타입 힌트 사용**

### 새로운 기능 추가 방법

#### 1. UI 요소 추가
```python
# hana_studio.py에서
def create_new_feature_ui(self, parent_layout):
    new_group = QGroupBox("🆕 새 기능")
    new_layout = QVBoxLayout(new_group)
    
    new_button = ModernButton("새 기능 실행", primary=True)
    new_button.clicked.connect(self.new_feature_action)
    new_layout.addWidget(new_button)
    
    parent_layout.addWidget(new_group)

def new_feature_action(self):
    self.log("새 기능 실행됨")
    # 기능 구현
```

#### 2. 설정 옵션 추가
```python
class Settings:
    def __init__(self):
        self.ai_model = 'isnet-general-use'
        self.output_quality = 95
        self.auto_save = True
        
    def save_to_file(self):
        # JSON으로 설정 저장
        pass
        
    def load_from_file(self):
        # JSON에서 설정 로드
        pass
```

#### 3. 새로운 AI 모델 지원
```python
# ProcessingThread에서
SUPPORTED_MODELS = {
    'isnet-general-use': '범용 모델 (권장)',
    'u2net': '빠른 처리',
    'silueta': '정밀한 처리'
}

def change_ai_model(self, model_name):
    if model_name in SUPPORTED_MODELS:
        self.session = new_session(model_name=model_name)
        self.log(f"AI 모델 변경: {SUPPORTED_MODELS[model_name]}")
```

## 📊 성능 최적화

### 메모리 사용량 최적화
```python
# 큰 이미지 처리 시
def resize_image_if_needed(self, image_path, max_size=2048):
    img = Image.open(image_path)
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        # 임시 파일로 저장
        temp_path = os.path.join('temp', f'resized_{os.path.basename(image_path)}')
        img.save(temp_path)
        return temp_path
    return image_path
```

### UI 반응성 개선
```python
# 장시간 작업 시 UI 업데이트
def long_running_task(self):
    QApplication.processEvents()  # UI 업데이트 허용
    # 작업 수행
    QApplication.processEvents()
```

## 🔐 보안 고려사항

### 파일 경로 검증
```python
def validate_image_path(self, file_path):
    # 안전한 파일 확장자 확인
    safe_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    ext = Path(file_path).suffix.lower()
    
    if ext not in safe_extensions:
        raise ValueError(f"지원하지 않는 파일 형식: {ext}")
    
    # 파일 크기 제한 (예: 50MB)
    if os.path.getsize(file_path) > 50 * 1024 * 1024:
        raise ValueError("파일 크기가 너무 큽니다 (최대 50MB)")
    
    return True
```

### 임시 파일 관리
```python
import tempfile
import atexit

class TempFileManager:
    def __init__(self):
        self.temp_files = []
        atexit.register(self.cleanup)
    
    def create_temp_file(self, suffix='.tmp'):
        temp_file = tempfile.NamedTemporaryFile(
            delete=False, 
            suffix=suffix,
            dir='temp'
        )
        self.temp_files.append(temp_file.name)
        return temp_file.name
    
    def cleanup(self):
        for temp_file in self.temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass
```

## 🌐 다국어 지원 (향후 계획)

### 기본 구조
```python
import gettext

class Translator:
    def __init__(self, language='ko'):
        self.language = language
        self.translator = gettext.translation(
            'hana_studio', 
            localedir='locales', 
            languages=[language],
            fallback=True
        )
        self.translator.install()
    
    def _(self, text):
        return self.translator.gettext(text)

# 사용 예시
t = Translator()
title = t._("이미지 선택")
```

## 📈 모니터링 및 분석

### 사용량 통계 (선택사항)
```python
import json
from datetime import datetime

class UsageStats:
    def __init__(self):
        self.stats_file = 'logs/usage_stats.json'
        self.stats = self.load_stats()
    
    def log_action(self, action):
        timestamp = datetime.now().isoformat()
        if action not in self.stats:
            self.stats[action] = []
        self.stats[action].append(timestamp)
        self.save_stats()
    
    def load_stats(self):
        try:
            with open(self.stats_file, 'r') as f:
                return json.load(f)
        except:
            return {}
    
    def save_stats(self):
        with open(self.stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
```

## 🎯 로드맵

### 단기 목표 (v1.1)
- [ ] 프린터 연동 완성
- [ ] 배치 처리 기능
- [ ] 설정 저장/로드
- [ ] 더 많은 AI 모델 지원

### 중기 목표 (v1.5)
- [ ] 실시간 미리보기
- [ ] 고급 이미지 편집 도구
- [ ] 플러그인 시스템
- [ ] 클라우드 연동

### 장기 목표 (v2.0)
- [ ] 3D 객체 지원
- [ ] 비디오 처리
- [ ] 웹 버전
- [ ] 모바일 앱

---

## 🤝 기여 가이드

### 코드 스타일
- **PEP 8** 준수
- **Type hints** 사용
- **Docstring** 작성
- **단위 테스트** 포함

### Pull Request 프로세스
1. Fork 생성
2. Feature branch 생성
3. 코드 작성 및 테스트
4. Pull Request 생성
5. 코드 리뷰 후 병합

### 이슈 리포팅
- 명확한 제목과 설명
- 재현 단계 포함
- 스크린샷 첨부 (UI 관련)
- 환경 정보 제공

---

**Happy Coding! 🚀**