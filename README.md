# 🎨 Hana Studio

AI 기반 이미지 배경 제거 도구

## ✨ 주요 기능

- 🤖 **AI 기반 배경 제거**: 최신 AI 모델을 사용한 정확한 배경 제거
- 🖼️ **실시간 미리보기**: 원본, 마스크, 합성 이미지를 동시에 확인
- 🎨 **현대적인 UI**: PySide6 기반의 직관적이고 아름다운 인터페이스
- ⚡ **빠른 처리**: CPU 환경에서도 최적화된 성능
- 💾 **간편한 저장**: 처리된 결과를 원하는 위치에 저장

## 🛠️ 시스템 요구사항

- **Python**: 3.8 이상
- **운영체제**: Windows 10/11, macOS, Linux
- **메모리**: 최소 4GB RAM (8GB 권장)
- **저장공간**: 최소 2GB 여유공간
- **인터넷**: 최초 AI 모델 다운로드 시 필요

## 📦 설치 방법

### 1. 프로젝트 클론 또는 다운로드

```bash
git clone https://github.com/yourusername/hana-studio.git
cd hana-studio
```

### 2. 자동 설치 (권장)

```bash
python setup.py
```

### 3. 수동 설치

```bash
# 가상환경 생성 (선택사항)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

## 🚀 실행 방법

```bash
python hana_studio.py
```

## 📖 사용 방법

### 1. 이미지 선택
- **"이미지 선택"** 버튼을 클릭하여 처리할 이미지를 선택합니다
- 지원 형식: PNG, JPG, JPEG, BMP, TIFF

### 2. 배경 제거
- **"배경 제거 시작"** 버튼을 클릭하여 AI 처리를 시작합니다
- 처리 과정이 진행상황 창에 실시간으로 표시됩니다

### 3. 결과 확인
- **원본 이미지**: 선택한 원본 이미지
- **마스크 이미지**: AI가 생성한 배경 제거 마스크
- **합성 미리보기**: 원본과 마스크가 합성된 미리보기

### 4. 결과 저장
- **"결과 저장"** 버튼을 클릭하여 처리된 이미지들을 저장합니다
- 마스크 이미지와 합성 이미지가 선택한 폴더에 저장됩니다

## 🎛️ 고급 설정

### AI 모델 설정
기본적으로 `isnet-general-use` 모델을 사용하며, 다음과 같은 다른 모델도 선택할 수 있습니다:

```python
# hana_studio.py 파일에서 모델 변경
self.session = new_session(model_name='u2net')  # 다른 모델 예시
```

**사용 가능한 모델들:**
- `isnet-general-use` (기본, 권장)
- `u2net`
- `u2netp`
- `silueta`

### 마스크 임계값 조정
더 정밀한 마스크를 원할 경우 임계값을 조정할 수 있습니다:

```python
# ProcessingThread의 run 메서드에서
alpha_threshold = 45  # 기본값, 0-255 범위에서 조정 가능
```

## 📁 프로젝트 구조

```
hana-studio/
├── hana_studio.py          # 메인 애플리케이션
├── setup.py               # 설치 스크립트
├── requirements.txt       # 필요한 패키지 목록
├── README.md             # 이 파일
├── output/               # 처리된 이미지 저장 폴더
├── temp/                 # 임시 파일 폴더
├── models/               # AI 모델 캐시 폴더
└── logs/                 # 로그 파일 폴더
```

## 🔧 문제 해결

### 일반적인 문제들

#### 1. AI 모델 다운로드 실패
```bash
# 인터넷 연결 확인 후 재시도
# 방화벽/프록시 설정 확인
```

#### 2. 메모리 부족 오류
- 더 작은 크기의 이미지 사용
- 다른 프로그램 종료 후 재시도
- 가상 메모리 설정 확인

#### 3. PySide6 설치 오류
```bash
# Windows의 경우
pip install PySide6 --force-reinstall

# macOS의 경우 (Apple Silicon)
pip install PySide6 --force-reinstall --no-cache-dir
```

#### 4. 이미지 로드 오류
- 지원되는 이미지 형식인지 확인
- 파일 경로에 한글이 포함되어 있지 않은지 확인
- 파일 권한 확인

### 로그 확인
애플리케이션 내 **"처리 로그"** 창에서 상세한 오류 정보를 확인할 수 있습니다.

## 🔄 업데이트

새로운 버전이 출시되면:

```bash
git pull origin main
pip install -r requirements.txt --upgrade
```

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 🙏 감사의 말

- [rembg](https://github.com/danielgatis/rembg) - AI 배경 제거 라이브러리
- [PySide6](https://wiki.qt.io/Qt_for_Python) - GUI 프레임워크
- [OpenCV](https://opencv.org/) - 이미지 처리 라이브러리

## 📞 지원

문제가 발생하거나 문의사항이 있으시면:

- **Issues**: GitHub Issues 페이지에 문제를 등록해주세요
- **Email**: support@hanastudio.com
- **Documentation**: [온라인 문서](https://docs.hanastudio.com)

---

**Hana Studio** - 2024 © Hana Tech. All rights reserved.