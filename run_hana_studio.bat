@echo off
chcp 65001 > nul
echo.
echo 🎨 Hana Studio 시작 중...
echo ==========================================
echo.

REM Python 버전 확인
python --version > nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python이 설치되어 있지 않습니다.
    echo    Python 3.8 이상을 설치해주세요.
    echo    다운로드: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Python 버전 표시
echo 📋 Python 버전 확인:
python --version

REM pip 업그레이드
echo 📦 pip 업그레이드 중...
python -m pip install --upgrade pip

REM 가상환경 존재 확인 및 생성
if not exist "venv" (
    echo 📂 가상환경을 생성하는 중...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo ❌ 가상환경 생성 실패
        pause
        exit /b 1
    )
    echo ✅ 가상환경 생성 완료
)

REM 가상환경 활성화
echo 🔧 가상환경 활성화 중...
call venv\Scripts\activate.bat

REM 패키지 설치 확인
echo 📋 필요한 패키지 확인 중...
pip show PySide6 > nul 2>&1
if %errorlevel% neq 0 (
    echo 📥 패키지 설치 중... (시간이 소요될 수 있습니다)
    echo    호환되는 최신 버전으로 설치합니다...
    pip install --upgrade -r requirements.txt
    if %errorlevel% neq 0 (
        echo ❌ 패키지 설치 실패
        echo 💡 문제 해결 방법:
        echo    1. 인터넷 연결을 확인해주세요
        echo    2. 방화벽/백신 프로그램을 잠시 해제해보세요
        echo    3. Python을 최신 버전으로 업데이트해보세요
        pause
        exit /b 1
    )
    echo ✅ 패키지 설치 완료
)

REM config.py 파일 확인
if not exist "config.py" (
    echo ❌ config.py 파일이 없습니다.
    echo    모든 프로젝트 파일이 같은 폴더에 있는지 확인해주세요.
    pause
    exit /b 1
)

REM Hana Studio 실행
echo.
echo ✅ 준비 완료! Hana Studio를 시작합니다...
echo.
python hana_studio.py

REM 에러 처리
if %errorlevel% neq 0 (
    echo.
    echo ❌ Hana Studio 실행 중 오류가 발생했습니다.
    echo 💡 문제가 지속되면 다음 명령어로 직접 실행해보세요:
    echo    venv\Scripts\activate.bat
    echo    python hana_studio.py
)

REM 종료 메시지
echo.
echo 👋 Hana Studio가 종료되었습니다.
pause