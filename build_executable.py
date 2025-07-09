"""
Hana Studio 실행파일 빌드 스크립트
PyInstaller를 사용하여 배포용 실행파일 생성
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path


class HanaStudioBuilder:
    """Hana Studio 빌드 클래스"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.build_dir = self.project_root / "build"
        self.dist_dir = self.project_root / "dist"
        self.spec_file = self.project_root / "hana_studio.spec"
        
    def clean_build_dirs(self):
        """기존 빌드 디렉토리 정리"""
        print("🧹 기존 빌드 파일 정리 중...")
        
        for dir_path in [self.build_dir, self.dist_dir]:
            if dir_path.exists():
                shutil.rmtree(dir_path)
                print(f"   삭제: {dir_path}")
        
        if self.spec_file.exists():
            self.spec_file.unlink()
            print(f"   삭제: {self.spec_file}")
    
    def check_dependencies(self):
        """필수 파일 존재 여부 확인"""
        print("📋 필수 파일 확인 중...")
        
        required_files = [
            "hana_studio.py",
            "main.py", 
            "config.py",
            "libDSRetransfer600App.dll",
            "Retransfer600_SDKCfg.xml",
            "requirements.txt"
        ]
        
        missing_files = []
        for file_name in required_files:
            file_path = self.project_root / file_name
            if file_path.exists():
                print(f"   ✅ {file_name}")
            else:
                print(f"   ❌ {file_name}")
                missing_files.append(file_name)
        
        # SDK 관련 파일들 확인
        sdk_files = list(self.project_root.glob("*.EWL"))
        if sdk_files:
            print(f"   ✅ EWL 파일: {len(sdk_files)}개")
        else:
            print("   ⚠️ EWL 파일을 찾을 수 없습니다")
        
        # R600StatusReference 파일들 확인
        status_files = list(self.project_root.glob("R600StatusReference*"))
        if status_files:
            print(f"   ✅ R600StatusReference: {len(status_files)}개")
        else:
            print("   ⚠️ R600StatusReference 파일을 찾을 수 없습니다")
        
        if missing_files:
            print(f"\n❌ 누락된 파일들: {', '.join(missing_files)}")
            return False
        
        print("✅ 모든 필수 파일 확인 완료")
        return True
    
    def create_pyinstaller_spec(self):
        """PyInstaller spec 파일 생성"""
        print("📝 PyInstaller spec 파일 생성 중...")
        
        # 추가 데이터 파일들 수집
        added_files = []
        
        # DLL 파일
        if (self.project_root / "libDSRetransfer600App.dll").exists():
            added_files.append("('libDSRetransfer600App.dll', '.')")
        
        # XML 설정 파일
        if (self.project_root / "Retransfer600_SDKCfg.xml").exists():
            added_files.append("('Retransfer600_SDKCfg.xml', '.')")
        
        # JSON 설정 파일
        if (self.project_root / "config.json").exists():
            added_files.append("('config.json', '.')")
        
        # EWL 파일들
        ewl_files = list(self.project_root.glob("*.EWL"))
        for ewl_file in ewl_files:
            added_files.append(f"('{ewl_file.name}', '.')")
        
        # R600StatusReference 파일들
        status_files = list(self.project_root.glob("R600StatusReference*"))
        for status_file in status_files:
            added_files.append(f"('{status_file.name}', '.')")
        
        # UI 디렉토리
        if (self.project_root / "ui").exists():
            added_files.append("('ui', 'ui')")
        
        # core 디렉토리
        if (self.project_root / "core").exists():
            added_files.append("('core', 'core')")
        
        # printer 디렉토리
        if (self.project_root / "printer").exists():
            added_files.append("('printer', 'printer')")
        
        # 기본 디렉토리들
        for dir_name in ["output", "temp", "models", "logs"]:
            dir_path = self.project_root / dir_name
            if not dir_path.exists():
                dir_path.mkdir(exist_ok=True)
            added_files.append(f"('{dir_name}', '{dir_name}')")
        
        # 추가 데이터 파일들을 올바른 형식으로 포맷팅
        added_files_str = ',\n    '.join(added_files) if added_files else ""
        
        spec_content = f"""# -*- mode: python ; coding: utf-8 -*-
import os

block_cipher = None

# 추가 데이터 파일들
added_files = [
    {added_files_str}
]

# 숨겨진 imports - AI 모델 및 의존성
hidden_imports = [
    'rembg',
    'rembg.models',
    'rembg.sessions',
    'onnxruntime',
    'onnxruntime.providers',
    'onnxruntime.providers.cpu',
    'cv2',
    'numpy',
    'PIL',
    'PIL.Image',
    'PIL.ImageQt',
    'PySide6.QtCore',
    'PySide6.QtGui', 
    'PySide6.QtWidgets',
    'config',
    'ui',
    'core',
    'printer',
    'ctypes',
    'cffi',
    'tempfile',
    'pathlib',
    'io',
    'threading',
    'time',
    'json'
]

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=added_files,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'matplotlib',
        'scipy',
        'pandas',
        'jupyter',
        'IPython',
        'test',
        'tests',
        'unittest'
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='HanaStudio',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # GUI 모드
    disable_windowed_traceback=False,
    icon='icon.ico' if os.path.exists('icon.ico') else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='HanaStudio',
)
"""
        
        with open(self.spec_file, 'w', encoding='utf-8') as f:
            f.write(spec_content)
        
        print(f"✅ Spec 파일 생성 완료: {self.spec_file}")
    
    def run_pyinstaller(self):
        """PyInstaller 실행"""
        print("🚀 PyInstaller 빌드 시작...")
        
        cmd = [
            sys.executable, "-m", "PyInstaller",
            "--clean",
            str(self.spec_file)
        ]
        
        print(f"실행 명령어: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, cwd=self.project_root)
            print("✅ PyInstaller 빌드 완료!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ PyInstaller 빌드 실패: {e}")
            return False
    
    def create_installer_assets(self):
        """설치 관련 파일들 생성"""
        print("📦 설치 관련 파일들 생성 중...")
        
        dist_hana = self.dist_dir / "HanaStudio"
        
        if not dist_hana.exists():
            print("❌ 빌드된 실행파일 디렉토리를 찾을 수 없습니다.")
            return False
        
        # README 파일
        readme_content = """# Hana Studio - AI 기반 이미지 배경 제거 도구

## 설치 방법
1. HanaStudio.exe를 실행하세요
2. 프린터 DLL이 같은 폴더에 있는지 확인하세요
3. 인터넷 연결이 필요합니다 (AI 모델 다운로드)

## 시스템 요구사항
- Windows 10/11 (64bit)
- 메모리: 최소 4GB RAM (8GB 권장)
- 저장공간: 최소 2GB
- 인터넷 연결 (최초 실행 시)

## 사용법
1. 앞면 이미지를 선택하세요
2. 양면 인쇄 시 뒷면 이미지도 선택하세요
3. "배경 제거 시작" 버튼을 클릭하세요
4. 결과를 확인하고 저장하거나 인쇄하세요

## 문제 해결
- 프린터 연결 오류: libDSRetransfer600App.dll 파일 확인
- AI 모델 다운로드 실패: 인터넷 연결 및 방화벽 확인
- 메모리 부족: 다른 프로그램 종료 후 재시도

## 지원
문의사항이 있으시면 고객지원팀에 연락하세요.
"""
        
        readme_path = dist_hana / "README.txt"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        # 버전 정보 파일
        version_content = """Hana Studio v1.0.0
빌드 날짜: {build_date}
AI 모델: isnet-general-use
프린터 지원: RTAI LUKA R600
기능: 양면 인쇄, 여러장 인쇄, 레이어 인쇄
""".format(build_date=str(Path(__file__).stat().st_mtime))
        
        version_path = dist_hana / "VERSION.txt"
        with open(version_path, 'w', encoding='utf-8') as f:
            f.write(version_content)
        
        print("✅ 설치 관련 파일들 생성 완료")
        return True
    
    def create_batch_runner(self):
        """실행용 배치 파일 생성"""
        print("📝 배치 실행 파일 생성 중...")
        
        dist_hana = self.dist_dir / "HanaStudio"
        
        batch_content = """@echo off
chcp 65001 > nul
echo.
echo 🎨 Hana Studio 시작 중...
echo.

REM 관리자 권한 확인
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️ 관리자 권한으로 실행하는 것을 권장합니다.
    echo   프린터 연결이 원활하지 않을 수 있습니다.
    echo.
    timeout /t 3 /nobreak > nul
)

REM DLL 파일 확인
if not exist "libDSRetransfer600App.dll" (
    echo ❌ 프린터 DLL 파일이 없습니다.
    echo    libDSRetransfer600App.dll 파일을 확인하세요.
    pause
    exit /b 1
)

REM 실행파일 확인
if not exist "HanaStudio.exe" (
    echo ❌ HanaStudio.exe 파일이 없습니다.
    pause
    exit /b 1
)

echo ✅ 파일 확인 완료, Hana Studio를 시작합니다...
echo.

REM Hana Studio 실행
start "" "HanaStudio.exe"

REM 에러 체크
if %errorlevel% neq 0 (
    echo ❌ 실행 중 오류가 발생했습니다.
    echo 💡 문제 해결 방법:
    echo    1. 관리자 권한으로 실행해보세요
    echo    2. 백신 프로그램을 잠시 해제해보세요
    echo    3. Windows Defender 예외 목록에 추가해보세요
    pause
    exit /b 1
)

echo 프로그램이 시작되었습니다.
"""
        
        batch_path = dist_hana / "Run_HanaStudio.bat"
        with open(batch_path, 'w', encoding='utf-8') as f:
            f.write(batch_content)
        
        print(f"✅ 배치 파일 생성 완료: {batch_path}")
    
    def optimize_distribution(self):
        """배포 파일 최적화"""
        print("⚡ 배포 파일 최적화 중...")
        
        dist_hana = self.dist_dir / "HanaStudio"
        
        if not dist_hana.exists():
            return False
        
        # 불필요한 파일들 제거
        unnecessary_patterns = [
            "*.pdb",
            "*.pyc", 
            "*.pyo",
            "*__pycache__*",
            "*.dist-info",
            "test_*",
            "*test*",
            "example*"
        ]
        
        removed_count = 0
        for pattern in unnecessary_patterns:
            for file_path in dist_hana.rglob(pattern):
                if file_path.is_file():
                    file_path.unlink()
                    removed_count += 1
                elif file_path.is_dir():
                    shutil.rmtree(file_path, ignore_errors=True)
                    removed_count += 1
        
        print(f"   정리된 파일/폴더: {removed_count}개")
        
        # 압축 가능한 파일들 확인
        total_size = sum(f.stat().st_size for f in dist_hana.rglob('*') if f.is_file())
        print(f"   총 크기: {total_size / (1024*1024):.1f} MB")
        
        print("✅ 배포 파일 최적화 완료")
        return True
    
    def create_zip_package(self):
        """ZIP 패키지 생성"""
        print("📦 ZIP 패키지 생성 중...")
        
        dist_hana = self.dist_dir / "HanaStudio"
        zip_path = self.dist_dir / "HanaStudio_v1.0.0_Portable.zip"
        
        if not dist_hana.exists():
            print("❌ 배포 폴더를 찾을 수 없습니다.")
            return False
        
        try:
            shutil.make_archive(
                zip_path.with_suffix(''), 
                'zip', 
                dist_hana.parent, 
                dist_hana.name
            )
            print(f"✅ ZIP 패키지 생성 완료: {zip_path}")
            
            # 파일 크기 표시
            zip_size = zip_path.stat().st_size / (1024*1024)
            print(f"   패키지 크기: {zip_size:.1f} MB")
            
            return True
        except Exception as e:
            print(f"❌ ZIP 패키지 생성 실패: {e}")
            return False
    
    def build(self):
        """전체 빌드 프로세스 실행"""
        print("🏗️ Hana Studio 빌드 시작")
        print("=" * 50)
        
        # 1. 사전 점검
        if not self.check_dependencies():
            print("\n❌ 빌드 중단: 필수 파일이 누락되었습니다.")
            return False
        
        # 2. 기존 빌드 정리
        self.clean_build_dirs()
        
        # 3. Spec 파일 생성
        self.create_pyinstaller_spec()
        
        # 4. PyInstaller 실행
        if not self.run_pyinstaller():
            print("\n❌ 빌드 실패")
            return False
        
        # 5. 추가 파일들 생성
        self.create_installer_assets()
        self.create_batch_runner()
        
        # 6. 최적화
        self.optimize_distribution()
        
        # 7. ZIP 패키지 생성
        self.create_zip_package()
        
        print("\n" + "=" * 50)
        print("🎉 Hana Studio 빌드 완료!")
        print(f"📁 배포 폴더: {self.dist_dir / 'HanaStudio'}")
        print(f"📦 ZIP 파일: {self.dist_dir / 'HanaStudio_v1.0.0_Portable.zip'}")
        print("\n💡 사용 방법:")
        print("   1. HanaStudio 폴더를 대상 컴퓨터에 복사")
        print("   2. Run_HanaStudio.bat를 실행")
        print("   3. 또는 HanaStudio.exe를 직접 실행")
        
        return True


def main():
    """메인 함수"""
    print("🎨 Hana Studio 실행파일 빌더")
    print("PyInstaller를 사용하여 배포용 실행파일을 생성합니다.")
    print("")
    
    # PyInstaller 설치 확인
    try:
        import PyInstaller
        print(f"✅ PyInstaller 버전: {PyInstaller.__version__}")
    except ImportError:
        print("❌ PyInstaller가 설치되어 있지 않습니다.")
        print("다음 명령어로 설치하세요: pip install pyinstaller")
        return
    
    # 빌드 시작
    builder = HanaStudioBuilder()
    success = builder.build()
    
    if success:
        print("\n✅ 빌드가 성공적으로 완료되었습니다!")
    else:
        print("\n❌ 빌드가 실패했습니다.")
        print("오류를 확인하고 다시 시도해주세요.")


if __name__ == "__main__":
    main()