"""
디버깅용 PyInstaller 빌드 스크립트 - 콘솔 창 포함
오류 메시지를 확인할 수 있도록 수정
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path


class DebugHanaStudioBuilder:
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.build_dir = self.project_root / "build"
        self.dist_dir = self.project_root / "dist"
        
    def clean_build_dirs(self):
        """기존 빌드 디렉토리 정리"""
        print("🧹 기존 빌드 파일 정리...")
        
        for dir_path in [self.build_dir, self.dist_dir]:
            if dir_path.exists():
                shutil.rmtree(dir_path)
    
    def check_dependencies(self):
        """필수 파일 존재 여부 확인"""
        print("📋 필수 파일 확인...")
        
        required_files = [
            "main.py",
            "hana_studio.py", 
            "config.py",
            "libDSRetransfer600App.dll",
            "Retransfer600_SDKCfg.xml",
            "requirements.txt"
        ]
        
        missing_files = []
        for file_name in required_files:
            if not (self.project_root / file_name).exists():
                missing_files.append(file_name)
        
        if missing_files:
            print(f"❌ 누락된 파일: {', '.join(missing_files)}")
            return False
        
        print("✅ 모든 필수 파일 확인 완료")
        return True
    
    def run_debug_pyinstaller(self):
        """디버깅용 PyInstaller 실행 - 콘솔 창 포함"""
        print("🚀 디버깅용 실행파일 빌드 시작...")
        
        cmd = [
            sys.executable, "-m", "PyInstaller",
            "--onefile",           # 단일 파일로 생성
            # "--windowed",        # ❌ 이 옵션 제거! (콘솔 창 표시)
            "--console",           # ✅ 콘솔 창 명시적으로 표시
            "--name", "HanaStudio_Debug",  # 디버그 버전임을 명시
            "--clean", 
            "--noconfirm",
            "--debug", "all"       # ✅ 디버그 정보 모두 출력
        ]
        
        # 아이콘 파일이 있으면 추가
        if (self.project_root / "hana.ico").exists():
            cmd.extend(["--icon", "hana.ico"])
        
        # 필수 데이터 파일들 추가
        data_files = [
            ("libDSRetransfer600App.dll", "."),
            ("Retransfer600_SDKCfg.xml", "."),
            ("config.json", "."),
        ]
        
        if (self.project_root / "hana.ico").exists():
            data_files.append(("hana.ico", "."))
        
        # 디렉토리들 추가
        for dir_name in ["ui", "core", "printer"]:
            if (self.project_root / dir_name).exists():
                data_files.append((dir_name, dir_name))
        
        # EWL 파일들 추가
        for ewl_file in self.project_root.glob("*.EWL"):
            data_files.append((ewl_file.name, "."))
        
        # R600StatusReference 파일들 추가
        for status_file in self.project_root.glob("R600StatusReference*"):
            data_files.append((status_file.name, "."))
        
        # 데이터 파일들을 명령어에 추가
        for src, dst in data_files:
            if os.path.exists(src) or os.path.isdir(src):
                if os.name == 'nt':  # Windows
                    cmd.extend(["--add-data", f"{src};{dst}"])
                else:  # Linux/Mac
                    cmd.extend(["--add-data", f"{src}:{dst}"])
                print(f"✅ 추가: {src}")
        
        # 🔧 완전한 숨김 imports - 모든 의존성 포함
        hidden_imports = [
            # Python 기본 모듈
            "sys", "os", "pathlib", "tempfile", "shutil", "time", "threading",
            "json", "uuid", "ctypes", "subprocess", "glob", "re",
            
            # rembg 관련 - 전체 포함
            "rembg",
            "rembg.bg", 
            "rembg.sessions",
            "rembg.sessions.base",
            "rembg.sessions.isnet",
            "rembg.sessions.u2net",
            "rembg.sessions.u2netp", 
            "rembg.sessions.silueta",
            
            # scipy 전체 - 누락 방지
            "scipy",
            "scipy.ndimage",
            "scipy.ndimage._filters",
            "scipy.ndimage._interpolation", 
            "scipy.ndimage._measurements",
            "scipy.ndimage._morphology",
            "scipy.ndimage.filters",
            "scipy.ndimage.interpolation",
            "scipy.ndimage.measurements", 
            "scipy.ndimage.morphology",
            "scipy.special",
            "scipy.special.cython_special",
            "scipy.sparse",
            "scipy.sparse.csgraph",
            
            # numpy 전체
            "numpy",
            "numpy.core",
            "numpy.core._methods",
            "numpy.lib",
            "numpy.lib.format",
            "numpy.random",
            
            # OpenCV 전체
            "cv2",
            "cv2.cv2",
            
            # PIL/Pillow 전체
            "PIL",
            "PIL.Image",
            "PIL.ImageOps",
            "PIL.ImageFilter", 
            "PIL.ImageDraw",
            "PIL.ImageFont",
            "PIL.ImageTk",
            
            # PySide6 전체
            "PySide6",
            "PySide6.QtCore",
            "PySide6.QtWidgets", 
            "PySide6.QtGui",
            "PySide6.QtOpenGL",
            
            # onnxruntime 전체
            "onnxruntime",
            "onnxruntime.capi",
            "onnxruntime.capi.onnxruntime_pybind11_state",
            
            # scikit-image 관련
            "skimage",
            "skimage.transform",
            "skimage.morphology",
            "skimage.filters",
            "skimage.measure",
            
            # 프로젝트 모듈들
            "config",
            "ui",
            "ui.components",
            "ui.components.modern_button",
            "ui.components.image_viewer", 
            "ui.components.control_panels",
            "ui.main_window",
            "ui.styles",
            "ui.loading_dialog",
            "core",
            "core.image_processor",
            "core.processing_thread",
            "core.file_manager",
            "printer",
            "printer.r600_printer",
            "printer.printer_thread",
            "printer.printer_utils",
            "printer.exceptions",
            "printer.printer_discovery",
        ]
        
        for module in hidden_imports:
            cmd.extend(["--hidden-import", module])
        
        # 최소한의 제외 모듈만
        exclude_modules = [
            "tkinter",
            "matplotlib", 
            "pandas",
            "jupyter",
            "IPython",
            "pytest"
        ]
        
        for module in exclude_modules:
            cmd.extend(["--exclude-module", module])
        
        # 메인 파일 추가
        cmd.append("main.py")
        
        try:
            print("🔧 실행 명령어:")
            print(" ".join(cmd))
            print("\n⏳ 빌드 중... (시간이 오래 걸릴 수 있습니다)")
            
            result = subprocess.run(cmd, check=True, cwd=self.project_root)
            print("✅ PyInstaller 빌드 완료!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ PyInstaller 빌드 실패: {e}")
            return False
    
    def create_debug_package(self):
        """디버깅용 패키지 생성"""
        print("📦 디버깅용 패키지 생성...")
        
        dist_exe = self.dist_dir / "HanaStudio_Debug.exe"
        if not dist_exe.exists():
            print("❌ 디버그 실행파일을 찾을 수 없습니다.")
            return False
        
        # debug 폴더 생성
        debug_dir = self.project_root / "debug"
        debug_dir.mkdir(exist_ok=True)
        
        # 실행파일 복사
        debug_exe = debug_dir / "HanaStudio_Debug.exe"
        shutil.copy2(dist_exe, debug_exe)
        
        # 파일 크기 확인
        file_size_mb = debug_exe.stat().st_size / (1024 * 1024)
        print(f"📊 디버그 실행파일 크기: {file_size_mb:.1f}MB")
        
        # 디버깅용 실행 배치 파일 생성
        debug_batch = debug_dir / "Run_Debug.bat"
        batch_content = '''@echo off
chcp 65001 > nul
echo 🐛 Hana Studio 디버깅 모드
echo ==========================

echo.
echo 📋 환경 정보:
echo OS: %OS%
echo 현재 경로: %CD%
echo 시간: %DATE% %TIME%

echo.
echo 📁 파일 확인:
if exist "HanaStudio_Debug.exe" (
    echo ✅ HanaStudio_Debug.exe 있음
    for %%f in (HanaStudio_Debug.exe) do echo    크기: %%~zf bytes
) else (
    echo ❌ HanaStudio_Debug.exe 없음
    pause
    exit /b 1
)

echo.
echo ⚠️ 디버깅 모드:
echo - 콘솔 창이 표시됩니다
echo - 모든 오류 메시지가 출력됩니다  
echo - 프로그램 종료 시까지 이 창을 닫지 마세요

echo.
echo 🚀 디버깅 실행 중...
echo.

HanaStudio_Debug.exe

echo.
echo 📋 실행 완료
echo 오류가 발생했다면 위의 메시지를 확인하세요
echo.
pause
'''
        
        try:
            with open(debug_batch, 'w', encoding='utf-8-sig') as f:
                f.write(batch_content)
            print(f"✅ 디버깅 배치 파일 생성: {debug_batch}")
        except Exception as e:
            print(f"⚠️ 배치 파일 생성 실패: {e}")
        
        # 디버깅 README 생성
        debug_readme = debug_dir / "DEBUG_README.txt"
        readme_content = '''🐛 Hana Studio 디버깅 버전
=============================

📋 사용 목적
-----------
이 버전은 오류 진단을 위한 디버깅 전용입니다.
콘솔 창이 표시되어 모든 오류 메시지를 확인할 수 있습니다.

🚀 실행 방법  
----------
1. Run_Debug.bat 실행 (권장)
2. 또는 HanaStudio_Debug.exe 직접 실행

📝 오류 확인 방법
---------------
1. 실행 후 콘솔 창의 메시지 확인
2. "ModuleNotFoundError" 등의 오류 메시지 캡처
3. 프로그램이 멈추면 콘솔 창의 마지막 메시지 확인

🔧 일반적인 오류들
----------------
- ModuleNotFoundError: 필요한 모듈이 누락됨
- DLL load failed: DLL 파일 문제
- Permission denied: 파일 권한 문제
- Network error: 인터넷 연결 필요

📞 버그 리포트 시 포함할 정보
--------------------------
1. 콘솔 창의 전체 오류 메시지
2. Windows 버전 (예: Windows 10/11)
3. 실행 환경 (Python 설치 여부)
4. 오류 발생 시점 (시작 시/특정 기능 사용 시)
'''
        
        try:
            with open(debug_readme, 'w', encoding='utf-8-sig') as f:
                f.write(readme_content)
            print("✅ 디버깅 README 생성 완료")
        except Exception as e:
            print(f"⚠️ README 생성 실패: {e}")
        
        print("✅ 디버깅용 패키지 생성 완료")
        return True
    
    def build_debug(self):
        """디버깅용 빌드 프로세스"""
        print("🐛 Hana Studio 디버깅용 빌드 시작")
        print("=" * 60)
        
        if not self.check_dependencies():
            return False
        
        self.clean_build_dirs()
        
        if not self.run_debug_pyinstaller():
            return False
        
        if not self.create_debug_package():
            return False
        
        print("\n" + "=" * 60)
        print("🐛 디버깅 빌드 완료!")
        print(f"📁 디버깅 실행파일: {self.project_root / 'debug' / 'HanaStudio_Debug.exe'}")
        print("")
        print("🧪 테스트 방법:")
        print("   1. debug/Run_Debug.bat 실행 (권장)")
        print("   2. 또는 debug/HanaStudio_Debug.exe 직접 실행")
        print("")
        print("🔍 오류 확인:")
        print("   - 콘솔 창에 모든 오류 메시지가 표시됩니다")
        print("   - 오류 발생 시 메시지를 캡처해주세요")
        print("")
        print("💡 Python 없는 컴퓨터에서 테스트해보세요!")
        
        return True


def main():
    """디버깅 빌드 실행"""
    # PyInstaller 설치 확인
    try:
        import PyInstaller
        print(f"✅ PyInstaller {PyInstaller.__version__}")
    except ImportError:
        print("❌ PyInstaller 미설치")
        print("설치 명령어: pip install pyinstaller")
        return
    
    builder = DebugHanaStudioBuilder()
    success = builder.build_debug()
    
    if not success:
        print("❌ 디버깅 빌드 실패")
        return
    
    print("\n📋 다음 단계:")
    print("1. debug/Run_Debug.bat 실행")
    print("2. 오류 메시지 확인 및 캡처")
    print("3. Python 없는 컴퓨터에서도 테스트")


if __name__ == "__main__":
    main()