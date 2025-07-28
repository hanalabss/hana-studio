"""
스마트 콘솔 모드 PyInstaller 실행파일 빌드 스크립트
- 시작 시 CMD 창으로 오류 확인 가능
- 프로그램 정상 실행되면 CMD 창 자동 숨김
- 오류 발생 시에만 CMD 창 유지
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path


class SmartConsoleHanaStudioBuilder:
    
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

    def create_smart_console_main(self):
        """현대적인 설치 다이얼로그를 사용하는 스마트 콘솔 메인 파일 생성"""
        print("📝 현대적인 스마트 콘솔 래퍼 생성...")
        
        smart_main_content = '''"""
현대적인 스마트 콘솔 모드 메인 파일
- 시작 시 콘솔 창 표시로 오류 확인 가능
- 현대적인 설치 다이얼로그 사용
- 프로그램 정상 실행 시 콘솔 창 숨김
"""

import sys
import os
import time
import ctypes
from ctypes import wintypes

def hide_console():
    """콘솔 창 숨기기 (Windows 전용)"""
    try:
        if os.name == 'nt':  # Windows에서만 실행
            kernel32 = ctypes.windll.kernel32
            user32 = ctypes.windll.user32
            
            console_window = kernel32.GetConsoleWindow()
            if console_window:
                # SW_HIDE = 0 (숨김)
                user32.ShowWindow(console_window, 0)
                print("✅ 콘솔 창 숨김 완료")
        else:
            print("⚠️ Windows가 아닌 환경에서는 콘솔 숨김을 지원하지 않습니다.")
    except Exception as e:
        print(f"⚠️ 콘솔 숨김 실패: {e}")

def show_console():
    """콘솔 창 표시 (오류 시 호출)"""
    try:
        if os.name == 'nt':
            kernel32 = ctypes.windll.kernel32
            user32 = ctypes.windll.user32
            
            console_window = kernel32.GetConsoleWindow()
            if console_window:
                # SW_SHOW = 5 (표시)
                user32.ShowWindow(console_window, 5)
                # 창을 맨 앞으로
                user32.SetForegroundWindow(console_window)
    except Exception as e:
        print(f"⚠️ 콘솔 표시 실패: {e}")

def setup_environment():
    """환경 변수 설정"""
    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "0"
    os.environ["QT_SCALE_FACTOR_ROUNDING_POLICY"] = "Floor"
    os.environ["QT_FONT_DPI"] = "96"
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "0"

def main():
    """메인 실행 함수 - 현대적인 설치 다이얼로그 사용"""
    startup_success = False
    
    try:
        print("🎨 Hana Studio 시작 중...")
        print("=" * 50)
        
        # 환경 정보 출력 (간단히)
        print(f"Python 버전: {sys.version.split()[0]}")
        print(f"작업 디렉토리: {os.getcwd()}")
        
        # PyInstaller 환경 확인
        if getattr(sys, 'frozen', False):
            print("✅ 배포된 실행파일에서 실행 중")
            if hasattr(sys, '_MEIPASS'):
                print(f"임시 디렉토리: {sys._MEIPASS}")
        else:
            print("🔧 개발 환경에서 실행 중")
        
        print("=" * 50)
        print()
        
        # 🚀 환경 설정 (빠르게)
        setup_environment()
        
        # PySide6 애플리케이션 import (빠르게)
        print("📦 UI 프레임워크 로딩 중...")
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QGuiApplication, QIcon
        
        print("✅ UI 프레임워크 로딩 완료")
        
        # 기본 설정 로딩
        from config import config, AppConstants, get_resource_path
        
        # Qt 애플리케이션 설정 (빠르게)
        try:
            QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
                Qt.HighDpiScaleFactorRoundingPolicy.Floor
            )
        except AttributeError:
            pass
        
        app = QApplication(sys.argv)
        
        # 앱 정보 설정
        app.setApplicationName(AppConstants.APP_NAME)
        app.setApplicationVersion(AppConstants.APP_VERSION)
        app.setOrganizationName(AppConstants.APP_AUTHOR)
        
        # 아이콘 설정
        try:
            icon_path = get_resource_path("hana.ico")
            if os.path.exists(icon_path):
                app.setWindowIcon(QIcon(icon_path))
        except Exception as e:
            print(f"⚠️ 아이콘 설정 실패: {e}")
        
        # 기본 스타일 설정
        app.setStyle('Fusion')
        
        print("✅ 애플리케이션 초기화 완료")
        
        # 🎯 현대적인 설치 다이얼로그 표시
        print("🚀 AI 엔진 설치 다이얼로그 시작...")
        
        # 콘솔 창을 잠시 숨김 (설치 다이얼로그 표시 중)
        hide_console()
        
        # 현대적인 설치 다이얼로그 사용
        from ui.installation_dialog import show_installation_dialog
        installation_success = show_installation_dialog()
        
        if not installation_success:
            print("❌ 설치 취소 또는 실패")
            show_console()  # 오류 시 콘솔 다시 표시
            input("Enter 키를 눌러 종료...")
            sys.exit(1)
        
        print("✅ AI 엔진 설치 완료")
        
        # 🚀 메인 애플리케이션 로딩 (지연 로딩)
        print("🖥️ 메인 애플리케이션 로딩 중...")
        
        # 지연 import로 빠른 시작
        from hana_studio import HanaStudio
        from ui.styles import get_light_palette
        
        # 스타일 적용
        app.setPalette(get_light_palette())
        
        # 메인 윈도우 생성
        print("🖥️ 메인 윈도우 생성 중...")
        window = HanaStudio()
        
        print("✅ 모든 초기화 완료!")
        print("🎉 Hana Studio 실행 준비 완료")
        print()
        
        startup_success = True
        
        # 메인 윈도우 표시 및 실행
        window.show()
        sys.exit(app.exec())
        
    except KeyboardInterrupt:
        print("\\n⚠️ 사용자에 의해 중단되었습니다.")
        show_console()
        input("Enter 키를 눌러 종료...")
        sys.exit(0)
        
    except ImportError as e:
        print(f"\\n❌ 모듈 로드 실패: {e}")
        print("=" * 50)
        import traceback
        traceback.print_exc()
        print("=" * 50)
        
        # 오류 발생 시 콘솔 창 유지/표시
        show_console()
        print("\\n🐛 필수 구성요소를 찾을 수 없습니다!")
        print("프로그램을 다시 설치하거나 지원팀에 문의해주세요.")
        input("\\nEnter 키를 눌러 종료...")
        sys.exit(1)
        
    except Exception as e:
        print(f"\\n❌ 시작 중 오류 발생: {e}")
        print("=" * 50)
        import traceback
        traceback.print_exc()
        print("=" * 50)
        
        # 오류 발생 시 콘솔 창 유지/표시
        show_console()
        print("\\n🐛 예상치 못한 오류가 발생했습니다!")
        print("위의 오류 메시지를 확인하고 지원팀에 문의해주세요.")
        input("\\nEnter 키를 눌러 종료...")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
        
        # 새로운 메인 파일 저장
        smart_main_path = self.project_root / "main_smart_console.py"
        with open(smart_main_path, 'w', encoding='utf-8') as f:
            f.write(smart_main_content)
        
        print(f"✅ 현대적인 스마트 콘솔 메인 파일 생성: {smart_main_path}")
        return smart_main_path

    def run_smart_console_pyinstaller(self):
        """스마트 콘솔 모드 PyInstaller 실행"""
        print("🚀 스마트 콘솔 모드 빌드 시작...")
        
        # 스마트 콘솔 메인 파일 생성
        smart_main_path = self.create_smart_console_main()
        
        cmd = [
            sys.executable, "-m", "PyInstaller",
            "--onefile",           # 단일 파일로 생성
            "--console",           # ✅ 콘솔 모드로 시작 (나중에 숨김)
            "--name", "HanaStudio",
            "--clean", 
            "--noconfirm"
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
        
        # 완전한 숨김 imports
        hidden_imports = [
            # Python 기본 모듈
            "sys", "os", "pathlib", "tempfile", "shutil", "time", "threading",
            "json", "uuid", "ctypes", "subprocess", "glob", "re",
            
            # rembg 관련
            "rembg",
            "rembg.bg", 
            "rembg.sessions",
            "rembg.sessions.base",
            "rembg.sessions.isnet",
            "rembg.sessions.u2net",
            "rembg.sessions.u2netp", 
            "rembg.sessions.silueta",
            
            # scipy 전체
            "scipy",
            "scipy.ndimage",
            "scipy.ndimage._filters",
            "scipy.ndimage._interpolation", 
            "scipy.ndimage._measurements",
            "scipy.ndimage._morphology",
            "scipy.special",
            "scipy.special.cython_special",
            
            # numpy
            "numpy",
            "numpy.core._methods",
            "numpy.lib.format",
            
            # OpenCV
            "cv2",
            
            # PIL/Pillow
            "PIL",
            "PIL.Image",
            "PIL.ImageOps",
            "PIL.ImageFilter", 
            
            # PySide6
            "PySide6.QtCore",
            "PySide6.QtWidgets", 
            "PySide6.QtGui",
            
            # onnxruntime
            "onnxruntime",
            "onnxruntime.capi",
            
            # scikit-image
            "skimage",
            "skimage.transform",
            "skimage.morphology",
            
            # 프로젝트 모듈들
            "config",
            "ui",
            "ui.components",
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
        
        # 제외 모듈
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
        
        # 스마트 콘솔 메인 파일 사용
        cmd.append(str(smart_main_path))
        
        try:
            print("🔧 실행 명령어:")
            print(" ".join(cmd[:10]) + " ...")
            print("⏳ 빌드 중...")
            
            result = subprocess.run(cmd, check=True, cwd=self.project_root)
            print("✅ PyInstaller 빌드 완료!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ PyInstaller 빌드 실패: {e}")
            return False
    
    def create_smart_package(self):
        """스마트 콘솔 패키지 생성"""
        print("📦 스마트 콘솔 패키지 생성...")
        
        dist_exe = self.dist_dir / "HanaStudio.exe"
        if not dist_exe.exists():
            print("❌ 실행파일을 찾을 수 없습니다.")
            return False
        
        # release 폴더 생성
        release_dir = self.project_root / "release"
        release_dir.mkdir(exist_ok=True)
        
        # 실행파일 복사
        release_exe = release_dir / "HanaStudio.exe"
        shutil.copy2(dist_exe, release_exe)
        
        # 파일 크기 확인
        file_size_mb = release_exe.stat().st_size / (1024 * 1024)
        print(f"📊 실행파일 크기: {file_size_mb:.1f}MB")
        
        # 사용 설명서 생성
        readme_content = '''🎨 Hana Studio - 스마트 콘솔 모드
=====================================

🚀 실행 방법
----------
HanaStudio.exe를 더블클릭하여 실행

⭐ 스마트 콘솔 모드 특징
---------------------
✅ 시작 시 콘솔 창이 표시되어 오류 확인 가능
✅ 프로그램 정상 실행 시 콘솔 창 자동 숨김
✅ 오류 발생 시에만 콘솔 창 유지
✅ Python 설치 없이 독립 실행 가능

🔧 시스템 요구사항
---------------
- Windows 10/11 (64bit)
- 메모리: 최소 4GB RAM 권장  
- 저장공간: 최소 1GB 여유공간
- 인터넷 연결: 최초 AI 모델 다운로드 시 필요

🐛 문제 해결
----------
- 실행 안됨: 콘솔 창의 오류 메시지 확인
- 콘솔 창이 계속 보임: 오류가 발생한 것이므로 메시지 확인
- 느린 실행: 최초 실행 시 AI 모델 다운로드 정상

📞 지원
------
문제 발생 시 콘솔 창의 오류 메시지와 함께 문의해주세요.
'''
        
        with open(release_dir / "README.txt", 'w', encoding='utf-8-sig') as f:
            f.write(readme_content)
        
        print("✅ 스마트 콘솔 패키지 생성 완료")
        return True
    
    def build(self):
        """전체 빌드 프로세스"""
        print("🧠 Hana Studio 스마트 콘솔 모드 빌드 시작")
        print("=" * 60)
        
        if not self.check_dependencies():
            return False
        
        self.clean_build_dirs()
        
        if not self.run_smart_console_pyinstaller():
            return False
        
        if not self.create_smart_package():
            return False
        
        print("\n" + "=" * 60)
        print("🎉 스마트 콘솔 빌드 완료!")
        print(f"📁 실행파일: {self.project_root / 'release' / 'HanaStudio.exe'}")
        print("")
        print("⭐ 스마트 콘솔 모드:")
        print("   - 시작 시 콘솔 창으로 오류 확인")
        print("   - 정상 실행 시 콘솔 창 자동 숨김")
        print("   - 오류 시에만 콘솔 창 유지")
        print("")
        print("🧪 테스트:")
        print("   1. release/HanaStudio.exe 실행")
        print("   2. 콘솔 창이 잠깐 나타났다가 사라지는지 확인")
        print("   3. Python 없는 컴퓨터에서도 테스트")
        
        return True


def main():
    """스마트 콘솔 빌드 실행"""
    # PyInstaller 설치 확인
    try:
        import PyInstaller
        print(f"✅ PyInstaller {PyInstaller.__version__}")
    except ImportError:
        print("❌ PyInstaller 미설치")
        print("설치 명령어: pip install pyinstaller")
        return
    
    builder = SmartConsoleHanaStudioBuilder()
    success = builder.build()
    
    if not success:
        print("❌ 빌드 실패")
        return
    
    # 임시 파일 정리
    smart_main_path = builder.project_root / "main_smart_console.py"
    if smart_main_path.exists():
        smart_main_path.unlink()
        print("🧹 임시 파일 정리 완료")


if __name__ == "__main__":
    main()