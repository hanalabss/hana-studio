"""
빠른 시작용 PyInstaller 빌드 스크립트
--onedir 모드로 시작 시간 대폭 단축
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path


class FastHanaStudioBuilder:
    
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

    def run_fast_pyinstaller(self):
        """빠른 시작을 위한 PyInstaller 실행 (onedir 모드)"""
        print("🚀 빠른 시작용 실행파일 빌드 시작 (폴더 형태)...")
        
        cmd = [
            sys.executable, "-m", "PyInstaller",
            "--onedir",            # ✅ 폴더 형태 (빠른 시작)
            "--windowed",          # GUI 모드 (CMD 창 숨김)
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
        
        # 필수 hidden imports + 누락된 종속성 포함
        essential_imports = [
            # 핵심 모듈들
            "PySide6.QtCore",
            "PySide6.QtWidgets", 
            "PySide6.QtGui",
            
            # rembg 핵심
            "rembg",
            "rembg.sessions.isnet",
            "rembg.sessions.base",
            "rembg.sessions.u2net",
            
            # pkg_resources 관련 누락 모듈들 (🔧 추가)
            "pkg_resources",
            "jaraco",
            "jaraco.text",
            "jaraco.functools",
            "jaraco.context",
            "jaraco.collections",
            "more_itertools",
            "zipp",
            "importlib_metadata",
            
            # setuptools 관련
            "setuptools",
            "setuptools.extern",
            "setuptools._vendor",
            
            # 기타 자주 누락되는 모듈들
            "distutils",
            "distutils.util",
            
            # 프로젝트 모듈들
            "config",
            "ui.installation_dialog",
            "core.image_processor",
            "printer.r600_printer",
        ]
        
        for module in essential_imports:
            cmd.extend(["--hidden-import", module])
        
        # 불필요한 모듈 제외 (빌드 속도 향상) - setuptools는 제외하지 않음
        exclude_modules = [
            "tkinter", "matplotlib", "pandas", "jupyter", 
            "IPython", "pytest", "pip"
        ]
        
        for module in exclude_modules:
            cmd.extend(["--exclude-module", module])
        
        # 메인 파일 추가
        cmd.append("main.py")
        
        try:
            print("🔧 실행 명령어:")
            print(" ".join(cmd[:10]) + " ...")
            print("⏳ 빌드 중... (폴더 형태로 빠른 빌드)")
            
            result = subprocess.run(cmd, check=True, cwd=self.project_root)
            print("✅ PyInstaller 빌드 완료!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ PyInstaller 빌드 실패: {e}")
            return False
    
    def create_fast_package(self):
        """빠른 시작용 패키지 생성"""
        print("📦 빠른 시작용 패키지 생성...")
        
        dist_folder = self.dist_dir / "HanaStudio"
        if not dist_folder.exists():
            print("❌ 빌드 폴더를 찾을 수 없습니다.")
            return False
        
        # release 폴더 생성
        release_dir = self.project_root / "release_fast"
        if release_dir.exists():
            shutil.rmtree(release_dir)
        
        # 전체 폴더 복사
        shutil.copytree(dist_folder, release_dir)
        
        # 폴더 크기 확인
        total_size = sum(f.stat().st_size for f in release_dir.rglob('*') if f.is_file())
        size_mb = total_size / (1024 * 1024)
        print(f"📊 빌드 폴더 크기: {size_mb:.1f}MB")
        
        # 실행 배치 파일 생성 (더블클릭 편의용)
        batch_file = release_dir / "Hana Studio 실행.bat"
        batch_content = '''@echo off
cd /d "%~dp0"
start "" "HanaStudio.exe"
'''
        
        with open(batch_file, 'w', encoding='cp949') as f:
            f.write(batch_content)
        
        # 사용 설명서 생성
        readme_content = '''🎨 Hana Studio - 빠른 시작 버전
====================================

🚀 실행 방법
----------
1. "Hana Studio 실행.bat" 더블클릭 (권장)
2. 또는 "HanaStudio.exe" 직접 실행

⚡ 빠른 시작 버전 특징
-------------------
✅ 즉시 시작 (2-3초 내 실행)
✅ CMD 창 완전 숨김
✅ AI 모델 다운로드 진행 상황 표시
✅ 폴더 형태로 모든 파일 포함

📁 파일 구조
----------
이 폴더의 모든 파일이 필요합니다.
다른 컴퓨터에 복사할 때는 전체 폴더를 복사하세요.

🔧 시스템 요구사항
---------------
- Windows 10/11 (64bit)
- 메모리: 최소 4GB RAM 권장  
- 저장공간: 약 500MB
- 인터넷 연결: 최초 AI 모델 다운로드 시 필요

🎯 배포 방법
----------
1. 이 폴더 전체를 ZIP으로 압축
2. 다른 컴퓨터에서 압축 해제
3. "Hana Studio 실행.bat" 실행

📞 지원
------
빠른 시작이 안 되면 HanaStudio.exe를 직접 실행해보세요.
'''
        
        with open(release_dir / "README.txt", 'w', encoding='utf-8-sig') as f:
            f.write(readme_content)
        
        print("✅ 빠른 시작용 패키지 생성 완료")
        return True
    
    def build(self):
        """전체 빌드 프로세스"""
        print("⚡ Hana Studio 빠른 시작 빌드 시작")
        print("=" * 60)
        
        if not self.check_dependencies():
            return False
        
        self.clean_build_dirs()
        
        if not self.run_fast_pyinstaller():
            return False
        
        if not self.create_fast_package():
            return False
        
        print("\n" + "=" * 60)
        print("⚡ 빠른 시작 빌드 완료!")
        print(f"📁 실행 폴더: {self.project_root / 'release_fast'}")
        print("")
        print("⚡ 빠른 시작 버전:")
        print("   - 2-3초 내 즉시 시작")
        print("   - 폴더 형태 (압축 해제 없음)")
        print("   - CMD 창 완전 숨김")
        print("")
        print("🧪 테스트:")
        print("   1. release_fast/Hana Studio 실행.bat 실행")
        print("   2. 즉시 시작되는지 확인")
        print("   3. AI 다이얼로그가 바로 표시되는지 확인")
        
        return True


def main():
    """빠른 시작 빌드 실행"""
    # PyInstaller 설치 확인
    try:
        import PyInstaller
        print(f"✅ PyInstaller {PyInstaller.__version__}")
    except ImportError:
        print("❌ PyInstaller 미설치")
        print("설치 명령어: pip install pyinstaller")
        return
    
    builder = FastHanaStudioBuilder()
    success = builder.build()
    
    if not success:
        print("❌ 빌드 실패")
        return
    
    print("\n📋 다음 단계:")
    print("1. release_fast/Hana Studio 실행.bat 실행")
    print("2. 즉시 시작되는지 확인")
    print("3. 전체 폴더를 ZIP으로 배포")


if __name__ == "__main__":
    main()