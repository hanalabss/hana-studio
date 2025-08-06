"""
빠른 시작용 PyInstaller 빌드 스크립트 - 중복 실행 문제 해결
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
        print("[INFO] 기존 빌드 파일 정리...")
        
        for dir_path in [self.build_dir, self.dist_dir]:
            if dir_path.exists():
                shutil.rmtree(dir_path)
    
    def check_dependencies(self):
        """필수 파일 존재 여부 확인"""
        print("[INFO] 필수 파일 확인...")
        
        # 기본 파이썬 파일들
        required_py_files = [
            "main.py",
            "hana_studio.py", 
            "config.py",
            "requirements.txt"
        ]
        
        # DLL 파일들 (dll 폴더 또는 루트에서 찾기)
        required_dll_files = [
            "libDSRetransfer600App.dll",
            "Retransfer600_SDKCfg.xml",
            "EWL",
            "R600StatusReference"
        ]
        
        missing_files = []
        
        # 기본 파이썬 파일들 확인
        for file_name in required_py_files:
            if not (self.project_root / file_name).exists():
                missing_files.append(file_name)
        
        # DLL 파일들 확인 (dll 폴더 또는 루트에서)
        for file_name in required_dll_files:
            # dll 폴더에서 먼저 찾기
            dll_path = self.project_root / "dll" / file_name
            root_path = self.project_root / file_name
            
            if not (dll_path.exists() or root_path.exists()):
                missing_files.append(f"{file_name} (dll 폴더 또는 루트)")
        
        # dll 폴더 자체 확인
        dll_dir = self.project_root / "dll"
        if not dll_dir.exists():
            missing_files.append("dll 폴더")
        else:
            dll_files = list(dll_dir.glob("*"))
            if len(dll_files) == 0:
                missing_files.append("dll 폴더 내 파일들")
            else:
                print(f"[OK] dll 폴더 발견: {len(dll_files)}개 파일")
                for dll_file in dll_files:
                    print(f"   - {dll_file.name}")
        
        if missing_files:
            print(f"[ERROR] 누락된 파일: {', '.join(missing_files)}")
            return False
        
        print("[OK] 모든 필수 파일 확인 완료")
        return True

    def run_fast_pyinstaller(self):
        """빠른 시작을 위한 PyInstaller 실행 (onedir 모드)"""
        print("[BUILD] 빠른 시작용 실행파일 빌드 시작 (폴더 형태)...")
        
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
            ("config.json", "."),
        ]
        
        # dll 폴더의 모든 파일들 추가 (프린터 관련)
        dll_dir = self.project_root / "dll"
        if dll_dir.exists():
            for dll_file in dll_dir.iterdir():
                if dll_file.is_file():
                    data_files.append((str(dll_file), "."))
                    print(f"[ADD] DLL 파일 추가: {dll_file.name}")
        
        # 루트에 있는 개별 DLL/설정 파일들도 확인 (하위 호환성)
        root_dll_files = [
            "libDSRetransfer600App.dll",
            "Retransfer600_SDKCfg.xml"
        ]
        for dll_file in root_dll_files:
            dll_path = self.project_root / dll_file
            if dll_path.exists():
                data_files.append((dll_file, "."))
                print(f"[ADD] 루트 DLL 파일 추가: {dll_file}")
        
        if (self.project_root / "hana.ico").exists():
            data_files.append(("hana.ico", "."))
        
        # 디렉토리들 추가
        for dir_name in ["ui", "core", "printer", "models", "dll"]:
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
                print(f"[ADD] 추가: {src}")
        
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
            print("[BUILD] 실행 명령어:")
            print(" ".join(cmd[:10]) + " ...")
            print("[BUILD] 빌드 중... (폴더 형태로 빠른 빌드)")
            
            result = subprocess.run(cmd, check=True, cwd=self.project_root)
            print("[OK] PyInstaller 빌드 완료!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] PyInstaller 빌드 실패: {e}")
            return False
    
    def create_fast_package(self):
        """빠른 시작용 패키지 생성 - 중복 실행 문제 해결"""
        print("[PACKAGE] 빠른 시작용 패키지 생성...")
        
        dist_folder = self.dist_dir / "HanaStudio"
        if not dist_folder.exists():
            print("[ERROR] 빌드 폴더를 찾을 수 없습니다.")
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
        print(f"[INFO] 빌드 폴더 크기: {size_mb:.1f}MB")
        
        # 🔧 수정된 실행 배치 파일 생성 (중복 실행 문제 해결)
        batch_file = release_dir / "Hana Studio 실행.bat"
        batch_content = '''@echo off
cd /d "%~dp0"
HanaStudio.exe
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

⚠️ 중요 사항
-----------
- 한 번만 실행하세요 (중복 실행 방지)
- 프로그램이 이미 실행 중이면 작업표시줄에서 확인하세요

⚡ 빠른 시작 버전 특징
-------------------
✅ 즉시 시작 (2-3초 내 실행)
✅ CMD 창 완전 숨김
✅ AI 모델 백그라운드 로딩 (네트워크 불필요)
✅ 프린터 DLL 파일 모두 포함
✅ 폴더 형태로 모든 파일 포함

📁 포함된 파일들
--------------
- AI 모델: models/ 폴더 (4개 모델 파일)
- 프린터: dll/ 폴더 (4개 DLL 파일)
- 설정: config.json, *.EWL 파일들
- 실행: HanaStudio.exe + 모든 종속성

🔧 시스템 요구사항
---------------
- Windows 10/11 (64bit)
- 메모리: 최소 4GB RAM 권장  
- 저장공간: 약 500MB
- 네트워크: AI 모델이 포함되어 연결 불필요

🎯 배포 방법
----------
1. 이 폴더 전체를 ZIP으로 압축
2. 다른 컴퓨터에서 압축 해제
3. "Hana Studio 실행.bat" 실행
4. 프린터 연결 후 바로 사용 가능

📞 지원
------
빠른 시작이 안 되면 HanaStudio.exe를 직접 실행해보세요.
'''
        
        with open(release_dir / "README.txt", 'w', encoding='utf-8-sig') as f:
            f.write(readme_content)
        
        print("[OK] 빠른 시작용 패키지 생성 완료")
        return True
    
    def build(self):
        """전체 빌드 프로세스"""
        print("[START] Hana Studio 빠른 시작 빌드 시작")
        print("=" * 60)
        
        if not self.check_dependencies():
            return False
        
        self.clean_build_dirs()
        
        if not self.run_fast_pyinstaller():
            return False
        
        if not self.create_fast_package():
            return False
        
        print("\n" + "=" * 60)
        print("[COMPLETE] 빠른 시작 빌드 완료!")
        print(f"[PATH] 실행 폴더: {self.project_root / 'release_fast'}")
        print("")
        print("[FEATURES] 빠른 시작 버전:")
        print("   - 2-3초 내 즉시 시작")
        print("   - AI 모델 포함 (네트워크 불필요)")
        print("   - 프린터 DLL 모두 포함")
        print("   - CMD 창 완전 숨김")
        print("   - 중복 실행 방지")
        print("")
        print("[TEST] 테스트 방법:")
        print("   1. release_fast/Hana Studio 실행.bat 실행")
        print("   2. 즉시 시작되는지 확인")
        print("   3. 배경제거 AI 자동 준비 확인")
        print("   4. 프린터 연결 테스트")
        
        return True


def main():
    """빠른 시작 빌드 실행"""
    # PyInstaller 설치 확인
    try:
        import PyInstaller
        print(f"[OK] PyInstaller {PyInstaller.__version__}")
    except ImportError:
        print("[ERROR] PyInstaller 미설치")
        print("설치 명령어: pip install pyinstaller")
        return
    
    builder = FastHanaStudioBuilder()
    success = builder.build()
    
    if not success:
        print("[ERROR] 빌드 실패")
        return
    
    print("\n[DEPLOY] 배포 단계:")
    print("1. release_fast 폴더 전체를 ZIP으로 압축")
    print("2. 배포 및 테스트")
    print("3. 사용자에게 '실행.bat' 파일 클릭 안내")
    print("4. AI 모델과 프린터 DLL 모두 포함됨")


if __name__ == "__main__":
    main()