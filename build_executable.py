"""
PyInstaller 실행파일 빌드 스크립트 - 완전 자동화
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path


class HanaStudioBuilder:
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.build_dir = self.project_root / "build"
        self.dist_dir = self.project_root / "dist"
        self.spec_file = self.project_root / "hana_studio.spec"
        
    def clean_build_dirs(self):
        """기존 빌드 디렉토리 정리"""
        print("🧹 기존 빌드 파일 정리...")
        
        for dir_path in [self.build_dir, self.dist_dir]:
            if dir_path.exists():
                shutil.rmtree(dir_path)
        
        if self.spec_file.exists():
            self.spec_file.unlink()
    
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
    
    def create_pyinstaller_spec(self):
        """PyInstaller spec 파일 생성"""
        print("📝 PyInstaller spec 파일 생성...")
        
        # 추가 데이터 파일들 수집
        added_files = []
        
        # DLL 및 설정 파일
        data_files = [
            "libDSRetransfer600App.dll",
            "Retransfer600_SDKCfg.xml",
            "config.json"
        ]
        
        for file_name in data_files:
            if (self.project_root / file_name).exists():
                added_files.append(f"('{file_name}', '.')")
        
        # EWL 파일들
        ewl_files = list(self.project_root.glob("*.EWL"))
        for ewl_file in ewl_files:
            added_files.append(f"('{ewl_file.name}', '.')")
        
        # R600StatusReference 파일들  
        status_files = list(self.project_root.glob("R600StatusReference*"))
        for status_file in status_files:
            added_files.append(f"('{status_file.name}', '.')")
        
        # 디렉토리들
        for dir_name in ["ui", "core", "printer", "output", "temp", "logs"]:
            if (self.project_root / dir_name).exists():
                added_files.append(f"('{dir_name}', '{dir_name}')")
        
        added_files_str = ',\n    '.join(added_files) if added_files else ""
        
        spec_content = f"""# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

# 추가 데이터 파일들
added_files = [
    {added_files_str}
]

# 숨겨진 imports - 핵심만
hidden_imports = [
    'config',
    'ui',
    'core', 
    'printer'
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
        'pandas',
        'jupyter',
        'IPython'
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
    console=False,
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
        
        print(f"✅ Spec 파일 생성: {self.spec_file}")
    
    def run_pyinstaller(self):
        """PyInstaller 실행 - spec 파일 없이 직접"""
        print("🚀 PyInstaller 빌드 시작...")
        
        cmd = [
            sys.executable, "-m", "PyInstaller",
            "--onedir",
            "--windowed",  # console 대신 windowed로 변경
            "--name", "HanaStudio",
            "--collect-all", "rembg",
            "--collect-all", "scipy", 
            "--collect-all", "numpy",
            "--collect-all", "cv2",
            "--clean", "--noconfirm",
            "main.py"
        ]
        
        # 존재하는 파일들만 추가
        data_files = [
            ("libDSRetransfer600App.dll", "."),
            ("Retransfer600_SDKCfg.xml", "."),
            ("config.json", "."),
            ("ui", "ui"),
            ("core", "core"),
            ("printer", "printer")
        ]
        
        for src, dst in data_files:
            if os.path.exists(src) or os.path.isdir(src):
                cmd.extend(["--add-data", f"{src};{dst}"])
                print(f"✅ 추가: {src}")
            else:
                print(f"⚠️ 파일 없음: {src}")
        
        # EWL 파일들 추가
        for ewl_file in self.project_root.glob("*.EWL"):
            cmd.extend(["--add-data", f"{ewl_file.name};."])
            print(f"✅ 추가: {ewl_file.name}")
        
        # R600StatusReference 파일들 추가
        for status_file in self.project_root.glob("R600StatusReference*"):
            cmd.extend(["--add-data", f"{status_file.name};."])
            print(f"✅ 추가: {status_file.name}")
        
        try:
            result = subprocess.run(cmd, check=True, cwd=self.project_root)
            print("✅ PyInstaller 빌드 완료!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ PyInstaller 빌드 실패: {e}")
            return False
    
    def create_installer_assets(self):
        """설치 관련 파일들 생성"""
        print("📦 설치 파일 생성...")
        
        dist_hana = self.dist_dir / "HanaStudio"
        if not dist_hana.exists():
            print("❌ 빌드 폴더를 찾을 수 없습니다.")
            return False
        
        # 실행 배치 파일
        batch_content = """@echo off
chcp 65001 > nul
echo 🎨 Hana Studio 디버그 모드

echo 파일 확인 중...
if not exist "libDSRetransfer600App.dll" (
    echo ❌ libDSRetransfer600App.dll 없음
) else (
    echo ✅ libDSRetransfer600App.dll 있음
)

if not exist "Retransfer600_SDKCfg.xml" (
    echo ❌ Retransfer600_SDKCfg.xml 없음  
) else (
    echo ✅ Retransfer600_SDKCfg.xml 있음
)

echo.
echo HanaStudio.exe 실행 중...
"HanaStudio.exe"

echo.
echo 프로그램이 종료되었습니다.
pause
"""
        
        batch_path = dist_hana / "Run_HanaStudio.bat"
        with open(batch_path, 'w', encoding='utf-8') as f:
            f.write(batch_content)
        
        print("✅ 설치 파일 생성 완료")
        return True
    
    def build(self):
        """전체 빌드 프로세스"""
        print("🏗️ Hana Studio 빌드 시작")
        print("=" * 50)
        
        if not self.check_dependencies():
            return False
        
        self.clean_build_dirs()
        # spec 파일 생성 제거
        
        if not self.run_pyinstaller():
            return False
        
        self.create_installer_assets()
        
        print("\n" + "=" * 50)
        print("🎉 빌드 완료!")
        print(f"📁 실행파일: {self.dist_dir / 'HanaStudio'}")
        print("💡 Run_HanaStudio.bat로 실행하세요")
        
        return True


def main():
    """메인 실행"""
    # PyInstaller 설치 확인
    try:
        import PyInstaller
        print(f"✅ PyInstaller {PyInstaller.__version__}")
    except ImportError:
        print("❌ PyInstaller 미설치: pip install pyinstaller")
        return
    
    builder = HanaStudioBuilder()
    builder.build()


if __name__ == "__main__":
    main()