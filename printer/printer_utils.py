"""
프린터 관련 유틸리티 함수들
"""

import os
import sys
import shutil
from typing import List, Optional, Tuple
from utils.safe_temp_path import ensure_ascii_path, create_safe_temp_file
from config import config
from .exceptions import DLLNotFoundError


def check_printer_dll(dll_path: str) -> bool:
    """DLL 파일 존재 여부 확인"""
    return os.path.exists(dll_path)

def get_executable_dir() -> str:
    """실행파일 디렉토리"""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
def get_default_dll_paths() -> List[str]:
    """기본 DLL 경로 목록 반환 - PyInstaller 호환"""
    base_dir = get_executable_dir()
    
    # PyInstaller 빌드된 경우 _internal 폴더도 확인
    paths = [
        os.path.join(base_dir, 'libDSRetransfer600App.dll'),
        os.path.join(base_dir, '_internal', 'libDSRetransfer600App.dll'),
        os.path.join(base_dir, '_internal', 'dll', 'libDSRetransfer600App.dll'),
        os.path.join(base_dir, 'dll', 'libDSRetransfer600App.dll'),
        os.path.join(base_dir, 'lib', 'libDSRetransfer600App.dll'),
        config.get('printer.dll_path', os.path.join(base_dir, 'libDSRetransfer600App.dll'))
    ]
    
    # 개발 환경에서도 release_fast 폴더 확인
    if not getattr(sys, 'frozen', False):
        release_paths = [
            os.path.join(base_dir, 'release_fast', '_internal', 'libDSRetransfer600App.dll'),
            os.path.join(base_dir, 'release_fast', '_internal', 'dll', 'libDSRetransfer600App.dll')
        ]
        paths.extend(release_paths)
    
    return paths

def find_printer_dll() -> Optional[str]:
    """사용 가능한 프린터 DLL 찾기 - 한글 경로 대응"""
    print(f"[DLL DEBUG] 실행 디렉토리: {get_executable_dir()}")
    print(f"[DLL DEBUG] Python frozen 상태: {getattr(sys, 'frozen', False)}")
    
    dll_paths = get_default_dll_paths()
    print(f"[DLL DEBUG] 검색할 경로 목록 ({len(dll_paths)}개):")
    for i, path in enumerate(dll_paths, 1):
        print(f"  {i}. {path}")
    
    for path in dll_paths:
        print(f"[DLL DEBUG] 확인 중: {path}")
        if check_printer_dll(path):
            print(f"[DLL DEBUG] 파일 존재 확인됨: {path}")
            # DLL 경로가 한글을 포함하는지 확인
            try:
                path.encode('ascii')
                print(f"✅ 프린터 DLL 발견: {path}")
                return path
            except UnicodeEncodeError:
                # 한글 경로인 경우 안전한 경로로 복사
                print(f"⚠️ DLL 경로에 한글 포함: {path}")
                safe_path = create_safe_temp_file(prefix="printer_dll", suffix=".dll")
                try:
                    shutil.copy2(path, safe_path)
                    print(f"✅ DLL을 안전한 경로로 복사: {safe_path}")
                    return safe_path
                except Exception as e:
                    print(f"❌ DLL 복사 실패: {e}")
                    continue
        else:
            print(f"[DLL DEBUG] 파일 없음: {path}")
    
    print("❌ 프린터 DLL을 찾을 수 없습니다.")
    print("[DLL DEBUG] 현재 디렉토리 내용:")
    try:
        base_dir = get_executable_dir()
        if os.path.exists(base_dir):
            for item in os.listdir(base_dir)[:20]:  # 처음 20개만
                print(f"  - {item}")
    except Exception as e:
        print(f"  디렉토리 읽기 실패: {e}")
    
    return None


def test_printer_connection() -> bool:
    """프린터 연결 테스트"""
    try:
        from .r600_printer import R600Printer
        
        dll_path = find_printer_dll()
        if not dll_path:
            return False
        
        with R600Printer(dll_path) as printer:
            printers = printer.enum_printers()
            return len(printers) > 0
    except Exception as e:
        print(f"프린터 연결 테스트 실패: {e}")
        return False


def test_print_modes():
    """인쇄 모드 테스트 함수"""
    try:
        from .r600_printer import R600Printer
        
        dll_path = find_printer_dll()
        if not dll_path:
            print("❌ DLL 파일을 찾을 수 없습니다.")
            return
        
        with R600Printer(dll_path) as printer:
            printers = printer.enum_printers()
            if not printers:
                print("❌ 사용 가능한 프린터가 없습니다.")
                return
            
            printer.set_timeout(10000)
            printer.select_printer(printers[0])
            
            # 테스트 이미지 경로 (실제 파일로 변경 필요)
            test_image = "test_image.jpg"
            test_mask = "test_mask.jpg"
            
            if os.path.exists(test_image):
                print("🖼️ 일반 인쇄 테스트...")
                printer.print_normal_card(test_image)
                
                if os.path.exists(test_mask):
                    print("🎭 레이어 인쇄 테스트...")
                    printer.print_layered_card(test_mask, test_image)
                else:
                    print("⚠️ 마스크 이미지가 없어 레이어 인쇄 테스트를 건너뜁니다.")
            else:
                print("⚠️ 테스트 이미지가 없어 인쇄 테스트를 건너뜁니다.")
                
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")