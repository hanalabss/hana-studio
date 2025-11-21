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
    
    # 개발 환경에서도 dist 폴더 확인
    if not getattr(sys, 'frozen', False):
        dist_paths = [
            os.path.join(base_dir, 'dist', 'HanaStudio', '_internal', 'libDSRetransfer600App.dll'),
            os.path.join(base_dir, 'dist', 'HanaStudio', '_internal', 'dll', 'libDSRetransfer600App.dll')
        ]
        paths.extend(dist_paths)
    
    return paths

def find_printer_dll() -> Optional[str]:
    """사용 가능한 프린터 DLL 찾기 - 한글/영어 사용자명 환경 모두 지원"""
    print("\n" + "="*60)
    print("🔍 프린터 DLL 탐색 시작")
    print("="*60)
    print(f"[DLL] 실행 디렉토리: {get_executable_dir()}")
    print(f"[DLL] Python frozen 상태: {getattr(sys, 'frozen', False)}")

    dll_paths = get_default_dll_paths()
    print(f"\n[DLL] 검색할 경로 목록 ({len(dll_paths)}개):")
    for i, path in enumerate(dll_paths, 1):
        # ASCII 여부 표시
        try:
            path.encode('ascii')
            ascii_status = "✓ ASCII"
        except UnicodeEncodeError:
            ascii_status = "⚠ 한글포함"
        print(f"  {i}. [{ascii_status}] {path}")

    print("\n[DLL] 경로 검증 시작...")
    for path in dll_paths:
        print(f"\n→ 확인 중: {path}")

        # 파일 존재 확인
        if not check_printer_dll(path):
            print(f"  ✗ 파일 없음")
            continue

        print(f"  ✓ 파일 존재 (크기: {os.path.getsize(path):,} bytes)")

        # ASCII 경로 확인
        try:
            path.encode('ascii')
            print(f"  ✓ ASCII 경로")
            print(f"✅ 프린터 DLL 발견: {path}")
            return path
        except UnicodeEncodeError:
            # 한글 경로인 경우 안전한 경로로 복사 (dll 폴더의 모든 4개 항목 복사)
            print(f"  ⚠ 한글 경로 감지 - ASCII 안전 경로로 복사 시도")
            try:
                # dll 폴더 경로
                dll_dir = os.path.dirname(path)
                dll_filename = os.path.basename(path)

                # 안전한 임시 디렉토리 생성
                from utils.safe_temp_path import get_safe_temp_dir
                safe_base_dir = get_safe_temp_dir()
                safe_dll_dir = os.path.join(safe_base_dir, "printer_dll_full")

                # 이미 존재하면 모든 필수 파일이 있는지 확인하고 재사용
                safe_dll_path = os.path.join(safe_dll_dir, dll_filename)
                required_items = ['EWL', 'libDSRetransfer600App.dll', 'R600StatusReference', 'Retransfer600_SDKCfg.xml']

                if os.path.exists(safe_dll_dir):
                    # 모든 필수 파일 존재 여부 확인
                    all_exist = True
                    for item in required_items:
                        item_path = os.path.join(safe_dll_dir, item)
                        if not os.path.exists(item_path):
                            print(f"  ⚠ 필수 파일 누락: {item}")
                            all_exist = False
                            break

                    if all_exist:
                        print(f"  ✓ 이미 복사된 DLL 및 설정 파일 발견 - 재사용")
                        print(f"✅ DLL 경로: {safe_dll_path}")
                        return safe_dll_path
                    else:
                        # 일부 파일 누락 - 디렉토리 삭제 후 재복사
                        print(f"  → 일부 파일 누락으로 전체 재복사 필요")
                        try:
                            shutil.rmtree(safe_dll_dir)
                        except Exception as e:
                            print(f"  ⚠ 기존 디렉토리 삭제 실패: {e}")

                os.makedirs(safe_dll_dir, exist_ok=True)

                print(f"  → 대상 디렉토리: {safe_dll_dir}")
                print(f"  → dll 폴더의 모든 항목 복사 중...")

                # dll 폴더의 모든 항목 확인 및 복사
                required_items = ['EWL', 'libDSRetransfer600App.dll', 'R600StatusReference', 'Retransfer600_SDKCfg.xml']
                copied_items = []

                for item_name in os.listdir(dll_dir):
                    src_path = os.path.join(dll_dir, item_name)
                    dst_path = os.path.join(safe_dll_dir, item_name)

                    try:
                        if os.path.isdir(src_path):
                            # 디렉토리 복사
                            shutil.copytree(src_path, dst_path)
                            size_info = f"(디렉토리)"
                            print(f"    ✓ {item_name}/ 복사 완료 {size_info}")
                        else:
                            # 파일 복사
                            shutil.copy2(src_path, dst_path)
                            size = os.path.getsize(dst_path)
                            size_info = f"({size:,} bytes)"
                            print(f"    ✓ {item_name} 복사 완료 {size_info}")

                        copied_items.append(item_name)
                    except Exception as e:
                        print(f"    ✗ {item_name} 복사 실패: {e}")

                # 필수 4개 항목이 모두 복사되었는지 확인
                print(f"\n  → 필수 항목 검증 중...")
                missing_items = []
                for required in required_items:
                    dst_path = os.path.join(safe_dll_dir, required)
                    if os.path.exists(dst_path):
                        if os.path.isdir(dst_path):
                            print(f"    ✓ {required}/ - 존재 (디렉토리)")
                        else:
                            size = os.path.getsize(dst_path)
                            print(f"    ✓ {required} - 존재 ({size:,} bytes)")
                    else:
                        print(f"    ✗ {required} - 없음!")
                        missing_items.append(required)

                if missing_items:
                    print(f"  ✗ 필수 항목 누락: {missing_items}")
                    print(f"  ✗ 프린터 목록을 찾을 수 없습니다")
                    shutil.rmtree(safe_dll_dir)
                    continue

                # DLL 파일 경로 반환
                safe_dll_path = os.path.join(safe_dll_dir, dll_filename)
                print(f"\n✅ dll 폴더의 모든 항목 복사 완료 ({len(copied_items)}개)")
                print(f"✅ DLL 경로: {safe_dll_path}")
                return safe_dll_path

            except RuntimeError as e:
                # get_safe_temp_dir()에서 발생한 에러
                print(f"  ✗ 안전 경로 생성 실패: {e}")
                print(f"  ℹ 임시 디렉토리를 생성할 수 없습니다")
                # 에러는 출력하지만 계속 진행 (다른 경로 시도)
                continue
            except Exception as e:
                print(f"  ✗ 복사 실패: {e}")
                import traceback
                traceback.print_exc()
                continue

    # DLL을 찾지 못한 경우
    print("\n" + "="*60)
    print("❌ 프린터 DLL을 찾을 수 없습니다")
    print("="*60)

    # 디버깅 정보
    print("\n[디버깅] 현재 디렉토리 내용 (최대 20개):")
    try:
        base_dir = get_executable_dir()
        if os.path.exists(base_dir):
            items = os.listdir(base_dir)[:20]
            for item in items:
                item_path = os.path.join(base_dir, item)
                if os.path.isdir(item_path):
                    print(f"  📁 {item}/")
                else:
                    print(f"  📄 {item}")
            if len(os.listdir(base_dir)) > 20:
                print(f"  ... (총 {len(os.listdir(base_dir))}개 항목)")
    except Exception as e:
        print(f"  ✗ 디렉토리 읽기 실패: {e}")

    print("\n[해결 방법]")
    print("1. libDSRetransfer600App.dll 파일이 프로그램과 같은 폴더에 있는지 확인")
    print("2. PyInstaller 빌드인 경우 _internal 폴더 확인")
    print("3. 파일이 존재하는데도 인식 안 되면:")
    print("   - 관리자 권한으로 프로그램 실행")
    print("   - 또는 프로그램을 한글 없는 경로에 설치 (예: C:\\Programs\\HanaStudio)")
    print("="*60 + "\n")

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