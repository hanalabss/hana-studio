"""
[EMOJI] [EMOJI] [EMOJI] [EMOJI] [EMOJI] - [EMOJI] [EMOJI] [EMOJI]
"""

import os
import sys
import tempfile
import uuid
from pathlib import Path


def get_safe_temp_dir():
    """한글 경로 문제 해결을 위한 안전한 임시 디렉토리 반환

    개선된 버전: 한글/영어 사용자명 환경 모두 지원
    onnxruntime/rembg는 전체 경로가 ASCII여야 합니다.
    한글 경로가 포함되면 모델 로딩이 실패합니다.
    """

    print("[TEMP] ASCII 안전 경로 탐색 시작...")

    def is_ascii_path(path):
        """전체 경로가 ASCII인지 확인"""
        try:
            path.encode('ascii')
            return True
        except UnicodeEncodeError:
            return False

    def test_write_permission(path):
        """쓰기 권한 테스트"""
        try:
            os.makedirs(path, exist_ok=True)
            test_file = os.path.join(path, 'test_write.tmp')
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write('test')
            os.remove(test_file)
            return True
        except Exception:
            return False

    # 우선순위 1: ProgramData (모든 사용자가 쓰기 가능, ASCII 경로)
    try:
        program_data = os.environ.get('ProgramData', 'C:\\ProgramData')
        safe_path = os.path.join(program_data, "HanaStudio", "temp")

        if is_ascii_path(safe_path) and test_write_permission(safe_path):
            print(f"[TEMP] ✅ ProgramData 경로 사용: {safe_path}")
            return safe_path
    except Exception as e:
        print(f"[TEMP] ⚠️ ProgramData 실패: {e}")

    # 우선순위 2: 시스템 드라이브 루트 (C:\Temp 등)
    priority_paths = [
        "C:\\Temp\\HanaStudio",
        "C:\\hana_temp",
        "D:\\Temp\\HanaStudio",
        "D:\\hana_temp",
    ]

    for path in priority_paths:
        try:
            if is_ascii_path(path) and test_write_permission(path):
                print(f"[TEMP] ✅ 시스템 드라이브 경로 사용: {path}")
                return path
        except Exception as e:
            print(f"[TEMP] ⚠️ {path} 실패: {e}")
            continue

    # 우선순위 3: 시스템 임시 디렉토리 (ASCII인 경우만)
    try:
        base_temp = tempfile.gettempdir()
        safe_subdir = os.path.join(base_temp, "hana_printer_safe")

        if is_ascii_path(safe_subdir) and test_write_permission(safe_subdir):
            print(f"[TEMP] ✅ 시스템 임시 폴더 사용: {safe_subdir}")
            return safe_subdir
        else:
            print(f"[TEMP] ⚠️ 시스템 임시 폴더에 한글 포함: {base_temp}")
    except Exception as e:
        print(f"[TEMP] ⚠️ 시스템 임시 폴더 실패: {e}")

    # 우선순위 4: 현재 작업 디렉토리 (ASCII인 경우만)
    try:
        cwd = os.getcwd()
        cwd_temp = os.path.join(cwd, ".hana_temp")

        if is_ascii_path(cwd_temp) and test_write_permission(cwd_temp):
            print(f"[TEMP] ✅ 작업 디렉토리 사용: {cwd_temp}")
            return cwd_temp
        else:
            print(f"[TEMP] ⚠️ 작업 디렉토리에 한글 포함: {cwd}")
    except Exception as e:
        print(f"[TEMP] ⚠️ 작업 디렉토리 실패: {e}")

    # 우선순위 5: 실행 파일 디렉토리 (빌드된 경우, ASCII인 경우만)
    if getattr(sys, 'frozen', False):
        try:
            exe_dir = os.path.dirname(sys.executable)
            temp_dir = os.path.join(exe_dir, '.temp')

            if is_ascii_path(temp_dir) and test_write_permission(temp_dir):
                print(f"[TEMP] ✅ 실행 디렉토리 사용: {temp_dir}")
                return temp_dir
            else:
                print(f"[TEMP] ⚠️ 실행 디렉토리에 한글 포함: {exe_dir}")
        except Exception as e:
            print(f"[TEMP] ⚠️ 실행 디렉토리 실패: {e}")

    # 모든 방법 실패 - 명확한 에러 메시지
    error_msg = """
❌ AI 모델 로딩을 위한 ASCII 경로를 찾을 수 없습니다.

한글 사용자명으로 인해 다음 경로들을 사용할 수 없습니다.

시도한 경로들:
1. C:\\ProgramData\\HanaStudio\\temp
2. C:\\Temp\\HanaStudio
3. 시스템 임시 폴더 (한글 포함)
4. 현재 작업 디렉토리 (한글 포함)
5. 실행 파일 디렉토리 (한글 포함)

해결 방법:
1. 프로그램을 관리자 권한으로 실행 (권장)
2. 또는 다음 명령어를 관리자 권한 CMD에서 실행:
   mkdir C:\\Temp\\HanaStudio
   icacls C:\\Temp\\HanaStudio /grant Users:(OI)(CI)F

3. 또는 프로그램을 한글이 없는 경로에 설치
   (예: C:\\HanaStudio)
"""
    print(error_msg)
    raise RuntimeError(error_msg)


def create_safe_temp_file(prefix="temp", suffix=".tmp", dir=None):
    """[EMOJI] [EMOJI] [EMOJI] [EMOJI]"""
    if dir is None:
        dir = get_safe_temp_dir()
    
    # ASCII [EMOJI] [EMOJI] [EMOJI] [EMOJI]
    safe_name = f"{prefix}_{str(uuid.uuid4())[:8]}{suffix}"
    temp_path = os.path.join(dir, safe_name)
    
    return temp_path


def ensure_ascii_path(file_path):
    """[EMOJI] [EMOJI] ASCII[EMOJI] [EMOJI] [EMOJI]
    [EMOJI] [EMOJI] [EMOJI] [EMOJI] [EMOJI] [EMOJI] [EMOJI]"""
    
    try:
        file_path.encode('ascii')
        # [EMOJI] ASCII [EMOJI] [EMOJI] [EMOJI]
        return file_path, False
    except UnicodeEncodeError:
        # [EMOJI] [EMOJI] [EMOJI] [EMOJI] [EMOJI] [EMOJI]
        import shutil
        
        ext = os.path.splitext(file_path)[1]
        temp_path = create_safe_temp_file(prefix="copy", suffix=ext)
        
        try:
            shutil.copy2(file_path, temp_path)
            print(f"[TEMP] [EMOJI] [EMOJI] [EMOJI] [EMOJI] [EMOJI] [EMOJI]: {temp_path}")
            return temp_path, True
        except Exception as e:
            print(f"[TEMP] [EMOJI] [EMOJI] [EMOJI]: {e}")
            # [EMOJI] [EMOJI] [EMOJI] [EMOJI] [EMOJI] [EMOJI]
            return file_path, False


# [EMOJI] [EMOJI] [EMOJI] [EMOJI]
_cached_temp_dir = None

def get_cached_safe_temp_dir():
    """[EMOJI] [EMOJI] [EMOJI] [EMOJI] [EMOJI]"""
    global _cached_temp_dir
    if _cached_temp_dir is None:
        _cached_temp_dir = get_safe_temp_dir()
    return _cached_temp_dir