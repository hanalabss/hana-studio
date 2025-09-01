"""
[EMOJI] [EMOJI] [EMOJI] [EMOJI] [EMOJI] - [EMOJI] [EMOJI] [EMOJI]
"""

import os
import sys
import tempfile
import uuid
from pathlib import Path


def get_safe_temp_dir():
    """[EMOJI] [EMOJI] [EMOJI] [EMOJI] [EMOJI] [EMOJI] [EMOJI] [EMOJI]"""
    
    # 1. [EMOJI] [EMOJI] [EMOJI] [EMOJI] temp [EMOJI] [EMOJI]
    if getattr(sys, 'frozen', False):
        # [EMOJI] [EMOJI] [EMOJI]
        exe_dir = os.path.dirname(sys.executable)
        # [EMOJI] [EMOJI] [EMOJI] [EMOJI] [EMOJI]
        try:
            exe_dir.encode('ascii')
            # ASCII[EMOJI] [EMOJI] [EMOJI] [EMOJI]
            temp_dir = os.path.join(exe_dir, '.temp')
            os.makedirs(temp_dir, exist_ok=True)
            
            # [EMOJI] [EMOJI] [EMOJI]
            test_file = os.path.join(temp_dir, 'test.tmp')
            try:
                with open(test_file, 'w') as f:
                    f.write('test')
                os.remove(test_file)
                print(f"[TEMP] [EMOJI] [EMOJI] [EMOJI] [EMOJI]: {temp_dir}")
                return temp_dir
            except:
                pass
        except UnicodeEncodeError:
            print("[TEMP] [EMOJI] [EMOJI] [EMOJI] [EMOJI], [EMOJI] [EMOJI] [EMOJI]")
    
    # 2. C:\temp [EMOJI] [EMOJI] [EMOJI] [EMOJI]
    fixed_paths = [
        "C:\\hana_temp",
        "C:\\temp",
        "D:\\temp",
        "C:\\ProgramData\\HanaStudio\\temp"
    ]
    
    for path in fixed_paths:
        try:
            os.makedirs(path, exist_ok=True)
            # [EMOJI] [EMOJI] [EMOJI]
            test_file = os.path.join(path, 'test.tmp')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            print(f"[TEMP] [EMOJI] [EMOJI] [EMOJI] [EMOJI]: {path}")
            return path
        except Exception as e:
            continue
    
    # 3. [EMOJI] [EMOJI] [EMOJI] [EMOJI] [EMOJI] ([EMOJI] [EMOJI] [EMOJI] [EMOJI] [EMOJI])
    default_temp = tempfile.gettempdir()
    
    # [EMOJI] [EMOJI] [EMOJI]
    try:
        default_temp.encode('ascii')
        print(f"[TEMP] [EMOJI] [EMOJI] [EMOJI] [EMOJI]: {default_temp}")
        return default_temp
    except UnicodeEncodeError:
        # [EMOJI] [EMOJI] [EMOJI], [EMOJI] ASCII [EMOJI] [EMOJI] [EMOJI]
        safe_subdir = os.path.join(default_temp, "hana_" + str(uuid.uuid4())[:8])
        try:
            os.makedirs(safe_subdir, exist_ok=True)
            print(f"[TEMP] [EMOJI] [EMOJI] [EMOJI] [EMOJI] [EMOJI] [EMOJI]: {safe_subdir}")
            return safe_subdir
        except:
            # [EMOJI] [EMOJI] [EMOJI] [EMOJI]
            print(f"[TEMP] [EMOJI]: [EMOJI] [EMOJI] [EMOJI] {default_temp}")
            return default_temp


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