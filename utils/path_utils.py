# utils/path_utils.py - PyInstaller 호환 경로 유틸리티

import os
import sys
from pathlib import Path


def get_resource_path(relative_path: str) -> str:
    """PyInstaller 호환 리소스 경로 반환"""
    try:
        # PyInstaller로 패키징된 경우
        base_path = sys._MEIPASS
    except AttributeError:
        # 개발 환경에서 실행되는 경우
        base_path = os.path.dirname(os.path.abspath(__file__))
        base_path = os.path.dirname(base_path)  # utils 상위 디렉토리
    
    return os.path.join(base_path, relative_path)


def get_executable_dir() -> str:
    """실행파일이 있는 디렉토리 반환"""
    if getattr(sys, 'frozen', False):
        # PyInstaller로 패키징된 경우
        return os.path.dirname(sys.executable)
    else:
        # 개발 환경
        return os.path.dirname(os.path.abspath(__file__))


def ensure_writable_dir(dir_path: str) -> str:
    """쓰기 가능한 디렉토리 확보"""
    if getattr(sys, 'frozen', False):
        # 실행파일과 같은 위치에 생성
        exe_dir = get_executable_dir()
        full_path = os.path.join(exe_dir, dir_path)
    else:
        full_path = dir_path
    
    os.makedirs(full_path, exist_ok=True)
    return full_path