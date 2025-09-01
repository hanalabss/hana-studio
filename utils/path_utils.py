# utils/path_utils.py - PyInstaller [EMOJI] [EMOJI] [EMOJI]

import os
import sys
from pathlib import Path


def get_resource_path(relative_path: str) -> str:
    """PyInstaller [EMOJI] [EMOJI] [EMOJI] [EMOJI]"""
    try:
        # PyInstaller[EMOJI] [EMOJI] [EMOJI]
        base_path = sys._MEIPASS
    except AttributeError:
        # [EMOJI] [EMOJI] [EMOJI] [EMOJI]
        base_path = os.path.dirname(os.path.abspath(__file__))
        base_path = os.path.dirname(base_path)  # utils [EMOJI] [EMOJI]
    
    return os.path.join(base_path, relative_path)


def get_executable_dir() -> str:
    """[EMOJI] [EMOJI] [EMOJI] [EMOJI]"""
    if getattr(sys, 'frozen', False):
        # PyInstaller[EMOJI] [EMOJI] [EMOJI]
        return os.path.dirname(sys.executable)
    else:
        # [EMOJI] [EMOJI]
        return os.path.dirname(os.path.abspath(__file__))


def ensure_writable_dir(dir_path: str) -> str:
    """[EMOJI] [EMOJI] [EMOJI] [EMOJI]"""
    if getattr(sys, 'frozen', False):
        # [EMOJI] [EMOJI] [EMOJI] [EMOJI]
        exe_dir = get_executable_dir()
        full_path = os.path.join(exe_dir, dir_path)
    else:
        full_path = dir_path
    
    os.makedirs(full_path, exist_ok=True)
    return full_path