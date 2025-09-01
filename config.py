# config.py
import os
import sys
import json
from pathlib import Path
from typing import Any, Dict, Optional

def get_resource_path(relative_path: str) -> str:
    """PyInstaller 실행 파일 경로 처리"""
    try:
        # PyInstaller로 빌드된 경우
        base_path = sys._MEIPASS
    except AttributeError:
        # 일반 Python 실행 환경
        base_path = os.path.dirname(os.path.abspath(__file__))
    
    return os.path.join(base_path, relative_path)

class Config:
    """애플리케이션 설정 관리 클래스"""
    
    DEFAULT_SETTINGS = {
        "ai_model": "isnet-general-use",
        "alpha_threshold": 45,
        "output_quality": 95,
        "max_image_size": 2048,
        "directories": {
            "output": "output",
            "masks": "masks",
            "temp": "temp"
        }
    }
    
    def __init__(self):
        self.settings = {}
        self.config_file = "config.json"
        self.load_settings()
        self._ensure_directories()
    
    def _get_safe_temp_dir(self):
        """안전한 임시 디렉토리 경로 반환"""
        if getattr(sys, 'frozen', False):
            # PyInstaller 환경: 실행 파일과 같은 위치
            exe_dir = Path(sys.executable).parent
            temp_dir = exe_dir / "HanaStudio_Temp"
        else:
            # 개발 환경: 현재 작업 디렉토리
            temp_dir = Path.cwd() / "HanaStudio_Temp"
        
        # 디렉토리 생성
        temp_dir.mkdir(exist_ok=True, parents=True)
        return str(temp_dir)
    
    def _ensure_directories(self):
        """필요한 디렉토리 생성 (한글 경로 지원)"""
        for dir_key, dir_path in self.settings.get('directories', {}).items():
            try:
                if getattr(sys, 'frozen', False):
                    # 실행 파일 위치 기준
                    base_dir = Path(sys.executable).parent
                else:
                    # 개발 환경 기준
                    base_dir = Path.cwd()
                
                full_path = base_dir / dir_path
                full_path.mkdir(exist_ok=True, parents=True)
                
            except Exception as e:
                # 권한 문제 등으로 실패 시 안전한 위치에 생성
                safe_dir = Path(self._get_safe_temp_dir()) / dir_path
                safe_dir.mkdir(exist_ok=True, parents=True)
                self.settings['directories'][dir_key] = str(safe_dir)
    
    def load_settings(self):
        """설정 파일 로드"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                    self.settings = self._merge_settings(self.DEFAULT_SETTINGS.copy(), loaded)
            else:
                self.settings = self.DEFAULT_SETTINGS.copy()
        except Exception as e:
            print(f"설정 로드 실패, 기본값 사용: {e}")
            self.settings = self.DEFAULT_SETTINGS.copy()
    
    def save_settings(self):
        """현재 설정을 파일에 저장"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, indent=2, ensure_ascii=False)
            print(f"설정 저장 완료: {self.config_file}")
        except Exception as e:
            print(f"설정 저장 실패: {e}")
    
    def _merge_settings(self, base, update):
        """설정 병합"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                base[key] = self._merge_settings(base[key], value)
            else:
                base[key] = value
        return base
    
    def get(self, key: str, default: Any = None) -> Any:
        """설정값 가져오기"""
        keys = key.split('.')
        value = self.settings
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """설정값 저장"""
        keys = key.split('.')
        target = self.settings
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]
        target[keys[-1]] = value

    SUPPORTED_AI_MODELS = {
        "isnet-general-use": {"name": "ISNet (일반용)", "quality": "최고"},
        "u2net": {"name": "U2Net", "quality": "높음"},
        "u2net_human_seg": {"name": "U2Net (인물용)", "quality": "높음"},
        "silueta": {"name": "Silueta", "quality": "중간"}
    }
    
    SUPPORTED_IMAGE_FORMATS = {
        ".jpg": "JPEG 이미지",
        ".jpeg": "JPEG 이미지",
        ".png": "PNG 이미지",
        ".bmp": "BMP 이미지",
        ".tiff": "TIFF 이미지",
        ".webp": "WebP 이미지"
    }
    
    def get_image_filter(self):
        """QFileDialog용 이미지 필터 문자열 반환"""
        filters = []
        
        # 모든 이미지 형식
        all_extensions = " ".join([f"*{ext}" for ext in self.SUPPORTED_IMAGE_FORMATS.keys()])
        filters.append(f"이미지 파일 ({all_extensions})")
        
        # 개별 형식
        for ext, desc in self.SUPPORTED_IMAGE_FORMATS.items():
            filters.append(f"{desc} (*{ext})")
        
        # 모든 파일
        filters.append("모든 파일 (*.*)")
        
        return ";;".join(filters)
    
    def is_supported_image(self, file_path: str) -> bool:
        """지원되는 이미지 형식인지 확인"""
        from pathlib import Path
        ext = Path(file_path).suffix.lower()
        return ext in self.SUPPORTED_IMAGE_FORMATS

# 전역 설정 인스턴스
config = Config()

def get_setting(key: str, default: Any = None):
    return config.get(key, default)

def set_setting(key: str, value: Any):
    config.set(key, value)

class AppConstants:
    APP_NAME = "Hana Studio"
    APP_VERSION = "1.0.0"
    
    class Colors:
        PRIMARY = "#4A90E2"
        SUCCESS = "#28A745"
        WARNING = "#FFC107"
        ERROR = "#DC3545"