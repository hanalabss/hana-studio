"""
Hana Studio 설정 파일
애플리케이션의 기본 설정값들을 관리합니다.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """애플리케이션 설정 관리 클래스"""
    
    # 기본 설정값
    DEFAULT_SETTINGS = {
        'ai_model': 'isnet-general-use',
        'alpha_threshold': 45,
        'output_quality': 95,
        'auto_save_results': False,
        'max_image_size': 2048,
        'timeout_seconds': 30,
        'language': 'ko',
        'theme': 'light',
        'window_geometry': {
            'width': 1600,
            'height': 900,
            'x': 100,
            'y': 100
        },
        'directories': {
            'output': 'output',
            'temp': 'temp',
            'models': 'models',
            'logs': 'logs'
        },
        'printer': {
            'dll_path': './libDSRetransfer600App.dll',
            'card_width': 53.98,
            'card_height': 85.6,
            'timeout_ms': 10000,
            'auto_detect': True
        },
        'watermark': {
            'threshold': 128,
            'thickness': 128,
            'rotation': 0,
            'need_mirror': False,
            'improve_mask_quality': True,
            'mask_kernel_size': 3,
            'mask_iterations': 1
        },
        'processing': {
            'auto_reprocess': False,
            'auto_save_original': False,
            'improve_mask_quality': True,
            'mask_kernel_size': 3,
            'mask_iterations': 1
        }
    }
    
    # 지원하는 AI 모델들
    SUPPORTED_AI_MODELS = {
        'isnet-general-use': {
            'name': '범용 모델 (권장)',
            'description': '대부분의 이미지에 적합한 고품질 모델',
            'performance': 'medium',
            'accuracy': 'high'
        },
        'u2net': {
            'name': 'U²-Net',
            'description': '빠른 처리 속도의 기본 모델',
            'performance': 'fast',
            'accuracy': 'medium'
        },
        'u2netp': {
            'name': 'U²-Net (Light)',
            'description': '경량화된 빠른 처리 모델',
            'performance': 'very_fast',
            'accuracy': 'medium'
        },
        'silueta': {
            'name': 'Silueta',
            'description': '정밀한 실루엣 처리 모델',
            'performance': 'slow',
            'accuracy': 'very_high'
        }
    }
    
    # 지원하는 이미지 형식
    SUPPORTED_IMAGE_FORMATS = {
        '.jpg': 'JPEG 이미지',
        '.jpeg': 'JPEG 이미지',
        '.png': 'PNG 이미지',
        '.bmp': 'Bitmap 이미지',
        '.tiff': 'TIFF 이미지',
        '.tif': 'TIFF 이미지',
        '.webp': 'WebP 이미지'
    }
    
    def __init__(self, config_file: str = 'config.json'):
        """
        설정 관리자 초기화
        
        Args:
            config_file: 설정 파일 경로
        """
        self.config_file = Path(config_file)
        self.settings = self.DEFAULT_SETTINGS.copy()
        self.load_settings()
        self.ensure_directories()
    
    def load_settings(self) -> None:
        """설정 파일에서 설정을 로드합니다."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded_settings = json.load(f)
                    # 기본 설정과 병합
                    self._merge_settings(self.settings, loaded_settings)
                print(f"✅ 설정 로드 완료: {self.config_file}")
            except Exception as e:
                print(f"⚠️ 설정 로드 실패: {e}, 기본 설정 사용")
        else:
            print("📄 설정 파일이 없습니다. 기본 설정을 생성합니다.")
            self.save_settings()
    
    def save_settings(self) -> None:
        """현재 설정을 파일에 저장합니다."""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, indent=2, ensure_ascii=False)
            print(f"✅ 설정 저장 완료: {self.config_file}")
        except Exception as e:
            print(f"❌ 설정 저장 실패: {e}")
    
    def _merge_settings(self, default: Dict, loaded: Dict) -> None:
        """기본 설정과 로드된 설정을 재귀적으로 병합합니다."""
        for key, value in loaded.items():
            if key in default:
                if isinstance(default[key], dict) and isinstance(value, dict):
                    self._merge_settings(default[key], value)
                else:
                    default[key] = value
    
    def ensure_directories(self) -> None:
        """필요한 디렉토리들을 생성합니다."""
        for dir_key, dir_path in self.settings['directories'].items():
            Path(dir_path).mkdir(exist_ok=True)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        설정값을 가져옵니다.
        
        Args:
            key: 설정 키 (점 표기법 지원, 예: 'window_geometry.width')
            default: 기본값
            
        Returns:
            설정값 또는 기본값
        """
        keys = key.split('.')
        value = self.settings
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        설정값을 변경합니다.
        
        Args:
            key: 설정 키 (점 표기법 지원)
            value: 설정값
        """
        keys = key.split('.')
        target = self.settings
        
        # 마지막 키를 제외한 경로까지 이동
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]
        
        # 마지막 키에 값 설정
        target[keys[-1]] = value
    
    def get_ai_model_info(self, model_name: str) -> Optional[Dict]:
        """AI 모델 정보를 가져옵니다."""
        return self.SUPPORTED_AI_MODELS.get(model_name)
    
    def get_supported_models(self) -> Dict:
        """지원되는 AI 모델 목록을 가져옵니다."""
        return self.SUPPORTED_AI_MODELS
    
    def get_image_filter(self) -> str:
        """파일 다이얼로그용 이미지 필터를 생성합니다."""
        formats = []
        for ext, desc in self.SUPPORTED_IMAGE_FORMATS.items():
            formats.append(f"*{ext}")
        
        filter_str = f"이미지 파일 ({' '.join(formats)})"
        for ext, desc in self.SUPPORTED_IMAGE_FORMATS.items():
            filter_str += f";;{desc} (*{ext})"
        
        return filter_str
    
    def is_supported_image(self, file_path: str) -> bool:
        """지원되는 이미지 형식인지 확인합니다."""
        ext = Path(file_path).suffix.lower()
        return ext in self.SUPPORTED_IMAGE_FORMATS
    
    def validate_settings(self) -> bool:
        """설정값들의 유효성을 검사합니다."""
        try:
            # AI 모델 확인
            if self.get('ai_model') not in self.SUPPORTED_AI_MODELS:
                print(f"⚠️ 지원하지 않는 AI 모델: {self.get('ai_model')}")
                self.set('ai_model', 'isnet-general-use')
            
            # 임계값 범위 확인
            threshold = self.get('alpha_threshold')
            if not (0 <= threshold <= 255):
                print(f"⚠️ 잘못된 알파 임계값: {threshold}")
                self.set('alpha_threshold', 45)
            
            # 품질 범위 확인
            quality = self.get('output_quality')
            if not (1 <= quality <= 100):
                print(f"⚠️ 잘못된 출력 품질: {quality}")
                self.set('output_quality', 95)
            
            return True
            
        except Exception as e:
            print(f"❌ 설정 검증 실패: {e}")
            return False
    
    def reset_to_defaults(self) -> None:
        """설정을 기본값으로 초기화합니다."""
        self.settings = self.DEFAULT_SETTINGS.copy()
        self.save_settings()
        print("🔄 설정이 기본값으로 초기화되었습니다.")
    
    def export_settings(self, file_path: str) -> bool:
        """설정을 다른 파일로 내보냅니다."""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, indent=2, ensure_ascii=False)
            print(f"✅ 설정 내보내기 완료: {file_path}")
            return True
        except Exception as e:
            print(f"❌ 설정 내보내기 실패: {e}")
            return False
    
    def import_settings(self, file_path: str) -> bool:
        """다른 파일에서 설정을 가져옵니다."""
        try:
            if not os.path.exists(file_path):
                print(f"❌ 파일이 존재하지 않습니다: {file_path}")
                return False
            
            with open(file_path, 'r', encoding='utf-8') as f:
                imported_settings = json.load(f)
            
            self._merge_settings(self.settings, imported_settings)
            self.save_settings()
            print(f"✅ 설정 가져오기 완료: {file_path}")
            return True
            
        except Exception as e:
            print(f"❌ 설정 가져오기 실패: {e}")
            return False


# 전역 설정 인스턴스
config = Config()


# 편의 함수들
def get_setting(key: str, default: Any = None) -> Any:
    """설정값을 가져오는 편의 함수"""
    return config.get(key, default)


def set_setting(key: str, value: Any) -> None:
    """설정값을 변경하는 편의 함수"""
    config.set(key, value)


def save_config() -> None:
    """설정을 저장하는 편의 함수"""
    config.save_settings()


# 애플리케이션 상수들
class AppConstants:
    """애플리케이션 상수 정의"""
    
    APP_NAME = "Hana Studio"
    APP_VERSION = "1.0.0"
    APP_AUTHOR = "Hana Tech"
    APP_DESCRIPTION = "AI 기반 이미지 배경 제거 도구"
    
    # 색상 팔레트
    class Colors:
        PRIMARY = "#4A90E2"
        PRIMARY_DARK = "#357ABD"
        PRIMARY_LIGHT = "#5BA0F2"
        
        SECONDARY = "#F8F9FA"
        SECONDARY_DARK = "#E9ECEF"
        SECONDARY_LIGHT = "#FFFFFF"
        
        SUCCESS = "#28A745"
        WARNING = "#FFC107"
        ERROR = "#DC3545"
        INFO = "#17A2B8"
        
        TEXT_PRIMARY = "#212529"
        TEXT_SECONDARY = "#6C757D"
        TEXT_MUTED = "#ADB5BD"
        
        BACKGROUND = "#FFFFFF"
        BACKGROUND_ALT = "#F8F9FA"
        BORDER = "#DEE2E6"
    
    # 크기 상수
    class Sizes:
        BUTTON_HEIGHT = 45
        ICON_SIZE = 16
        BORDER_RADIUS = 8
        SPACING_SMALL = 10
        SPACING_MEDIUM = 15
        SPACING_LARGE = 20
        WINDOW_MIN_WIDTH = 1200
        WINDOW_MIN_HEIGHT = 800


if __name__ == "__main__":
    # 설정 테스트
    print("🔧 Hana Studio 설정 테스트")
    print(f"AI 모델: {config.get('ai_model')}")
    print(f"출력 디렉토리: {config.get('directories.output')}")
    print(f"지원 모델: {list(config.get_supported_models().keys())}")
    print(f"설정 유효성: {config.validate_settings()}")