"""
이미지 처리 로직
"""

import io
import numpy as np
import cv2
from PIL import Image
from rembg import remove, new_session
from config import config


class ImageProcessor:
    """이미지 처리를 담당하는 클래스"""
    
    def __init__(self):
        self.session = None
        self._initialize_ai_model()
    
    def _initialize_ai_model(self):
        """AI 모델 초기화"""
        try:
            model_name = config.get('ai_model', 'isnet-general-use')
            print(f"AI 모델 초기화 중: {model_name}")
            self.session = new_session(model_name=model_name)
            
            model_info = config.get_ai_model_info(model_name)
            if model_info:
                print(f"✅ {model_info['name']} 로드 완료!")
            else:
                print("✅ AI 모델 로드 완료!")
                
        except Exception as e:
            print(f"❌ AI 모델 로드 실패: {e}")
            raise
    
    def is_model_ready(self) -> bool:
        """모델이 준비되었는지 확인"""
        return self.session is not None
    
    def remove_background(self, image_path: str) -> np.ndarray:
        """배경 제거 처리"""
        if not self.session:
            raise RuntimeError("AI 모델이 초기화되지 않았습니다.")
        
        try:
            # 이미지 파일 읽기
            with open(image_path, 'rb') as f:
                input_data = f.read()
            
            # 배경 제거 처리
            result = remove(input_data, session=self.session)
            
            # 마스크 생성
            img_rgba = Image.open(io.BytesIO(result)).convert("RGBA")
            alpha = np.array(img_rgba.split()[-1])
            
            # 실루엣 마스크 생성 (배경은 흰색, 객체는 검은색)
            alpha_threshold = config.get('alpha_threshold', 45)
            mask = np.where(alpha > alpha_threshold, 0, 255).astype(np.uint8)
            mask_rgb = cv2.merge([mask, mask, mask])
            
            return mask_rgb
            
        except Exception as e:
            raise RuntimeError(f"배경 제거 처리 실패: {e}")
    
    def create_composite_preview(self, original_image: np.ndarray, mask_image: np.ndarray) -> np.ndarray:
        """합성 미리보기 생성"""
        try:
            # 간단한 합성 미리보기 (원본 + 마스크 오버레이)
            composite = original_image.copy()
            
            # 마스크를 반투명하게 오버레이
            mask_colored = cv2.applyColorMap(mask_image, cv2.COLORMAP_JET)
            composite = cv2.addWeighted(composite, 0.7, mask_colored, 0.3, 0)
            
            return composite
            
        except Exception as e:
            raise RuntimeError(f"합성 미리보기 생성 실패: {e}")
    
    def validate_image(self, image_path: str) -> tuple[bool, str]:
        """이미지 파일 유효성 검사"""
        try:
            # 지원되는 형식인지 확인
            if not config.is_supported_image(image_path):
                return False, "지원하지 않는 이미지 형식입니다."
            
            # 파일 크기 확인
            import os
            max_size_mb = 50
            file_size_mb = os.path.getsize(image_path) / (1024 * 1024)
            if file_size_mb > max_size_mb:
                return False, f"파일 크기가 너무 큽니다. (최대 {max_size_mb}MB)"
            
            return True, ""
            
        except Exception as e:
            return False, f"파일 검사 중 오류: {e}"
            
            # 마스크 생성
            img_rgba = Image.open(io.BytesIO(result)).convert("RGBA")
            alpha = np.array(img_rgba.split()[-1])
            
            # 실루엣 마스크 생성 (배경은 흰색, 객체는 검은색)
            alpha_threshold = config.get('alpha_threshold', 45)
            mask = np.where(alpha > alpha_threshold, 0, 255).astype(np.uint8)
            mask_rgb = cv2.merge([mask, mask, mask])
            
            return mask_rgb
            
        except Exception as e:
            raise RuntimeError(f"배경 제거 처리 실패: {e}")
    
    def create_composite_preview(self, original_image: np.ndarray, mask_image: np.ndarray) -> np.ndarray:
        """합성 미리보기 생성"""
        try:
            # 간단한 합성 미리보기 (원본 + 마스크 오버레이)
            composite = original_image.copy()
            
            # 마스크를 반투명하게 오버레이
            mask_colored = cv2.applyColorMap(mask_image, cv2.COLORMAP_JET)
            composite = cv2.addWeighted(composite, 0.7, mask_colored, 0.3, 0)
            
            return composite
            
        except Exception as e:
            raise RuntimeError(f"합성 미리보기 생성 실패: {e}")
    
    def validate_image(self, image_path: str) -> tuple[bool, str]:
        """이미지 파일 유효성 검사"""
        try:
            # 지원되는 형식인지 확인
            if not config.is_supported_image(image_path):
                return False, "지원하지 않는 이미지 형식입니다."
            
            # 파일 크기 확인
            import os
            max_size_mb = 50
            file_size_mb = os.path.getsize(image_path) / (1024 * 1024)
            if file_size_mb > max_size_mb:
                return False, f"파일 크기가 너무 큽니다. (최대 {max_size_mb}MB)"
            
            return True, ""
            
        except Exception as e:
            return False, f"파일 검사 중 오류: {e}"