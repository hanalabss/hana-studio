"""
core/image_processor.py 수정
동적 임계값 지원
"""

import io
import numpy as np
import cv2
from PIL import Image
from rembg import remove, new_session
from config import config


class ImageProcessor:
    """이미지 처리를 담당하는 클래스 - 동적 임계값 지원"""
    
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
    
    def remove_background(self, image_path: str, alpha_threshold: int = None) -> np.ndarray:
        """
        배경 제거 처리 - EXIF 회전 무시 (인쇄 결과와 일치)
        
        Args:
            image_path: 이미지 파일 경로
            alpha_threshold: 알파 임계값 (None이면 config에서 가져옴)
        """
        if not self.session:
            raise RuntimeError("AI 모델이 초기화되지 않았습니다.")
        
        try:
            # 임계값 결정
            if alpha_threshold is None:
                alpha_threshold = config.get('alpha_threshold', 200)
            
            print(f"[DEBUG] 배경 제거 시작 - 임계값: {alpha_threshold} (EXIF 회전 무시)")
            
            # 이미지 파일 읽기
            with open(image_path, 'rb') as f:
                input_data = f.read()
            
            # 배경 제거 처리
            result = remove(input_data, session=self.session)
            
            # 마스크 생성 (EXIF 회전 적용하지 않음)
            img_rgba = Image.open(io.BytesIO(result)).convert("RGBA")
            alpha = np.array(img_rgba.split()[-1])
            
            print(f"[DEBUG] 원본 마스크 크기: {alpha.shape} (EXIF 회전 무시)")
            
            # 실루엣 마스크 생성 (배경은 흰색, 객체는 검은색)
            mask = np.where(alpha > alpha_threshold, 0, 255).astype(np.uint8)
            mask_rgb = cv2.merge([mask, mask, mask])
            
            print(f"[DEBUG] 마스크 생성 완료 - 임계값: {alpha_threshold}, 마스크 크기: {mask_rgb.shape}")
            
            # 마스크 통계 출력 (디버깅용)
            black_pixels = np.sum(mask == 0)  # 객체 픽셀  
            white_pixels = np.sum(mask == 255)  # 배경 픽셀
            total_pixels = mask.size
            object_ratio = (black_pixels / total_pixels) * 100
            
            print(f"[DEBUG] 마스크 통계 - 객체: {object_ratio:.1f}%, 배경: {100-object_ratio:.1f}%")
            
            # 🔍 최종 확인
            height, width = mask_rgb.shape[:2]
            if height > width:
                print(f"[DEBUG] ✅ 마스크: 세로 형태 ({width}x{height}) - 원본과 일치")
            else:
                print(f"[DEBUG] ✅ 마스크: 가로 형태 ({width}x{height}) - 원본과 일치")
            
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
    
    def analyze_threshold_effectiveness(self, image_path: str, threshold_range: tuple = (50, 250), step: int = 50):
        """
        임계값 효과 분석 (개발/디버깅용)
        
        Args:
            image_path: 이미지 경로
            threshold_range: 테스트할 임계값 범위 (min, max)
            step: 임계값 증가 단위
            
        Returns:
            dict: 각 임계값별 객체 비율
        """
        if not self.session:
            raise RuntimeError("AI 모델이 초기화되지 않았습니다.")
        
        try:
            # 배경 제거 한 번만 수행
            with open(image_path, 'rb') as f:
                input_data = f.read()
            result = remove(input_data, session=self.session)
            img_rgba = Image.open(io.BytesIO(result)).convert("RGBA")
            alpha = np.array(img_rgba.split()[-1])
            
            # 각 임계값별 분석
            analysis_results = {}
            min_threshold, max_threshold = threshold_range
            
            for threshold in range(min_threshold, max_threshold + 1, step):
                mask = np.where(alpha > threshold, 0, 255).astype(np.uint8)
                object_pixels = np.sum(mask == 0)
                total_pixels = mask.size
                object_ratio = (object_pixels / total_pixels) * 100
                
                analysis_results[threshold] = {
                    'object_ratio': object_ratio,
                    'background_ratio': 100 - object_ratio
                }
            
            print(f"[DEBUG] 임계값 분석 완료: {len(analysis_results)}개 임계값 테스트")
            return analysis_results
            
        except Exception as e:
            raise RuntimeError(f"임계값 분석 실패: {e}")
    
    def get_recommended_threshold(self, image_path: str) -> int:
        """
        이미지에 최적화된 임계값 추천 (실험적 기능)
        
        Returns:
            int: 추천 임계값
        """
        try:
            analysis = self.analyze_threshold_effectiveness(image_path, (100, 250), 25)
            
            # 객체 비율이 5-40% 사이인 임계값 중에서 선택
            candidates = []
            for threshold, data in analysis.items():
                object_ratio = data['object_ratio']
                if 5 <= object_ratio <= 40:  # 적절한 객체 비율 범위
                    candidates.append((threshold, object_ratio))
            
            if candidates:
                # 객체 비율이 15-25% 사이에 가장 가까운 임계값 선택
                target_ratio = 20
                best_threshold = min(candidates, key=lambda x: abs(x[1] - target_ratio))[0]
                print(f"[DEBUG] 추천 임계값: {best_threshold}")
                return best_threshold
            else:
                print("[DEBUG] 적절한 임계값을 찾지 못함, 기본값 사용")
                return 200
                
        except Exception as e:
            print(f"[DEBUG] 임계값 추천 실패: {e}, 기본값 사용")
            return 200
    
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