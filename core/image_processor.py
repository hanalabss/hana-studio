"""
core/image_processor.py
이미지 처리 핵심 모듈
"""

import io
import numpy as np
import cv2
from PIL import Image
# rembg는 지연 import (시작 시간 최적화)
from config import config
from .model_loader import get_ai_session


class ImageProcessor:
    """이미지 처리 핵심 클래스 - 세션을 외부에서 주입받아 사용"""

    def __init__(self):
        pass

    def is_model_ready(self) -> bool:
        """모델 준비 상태 확인 (세션 생성 안 함)"""
        from .model_loader import is_ai_model_ready
        return is_ai_model_ready()
    
    def remove_background(self, image_path: str, session, alpha_threshold: int = None) -> np.ndarray:
        """
        배경 제거 및 마스크 생성 - EXIF 정보 보존 (자동 회전 방지)

        Args:
            image_path: 처리할 이미지 경로
            session: AI 모델 세션 (필수, 외부에서 전달)
            alpha_threshold: 알파 임계값 (None이면 config에서 가져옴)

        Raises:
            ValueError: 세션이 None인 경우
            RuntimeError: 배경 제거 처리 중 오류 발생
        """
        if not session:
            raise ValueError("AI 세션이 제공되지 않았습니다. ModelLoadingManager에서 세션을 먼저 로딩해야 합니다.")
        
        try:
            # 임계값 설정
            if alpha_threshold is None:
                alpha_threshold = config.get('alpha_threshold', 200)
            
            print(f"[DEBUG] 배경 제거 시작 - 임계값: {alpha_threshold} (EXIF 정보 보존)")
            
            # 이미지 파일 읽기
            with open(image_path, 'rb') as f:
                input_data = f.read()
            
            # 배경 제거 수행 (지연 import)
            from rembg import remove
            result = remove(input_data, session=session)
            
            # 알파 채널 추출 (EXIF 정보 무시하여 자동 회전 방지)
            img_rgba = Image.open(io.BytesIO(result)).convert("RGBA")
            alpha = np.array(img_rgba.split()[-1])
            
            print(f"[DEBUG] 알파 채널 추출 완료: {alpha.shape} (EXIF 정보 무시)")
            
            # 마스크 이미지 생성 (흰색: 배경, 검은색: 객체)
            mask = np.where(alpha > alpha_threshold, 0, 255).astype(np.uint8)
            mask_rgb = cv2.merge([mask, mask, mask])
            
            print(f"[DEBUG] 마스크 생성 완료 - 임계값: {alpha_threshold}, 마스크 크기: {mask_rgb.shape}")
            
            # 객체 비율 계산 (검정색 픽셀)
            black_pixels = np.sum(mask == 0)  # 객체 픽셀  
            white_pixels = np.sum(mask == 255)  # 배경 픽셀
            total_pixels = mask.size
            object_ratio = (black_pixels / total_pixels) * 100
            
            print(f"[DEBUG] 픽셀 분석 - 객체: {object_ratio:.1f}%, 배경: {100-object_ratio:.1f}%")
            
            # 이미지 방향 확인
            height, width = mask_rgb.shape[:2]
            if height > width:
                print(f"[DEBUG] [OK] 감지됨: 세로 이미지 ({width}x{height}) - 자동 회전 방지됨")
            else:
                print(f"[DEBUG] [OK] 감지됨: 가로 이미지 ({width}x{height}) - 자동 회전 방지됨")
            
            return mask_rgb
            
        except Exception as e:
            raise RuntimeError(f"배경 제거 처리 중 오류: {e}")
        
    def create_composite_preview(self, original_image: np.ndarray, mask_image: np.ndarray) -> np.ndarray:
        """합성 미리보기 이미지 생성"""
        try:
            # 합성 이미지 생성 (원본 + 마스크 오버레이)
            composite = original_image.copy()
            
            # 마스크 컬러맵 적용
            mask_colored = cv2.applyColorMap(mask_image, cv2.COLORMAP_JET)
            composite = cv2.addWeighted(composite, 0.7, mask_colored, 0.3, 0)
            
            return composite
            
        except Exception as e:
            raise RuntimeError(f"합성 미리보기 생성 중 오류: {e}")
    
    def analyze_threshold_effectiveness(self, image_path: str, session, threshold_range: tuple = (50, 250), step: int = 50):
        """
        임계값 자동 분석 (객체/배경 비율)

        Args:
            image_path: 분석할 이미지 경로
            session: AI 모델 세션 (필수, 외부에서 전달)
            threshold_range: 임계값 테스트 범위 (min, max)
            step: 임계값 증가 단계

        Returns:
            dict: 각 임계값별 분석 결과
        """
        if not session:
            raise ValueError("AI 세션이 제공되지 않았습니다.")
        
        try:
            # 원본 이미지 한 번만 처리
            from rembg import remove
            with open(image_path, 'rb') as f:
                input_data = f.read()
            result = remove(input_data, session=session)
            img_rgba = Image.open(io.BytesIO(result)).convert("RGBA")
            alpha = np.array(img_rgba.split()[-1])
            
            # 다양한 임계값 테스트
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
            
            print(f"[DEBUG] 임계값 분석 완료: {len(analysis_results)}개 테스트 완료")
            return analysis_results
            
        except Exception as e:
            raise RuntimeError(f"임계값 분석 중 오류: {e}")
    
    def get_recommended_threshold(self, image_path: str, session) -> int:
        """
        분석 결과에서 최적 임계값 선택 (명함 기준)

        Args:
            image_path: 분석할 이미지 경로
            session: AI 모델 세션

        Returns:
            int: 최적 임계값
        """
        try:
            analysis = self.analyze_threshold_effectiveness(image_path, session, (100, 250), 25)
            
            # 객체 비율이 5-40% 사이인 후보들 찾기
            candidates = []
            for threshold, data in analysis.items():
                object_ratio = data['object_ratio']
                if 5 <= object_ratio <= 40:  # 명함 크기에 적합한 범위
                    candidates.append((threshold, object_ratio))
            
            if candidates:
                # 객체 비율이 15-25% 사이가 되도록 가장 가까운 값 선택
                target_ratio = 20
                best_threshold = min(candidates, key=lambda x: abs(x[1] - target_ratio))[0]
                print(f"[DEBUG] 최적 임계값 선택: {best_threshold}")
                return best_threshold
            else:
                print("[DEBUG] 적합한 임계값을 찾지 못함, 기본값 사용")
                return 200
                
        except Exception as e:
            print(f"[DEBUG] 최적 임계값 선택 중 오류: {e}, 기본값 사용")
            return 200
    
    def validate_image(self, image_path: str) -> tuple[bool, str]:
        """이미지 파일 유효성 검증"""
        try:
            # 지원 형식 확인
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
            return False, f"이미지 검증 중 오류 발생: {e}"