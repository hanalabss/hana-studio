"""
파일 관리 로직
"""

import os
import shutil
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from config import config


class FileManager:
    """파일 저장/로드를 관리하는 클래스"""
    
    def __init__(self):
        self.temp_dir = config.get('directories.temp', 'temp')
        self.output_dir = config.get('directories.output', 'output')
        self._ensure_directories()
    
    def _ensure_directories(self):
        """필요한 디렉토리 생성"""
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
    
    def save_mask_for_printing(self, mask_image: np.ndarray, original_image_path: str) -> Optional[str]:
        """프린터용 마스크 이미지 저장"""
        try:
            base_name = os.path.splitext(os.path.basename(original_image_path))[0]
            mask_filename = f"{base_name}_mask_print.jpg"
            mask_path = os.path.join(self.temp_dir, mask_filename)
            
            # 마스크 이미지 저장
            quality = config.get('output_quality', 95)
            cv2.imwrite(mask_path, mask_image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            
            print(f"프린터용 마스크 저장: {mask_path}")
            return mask_path
            
        except Exception as e:
            print(f"❌ 마스크 저장 실패: {e}")
            return None
    
    def export_results(self, original_image_path: str, mask_image: np.ndarray, 
                      composite_image: Optional[np.ndarray] = None, 
                      output_folder: Optional[str] = None) -> Tuple[bool, str]:
        """처리 결과 저장"""
        try:
            # 저장 폴더 결정
            if output_folder is None:
                output_folder = self.output_dir
            
            base_name = os.path.splitext(os.path.basename(original_image_path))[0]
            quality = config.get('output_quality', 95)
            
            saved_files = []
            
            # 마스크 이미지 저장
            mask_path = os.path.join(output_folder, f"{base_name}_mask.jpg")
            cv2.imwrite(mask_path, mask_image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            saved_files.append(mask_path)
            
            # 합성 이미지 저장 (있는 경우)
            if composite_image is not None:
                composite_path = os.path.join(output_folder, f"{base_name}_composite.jpg")
                cv2.imwrite(composite_image, composite_path, [cv2.IMWRITE_JPEG_QUALITY, quality])
                saved_files.append(composite_path)
            
            # 원본 이미지도 복사 (선택사항)
            if config.get('auto_save_original', False):
                original_path = os.path.join(output_folder, 
                                           f"{base_name}_original{Path(original_image_path).suffix}")
                shutil.copy2(original_image_path, original_path)
                saved_files.append(original_path)
            
            success_msg = f"결과 저장 완료: {output_folder}\n저장된 파일: {len(saved_files)}개"
            return True, success_msg
            
        except Exception as e:
            error_msg = f"저장 중 오류가 발생했습니다: {e}"
            return False, error_msg
    
    def get_file_info(self, file_path: str) -> Tuple[str, float]:
        """파일 정보 반환 (이름, 크기MB)"""
        try:
            file_name = os.path.basename(file_path)
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            return file_name, file_size_mb
        except Exception:
            return "Unknown", 0.0
    
    def cleanup_temp_files(self):
        """임시 파일 정리"""
        try:
            if os.path.exists(self.temp_dir):
                for file in os.listdir(self.temp_dir):
                    if file.endswith(('_mask_print.jpg', '_temp.jpg')):
                        file_path = os.path.join(self.temp_dir, file)
                        os.remove(file_path)
                        print(f"임시 파일 삭제: {file}")
        except Exception as e:
            print(f"임시 파일 정리 중 오류: {e}")
    
    def validate_output_directory(self, directory: str) -> bool:
        """출력 디렉토리 유효성 확인"""
        try:
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            
            # 쓰기 권한 확인
            test_file = os.path.join(directory, "test_write.tmp")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            
            return True
        except Exception:
            return False