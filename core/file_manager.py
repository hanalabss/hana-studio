"""
파일 관리 로직 - 한글 파일명 지원 및 양면 저장 지원
EXIF 회전 정보 무시하여 항상 픽셀 데이터 그대로 표시
"""

import os
import shutil
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
import uuid
from utils.safe_temp_path import get_cached_safe_temp_dir, create_safe_temp_file
from config import config


class FileManager:
    """파일 저장/로드를 관리하는 클래스 - 한글 파일명 및 양면 이미지 지원"""
    
    def __init__(self):
        # 안전한 임시 디렉토리 사용
        self.temp_dir = get_cached_safe_temp_dir()
        self.output_dir = config.get('directories.output', 'output')
        self._ensure_directories()
    
    def _ensure_directories(self):
        """필요한 디렉토리 생성"""
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _safe_imread(self, image_path: str) -> Optional[np.ndarray]:
        """PyInstaller 호환 안전한 이미지 읽기 - EXIF 회전 정보 적용"""
        try:
            print(f"[DEBUG] 이미지 읽기 시도 (EXIF 회전 적용): {image_path}")
            
            # 파일 존재 확인
            if not os.path.exists(image_path):
                print(f"[ERROR] 파일이 존재하지 않음: {image_path}")
                return None
            
            # 파일 크기 확인
            try:
                file_size = os.path.getsize(image_path)
                if file_size == 0:
                    print(f"[ERROR] 빈 파일: {image_path}")
                    return None
                print(f"[DEBUG] 파일 크기: {file_size} bytes")
            except Exception as e:
                print(f"[ERROR] 파일 크기 확인 실패: {e}")
                return None
            
            # [TARGET] PIL로 EXIF 회전 정보를 적용하여 올바른 방향으로 읽기
            try:
                from PIL import Image, ImageOps
                
                print("[DEBUG] PIL + EXIF 회전 적용 방식 사용")
                
                # PIL로 이미지 열기
                pil_image = Image.open(image_path)
                
                # [TARGET] EXIF 회전 정보 적용 - 올바른 방향으로 회전
                pil_image = ImageOps.exif_transpose(pil_image)
                
                print(f"[DEBUG] EXIF 회전 적용 후 크기: {pil_image.size}")
                
                # RGB 변환
                if pil_image.mode == 'RGBA':
                    pil_image = pil_image.convert('RGB')
                elif pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                
                # OpenCV 형식으로 변환 (RGB → BGR)
                opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                
                print(f"[DEBUG] PIL + EXIF 적용 성공: {opencv_image.shape}")
                return opencv_image
                
            except Exception as e:
                print(f"[ERROR] PIL + EXIF 방식 실패: {e}")
            
            # 백업 방법 1: OpenCV 직접 시도 (영문 경로인 경우)
            try:
                # 경로에 한글이 없는 경우 직접 시도
                image_path.encode('ascii')
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                if image is not None:
                    print(f"[DEBUG] OpenCV 직접 읽기 성공 (EXIF 무시): {image.shape}")
                    print("[WARNING] EXIF 회전 정보가 적용되지 않을 수 있습니다.")
                    return image
            except UnicodeEncodeError:
                print("[DEBUG] 한글 경로 감지, 바이트 방식 사용")
            except Exception as e:
                print(f"[DEBUG] OpenCV 직접 읽기 실패: {e}")
            
            # 백업 방법 2: 바이트 읽기 방식 (한글 경로 대응)
            try:
                with open(image_path, 'rb') as f:
                    image_data = f.read()
                
                if len(image_data) == 0:
                    print("[ERROR] 읽은 데이터가 비어있음")
                    return None
                
                print(f"[DEBUG] 바이트 데이터 크기: {len(image_data)}")
                
                # numpy 배열로 변환
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if image is None:
                    print("[ERROR] OpenCV 디코딩 실패")
                    return None
                    
                print(f"[DEBUG] 바이트 방식 읽기 성공 (EXIF 무시): {image.shape}")
                print("[WARNING] EXIF 회전 정보가 적용되지 않을 수 있습니다.")
                return image
                
            except Exception as e:
                print(f"[ERROR] 바이트 읽기 실패: {e}")
            
            return None
            
        except Exception as e:
            print(f"[ERROR] 전체 이미지 읽기 실패: {image_path}, 오류: {e}")
            import traceback
            traceback.print_exc()
            return None
        
    def _safe_imwrite(self, image_path: str, image: np.ndarray, quality: int = 95) -> bool:
        """한글 경로를 지원하는 안전한 이미지 저장"""
        try:
            # OpenCV가 한글 경로에 저장하지 못하는 문제 해결
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            result, encoded_img = cv2.imencode('.jpg', image, encode_param)
            
            if result:
                with open(image_path, 'wb') as f:
                    f.write(encoded_img.tobytes())
                return True
            else:
                print(f"❌ 이미지 인코딩 실패: {image_path}")
                return False
                
        except Exception as e:
            print(f"❌ 이미지 저장 실패: {image_path}, 오류: {e}")
            return False
    
    def _generate_safe_filename(self, original_path: str, side: str, suffix: str) -> str:
        """안전한 파일명 생성 (한글 문제 방지)"""
        try:
            # 원본 파일명에서 확장자 제거
            base_name = os.path.splitext(os.path.basename(original_path))[0]
            
            # 한글이 포함된 경우 UUID로 대체
            try:
                base_name.encode('ascii')
                safe_name = base_name
            except UnicodeEncodeError:
                # 한글이 포함된 경우 짧은 UUID 사용
                safe_name = f"image_{str(uuid.uuid4())[:8]}"
                print(f"한글 파일명 감지, 안전한 이름으로 변경: {base_name} -> {safe_name}")
            
            return f"{safe_name}_{side}_{suffix}.jpg"
            
        except Exception as e:
            # 최후의 수단: 완전 랜덤 이름
            random_name = f"image_{str(uuid.uuid4())[:8]}_{side}_{suffix}.jpg"
            print(f"파일명 생성 실패, 랜덤 이름 사용: {random_name}")
            return random_name
    
    def save_mask_for_printing(self, mask_image: np.ndarray, original_image_path: str, side: str = "front") -> Optional[str]:
        """프린터용 마스크 이미지 저장 - 한글 파일명 지원"""
        try:
            # 안전한 파일명 생성
            mask_filename = self._generate_safe_filename(original_image_path, side, "mask_print")
            mask_path = os.path.join(self.temp_dir, mask_filename)
            
            # 한글 경로를 지원하는 이미지 저장
            quality = config.get('output_quality', 95)
            success = self._safe_imwrite(mask_path, mask_image, quality)
            
            if success:
                print(f"프린터용 {side} 마스크 저장: {mask_path}")
                return mask_path
            else:
                print(f"❌ 프린터용 {side} 마스크 저장 실패")
                return None
            
        except Exception as e:
            print(f"❌ {side} 마스크 저장 실패: {e}")
            return None
    
    def export_single_result(self, original_image_path: str, mask_image: np.ndarray, 
                           composite_image: Optional[np.ndarray] = None, 
                           output_folder: str = None, side: str = "front") -> Tuple[bool, str]:
        """단일 이미지 결과 저장 - 한글 파일명 지원"""
        try:
            # 저장 폴더 결정
            if output_folder is None:
                output_folder = self.output_dir
            
            quality = config.get('output_quality', 95)
            saved_files = []
            
            # 마스크 이미지 저장
            mask_filename = self._generate_safe_filename(original_image_path, side, "mask")
            mask_path = os.path.join(output_folder, mask_filename)
            
            if self._safe_imwrite(mask_path, mask_image, quality):
                saved_files.append(mask_path)
                print(f"[OK] {side} 마스크 저장: {mask_filename}")
            else:
                print(f"❌ {side} 마스크 저장 실패")
            
            # 합성 이미지 저장 (있는 경우)
            if composite_image is not None:
                composite_filename = self._generate_safe_filename(original_image_path, side, "composite")
                composite_path = os.path.join(output_folder, composite_filename)
                
                if self._safe_imwrite(composite_path, composite_image, quality):
                    saved_files.append(composite_path)
                    print(f"[OK] {side} 합성 이미지 저장: {composite_filename}")
                else:
                    print(f"❌ {side} 합성 이미지 저장 실패")
            
            # 원본 이미지도 복사 (선택사항)
            if config.get('auto_save_original', False):
                try:
                    original_filename = self._generate_safe_filename(original_image_path, side, "original")
                    # 확장자는 원본과 동일하게
                    original_ext = Path(original_image_path).suffix
                    original_filename = original_filename.replace('.jpg', original_ext)
                    original_path = os.path.join(output_folder, original_filename)
                    
                    shutil.copy2(original_image_path, original_path)
                    saved_files.append(original_path)
                    print(f"[OK] {side} 원본 복사: {original_filename}")
                except Exception as e:
                    print(f"[WARNING] {side} 원본 복사 실패: {e}")
            
            if saved_files:
                success_msg = f"{side} 결과 저장 완료: {len(saved_files)}개 파일"
                return True, success_msg
            else:
                error_msg = f"{side} 저장 실패: 저장된 파일이 없습니다"
                return False, error_msg
            
        except Exception as e:
            error_msg = f"{side} 저장 중 오류가 발생했습니다: {e}"
            return False, error_msg
    
    def export_dual_results(self, 
                          front_image_path: str, front_mask_image: np.ndarray,
                          back_image_path: Optional[str] = None, back_mask_image: Optional[np.ndarray] = None,
                          front_composite: Optional[np.ndarray] = None,
                          back_composite: Optional[np.ndarray] = None,
                          output_folder: Optional[str] = None) -> Tuple[bool, str]:
        """양면 이미지 결과 저장 - 한글 파일명 지원"""
        try:
            # 저장 폴더 결정
            if output_folder is None:
                output_folder = self.output_dir
            
            saved_files = []
            all_success = True
            messages = []
            
            # 앞면 저장
            success, msg = self.export_single_result(
                front_image_path, front_mask_image, front_composite, output_folder, "front"
            )
            if success:
                saved_files.append("앞면 결과")
                messages.append(msg)
            else:
                all_success = False
                messages.append(f"❌ {msg}")
            
            # 뒷면 저장 (있는 경우)
            if back_image_path and back_mask_image is not None:
                success, msg = self.export_single_result(
                    back_image_path, back_mask_image, back_composite, output_folder, "back"
                )
                if success:
                    saved_files.append("뒷면 결과")
                    messages.append(msg)
                else:
                    all_success = False
                    messages.append(f"❌ {msg}")
            
            # 최종 메시지 구성
            if all_success and saved_files:
                final_msg = f"양면 결과 저장 완료!\n📁 저장 위치: {output_folder}\n[OK] {', '.join(saved_files)}"
            elif saved_files:
                final_msg = f"일부 저장 완료: {', '.join(saved_files)}\n[WARNING] 일부 오류 발생"
            else:
                final_msg = "저장 실패\n" + "\n".join(messages)
            
            return all_success, final_msg
            
        except Exception as e:
            error_msg = f"양면 저장 중 오류가 발생했습니다: {e}"
            return False, error_msg
    
    def export_results(self, original_image_path: str, mask_image: np.ndarray, 
                      composite_image: Optional[np.ndarray] = None, 
                      output_folder: Optional[str] = None) -> Tuple[bool, str]:
        """기존 단일 결과 저장 (하위 호환성)"""
        return self.export_single_result(
            original_image_path, mask_image, composite_image, output_folder, "result"
        )
    
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
                    if any(pattern in file for pattern in ['_mask_print.jpg', '_temp.jpg', '_front_', '_back_']):
                        file_path = os.path.join(self.temp_dir, file)
                        try:
                            os.remove(file_path)
                            print(f"임시 파일 삭제: {file}")
                        except Exception as e:
                            print(f"임시 파일 삭제 실패: {file}, 오류: {e}")
        except Exception as e:
            print(f"임시 파일 정리 중 오류: {e}")
    
    def validate_output_directory(self, directory: str) -> bool:
        """출력 디렉토리 유효성 확인"""
        try:
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            
            # 쓰기 권한 확인
            test_file = os.path.join(directory, "test_write.tmp")
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write("test")
            os.remove(test_file)
            
            return True
        except Exception:
            return False

    def get_saved_files_info(self, output_folder: str, base_name: str) -> List[str]:
        """저장된 파일들의 정보 반환"""
        saved_files = []
        
        try:
            # 저장 폴더의 모든 파일 확인
            if os.path.exists(output_folder):
                for file in os.listdir(output_folder):
                    if any(pattern in file for pattern in ['_front_', '_back_', '_mask', '_composite']):
                        file_path = os.path.join(output_folder, file)
                        if os.path.exists(file_path):
                            file_size = os.path.getsize(file_path) / 1024  # KB
                            saved_files.append(f"{file} ({file_size:.1f}KB)")
        except Exception as e:
            print(f"저장된 파일 정보 조회 실패: {e}")
        
        return saved_files