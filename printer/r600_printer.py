"""
printer/r600_printer.py 수정
개별 면 방향 지원 + 세로형 뒷면 마스킹 180도 회전 수정
"""

import ctypes
import os
import time
from typing import List, Optional, Tuple
from .exceptions import R600PrinterError, PrinterInitializationError, DLLNotFoundError
from .printer_discovery import PrinterInfo


class R600Printer:
    """R600 프린터 제어 클래스 - 개별 면 방향 지원 + 마스킹 회전 수정"""
    
    def __init__(self, dll_path: str = './libDSRetransfer600App.dll', selected_printer: Optional[PrinterInfo] = None):
        """R600 프린터 초기화"""
        self.lib = None
        self.selected_printer_info = selected_printer
        self.selected_printer_name = None
        self.front_img_info = None
        self.back_img_info = None
        self.is_initialized = False

        try:
            print(f"[DEBUG] DLL 경로 시도: {dll_path}")
            if not os.path.exists(dll_path):
                raise DLLNotFoundError(f"DLL 파일이 존재하지 않음: {dll_path}")

            self.lib = ctypes.CDLL(dll_path)
            print("[DEBUG] DLL 로드 성공")

            self._setup_function_signatures()
            self._initialize_library()
            self.is_initialized = True
            
            # 선택된 프린터가 있으면 자동으로 설정
            if self.selected_printer_info:
                self.auto_select_printer()
            
        except Exception as e:
            print(f"[DEBUG] 예외 발생 위치: {type(e).__name__}: {e}")
            raise PrinterInitializationError(f"프린터 초기화 실패: {e}")

    def _setup_function_signatures(self):
        """함수 시그니처 정의"""
        # 기본 함수들
        self.lib.R600EnumTcpPrt.argtypes = [
            ctypes.POINTER(ctypes.c_char), 
            ctypes.POINTER(ctypes.c_uint), 
            ctypes.POINTER(ctypes.c_int)
        ]
        self.lib.R600EnumTcpPrt.restype = ctypes.c_uint
        
        # USB 프린터 열거 함수 추가
        self.lib.R600EnumUsbPrt.argtypes = [
            ctypes.POINTER(ctypes.c_char), 
            ctypes.POINTER(ctypes.c_uint), 
            ctypes.POINTER(ctypes.c_int)
        ]
        self.lib.R600EnumUsbPrt.restype = ctypes.c_uint
        
        self.lib.R600TcpSetTimeout.argtypes = [ctypes.c_int, ctypes.c_int]
        self.lib.R600TcpSetTimeout.restype = ctypes.c_uint
        
        self.lib.R600SelectPrt.argtypes = [ctypes.POINTER(ctypes.c_char)]
        self.lib.R600SelectPrt.restype = ctypes.c_uint
        
        # 카드 관련
        self.lib.R600CardInject.argtypes = [ctypes.c_int]
        self.lib.R600CardInject.restype = ctypes.c_uint
        
        self.lib.R600CardEject.argtypes = [ctypes.c_int]
        self.lib.R600CardEject.restype = ctypes.c_uint
        
        # 카드 뒤집기 - 양면 인쇄의 핵심
        self.lib.R600CardTurnover.argtypes = []
        self.lib.R600CardTurnover.restype = ctypes.c_uint
        
        # 리본 옵션
        self.lib.R600SetRibbonOpt.argtypes = [
            ctypes.c_ubyte, ctypes.c_uint, ctypes.c_char_p, ctypes.c_uint
        ]
        self.lib.R600SetRibbonOpt.restype = ctypes.c_uint
        
        # 캔버스 관련
        self.lib.R600SetCanvasPortrait.argtypes = [ctypes.c_int]
        self.lib.R600SetCanvasPortrait.restype = ctypes.c_uint
        
        self.lib.R600PrepareCanvas.argtypes = [ctypes.c_int, ctypes.c_int]
        self.lib.R600PrepareCanvas.restype = ctypes.c_uint
        
        self.lib.R600ClearCanvas.argtypes = []
        self.lib.R600ClearCanvas.restype = ctypes.c_uint
        
        self.lib.R600CommitCanvas.argtypes = [
            ctypes.POINTER(ctypes.c_char), 
            ctypes.POINTER(ctypes.c_uint)
        ]
        self.lib.R600CommitCanvas.restype = ctypes.c_uint
        
        # 이미지 관련
        self.lib.R600SetImagePara.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_float]
        self.lib.R600SetImagePara.restype = ctypes.c_uint
        
        self.lib.R600DrawWaterMark.argtypes = [
            ctypes.c_double, ctypes.c_double, ctypes.c_double, 
            ctypes.c_double, ctypes.c_char_p
        ]
        self.lib.R600DrawWaterMark.restype = ctypes.c_uint
        
        self.lib.R600DrawImage.argtypes = [
            ctypes.c_double, ctypes.c_double, ctypes.c_double, 
            ctypes.c_double, ctypes.c_char_p, ctypes.c_int
        ]
        self.lib.R600DrawImage.restype = ctypes.c_uint
        
        # 양면 인쇄 - 앞면과 뒷면을 모두 받는 함수
        self.lib.R600PrintDraw.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        self.lib.R600PrintDraw.restype = ctypes.c_uint
    
    def _initialize_library(self):
        """라이브러리 초기화"""
        ret = self.lib.R600LibInit()
        if ret != 0:
            raise R600PrinterError(f"라이브러리 초기화 실패: {ret}")
        print(f"라이브러리 초기화 성공: {ret}")
    
    def _check_result(self, result: int, operation: str):
        """결과 코드 확인"""
        if result == 0:
            print(f"{operation} 성공")
        else:
            raise R600PrinterError(f"{operation} 실패: 오류 코드 {result}")
    
    def auto_select_printer(self):
        """초기화 시 선택된 프린터 자동 설정"""
        if not self.selected_printer_info:
            print("⚠️ 선택된 프린터 정보가 없습니다.")
            return False
        
        try:
            # 선택된 프린터로 바로 설정
            printer_name = self.selected_printer_info.name
            print(f"🎯 선택된 프린터로 자동 설정: {printer_name} ({self.selected_printer_info.connection_type})")
            
            ret = self.lib.R600SelectPrt(printer_name.encode('cp949'))
            self._check_result(ret, f"프린터 자동 선택 ({printer_name})")
            self.selected_printer_name = printer_name
            
            print(f"✅ 프린터 자동 선택 완료: {printer_name}")
            return True
            
        except Exception as e:
            print(f"❌ 프린터 자동 선택 실패: {e}")
            return False
    
    def enum_printers(self) -> List[str]:
        """프린터 목록 조회 - 선택된 프린터 타입에 맞춰 조회"""
        if self.selected_printer_info:
            # 이미 선택된 프린터가 있으면 그것만 반환
            return [self.selected_printer_info.name]
        
        # 선택된 프린터 정보가 없으면 TCP와 USB 모두 조회
        all_printers = []
        
        # TCP 프린터 조회
        tcp_printers = self._enum_tcp_printers()
        all_printers.extend(tcp_printers)
        
        # USB 프린터 조회
        usb_printers = self._enum_usb_printers()
        all_printers.extend(usb_printers)
        
        return all_printers
    
    def _enum_tcp_printers(self) -> List[str]:
        """TCP 프린터 목록 조회"""
        printers = []
        
        try:
            list_buffer_size = 1024
            printer_list_buffer = ctypes.create_string_buffer(list_buffer_size)
            enum_list_len = ctypes.c_uint(list_buffer_size)
            num_printers = ctypes.c_int()
            
            ret = self.lib.R600EnumTcpPrt(
                printer_list_buffer, 
                ctypes.byref(enum_list_len), 
                ctypes.byref(num_printers)
            )
            
            if ret == 0:
                actual_len = enum_list_len.value
                printer_count = num_printers.value
                
                if actual_len > 0 and printer_count > 0:
                    printer_names_str = printer_list_buffer.value.decode('cp949')
                    printer_names = [name.strip() for name in printer_names_str.split('\n') if name.strip()]
                    printers.extend(printer_names)
                    print(f"📡 TCP 프린터 {len(printer_names)}대 발견")
        
        except Exception as e:
            print(f"❌ TCP 프린터 조회 중 오류: {e}")
        
        return printers
    
    def _enum_usb_printers(self) -> List[str]:
        """USB 프린터 목록 조회"""
        printers = []
        
        try:
            list_buffer_size = 1024
            printer_list_buffer = ctypes.create_string_buffer(list_buffer_size)
            enum_list_len = ctypes.c_uint(list_buffer_size)
            num_printers = ctypes.c_int()
            
            ret = self.lib.R600EnumUsbPrt(
                printer_list_buffer, 
                ctypes.byref(enum_list_len), 
                ctypes.byref(num_printers)
            )
            
            if ret == 0:
                actual_len = enum_list_len.value
                printer_count = num_printers.value
                
                if actual_len > 0 and printer_count > 0:
                    printer_names_str = printer_list_buffer.value.decode('cp949')
                    printer_names = [name.strip() for name in printer_names_str.split('\n') if name.strip()]
                    printers.extend(printer_names)
                    print(f"🔌 USB 프린터 {len(printer_names)}대 발견")
        
        except Exception as e:
            print(f"❌ USB 프린터 조회 중 오류: {e}")
        
        return printers
    
    def set_timeout(self, timeout_ms: int = 10000):
        """타임아웃 설정"""
        ret = self.lib.R600TcpSetTimeout(timeout_ms, timeout_ms)
        self._check_result(ret, "타임아웃 설정")
    
    def select_printer(self, printer_name: str):
        """프린터 선택"""
        ret = self.lib.R600SelectPrt(printer_name.encode('cp949'))
        self._check_result(ret, f"프린터 선택 ({printer_name})")
        self.selected_printer_name = printer_name
    
    def inject_card(self):
        """카드 삽입"""
        ret = self.lib.R600CardInject(0)
        self._check_result(ret, "카드 삽입")
    
    def eject_card(self):
        """카드 배출"""
        ret = self.lib.R600CardEject(0)
        self._check_result(ret, "카드 배출")
    
    def turnover_card(self):
        """카드 뒤집기 - 양면 인쇄의 핵심 기능"""
        ret = self.lib.R600CardTurnover()
        self._check_result(ret, "카드 뒤집기")
    
    def set_ribbon_option(self, ribbon_type: int = 1, key: int = 0, value: str = "2"):
        """리본 옵션 설정"""
        value_bytes = value.encode('cp949')
        ret = self.lib.R600SetRibbonOpt(ribbon_type, key, value_bytes, len(value_bytes))
        self._check_result(ret, "리본 옵션 설정")
    
    def setup_canvas(self, card_orientation: str = "portrait", rotation: int = 0):
        """카드 방향을 고려한 캔버스 설정 + 회전"""
        portrait_mode = (card_orientation == "portrait")
        ret = self.lib.R600SetCanvasPortrait(1 if portrait_mode else 0)
        self._check_result(ret, f"캔버스 방향 설정 ({card_orientation})")

        ret = self.lib.R600PrepareCanvas(0, 0)
        self._check_result(ret, "캔버스 준비")

        # 회전 각도 적용
        ret = self.lib.R600SetImagePara(1, rotation, 0.0)
        self._check_result(ret, f"이미지 파라미터 설정 (회전 {rotation}도)")

        
    def get_card_dimensions(self, orientation: str) -> tuple:
        """카드 방향에 따른 크기 반환"""
        if orientation == "portrait":
            return 55, 86.6  # 세로형
        else:
            return 86.6, 55  # 가로형
        
    def clear_canvas(self):
        """캔버스 클리어"""
        ret = self.lib.R600ClearCanvas()
        self._check_result(ret, "캔버스 클리어")
    
    def draw_watermark(self, x: float, y: float, width: float, height: float, image_path: str):
        """워터마크 그리기 - EXIF 제거 후 전송"""
        if image_path and not os.path.exists(image_path):
            raise R600PrinterError(f"이미지 파일 {image_path}이 존재하지 않습니다.")
        
        if not image_path:
            path_encoded = b""
        else:
            # 🔧 EXIF 정보를 제거한 임시 워터마크 생성
            try:
                from PIL import Image as PILImage
                import tempfile
                
                print(f"[PRINTER DEBUG] 워터마크 EXIF 제거 처리")
                
                # 원본 워터마크 열기
                original_watermark = PILImage.open(image_path)
                print(f"  워터마크 원본 크기: {original_watermark.size}")
                
                # RGB 모드로 변환 (EXIF 정보 자동 제거됨)
                if original_watermark.mode in ('RGBA', 'LA', 'P'):
                    clean_watermark = original_watermark.convert('RGB')
                elif original_watermark.mode != 'RGB':
                    clean_watermark = original_watermark.convert('RGB')
                else:
                    # 이미 RGB인 경우에도 새 이미지로 복사해서 EXIF 제거
                    clean_watermark = PILImage.new('RGB', original_watermark.size)
                    clean_watermark.paste(original_watermark)
                
                # 임시 파일로 저장 (EXIF 없음)
                temp_dir = tempfile.gettempdir()
                temp_name = f"watermark_clean_{int(time.time())}.jpg"
                temp_path = os.path.join(temp_dir, temp_name)
                
                # 고품질로 저장
                clean_watermark.save(temp_path, 'JPEG', quality=95, optimize=True)
                
                print(f"  EXIF 제거된 워터마크: {temp_path}")
                
                path_encoded = temp_path.encode('cp949')
                
            except Exception as e:
                print(f"[PRINTER DEBUG] 워터마크 EXIF 제거 실패: {e}")
                # 실패 시 원본 파일 사용
                try:
                    image_path = os.path.abspath(image_path)
                    path_encoded = image_path.encode('cp949')
                except UnicodeEncodeError:
                    # 한글 경로 처리
                    import shutil
                    temp_dir = tempfile.gettempdir()
                    temp_name = f"temp_watermark_{int(time.time())}.jpg"
                    temp_path = os.path.join(temp_dir, temp_name)
                    shutil.copy2(image_path, temp_path)
                    path_encoded = temp_path.encode('cp949')
                    print(f"워터마크 임시 파일 생성: {temp_path}")

        adjusted_x = x - 0  # 좌표 조정 없음
        adjusted_y = y - 0  # 좌표 조정 없음

        ret = self.lib.R600DrawWaterMark(adjusted_x, adjusted_y, width, height, path_encoded)
        self._check_result(ret, f"워터마크 그리기 ({image_path})")
        
    def draw_watermark_rotated(self, x: float, y: float, width: float, height: float, 
                              image_path: str, rotation: int):
        """마스킹 이미지를 회전하여 그리기"""
        if not image_path or not os.path.exists(image_path):
            raise R600PrinterError(f"이미지 파일 {image_path}이 존재하지 않습니다.")
        
        if rotation == 0:
            # 회전 없으면 기존 메서드 사용
            self.draw_watermark(x, y, width, height, image_path)
            return
        
        # 이미지를 회전시켜 임시 파일로 저장
        rotated_path = self._create_rotated_mask(image_path, rotation)
        
        # 회전된 이미지로 워터마크 그리기
        self.draw_watermark(x, y, width, height, rotated_path)

    def _create_rotated_mask(self, image_path: str, rotation: int) -> str:
        """마스킹 이미지를 회전시켜 임시 파일로 저장"""
        import cv2
        import numpy as np
        from utils.safe_temp_path import create_safe_temp_file
        
        try:
            print(f"마스킹 이미지 {rotation}도 회전 중: {image_path}")
            
            # 이미지 읽기 (한글 경로 대응)
            try:
                image = cv2.imread(image_path)
            except:
                # 한글 경로 대응
                with open(image_path, 'rb') as f:
                    image_data = f.read()
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise R600PrinterError(f"마스킹 이미지 로드 실패: {image_path}")
            
            # 회전 적용
            if rotation == 180:
                rotated_image = cv2.rotate(image, cv2.ROTATE_180)
            elif rotation == 90:
                rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            elif rotation == 270:
                rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            else:
                rotated_image = image
            
            # 안전한 임시 파일 경로 생성
            temp_path = create_safe_temp_file(
                prefix=f"rotated_mask_{rotation}deg",
                suffix=".jpg"
            )
            
            # 높은 품질로 저장
            cv2.imwrite(temp_path, rotated_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            print(f"회전된 마스킹 이미지 저장: {temp_path}")
            return temp_path
            
        except Exception as e:
            print(f"마스킹 이미지 회전 실패: {e}")
            # 실패 시 원본 반환
            return image_path
    
    def draw_image(self, x: float, y: float, width: float, height: float, 
                image_path: str, mode: int = 1):
        """이미지 그리기 - EXIF 회전 적용 후 EXIF 제거"""
        if not os.path.exists(image_path):
            raise R600PrinterError(f"이미지 파일 {image_path}이 존재하지 않습니다.")
        
        # 🔧 EXIF 회전을 적용한 후 EXIF 정보 제거
        try:
            from PIL import Image as PILImage, ImageOps
            from utils.safe_temp_path import create_safe_temp_file
            
            print(f"[PRINTER DEBUG] EXIF 회전 적용 후 제거 처리")
            
            # 원본 이미지 열기
            original_image = PILImage.open(image_path)
            print(f"  원본 크기: {original_image.size}")
            
            # EXIF 정보 확인
            if hasattr(original_image, '_getexif') and original_image._getexif() is not None:
                exif_data = original_image._getexif()
                orientation = exif_data.get(274, 1)
                print(f"  적용할 EXIF Orientation: {orientation}")
            else:
                print(f"  EXIF 정보 없음")
                orientation = 1
            
            # 🎯 핵심: EXIF 회전을 실제 픽셀에 적용
            rotated_image = ImageOps.exif_transpose(original_image)
            print(f"  EXIF 회전 적용 후 크기: {rotated_image.size}")
            
            # 회전이 적용되었는지 확인
            if original_image.size != rotated_image.size:
                print(f"  ✅ 이미지 회전됨: {original_image.size} → {rotated_image.size}")
            else:
                print(f"  ℹ️ 회전 없음 (이미 올바른 방향)")
            
            # RGB 모드로 변환 (EXIF 정보 자동 제거됨)
            if rotated_image.mode in ('RGBA', 'LA', 'P'):
                clean_image = rotated_image.convert('RGB')
            elif rotated_image.mode != 'RGB':
                clean_image = rotated_image.convert('RGB')
            else:
                # 이미 RGB인 경우에도 새 이미지로 복사해서 EXIF 제거
                clean_image = PILImage.new('RGB', rotated_image.size)
                clean_image.paste(rotated_image)
            
            print(f"  최종 이미지 크기: {clean_image.size}")
            
            # 안전한 임시 파일로 저장 (EXIF 없음, 회전 적용됨)
            temp_path = create_safe_temp_file(
                prefix="printer_rotated",
                suffix=".jpg"
            )
            
            # 고품질로 저장 (EXIF 정보는 저장되지 않음)
            clean_image.save(temp_path, 'JPEG', quality=95, optimize=True)
            
            print(f"  회전+EXIF제거 파일: {temp_path}")
            
            # 검증: 저장된 파일에 EXIF가 없는지 확인
            verify_image = PILImage.open(temp_path)
            if hasattr(verify_image, '_getexif') and verify_image._getexif() is not None:
                print(f"  ⚠️ EXIF 제거 실패!")
            else:
                print(f"  ✅ EXIF 제거 성공!")
                
            # 최종 형태 확인
            if verify_image.size[0] > verify_image.size[1]:
                print(f"  📄 최종: 가로 형태 ({verify_image.size[0]}x{verify_image.size[1]})")
            else:
                print(f"  📄 최종: 세로 형태 ({verify_image.size[0]}x{verify_image.size[1]})")
            
            # 프린터에 회전된+EXIF없는 파일 전송
            path_encoded = temp_path.encode('cp949')
            image_path_for_log = temp_path
            
        except Exception as e:
            print(f"[PRINTER DEBUG] EXIF 처리 실패: {e}")
            # 실패 시 원본 파일 사용 (기존 방식)
            try:
                image_path = os.path.abspath(image_path)
                path_encoded = image_path.encode('cp949')
                image_path_for_log = image_path
            except UnicodeEncodeError:
                # 한글 경로 처리
                import shutil
                temp_dir = tempfile.gettempdir()
                temp_name = f"temp_image_{int(time.time())}.jpg"
                temp_path = os.path.join(temp_dir, temp_name)
                shutil.copy2(image_path, temp_path)
                path_encoded = temp_path.encode('cp949')
                image_path_for_log = temp_path
                print(f"임시 파일 생성: {temp_path}")

        adjusted_x = x - 0  # 좌표 조정 없음
        adjusted_y = y - 0  # 좌표 조정 없음
        ret = self.lib.R600DrawImage(adjusted_x, adjusted_y, width, height, path_encoded, mode)
        self._check_result(ret, f"이미지 그리기 ({image_path_for_log})")
        
    def commit_canvas(self) -> str:
        """캔버스 커밋"""
        img_info_buffer_size = 200
        img_info_buffer = ctypes.create_string_buffer(img_info_buffer_size)
        p_img_info_len = ctypes.pointer(ctypes.c_uint(img_info_buffer_size))
        
        ret = self.lib.R600CommitCanvas(img_info_buffer, p_img_info_len)
        if ret != 0:
            raise R600PrinterError(f"캔버스 커밋 실패: {ret}")
        
        img_info = img_info_buffer.value.decode('cp949')
        print(f"캔버스 커밋 성공. 이미지 정보: {img_info}")
        print(f"실제 이미지 정보 길이: {p_img_info_len.contents.value}")
        
        return img_info
    
    def prepare_front_canvas(self, front_image_path: str, watermark_path: Optional[str] = None,
                           card_width: float = 55, card_height: float = 86.6,
                           card_orientation: str = "portrait") -> str:
        """카드 방향을 고려한 앞면 캔버스 준비"""
        print(f"=== 앞면 캔버스 준비 ({card_orientation}) ===")
        
        # 캔버스 설정 (카드 방향 적용)
        self.setup_canvas(card_orientation)
        
        # 워터마크 그리기 (레이어 모드인 경우)
        if watermark_path:
            self.draw_watermark(0.0, 0.0, card_width, card_height, watermark_path)
        
        # 앞면 이미지 그리기
        self.draw_image(0.0, 0.0, card_width, card_height, front_image_path)
        
        # 캔버스 커밋
        self.front_img_info = self.commit_canvas()
        return self.front_img_info
    
    def prepare_back_canvas(self, back_image_path: Optional[str] = None, 
                          watermark_path: Optional[str] = None,
                          card_width: float = 55, card_height: float = 86.6,
                          card_orientation: str = "portrait") -> str:
        """카드 방향을 고려한 뒷면 캔버스 준비 - 세로형 마스킹만 추가 회전"""
        print(f"=== 뒷면 캔버스 준비 ({card_orientation}) ===")
        
        # 캔버스 클리어 및 재설정
        self.clear_canvas()
        
        # 🎯 원본 이미지 회전은 기존 로직 유지
        rotation = 0 if card_orientation == "landscape" else 180
        print(f"뒷면 회전 각도: {rotation}도 (방향: {card_orientation})")
        self.setup_canvas(card_orientation, rotation)
        
        # 뒷면 이미지가 있는 경우
        if back_image_path and os.path.exists(back_image_path):
            # 🎯 워터마크(마스킹) 그리기 - 세로형일 때만 추가 180도 회전
            if watermark_path:
                if card_orientation == "portrait":
                    # 세로형: 마스킹을 180도 더 회전 (총 360도 = 0도와 동일한 효과)
                    print("세로형 뒷면: 마스킹 이미지 180도 추가 회전 적용")
                    self.draw_watermark_rotated(0.0, 0.0, card_width, card_height, watermark_path, 180)
                else:
                    # 가로형: 마스킹 회전 없음 (현재 상태 유지)
                    print("가로형 뒷면: 마스킹 이미지 회전 없음")
                    self.draw_watermark(0.0, 0.0, card_width, card_height, watermark_path)
            
            # 뒷면 이미지 그리기 (기존과 동일)
            self.draw_image(0.0, 0.0, card_width, card_height, back_image_path)
        else:
            # 뒷면 이미지가 없으면 빈 캔버스 또는 기본 이미지
            print("뒷면 이미지가 없습니다. 빈 뒷면으로 설정합니다.")
        
        # 캔버스 커밋
        self.back_img_info = self.commit_canvas()
        return self.back_img_info
    
    def print_dual_side_card(self, front_image_path: str, back_image_path: Optional[str] = None,
                           front_watermark_path: Optional[str] = None,
                           back_watermark_path: Optional[str] = None,
                           front_orientation: str = "portrait",  # 개별 면 방향 추가
                           back_orientation: str = "portrait",   # 개별 면 방향 추가
                           print_mode: str = "normal"):
        """양면 카드 인쇄 - 개별 면 방향 지원"""
        try:
            front_orientation_text = "세로형" if front_orientation == "portrait" else "가로형"
            back_orientation_text = "세로형" if back_orientation == "portrait" else "가로형"
            print(f"=== 양면 카드 인쇄 시작: 앞면({front_orientation_text}), 뒷면({back_orientation_text}) ===")
            
            # 1. 카드 삽입
            self.inject_card()
            
            # 2. 리본 옵션 설정
            self.set_ribbon_option(ribbon_type=1, key=0, value="2")
            
            # 3. 앞면 캔버스 준비 - 개별 방향 적용
            front_width, front_height = self.get_card_dimensions(front_orientation)
            if print_mode == "layered":
                front_img_info = self.prepare_front_canvas(
                    front_image_path, front_watermark_path, front_width, front_height, front_orientation
                )
            else:
                front_img_info = self.prepare_front_canvas(
                    front_image_path, None, front_width, front_height, front_orientation
                )
            
            # 4. 뒷면 캔버스 준비 - 개별 방향 적용
            back_width, back_height = self.get_card_dimensions(back_orientation)
            if print_mode == "layered":
                back_img_info = self.prepare_back_canvas(
                    back_image_path, back_watermark_path, back_width, back_height, back_orientation
                )
            else:
                back_img_info = self.prepare_back_canvas(
                    back_image_path, None, back_width, back_height, back_orientation
                )
            
            # 5. 양면 인쇄 실행
            print(f"양면 인쇄 실행 중: 앞면({front_orientation_text}), 뒷면({back_orientation_text})")
            ret = self.lib.R600PrintDraw(
                front_img_info.encode('cp949') if front_img_info else ctypes.c_char_p(None),
                back_img_info.encode('cp949') if back_img_info else ctypes.c_char_p(None)
            )
            self._check_result(ret, f"양면 인쇄 실행")
            
            # 6. 인쇄 완료 대기
            time.sleep(1)
            
            # 7. 카드 배출
            self.eject_card()
            
            # 8. 배출 완료 대기
            time.sleep(1)
            
            print(f"=== 양면 카드 인쇄 완료: 앞면({front_orientation_text}), 뒷면({back_orientation_text}) ===")
            
        except R600PrinterError as e:
            print(f"양면 카드 인쇄 중 오류 발생: {e}")
            # 오류 발생 시에도 카드 배출 시도
            try:
                self.eject_card()
            except:
                pass
            raise
        
    def print_single_side_card(self, image_path: str, watermark_path: Optional[str] = None,
                             card_orientation: str = "portrait",  # 개별 면 방향 추가
                             print_mode: str = "normal"):
        """단면 카드 인쇄 - 개별 면 방향 지원"""
        try:
            orientation_text = "세로형" if card_orientation == "portrait" else "가로형"
            print(f"=== {orientation_text} 단면 카드 인쇄 시작 ===")
            
            # 1. 카드 삽입
            self.inject_card()
            
            # 2. 리본 옵션 설정
            self.set_ribbon_option(ribbon_type=1, key=0, value="2")
            
            # 3. 캔버스 준비 - 개별 방향 적용
            card_width, card_height = self.get_card_dimensions(card_orientation)
            if print_mode == "layered":
                img_info = self.prepare_front_canvas(
                    image_path, watermark_path, card_width, card_height, card_orientation
                )
            else:
                img_info = self.prepare_front_canvas(
                    image_path, None, card_width, card_height, card_orientation
                )
            
            # 4. 단면 인쇄 실행 (뒷면은 None)
            ret = self.lib.R600PrintDraw(
                img_info.encode('cp949'),
                ctypes.c_char_p(None)
            )
            self._check_result(ret, f"{orientation_text} 단면 인쇄 실행")
            
            # 5. 인쇄 완료 대기
            time.sleep(1)
            
            # 6. 카드 배출
            self.eject_card()
            
            # 7. 배출 완료 대기
            time.sleep(1)
            
            print(f"=== {orientation_text} 단면 카드 인쇄 완료 ===")
            
        except R600PrinterError as e:
            print(f"{orientation_text} 단면 카드 인쇄 중 오류 발생: {e}")
            # 오류 발생 시에도 카드 배출 시도
            try:
                self.eject_card()
            except:
                pass
            raise
        
    # 하위 호환성을 위한 기존 메서드들 - 개별 면 방향 지원
    def print_normal_card(self, image_path: str, card_orientation: str = "portrait"):
        """일반 카드 인쇄 (하위 호환성) - 개별 면 방향 지원"""
        self.print_single_side_card(image_path, None, card_orientation, "normal")

    def print_layered_card(self, watermark_path: str, image_path: Optional[str] = None,
                          card_orientation: str = "portrait"):
        """레이어 카드 인쇄 (하위 호환성) - 개별 면 방향 지원"""
        self.print_single_side_card(image_path or "", watermark_path, card_orientation, "layered")

    def cleanup_and_close(self):
        """강화된 리소스 정리 - 회전된 마스킹 임시 파일 포함"""
        try:
            if not self.is_initialized:
                return
                
            print("[DEBUG] 프린터 리소스 정리 시작...")
            
            # 1. 카드 상태 확인 및 정리
            try:
                self.eject_card()
                print("[DEBUG] 카드 배출 완료")
            except:
                print("[DEBUG] 카드 배출 건너뜀 (정상)")
            
            # 2. 임시 파일 정리 (회전된 마스킹 파일 포함)
            try:
                import tempfile
                import glob
                temp_dir = tempfile.gettempdir()
                temp_files = glob.glob(os.path.join(temp_dir, "temp_watermark_*.jpg"))
                temp_files.extend(glob.glob(os.path.join(temp_dir, "temp_image_*.jpg")))
                temp_files.extend(glob.glob(os.path.join(temp_dir, "rotated_mask_*.jpg")))  # 🎯 추가
                
                for temp_file in temp_files:
                    try:
                        os.remove(temp_file)
                        print(f"[DEBUG] 임시 파일 삭제: {temp_file}")
                    except:
                        pass
            except Exception as e:
                print(f"[DEBUG] 임시 파일 정리 오류: {e}")
            
            # 3. 라이브러리 정리
            if self.lib:
                try:
                    ret = self.lib.R600LibClear()
                    print(f"[DEBUG] 라이브러리 정리 결과: {ret}")
                except:
                    print("[DEBUG] 라이브러리 정리 함수 없음 (정상)")
            
            # 4. 상태 초기화
            self.selected_printer_name = None
            self.front_img_info = None
            self.back_img_info = None
            self.is_initialized = False
            
            print("[DEBUG] 프린터 리소스 정리 완료")
            
        except Exception as e:
            print(f"[DEBUG] 리소스 정리 중 오류: {e}")
    
    def close(self):
        """기존 close 메서드 - cleanup_and_close 호출"""
        self.cleanup_and_close()

    def __enter__(self):
        """컨텍스트 매니저 진입"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 매니저 종료"""
        self.close()