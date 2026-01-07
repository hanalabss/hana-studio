"""
License manager module.
Handles license verification through Supabase RPC.
"""

from typing import Optional
from dataclasses import dataclass

from supabase import create_client, Client

from .config import SUPABASE_URL, SUPABASE_KEY
from .device import get_device_hash


# 서버 응답 코드 → 한글 메시지 매핑
MESSAGES = {
    'ACTIVATED': '라이선스 활성화 완료',
    'VERIFIED': '인증 성공',
    'INVALID_KEY': '유효하지 않은 라이선스 키',
    'REVOKED': '정지된 라이선스',
    'EXPIRED': '만료된 라이선스',
    'DEVICE_MISMATCH': '다른 PC에 등록된 라이선스입니다',
}


@dataclass
class LicenseResult:
    """라이선스 검증 결과"""
    success: bool
    message: str
    code: Optional[str] = None


class LicenseManager:
    """라이선스 관리 클래스"""

    def __init__(self):
        self._client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    def verify_license(self, license_key: str) -> LicenseResult:
        """
        라이선스 검증

        1. license_key의 device_hash가 NULL이면 → 현재 device_hash 등록
        2. NULL이 아니면 → 현재 device_hash와 일치 여부 검증

        Args:
            license_key: 라이선스 키

        Returns:
            LicenseResult: 검증 결과
        """
        try:
            # 디바이스 해시 생성
            device_hash = get_device_hash()

            # Supabase RPC 호출
            result = self._client.rpc('verify_license', {
                'p_license_key': license_key,
                'p_device_hash': device_hash
            }).execute()

            data = result.data
            code = data.get('code')
            success = data.get('success', False)
            message = MESSAGES.get(code, '알 수 없는 오류')

            return LicenseResult(
                success=success,
                message=message,
                code=code
            )

        except RuntimeError as e:
            # 디바이스 식별자 수집 실패
            return LicenseResult(
                success=False,
                message='디바이스 식별 실패',
                code='DEVICE_ERROR'
            )

        except Exception as e:
            # 네트워크 오류 등
            return LicenseResult(
                success=False,
                message='서버 연결 실패',
                code='CONNECTION_ERROR'
            )


# 싱글톤 인스턴스
_manager: Optional[LicenseManager] = None


def get_license_manager() -> LicenseManager:
    """LicenseManager 싱글톤 인스턴스 반환"""
    global _manager
    if _manager is None:
        _manager = LicenseManager()
    return _manager


def verify_license(license_key: str) -> LicenseResult:
    """
    라이선스 검증 (편의 함수)

    Usage:
        from licensing.manager import verify_license

        result = verify_license('TEST-1234-5678-ABCD')
        if result.success:
            print(result.message)
        else:
            print(f"Error: {result.error}")
    """
    return get_license_manager().verify_license(license_key)


if __name__ == "__main__":
    # 테스트용
    test_key = input("License Key: ")
    result = verify_license(test_key)

    if result.success:
        print(f"✅ {result.message}")
    else:
        print(f"❌ {result.message}")

    print(f"Code: {result.code}")
