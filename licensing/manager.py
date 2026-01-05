"""
License manager module.
Handles license verification through Supabase RPC.
"""

from typing import Optional
from dataclasses import dataclass

import httpx

from .config import SUPABASE_URL, SUPABASE_KEY
from .device import get_device_hash


@dataclass
class LicenseResult:
    """라이선스 검증 결과"""
    success: bool
    message: str
    error: Optional[str] = None


class LicenseManager:
    """라이선스 관리 클래스"""

    def __init__(self):
        self._rpc_url = f"{SUPABASE_URL}/rest/v1/rpc/verify_license"
        self._headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json",
        }

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

            # HTTP로 직접 RPC 호출
            with httpx.Client(timeout=30.0) as client:
                response = client.post(
                    self._rpc_url,
                    headers=self._headers,
                    json={
                        "p_license_key": license_key,
                        "p_device_hash": device_hash
                    }
                )
                response.raise_for_status()
                data = response.json()

            if data.get('success'):
                return LicenseResult(
                    success=True,
                    message=data.get('message', '인증 성공')
                )
            else:
                return LicenseResult(
                    success=False,
                    message=data.get('message', '인증 실패'),
                    error=data.get('error')
                )

        except RuntimeError as e:
            # 디바이스 식별자 수집 실패
            return LicenseResult(
                success=False,
                message='디바이스 식별 실패',
                error=str(e)
            )

        except httpx.HTTPStatusError as e:
            # HTTP 에러 (4xx, 5xx)
            return LicenseResult(
                success=False,
                message='서버 오류',
                error=f"HTTP {e.response.status_code}"
            )

        except Exception as e:
            # 네트워크 오류 등
            return LicenseResult(
                success=False,
                message='서버 연결 실패',
                error=str(e)
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
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")
    if result.error:
        print(f"Error: {result.error}")
