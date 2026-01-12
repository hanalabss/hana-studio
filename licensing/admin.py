"""
Admin utilities for license management.
- License key generation
- Admin-specific operations
"""

import secrets
import string
from datetime import datetime, timedelta
from typing import Optional


def generate_license_key(prefix: str = "HANA") -> str:
    """
    라이선스 키 생성

    형식: PREFIX-XXXX-XXXX-XXXX (12자리 영숫자, 총 19자)

    Args:
        prefix: 키 접두사 (HANA=일반, ADMIN=관리자)

    Returns:
        생성된 라이선스 키

    Example:
        >>> generate_license_key()
        'HANA-A1B2-C3D4-E5F6'
        >>> generate_license_key("ADMIN")
        'ADMIN-X7Y8-Z9A0-B1C2'
    """
    # 대문자 + 숫자 (혼동 방지: O, 0, I, 1 제외)
    chars = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"

    segments = [
        ''.join(secrets.choice(chars) for _ in range(4))
        for _ in range(3)
    ]

    return f"{prefix}-{'-'.join(segments)}"


def generate_admin_key() -> str:
    """Admin 전용 키 생성"""
    return generate_license_key(prefix="ADMIN")


def generate_user_key() -> str:
    """일반 사용자 키 생성"""
    return generate_license_key(prefix="HANA")


def calculate_expiry_date(days: int) -> Optional[str]:
    """
    만료일 계산

    Args:
        days: 오늘부터 N일 후 (0 = 무제한)

    Returns:
        ISO 형식 날짜 문자열 또는 None (무제한)
    """
    if days <= 0:
        return None

    expiry = datetime.now() + timedelta(days=days)
    return expiry.isoformat()


def validate_key_format(license_key: str) -> bool:
    """
    라이선스 키 형식 검증

    Args:
        license_key: 검증할 키

    Returns:
        True if valid format
    """
    parts = license_key.split('-')

    # 최소 4개 파트 (PREFIX-XXXX-XXXX-XXXX)
    if len(parts) != 4:
        return False

    # 첫 번째 파트는 PREFIX
    if parts[0] not in ('HANA', 'ADMIN'):
        return False

    # 나머지 파트는 4자리씩
    for part in parts[1:]:
        if len(part) != 4:
            return False
        if not part.isalnum():
            return False

    return True


def is_admin_key(license_key: str) -> bool:
    """
    Admin 키 여부 확인 (prefix 기반)

    Args:
        license_key: 확인할 키

    Returns:
        True if admin key
    """
    return license_key.startswith("ADMIN-")


if __name__ == "__main__":
    # 테스트
    print("=== License Key Generator ===")
    print(f"User Key:  {generate_user_key()}")
    print(f"Admin Key: {generate_admin_key()}")
    print()
    print(f"Expiry (30 days): {calculate_expiry_date(30)}")
    print(f"Expiry (365 days): {calculate_expiry_date(365)}")
    print(f"Expiry (unlimited): {calculate_expiry_date(0)}")
