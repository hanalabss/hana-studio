# Licensing module

from .manager import verify_license, LicenseResult, LicenseManager
from .dialog import check_license, LicenseDialog
from .admin import generate_license_key, generate_admin_key, generate_user_key
from .admin_panel import AdminPanel

__all__ = [
    'verify_license',
    'LicenseResult',
    'LicenseManager',
    'check_license',
    'LicenseDialog',
    'generate_license_key',
    'generate_admin_key',
    'generate_user_key',
    'AdminPanel',
]
