"""
라이선스 인증 다이얼로그
- 최초 실행: 라이선스 키 입력 폼 표시
- 이후 실행: 저장된 키로 자동 인증
"""

import json
import os
import sys
from typing import Optional
from dataclasses import dataclass

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QFrame, QWidget
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont

from .manager import verify_license, LicenseResult


def _get_app_dir() -> str:
    """exe 폴더 (빌드) 또는 프로젝트 루트 (개발)"""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# 라이선스 파일 경로 (exe 옆 / 프로젝트 루트)
LICENSE_FILE = os.path.join(_get_app_dir(), "license.json")

# 이전 버전 경로 (빌드 시 _internal 폴더 안에 저장되던 위치)
LEGACY_LICENSE_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "license.json")

# 키 자체의 문제가 아닌 실패 → 저장된 키 유지
KEEP_KEY_CODES = {'CONNECTION_ERROR', 'DEVICE_ERROR'}


def _is_same_path(a: str, b: str) -> bool:
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def load_license_key() -> Optional[str]:
    """저장된 라이선스 키 로드 (새 경로 → 이전 경로 순)"""
    for path in (LICENSE_FILE, LEGACY_LICENSE_FILE):
        try:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    key = json.load(f).get('license_key')
                if key:
                    return key
        except Exception as e:
            print(f"[LICENSE] 라이선스 파일 읽기 실패 ({path}): {type(e).__name__}: {e}")
    return None


def save_license_key(license_key: str) -> bool:
    """라이선스 키 저장 (이전 경로 파일은 정리)"""
    try:
        with open(LICENSE_FILE, 'w', encoding='utf-8') as f:
            json.dump({'license_key': license_key}, f, indent=2)
    except Exception as e:
        print(f"[LICENSE] 라이선스 파일 저장 실패 ({LICENSE_FILE}): {type(e).__name__}: {e}")
        return False

    if not _is_same_path(LICENSE_FILE, LEGACY_LICENSE_FILE) and os.path.exists(LEGACY_LICENSE_FILE):
        try:
            os.remove(LEGACY_LICENSE_FILE)
        except Exception:
            pass
    return True


def clear_license_key() -> bool:
    """저장된 라이선스 키 삭제 (새 경로 + 이전 경로)"""
    success = True
    for path in {LICENSE_FILE, LEGACY_LICENSE_FILE}:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            success = False
    return success


class VerifyThread(QThread):
    """라이선스 검증 스레드"""
    finished = Signal(LicenseResult)

    def __init__(self, license_key: str):
        super().__init__()
        self.license_key = license_key

    def run(self):
        result = verify_license(self.license_key)
        self.finished.emit(result)


class LicenseDialog(QDialog):
    """라이선스 인증 다이얼로그"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.verify_thread: Optional[VerifyThread] = None
        self.is_verified = False
        self.license_result: Optional[LicenseResult] = None

        self._setup_dialog()
        self._setup_ui()

    def _setup_dialog(self):
        """다이얼로그 설정"""
        self.setWindowTitle("라이선스 인증 - Hana Studio")
        self.setModal(True)
        self.setFixedSize(400, 280)
        self.setStyleSheet("""
            QDialog {
                background-color: #F9FAFB;
                font-family: 'Segoe UI', system-ui, sans-serif;
            }
        """)

    def _setup_ui(self):
        """UI 구성"""
        layout = QVBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        # 헤더
        self._create_header(layout)

        # 컨텐츠
        content = QWidget()
        content.setStyleSheet("background-color: #FFFFFF;")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(24, 20, 24, 20)
        content_layout.setSpacing(16)

        # 입력 필드
        self._create_input_field(content_layout)

        # 상태 메시지
        self._create_status_section(content_layout)

        content_layout.addStretch()
        layout.addWidget(content, 1)

        # 푸터
        self._create_footer(layout)

    def _create_header(self, parent_layout):
        """헤더"""
        header = QFrame()
        header.setFixedHeight(70)
        header.setStyleSheet("QFrame { background-color: #FFFFFF; }")

        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(24, 16, 24, 12)
        header_layout.setSpacing(4)

        title = QLabel("라이선스 인증")
        title.setFont(QFont("Segoe UI", 16, QFont.Weight.DemiBold))
        title.setStyleSheet("color: #111827; background: transparent;")

        subtitle = QLabel("라이선스 키를 입력해주세요")
        subtitle.setFont(QFont("Segoe UI", 10))
        subtitle.setStyleSheet("color: #6B7280; background: transparent;")

        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        parent_layout.addWidget(header)

    def _create_input_field(self, parent_layout):
        """입력 필드"""
        label = QLabel("라이선스 키")
        label.setFont(QFont("Segoe UI", 10, QFont.Weight.Medium))
        label.setStyleSheet("color: #374151; background: transparent;")

        self.key_input = QLineEdit()
        self.key_input.setPlaceholderText("XXXX-XXXX-XXXX-XXXX")
        self.key_input.setFixedHeight(40)
        self.key_input.setFont(QFont("Consolas", 11))
        self.key_input.setStyleSheet("""
            QLineEdit {
                background-color: #FFFFFF;
                border: 1px solid #D1D5DB;
                border-radius: 6px;
                padding: 0 12px;
                color: #1F2937;
            }
            QLineEdit:focus {
                border-color: #3B82F6;
            }
            QLineEdit:disabled {
                background-color: #F3F4F6;
                color: #9CA3AF;
            }
        """)
        self.key_input.returnPressed.connect(self._on_verify_clicked)

        parent_layout.addWidget(label)
        parent_layout.addWidget(self.key_input)

    def _create_status_section(self, parent_layout):
        """상태 메시지 섹션"""
        self.status_frame = QFrame()
        self.status_frame.setFixedHeight(36)
        self.status_frame.setVisible(False)

        status_layout = QHBoxLayout(self.status_frame)
        status_layout.setContentsMargins(12, 8, 12, 8)

        self.status_icon = QLabel()
        self.status_icon.setFont(QFont("Segoe UI", 10))
        self.status_icon.setStyleSheet("background: transparent;")

        self.status_text = QLabel()
        self.status_text.setFont(QFont("Segoe UI", 9))
        self.status_text.setStyleSheet("background: transparent;")

        status_layout.addWidget(self.status_icon)
        status_layout.addWidget(self.status_text)
        status_layout.addStretch()

        parent_layout.addWidget(self.status_frame)

    def _create_footer(self, parent_layout):
        """푸터"""
        footer = QFrame()
        footer.setFixedHeight(64)
        footer.setStyleSheet("QFrame { background-color: #F9FAFB; }")

        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(24, 16, 24, 16)
        footer_layout.setSpacing(12)

        footer_layout.addStretch()

        # 취소 버튼
        self.cancel_btn = QPushButton("취소")
        self.cancel_btn.setFixedSize(70, 32)
        self.cancel_btn.setFont(QFont("Segoe UI", 9))
        self.cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #FFFFFF;
                border: 1px solid #D1D5DB;
                border-radius: 6px;
                color: #374151;
            }
            QPushButton:hover {
                background-color: #F9FAFB;
            }
            QPushButton:pressed {
                background-color: #F3F4F6;
            }
        """)
        self.cancel_btn.clicked.connect(self.reject)

        # 인증 버튼
        self.verify_btn = QPushButton("인증")
        self.verify_btn.setFixedSize(70, 32)
        self.verify_btn.setFont(QFont("Segoe UI", 9, QFont.Weight.Medium))
        self.verify_btn.setStyleSheet("""
            QPushButton {
                background-color: #3B82F6;
                border: none;
                border-radius: 6px;
                color: #FFFFFF;
            }
            QPushButton:hover {
                background-color: #2563EB;
            }
            QPushButton:pressed {
                background-color: #1D4ED8;
            }
            QPushButton:disabled {
                background-color: #93C5FD;
            }
        """)
        self.verify_btn.clicked.connect(self._on_verify_clicked)

        footer_layout.addWidget(self.cancel_btn)
        footer_layout.addWidget(self.verify_btn)
        parent_layout.addWidget(footer)

    def _show_status(self, success: bool, message: str):
        """상태 메시지 표시"""
        self.status_frame.setVisible(True)

        if success:
            self.status_frame.setStyleSheet("""
                QFrame {
                    background-color: #ECFDF5;
                    border-radius: 6px;
                }
            """)
            self.status_icon.setText("✓")
            self.status_icon.setStyleSheet("color: #10B981; background: transparent;")
            self.status_text.setStyleSheet("color: #065F46; background: transparent;")
        else:
            self.status_frame.setStyleSheet("""
                QFrame {
                    background-color: #FEF2F2;
                    border-radius: 6px;
                }
            """)
            self.status_icon.setText("✗")
            self.status_icon.setStyleSheet("color: #EF4444; background: transparent;")
            self.status_text.setStyleSheet("color: #991B1B; background: transparent;")

        self.status_text.setText(message)

    def _set_loading(self, loading: bool):
        """로딩 상태 설정"""
        self.key_input.setEnabled(not loading)
        self.verify_btn.setEnabled(not loading)
        self.cancel_btn.setEnabled(not loading)

        if loading:
            self.verify_btn.setText("인증 중...")
        else:
            self.verify_btn.setText("인증")

    def _on_verify_clicked(self):
        """인증 버튼 클릭"""
        license_key = self.key_input.text().strip()

        if not license_key:
            self._show_status(False, "라이선스 키를 입력해주세요")
            return

        self._set_loading(True)
        self.status_frame.setVisible(False)

        # 백그라운드 스레드에서 인증
        self.verify_thread = VerifyThread(license_key)
        self.verify_thread.finished.connect(self._on_verify_finished)
        self.verify_thread.start()

    def _on_verify_finished(self, result: LicenseResult):
        """인증 완료"""
        self._set_loading(False)
        self._show_status(result.success, result.message)
        self.license_result = result

        if result.success:
            # 키 저장
            save_license_key(self.key_input.text().strip())
            self.is_verified = True

            # 1초 후 다이얼로그 닫기
            from PySide6.QtCore import QTimer
            QTimer.singleShot(1000, self.accept)

    def closeEvent(self, event):
        """다이얼로그 닫기"""
        if self.verify_thread and self.verify_thread.isRunning():
            self.verify_thread.quit()
            self.verify_thread.wait()
        event.accept()


def check_license(parent=None) -> tuple:
    """
    라이선스 확인 (앱 시작 시 호출)

    Returns:
        tuple: (success: bool, is_admin: bool)
            - success: 인증 성공 여부
            - is_admin: Admin 권한 여부
    """
    # 저장된 키 확인
    saved_key = load_license_key()
    failed_result: Optional[LicenseResult] = None

    if saved_key:
        # 자동 인증 시도
        result = verify_license(saved_key)
        if result.success:
            # 이전 경로(_internal)에 있던 키는 새 경로로 이동
            if not os.path.exists(LICENSE_FILE):
                save_license_key(saved_key)
            return (True, result.is_admin)

        print(f"[LICENSE] 자동 인증 실패: {result.code}")
        failed_result = result
        # 키 자체가 무효한 경우에만 삭제 (네트워크/디바이스 오류는 유지)
        if result.code not in KEEP_KEY_CODES:
            clear_license_key()

    # 다이얼로그 표시
    dialog = LicenseDialog(parent)
    if failed_result:
        # 실패 사유 표시, 유지된 키는 미리 채워서 재시도만 누르면 되도록
        if failed_result.code in KEEP_KEY_CODES:
            dialog.key_input.setText(saved_key)
        dialog._show_status(False, failed_result.message)
    if dialog.exec() == QDialog.DialogCode.Accepted and dialog.is_verified:
        return (True, dialog.license_result.is_admin if dialog.license_result else False)

    return (False, False)
