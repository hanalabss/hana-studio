"""
Admin Panel - 라이선스 관리 UI
Admin 키로 인증 시 접근 가능
"""

from typing import Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QWidget, QTableWidget,
    QTableWidgetItem, QHeaderView, QLineEdit,
    QSpinBox, QCheckBox, QMessageBox, QMenu
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont, QCursor

from supabase import create_client
from .config import SUPABASE_URL, SUPABASE_KEY
from .admin import generate_user_key, generate_admin_key


@dataclass
class LicenseInfo:
    """라이선스 정보"""
    license_key: str
    status: str
    is_admin: bool
    device_hash: Optional[str]
    expires_at: Optional[str]
    memo: str
    created_at: str


class LoadLicensesThread(QThread):
    """라이선스 목록 로드 스레드"""
    finished = Signal(list)
    error = Signal(str)

    def run(self):
        try:
            client = create_client(SUPABASE_URL, SUPABASE_KEY)
            result = client.table('licenses').select('*').order('created_at', desc=True).execute()

            licenses = []
            for row in result.data:
                licenses.append(LicenseInfo(
                    license_key=row.get('license_key', ''),
                    status=row.get('status', 'active'),
                    is_admin=row.get('is_admin', False),
                    device_hash=row.get('device_hash'),
                    expires_at=row.get('expires_at'),
                    memo=row.get('memo', ''),
                    created_at=row.get('created_at', '')
                ))

            self.finished.emit(licenses)
        except Exception as e:
            self.error.emit(str(e))


class CreateLicenseThread(QThread):
    """라이선스 생성 스레드"""
    finished = Signal(str)  # 생성된 키
    error = Signal(str)

    def __init__(self, is_admin: bool, days: int, memo: str):
        super().__init__()
        self.is_admin = is_admin
        self.days = days
        self.memo = memo

    def run(self):
        try:
            client = create_client(SUPABASE_URL, SUPABASE_KEY)

            # 키 생성
            if self.is_admin:
                license_key = generate_admin_key()
            else:
                license_key = generate_user_key()

            # 만료일 계산
            expires_at = None
            if self.days > 0 and not self.is_admin:
                expires_at = (datetime.now() + timedelta(days=self.days)).isoformat()

            # DB 등록
            client.table('licenses').insert({
                'license_key': license_key,
                'is_admin': self.is_admin,
                'expires_at': expires_at,
                'memo': self.memo,
                'status': 'active'
            }).execute()

            self.finished.emit(license_key)
        except Exception as e:
            self.error.emit(str(e))


class UpdateStatusThread(QThread):
    """상태 업데이트 스레드"""
    finished = Signal()
    error = Signal(str)

    def __init__(self, license_key: str, new_status: str):
        super().__init__()
        self.license_key = license_key
        self.new_status = new_status

    def run(self):
        try:
            client = create_client(SUPABASE_URL, SUPABASE_KEY)
            client.table('licenses').update({
                'status': self.new_status
            }).eq('license_key', self.license_key).execute()

            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))


class AdminPanel(QDialog):
    """Admin 라이선스 관리 패널"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.licenses: List[LicenseInfo] = []
        self.load_thread: Optional[LoadLicensesThread] = None
        self.create_thread: Optional[CreateLicenseThread] = None
        self.update_thread: Optional[UpdateStatusThread] = None

        self._setup_dialog()
        self._setup_ui()
        self._load_licenses()

    def _setup_dialog(self):
        """다이얼로그 설정"""
        self.setWindowTitle("라이선스 관리 - Admin Panel")
        self.setModal(True)
        self.setMinimumSize(800, 500)
        self.resize(900, 600)
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

        # 툴바
        self._create_toolbar(layout)

        # 테이블
        self._create_table(layout)

        # 푸터
        self._create_footer(layout)

    def _create_header(self, parent_layout):
        """헤더"""
        header = QFrame()
        header.setFixedHeight(70)
        header.setStyleSheet("QFrame { background-color: #FFFFFF; border-bottom: 1px solid #E5E7EB; }")

        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(24, 16, 24, 12)
        header_layout.setSpacing(4)

        title = QLabel("라이선스 관리")
        title.setFont(QFont("Segoe UI", 16, QFont.Weight.DemiBold))
        title.setStyleSheet("color: #111827; background: transparent; border: none;")

        subtitle = QLabel("등록된 라이선스를 관리합니다")
        subtitle.setFont(QFont("Segoe UI", 10))
        subtitle.setStyleSheet("color: #6B7280; background: transparent; border: none;")

        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        parent_layout.addWidget(header)

    def _create_toolbar(self, parent_layout):
        """툴바"""
        toolbar = QFrame()
        toolbar.setFixedHeight(60)
        toolbar.setStyleSheet("QFrame { background-color: #FFFFFF; border-bottom: 1px solid #E5E7EB; }")

        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(24, 12, 24, 12)
        toolbar_layout.setSpacing(12)

        # 새 라이선스 버튼
        self.create_btn = QPushButton("+ 새 라이선스")
        self.create_btn.setFixedHeight(36)
        self.create_btn.setFont(QFont("Segoe UI", 9, QFont.Weight.Medium))
        self.create_btn.setStyleSheet("""
            QPushButton {
                background-color: #3B82F6;
                border: none;
                border-radius: 6px;
                color: #FFFFFF;
                padding: 0 16px;
            }
            QPushButton:hover { background-color: #2563EB; }
            QPushButton:pressed { background-color: #1D4ED8; }
            QPushButton:disabled { background-color: #93C5FD; }
        """)
        self.create_btn.clicked.connect(self._show_create_dialog)

        # 새로고침 버튼
        self.refresh_btn = QPushButton("새로고침")
        self.refresh_btn.setFixedHeight(36)
        self.refresh_btn.setFont(QFont("Segoe UI", 9))
        self.refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #FFFFFF;
                border: 1px solid #D1D5DB;
                border-radius: 6px;
                color: #374151;
                padding: 0 16px;
            }
            QPushButton:hover { background-color: #F9FAFB; }
            QPushButton:pressed { background-color: #F3F4F6; }
        """)
        self.refresh_btn.clicked.connect(self._load_licenses)

        # 상태 레이블
        self.status_label = QLabel()
        self.status_label.setFont(QFont("Segoe UI", 9))
        self.status_label.setStyleSheet("color: #6B7280; background: transparent; border: none;")

        toolbar_layout.addWidget(self.create_btn)
        toolbar_layout.addWidget(self.refresh_btn)
        toolbar_layout.addStretch()
        toolbar_layout.addWidget(self.status_label)

        parent_layout.addWidget(toolbar)

    def _create_table(self, parent_layout):
        """테이블"""
        table_container = QWidget()
        table_container.setStyleSheet("background-color: #FFFFFF;")
        table_layout = QVBoxLayout(table_container)
        table_layout.setContentsMargins(24, 16, 24, 16)

        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            "라이선스 키", "타입", "상태", "만료일", "디바이스", "메모"
        ])

        # 테이블 스타일
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: #FFFFFF;
                border: 1px solid #E5E7EB;
                border-radius: 8px;
                gridline-color: #F3F4F6;
            }
            QTableWidget::item {
                padding: 8px;
                border-bottom: 1px solid #F3F4F6;
            }
            QTableWidget::item:selected {
                background-color: #EFF6FF;
                color: #1E40AF;
            }
            QHeaderView::section {
                background-color: #F9FAFB;
                color: #374151;
                font-weight: 600;
                padding: 10px 8px;
                border: none;
                border-bottom: 1px solid #E5E7EB;
            }
        """)

        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._show_context_menu)

        # 컬럼 너비
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)

        self.table.setColumnWidth(0, 180)  # 키
        self.table.setColumnWidth(1, 70)   # 타입
        self.table.setColumnWidth(2, 70)   # 상태
        self.table.setColumnWidth(3, 100)  # 만료일
        self.table.setColumnWidth(4, 80)   # 디바이스

        table_layout.addWidget(self.table)
        parent_layout.addWidget(table_container, 1)

    def _create_footer(self, parent_layout):
        """푸터"""
        footer = QFrame()
        footer.setFixedHeight(56)
        footer.setStyleSheet("QFrame { background-color: #F9FAFB; border-top: 1px solid #E5E7EB; }")

        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(24, 12, 24, 12)

        hint_label = QLabel("우클릭으로 라이선스 상태 변경")
        hint_label.setFont(QFont("Segoe UI", 9))
        hint_label.setStyleSheet("color: #9CA3AF; background: transparent; border: none;")

        close_btn = QPushButton("닫기")
        close_btn.setFixedSize(80, 32)
        close_btn.setFont(QFont("Segoe UI", 9))
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #FFFFFF;
                border: 1px solid #D1D5DB;
                border-radius: 6px;
                color: #374151;
            }
            QPushButton:hover { background-color: #F9FAFB; }
        """)
        close_btn.clicked.connect(self.close)

        footer_layout.addWidget(hint_label)
        footer_layout.addStretch()
        footer_layout.addWidget(close_btn)

        parent_layout.addWidget(footer)

    def _load_licenses(self):
        """라이선스 목록 로드"""
        self.status_label.setText("로딩 중...")
        self.refresh_btn.setEnabled(False)

        self.load_thread = LoadLicensesThread()
        self.load_thread.finished.connect(self._on_licenses_loaded)
        self.load_thread.error.connect(self._on_load_error)
        self.load_thread.start()

    def _on_licenses_loaded(self, licenses: List[LicenseInfo]):
        """로드 완료"""
        self.licenses = licenses
        self._update_table()
        self.status_label.setText(f"총 {len(licenses)}개")
        self.refresh_btn.setEnabled(True)

    def _on_load_error(self, error: str):
        """로드 에러"""
        self.status_label.setText(f"오류: {error}")
        self.refresh_btn.setEnabled(True)

    def _update_table(self):
        """테이블 업데이트"""
        self.table.setRowCount(len(self.licenses))

        for row, lic in enumerate(self.licenses):
            # 키
            key_item = QTableWidgetItem(lic.license_key)
            key_item.setFont(QFont("Consolas", 9))
            self.table.setItem(row, 0, key_item)

            # 타입
            type_text = "Admin" if lic.is_admin else "User"
            type_item = QTableWidgetItem(type_text)
            type_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if lic.is_admin:
                type_item.setForeground(Qt.GlobalColor.darkMagenta)
            self.table.setItem(row, 1, type_item)

            # 상태
            status_map = {'active': '✅ 활성', 'revoked': '❌ 정지'}
            status_text = status_map.get(lic.status, lic.status)
            status_item = QTableWidgetItem(status_text)
            status_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 2, status_item)

            # 만료일
            if lic.expires_at:
                try:
                    exp_date = datetime.fromisoformat(lic.expires_at.replace('Z', '+00:00'))
                    expires_text = exp_date.strftime('%Y-%m-%d')
                    if exp_date < datetime.now(exp_date.tzinfo):
                        expires_text = f"⏳ {expires_text}"
                except:
                    expires_text = lic.expires_at[:10]
            else:
                expires_text = "무제한"
            expires_item = QTableWidgetItem(expires_text)
            expires_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 3, expires_item)

            # 디바이스
            device_text = "등록됨" if lic.device_hash else "-"
            device_item = QTableWidgetItem(device_text)
            device_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 4, device_item)

            # 메모
            memo_item = QTableWidgetItem(lic.memo or "")
            self.table.setItem(row, 5, memo_item)

    def _show_context_menu(self, pos):
        """컨텍스트 메뉴"""
        row = self.table.rowAt(pos.y())
        if row < 0 or row >= len(self.licenses):
            return

        lic = self.licenses[row]
        menu = QMenu(self)

        # 키 복사
        copy_action = menu.addAction("키 복사")
        copy_action.triggered.connect(lambda: self._copy_key(lic.license_key))

        menu.addSeparator()

        # 상태 변경
        if lic.status == 'active':
            revoke_action = menu.addAction("❌ 정지")
            revoke_action.triggered.connect(lambda: self._update_status(lic.license_key, 'revoked'))
        else:
            activate_action = menu.addAction("✅ 활성화")
            activate_action.triggered.connect(lambda: self._update_status(lic.license_key, 'active'))

        menu.exec(QCursor.pos())

    def _copy_key(self, key: str):
        """키 복사"""
        from PySide6.QtWidgets import QApplication
        QApplication.clipboard().setText(key)
        self.status_label.setText("키가 복사되었습니다")

    def _update_status(self, license_key: str, new_status: str):
        """상태 업데이트"""
        self.status_label.setText("업데이트 중...")

        self.update_thread = UpdateStatusThread(license_key, new_status)
        self.update_thread.finished.connect(self._on_status_updated)
        self.update_thread.error.connect(self._on_update_error)
        self.update_thread.start()

    def _on_status_updated(self):
        """상태 업데이트 완료"""
        self._load_licenses()

    def _on_update_error(self, error: str):
        """업데이트 에러"""
        QMessageBox.warning(self, "오류", f"상태 변경 실패: {error}")
        self.status_label.setText("업데이트 실패")

    def _show_create_dialog(self):
        """라이선스 생성 다이얼로그"""
        dialog = CreateLicenseDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._create_license(
                is_admin=dialog.is_admin,
                days=dialog.days,
                memo=dialog.memo
            )

    def _create_license(self, is_admin: bool, days: int, memo: str):
        """라이선스 생성"""
        self.status_label.setText("생성 중...")
        self.create_btn.setEnabled(False)

        self.create_thread = CreateLicenseThread(is_admin, days, memo)
        self.create_thread.finished.connect(self._on_license_created)
        self.create_thread.error.connect(self._on_create_error)
        self.create_thread.start()

    def _on_license_created(self, license_key: str):
        """생성 완료"""
        self.create_btn.setEnabled(True)

        # 키 복사
        from PySide6.QtWidgets import QApplication
        QApplication.clipboard().setText(license_key)

        QMessageBox.information(
            self,
            "라이선스 생성 완료",
            f"새 라이선스가 생성되었습니다.\n\n{license_key}\n\n(클립보드에 복사됨)"
        )

        self._load_licenses()

    def _on_create_error(self, error: str):
        """생성 에러"""
        self.create_btn.setEnabled(True)
        QMessageBox.warning(self, "오류", f"라이선스 생성 실패: {error}")


class CreateLicenseDialog(QDialog):
    """라이선스 생성 다이얼로그"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_admin = False
        self.days = 365
        self.memo = ""

        self._setup_ui()

    def _setup_ui(self):
        self.setWindowTitle("새 라이선스 생성")
        self.setFixedSize(350, 280)
        self.setStyleSheet("""
            QDialog {
                background-color: #FFFFFF;
                font-family: 'Segoe UI', system-ui, sans-serif;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        # Admin 체크박스
        self.admin_check = QCheckBox("Admin 라이선스")
        self.admin_check.setFont(QFont("Segoe UI", 10))
        self.admin_check.toggled.connect(self._on_admin_toggled)

        # 만료일
        days_layout = QHBoxLayout()
        days_label = QLabel("유효 기간:")
        days_label.setFont(QFont("Segoe UI", 10))
        self.days_spin = QSpinBox()
        self.days_spin.setRange(0, 3650)
        self.days_spin.setValue(365)
        self.days_spin.setSuffix(" 일")
        self.days_spin.setSpecialValueText("무제한")
        self.days_spin.setFixedWidth(100)
        days_layout.addWidget(days_label)
        days_layout.addWidget(self.days_spin)
        days_layout.addStretch()

        # 메모
        memo_label = QLabel("메모:")
        memo_label.setFont(QFont("Segoe UI", 10))
        self.memo_input = QLineEdit()
        self.memo_input.setPlaceholderText("회사명, 담당자 등")
        self.memo_input.setFixedHeight(36)
        self.memo_input.setStyleSheet("""
            QLineEdit {
                border: 1px solid #D1D5DB;
                border-radius: 6px;
                padding: 0 12px;
            }
            QLineEdit:focus { border-color: #3B82F6; }
        """)

        # 버튼
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)

        cancel_btn = QPushButton("취소")
        cancel_btn.setFixedSize(80, 36)
        cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #FFFFFF;
                border: 1px solid #D1D5DB;
                border-radius: 6px;
                color: #374151;
            }
            QPushButton:hover { background-color: #F9FAFB; }
        """)
        cancel_btn.clicked.connect(self.reject)

        create_btn = QPushButton("생성")
        create_btn.setFixedSize(80, 36)
        create_btn.setStyleSheet("""
            QPushButton {
                background-color: #3B82F6;
                border: none;
                border-radius: 6px;
                color: #FFFFFF;
            }
            QPushButton:hover { background-color: #2563EB; }
        """)
        create_btn.clicked.connect(self._on_create)

        btn_layout.addStretch()
        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(create_btn)

        layout.addWidget(self.admin_check)
        layout.addLayout(days_layout)
        layout.addWidget(memo_label)
        layout.addWidget(self.memo_input)
        layout.addStretch()
        layout.addLayout(btn_layout)

    def _on_admin_toggled(self, checked: bool):
        """Admin 체크 토글"""
        self.days_spin.setEnabled(not checked)
        if checked:
            self.days_spin.setValue(0)  # Admin은 무제한

    def _on_create(self):
        """생성 버튼"""
        self.is_admin = self.admin_check.isChecked()
        self.days = self.days_spin.value()
        self.memo = self.memo_input.text().strip()
        self.accept()
