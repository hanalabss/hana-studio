"""
Hana Studio v2 - RTAI 전용 카드 디자인 & 인쇄 툴
MVP (Minimum Viable Product)

캔버스 기반 디자인 툴로 완전히 재설계된 버전
- 드래그 앤 드롭 중심 UI
- 이미지/텍스트 레이어 시스템
- 600DPI 인쇄 지원 (다음 단계)
"""

import sys
import os
import tempfile
from pathlib import Path


# ============================================================
# 안전한 로깅 시스템 (PyInstaller --windowed 모드 대응)
# ============================================================

def _setup_safe_logging():
    """
    PyInstaller --windowed 모드에서 안전한 로깅 설정

    문제: --windowed 옵션으로 빌드하면 콘솔이 없어서
          sys.stdout/stderr가 None이 됨 → print() 크래시

    해결: 파일로 리다이렉트하여 디버깅 가능하게 유지
    """
    if sys.stdout is None or sys.stderr is None:
        log_path = os.path.join(os.getcwd(), "hana_studio_v2_debug.log")
        try:
            log_file = open(log_path, "w", encoding="utf-8", buffering=1)
            if sys.stdout is None:
                sys.stdout = log_file
            if sys.stderr is None:
                sys.stderr = log_file
        except Exception:
            # 파일 생성 실패 시 devnull로 폴백
            devnull = open(os.devnull, 'w', encoding='utf-8')
            if sys.stdout is None:
                sys.stdout = devnull
            if sys.stderr is None:
                sys.stderr = devnull


# 모듈 로드 시 즉시 실행
_setup_safe_logging()


# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent))

# 앱 고유 식별자
APP_MUTEX_NAME = "HanaStudioV2_SingleInstance_Mutex"
APP_LOCK_FILE = os.path.join(tempfile.gettempdir(), "hana_studio_v2.lock")

# Mutex 핸들 (전역으로 유지해야 함)
_mutex_handle = None


def check_single_instance() -> bool:
    """
    단일 인스턴스 실행 확인 (Named Mutex + Lock 파일 하이브리드)

    Returns:
        True: 첫 번째 인스턴스 (실행 가능)
        False: 이미 실행 중인 인스턴스 존재
    """
    global _mutex_handle

    if sys.platform == "win32":
        # Windows: Named Mutex 사용 (가장 안정적)
        try:
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.windll.kernel32

            # CreateMutexW 호출
            _mutex_handle = kernel32.CreateMutexW(
                None,  # 기본 보안 속성
                True,  # 초기 소유권 요청
                APP_MUTEX_NAME  # Mutex 이름
            )

            # ERROR_ALREADY_EXISTS (183) 확인
            last_error = kernel32.GetLastError()
            if last_error == 183:  # ERROR_ALREADY_EXISTS
                # 이미 실행 중 - 하지만 좀비인지 확인
                if _is_zombie_process():
                    # 좀비면 강제로 뮤텍스 해제하고 재시도
                    kernel32.CloseHandle(_mutex_handle)
                    _cleanup_lock_file()
                    _mutex_handle = kernel32.CreateMutexW(None, True, APP_MUTEX_NAME)
                    if kernel32.GetLastError() == 183:
                        return False
                else:
                    return False

            # Lock 파일에 현재 PID 저장
            _write_lock_file()
            return True

        except Exception as e:
            print(f"[WARN] Mutex 생성 실패, Lock 파일 방식 사용: {e}")
            return _check_lock_file()
    else:
        # Linux/Mac: Lock 파일 방식
        return _check_lock_file()


def _is_zombie_process() -> bool:
    """Lock 파일의 PID가 좀비(존재하지 않는 프로세스)인지 확인"""
    try:
        if not os.path.exists(APP_LOCK_FILE):
            return True

        with open(APP_LOCK_FILE, 'r') as f:
            old_pid = int(f.read().strip())

        # 해당 PID가 실제로 존재하는지 확인
        if sys.platform == "win32":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, old_pid)
            if handle:
                kernel32.CloseHandle(handle)
                return False  # 프로세스 존재
            return True  # 프로세스 없음 (좀비)
        else:
            os.kill(old_pid, 0)  # 시그널 0으로 존재 확인
            return False
    except (ValueError, FileNotFoundError, OSError, ProcessLookupError):
        return True


def _write_lock_file():
    """현재 PID를 Lock 파일에 저장"""
    try:
        with open(APP_LOCK_FILE, 'w') as f:
            f.write(str(os.getpid()))
    except Exception:
        pass


def _cleanup_lock_file():
    """Lock 파일 삭제"""
    try:
        if os.path.exists(APP_LOCK_FILE):
            os.remove(APP_LOCK_FILE)
    except Exception:
        pass


def _check_lock_file() -> bool:
    """Lock 파일 기반 단일 인스턴스 확인 (폴백)"""
    if _is_zombie_process():
        _cleanup_lock_file()

    if os.path.exists(APP_LOCK_FILE):
        return False

    _write_lock_file()
    return True


def cleanup_on_exit():
    """종료 시 리소스 정리"""
    global _mutex_handle

    _cleanup_lock_file()

    if _mutex_handle and sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.kernel32.ReleaseMutex(_mutex_handle)
            ctypes.windll.kernel32.CloseHandle(_mutex_handle)
        except Exception:
            pass


def create_loading_screen(app):
    """로딩 화면 생성"""
    from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QProgressBar
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QFont

    loading = QWidget()
    loading.setWindowTitle("Hana Studio v2")
    loading.setFixedSize(360, 160)
    loading.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
    loading.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
    loading.setStyleSheet("""
        QWidget {
            background-color: #1a1a2e;
            border-radius: 12px;
        }
    """)

    layout = QVBoxLayout(loading)
    layout.setContentsMargins(32, 28, 32, 28)
    layout.setSpacing(16)

    # 타이틀
    title = QLabel("Hana Studio v2")
    title.setAlignment(Qt.AlignmentFlag.AlignCenter)
    title.setFont(QFont("Segoe UI", 18, QFont.Weight.DemiBold))
    title.setStyleSheet("color: #eef1ff; background: transparent;")
    layout.addWidget(title)

    # 상태 텍스트
    status = QLabel("시작하는 중...")
    status.setAlignment(Qt.AlignmentFlag.AlignCenter)
    status.setFont(QFont("Segoe UI", 10))
    status.setStyleSheet("color: #a0a0b0; background: transparent;")
    layout.addWidget(status)

    # 프로그레스 바
    progress = QProgressBar()
    progress.setFixedHeight(4)
    progress.setTextVisible(False)
    progress.setRange(0, 100)
    progress.setValue(0)
    progress.setStyleSheet("""
        QProgressBar {
            background-color: #2a2a4e;
            border: none;
            border-radius: 2px;
        }
        QProgressBar::chunk {
            background-color: #6c63ff;
            border-radius: 2px;
        }
    """)
    layout.addWidget(progress)

    # 화면 중앙 배치
    screen = app.primaryScreen().geometry()
    x = (screen.width() - loading.width()) // 2
    y = (screen.height() - loading.height()) // 2
    loading.move(x, y)

    return loading, status, progress


def run_loading_sequence(app, loading, status, progress):
    """단계별 로딩 실행"""
    steps = [
        ("NumPy 로딩...", 15, lambda: __import__('numpy')),
        ("OpenCV 로딩...", 35, lambda: __import__('cv2')),
        ("ONNX Runtime 로딩...", 55, lambda: __import__('onnxruntime')),
        ("UI 모듈 로딩...", 75, lambda: __import__('ui_v2')),
        ("메인 윈도우 준비...", 90, None),
    ]

    for msg, prog, loader in steps:
        status.setText(msg)
        progress.setValue(prog)
        app.processEvents()

        if loader:
            try:
                loader()
                print(f"[OK] {msg.replace('...', '')}")
            except Exception as e:
                print(f"[WARN] {msg} 실패: {e}")

        app.processEvents()

    progress.setValue(100)
    app.processEvents()


def main():
    """메인 함수"""
    import atexit

    # 단일 인스턴스 확인
    if not check_single_instance():
        print("[EXIT] 이미 실행 중인 Hana Studio v2가 있습니다.")
        sys.exit(0)

    # 종료 시 정리 등록
    atexit.register(cleanup_on_exit)

    # Qt import (단일 인스턴스 확인 후)
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import Qt

    # Qt 애플리케이션 생성
    app = QApplication(sys.argv)

    # 고해상도 디스플레이 지원
    app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    # 애플리케이션 정보
    app.setOrganizationName("Hana Studio")
    app.setApplicationName("Hana Studio v2")
    app.setApplicationVersion("2.0.0-MVP")

    print("=" * 60)
    print(" Hana Studio v2 - RTAI 카드 디자인 툴")
    print("=" * 60)

    # 라이선스 인증 (로딩 화면 전에)
    from licensing.dialog import check_license
    if not check_license():
        print("[EXIT] 라이선스 인증 실패")
        sys.exit(0)

    print("[OK] 라이선스 인증 완료")

    # 로딩 화면 표시
    loading, status, progress = create_loading_screen(app)
    loading.show()
    app.processEvents()

    # 단계별 로딩
    run_loading_sequence(app, loading, status, progress)

    # 메인 윈도우 생성
    from ui_v2 import HanaStudioMainWindowV2
    window = HanaStudioMainWindowV2()

    # 로딩 화면 닫고 메인 윈도우 표시
    loading.close()
    window.show()

    print("[OK] Hana Studio v2 시작 완료")

    # 이벤트 루프 실행
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
