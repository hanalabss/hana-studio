"""
main.py - 응답없음 완전 방지 시스템 v2
단순화된 접근: 로딩 화면만 표시하고 모든 무거운 작업은 show() 후에 처리
"""

import os
import sys

# GUI 모드에서 stdout/stderr가 None인 경우 처리 (PyInstaller --windowed)
def _debug_log_path() -> str:
    """기록 파일 위치. 실행 폴더에 쓸 수 없으면(관리자 권한 없이 Program Files) 사용자 폴더를 쓴다."""
    path = os.path.join(os.getcwd(), "hana_debug.log")
    try:
        with open(path, "a", encoding="utf-8"):
            pass
        return path
    except Exception:
        folder = os.path.join(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"), "HanaStudio", "logs")
        os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, "hana_debug.log")


if sys.stdout is None or sys.stderr is None:
    try:
        log_path = _debug_log_path()
        log_file = open(log_path, "w", encoding="utf-8", buffering=1)
        if sys.stdout is None:
            sys.stdout = log_file
        if sys.stderr is None:
            sys.stderr = log_file
    except Exception:
        devnull = open(os.devnull, 'w', encoding='utf-8')
        if sys.stdout is None:
            sys.stdout = devnull
        if sys.stderr is None:
            sys.stderr = devnull

# 안전한 print 함수
_original_print = print
_log_file = None

def safe_print(*args, **kwargs):
    """GUI 모드에서도 안전한 print 함수"""
    global _log_file
    if _log_file is None:
        try:
            log_path = _debug_log_path()
            _log_file = open(log_path, "a", encoding="utf-8")
        except Exception:
            _log_file = False

    try:
        _original_print(*args, **kwargs)
    except:
        pass

    if _log_file and _log_file is not False:
        try:
            message = " ".join(str(arg) for arg in args)
            _log_file.write(message + "\n")
            _log_file.flush()
        except:
            pass

import builtins
builtins.print = safe_print


def check_single_instance():
    """단일 인스턴스 실행 확인"""
    try:
        import psutil
        current_pid = os.getpid()

        if getattr(sys, 'frozen', False):
            target_name = "HanaStudio.exe"
        else:
            target_name = "python.exe"

        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['pid'] == current_pid:
                    continue
                if not proc.info['name'] or proc.info['name'].lower() != target_name.lower():
                    continue

                if getattr(sys, 'frozen', False):
                    return False
                else:
                    cmdline = proc.info.get('cmdline', [])
                    is_main_py = any('main.py' in arg for arg in cmdline if arg)
                    if is_main_py:
                        return False
            except:
                continue

        return True
    except ImportError:
        return True
    except Exception:
        return True


def main():
    """메인 함수 - 응답없음 완전 방지"""
    try:
        print("[START] Hana Studio 시작...")

        # 중복 실행 방지 (임시 비활성화)
        # if not check_single_instance():
        #     print("[INFO] 이미 실행 중인 인스턴스가 있습니다.")
        #     sys.exit(0)

        try:
            from utils.console import ensure_utf8_console
            ensure_utf8_console()
        except Exception:
            pass

        # 최소한의 Qt import만
        from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel
        from PySide6.QtCore import Qt, QTimer
        from PySide6.QtGui import QFont

        # 화면 배율(125%, 150%)에서도 새 화면이 선명하게 보이도록 설정
        from ui.theme import configure_hidpi
        configure_hidpi()

        # Qt 애플리케이션 생성
        # 작업 표시줄에 파이썬 대신 앱 아이콘이 보이게 한다(창을 만들기 전에)
        from ui.theme.brand import app_icon, logo_pixmap, set_windows_app_id
        set_windows_app_id()

        app = QApplication(sys.argv)
        app.setWindowIcon(app_icon())
        app.setStyle('Fusion')
        app.setQuitOnLastWindowClosed(True)

        print("[OK] QApplication 생성 완료")

        # 라이선스 인증
        from licensing.dialog import check_license
        license_success, is_admin = check_license()
        if not license_success:
            print("[EXIT] 라이선스 인증 실패")
            sys.exit(0)

        print(f"[OK] 라이선스 인증 완료 (Admin: {is_admin})")

        # 새 화면 테마 적용 (라이선스 창은 기존 모습 유지)
        from ui.theme import ThemeManager
        ThemeManager.instance().apply(app)

        # 로딩 화면: 로고, 이름, 지금 하는 일, 아래에 리본 색 띠
        loading = QWidget()
        loading.setWindowTitle("하나스튜디오")
        loading.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        loading.setFixedSize(420, 240)
        loading.setObjectName("Loading")
        loading.setAttribute(Qt.WA_StyledBackground, True)
        loading.setStyleSheet("#Loading { background-color: #FFFFFF; border: 1px solid #DDE1E7; }")

        outer = QVBoxLayout(loading)
        outer.setContentsMargins(1, 1, 1, 1)
        outer.setSpacing(0)
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(32, 36, 32, 24)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(10)

        logo = QLabel()
        logo.setAlignment(Qt.AlignCenter)
        logo.setPixmap(logo_pixmap(64, loading.devicePixelRatioF()))
        layout.addWidget(logo)

        title = QLabel("하나스튜디오")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Malgun Gothic", 17, QFont.Bold))
        title.setStyleSheet("color: #1B2430;")
        layout.addWidget(title)

        status = QLabel("시작하는 중...")
        status.setAlignment(Qt.AlignCenter)
        status.setFont(QFont("Malgun Gothic", 9))
        status.setStyleSheet("color: #5B6676;")
        layout.addWidget(status)
        outer.addWidget(body, 1)

        ribbon = QWidget()
        ribbon.setFixedHeight(6)
        ribbon_row = QHBoxLayout(ribbon)
        ribbon_row.setContentsMargins(0, 0, 0, 0)
        ribbon_row.setSpacing(0)
        for color in ("#F5C400", "#E0338A", "#1AA7E0", "#1B2430"):
            block = QWidget()
            block.setStyleSheet(f"background-color: {color};")
            ribbon_row.addWidget(block)
        outer.addWidget(ribbon)

        # 화면 중앙에 배치
        screen = app.primaryScreen().availableGeometry()
        loading.move(screen.x() + (screen.width() - loading.width()) // 2,
                     screen.y() + (screen.height() - loading.height()) // 2)

        # 로딩 화면 표시
        loading.show()
        app.processEvents()

        print("[OK] 로딩 화면 표시")

        # 상태 변수
        state = {
            'step': 0,
            'main_window': None,
            'is_admin': is_admin
        }

        def do_loading_step():
            """각 로딩 단계 실행"""
            step = state['step']

            try:
                if step == 0:
                    status.setText("계산 도구를 불러오는 중...")
                    app.processEvents()
                    import numpy
                    print("[IMPORT] numpy")

                elif step == 1:
                    status.setText("그림 도구를 불러오는 중...")
                    app.processEvents()
                    import cv2
                    print("[IMPORT] cv2")

                elif step == 2:
                    status.setText("배경 지우기 기능을 준비하는 중...")
                    app.processEvents()
                    import onnxruntime
                    print("[IMPORT] onnxruntime")

                elif step == 3:
                    status.setText("화면을 준비하는 중...")
                    app.processEvents()
                    from ui.components.control_panels import (
                        PrinterPanel, FileSelectionPanel, PrintModePanel,
                        PositionAdjustPanel, ProgressPanel, PrintQuantityPanel
                    )
                    print("[IMPORT] control_panels")

                elif step == 4:
                    status.setText("화면을 준비하는 중...")
                    app.processEvents()
                    from ui.components.image_viewer import ImageViewer
                    from ui import main_window
                    print("[IMPORT] image_viewer")

                elif step == 5:
                    status.setText("거의 다 됐습니다...")
                    app.processEvents()
                    from ui.app import build_main_window
                    state['main_window'] = build_main_window(is_admin=state['is_admin'])
                    print(f"[INIT] 메인 화면 생성 (Admin: {state['is_admin']})")

                elif step in (6, 7):
                    # 기존 화면(빠른 인쇄)의 패널·뷰어는 그 화면을 처음 열 때 초기화된다.
                    pass

                elif step == 8:
                    # 완료 - 로딩 화면 닫고 메인 윈도우 표시
                    loading.close()
                    state['main_window'].showMaximized()
                    print("[OK] 메인 윈도우 표시")
                    return  # 더 이상 단계 없음

            except Exception as e:
                print(f"[ERROR] Step {step}: {e}")
                import traceback
                traceback.print_exc()
                # 에러가 나도 계속 진행
                if step >= 8:
                    return

            # 다음 단계 예약
            state['step'] += 1
            app.processEvents()
            QTimer.singleShot(10, do_loading_step)

        # 첫 로딩 단계 시작 (50ms 후)
        QTimer.singleShot(50, do_loading_step)

        # 앱 실행
        exit_code = app.exec()
        print(f"종료 Hana Studio 종료 (코드: {exit_code})")
        sys.exit(exit_code)

    except Exception as e:
        print(f"❌ 시작 오류: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
