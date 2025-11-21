"""
main.py - 응답없음 완전 방지 시스템 v2
단순화된 접근: 로딩 화면만 표시하고 모든 무거운 작업은 show() 후에 처리
"""

import os
import sys

# GUI 모드에서 stdout/stderr가 None인 경우 처리 (PyInstaller --windowed)
if sys.stdout is None or sys.stderr is None:
    try:
        log_path = os.path.join(os.getcwd(), "hana_debug.log")
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
            log_path = os.path.join(os.getcwd(), "hana_debug.log")
            _log_file = open(log_path, "w", encoding="utf-8")
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

        # 중복 실행 방지
        if not check_single_instance():
            print("[INFO] 이미 실행 중인 인스턴스가 있습니다.")
            sys.exit(0)

        # Qt 환경 변수 설정
        os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "0"
        os.environ["QT_SCALE_FACTOR_ROUNDING_POLICY"] = "Floor"

        # 최소한의 Qt import만
        from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel
        from PySide6.QtCore import Qt, QTimer
        from PySide6.QtGui import QFont

        # Qt 애플리케이션 생성
        app = QApplication(sys.argv)
        app.setStyle('Fusion')
        app.setQuitOnLastWindowClosed(True)

        print("[OK] QApplication 생성 완료")

        # 매우 심플한 로딩 화면
        loading = QWidget()
        loading.setWindowTitle("Hana Studio")
        loading.setFixedSize(400, 200)
        loading.setStyleSheet("background-color: #F8F9FA;")

        layout = QVBoxLayout(loading)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(15)

        # 타이틀
        title = QLabel("Hana Studio")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Segoe UI", 24, QFont.Bold))
        title.setStyleSheet("color: #343A40;")
        layout.addWidget(title)

        # 상태
        status = QLabel("Loading...")
        status.setAlignment(Qt.AlignCenter)
        status.setStyleSheet("color: #6C757D; font-size: 12px;")
        layout.addWidget(status)

        # 화면 중앙에 배치
        screen = app.primaryScreen().geometry()
        x = (screen.width() - loading.width()) // 2
        y = (screen.height() - loading.height()) // 2
        loading.move(x, y)

        # 로딩 화면 표시
        loading.show()
        app.processEvents()

        print("[OK] 로딩 화면 표시")

        # 상태 변수
        state = {
            'step': 0,
            'main_window': None
        }

        def do_loading_step():
            """각 로딩 단계 실행"""
            step = state['step']

            try:
                if step == 0:
                    status.setText("NumPy 로딩...")
                    app.processEvents()
                    import numpy
                    print("[IMPORT] numpy")

                elif step == 1:
                    status.setText("OpenCV 로딩...")
                    app.processEvents()
                    import cv2
                    print("[IMPORT] cv2")

                elif step == 2:
                    status.setText("ONNX Runtime 로딩...")
                    app.processEvents()
                    import onnxruntime
                    print("[IMPORT] onnxruntime")

                elif step == 3:
                    status.setText("UI 컴포넌트 로딩...")
                    app.processEvents()
                    from ui.components.control_panels import (
                        PrinterPanel, FileSelectionPanel, PrintModePanel,
                        PositionAdjustPanel, ProgressPanel, PrintQuantityPanel
                    )
                    print("[IMPORT] control_panels")

                elif step == 4:
                    status.setText("이미지 뷰어 로딩...")
                    app.processEvents()
                    from ui.components.image_viewer import ImageViewer
                    from ui import main_window
                    print("[IMPORT] image_viewer")

                elif step == 5:
                    status.setText("HanaStudio 생성 중...")
                    app.processEvents()
                    from hana_studio import HanaStudio
                    state['main_window'] = HanaStudio()
                    print("[INIT] HanaStudio 생성")

                elif step == 6:
                    status.setText("패널 초기화...")
                    app.processEvents()
                    state['main_window'].ui.initialize_panels_deferred()
                    print("[INIT] 패널 초기화")

                elif step == 7:
                    status.setText("뷰어 초기화...")
                    app.processEvents()
                    state['main_window'].ui.initialize_viewers_deferred()
                    print("[INIT] 뷰어 초기화")

                elif step == 8:
                    # 완료 - 로딩 화면 닫고 메인 윈도우 표시
                    loading.close()
                    state['main_window'].show()
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
