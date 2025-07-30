"""
Hana Studio 메인 시작 스크립트
초고속 UI 표시 및 단일 인스턴스 보장
"""

import sys
import os


def check_single_instance():
    """단일 인스턴스 실행 확인 - 중복 실행 방지"""
    try:
        import psutil
        current_pid = os.getpid()
        
        if getattr(sys, 'frozen', False):
            current_name = "HanaStudio.exe"
        else:
            current_name = "python.exe"
        
        # 같은 이름의 프로세스 찾기
        running_processes = []
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                proc_info = proc.info
                
                if proc_info['name'] and proc_info['name'].lower() == current_name.lower():
                    if proc_info['pid'] != current_pid:
                        # 개발 환경에서 명령줄 인수 확인
                        if not getattr(sys, 'frozen', False):
                            cmdline = proc_info.get('cmdline', [])
                            hana_related = any('main.py' in arg or 'hana_studio' in arg.lower() for arg in cmdline)
                            if hana_related:
                                running_processes.append(proc_info['pid'])
                        else:
                            # 실행파일인 경우 모든 동일한 이름의 프로세스
                            running_processes.append(proc_info['pid'])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        if running_processes:
            print(f"⚠️ Hana Studio가 이미 실행 중입니다 (PID: {running_processes})")
            print("기존 실행 중인 창을 확인하세요.")
            return False
        
        return True
        
    except ImportError:
        # psutil이 없는 경우 락 파일 방식 사용
        return check_single_instance_lockfile()
    except Exception as e:
        print(f"인스턴스 확인 오류: {e}")
        return True  # 오류 시 실행 허용


def check_single_instance_lockfile():
    """락 파일을 이용한 단일 인스턴스 확인"""
    try:
        import tempfile
        
        # 임시 락 파일 경로
        lock_file_path = os.path.join(tempfile.gettempdir(), "hana_studio.lock")
        
        if sys.platform == 'win32':
            # Windows: 파일 잠금으로 확인
            try:
                lock_file = open(lock_file_path, 'w')
                lock_file.write(str(os.getpid()))
                lock_file.flush()
                print(f"✅ 락 파일 생성: {lock_file_path}")
                return True  # 락 파일 생성 성공
            except IOError:
                print(f"⚠️ Hana Studio가 이미 실행 중입니다 (락 파일: {lock_file_path})")
                return False  # 이미 실행 중
        else:
            # Linux/Mac: fcntl 사용
            try:
                import fcntl
                lock_file = open(lock_file_path, 'w')
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                print(f"✅ 락 파일 생성: {lock_file_path}")
                return True
            except IOError:
                print(f"⚠️ Hana Studio가 이미 실행 중입니다 (락 파일: {lock_file_path})")
                return False
    except Exception as e:
        print(f"락 파일 확인 오류: {e}")
        return True  # 오류 시 실행 허용


def main():
    """메인 함수 - 단일 인스턴스 보장"""
    
    try:
        print("🚀 Hana Studio 시작...")
        
        # 중복 실행 방지 체크
        if not check_single_instance():
            print("프로그램을 종료합니다.")
            sys.exit(0)
        
        # Qt 환경 변수 설정
        os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "0"
        os.environ["QT_SCALE_FACTOR_ROUNDING_POLICY"] = "Floor"
        
        # Qt 애플리케이션 생성
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import Qt
        
        app = QApplication(sys.argv)
        app.setStyle('Fusion')
        app.setQuitOnLastWindowClosed(True)
        
        print("✅ QApplication 생성 완료")
        
        # 로딩 윈도우 표시
        from ui.simple_loading import SimpleLoadingWindow
        
        loading_window = SimpleLoadingWindow()
        # SimpleLoadingWindow에서 자동으로 show() 호출됨
        
        print("✅ 로딩 윈도우 표시 완료")
        print("⏳ 백그라운드에서 초기화 진행 중...")
        
        # 애플리케이션 실행
        exit_code = app.exec()
        
        print(f"🔚 Hana Studio 종료 (코드: {exit_code})")
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print("\n⏹️ 사용자에 의해 중단됨")
        sys.exit(0)
        
    except Exception as e:
        print(f"❌ 시작 오류: {e}")
        import traceback
        traceback.print_exc()
        
        # 오류 다이얼로그 표시
        try:
            from PySide6.QtWidgets import QMessageBox, QApplication
            
            if not QApplication.instance():
                app = QApplication(sys.argv)
            
            QMessageBox.critical(
                None,
                "시작 오류",
                f"프로그램 시작 중 오류가 발생했습니다:\n\n{str(e)}\n\n"
                "이미 실행 중인 프로그램이 있는지 확인하고 다시 시도하세요."
            )
        except Exception:
            # 다이얼로그 표시도 실패한 경우 무시
            pass
        
        sys.exit(1)


if __name__ == "__main__":
    main()