"""
main.py의 중복 실행 방지 로직 수정
더 정확하고 안정적인 프로세스 검사
"""

import os
import sys
import time
import io

# GUI 모드에서 print를 안전하게 처리하기 위한 설정
_original_print = print

def safe_print(*args, **kwargs):
    """GUI 모드에서도 안전한 print 함수"""
    try:
        _original_print(*args, **kwargs)
    except:
        # GUI 모드에서 print 실패 시 무시
        pass

# 전역 print를 safe_print로 교체
import builtins
builtins.print = safe_print

# UTF-8 인코딩 설정 (GUI 모드에서 안전하게)
try:
    # stdout과 stderr가 있는지 확인
    if sys.stdout and hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    if sys.stderr and hasattr(sys.stderr, 'buffer'):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
except:
    # GUI 모드에서는 stdout/stderr가 없을 수 있음
    pass


def check_single_instance():
    """단일 인스턴스 실행 확인 - 수정된 버전"""
    try:
        import psutil
        current_pid = os.getpid()
        
        if getattr(sys, 'frozen', False):
            # PyInstaller 실행파일인 경우
            target_name = "HanaStudio.exe"
            print(f"[DEBUG] 실행파일 모드: {target_name} 검사")
        else:
            # 개발 환경인 경우
            target_name = "python.exe"
            print(f"[DEBUG] 개발 모드: main.py를 실행하는 {target_name} 검사")
        
        # 중복 프로세스 찾기
        duplicate_found = False
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                # 자기 자신은 제외
                if proc.info['pid'] == current_pid:
                    continue
                
                # 프로세스 이름 확인
                if not proc.info['name'] or proc.info['name'].lower() != target_name.lower():
                    continue
                
                if getattr(sys, 'frozen', False):
                    # 실행파일: 같은 이름이면 중복
                    duplicate_found = True
                    print(f"[DEBUG] 중복 실행파일 발견: PID {proc.info['pid']}")
                    break
                else:
                    # 개발 환경: main.py를 실행하는지 확인
                    cmdline = proc.info.get('cmdline', [])
                    
                    # main.py가 명령줄에 있는지 확인
                    is_main_py = any(
                        arg.endswith('main.py') or 'main.py' in arg 
                        for arg in cmdline if arg
                    )
                    
                    if is_main_py:
                        duplicate_found = True
                        print(f"[DEBUG] 중복 main.py 프로세스 발견: PID {proc.info['pid']}")
                        print(f"[DEBUG] 명령줄: {' '.join(cmdline)}")
                        break
                        
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                # 프로세스가 사라졌거나 접근 권한이 없는 경우 무시
                continue
            except Exception as e:
                # 기타 오류는 로그만 출력하고 계속
                print(f"[DEBUG] 프로세스 검사 오류: {e}")
                continue
        
        if duplicate_found:
            print("⚠️ Hana Studio가 이미 실행 중입니다.")
            print("기존 실행 중인 창을 확인하세요.")
            
            # 사용자에게 강제 실행 여부 확인
            try:
                answer = input("그래도 실행하시겠습니까? (y/N): ").strip().lower()
                if answer in ['y', 'yes']:
                    print("✅ 강제 실행합니다.")
                    return True
                else:
                    return False
            except (KeyboardInterrupt, EOFError):
                print("\n❌ 실행을 취소합니다.")
                return False
        
        print("✅ 중복 실행 없음, 정상 시작합니다.")
        return True
        
    except ImportError:
        print("[DEBUG] psutil이 설치되지 않음, 락 파일 방식으로 전환")
        return check_single_instance_lockfile()
    except Exception as e:
        print(f"[DEBUG] 중복 검사 중 오류 발생: {e}")
        print("⚠️ 오류가 발생했지만 실행을 계속합니다.")
        return True


def check_single_instance_lockfile():
    """락 파일을 이용한 중복 실행 방지 (psutil 대체)"""
    try:
        import tempfile
        
        # 락 파일 경로 생성
        if getattr(sys, 'frozen', False):
            lock_filename = "hana_studio_exe.lock"
        else:
            lock_filename = "hana_studio_dev.lock"
        
        lock_path = os.path.join(tempfile.gettempdir(), lock_filename)
        
        try:
            # 락 파일 생성 시도
            with open(lock_path, 'x') as lock_file:
                lock_file.write(f"PID:{os.getpid()}\nTime:{time.time()}")
            
            print(f"✅ 락 파일 생성 성공: {lock_path}")
            return True
            
        except FileExistsError:
            # 락 파일이 이미 존재하는 경우
            print(f"⚠️ 락 파일이 이미 존재: {lock_path}")
            
            # 락 파일이 오래된 경우 정리 시도
            try:
                file_time = os.path.getmtime(lock_path)
                current_time = time.time()
                
                # 1시간 이상 된 락 파일은 삭제
                if current_time - file_time > 3600:
                    os.remove(lock_path)
                    print("🧹 오래된 락 파일을 삭제했습니다.")
                    return check_single_instance_lockfile()  # 재시도
                    
            except Exception as cleanup_error:
                print(f"[DEBUG] 락 파일 정리 실패: {cleanup_error}")
            
            print("다른 Hana Studio 인스턴스가 실행 중일 수 있습니다.")
            
            # 사용자에게 강제 실행 여부 확인
            try:
                answer = input("그래도 실행하시겠습니까? (y/N): ").strip().lower()
                if answer in ['y', 'yes']:
                    # 기존 락 파일 강제 삭제
                    try:
                        os.remove(lock_path)
                        print("✅ 락 파일을 강제 삭제하고 실행합니다.")
                        return True
                    except Exception as force_error:
                        print(f"❌ 락 파일 삭제 실패: {force_error}")
                        return False
                else:
                    return False
            except (KeyboardInterrupt, EOFError):
                print("\n❌ 실행을 취소합니다.")
                return False
                
    except Exception as e:
        print(f"[DEBUG] 락 파일 방식 오류: {e}")
        print("⚠️ 중복 검사를 건너뛰고 실행합니다.")
        return True


def check_single_instance_simple():
    """간단한 중복 실행 방지 (개발용)"""
    print("[DEBUG] 간단한 중복 검사 모드")
    
    # 환경 변수로 중복 실행 허용 여부 확인
    if os.environ.get('HANA_ALLOW_MULTIPLE', '').lower() == 'true':
        print("✅ 환경 변수로 중복 실행이 허용됨")
        return True
    
    # 기본적으로 허용
    return True


# main.py에서 사용할 최종 함수
def main():
    """메인 함수 - 개선된 중복 검사 및 스플래시 스크린"""
    try:
        print("[START] Hana Studio 시작...")
        
        # 중복 실행 방지 (더 안정적인 버전)
        if not check_single_instance():
            print("프로그램을 종료합니다.")
            sys.exit(0)
        
        # Qt 환경 변수 설정
        os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "0"
        os.environ["QT_SCALE_FACTOR_ROUNDING_POLICY"] = "Floor"
        
        # Qt 애플리케이션 생성
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import Qt, QTimer
        
        app = QApplication(sys.argv)
        app.setStyle('Fusion')
        app.setQuitOnLastWindowClosed(True)
        
        print("[OK] QApplication 생성 완료")
        
        # 스플래시 스크린 표시
        from ui.splash_screen import HanaStudioSplash
        
        splash = HanaStudioSplash()
        splash.show()
        app.processEvents()  # 즉시 표시되도록
        
        # 초기화 단계 1: 환경 설정
        splash.update_status("환경 설정 중...", 20)
        QTimer.singleShot(100, app.processEvents)  # UI 업데이트를 위한 짧은 대기
        
        # 초기화 단계 2: AI 모델 백그라운드 로딩 시작
        splash.update_status("AI 모델 준비 중...", 40)
        from core.model_loader import preload_ai_model
        preload_ai_model()
        
        # 초기화 단계 3: UI 모듈 로드
        splash.update_status("UI 모듈 로딩 중...", 60)
        from hana_studio import HanaStudio
        
        # 초기화 단계 4: 메인 윈도우 생성
        splash.update_status("메인 윈도우 생성 중...", 80)
        main_window = HanaStudio()
        
        # 초기화 단계 5: 최종 준비
        splash.update_status("시작 준비 완료!", 100)
        QTimer.singleShot(500, app.processEvents)  # 완료 메시지를 보여주기 위한 짧은 대기
        
        # 스플래시 종료 및 메인 윈도우 표시
        splash.finish(main_window)
        main_window.show()
        
        print("[OK] Hana Studio 시작 완료")
        print("[AI] 백그라운드에서 배경제거 AI 준비 중...")
        
        # 애플리케이션 실행
        exit_code = app.exec()
        
        print(f"종료 Hana Studio 종료 (코드: {exit_code})")
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
                "문제가 지속되면 프로그램을 다시 설치해보세요."
            )
        except Exception:
            pass
        
        sys.exit(1)


if __name__ == "__main__":
    main()