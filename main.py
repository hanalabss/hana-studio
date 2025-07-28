"""
초고속 UI 표시 main.py - 멀티스레드 버전
실행 즉시 로딩 윈도우를 표시하고 백그라운드에서 초기화 수행
"""

import sys
import os

def main():
    """초고속 시작 메인 함수 - 멀티스레드"""
    
    try:
        print("🚀 Hana Studio 시작...")
        
        # 환경 변수 설정 (빠르게)
        os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "0"
        os.environ["QT_SCALE_FACTOR_ROUNDING_POLICY"] = "Floor"
        
        # 🎯 최소한의 import로 QApplication 생성
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import Qt
        
        app = QApplication(sys.argv)
        app.setStyle('Fusion')
        
        print("✅ QApplication 생성 완료")
        
        # 🚀 즉시 로딩 윈도우 표시 (모든 작업이 백그라운드에서 처리됨)
        from ui.simple_loading import SimpleLoadingWindow
        
        loading_window = SimpleLoadingWindow()
        loading_window.show()
        
        print("✅ 로딩 윈도우 표시 완료")
        print("⏳ 백그라운드에서 초기화 진행 중...")
        
        # 애플리케이션 실행 (로딩 윈도우가 모든 것을 처리함)
        sys.exit(app.exec())
        
    except Exception as e:
        print(f"❌ 시작 오류: {e}")
        import traceback
        traceback.print_exc()
        
        try:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(
                None,
                "시작 오류",
                f"프로그램 시작 중 오류가 발생했습니다:\n\n{str(e)}"
            )
        except Exception:
            pass
        
        sys.exit(1)


if __name__ == "__main__":
    main()