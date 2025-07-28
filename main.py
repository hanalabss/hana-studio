"""
최적화된 Hana Studio 메인 진입점
- 지연 로딩으로 빠른 시작
- AI 모델을 필요할 때만 로딩
- UI 우선 표시로 사용자 경험 개선
"""

import sys
import os
from PySide6.QtWidgets import QApplication, QSplashScreen, QLabel
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QGuiApplication, QIcon, QPixmap, QFont

from config import config, AppConstants, get_resource_path


def setup_environment():
    """환경 변수 설정"""
    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "0"
    os.environ["QT_SCALE_FACTOR_ROUNDING_POLICY"] = "Floor"
    os.environ["QT_FONT_DPI"] = "96"
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "0"


def setup_application() -> QApplication:
    """QApplication 설정 - 최적화"""
    try:
        QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.Floor
        )
    except AttributeError:
        pass
    
    app = QApplication(sys.argv)
    
    # 앱 정보 설정
    app.setApplicationName(AppConstants.APP_NAME)
    app.setApplicationVersion(AppConstants.APP_VERSION)
    app.setOrganizationName(AppConstants.APP_AUTHOR)
    
    # 아이콘 설정 (실패해도 무시)
    try:
        icon_path = get_resource_path("hana.ico")
        if os.path.exists(icon_path):
            app.setWindowIcon(QIcon(icon_path))
    except Exception:
        pass
    
    # 기본 스타일만 설정 (팔레트는 나중에)
    app.setStyle('Fusion')
    
    return app


def create_splash_screen(app) -> QSplashScreen:
    """스플래시 화면 생성"""
    try:
        # 아이콘이 있으면 사용, 없으면 텍스트만
        icon_path = get_resource_path("hana.ico")
        if os.path.exists(icon_path):
            pixmap = QPixmap(icon_path).scaled(200, 200, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        else:
            # 간단한 텍스트 스플래시
            pixmap = QPixmap(400, 200)
            pixmap.fill(Qt.GlobalColor.white)
        
        splash = QSplashScreen(pixmap)
        splash.setStyleSheet("""
            QSplashScreen {
                background-color: white;
                border: 2px solid #4A90E2;
                border-radius: 10px;
            }
        """)
        
        # 로딩 메시지
        splash.showMessage(
            "🎨 Hana Studio 시작 중...",
            Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter,
            Qt.GlobalColor.black
        )
        
        return splash
        
    except Exception:
        return None


def validate_config():
    """빠른 설정 검증"""
    try:
        if not config.validate_settings():
            config.reset_to_defaults()
    except Exception:
        pass  # 설정 오류는 나중에 처리


def main():
    """최적화된 메인 실행 함수"""
    print("🚀 Hana Studio 시작 중...")
    
    # 1단계: 빠른 환경 설정
    setup_environment()
    validate_config()
    
    # 2단계: Qt 애플리케이션 생성
    app = setup_application()
    
    # 3단계: 스플래시 화면 표시 (즉시)
    splash = create_splash_screen(app)
    if splash:
        splash.show()
        app.processEvents()  # 즉시 화면에 표시
    
    try:
        # 4단계: UI 스타일 로딩 (지연)
        if splash:
            splash.showMessage("UI 테마 로딩 중...", Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter)
            app.processEvents()
        
        from ui.styles import get_light_palette
        app.setPalette(get_light_palette())
        
        # 5단계: 메인 윈도우 생성 (지연 import)
        if splash:
            splash.showMessage("메인 화면 준비 중...", Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter)
            app.processEvents()
        
        # 지연 import로 빠른 시작
        from hana_studio import HanaStudio
        
        # 6단계: AI 모델 로딩 (별도 다이얼로그, 비동기)
        if splash:
            splash.showMessage("AI 모델 준비 중...", Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter)
            app.processEvents()
        
        # 🎯 현대적인 설치 다이얼로그 사용
        from ui.installation_dialog import show_installation_dialog
        
        # 스플래시 화면 숨기기
        if splash:
            splash.hide()
        
        # 설치 다이얼로그 표시
        print("🚀 설치 다이얼로그 시작...")
        installation_success = show_installation_dialog()
        
        if not installation_success:
            print("❌ 설치 취소 또는 실패")
            sys.exit(1)
        
        # 7단계: 메인 윈도우 표시
        print("✅ 메인 윈도우 생성 중...")
        window = HanaStudio()
        window.show()
        
        print("🎉 Hana Studio 시작 완료!")
        
        # 애플리케이션 실행
        sys.exit(app.exec())
        
    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 중단되었습니다.")
        sys.exit(0)
        
    except ImportError as e:
        error_msg = f"필수 모듈 로드 실패: {e}"
        print(f"❌ {error_msg}")
        
        # 사용자에게 친화적인 오류 메시지
        try:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(
                None,
                "시작 오류",
                f"프로그램 시작에 필요한 구성요소를 찾을 수 없습니다.\n\n"
                f"오류 세부사항:\n{error_msg}\n\n"
                "프로그램을 다시 설치해주세요."
            )
        except Exception:
            pass
        
        sys.exit(1)
        
    except Exception as e:
        error_msg = f"시작 중 오류 발생: {e}"
        print(f"❌ {error_msg}")
        
        # 스플래시 화면이 있으면 숨기기
        if 'splash' in locals() and splash:
            splash.hide()
        
        # 상세한 오류 정보 출력 (개발자용)
        import traceback
        traceback.print_exc()
        
        # 사용자 친화적 오류 다이얼로그
        try:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(
                None,
                "시작 오류",
                f"프로그램 시작 중 예상치 못한 오류가 발생했습니다.\n\n"
                f"오류 내용:\n{str(e)[:200]}...\n\n"
                "문제가 계속되면 지원팀에 문의해주세요."
            )
        except Exception:
            pass
        
        sys.exit(1)


if __name__ == "__main__":
    main()