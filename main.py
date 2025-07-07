"""
Hana Studio 메인 진입점 - 리팩토링된 버전
"""

import sys
import os
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from hana_studio import HanaStudio
from ui.styles import get_light_palette
from config import config, AppConstants


def setup_environment():
    """환경 변수 설정"""
    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "0"
    os.environ["QT_SCALE_FACTOR_ROUNDING_POLICY"] = "Floor"
    os.environ["QT_FONT_DPI"] = "96"
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "0"


def setup_application() -> QApplication:
    """QApplication 설정"""
    app = QApplication(sys.argv)
    
    # 고해상도 DPI 지원 설정
    try:
        app.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, False)
        app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)
    except AttributeError:
        print("일부 DPI 속성을 설정할 수 없습니다. 버전 호환성 문제일 수 있습니다.")
    
    # 앱 정보 설정
    app.setApplicationName(AppConstants.APP_NAME)
    app.setApplicationVersion(AppConstants.APP_VERSION)
    app.setOrganizationName(AppConstants.APP_AUTHOR)
    
    # 라이트 테마 팔레트 설정
    app.setPalette(get_light_palette())
    app.setStyle('Fusion')
    
    return app


def validate_config():
    """설정 검증"""
    if not config.validate_settings():
        print("⚠️ 설정 검증 실패, 기본값으로 복원합니다.")
        config.reset_to_defaults()


def main():
    """메인 실행 함수"""
    print("🎨 Hana Studio 시작 중...")
    
    # 환경 설정
    setup_environment()
    
    # 설정 검증
    validate_config()
    
    # 애플리케이션 생성
    app = setup_application()
    
    try:
        # 메인 윈도우 생성 및 표시
        window = HanaStudio()
        window.show()
        
        print("✅ Hana Studio 시작 완료!")
        
        # 애플리케이션 실행
        sys.exit(app.exec())
        
    except KeyboardInterrupt:
        print("\n프로그램이 사용자에 의해 중단되었습니다.")
        sys.exit(0)
    except Exception as e:
        print(f"❌ 시작 중 오류 발생: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()