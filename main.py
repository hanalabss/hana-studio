"""
Hana Studio 메인 진입점 - High DPI 경고 수정
"""

import sys
import os
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication

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
    """QApplication 설정 - High DPI 경고 수정"""
    
    # Qt6에서 권장하는 방식으로 High DPI 설정 (QApplication 생성 전)
    try:
        # Qt6에서는 이 방법이 권장됨
        QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.Floor
        )
    except AttributeError:
        # 구버전 Qt에서는 이 속성이 없을 수 있음
        pass
    
    app = QApplication(sys.argv)
    
    # Qt6에서 deprecated된 속성들 제거
    # 대신 환경변수로 처리했음
    
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
    # 환경 설정
    setup_environment()
    validate_config()
    
    # 애플리케이션 생성
    app = setup_application()
    
    try:
        # AI 모델 로딩 다이얼로그 표시
        from ui.loading_dialog import LoadingDialog
        loading = LoadingDialog()
        
        if loading.exec() != LoadingDialog.DialogCode.Accepted:
            print("AI 모델 로드 실패")
            sys.exit(1)
        
        # 메인 윈도우 생성 및 표시
        window = HanaStudio()
        window.show()
        
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