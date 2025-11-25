"""
Hana Studio v2 - RTAI 전용 카드 디자인 & 인쇄 툴
MVP (Minimum Viable Product)

캔버스 기반 디자인 툴로 완전히 재설계된 버전
- 드래그 앤 드롭 중심 UI
- 이미지/텍스트 레이어 시스템
- 600DPI 인쇄 지원 (다음 단계)
"""

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent))

from ui_v2 import HanaStudioMainWindowV2


def main():
    """메인 함수"""
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
    print()
    print("MVP 기능:")
    print("  ✅ RTAI 규격 캔버스 (600 DPI)")
    print("  ✅ 이미지 레이어 (드래그, 크기 조절, 회전)")
    print("  ✅ 텍스트 레이어 (폰트, 색상, 크기)")
    print("  ✅ Canva 스타일 UI")
    print("  ✅ 줌/팬 기능")
    print()
    print("다음 단계:")
    print("  ⬜ 600DPI 인쇄 렌더링")
    print("  ⬜ 프린터 모듈 통합")
    print("  ⬜ AI 배경제거 통합")
    print("  ⬜ 템플릿/프리셋 시스템")
    print()
    print("=" * 60)
    print()

    # 메인 윈도우 생성 및 표시
    window = HanaStudioMainWindowV2()
    window.show()

    # 이벤트 루프 실행
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
