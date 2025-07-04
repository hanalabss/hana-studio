import os
import time
import sys

# 상위 디렉토리 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from printer_integration import R600Printer, find_printer_dll

def run_white_only_layer_test():
    white_mask_path = r"C:\Users\user\Desktop\Hana Studio\temp\_binary_white_layer.jpg"

    if not os.path.exists(white_mask_path):
        print(f"❌ 화이트 마스크 이미지 없음: {white_mask_path}")
        return

    dll_path = find_printer_dll()
    if not dll_path:
        print("❌ 프린터 DLL을 찾을 수 없습니다.")
        return

    try:
        with R600Printer(dll_path) as printer:
            printers = printer.enum_printers()
            if not printers:
                print("❌ 사용 가능한 프린터가 없습니다.")
                return

            print("[DEBUG] 프린터 연결 및 설정 시작")
            printer.set_timeout(15000)
            printer.select_printer(printers[0])

            # 카드 삽입
            print("[DEBUG] 카드 삽입 중")
            printer.inject_card()

            # 리본 설정 (화이트 → 컬러 순서)
            print("[DEBUG] 리본 옵션 설정: White → Color (value='0')")
            printer.set_ribbon_option(1, 0, "0")

            # 캔버스 준비
            print("[DEBUG] 캔버스 준비")
            printer.setup_canvas()

            # 화이트 임계값 설정
            threshold = 245
            print(f"[DEBUG] 화이트 임계값 설정: {threshold}")
            printer.set_white_layer_threshold(threshold)

            # 이미지 위치 및 크기 계산 (카드 중앙 배치)
            card_width = 85.6
            card_height = 53.98
            img_width = 40.0  # mm
            img_height = 40.0
            x_offset = (card_width - img_width) / 2
            y_offset = (card_height - img_height) / 2

            print(f"[DEBUG] 화이트 레이어 그리기: 위치=({x_offset}, {y_offset}), 크기=({img_width}x{img_height})")
            printer.draw_layer_white(x_offset, y_offset, img_width, img_height, white_mask_path)

            # 캔버스 커밋
            print("[DEBUG] 캔버스 커밋")
            printer.commit_canvas()

            # 인쇄 실행
            print("[DEBUG] 인쇄 실행")
            printer.print_draw()
            time.sleep(2)

            # 카드 배출
            print("[DEBUG] 카드 배출")
            printer.eject_card()
            time.sleep(1)

            print("✅ 화이트-only 인쇄 완료!")

    except Exception as e:
        print(f"❌ 테스트 중 오류 발생: {e}")

if __name__ == "__main__":
    run_white_only_layer_test()
