"""
실기 테스트 인쇄 스크립트. 카드 한 장과 리본을 소모하므로 --yes 없이는 실행하지 않는다.

    python scripts/hw_print_test.py --list                 # 프린터 목록만 보기
    python scripts/hw_print_test.py --index 1 --yes        # 1번 프린터로 테스트 카드 인쇄
    python scripts/hw_print_test.py --family smart --dual --yes
    python scripts/hw_print_test.py --index 1 --image card.png --orientation landscape --yes
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

from hw_common import connection_label, family_label, quiet, say, setup


def make_test_image(path: str, orientation: str, label: str) -> str:
    """색 막대와 테두리가 있는 테스트 카드 이미지(300dpi, CR80)를 만든다."""
    from PIL import Image, ImageDraw

    width, height = (1012, 636) if orientation == "landscape" else (636, 1012)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    colors = [(255, 0, 0), (0, 160, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255),
              (255, 255, 0), (0, 0, 0), (128, 128, 128)]
    bar = (width - 80) // len(colors)
    for index, color in enumerate(colors):
        draw.rectangle([40 + index * bar, height // 3, 40 + (index + 1) * bar, height * 2 // 3],
                       fill=color)
    draw.rectangle([12, 12, width - 13, height - 13], outline=(0, 0, 0), width=4)
    draw.text((40, 40), f"HANA STUDIO PRINT TEST - {label}", fill=(0, 0, 0))
    draw.text((40, height - 70), f"{width} x {height} px / {orientation}", fill=(0, 0, 0))
    image.save(path)
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="테스트 카드 한 장 인쇄 (카드·리본 소모)")
    parser.add_argument("--yes", action="store_true", help="카드와 리본 소모에 동의하고 실제로 인쇄")
    parser.add_argument("--list", action="store_true", help="프린터 목록만 보여 주고 끝냄")
    parser.add_argument("--family", choices=["all", "rtai", "smart"], default="all")
    parser.add_argument("--index", type=int, default=1, help="목록에서 고를 프린터 번호 (1부터)")
    parser.add_argument("--image", help="앞면 이미지 파일 (없으면 테스트 무늬를 만든다)")
    parser.add_argument("--back-image", help="뒷면 이미지 파일 (--dual과 함께)")
    parser.add_argument("--dual", action="store_true", help="양면 인쇄")
    parser.add_argument("--orientation", choices=["portrait", "landscape"], default="landscape")
    parser.add_argument("--offset-x", type=float, default=0.0, help="가로 위치 조정(mm)")
    parser.add_argument("--offset-y", type=float, default=0.0, help="세로 위치 조정(mm)")
    parser.add_argument("--verbose", action="store_true", help="SDK와 기존 코드의 출력도 표시")
    parser.add_argument("--fake", action="store_true", help="실기 대신 가상 프린터 사용")
    args = parser.parse_args(argv)

    setup()
    with quiet(args.verbose):
        from printer import registry
        from printer.base import CardJob, PrinterError, SideImages
    if args.fake:
        registry.use_fakes(True)

    say("=== 하나스튜디오 테스트 인쇄 ===")
    families = registry.ALL_FAMILIES if args.family == "all" else (args.family,)
    with quiet(args.verbose):
        devices = registry.discover(families)
    for line in devices.diagnostics:
        say(f"  - {line}")
    if not devices:
        say("인쇄할 프린터를 찾지 못했습니다.")
        return 1
    for number, device in enumerate(devices, start=1):
        say(f"  {number}. {family_label(device.family)} / {device.name} ({connection_label(device.connection)})")
    if args.list:
        return 0
    if not 1 <= args.index <= len(devices):
        say(f"--index 값이 범위를 벗어났습니다 (1~{len(devices)}).")
        return 1
    device = devices[args.index - 1]
    sides = "양면" if args.dual else "단면"
    say()
    say(f"대상: {args.index}번 {family_label(device.family)} / {device.name}")
    say(f"내용: {sides}, {'가로형' if args.orientation == 'landscape' else '세로형'}, "
        f"위치 조정 X={args.offset_x:+.2f}mm Y={args.offset_y:+.2f}mm")

    if not args.yes:
        say()
        say("이 명령은 카드 1장과 리본(재전사 프린터는 필름 포함)을 소모합니다.")
        say("실제로 인쇄하려면 같은 명령에 --yes 를 붙여 다시 실행해 주세요.")
        return 2

    work_dir = tempfile.mkdtemp(prefix="hana_print_test_")
    front_path = args.image or make_test_image(os.path.join(work_dir, "front.png"),
                                               args.orientation, "FRONT")
    back_path = None
    if args.dual:
        back_path = args.back_image or make_test_image(os.path.join(work_dir, "back.png"),
                                                       args.orientation, "BACK")
    for path in filter(None, (front_path, back_path)):
        if not os.path.isfile(path):
            say(f"이미지 파일을 찾을 수 없습니다: {path}")
            return 1

    job = CardJob(
        front=SideImages(front_path, orientation=args.orientation),
        back=SideImages(back_path, orientation=args.orientation) if back_path else None,
        options={"offset_x_mm": args.offset_x, "offset_y_mm": args.offset_y},
    )
    try:
        with quiet(args.verbose):
            printer = registry.open(device)
    except PrinterError as exc:
        say(f"연결 실패: {exc}")
        return 1
    try:
        with quiet(args.verbose):
            status = printer.status()
        say(f"상태: {'인쇄 가능' if status.ready else '인쇄 불가'} - {status.message}")
        if args.dual and not printer.supports("dual_side"):
            say("이 프린터는 양면 인쇄를 지원하지 않습니다.")
            return 1
        say("인쇄를 시작합니다...")
        with quiet(args.verbose):
            printer.print_card(job, progress=lambda percent, message: say(f"  [{percent:3d}%] {message}"))
        say("인쇄가 끝났습니다. 카드의 위치·색·방향을 확인해 주세요.")
        return 0
    except PrinterError as exc:
        say(f"인쇄 실패: {exc}")
        if exc.detail:
            say(f"상세: {exc.detail}")
        say("다시 시도할 수 있는 오류입니다." if exc.recoverable else "설정이나 장비 점검이 필요한 오류입니다.")
        return 1
    finally:
        with quiet(args.verbose):
            printer.close()


if __name__ == "__main__":
    sys.exit(main())
