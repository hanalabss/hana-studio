"""
실기 점검 스크립트 (카드·리본을 소모하지 않는다).

RTAI와 SMART 두 제품군의 프린터를 찾고, 한 대씩 열어 상태·정보·리본 잔량을 읽은 뒤 닫는다.
카드를 넣거나 빼는 명령은 보내지 않는다.

    python scripts/hw_probe.py              # 두 제품군 모두
    python scripts/hw_probe.py --family smart
    python scripts/hw_probe.py --verbose    # SDK·기존 코드의 디버그 출력도 보기
    python scripts/hw_probe.py --fake       # 가상 프린터로 동작만 확인
"""

from __future__ import annotations

import argparse
import sys
import traceback

from hw_common import connection_label, family_label, quiet, say, setup


def describe(value) -> str:
    if isinstance(value, dict):
        return ", ".join(f"{key}={describe(item)}" for key, item in value.items())
    return str(value)


def probe_device(registry, device, verbose: bool) -> bool:
    from printer.base import PrinterError

    say(f"  ■ {family_label(device.family)} / {device.name} ({connection_label(device.connection)})")
    if device.address and device.address != device.name:
        say(f"    주소: {device.address}")
    try:
        with quiet(verbose):
            printer = registry.open(device)
    except PrinterError as exc:
        say(f"    연결 실패: {exc}")
        if exc.detail:
            say(f"    상세: {exc.detail}")
        return False
    try:
        say("    연결: 성공")
        with quiet(verbose):
            capabilities = sorted(printer.capabilities())
            status = printer.status()
            info = printer.info()
        say(f"    지원 기능: {', '.join(capabilities)}")
        say(f"    상태: {'인쇄 가능' if status.ready else '인쇄 불가'} - {status.message}")
        if status.code is not None:
            say(f"    상태 코드: {status.code} (0x{status.code & 0xFFFFFFFFFFFFFFFF:X})")
        for warning in status.warnings:
            say(f"    경고: {warning}")
        say(f"    카드 공급함: {'비어 있음' if status.feeder_empty else '카드 있음' if status.feeder_empty is False else '확인 불가'}")
        if status.card_inside is not None:
            say(f"    프린터 내부 카드: {'있음' if status.card_inside else '없음'}")
        say(f"    리본 잔량: {status.ribbon_remaining if status.ribbon_remaining is not None else '확인 불가'}")
        if status.film_remaining is not None:
            say(f"    필름 잔량: {status.film_remaining}")
        if status.ribbon_type is not None:
            ribbon = status.details.get("ribbon", {}).get("name")
            say(f"    리본 종류: {status.ribbon_type}" + (f" ({ribbon})" if ribbon else ""))
        if info:
            say("    장치 정보:")
            for key, value in info.items():
                say(f"      {key}: {describe(value)}")
        else:
            say("    장치 정보: 읽지 못했습니다.")
        if verbose and status.details:
            say(f"    원본 상태 값: {describe(status.details)}")
        return True
    except PrinterError as exc:
        say(f"    점검 중 오류: {exc}")
        return False
    finally:
        with quiet(verbose):
            printer.close()
        say("    연결 종료")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="프린터 실기 점검 (카드·리본 소모 없음)")
    parser.add_argument("--family", choices=["all", "rtai", "smart"], default="all",
                        help="점검할 제품군 (기본: all)")
    parser.add_argument("--verbose", action="store_true", help="SDK와 기존 코드의 출력도 표시")
    parser.add_argument("--fake", action="store_true", help="실기 대신 가상 프린터 사용")
    args = parser.parse_args(argv)

    setup()
    with quiet(args.verbose):
        from printer import registry
        from printer import smart_printer
        from printer.printer_utils import get_default_dll_paths
    if args.fake:
        registry.use_fakes(True)

    say("=== 하나스튜디오 프린터 점검 (카드·리본을 소모하지 않습니다) ===")
    say(f"Python {sys.version.split()[0]} / {'64' if sys.maxsize > 2 ** 32 else '32'}비트")
    if not args.fake:
        import os
        with quiet(args.verbose):
            rtai_paths = [path for path in get_default_dll_paths() if os.path.isfile(path)]
        smart_path = smart_printer.find_smart_dll()
        say(f"RTAI 라이브러리: {rtai_paths[0] if rtai_paths else '찾을 수 없음'}")
        say(f"SMART 라이브러리: {smart_path or '찾을 수 없음'}")
    say()

    families = registry.ALL_FAMILIES if args.family == "all" else (args.family,)
    say("[1] 프린터 탐지")
    with quiet(args.verbose):
        devices = registry.discover(families)
    for line in devices.diagnostics:
        say(f"  - {line}")
    say(f"  찾은 프린터: {len(devices)}대")
    say()

    say("[2] 장치별 점검")
    if not devices:
        say("  점검할 프린터가 없습니다. 전원과 USB/네트워크 연결을 확인한 뒤 다시 실행해 주세요.")
    succeeded = 0
    for device in devices:
        try:
            succeeded += 1 if probe_device(registry, device, args.verbose) else 0
        except Exception:
            say("    예상하지 못한 오류:")
            say(traceback.format_exc())
    say()
    say(f"결과: 프린터 {len(devices)}대 중 {succeeded}대 점검 성공")
    return 0


if __name__ == "__main__":
    sys.exit(main())
