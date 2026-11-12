"""하나스튜디오 2 배포판 만들기 (온라인 업데이트용: Velopack + GitHub 릴리스).

    python scripts/release.py                  빌드하고 releases_v2/ 에 설치 파일과 업데이트 파일을 만든다
    python scripts/release.py --skip-build     dist_v2/HanaStudio 를 그대로 써서 포장만 한다
    python scripts/release.py --upload         만든 뒤 GitHub 릴리스로 올리고, 릴리스를 2개(새 버전과 직전 버전)만 남긴다
    python scripts/release.py --version 2.0.1  config.py의 APP_VERSION 대신 이 버전으로 만든다

올리기(--upload)는 누구나 받을 수 있는 곳에 공개하는 일이다. 버전을 올리는 방법:
config.py의 APP_VERSION을 바꾸고 커밋한 뒤 이 스크립트를 --upload로 돌린다.

필요한 것: PyInstaller, vpk(Velopack CLI, `dotnet tool install vpk --tool-path D:\\tools\\vpk`), gh(올릴 때만).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACK_ID = "HanaStudio2"
TITLE = "하나스튜디오 2"
REPO_URL = "https://github.com/hanalabss/hana-studio"
REPO = "hanalabss/hana-studio"
DIST = ROOT / "dist_v2" / "HanaStudio"
OUT = ROOT / "releases_v2"
KEEP_RELEASES = 2
RUNTIME_LEFTOVERS = ("masks", "output", "temp", "license.json", "hana_debug.log")


def run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    print(">", " ".join(str(part) for part in command), flush=True)
    return subprocess.run(command, cwd=ROOT, check=True, **kwargs)


def app_version() -> str:
    text = (ROOT / "config.py").read_text(encoding="utf-8")
    found = re.search(r'APP_VERSION\s*=\s*"([^"]+)"', text)
    if not found:
        raise SystemExit("config.py에서 APP_VERSION을 찾지 못했습니다.")
    return found.group(1)


def find_vpk() -> str:
    for candidate in (os.environ.get("VPK_EXE"), r"D:\tools\vpk\vpk.exe", shutil.which("vpk")):
        if candidate and os.path.isfile(candidate):
            return candidate
    raise SystemExit("vpk를 찾지 못했습니다. `dotnet tool install vpk --tool-path D:\\tools\\vpk`로 설치하세요.")


def build() -> None:
    run([sys.executable, "-m", "PyInstaller", "HanaStudio.spec", "--noconfirm",
         "--distpath", "dist_v2", "--workpath", "build_v2"])


def clean_dist() -> None:
    """개발 PC에서 실행하며 생긴 파일(인증번호, 기록, 임시 폴더)이 배포판에 섞이지 않게 지운다."""
    for name in RUNTIME_LEFTOVERS:
        target = DIST / name
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
        elif target.exists():
            target.unlink()


def gh_token() -> str:
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        return token
    return subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, check=True).stdout.strip()


def fetch_previous(vpk: str, token: str | None) -> None:
    """직전 버전을 받아 둔다. 있어야 '바뀐 부분만 담은 파일'(delta)이 만들어진다. 첫 배포면 받을 것이 없다."""
    command = [vpk, "download", "github", "--repoUrl", REPO_URL, "--outputDir", str(OUT)]
    if token:
        command += ["--token", token]
    try:
        run(command)
    except subprocess.CalledProcessError:
        print("직전 버전을 받지 못했습니다(첫 배포라면 정상입니다). 전체 파일만 만듭니다.")


def pack(vpk: str, version: str) -> None:
    if not (DIST / "HanaStudio.exe").is_file():
        raise SystemExit(f"{DIST}에 HanaStudio.exe가 없습니다. 먼저 빌드하세요.")
    OUT.mkdir(exist_ok=True)
    run([vpk, "pack", "--packId", PACK_ID, "--packVersion", version, "--packDir", str(DIST),
         "--mainExe", "HanaStudio.exe", "--packTitle", TITLE, "--packAuthors", "Hana Platform",
         "--icon", str(ROOT / "hana.ico"), "--outputDir", str(OUT),
         "--noPortable"])        # 설치 파일과 업데이트 파일만 만든다(휴대용 zip 432MB는 올리지 않는다)


def upload(vpk: str, version: str, token: str) -> None:
    run([vpk, "upload", "github", "--repoUrl", REPO_URL, "--outputDir", str(OUT), "--token", token,
         "--publish", "--tag", f"v{version}", "--releaseName", f"{TITLE} {version}"])


def prune_releases(keep: int = KEEP_RELEASES) -> list[str]:
    """v로 시작하는 정식 릴리스 가운데 최근 keep개만 남기고 지운다(태그도 함께). 지운 태그를 돌려준다."""
    listing = subprocess.run(
        ["gh", "release", "list", "-R", REPO, "--limit", "100", "--json", "tagName,isPrerelease,isDraft,publishedAt"],
        capture_output=True, text=True, check=True, cwd=ROOT).stdout
    releases = [r for r in json.loads(listing)
                if re.match(r"^v\d", r["tagName"]) and not r["isPrerelease"] and not r["isDraft"]]
    releases.sort(key=lambda r: r["publishedAt"], reverse=True)
    removed = []
    for release in releases[keep:]:
        run(["gh", "release", "delete", release["tagName"], "-R", REPO, "--yes", "--cleanup-tag"])
        removed.append(release["tagName"])
    return removed


def prune_local(keep: int = KEEP_RELEASES) -> None:
    """releases_v2/ 에 전체 파일이 keep개를 넘게 쌓이지 않게 오래된 것을 지운다."""
    def version_key(path: Path) -> tuple:
        found = re.search(r"-(\d+(?:\.\d+)*)-", path.name)
        return tuple(int(part) for part in found.group(1).split(".")) if found else ()

    for pattern in (f"{PACK_ID}-*-full.nupkg", f"{PACK_ID}-*-delta.nupkg"):
        files = sorted(OUT.glob(pattern), key=version_key, reverse=True)
        for old in files[keep:]:
            old.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="하나스튜디오 2 배포판 만들기")
    parser.add_argument("--version", help="버전(기본: config.py의 APP_VERSION)")
    parser.add_argument("--skip-build", action="store_true", help="PyInstaller 빌드를 건너뛴다")
    parser.add_argument("--no-previous", action="store_true", help="직전 버전을 받지 않는다(delta 없이 전체 파일만)")
    parser.add_argument("--upload", action="store_true", help="GitHub 릴리스로 올린다(공개된다)")
    parser.add_argument("--pack-id", help="시험용: 다른 이름으로 포장한다(실제 설치본과 섞이지 않게)")
    parser.add_argument("--title", help="시험용: 설치되는 프로그램 이름")
    parser.add_argument("--out", help="시험용: 만든 파일을 둘 폴더")
    args = parser.parse_args()
    global PACK_ID, TITLE, OUT
    if args.pack_id or args.title or args.out:
        if args.upload:
            raise SystemExit("시험용 이름으로 만든 것은 올릴 수 없습니다.")
        PACK_ID, TITLE = args.pack_id or PACK_ID, args.title or TITLE
        OUT = Path(args.out).resolve() if args.out else OUT
        args.no_previous = True

    version = args.version or app_version()
    vpk = find_vpk()
    token = gh_token() if args.upload else None
    if not args.skip_build:
        build()
    clean_dist()
    if not args.no_previous:
        fetch_previous(vpk, token)
    pack(vpk, version)
    prune_local()
    print(f"\n만든 곳: {OUT}")
    for item in sorted(OUT.iterdir()):
        print(f"  {item.name}  ({item.stat().st_size / 1024 / 1024:.1f} MB)")
    if args.upload:
        upload(vpk, version, token)
        removed = prune_releases()
        print(f"올렸습니다: {REPO_URL}/releases/tag/v{version}")
        if removed:
            print("지운 예전 릴리스:", ", ".join(removed))
    else:
        print("\n올리지 않았습니다. 올리려면 --upload 를 붙여 다시 실행하세요.")


if __name__ == "__main__":
    main()
