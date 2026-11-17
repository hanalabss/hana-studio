"""화면에 보이는 짧은 한국어 문구(버튼·메뉴·제목)를 모아 ui/i18n/strings_ko.json으로 저장한다.

다른 언어 번역표(ui/i18n/<언어>.json)는 이 목록을 기준으로 채운다. 번역이 없는 문구는 한국어로 보인다.
    python scripts/extract_ui_strings.py
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HANGUL = re.compile("[가-힣]")
MAX_LENGTH = 30
SKIP = ("main_window.py",)


def extract() -> list[str]:
    found: set[str] = set()
    for path in sorted((ROOT / "ui").rglob("*.py")):
        if path.name in SKIP:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        skip: set[int] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.JoinedStr):
                skip.update(id(v) for v in ast.walk(node))
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                skip.add(id(node.value))            # 설명 글(docstring)
            elif isinstance(node, ast.Call) and getattr(node.func, "attr", "") in ("debug", "info", "warning",
                                                                                   "error", "exception"):
                skip.update(id(v) for v in ast.walk(node))   # 로그
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in skip:
                text = node.value
                if HANGUL.search(text) and len(text) <= MAX_LENGTH and "\n" not in text and "{" not in text:
                    found.add(text)
    return sorted(found)


def main() -> None:
    strings = extract()
    out = ROOT / "ui" / "i18n" / "strings_ko.json"
    out.write_text(json.dumps(strings, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"{len(strings)}개 → {out}")


if __name__ == "__main__":
    main()
