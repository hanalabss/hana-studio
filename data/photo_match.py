"""사진 폴더만 고르면 명단의 사람과 사진 파일을 알아서 짝짓는다.

파일 이름이 명단의 한 칸과 똑같지 않아도 된다. '부서_이름_직급.jpg'처럼 여러 항목을 이어 붙인 이름이나
'이름(사번).jpg' 같은 이름도, 그 안에 든 낱말이 명단의 값과 얼마나 겹치는지로 찾는다.
"""
from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from .values import IMAGE_EXTENSIONS, is_empty

FOUND = "found"            # 한 장으로 정해졌다
AMBIGUOUS = "ambiguous"    # 후보가 여럿이라 정하지 못했다
MISSING = "missing"        # 맞는 파일이 없다

_EXACT = 100               # 파일 이름 전체가 값과 같다 (예: 사번.jpg)
_TOKEN = 10                # 파일 이름의 한 낱말이 값과 같다
_PART = 4                  # 파일 이름 안에 값이 들어 있다 (낱말 경계와는 안 맞는다)
# 사람을 가려내는 칸의 이름에 흔히 들어가는 말. 이런 칸이 있으면 그 칸만 근거로 삼는다.
PERSON_WORDS = ("이름", "성명", "성함", "name", "사번", "사원번호", "직원번호", "학번", "회원번호", "id", "코드")
_SPLIT = re.compile(r"[\s_\-.,()\[\]{}+&·/\\]+")


@dataclass
class PhotoMatch:
    status: str = MISSING
    file_name: str = ""                                  # 찾았을 때의 파일 이름(폴더 제외)
    candidates: list[str] = field(default_factory=list)  # 정하지 못했을 때의 후보 파일 이름

    @property
    def found(self) -> bool:
        return self.status == FOUND


def _norm(text: Any) -> str:
    if is_empty(text):
        return ""
    if isinstance(text, float) and text.is_integer():
        text = int(text)
    return unicodedata.normalize("NFC", str(text)).strip().lower()


def _squash(text: str) -> str:
    return _SPLIT.sub("", text)


def list_photos(folder: str | os.PathLike | None, extensions: Iterable[str] | None = None) -> list[str]:
    """폴더에 든 그림 파일 이름들(하위 폴더 제외)."""
    if not folder:
        return []
    allowed = tuple(e.lower() for e in (extensions or IMAGE_EXTENSIONS))
    names = []
    try:
        with os.scandir(folder) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.lower().endswith(allowed):
                    names.append(entry.name)
    except OSError:
        return []
    return sorted(names)


def folder_has_photos(folder: str | os.PathLike | None) -> bool:
    return bool(list_photos(folder))


def _score(values: Sequence[str], stem: str, tokens: set[str], squashed: str) -> tuple[int, bool]:
    """(점수, 이 사람을 가리킨다고 볼 만한 근거가 있는가)."""
    score, strong = 0, False
    for value, identifying in values:
        if not value:
            continue
        compact = _squash(value)
        if not compact:
            continue
        if value == stem or compact == squashed:
            score += _EXACT
            strong = strong or identifying
        elif compact in tokens:
            score += _TOKEN
            strong = strong or identifying
        elif len(compact) >= 2 and compact in squashed:
            score += _PART
            strong = strong or (identifying and len(compact) >= 3)
    return score, strong


def match_photos(rows: Sequence[Sequence[Any]], file_names: Sequence[str],
                 headers: Sequence[str] | None = None) -> list[PhotoMatch]:
    """rows의 줄마다 사진 한 장을 고른다. file_names는 폴더의 그림 파일 이름들. headers는 칸 이름들.

    - 사람을 가려내는 칸(이름, 사번 등)의 값이 파일 이름에 들어 있어야 후보가 된다.
      부서나 직급만 같은 파일은 후보가 아니다. 칸 이름으로 알 수 없으면 값이 줄마다 대체로 다른 칸을 쓴다.
    - 점수가 가장 높은 파일이 하나면 그 파일로 정한다. 같은 점수가 여럿이면 정하지 않고 후보로 돌려준다.
    - 한 파일을 두 사람이 원하면 점수가 높은 쪽이 갖는다. 같으면 둘 다 정하지 않는다.
    """
    files = []
    for name in file_names:
        stem = _norm(Path(name).stem)
        files.append((name, stem, {t for t in _SPLIT.split(stem) if t}, _squash(stem)))

    width = max((len(row) for row in rows), default=0)
    columns = [[_norm(row[i]) if i < len(row) else "" for row in rows] for i in range(width)]
    identifying = []
    for values in columns:
        filled = [v for v in values if v]
        identifying.append(bool(filled) and len(set(filled)) / len(filled) > 0.8)
    if headers:
        keys = [_squash(_norm(name)) for name in list(headers)[:width]]
        named = [any(word in key for word in PERSON_WORDS) for key in keys] + [False] * (width - len(keys))
        if any(named):
            identifying = named      # 칸 이름으로 알 수 있으면 그것을 믿는다(동명이인이 있어도 이름 칸은 이름 칸이다)

    choices: list[list[tuple[int, str]]] = []
    for index in range(len(rows)):
        values = [(columns[i][index], identifying[i]) for i in range(width)]
        scored = []
        for name, stem, tokens, squashed in files:
            score, strong = _score(values, stem, tokens, squashed)
            if strong:
                scored.append((score, name))
        scored.sort(key=lambda item: (-item[0], item[1]))
        choices.append(scored)

    results = [PhotoMatch() for _ in rows]
    claims: dict[str, list[tuple[int, int]]] = {}
    for index, scored in enumerate(choices):
        if not scored:
            continue
        top = scored[0][0]
        best = [name for score, name in scored if score == top]
        if len(best) == 1:
            claims.setdefault(best[0], []).append((top, index))
        else:
            results[index] = PhotoMatch(AMBIGUOUS, candidates=best)
    for name, wanted in claims.items():
        wanted.sort(reverse=True)
        top = wanted[0][0]
        winners = [index for score, index in wanted if score == top]
        if len(winners) == 1:
            results[winners[0]] = PhotoMatch(FOUND, file_name=name)
        else:
            for index in winners:
                results[index] = PhotoMatch(AMBIGUOUS, candidates=[name])
    return results


def match_folder(rows: Sequence[Sequence[Any]], folder: str | os.PathLike | None,
                 headers: Sequence[str] | None = None) -> list[PhotoMatch]:
    return match_photos(rows, list_photos(folder), headers)
