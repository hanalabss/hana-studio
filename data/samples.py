"""예시 명단 만들기: 엑셀 파일 하나와, 사번으로 이름 붙인 사진 폴더 하나.

    files = write_sample("C:/Users/…/Documents")
    files.workbook_path   # …/하나스튜디오 명단 예시.xlsx
    files.photo_dir       # …/사진 예시  (HN260411.jpg …, 한 명은 일부러 없다)

- 사람과 사진은 모두 지어낸 것이다. 사진은 그 자리에서 그린 그림이다(내려받거나 넣어 둔 파일이 없다).
- 이미 같은 이름이 있으면 덮어쓰지 않는다. 엑셀 파일과 사진 폴더에 같은 번호를 붙인다:
  "하나스튜디오 명단 예시 (2).xlsx" + "사진 예시 (2)".
- headers를 주면(카드가 쓰는 칸 이름) 그 이름으로 첫 줄을 만든다. 사진 칸 이름은 열로 만들지 않는다.
  사진은 파일 이름을 사번과 같게 해서 짝을 맞추기 때문이다. 사번처럼 쓸 열이 없으면 '사번'을 덧붙인다.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from .errors import DataError

WORKBOOK_NAME = "하나스튜디오 명단 예시"
PHOTO_DIR_NAME = "사진 예시"
LIST_SHEET = "명단"
GUIDE_SHEET = "사용법"
DEFAULT_HEADERS = ("이름", "부서", "직급", "사번")
ID_HEADER = "사번"
PHOTO_SIZE = (480, 640)

_WRITE_FAILED = "이 폴더에는 예시 파일을 저장할 수 없습니다. '문서' 폴더처럼 저장할 수 있는 다른 폴더를 골라 주세요."


@dataclass(frozen=True)
class SamplePerson:
    name: str
    team: str
    rank: str
    id: str
    joined: str
    born: str
    gender: str


# 모두 지어낸 사람이다.
PEOPLE: tuple[SamplePerson, ...] = (
    SamplePerson("김하늘", "경영지원팀", "사원", "HN260411", "2026-03-02", "1999-04-12", "여"),
    SamplePerson("이바다", "영업1팀", "대리", "HN260412", "2026-03-02", "1994-08-23", "남"),
    SamplePerson("박새벽", "개발팀", "과장", "HN260413", "2026-03-02", "1989-11-05", "남"),
    SamplePerson("최가람", "디자인팀", "사원", "HN260414", "2026-03-09", "2000-01-30", "여"),
    SamplePerson("정나래", "영업2팀", "차장", "HN260415", "2026-03-09", "1985-06-17", "여"),
    SamplePerson("강다온", "개발팀", "사원", "HN260416", "2026-03-09", "1998-09-09", "남"),
    SamplePerson("윤보람", "경영지원팀", "대리", "HN260417", "2026-03-16", "1993-02-14", "여"),
    SamplePerson("한마루", "생산팀", "부장", "HN260418", "2026-03-16", "1981-12-01", "남"),
    SamplePerson("서이슬", "디자인팀", "사원", "HN260419", "2026-03-16", "2001-07-21", "여"),
    SamplePerson("오한결", "생산팀", "과장", "HN260420", "2026-03-23", "1990-05-28", "남"),
)
MISSING_PHOTO_ID = "HN260419"        # 이 사람은 일부러 사진을 만들지 않는다
EXAMPLE_ID = "HN260417"              # 사용법에 적는 예

# (배경, 어깨, 머리)
_PALETTE = (
    ((221, 232, 247), (71, 108, 168), (124, 157, 209)),
    ((250, 231, 214), (176, 108, 62), (214, 152, 108)),
    ((220, 240, 228), (62, 132, 96), (112, 176, 142)),
    ((240, 226, 244), (128, 86, 152), (172, 134, 194)),
    ((250, 240, 208), (160, 126, 44), (206, 174, 96)),
    ((215, 238, 242), (48, 124, 140), (104, 172, 186)),
    ((248, 224, 228), (168, 82, 100), (208, 132, 148)),
    ((228, 230, 236), (88, 96, 116), (140, 148, 168)),
    ((232, 240, 212), (108, 132, 56), (156, 180, 104)),
    ((224, 228, 250), (84, 92, 172), (136, 144, 212)),
)


@dataclass
class SampleFiles:
    directory: Path
    workbook_path: Path
    photo_dir: Path
    headers: list[str]                    # 엑셀 첫 줄에 실제로 적은 이름
    id_header: str                        # 사진 파일 이름과 같은 값이 든 열
    people: int
    missing_photo_id: str
    missing_photo_name: str
    photo_paths: list[Path] = field(default_factory=list)
    photo_headers: list[str] = field(default_factory=list)   # 사진 칸이라 열로 만들지 않은 이름


# ---------------------------------------------------------------- 첫 줄 이름 -> 넣을 값

def _key(name: str) -> str:
    return re.sub(r"[\s_\-./()\[\]:·]+", "", str(name)).lower()


def _has(key: str, *words: str) -> bool:
    return any(word in key for word in words)


def _kind(header: str) -> str:
    key = _key(header)
    if _has(key, "사진", "이미지", "photo", "image", "picture"):
        return "photo"
    if _has(key, "전화", "연락처", "휴대폰", "휴대전화", "핸드폰", "phone", "mobile") or key == "tel":
        return "phone"
    if _has(key, "이메일", "메일", "email", "mail"):
        return "email"
    if _has(key, "입사", "joindate", "hiredate"):
        return "joined"
    if _has(key, "생년월일", "생일", "birth") or key == "dob":
        return "born"
    if _has(key, "발급일", "발행일", "날짜", "일자", "date"):
        return "issued"
    if _has(key, "유효기간", "만료"):
        return "expires"
    if _has(key, "순번", "연번"):
        return "order"
    if _has(key, "회사", "기관", "company"):
        return "company"
    if _has(key, "부서", "소속", "dept", "department", "team") or key.endswith("팀"):
        return "team"
    if _has(key, "직급", "직위", "직책", "position", "jobtitle", "rank") or key == "title":
        return "rank"
    if _has(key, "성별", "gender"):
        return "gender"
    if _has(key, "주소", "address"):
        return "address"
    if _has(key, "사번", "사원번호", "직원번호", "학번", "회원번호", "등록번호", "번호", "코드") \
            or key in ("id", "no") or key.endswith("id") or key.endswith("no"):
        return "id"
    if _has(key, "이름", "성명", "성함", "name"):
        return "name"
    if _has(key, "비고", "메모", "memo", "note"):
        return "note"
    return "other"


def is_id_header(header: str) -> bool:
    """사진 파일 이름과 짝을 맞출 만한 열 이름인지(사번, 학번, 번호 …)."""
    return _kind(header) == "id"


def _value(header: str, kind: str, person: SamplePerson, index: int) -> str:
    tail = person.id[-4:]
    if kind == "name":
        return person.name
    if kind == "team":
        return person.team
    if kind == "rank":
        return person.rank
    if kind == "id":
        return person.id
    if kind == "phone":
        return f"010-0000-{tail}"
    if kind == "email":
        return f"{person.id.lower()}@example.com"
    if kind == "joined":
        return person.joined
    if kind == "born":
        return person.born
    if kind == "issued":
        return "2026-04-01"
    if kind == "expires":
        return "2028-03-31"
    if kind == "order":
        return str(index + 1)
    if kind == "company":
        return "하나상사"
    if kind == "gender":
        return person.gender
    if kind == "address":
        return f"서울시 예시구 예시로 {index + 1}"
    if kind == "note":
        return "신규 발급"
    return f"{header} {index + 1}"


def sample_headers(headers: Iterable[str] | None = None,
                   photo_headers: Iterable[str] = ()) -> tuple[list[str], str, list[str]]:
    """(엑셀 첫 줄 이름들, 사번처럼 쓸 열, 사진 칸이라 뺀 이름들)."""
    photo_keys = {_key(name) for name in photo_headers}
    names: list[str] = []
    dropped: list[str] = []
    for raw in (headers if headers is not None else DEFAULT_HEADERS):
        name = re.sub(r"\s+", " ", str(raw or "")).strip()
        if not name or name in names or name in dropped:
            continue
        if _key(name) in photo_keys or _kind(name) == "photo":
            dropped.append(name)
        else:
            names.append(name)
    if not names:
        names = [n for n in DEFAULT_HEADERS]
    id_header = next((name for name in names if is_id_header(name)), None)
    if id_header is None:
        id_header = ID_HEADER
        names.append(id_header)
    return names, id_header, dropped


def sample_rows(headers: Sequence[str]) -> list[list[str]]:
    kinds = [_kind(name) for name in headers]
    return [[_value(name, kind, person, index) for name, kind in zip(headers, kinds)]
            for index, person in enumerate(PEOPLE)]


# ---------------------------------------------------------------- 파일 이름 (덮어쓰지 않는다)

def free_names(directory: str | Path) -> tuple[Path, Path]:
    """아직 없는 (엑셀 파일, 사진 폴더) 한 쌍. 둘에 같은 번호가 붙는다."""
    directory = Path(directory)
    number = 1
    while True:
        suffix = "" if number == 1 else f" ({number})"
        workbook = directory / f"{WORKBOOK_NAME}{suffix}.xlsx"
        photos = directory / f"{PHOTO_DIR_NAME}{suffix}"
        if not workbook.exists() and not photos.exists():
            return workbook, photos
        number += 1


# ---------------------------------------------------------------- 사진

def draw_portrait(index: int, size: tuple[int, int] = PHOTO_SIZE):
    """사람 모양 그림(머리와 어깨). 실제 사람이 아니다."""
    from PIL import Image, ImageDraw

    background, body, head = _PALETTE[index % len(_PALETTE)]
    scale = 2
    width, height = size[0] * scale, size[1] * scale
    image = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(image)

    def box(left: float, top: float, right: float, bottom: float) -> tuple[int, int, int, int]:
        return round(left * width), round(top * height), round(right * width), round(bottom * height)

    draw.rounded_rectangle(box(0.07, 0.57, 0.93, 1.30), radius=round(0.26 * width), fill=body)   # 어깨
    draw.rounded_rectangle(box(0.42, 0.44, 0.58, 0.68), radius=round(0.05 * width), fill=head)   # 목
    draw.ellipse(box(0.27, 0.16, 0.73, 0.16 + 0.46 * width / height), fill=head)                 # 머리
    return image.resize(size, Image.LANCZOS)


def _write_photos(folder: Path) -> list[Path]:
    folder.mkdir(parents=True)
    paths = []
    for index, person in enumerate(PEOPLE):
        if person.id == MISSING_PHOTO_ID:
            continue
        path = folder / f"{person.id}.jpg"
        draw_portrait(index).save(path, "JPEG", quality=88)
        paths.append(path)
    return paths


# ---------------------------------------------------------------- 엑셀

def _text_width(text: str) -> int:
    return sum(2 if ord(ch) > 0x2E7F else 1 for ch in text)


def guide_lines(id_header: str, photo_dir_name: str, missing_name: str) -> list[str]:
    return [
        "하나스튜디오 명단은 이렇게 만드세요",
        "",
        f"1. 첫 줄에는 항목 이름을 적습니다. (예: {', '.join(DEFAULT_HEADERS)})",
        "2. 둘째 줄부터 한 줄에 한 사람씩 적습니다. 한 줄이 카드 한 장이 됩니다.",
        "3. 칸을 합치지 마세요. 한 칸에는 값 하나만 적습니다.",
        "4. 엑셀 파일(.xlsx)이나 CSV 파일(.csv)로 저장합니다.",
        f"5. 사진 파일 이름을 {id_header}과 같게: {EXAMPLE_ID}.jpg" if _ends_with_consonant(id_header)
        else f"5. 사진 파일 이름을 {id_header}와 같게: {EXAMPLE_ID}.jpg",
        "   사진은 한 폴더에 모아 두고, 가져올 때 그 폴더를 고르면 알아서 짝을 맞춥니다.",
        "",
        f"이 예시에서는 '{LIST_SHEET}' 시트가 명단이고, 사진은 같은 곳의 '{photo_dir_name}' 폴더에 있습니다.",
        f"{missing_name} 님은 일부러 사진을 넣지 않았습니다. 사진이 없는 사람이 프로그램에서 어떻게 보이는지 확인해 보세요.",
        "여기 적힌 사람은 모두 지어낸 예시입니다. 지우고 우리 명단으로 바꿔 쓰셔도 됩니다.",
    ]


def _ends_with_consonant(word: str) -> bool:
    for ch in reversed(word.strip()):
        if "가" <= ch <= "힣":
            return (ord(ch) - 0xAC00) % 28 != 0
        break
    return False


def _write_workbook(path: Path, headers: Sequence[str], rows: Sequence[Sequence[str]],
                    id_header: str, photo_dir_name: str, missing_name: str) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = LIST_SHEET
    fill = PatternFill("solid", fgColor="FFF4D6")
    line = Side(style="thin", color="D9C89A")
    for column, name in enumerate(headers, start=1):
        cell = sheet.cell(row=1, column=column, value=name)
        cell.font = Font(bold=True)
        cell.fill = fill
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = Border(bottom=line)
    for r, row in enumerate(rows, start=2):
        for column, value in enumerate(row, start=1):
            cell = sheet.cell(row=r, column=column, value=str(value))
            cell.number_format = "@"          # 사번·전화번호의 앞자리 0, 긴 숫자가 그대로 남는다
            cell.alignment = Alignment(vertical="center")
    sheet.freeze_panes = "A2"
    sheet.row_dimensions[1].height = 24
    for column, name in enumerate(headers, start=1):
        widest = max([_text_width(name)] + [_text_width(str(row[column - 1])) for row in rows])
        sheet.column_dimensions[get_column_letter(column)].width = min(40, max(10, widest + 4))

    guide = workbook.create_sheet(GUIDE_SHEET)
    guide.column_dimensions["A"].width = 96
    for r, text in enumerate(guide_lines(id_header, photo_dir_name, missing_name), start=1):
        cell = guide.cell(row=r, column=1, value=text or None)
        cell.alignment = Alignment(vertical="center", wrap_text=True)
        if r == 1:
            cell.font = Font(bold=True, size=14)
            guide.row_dimensions[1].height = 28
    workbook.save(path)


# ---------------------------------------------------------------- 공개 함수

def write_sample(directory: str | Path, headers: Iterable[str] | None = None,
                 photo_headers: Iterable[str] = ()) -> SampleFiles:
    """directory에 예시 엑셀 파일과 사진 폴더를 만든다. 이미 있는 것은 덮어쓰지 않는다.

    headers: 첫 줄에 쓸 항목 이름(카드가 쓰는 칸). 없으면 이름/부서/직급/사번.
    photo_headers: headers 가운데 사진 칸인 이름. 열로 만들지 않는다.
    저장할 수 없으면 만들던 것을 치우고 DataError(사용자에게 보여 줄 말)를 낸다.
    """
    directory = Path(directory)
    names, id_header, dropped = sample_headers(headers, photo_headers)
    rows = sample_rows(names)
    missing = next(person for person in PEOPLE if person.id == MISSING_PHOTO_ID)
    workbook_path = photo_dir = None
    try:
        directory.mkdir(parents=True, exist_ok=True)
        workbook_path, photo_dir = free_names(directory)
        photo_paths = _write_photos(photo_dir)
        _write_workbook(workbook_path, names, rows, id_header, photo_dir.name, missing.name)
    except OSError as exc:
        if photo_dir is not None:
            shutil.rmtree(photo_dir, ignore_errors=True)
        if workbook_path is not None:
            try:
                workbook_path.unlink()
            except OSError:
                pass
        raise DataError(_WRITE_FAILED) from exc
    return SampleFiles(
        directory=directory,
        workbook_path=workbook_path,
        photo_dir=photo_dir,
        headers=names,
        id_header=id_header,
        people=len(PEOPLE),
        missing_photo_id=missing.id,
        missing_photo_name=missing.name,
        photo_paths=photo_paths,
        photo_headers=dropped,
    )
