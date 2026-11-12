""".hsd 파일: ZIP(document.json + assets/ + thumbnail.png)."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from PIL import Image

from design.model import Document, ImageItem
from design.presets import CUT_54X86, CUT_54X86_BLEED, CardPreset

EXTENSION = ".hsd"
DOCUMENT_ENTRY = "document.json"
THUMBNAIL_ENTRY = "thumbnail.png"
ASSETS_DIR = "assets"
THUMBNAIL_WIDTH_PX = 360

_MAX_ENTRIES = 5000
_MAX_TOTAL_BYTES = 1024 * 1024 * 1024
_MAX_DOCUMENT_BYTES = 32 * 1024 * 1024
_RATIO_TOLERANCE = 0.005


class StorageError(Exception):
    """사용자에게 그대로 보여줄 수 있는 한국어 메시지를 담는다."""


def new_workdir() -> str:
    """새 문서가 쓸 작업 폴더. `document.base_dir`에 넣어 두고 닫을 때 `cleanup_workdir`로 지운다."""
    workdir = tempfile.mkdtemp(prefix="hsd_")
    os.makedirs(os.path.join(workdir, ASSETS_DIR), exist_ok=True)
    return workdir


def cleanup_workdir(document_or_path) -> None:
    path = getattr(document_or_path, "base_dir", document_or_path)
    if path and os.path.isdir(path) and os.path.basename(os.path.normpath(path)).startswith("hsd_"):
        shutil.rmtree(path, ignore_errors=True)


def _file_hash(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def import_asset(document_workdir: str | os.PathLike, file: str | os.PathLike) -> str:
    """파일을 작업 폴더의 assets/로 복사하고 상대 경로를 돌려준다. 내용이 같으면 한 번만 둔다."""
    source = os.fspath(file)
    if not os.path.isfile(source):
        raise StorageError(f"파일을 찾을 수 없습니다: {source}")
    extension = os.path.splitext(source)[1].lower()
    if not extension.replace(".", "").isalnum():
        extension = ""
    name = f"{_file_hash(source)[:20]}{extension}"
    assets = os.path.join(os.fspath(document_workdir), ASSETS_DIR)
    os.makedirs(assets, exist_ok=True)
    target = os.path.join(assets, name)
    if not os.path.exists(target):
        temporary = target + ".part"
        shutil.copyfile(source, temporary)
        os.replace(temporary, target)
    return f"{ASSETS_DIR}/{name}"


def _safe_member(name: str) -> PurePosixPath:
    """ZIP 항목 이름이 작업 폴더 밖을 가리키면 거부한다."""
    if "\\" in name or "\x00" in name or ":" in name or name.startswith("/"):
        raise StorageError(f"파일 안에 안전하지 않은 경로가 있어 열 수 없습니다: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in ("..", "") for part in path.parts):
        raise StorageError(f"파일 안에 안전하지 않은 경로가 있어 열 수 없습니다: {name!r}")
    return path


def _make_thumbnail(document: Document) -> bytes | None:
    try:
        from design.renderer import render_side

        width_mm, _ = document.side_size_mm("front")
        dpi = THUMBNAIL_WIDTH_PX / width_mm * 25.4
        image = render_side(document, "front", None, dpi=dpi).color
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()
    except Exception:
        # 미리보기 그림을 못 만들어도 문서 저장은 계속한다.
        return None


def save(document: Document, path: str | os.PathLike, *, thumbnail: bool = True) -> str:
    """문서를 .hsd로 저장한다. 같은 폴더의 임시 파일에 쓴 뒤 바꿔치기해 중간에 실패해도 원본이 남는다."""
    target = os.fspath(path)
    payload = json.dumps(document.to_dict(), ensure_ascii=False, indent=2).encode("utf-8")

    assets: list[tuple[str, str]] = []
    assets_root = os.path.join(document.base_dir, ASSETS_DIR) if document.base_dir else None
    if assets_root and os.path.isdir(assets_root):
        for folder, _, files in os.walk(assets_root):
            for file_name in sorted(files):
                if file_name.endswith(".part"):
                    continue
                full = os.path.join(folder, file_name)
                relative = os.path.relpath(full, document.base_dir).replace(os.sep, "/")
                assets.append((relative, full))
    present = {relative for relative, _ in assets}
    missing = sorted(ref for ref in document.asset_refs() if not os.path.isabs(ref) and ref not in present)
    if missing:
        raise StorageError("문서가 쓰는 이미지 파일이 없어 저장할 수 없습니다: " + ", ".join(missing))

    thumbnail_bytes = _make_thumbnail(document) if thumbnail else None

    directory = os.path.dirname(os.path.abspath(target))
    os.makedirs(directory, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=".hsd_", suffix=".tmp", dir=directory)
    os.close(handle)
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(DOCUMENT_ENTRY, payload)
            for relative, full in sorted(assets):
                archive.write(full, relative, compress_type=zipfile.ZIP_STORED)
            if thumbnail_bytes:
                archive.writestr(THUMBNAIL_ENTRY, thumbnail_bytes)
        os.replace(temporary, target)
    except OSError as exc:
        raise StorageError(f"파일을 저장하지 못했습니다: {exc}") from exc
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return target


def _open_archive(path: str) -> zipfile.ZipFile:
    if not os.path.isfile(path):
        raise StorageError(f"파일을 찾을 수 없습니다: {path}")
    try:
        return zipfile.ZipFile(path, "r")
    except (zipfile.BadZipFile, OSError) as exc:
        raise StorageError(f"하나스튜디오 디자인 파일이 아니거나 손상되었습니다: {path}") from exc


def load(path: str | os.PathLike, workdir: str | os.PathLike | None = None) -> Document:
    """.hsd를 연다. assets/는 workdir(없으면 새 임시 폴더)에 풀리고 `document.base_dir`가 그곳을 가리킨다.

    더 새 버전에서 저장한 문서도 열며, 모르는 필드는 보존된다(`document.is_newer_format`으로 확인).
    """
    source = os.fspath(path)
    created = workdir is None
    with _open_archive(source) as archive:
        entries = archive.infolist()
        if len(entries) > _MAX_ENTRIES:
            raise StorageError("파일 안의 항목이 너무 많아 열 수 없습니다.")
        if sum(entry.file_size for entry in entries) > _MAX_TOTAL_BYTES:
            raise StorageError("파일이 너무 커서 열 수 없습니다.")
        members = {entry.filename: _safe_member(entry.filename) for entry in entries if not entry.is_dir()}
        if DOCUMENT_ENTRY not in members:
            raise StorageError("하나스튜디오 디자인 파일이 아닙니다. (document.json 없음)")
        if archive.getinfo(DOCUMENT_ENTRY).file_size > _MAX_DOCUMENT_BYTES:
            raise StorageError("문서 정보가 너무 커서 열 수 없습니다.")
        try:
            data = json.loads(archive.read(DOCUMENT_ENTRY).decode("utf-8"))
            document = Document.from_dict(data)
        except (ValueError, TypeError, AttributeError, KeyError) as exc:
            raise StorageError(f"문서 내용을 읽을 수 없습니다: {exc}") from exc

        root = Path(tempfile.mkdtemp(prefix="hsd_") if created else os.fspath(workdir)).resolve()
        try:
            (root / ASSETS_DIR).mkdir(parents=True, exist_ok=True)
            for name, member in members.items():
                if member.parts[0] != ASSETS_DIR or len(member.parts) < 2:
                    continue
                destination = (root / Path(*member.parts)).resolve()
                if root not in destination.parents:
                    raise StorageError(f"파일 안에 안전하지 않은 경로가 있어 열 수 없습니다: {name!r}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(name) as src, open(destination, "wb") as dst:
                    shutil.copyfileobj(src, dst)
        except (StorageError, OSError, zipfile.BadZipFile) as exc:
            if created:
                shutil.rmtree(root, ignore_errors=True)
            if isinstance(exc, StorageError):
                raise
            raise StorageError(f"파일을 여는 중 오류가 났습니다: {exc}") from exc
    document.base_dir = str(root)
    return document


def read_thumbnail(path: str | os.PathLike) -> bytes | None:
    """문서를 열지 않고 미리보기 PNG만 읽는다(파일 목록 화면용). 없으면 None."""
    try:
        with _open_archive(os.fspath(path)) as archive:
            if THUMBNAIL_ENTRY not in archive.namelist():
                return None
            if archive.getinfo(THUMBNAIL_ENTRY).file_size > _MAX_DOCUMENT_BYTES:
                return None
            return archive.read(THUMBNAIL_ENTRY)
    except (StorageError, zipfile.BadZipFile, OSError):
        return None


def _ratio_matches(ratio: float, target: float) -> bool:
    return abs(ratio - target) / target <= _RATIO_TOLERANCE


def document_from_single_image(
    path: str | os.PathLike,
    *,
    back_path: str | os.PathLike | None = None,
    preset: CardPreset | None = None,
    fit: str = "stretch",
    mask_path: str | os.PathLike | None = None,
    workdir: str | os.PathLike | None = None,
    name: str | None = None,
) -> Document:
    """기존 "면마다 이미지 한 장" 방식을 ImageItem 하나짜리 문서로 바꾼다.

    preset이 None이면 기존 규칙을 따른다: 이미지 비율이 도련 포함 규격(58×90)과 같으면
    도련 2mm 문서, 아니면 54×86 문서. 방향은 이미지의 가로·세로 비로 정한다.
    """
    front = os.fspath(path)
    try:
        with Image.open(front) as probe:
            width, height = probe.size
    except (OSError, ValueError) as exc:
        raise StorageError(f"이미지를 열 수 없습니다: {front}") from exc

    if preset is None:
        ratio = min(width, height) / max(width, height)
        bleed = CUT_54X86_BLEED
        preset = bleed if _ratio_matches(ratio, bleed.height_mm / bleed.width_mm) else CUT_54X86
    orientation = "landscape" if width >= height else "portrait"

    base_dir = os.fspath(workdir) if workdir is not None else new_workdir()
    document = Document(
        name=name or os.path.splitext(os.path.basename(front))[0],
        card_width_mm=preset.width_mm, card_height_mm=preset.height_mm,
        bleed_mm=preset.bleed_mm, dual_side=back_path is not None, base_dir=base_dir,
    )
    for side_key, image_path, mask in (("front", front, mask_path), ("back", back_path, None)):
        side = document.side(side_key)
        side.orientation = orientation
        if image_path is None:
            continue
        side_w, side_h = document.side_size_mm(side_key)
        side.add_item(ImageItem(
            name="전체 이미지", x=0.0, y=0.0, w=side_w, h=side_h, fit=fit,
            asset=import_asset(base_dir, image_path),
            mask_asset=import_asset(base_dir, mask) if mask is not None else None,
        ))
    return document
