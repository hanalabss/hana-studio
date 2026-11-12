"""core.image_processor.extract_alpha: 실제 모델(rembg/onnx)은 불러오지 않고 가짜 rembg로 확인한다."""

import io
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = str(Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import core.image_processor as image_processor  # noqa: E402
from core.image_processor import ImageProcessor  # noqa: E402


@pytest.fixture
def photo(tmp_path) -> str:
    path = tmp_path / "photo.png"
    Image.new("RGB", (8, 6), (40, 90, 200)).save(path)
    return str(path)


@pytest.fixture
def fake_rembg(monkeypatch):
    """알파가 0 / 100 / 255인 6×8 결과를 돌려주는 가짜 rembg. 부른 내용을 기록한다."""
    calls = []

    def remove(data, session=None):
        calls.append({"bytes": len(data), "session": session})
        result = Image.new("RGBA", (8, 6), (10, 20, 30, 0))
        result.putpixel((3, 3), (255, 255, 255, 255))
        result.putpixel((4, 3), (255, 255, 255, 100))
        buffer = io.BytesIO()
        result.save(buffer, "PNG")
        return buffer.getvalue()

    monkeypatch.setitem(sys.modules, "rembg", types.SimpleNamespace(remove=remove))
    return calls


def test_extract_alpha_returns_the_soft_matte(photo, fake_rembg):
    alpha = ImageProcessor().extract_alpha(photo, session="SESSION")
    assert isinstance(alpha, np.ndarray) and alpha.dtype == np.uint8
    assert alpha.shape == (6, 8)                                # (높이, 너비)
    assert alpha[3, 3] == 255 and alpha[3, 4] == 100 and alpha[0, 0] == 0   # 임계값을 적용하지 않은 그대로
    assert fake_rembg == [{"bytes": fake_rembg[0]["bytes"], "session": "SESSION"}]
    assert fake_rembg[0]["bytes"] > 0


def test_one_alpha_serves_any_threshold_without_running_the_model_again(photo, fake_rembg):
    alpha = ImageProcessor().extract_alpha(photo, session="SESSION")
    kept = {threshold: int(np.count_nonzero(alpha > threshold)) for threshold in (50, 128, 200)}
    assert kept == {50: 2, 128: 1, 200: 1}
    assert len(fake_rembg) == 1


def test_extract_alpha_falls_back_to_the_loaded_session(photo, fake_rembg, monkeypatch):
    monkeypatch.setattr(image_processor, "get_ai_session", lambda: "LOADED")
    ImageProcessor().extract_alpha(photo)
    assert fake_rembg[0]["session"] == "LOADED"


def test_extract_alpha_without_any_session_is_refused(photo, fake_rembg, monkeypatch):
    monkeypatch.setattr(image_processor, "get_ai_session", lambda: None)
    with pytest.raises(ValueError):
        ImageProcessor().extract_alpha(photo)
    assert fake_rembg == []


def test_extract_alpha_wraps_failures(tmp_path, fake_rembg):
    with pytest.raises(RuntimeError):
        ImageProcessor().extract_alpha(str(tmp_path / "없는 파일.png"), session="SESSION")
    assert fake_rembg == []


def test_legacy_mask_is_unchanged_and_agrees_with_the_alpha(photo, fake_rembg):
    """기존 remove_background는 그대로다: 흰색(255) = 배경, 3채널."""
    processor = ImageProcessor()
    mask = processor.remove_background(photo, "SESSION", alpha_threshold=128)
    alpha = processor.extract_alpha(photo, session="SESSION")
    assert mask.shape == (6, 8, 3) and mask.dtype == np.uint8
    assert np.array_equal(mask[..., 0], np.where(alpha > 128, 0, 255))
    assert mask[3, 3, 0] == 0 and mask[3, 4, 0] == 255 and mask[0, 0, 0] == 255


def test_importing_the_processor_does_not_load_the_model():
    import subprocess

    code = (
        "import sys, core.image_processor; "
        "loaded = {'rembg', 'onnxruntime'} & set(sys.modules); "
        "sys.exit(1 if loaded else 0)"
    )
    assert subprocess.run([sys.executable, "-c", code], cwd=ROOT, timeout=120).returncode == 0
