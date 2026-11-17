"""배경 지우기는 CPU로만 한다(GPU 연결 파일은 설치본에 묶지 않는다)."""
import sys
import types
from pathlib import Path


def test_model_session_is_created_for_cpu_only(qapp, monkeypatch):
    calls = []
    fake = types.ModuleType("rembg")
    fake.new_session = lambda **kwargs: calls.append(kwargs) or object()
    monkeypatch.setitem(sys.modules, "rembg", fake)
    from core.model_loader import ModelLoadingManager

    loader = ModelLoadingManager()
    loader._load_model_worker()
    assert calls and calls[0]["providers"] == ["CPUExecutionProvider"]


def test_spec_leaves_out_gpu_provider_files():
    spec = (Path(__file__).resolve().parents[2] / "HanaStudio.spec").read_text(encoding="utf-8")
    assert "onnxruntime_providers_cuda" in spec and "a.binaries = [b for b in a.binaries" in spec
