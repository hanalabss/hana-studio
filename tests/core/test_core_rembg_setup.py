"""rembg를 불러와도 쓰지 않는 pymatting(numba 컴파일, 배포판 65초)을 불러오지 않는다."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_rembg_loads_without_pymatting_or_numba():
    code = (
        "import sys; sys.modules['numba'] = None; sys.modules['llvmlite'] = None\n"
        "from core.rembg_setup import prepare_rembg; prepare_rembg()\n"
        "import rembg.bg\n"
        "assert not hasattr(sys.modules['pymatting'], '__file__'), '진짜 pymatting을 불러왔다'\n"
        "try:\n"
        "    sys.modules['pymatting.util.util'].stack_images(None)\n"
        "except RuntimeError as exc:\n"
        "    assert 'alpha matting' in str(exc)\n"
        "else:\n"
        "    raise AssertionError('빈 함수가 조용히 지나갔다')\n"
        "print('ok')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0 and "ok" in result.stdout, result.stderr[-2000:]


def test_prepare_twice_and_real_pymatting_is_left_alone(monkeypatch):
    import types

    from core.rembg_setup import prepare_rembg

    real = types.ModuleType("pymatting")
    monkeypatch.setitem(sys.modules, "pymatting", real)
    prepare_rembg()
    prepare_rembg()
    assert sys.modules["pymatting"] is real          # 이미 불린 진짜 모듈은 바꾸지 않는다


def test_spec_keeps_numba_out_of_the_build():
    spec = (ROOT / "HanaStudio.spec").read_text(encoding="utf-8")
    assert "'numba', 'llvmlite', 'pymatting'," in spec and "collect_submodules('numba')" not in spec


def test_model_file_is_validated_once(tmp_path, monkeypatch):
    import core.model_downloader as downloader

    model = tmp_path / "m.onnx"
    model.write_bytes(b"x")
    made = []

    class FakeOrt:
        @staticmethod
        def InferenceSession(path, providers):
            made.append(path)
            return object()

    monkeypatch.setitem(sys.modules, "onnxruntime", FakeOrt)
    monkeypatch.setattr(downloader, "_VALIDATED", set())
    assert downloader.validate_model_file(model) and downloader.validate_model_file(model)
    assert len(made) == 1
