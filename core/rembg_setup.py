"""rembg를 불러오기 전에 부른다: 쓰지 않는 pymatting(numba)을 빈 모듈로 바꿔 둔다.

rembg.bg는 맨 위에서 pymatting을 불러오는데, pymatting은 불러올 때 numba로 함수 수십 개를 컴파일한다.
배포판(PyInstaller)에서는 numba가 컴파일 결과를 저장하지 못해 켤 때마다 다시 컴파일하고,
이 PC에서 65초가 걸렸다(설치본 로그 '배경제거 AI 준비 완료! (66.3초)').
pymatting은 rembg의 alpha_matting=True에서만 쓰이는데, 하나스튜디오는 그 옵션을 쓰지 않는다.
"""

from __future__ import annotations

import sys
import types

_STUBS = {
    "pymatting.alpha.estimate_alpha_cf": ("estimate_alpha_cf",),
    "pymatting.foreground.estimate_foreground_ml": ("estimate_foreground_ml",),
    "pymatting.util.util": ("stack_images",),
}


def _unused(*_args, **_kwargs):
    raise RuntimeError("하나스튜디오는 rembg의 alpha matting을 쓰지 않습니다(pymatting을 넣지 않았습니다).")


def prepare_rembg() -> None:
    """pymatting이 아직 불려 오지 않았으면 빈 모듈을 끼워 둔다. 여러 번 불러도 된다."""
    if "pymatting" in sys.modules:
        return
    for name in ("pymatting", "pymatting.alpha", "pymatting.foreground", "pymatting.util"):
        sys.modules[name] = types.ModuleType(name)
    for name, functions in _STUBS.items():
        module = types.ModuleType(name)
        for function in functions:
            setattr(module, function, _unused)
        sys.modules[name] = module
