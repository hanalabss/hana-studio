# -*- mode: python ; coding: utf-8 -*-

import sys
import os
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

### ====== 프로젝트 내부 패키지 ======
# main.py에서 아직 임포트하지 않는(또는 함수 안에서 늦게 임포트하는) 모듈도 빠지지 않도록
# 폴더를 훑어 모듈 이름을 모두 넣는다. 임포트하지 않고 파일 이름만 읽는다.
PROJECT_ROOT = SPECPATH
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def local_modules(*packages):
    names = set()
    for package in packages:
        root = os.path.join(PROJECT_ROOT, package)
        for folder, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d != '__pycache__' and not d.startswith('.')]
            dotted = os.path.relpath(folder, PROJECT_ROOT).replace(os.sep, '.')
            for file_name in files:
                if not file_name.endswith('.py'):
                    continue
                names.add(dotted if file_name == '__init__.py' else f"{dotted}.{file_name[:-3]}")
    return sorted(names)


# design(카드 디자인), data(명단), services(인쇄 흐름), printer(RTAI·SMART 드라이버),
# ui(ui.editor / ui.pages.* / ui.shell / ui.theme / ui.widgets / ui.components), utils, core, licensing
project_hidden = local_modules('design', 'data', 'services', 'printer', 'ui', 'utils', 'core', 'licensing')

### ====== 명단 가져오기·내보내기 / 바코드 · QR ======
# openpyxl(엑셀), qrcode, python-barcode는 함수 안에서 늦게 임포트한다.
openpyxl_hidden = collect_submodules('openpyxl') + ['et_xmlfile']
openpyxl_datas = collect_data_files('openpyxl')
qrcode_hidden = collect_submodules('qrcode')
barcode_hidden = collect_submodules('barcode')
barcode_datas = collect_data_files('barcode')        # barcode/fonts/DejaVuSansMono.ttf

### ====== SMART 프린터 SDK (SmartComm2.dll + 색 프로파일) ======
# 실행 파일 기준 _internal/dll/smart 에서 찾는다(printer.smart_printer.default_dll_candidates).
# 아래 ('dll', 'dll')이 폴더째 넣지만, 빠지면 SMART 인쇄가 통째로 안 되므로 따로 적어 둔다.
smart_datas = [('dll/smart', 'dll/smart')] if os.path.isdir(os.path.join(PROJECT_ROOT, 'dll', 'smart')) else []

### ====== NUMBA 필수 리소스 ======
numba_datas = collect_data_files('numba')
numba_hidden = collect_submodules('numba')

### ====== SCIPY C-EXTENSION + DATA 포함 ======
scipy_datas = collect_data_files('scipy')
scipy_hidden = collect_submodules('scipy')

### ====== SUPABASE 전체 의존성 ======
supabase_hidden = collect_submodules('supabase')
gotrue_hidden = collect_submodules('gotrue')
postgrest_hidden = collect_submodules('postgrest')
realtime_hidden = collect_submodules('realtime')
storage3_hidden = collect_submodules('storage3')
supafunc_hidden = collect_submodules('supafunc')
httpx_hidden = collect_submodules('httpx')
httpcore_hidden = collect_submodules('httpcore')

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('config.json', '.'),
        # === 프린터 DLL 및 설정 파일 ===
        ('dll/EWL', '_internal/dll'),
        ('dll/libDSRetransfer600App.dll', '_internal/dll'),
        ('dll/R600StatusReference', '_internal/dll'),
        ('dll/Retransfer600_SDKCfg.xml', '_internal/dll'),
        ('hana.ico', '.'),

        # === 프로젝트 내부 모듈 ===
        ('dll', 'dll'),
        ('models', '_internal/models'),
    ] + smart_datas + openpyxl_datas + barcode_datas + numba_datas + scipy_datas,  # <<< numba + scipy 데이터 포함

    hiddenimports=[
        # === Qt ===
        'PySide6.QtCore', 'PySide6.QtWidgets', 'PySide6.QtGui',

        # === rembg ===
        'rembg', 'rembg.bg', 'rembg.sessions', 'rembg.sessions.base',
        'rembg.sessions.u2net', 'rembg.sessions.u2netp',
        'rembg.sessions.u2net_human_seg', 'rembg.sessions.u2net_cloth_seg',
        'rembg.sessions.silueta', 'rembg.sessions.isnet_general_use',
        'rembg.sessions.isnet_anime', 'rembg.sessions.sam',

        # === onnxruntime ===
        'onnxruntime', 'onnxruntime.capi',
        'onnxruntime.capi._pybind_state',
        'onnxruntime.capi.onnxruntime_pybind11_state',

        # === pymatting ===
        'pymatting', 'pymatting.util.util',

        # === printer code ===
        'printer.r600_printer',

        # === 명단 파일(SQLite) · 엑셀 ===
        'sqlite3', 'csv', 'openpyxl', 'et_xmlfile',

        # === 바코드 · QR ===
        'qrcode', 'barcode',

        # === pooch (모델 다운로드 핸들링) ===
        'pooch', 'pooch.core', 'pooch.utils',

        # === jaraco (pooch 의존성) ===
        'jaraco', 'jaraco.text', 'jaraco.functools', 'jaraco.context',

        # === 기타 필요한 scipymodule ===
        'scipy.sparse.csgraph._validation',
        'scipy._lib._ccallback_c',
        'scipy._lib._testutils',
        'scipy.linalg.cython_blas',
        'scipy.linalg.cython_lapack',
        'scipy.spatial.transform.rotation',

        # === Numba 꼭 포함해야 하는 모듈 ===
        'numba.core', 'numba.core.types', 'numba.core.types.old_scalars',
        'numba.np.ufunc',

        # === 라이선스 모듈 ===
        'licensing', 'licensing.manager', 'licensing.dialog',
        'licensing.device', 'licensing.config',

        # === supabase (라이선스 서버 연동) ===
        'supabase', 'supabase._sync.client', 'supabase._async.client',
        'gotrue', 'postgrest', 'realtime', 'storage3', 'supafunc',
        'httpx', 'httpcore', 'h11', 'sniffio', 'anyio',
    ] + project_hidden + openpyxl_hidden + qrcode_hidden + barcode_hidden + numba_hidden + scipy_hidden + supabase_hidden + gotrue_hidden + postgrest_hidden + realtime_hidden + storage3_hidden + supafunc_hidden + httpx_hidden + httpcore_hidden,  # <<< 자동 포함된 모듈 추가

    hookspath=[],
    hooksconfig={},
    runtime_hooks=['pyi_rth_pkgutil_fix.py'],
    excludes=[
        # 🚫 불필요한 큰 라이브러리 제외
        'tkinter', 'matplotlib', 'pandas', 'jupyter', 'IPython', 'pytest',
        'pip', 'torch', 'tensorflow', 'transformers', 'flask', 'fastapi',
        'uvicorn', 'starlette', 'selenium', 'webdriver_manager', 'datasets',
        'accelerate', 'huggingface_hub', 'scikit-image', 'scikit-learn',
        'black', 'coverage', 'pygments'
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='HanaStudio',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    icon=['hana.ico'],
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='HanaStudio',
)
