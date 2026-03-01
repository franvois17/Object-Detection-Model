# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Detector de Inventario.

Build command (from repo root):
    pyinstaller detector_inventario.spec --clean
"""

from PyInstaller.utils.hooks import collect_all, collect_data_files

# ── Collect everything needed from heavy packages ────────────────────────────
ultralytics_datas, ultralytics_binaries, ultralytics_hiddenimports = collect_all('ultralytics')
albumentations_datas, albumentations_binaries, albumentations_hiddenimports = collect_all('albumentations')
pyside6_datas, pyside6_binaries, pyside6_hiddenimports = collect_all('PySide6')

# ── Analysis ─────────────────────────────────────────────────────────────────
a = Analysis(
    ['src/app/main.py'],
    pathex=['.'],
    binaries=ultralytics_binaries + albumentations_binaries + pyside6_binaries,
    datas=ultralytics_datas + albumentations_datas + pyside6_datas,
    hiddenimports=[
        # PySide6
        *pyside6_hiddenimports,
        'PySide6.QtCore',
        'PySide6.QtGui',
        'PySide6.QtWidgets',
        'PySide6.QtMultimedia',
        'PySide6.QtMultimediaWidgets',
        # PyTorch & torchvision
        'torch',
        'torch.nn',
        'torch.nn.functional',
        'torchvision',
        'torchvision.models',
        'torchvision.transforms',
        # Ultralytics / YOLO
        *ultralytics_hiddenimports,
        # Albumentations
        *albumentations_hiddenimports,
        # OpenCV
        'cv2',
        # FAISS
        'faiss',
        # SQLAlchemy
        'sqlalchemy',
        'sqlalchemy.dialects.sqlite',
        'sqlalchemy.orm',
        # Other
        'PIL',
        'PIL.Image',
        'tqdm',
        'numpy',
        # App source packages
        'src',
        'src.app',
        'src.core',
        'src.data',
        'src.ml',
        'src.video',
        'src.workers',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude unused heavy packages to reduce bundle size
        'tkinter',
        '_tkinter',
        'matplotlib',
        'jupyter',
        'notebook',
        'IPython',
        'scipy',
        'pandas',
        'sklearn',
        'skimage',
        'tensorflow',
        'keras',
        'jax',
        'pytest',
        'setuptools',
        'distutils',
    ],
    noarchive=False,
    optimize=1,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='DetectorInventario',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,          # GUI app — no terminal window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/icon.ico',
    version_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='DetectorInventario',
)
