# -*- mode: python ; coding: utf-8 -*-
# ─────────────────────────────────────────────────────────────────────────────
# NeuroBench Studio — PyInstaller Spec
# ─────────────────────────────────────────────────────────────────────────────
# Build:
#   python -m PyInstaller neurobench.spec
#
# Strategy: torch is EXCLUDED from the bundle to keep size manageable.
# The app performs lazy imports of torch only when a pipeline is executed,
# and shows a user-friendly prompt to install torch if it is absent.
# ─────────────────────────────────────────────────────────────────────────────

import sys
import os
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(SPECPATH)   # Project root (where this .spec lives)

# ── Icon — conditional per platform ───────────────────────────────────────────────────
ICNS_PATH = ROOT / 'build' / 'assets' / 'icon.icns'
ICO_PATH  = ROOT / 'build' / 'assets' / 'icon.ico'

if sys.platform == 'darwin':
    # macOS only needs .icns
    ICON = str(ICNS_PATH) if ICNS_PATH.exists() else None
elif sys.platform == 'win32':
    # Windows only needs .ico
    ICON = str(ICO_PATH) if ICO_PATH.exists() else None
else:
    # Linux — no icon for EXE, icon goes in the AppDir instead
    ICON = None

# ── Data files to bundle ─────────────────────────────────────────────────────
# Format: (source_glob_or_dir, destination_inside_bundle)
datas = [
    # Flask templates
    (str(ROOT / 'src' / 'dashboard' / 'templates'), 'src/dashboard/templates'),
    # Static assets (CSS, JS, images)
    (str(ROOT / 'src' / 'dashboard' / 'static'),    'src/dashboard/static'),
    # Sample configs (read-only reference copies)
    (str(ROOT / 'configs'),                          'configs'),
    # MNE data files (channel locations, etc.)
    *collect_data_files('mne'),
    # Braindecode data files
    *collect_data_files('braindecode'),
    # scikit-learn data files
    *collect_data_files('sklearn'),
]

# ── Hidden imports ────────────────────────────────────────────────────────────
# Modules that PyInstaller's static analyser misses (dynamic imports, plugins)
hidden_imports = [
    # Flask internals
    'flask',
    'flask.templating',
    'jinja2',
    'werkzeug',
    'werkzeug.serving',
    'werkzeug.debug',
    # Numerical
    'numpy',
    'pandas',
    'scipy',
    'scipy.signal',
    'scipy.linalg',
    'scipy.stats',
    'sklearn',
    'sklearn.utils._cython_blas',
    'sklearn.neighbors._partition_nodes',
    'sklearn.utils._typedefs',
    # EEG
    'mne',
    'mne.datasets',
    # pywebview backends — include all, pywebview picks the best at runtime
    'webview',
    'webview.platforms.cocoa',    # macOS
    'webview.platforms.winforms', # Windows
    'webview.platforms.gtk',      # Linux
    'webview.platforms.qt',       # Qt fallback
    # Plotting (non-interactive backend for frozen app)
    'matplotlib',
    'matplotlib.backends.backend_agg',
    'plotly',
    'plotly.graph_objects',
    # Stats
    'statsmodels',
    'pingouin',
    # Misc — note: the importable name is 'yaml', not 'pyyaml'
    'yaml',
    'tkinter',
    'tkinter.messagebox',
    'einops',
] + collect_submodules('mne')

# ── Excluded modules (keep bundle small) ─────────────────────────────────────
# torch and torchvision are loaded lazily at runtime when the user runs a
# deep-learning pipeline. If torch is not found, a helpful install message
# is displayed in the UI.
excludes = [
    'mlflow',           # optional experiment tracking — not needed for core UI
    'streamlit',        # legacy; Flask dashboard replaces it
    'IPython',
    'ipykernel',
    'notebook',
    'jupyter',
    'test',
    'tests',
    'pdb',
]

# ─────────────────────────────────────────────────────────────────────────────
# Analysis
# ─────────────────────────────────────────────────────────────────────────────

a = Analysis(
    [str(ROOT / 'main.py')],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[str(ROOT / 'build' / 'hooks')],
    hooksconfig={},
    runtime_hooks=[str(ROOT / 'build' / 'runtime_hook.py')],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# ─────────────────────────────────────────────────────────────────────────────
# PYZ archive of pure-Python bytecode
# ─────────────────────────────────────────────────────────────────────────────

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# ─────────────────────────────────────────────────────────────────────────────
# Executable
# ─────────────────────────────────────────────────────────────────────────────

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,   # use COLLECT (folder mode) for smaller builds
    name='NeuroBenchStudio',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,                # compress binaries if UPX is installed
    console=False,           # no terminal window on Windows/Mac
    disable_windowed_traceback=False,
    argv_emulation=True,     # macOS: handle open-file events
    target_arch=None,        # auto-detect (arm64 on Apple Silicon, x86_64 on Intel)
    codesign_identity=None,  # set to your Apple Developer ID for signing
    entitlements_file=None,
    icon=ICON,               # platform-appropriate icon, or None if missing
)

# ─────────────────────────────────────────────────────────────────────────────
# COLLECT — output folder with all binaries and data
# ─────────────────────────────────────────────────────────────────────────────

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='NeuroBenchStudio',
)

# ─────────────────────────────────────────────────────────────────────────────
# macOS App Bundle (.app)
# ─────────────────────────────────────────────────────────────────────────────

app = BUNDLE(
    coll,
    name='NeuroBench Studio.app',
    icon=str(ICNS_PATH) if ICNS_PATH.exists() else None,
    bundle_identifier='io.neurobench.studio',
    info_plist={
        'CFBundleName': 'NeuroBench Studio',
        'CFBundleDisplayName': 'NeuroBench Studio',
        'CFBundleVersion': '2.0.0',
        'CFBundleShortVersionString': '2.0.0',
        'NSHighResolutionCapable': True,
        'NSRequiresAquaSystemAppearance': False,  # support Dark Mode
        'LSMinimumSystemVersion': '11.0',
        'NSHumanReadableCopyright': 'Copyright 2025 NeuroBench Studio',
    },
)
