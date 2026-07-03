# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for the Crop Studio desktop exe.
# Build with:  .venv\Scripts\pyinstaller.exe --noconfirm CropStudio.spec
from PyInstaller.utils.hooks import collect_all

pyzbar_datas, pyzbar_binaries, pyzbar_hidden = collect_all('pyzbar')

a = Analysis(
    ['cropstudio_desktop.py'],
    pathex=[],
    binaries=pyzbar_binaries,
    datas=[('cropstudio/static', 'cropstudio/static')] + pyzbar_datas,
    hiddenimports=['pyzbar.pyzbar', 'zxingcpp'] + pyzbar_hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['rembg', 'onnxruntime'],   # cropstudio never imports redboxflip.cutout
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='CropStudio',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name='CropStudio',
)
