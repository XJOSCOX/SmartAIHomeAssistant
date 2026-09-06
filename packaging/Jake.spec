# Onedir Windows build. Only installed runtime package data is collected.
# Never add repository directories, config/local.toml, models or biometric stores.
import os
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules, copy_metadata

os.environ["YOLO_OFFLINE"] = "true"
os.environ["YOLO_AUTOINSTALL"] = "false"
# Do not resolve Qt DLLs from unrelated tools (e.g. Conda/Poppler ICU on PATH).
# Package hooks add their own native-library directories explicitly.
windows = Path(os.environ["SystemRoot"])
os.environ["PATH"] = os.pathsep.join((str(Path(sys.executable).parent),
                                      str(windows / "System32"), str(windows)))
root = Path(SPECPATH).parent
hidden = collect_submodules("jake.adapters") + [
    "ultralytics", "openvino", "keyring.backends.Windows", "tomli_w", "tzdata",
]
data = collect_data_files("ultralytics", includes=["cfg/**/*.yaml"])
# OpenVINO discovers its IR reader and CPU device plugin dynamically. Import
# analysis only finds openvino.dll and Python frontend DLLs, leaving model
# reading/CPU compilation broken even though importing openvino succeeds.
binaries = collect_dynamic_libs("openvino")
# Torchvision 0.29 uses _C_stable.pyd (loaded via torch.ops.load_library).
# Older bundled hooks only request torchvision._C and silently miss NMS.
binaries += collect_dynamic_libs("torchvision", search_patterns=["*.pyd", "*.dll"])
for package in ("ultralytics", "openvino", "torch", "jake-home-assistant"):
    data += copy_metadata(package)
a = Analysis([str(root / "packaging" / "desktop_entry.py")],
             pathex=[str(root / "src")], binaries=binaries, datas=data, hiddenimports=hidden,
             hookspath=[], runtime_hooks=[],
             excludes=["PyQt5", "PyQt6", "PySide2", "tkinter", "IPython", "pytest"],
             noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="Jake", debug=False,
          bootloader_ignore_signals=False, strip=False, upx=False, console=os.environ.get("JAKE_BUILD_CONSOLE") == "1",
          disable_windowed_traceback=True)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="Jake")
