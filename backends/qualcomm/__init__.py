import ctypes
import os
import pathlib

from .scripts.download_qnn_sdk import (
    _download_qnn_sdk,
    _get_libcxx_dir,
    _load_libcxx_libs,
    _stage_libcxx,
)

# # Path to executorch/backends/qualcomm/
# PKG_ROOT = pathlib.Path(__file__).parent.parent / "qualcomm"

# # --- Qualcomm SDK handling ---
# qnn_root = os.environ.get("QNN_SDK_ROOT")
# if qnn_root:
#     QNN_SDK_DIR = pathlib.Path(qnn_root)
# else:
#     QNN_SDK_DIR = PKG_ROOT / "sdk" / "qnn"
#     if not QNN_SDK_DIR.exists():
#         print("Qualcomm SDK not found. Downloading...")
#         _download_qnn_sdk()

# os.environ["QNN_SDK_ROOT"] = str(QNN_SDK_DIR)

# # Load QNN library
# qnn_lib = QNN_SDK_DIR / "lib" / "x86_64-linux-clang" / "libQnnHtp.so"
# try:
#     ctypes.CDLL(str(qnn_lib), mode=ctypes.RTLD_GLOBAL)
#     print(f"Loaded QNN library from {qnn_lib}")
# except OSError as e:
#     print(f"[ERROR] Failed to load QNN library at {qnn_lib}: {e}")
#     raise

# --- libc++ handling ---
# Path to executorch/backends/
# _libcxx_dir = _get_libcxx_dir(PKG_ROOT)

# # Stage (download if missing) and load
# try:
#     _stage_libcxx(_libcxx_dir)
#     _load_libcxx_libs(_libcxx_dir)
# except Exception as e:
#     print(f"[libcxx] Warning: failed to stage/load libc++: {e}")
