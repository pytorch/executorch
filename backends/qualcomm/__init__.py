import ctypes
import os
import pathlib

from .scripts.download_qnn_sdk import _download_qnn_sdk, _load_libcxx_libs, SDK_DIR

LLVM_VERSION = "14.0.0"
PKG_ROOT = pathlib.Path(__file__).parent.parent
LIBCXX_DIR = PKG_ROOT / "sdk" / f"libcxx-{LLVM_VERSION}"

# --- Qualcomm SDK handling ---
qnn_root = os.environ.get("QNN_SDK_ROOT")
if qnn_root:
    QNN_SDK_DIR = pathlib.Path(qnn_root)
else:
    QNN_SDK_DIR = PKG_ROOT / "sdk" / "qnn"
    if not QNN_SDK_DIR.exists():
        print("Qualcomm SDK not found. Downloading...")
        _download_qnn_sdk()

os.environ["QNN_SDK_ROOT"] = str(QNN_SDK_DIR)

# Load QNN library
qnn_lib = QNN_SDK_DIR / "lib" / "x86_64-linux-clang" / "libQnnHtp.so"
try:
    ctypes.CDLL(str(qnn_lib), mode=ctypes.RTLD_GLOBAL)
    print(f"Loaded QNN library from {qnn_lib}")
except OSError as e:
    print(f"[ERROR] Failed to load QNN library at {qnn_lib}: {e}")
    raise

# --- libc++ handling ---
if LIBCXX_DIR.exists():
    include_path = LIBCXX_DIR / "include"
    lib_path = LIBCXX_DIR / "lib"

    # Prepend paths to environment
    os.environ["CPLUS_INCLUDE_PATH"] = (
        f"{include_path}:{os.environ.get('CPLUS_INCLUDE_PATH','')}"
    )
    os.environ["LD_LIBRARY_PATH"] = f"{lib_path}:{os.environ.get('LD_LIBRARY_PATH','')}"
    os.environ["LIBRARY_PATH"] = f"{lib_path}:{os.environ.get('LIBRARY_PATH','')}"

    _load_libcxx_libs(lib_path)
else:
    print(f"libc++ not found at {LIBCXX_DIR}, please check installation.")
