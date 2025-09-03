import ctypes
import os
import pathlib
import platform
import shutil
import tarfile
import tempfile
import urllib.request
import zipfile
from typing import Optional

from tqdm import tqdm

# === executorch/backends/qualcomm path ===
PKG_ROOT = pathlib.Path(__file__).parent.parent
QNN_SDK_PATH = PKG_ROOT / "sdk" / "qnn"
libcxx_DIR_PATH


def is_linux_x86() -> bool:
    """Return True if running on Linux x86/x86_64."""
    return platform.system().lower() == "linux" and platform.machine().lower() in (
        "x86_64",
        "amd64",
        "i386",
        "i686",
    )


# ========================
# === QNN SDK handling ===
# ========================


def _get_qnn_sdk_dir() -> pathlib.Path:
    # === executorch/backends/qualcomm/sdk/qnn path ===
    return PKG_ROOT / "sdk" / "qnn"


def _download_qnn_sdk() -> Optional[pathlib.Path]:
    """Download and extract the Qualcomm SDK into SDK_DIR (Linux x86 only)."""
    if not is_linux_x86():
        print("Skipping Qualcomm SDK (only supported on Linux x86).")
        return None

    # Already exists? Skip
    if QNN_SDK_PATH.exists() and any(QNN_SDK_PATH.iterdir()):
        print(f"Qualcomm SDK already present at {QNN_SDK_PATH}, skipping download.")
        return QNN_SDK_PATH

    QNN_SDK_PATH.mkdir(parents=True, exist_ok=True)
    QAIRT_URL = "https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/2.34.0.250424/v2.34.0.250424.zip"
    QAIRT_CONTENT_DIR = "qairt/2.34.0.250424"

    with tempfile.TemporaryDirectory() as tmpdir:
        archive_path = pathlib.Path(tmpdir) / pathlib.Path(QAIRT_URL).name

        try:
            print(f"Downloading Qualcomm SDK from {QAIRT_URL}...")
            urllib.request.urlretrieve(QAIRT_URL, archive_path)
        except Exception as e:
            print(f"Error during download: {e}")
            return None

        if not archive_path.exists() or archive_path.stat().st_size == 0:
            print("ERROR: Download failed or file empty.")
            return None

        # Extract
        if QAIRT_URL.endswith(".zip"):
            _extract_zip(archive_path, QAIRT_CONTENT_DIR, QNN_SDK_PATH)
        elif QAIRT_URL.endswith((".tar.gz", ".tgz")):
            _extract_tar(archive_path, QAIRT_CONTENT_DIR, QNN_SDK_PATH)
        else:
            raise ValueError(f"Unsupported archive format: {QAIRT_URL}")

    print(f"Qualcomm SDK extracted to {QNN_SDK_PATH}")
    return QNN_SDK_PATH


# === End of QNN SDK handling ===


def _extract_zip(archive_path, content_dir, target_dir):
    print(f"Extracting {archive_path} to {target_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path, "r") as zip_ref:
        files_to_extract = [f for f in zip_ref.namelist() if f.startswith(content_dir)]
        for file in tqdm(files_to_extract, desc="Extracting files"):
            relpath = os.path.relpath(file, content_dir)
            if relpath == ".":
                continue
            out_path = target_dir / relpath
            if file.endswith("/"):
                out_path.mkdir(parents=True, exist_ok=True)
            else:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with zip_ref.open(file) as src, open(out_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)


def _extract_tar(archive_path: pathlib.Path, prefix: str, target_dir: pathlib.Path):
    """Extract files from a tar.gz archive into target_dir, stripping prefix."""
    with tarfile.open(archive_path, "r:gz") as tf:
        for m in tf.getmembers():
            if not m.name.startswith(prefix + "/"):
                continue
            relpath = pathlib.Path(m.name).relative_to(prefix)
            if not relpath.parts or relpath.parts[0] == "..":
                continue
            out_path = target_dir / relpath
            if m.isdir():
                out_path.mkdir(parents=True, exist_ok=True)
            else:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                src = tf.extractfile(m)
                if src:
                    with src, open(out_path, "wb") as dst:
                        dst.write(src.read())


# =========================
# === libcxx handling ====
# =========================

LLVM_VERSION = "14.0.0"


def _get_libcxx_dir() -> pathlib.Path:
    return PKG_ROOT / "sdk" / f"libcxx-{LLVM_VERSION}"


def _stage_libcxx(target_dir: pathlib.Path):
    """
    Stage required libc++ libraries by copying from extracted LLVM tarball.
    Creates symlinks to map different soname versions.
    """
    LIBCXX_BASE_NAME = f"clang+llvm-{LLVM_VERSION}-x86_64-linux-gnu-ubuntu-18.04"
    LLVM_URL = f"https://github.com/llvm/llvm-project/releases/download/llvmorg-{LLVM_VERSION}/{LIBCXX_BASE_NAME}.tar.xz"

    REQUIRED_LIBS = [
        "libc++.so.1.0",
        "libc++abi.so.1.0",
        "libunwind.so.1",
        "libm.so.6",
        "libpython3.10.so.1.0",
    ]

    # Mapping of symlink → target file
    SYMLINKS = {
        "libc++.so.1": "libc++.so.1.0",
        "libc++.so": "libc++.so.1",
        "libc++abi.so.1": "libc++abi.so.1.0",
        "libc++abi.so": "libc++abi.so.1",
    }

    target_dir.mkdir(parents=True, exist_ok=True)

    # Check if all required files already staged
    if all((target_dir / lib).exists() for lib in REQUIRED_LIBS):
        print(f"[libcxx] Already staged at {target_dir}, skipping")
        return

    # Download & extract LLVM tarball if not exists
    temp_tar = pathlib.Path("/tmp") / f"{LIBCXX_BASE_NAME}.tar.xz"
    temp_extract = pathlib.Path("/tmp") / LIBCXX_BASE_NAME
    if not temp_tar.exists():
        print(f"[libcxx] Downloading {LLVM_URL}")
        urllib.request.urlretrieve(LLVM_URL, temp_tar)

    print(f"[libcxx] Extracting {temp_tar}")
    with tarfile.open(temp_tar, "r:xz") as tar:
        tar.extractall(temp_extract.parent)

    lib_src = temp_extract / "lib"

    # Copy required files
    for fname in REQUIRED_LIBS:
        src_path = lib_src / fname
        dst_path = target_dir / fname
        if src_path.exists():
            shutil.copy(src_path, dst_path)
            print(f"[libcxx] Copied {src_path.name} -> {dst_path}")
        else:
            print(f"[libcxx] Warning: {fname} not found in LLVM package")

    # Create symlinks according to mapping
    for link_name, target_name in SYMLINKS.items():
        link_path = target_dir / link_name
        target_path = target_dir / target_name
        if not link_path.exists() and target_path.exists():
            os.symlink(target_name, link_path)  # symlink relative to target_dir
            print(f"[libcxx] Symlinked {link_name} -> {target_name}")

    print(f"[libcxx] Staged libc++ to {target_dir}")


def _load_libcxx_libs(lib_path):
    """Load libc++ shared libs from given directory."""
    candidates = list(lib_path.glob("*.so*"))
    priority = ["libc++abi", "libc++"]

    sorted_candidates = [
        f for p in priority for f in candidates if f.name.startswith(p)
    ] + [f for f in candidates if not any(f.name.startswith(p) for p in priority)]

    for sofile in sorted_candidates:
        try:
            ctypes.CDLL(str(sofile), mode=ctypes.RTLD_GLOBAL)
            print(f"Loaded {sofile.name}")
        except OSError as e:
            print(f"[WARN] Failed to load {sofile.name}: {e}")


# === End of libcxx handling ===


def install_qnn_sdk(force_download: bool = True) -> bool:
    """Ensure QNN SDK + libc++ are available and loaded."""
    # --- set up QNN SDK ---
    if not QNN_SDK_PATH.exists():
        if force_download:
            if not _download_qnn_sdk():
                return False
        else:
            print("[INIT] Qualcomm SDK not found and force_download=False")
            return False

    os.environ["QNN_SDK_ROOT"] = str(QNN_SDK_PATH)

    qnn_lib = QNN_SDK_PATH / "lib" / "x86_64-linux-clang" / "libQnnHtp.so"
    try:
        ctypes.CDLL(str(qnn_lib), mode=ctypes.RTLD_GLOBAL)
        print(f"[INIT] Loaded QNN library from {qnn_lib}")
    except OSError as e:
        print(f"[ERROR] Failed to load QNN library: {e}")
        return False
    # --- End of QNN SDK ---

    # --- Set up libcxx ---
    try:
        libcxx_dir = _get_libcxx_dir()
        _stage_libcxx(libcxx_dir)
        _load_libcxx_libs(libcxx_dir)
    except Exception as e:
        print(f"[libcxx] Warning: failed to stage/load libc++: {e}")
    # --- End of libcxx ---

    return True
