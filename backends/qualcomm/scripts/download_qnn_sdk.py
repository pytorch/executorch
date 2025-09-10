# Add these imports for additional logging
import ctypes
import os
import pathlib
import platform
import shutil
import tarfile
import tempfile
import urllib.request
import zipfile
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

PKG_ROOT = pathlib.Path(__file__).parent.parent
SDK_DIR = PKG_ROOT / "sdk" / "qnn"


def is_linux_x86() -> bool:
    """
    Check if the current platform is Linux x86_64.

    Returns:
        bool: True if the system is Linux x86_64, False otherwise.
    """
    return platform.system().lower() == "linux" and platform.machine().lower() in (
        "x86_64",
        "amd64",
        "i386",
        "i686",
    )


QNN_VERSION = "2.37.0.250724"


def get_qnn_version() -> str:
    return QNN_VERSION


def _download_qnn_sdk(dst_folder=SDK_DIR) -> Optional[pathlib.Path]:
    """
    Download and extract the Qualcomm SDK into dst_folder.

    Notes:
        - Only runs on Linux x86 platforms. Skips otherwise.
    """
    print("Downloading Qualcomm SDK...")
    QAIRT_URL = f"https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/{QNN_VERSION}/v{QNN_VERSION}.zip"
    QAIRT_CONTENT_DIR = f"qairt/{QNN_VERSION}"

    if not is_linux_x86():
        print("Skipping Qualcomm SDK (only supported on Linux x86).")
        return None

    dst_folder.mkdir(parents=True, exist_ok=True)
    print(f"dst_folder is {dst_folder}, exists: {dst_folder.exists()}")
    print(f"Current working directory: {os.getcwd()}")

    with tempfile.TemporaryDirectory() as tmpdir:
        archive_path = pathlib.Path(tmpdir) / pathlib.Path(QAIRT_URL).name
        print(f"Temporary directory: {tmpdir}")
        print(f"Archive will be saved to: {archive_path}")

        print(f"Downloading Qualcomm SDK from {QAIRT_URL}...")
        try:

            def make_report_progress():
                last_reported = 0

                def report_progress(block_num, block_size, total_size):
                    nonlocal last_reported
                    downloaded = block_num * block_size
                    percent = downloaded / total_size * 100
                    if percent - last_reported >= 20 or percent >= 100:
                        print(
                            f"Downloaded: {downloaded}/{total_size} bytes ({percent:.2f}%)"
                        )
                        last_reported = percent

                return report_progress

            urllib.request.urlretrieve(QAIRT_URL, archive_path, make_report_progress())
            print("Download completed!")

            if archive_path.exists() and archive_path.stat().st_size == 0:
                print("WARNING: Downloaded file is empty!")
            elif not archive_path.exists():
                print("ERROR: File was not downloaded!")
                return None

        except Exception as e:
            print(f"Error during download: {e}")
            return None

        if QAIRT_URL.endswith(".zip"):
            print("Extracting ZIP archive...")
            _extract_zip(archive_path, QAIRT_CONTENT_DIR, dst_folder)
        elif QAIRT_URL.endswith((".tar.gz", ".tgz")):
            print("Extracting TAR archive...")
            _extract_tar(archive_path, QAIRT_CONTENT_DIR, dst_folder)
        else:
            raise ValueError(f"Unsupported archive format: {QAIRT_URL}")

        print(f"Verifying extraction to {dst_folder}")
        if dst_folder.exists():
            print(f"SDK directory exists. Contents:")
            for item in dst_folder.iterdir():
                print(f"  {item.name}")
        else:
            print("ERROR: SDK directory was not created!")

        print(f"Qualcomm SDK extracted to {dst_folder}")

    return dst_folder


def _extract_zip(archive_path, content_dir, target_dir):
    print(f"Extracting {archive_path} to {target_dir}")
    print(f"Looking for content in subdirectory: {content_dir}")

    target_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path, "r") as zip_ref:
        files_to_extract = [f for f in zip_ref.namelist() if f.startswith(content_dir)]

        for file in tqdm(files_to_extract, desc="Extracting files"):
            relative_path = pathlib.Path(file).relative_to(content_dir)
            if relative_path == pathlib.Path("."):
                continue

            out_path = target_dir / relative_path
            if file.endswith("/"):
                out_path.mkdir(parents=True, exist_ok=True)
            else:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with zip_ref.open(file) as src, open(out_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)


def _extract_tar(archive_path: pathlib.Path, prefix: str, target_dir: pathlib.Path):
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
                if src is None:
                    continue
                with src, open(out_path, "wb") as dst:
                    dst.write(src.read())


LLVM_VERSION = "14.0.0"
LIBCXX_BASE_NAME = f"clang+llvm-{LLVM_VERSION}-x86_64-linux-gnu-ubuntu-18.04"
LLVM_URL = f"https://github.com/llvm/llvm-project/releases/download/llvmorg-{LLVM_VERSION}/{LIBCXX_BASE_NAME}.tar.xz"
REQUIRED_LIBCXX_LIBS = [
    "libc++.so.1.0",
    "libc++abi.so.1.0",
    "libunwind.so.1",
]


def _stage_libcxx(target_dir: pathlib.Path):
    target_dir.mkdir(parents=True, exist_ok=True)

    if all((target_dir / libname).exists() for libname in REQUIRED_LIBCXX_LIBS):
        print(f"[libcxx] Already staged at {target_dir}, skipping download")
        return

    temp_tar = pathlib.Path("/tmp") / f"{LIBCXX_BASE_NAME}.tar.xz"
    temp_extract = pathlib.Path("/tmp") / LIBCXX_BASE_NAME

    if not temp_tar.exists():
        print(f"[libcxx] Downloading {LLVM_URL}")
        urllib.request.urlretrieve(LLVM_URL, temp_tar)

    print(f"[libcxx] Extracting {temp_tar}")
    with tarfile.open(temp_tar, "r:xz") as tar:
        tar.extractall(temp_extract.parent)

    lib_src = temp_extract / "lib" / "x86_64-unknown-linux-gnu"
    for fname in REQUIRED_LIBCXX_LIBS:
        src_path = lib_src / fname
        if not src_path.exists():
            print(
                f"[libcxx] Warning: {fname} not found in extracted LLVM src_path {src_path}"
            )
            continue
        shutil.copy(src_path, target_dir / fname)

    print(f"[libcxx] Staged libc++ to {target_dir}")


REQUIRED_QNN_LIBS: List[str] = [
    "libQnnHtp.so",
]


def _ld_library_paths() -> List[pathlib.Path]:
    """Split LD_LIBRARY_PATH into ordered directories (skip empties)."""
    raw = os.environ.get("LD_LIBRARY_PATH", "")
    return [pathlib.Path(p) for p in raw.split(":") if p.strip()]


def _find_lib_in_ld_paths(
    libname: str, ld_dirs: Optional[List[pathlib.Path]] = None
) -> Optional[pathlib.Path]:
    """Return first matching path to `libname` in LD_LIBRARY_PATH, or None."""
    if ld_dirs is None:
        ld_dirs = _ld_library_paths()
    for d in ld_dirs:
        candidate = d / libname
        try:
            if candidate.exists():
                return candidate.resolve()
        except Exception:
            # Ignore unreadable / permission issues, keep looking.
            pass
    return None


def _check_libs_in_ld(
    libnames: List[str],
) -> Tuple[bool, Dict[str, Optional[pathlib.Path]]]:
    """
    Check if each lib in `libnames` exists in LD_LIBRARY_PATH directories.

    Returns:
        all_present: True iff every lib was found
        locations:   mapping lib -> path (or None if missing)
    """
    ld_dirs = _ld_library_paths()
    locations: Dict[str, Optional[pathlib.Path]] = {}
    for lib in libnames:
        locations[lib] = _find_lib_in_ld_paths(lib, ld_dirs)
    all_present = all(locations[lib] is not None for lib in libnames)
    return all_present, locations


# -----------------------
# Ensure QNN SDK library
# -----------------------
def _ensure_qnn_sdk_lib() -> bool:
    """
    Ensure libQnnHtp.so is available.
      - If found in LD_LIBRARY_PATH: do nothing, return True.
      - Otherwise: ensure packaged SDK is present, then load libQnnHtp.so from it.
    """
    all_present, locs = _check_libs_in_ld(REQUIRED_QNN_LIBS)
    if all_present:
        print("[QNN] libQnnHtp.so found in LD_LIBRARY_PATH; skipping SDK install.")
        for lib, p in locs.items():
            print(f"      - {lib}: {p}")
        return True

    # Not found → use packaged SDK
    qnn_sdk_dir = SDK_DIR
    print(f"[QNN] libQnnHtp.so not found in LD_LIBRARY_PATH.")
    if not qnn_sdk_dir.exists():
        print("[QNN] SDK dir missing; downloading...")
        _download_qnn_sdk()
    else:
        print(f"[QNN] Using existing SDK at {qnn_sdk_dir}")

    os.environ["QNN_SDK_ROOT"] = str(qnn_sdk_dir)

    qnn_lib = qnn_sdk_dir / "lib" / "x86_64-linux-clang" / "libQnnHtp.so"
    print(f"[QNN] Loading {qnn_lib}")
    try:
        ctypes.CDLL(str(qnn_lib), mode=ctypes.RTLD_GLOBAL)
        print("[QNN] Loaded libQnnHtp.so from packaged SDK.")
        return True
    except OSError as e:
        print(f"[QNN][ERROR] Failed to load {qnn_lib}: {e}")
        return False


def _load_libcxx_libs(lib_path):
    print("running _load_libcxx_libs")
    candidates = list(lib_path.glob("*.so*"))
    priority = ["libc++abi", "libc++"]
    sorted_candidates = [
        f for name in priority for f in candidates if f.name.startswith(name)
    ]
    sorted_candidates += [f for f in candidates if f not in sorted_candidates]
    print("sorted_candidates: ", sorted_candidates)
    for sofile in sorted_candidates:
        try:
            ctypes.CDLL(str(sofile), mode=ctypes.RTLD_GLOBAL)
            print(f"Loaded {sofile.name}")
        except OSError as e:
            print(f"[WARN] Failed to load {sofile.name}: {e}")


# ---------------------
# Ensure libc++ family
# ---------------------
def _ensure_libcxx_stack() -> bool:
    """
    Ensure libc++ stack is available.
      - If all required libc++ libs are found in LD_LIBRARY_PATH: do nothing.
      - Otherwise: stage and load the packaged libc++ bundle.
    """
    all_present, locs = _check_libs_in_ld(REQUIRED_LIBCXX_LIBS)
    if all_present:
        print("[libcxx] All libc++ libs present in LD_LIBRARY_PATH; skipping staging.")
        for lib, p in locs.items():
            print(f"         - {lib}: {p}")
        return True

    print(
        "[libcxx] Some libc++ libs missing in LD_LIBRARY_PATH; staging packaged libc++..."
    )
    try:
        libcxx_dir = PKG_ROOT / "sdk" / f"libcxx-{LLVM_VERSION}"
        _stage_libcxx(libcxx_dir)
        _load_libcxx_libs(libcxx_dir)
        print(f"[libcxx] Staged and loaded libc++ from {libcxx_dir}")
        return True
    except Exception as e:
        print(f"[libcxx][ERROR] Failed to stage/load libc++: {e}")
        return False


# ---------------
# Public entrypoint
# ---------------
def install_qnn_sdk() -> bool:
    """
    Initialize Qualcomm backend with separated logic:

    QNN SDK:
      - If libQnnHtp.so exists in LD_LIBRARY_PATH: do nothing.
      - Else: ensure packaged SDK, load libQnnHtp.so.

    libc++ stack:
      - If required libc++ libs exist in LD_LIBRARY_PATH: do nothing.
      - Else: stage and load packaged libc++.

    Returns:
        True if both steps succeeded (or were already satisfied), else False.
    """
    ok_libcxx = _ensure_libcxx_stack()
    ok_qnn = _ensure_qnn_sdk_lib()
    return bool(ok_qnn and ok_libcxx)
