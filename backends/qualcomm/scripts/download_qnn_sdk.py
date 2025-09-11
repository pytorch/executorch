# Add these imports for additional logging
import ctypes
import logging
import os
import pathlib
import platform
import shutil
import tarfile
import tempfile
import urllib.request
import zipfile
from typing import Dict, List, Optional, Tuple

# Module logger (library-friendly)
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

PKG_ROOT = pathlib.Path(__file__).parent.parent
SDK_DIR = PKG_ROOT / "sdk" / "qnn"


def is_linux_x86() -> bool:
    """
    Check if the current platform is Linux x86_64.

    Returns:
        bool: True if the system is Linux x86_64, False otherwise.
    """
    print("platform.system().lower(): ", platform.system().lower())
    print("platform.machine().lower(): ", platform.machine().lower())
    print("os.name: ", os.name)

    return platform.system().lower() == "linux" and platform.machine().lower() in (
        "x86_64",
        "amd64",
        "i386",
        "i686",
    )


REQUIRED_QNN_LIBS = ["libc.so.6"]


def check_glibc_exist() -> bool:
    """
    Check if users have glibc installed.
    """
    print("[QNN] Checking glibc exist running on Linux x86")
    paths = ["/lib/x86_64-linux-gnu/libc.so.6", "/lib64/libc.so.6", "/lib/libc.so.6"]

    exists = any(os.path.isfile(p) for p in paths)
    if not exists:
        logger.error(
            r""""
            [QNN] glibc not found. Please install glibc following the commands below.
            Ubuntu/Debian:
                sudo apt update
                sudo apt install libc6

            Fedora/Red Hat:
                sudo dnf install glibc

            Arch Linux:
                sudo pacman -S glibc
            """
        )
    print("[QNN] glibc exists: ", exists)
    return exists


def _download_archive(url: str, archive_path: pathlib.Path) -> bool:
    """Download archive from URL with progress reporting."""
    logger.debug("Archive will be saved to: %s", archive_path)

    try:
        urllib.request.urlretrieve(url, archive_path, _make_report_progress())
        logger.info("Download completed!")
    except Exception as e:
        logger.exception("Error during download: %s", e)
        return False

    if archive_path.exists() and archive_path.stat().st_size == 0:
        logger.warning("Downloaded file is empty!")
        return False
    elif not archive_path.exists():
        logger.error("File was not downloaded!")
        return False
    return True


def _make_report_progress():
    """Return a callback to report download progress."""
    last_reported = 0

    def report_progress(block_num, block_size, total_size):
        nonlocal last_reported
        try:
            downloaded = block_num * block_size
            percent = downloaded / total_size * 100 if total_size else 100.0
        except Exception:
            percent, downloaded, total_size = 0.0, block_num * block_size, 0
        if percent - last_reported >= 20 or percent >= 100:
            logger.info(
                "Downloaded: %d/%d bytes (%.2f%%)", downloaded, total_size, percent
            )
            last_reported = percent

    return report_progress


def _extract_archive(
    url: str, archive_path: pathlib.Path, content_dir: str, dst_folder: pathlib.Path
):
    """Extract archive based on type (zip or tar)."""
    if url.endswith(".zip"):
        logger.info("Extracting ZIP archive...")
        _extract_zip(archive_path, content_dir, dst_folder)
    elif url.endswith((".tar.gz", ".tgz")):
        logger.info("Extracting TAR archive...")
        _extract_tar(archive_path, content_dir, dst_folder)
    else:
        raise ValueError(f"Unsupported archive format: {url}")


def _verify_extraction(dst_folder: pathlib.Path):
    """Check if extraction succeeded and log contents."""
    logger.info("Verifying extraction to %s", dst_folder)
    if dst_folder.exists():
        logger.debug("SDK directory exists. Contents:")
        for item in dst_folder.iterdir():
            logger.debug("  %s", item.name)
    else:
        logger.error("SDK directory was not created!")


def _download_qnn_sdk(dst_folder=SDK_DIR) -> Optional[pathlib.Path]:
    """
    Download and extract the Qualcomm SDK into dst_folder.
    Only runs on Linux x86 platforms.
    """
    QNN_VERSION = "2.37.0.250724"
    logger.info("Downloading Qualcomm SDK...")
    QAIRT_URL = (
        f"https://softwarecenter.qualcomm.com/api/download/software/sdks/"
        f"Qualcomm_AI_Runtime_Community/All/{QNN_VERSION}/v{QNN_VERSION}.zip"
    )
    QAIRT_CONTENT_DIR = f"qairt/{QNN_VERSION}"

    if not is_linux_x86() or not check_glibc_exist():
        logger.info("Skipping Qualcomm SDK (only supported on Linux x86).")
        return None
    else:
        logger.info("Downloading Qualcomm SDK for Linux x86!!!!")

    dst_folder.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        archive_path = pathlib.Path(tmpdir) / pathlib.Path(QAIRT_URL).name
        if not _download_archive(QAIRT_URL, archive_path):
            return None

        _extract_archive(QAIRT_URL, archive_path, QAIRT_CONTENT_DIR, dst_folder)
        _verify_extraction(dst_folder)

    return dst_folder


def _extract_zip(archive_path, content_dir, target_dir):
    logger.debug("Extracting %s to %s", archive_path, target_dir)
    logger.debug("Looking for content in subdirectory: %s", content_dir)

    target_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path, "r") as zip_ref:
        files_to_extract = [f for f in zip_ref.namelist() if f.startswith(content_dir)]

        for file in files_to_extract:
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
        logger.info("[libcxx] Already staged at %s, skipping download", target_dir)
        return

    temp_tar = pathlib.Path("/tmp") / f"{LIBCXX_BASE_NAME}.tar.xz"
    temp_extract = pathlib.Path("/tmp") / LIBCXX_BASE_NAME

    if not temp_tar.exists():
        logger.info("[libcxx] Downloading %s", LLVM_URL)
        urllib.request.urlretrieve(LLVM_URL, temp_tar)

    logger.info("[libcxx] Extracting %s", temp_tar)
    with tarfile.open(temp_tar, "r:xz") as tar:
        tar.extractall(temp_extract.parent)

    lib_src = temp_extract / "lib" / "x86_64-unknown-linux-gnu"
    for fname in REQUIRED_LIBCXX_LIBS:
        src_path = lib_src / fname
        if not src_path.exists():
            logger.warning(
                "[libcxx] %s not found in extracted LLVM src_path %s", fname, src_path
            )
            continue
        shutil.copy(src_path, target_dir / fname)

    logger.info("[libcxx] Staged libc++ to %s", target_dir)


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
        logger.info(
            "[QNN] libQnnHtp.so found in LD_LIBRARY_PATH; skipping SDK install."
        )
        for lib, p in locs.items():
            logger.info("      - %s: %s", lib, p)
        return True

    # Not found → use packaged SDK
    qnn_sdk_dir = SDK_DIR
    logger.info("[QNN] libQnnHtp.so not found in LD_LIBRARY_PATH.")
    if not qnn_sdk_dir.exists():
        logger.info("[QNN] SDK dir missing; downloading...")
        _download_qnn_sdk()
    else:
        logger.info("[QNN] Using existing SDK at %s", qnn_sdk_dir)

    os.environ["QNN_SDK_ROOT"] = str(qnn_sdk_dir)

    qnn_lib = qnn_sdk_dir / "lib" / "x86_64-linux-clang" / "libQnnHtp.so"
    logger.info("[QNN] Loading %s", qnn_lib)
    lib_loaded = False
    try:
        ctypes.CDLL(str(qnn_lib), mode=ctypes.RTLD_GLOBAL)
        logger.info("[QNN] Loaded libQnnHtp.so from packaged SDK.")
        lib_loaded = True
    except OSError as e:
        logger.error("[QNN][ERROR] Failed to load %s: %s", qnn_lib, e)
    return lib_loaded


def _load_libcxx_libs(lib_path):
    logger.debug("running _load_libcxx_libs")
    candidates = list(lib_path.glob("*.so*"))
    priority = ["libc++abi", "libc++"]
    sorted_candidates = [
        f for name in priority for f in candidates if f.name.startswith(name)
    ]
    sorted_candidates += [f for f in candidates if f not in sorted_candidates]
    logger.debug("sorted_candidates: %s", sorted_candidates)
    for sofile in sorted_candidates:
        try:
            ctypes.CDLL(str(sofile), mode=ctypes.RTLD_GLOBAL)
            logger.info("Loaded %s", sofile.name)
        except OSError as e:
            logger.warning("[WARN] Failed to load %s: %s", sofile.name, e)


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
        logger.info(
            "[libcxx] All libc++ libs present in LD_LIBRARY_PATH; skipping staging."
        )
        for lib, p in locs.items():
            logger.info("         - %s: %s", lib, p)
        return True

    logger.info(
        "[libcxx] Some libc++ libs missing in LD_LIBRARY_PATH; staging packaged libc++..."
    )
    lib_loaded = False
    try:
        libcxx_dir = PKG_ROOT / "sdk" / f"libcxx-{LLVM_VERSION}"
        _stage_libcxx(libcxx_dir)
        _load_libcxx_libs(libcxx_dir)
        logger.info("[libcxx] Staged and loaded libc++ from %s", libcxx_dir)
        lib_loaded = True
    except Exception as e:
        logger.exception("[libcxx][ERROR] Failed to stage/load libc++: %s", e)
    return lib_loaded


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
